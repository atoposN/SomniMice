from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from .configs import EEGTransformerConfig, PreprocessConfig, TrainingConfig
from .data import SleepEpochDataset, SubjectEpochs, load_subject_epochs
from .losses import CenterLoss
from .models import build_model


def create_subject_splits(
    data_dir: Path,
    subjects: Sequence[str],
    cfg: PreprocessConfig,
    cache_dir: Path | None = None,
    data_format: str = "mat",
) -> list[SubjectEpochs]:
    records = []
    for sid in subjects:
        records.append(load_subject_epochs(sid, data_dir, cfg, cache_dir, data_format=data_format))
    return records


def describe_label_distribution(subjects: Sequence[SubjectEpochs], name: str, num_classes: int) -> None:
    counts = np.zeros(num_classes, dtype=np.int64)
    for subj in subjects:
        counts += np.bincount(subj.labels, minlength=num_classes)
    total = counts.sum()
    print(f"[Stats] {name} label distribution:")
    for cls, cnt in enumerate(counts):
        frac = cnt / max(total, 1)
        print(f"  class {cls}: {cnt} ({frac:.2%})")


def compute_class_weights(
    subjects: Sequence[SubjectEpochs],
    num_classes: int,
) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for subj in subjects:
        counts += np.bincount(subj.labels, minlength=num_classes)
    counts = np.clip(counts, 1.0, None)
    inv = 1.0 / counts
    inv = inv * (num_classes / inv.sum())
    return torch.tensor(inv, dtype=torch.float32)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.param_names = {name for name, _ in model.named_parameters()}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name, tensor in current.items():
            if name not in self.shadow:
                self.shadow[name] = tensor.detach().clone()
            if name in self.param_names:
                self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(tensor.detach())

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup:
            model.load_state_dict(self.backup, strict=True)
            self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}


def build_train_loader(
    dataset: SleepEpochDataset,
    batch_size: int,
    num_classes: int,
    balanced_sampling: bool,
) -> DataLoader:
    if len(dataset) == 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    if balanced_sampling:
        counts = np.bincount(dataset.flat_labels, minlength=num_classes)
        counts = np.clip(counts, 1, None)
        weights = 1.0 / counts
        sample_weights = weights[dataset.flat_labels]
        sampler = WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)


def train_model(
    data_dir: Path,
    cfg_prep: PreprocessConfig,
    cfg_model: EEGTransformerConfig,
    train_cfg: TrainingConfig,
    model_name: str = "CausalTransformer",
    cache_dir: Path | None = None,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_subjects = create_subject_splits(
        data_dir, train_cfg.train_subjects, cfg_prep, cache_dir, data_format=train_cfg.data_format
    )
    val_subjects = create_subject_splits(
        data_dir, train_cfg.val_subjects, cfg_prep, cache_dir, data_format=train_cfg.data_format
    )
    describe_label_distribution(train_subjects, "train", cfg_model.num_classes)
    describe_label_distribution(val_subjects, "val", cfg_model.num_classes)

    window = max(1, cfg_model.context_window)
    train_dataset = SleepEpochDataset(
        train_subjects,
        window,
        augment=False,
    )
    val_dataset = SleepEpochDataset(
        val_subjects,
        window,
        augment=False,
    )
    train_loader = build_train_loader(
        train_dataset,
        train_cfg.batch_size,
        cfg_model.num_classes,
        train_cfg.balanced_sampling,
    )
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = build_model(model_name, cfg_model).to(device)

    class_weights = None
    if train_cfg.class_weighting:
        class_weights = compute_class_weights(train_subjects, cfg_model.num_classes).to(device)
        print(f"[Stats] class weights: {class_weights.tolist()}")
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    center_loss = CenterLoss(cfg_model.num_classes, cfg_model.d_model).to(device)
    params = list(model.parameters()) + list(center_loss.parameters())
    optimizer = AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    ema = ModelEMA(model, train_cfg.ema_decay) if train_cfg.ema_decay > 0 else None

    best_val_bacc = -1.0
    epochs_without_improve = 0
    ckpt_dir = train_cfg.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{model_name}_best.pt"

    for epoch in range(1, train_cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for step, (batch_x, batch_y) in enumerate(train_loader, 1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, features = model(batch_x, return_features=True)
            loss_cls = ce_loss(logits, batch_y) * train_cfg.w_wce
            loss_center = center_loss(features, batch_y) * train_cfg.w_ct
            loss = loss_cls + loss_center
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            running_loss += float(loss.item()) * batch_x.size(0)
            total += batch_x.size(0)
            if step % train_cfg.log_interval == 0:
                print(
                    f"[Epoch {epoch:02d}] step {step:04d} "
                    f"loss={loss.item():.4f} (ce={loss_cls.item():.4f}, center={loss_center.item():.4f})"
                )
        train_loss = running_loss / max(total, 1)

        if ema is not None:
            ema.apply_shadow(model)
        model.eval()
        correct = 0
        total = 0
        per_class_tp = torch.zeros(cfg_model.num_classes, device=device)
        per_class_fn = torch.zeros(cfg_model.num_classes, device=device)
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                preds = logits.argmax(dim=-1)
                correct += int((preds == batch_y).sum().item())
                total += batch_y.size(0)
                for cls in range(cfg_model.num_classes):
                    mask = batch_y == cls
                    per_class_tp[cls] += (preds[mask] == cls).sum()
                    per_class_fn[cls] += mask.sum()
        if ema is not None:
            ema.restore(model)
        acc = correct / max(total, 1)
        per_class_recall = per_class_tp / torch.clamp(per_class_fn, min=1)
        bacc = float(per_class_recall.mean().item())
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} val_acc={acc:.4f} val_bacc={bacc:.4f}")

        if bacc > best_val_bacc:
            best_val_bacc = bacc
            preprocess_meta = {
                "context_window": cfg_model.context_window,
                "standardization": "exp_mov",
            }
            state_to_save = ema.state_dict() if ema is not None else model.state_dict()
            torch.save({"model_state": state_to_save, "config": cfg_model, "preprocess": preprocess_meta}, checkpoint_path)
            print(f"  -> New best balanced accuracy, checkpoint saved to {checkpoint_path}")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if train_cfg.early_stop_patience > 0 and epochs_without_improve >= train_cfg.early_stop_patience:
                print(
                    f"Early stopping triggered after {epochs_without_improve} epochs without improvement "
                    f"(best val_bacc={best_val_bacc:.4f})."
                )
                break

    print(f"Training complete. Best validation bacc={best_val_bacc:.4f}")
    return checkpoint_path
