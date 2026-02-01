'''
python finetune.py --config config.yaml --preprocessed-dir data/finetune --subjects mouse --epochs 10
'''

from __future__ import annotations

import argparse
import json
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

from src.models import build_model
from src.utils import get_logger, load_config, prepare_experiment_dir, set_seed

logger = get_logger(__name__)


def _load_h5(file_path: Path, eeg_key: str, emg_key: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(file_path, "r") as f:
        eeg = np.asarray(f[f"{eeg_key}_epochs"][:], dtype=np.float32)
        emg = np.asarray(f[f"{emg_key}_epochs"][:], dtype=np.float32)
        labels = np.asarray(f["labels"][:], dtype=np.int64)
    data = np.stack([eeg, emg], axis=1)  # (epochs, channels, samples)
    return data, labels


def _pad_indices(n: int, center: int, half: int, total: int) -> List[int]:
    """Return indices of length 'total' centered on 'center' using edge padding."""
    start = center - half
    end = start + total
    if start < 0:
        offset = -start
        start = 0
        end = min(n, total - offset)
        idx = [0] * offset + list(range(start, end))
        while len(idx) < total:
            idx.append(idx[-1] if idx else 0)
        return idx
    if end > n:
        overflow = end - n
        start = max(0, n - total)
        idx = list(range(start, n)) + [n - 1] * overflow
        while len(idx) > total:
            idx.pop()
        return idx
    return list(range(start, end))


@dataclass
class FinetuneConfig:
    config_path: Path
    preprocessed_dir: Path
    subjects: Optional[Sequence[str]]
    batch_size: int
    max_epochs: int
    sequence_length: int
    sample_rate: int
    device: torch.device
    lr: float
    weight_decay: float
    lr_multiplier: float
    freeze_backbone: bool
    train_ratio: float
    seed: int


class SequenceDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        seq_len: int,
        means: Optional[np.ndarray],
        stds: Optional[np.ndarray],
        apply_zscore: bool,
    ):
        super().__init__()
        assert data.ndim == 3  # (epochs, channels, samples)
        self.data = data
        self.labels = labels.astype(np.int64)
        self.seq_len = int(seq_len)
        self.half = self.seq_len // 2
        self.apply_zscore = bool(apply_zscore)
        if means is None:
            means = np.zeros((self.data.shape[1],), dtype=np.float32)
        if stds is None:
            stds = np.ones((self.data.shape[1],), dtype=np.float32)
        self.means = means.astype(np.float32)
        self.stds = np.where(stds == 0.0, 1.0, stds).astype(np.float32)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        n = self.data.shape[0]
        indices = _pad_indices(n, idx, self.half, self.seq_len)
        seq = self.data[indices]  # (seq_len, channels, samples)
        if self.apply_zscore:
            # Apply per-channel z-score using finetune statistics
            seq = (seq - self.means[None, :, None]) / self.stds[None, :, None]
        return {
            "sequence": torch.tensor(seq, dtype=torch.float32),
            "label": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def compute_channel_stats(files: Sequence[Path], eeg_key: str, emg_key: str) -> Tuple[np.ndarray, np.ndarray]:
    means = []
    sq_means = []
    total = 0
    for fp in files:
        x, _ = _load_h5(fp, eeg_key, emg_key)  # (epochs, 2, samples)
        # reshape to (epochs*samples, channels)
        c = x.shape[1]
        flat = x.transpose(0, 2, 1).reshape(-1, c)  # (N, C)
        means.append(np.mean(flat, axis=0))
        sq_means.append(np.mean(flat ** 2, axis=0))
        total += flat.shape[0]
    if not means:
        raise ValueError("No data found to compute channel statistics.")
    mean = np.average(np.stack(means, axis=0), axis=0)
    sq_mean = np.average(np.stack(sq_means, axis=0), axis=0)
    var = np.maximum(sq_mean - mean ** 2, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _collect_files(dir_path: Path, subjects: Optional[Sequence[str]], suffix: str = ".h5") -> List[Path]:
    all_files = sorted(dir_path.rglob(f"*{suffix}"))
    if not subjects:
        return all_files
    wanted = {s.lower() for s in subjects}
    return [fp for fp in all_files if fp.stem.lower() in wanted]


def finetune(args: FinetuneConfig) -> Tuple[Path, Path]:
    cfg = load_config(args.config_path)
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    # Resolve keys
    eeg_key = data_cfg.get("channels", {}).get("eeg", "EEG_P")
    emg_key = data_cfg.get("channels", {}).get("emg", "EMG_diff")

    files = _collect_files(args.preprocessed_dir, args.subjects, suffix=data_cfg.get("file_extension", ".h5"))
    if not files:
        raise FileNotFoundError(f"No preprocessed files found in {args.preprocessed_dir}")

    apply_zscore = bool(data_cfg.get("use_zscore", data_cfg.get("zscore", False)))
    logger.info("Computing channel statistics on %d files...", len(files))
    means, stds = compute_channel_stats(files, eeg_key, emg_key)  # shape (2,)
    if apply_zscore:
        logger.info("Channel stats | mean=%s std=%s (z-score enabled)", np.round(means, 4), np.round(stds, 4))
    else:
        logger.info("Channel stats | mean=%s std=%s (z-score disabled; not applied)", np.round(means, 4), np.round(stds, 4))

    # Load all data for simplicity
    data_list = []
    label_list = []
    for fp in files:
        x, y = _load_h5(fp, eeg_key, emg_key)
        data_list.append(x)
        label_list.append(y)
    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    # ------------------------------------------------------------------ Label remapping to contiguous ids
    unique_labels = sorted({int(l) for l in labels.tolist()})
    remap = {orig: idx for idx, orig in enumerate(unique_labels)}
    if len(remap) == 0:
        raise ValueError("No labels found in finetune data.")
    if len(remap) > model_cfg.get("classifier", {}).get("num_classes", model_cfg.get("num_classes", len(remap))):
        logger.warning(
            "Number of unique labels (%d) exceeds model num_classes; updating classifier.num_classes accordingly.",
            len(remap),
        )

    labels = np.asarray([remap[int(l)] for l in labels], dtype=np.int64)
    num_classes = len(remap)

    # Update model config (local copy) to match the data label count
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.setdefault("classifier", {})
    model_cfg["classifier"]["num_classes"] = num_classes
    cfg = copy.deepcopy(cfg)
    cfg["model"] = model_cfg

    logger.info("Label remap for finetune: %s (num_classes=%d)", remap, num_classes)

    dataset = SequenceDataset(
        data,
        labels,
        seq_len=args.sequence_length,
        means=means,
        stds=stds,
        apply_zscore=apply_zscore,
    )

    # Train/val split within the same subjects if no explicit val set
    train_ratio = max(0.5, min(1.0, float(args.train_ratio)))
    total_len = len(dataset)
    set_seed(args.seed)
    if total_len < 2 or train_ratio >= 0.999:
        train_ds = dataset
        val_ds = None
        logger.info("Using full finetune set for training (train_ratio=%.3f); validation disabled.", train_ratio)
    else:
        train_len = int(total_len * train_ratio)
        val_len = max(1, total_len - train_len)
        if train_len < 1:
            train_len = max(1, total_len - 1)
            val_len = 1
        train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if val_ds else None

    # Build model and optionally freeze backbone
    model = build_model(cfg)
    # Ensure the classifier exists as 'classifier' attribute for common patterns
    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    base_lr = float(training_cfg.get("optimizer", {}).get("lr", 5e-4))
    lr = float(args.lr) if args.lr > 0 else base_lr * float(args.lr_multiplier)
    wd = float(args.weight_decay) if args.weight_decay >= 0 else float(training_cfg.get("optimizer", {}).get("weight_decay", 1e-2))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    device = args.device
    model.to(device)

    # Simple CE loss
    criterion = nn.CrossEntropyLoss()

    exp_dir = prepare_experiment_dir(
        {
            "experiment": {"output_dir": str(Path(cfg.get("experiment", {}).get("output_dir", "outputs")) / "finetune"), "timestamp": True},
            "model": model_cfg,
        },
        experiment_name="finetune",
    )

    best_bacc = -1.0
    best_state = None
    epochs = int(args.max_epochs)
    logger.info("Starting finetune for %d epochs (batch_size=%d, lr=%.2e, wd=%.2e)", epochs, args.batch_size, lr, wd)
    epoch_iter = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        model.train()
        total_loss = 0.0
        total_samples = 0
        train_iter = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", unit="batch", leave=False)
        for batch in train_iter:
            x = batch["sequence"].to(device)  # (B, seq, C, S)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model({"sequence": x})
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
        train_loss = total_loss / max(total_samples, 1)

        if val_loader is not None:
            # Validation
            model.eval()
            preds = []
            gts = []
            val_loss = 0.0
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc=f"Val {epoch}/{epochs}", unit="batch", leave=False)
                for batch in val_iter:
                    x = batch["sequence"].to(device)
                    y = batch["label"].to(device)
                    logits = model({"sequence": x})
                    val_loss += float(criterion(logits, y).item()) * y.size(0)
                    preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
                    gts.extend(y.cpu().numpy().tolist())
            val_loss = val_loss / max(len(val_ds), 1)
            bacc = balanced_accuracy_score(gts, preds) if gts else 0.0
            logger.info("[Epoch %03d] train_loss=%.4f val_loss=%.4f val_bacc=%.4f", epoch, train_loss, val_loss, bacc)
            if bacc > best_bacc:
                best_bacc = bacc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            logger.info("[Epoch %03d] train_loss=%.4f (no validation)", epoch, train_loss)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    # Save artifacts with date tag
    date_tag = datetime.now().strftime("%Y%m%d")
    best_ckpt = exp_dir / f"best_model_{date_tag}.pt"
    norm_path = exp_dir / f"normalization_stats_{date_tag}.json"
    assert best_state is not None
    torch.save(best_state, best_ckpt)
    with norm_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "channel_names": ["eeg", "emg"],
                "means": [float(x) for x in means] if means is not None else [0.0, 0.0],
                "stds": [float(x) for x in stds] if stds is not None else [1.0, 1.0],
                "sequence_length": int(args.sequence_length),
                "sample_rate": int(args.sample_rate),
                "computed_at": datetime.now().isoformat(),
                "notes": "Per-channel z-score parameters; z-score applied during training: "
                + ("yes" if apply_zscore else "no (LLM preprocessing only)."),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Saved best checkpoint to %s and normalization stats to %s", best_ckpt, norm_path)

    # Record finetune timestamp to data/OnlineRecord for bookkeeping alongside txt outputs
    try:
        timestamp_dir = Path("data/OnlineRecord")
        timestamp_dir.mkdir(parents=True, exist_ok=True)
        stamp_file = timestamp_dir / f"finetune_timestamp_{date_tag}.txt"
        stamp_file.write_text(
            f"finished_at={datetime.now().isoformat()}\n"
            f"best_ckpt={best_ckpt}\n"
            f"normalization={norm_path}\n",
            encoding="utf-8",
        )
    except Exception:
        logger.warning("Failed to write finetune timestamp record to data/OnlineRecord", exc_info=True)
    return best_ckpt, norm_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute z-score stats and finetune model on labeled pre-recorded data.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("training_config.yaml"),
        help="Training config (used for model and data defaults).",
    )
    p.add_argument("--preprocessed-dir", type=Path, default=None, help="Directory with preprocessed .h5 files.")
    p.add_argument("--subjects", type=str, nargs="*", default=None, help="Optional list of subject IDs to include.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="Number of finetune epochs.")
    p.add_argument("--lr", type=float, default=-1.0, help="Override learning rate. Default uses config * lr_multiplier.")
    p.add_argument("--weight-decay", type=float, default=-1.0)
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    train_cfg = cfg.get("training", {})

    pre_dir = args.preprocessed_dir or Path(data_cfg.get("preprocessed_dir", "data/preprocessed"))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 64))
    epochs = args.epochs or int(eval_cfg.get("finetune_epochs", 10))
    seq_len = int(data_cfg.get("sequence_length", 11))
    sample_rate = int(data_cfg.get("sample_rate", 256))
    lr_mult = float(eval_cfg.get("finetune_settings", {}).get("lr_multiplier", 0.1))
    freeze_backbone = bool(eval_cfg.get("finetune_settings", {}).get("freeze_backbone", False))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    wd = float(train_cfg.get("optimizer", {}).get("weight_decay", 1e-2))
    device = torch.device(args.device) if args.device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    seed = int(cfg.get("experiment", {}).get("seed", 42))

    ft_cfg = FinetuneConfig(
        config_path=Path(args.config),
        preprocessed_dir=pre_dir,
        subjects=args.subjects or data_cfg.get("train_subjects"),
        batch_size=batch_size,
        max_epochs=epochs,
        sequence_length=seq_len,
        sample_rate=sample_rate,
        device=device,
        lr=args.lr,
        weight_decay=wd,
        lr_multiplier=lr_mult,
        freeze_backbone=freeze_backbone,
        train_ratio=train_ratio,
        seed=seed,
    )
    finetune(ft_cfg)


if __name__ == "__main__":
    main()
