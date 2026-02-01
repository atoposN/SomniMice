from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .configs import EEGTransformerConfig, PreprocessConfig
from .data import (
    build_label_array,
    load_mouse_recording,
    parse_label_file,
    resolve_subject_paths,
    slice_epochs_with_labels,
)
from .models import build_model
from .preprocessing import apply_bandpass, build_channel_filters, exponential_moving_standardize

LABEL_MAP = {
    0: (1, "NREM"),
    1: (2, "REM"),
    2: (3, "Wake"),
}


@dataclass(slots=True)
class OnlineAdaptationConfig:
    enabled: bool = False
    lr: float = 1e-4
    steps: int = 1
    update_every: int = 32
    buffer_size: int = 64
    batch_size: int = 16
    param_scope: str = "head"
    grad_clip: float = 1.0


def _prepare_sequences(
    epochs: np.ndarray,
    labels: np.ndarray,
    context_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    seq_labels: list[int] = []
    buffer: list[np.ndarray] = []
    epoch_shape = epochs.shape[1:] if epochs.ndim >= 3 else ()
    for idx, epoch in enumerate(epochs):
        buffer.append(epoch)
        if len(buffer) > context_window:
            buffer.pop(0)
        seq = list(buffer)
        if len(seq) < context_window:
            seq = [seq[0]] * (context_window - len(seq)) + seq
        arr = np.stack(seq, axis=0)
        sequences.append(arr.astype(np.float32))
        seq_labels.append(int(labels[idx]))
    if not sequences:
        return np.empty((0, context_window, *epoch_shape), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.stack(sequences, axis=0), np.asarray(seq_labels, dtype=np.int64)


def _finetune_model_on_prefix(
    model: torch.nn.Module,
    sequences: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    if sequences.size == 0 or labels.size == 0 or epochs <= 0:
        return
    dataset = TensorDataset(
        torch.from_numpy(sequences),
        torch.from_numpy(labels.astype(np.int64)),
    )
    if len(dataset) == 0:
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch_idx in range(epochs):
        running = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * batch_x.size(0)
            count += batch_x.size(0)
        avg_loss = running / max(count, 1)
        print(f"[Finetune] epoch {epoch_idx + 1:02d} loss={avg_loss:.4f}")
    model.eval()


def _select_adaptation_params(model: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    if scope == "head":
        head = getattr(model, "head", None)
        if head is not None:
            return [p for p in head.parameters() if p.requires_grad]
    return [p for p in model.parameters() if p.requires_grad]


def _run_online_adaptation(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
    batch: torch.Tensor,
    cfg: OnlineAdaptationConfig,
) -> None:
    if batch.size(0) == 0:
        return
    model.train()
    for _ in range(max(1, cfg.steps)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1).mean()
        entropy.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
    model.eval()


def _seconds_to_code(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{seconds:02d}"


def _write_txt_predictions(
    txt_path: Path,
    records: list[dict],
    base_seconds: float,
    epoch_seconds: float,
    hop_seconds: float,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write("READ ONLY! DO NOT EDIT!\n")
        handle.write("4-INIT 3-Wake 2-REM 1-NREM\n")
        handle.write("==========Marker==========\n")
        handle.write("==========Start-End==========\n")
        handle.write("==========Sleep stage==========\n")
        unit = int(round(epoch_seconds))
        for rec in records:
            idx = rec["epoch_index"]
            start_seconds = base_seconds + idx * hop_seconds
            end_seconds = start_seconds + epoch_seconds
            start_idx = int(round(idx * hop_seconds))
            end_idx = start_idx + unit
            label_id, label_name = LABEL_MAP.get(rec["pred"], (rec["pred"], f"class_{rec['pred']}"))
            line = (
                f"{_seconds_to_code(start_seconds)}, {start_idx}, 1, "
                f"{_seconds_to_code(end_seconds)}, {end_idx}, 0, {label_id}, {label_name}\n"
            )
            handle.write(line)


def load_checkpoint(model_name: str, checkpoint_path: Path) -> EEGTransformerConfig:
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg: EEGTransformerConfig = payload["config"]
    model = build_model(model_name, cfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def simulate_subject_online(
    model: torch.nn.Module,
    subject_id: str,
    data_dir: Path,
    preprocess_cfg: PreprocessConfig,
    device: torch.device,
    txt_output: Optional[Path] = None,
    csv_output: Optional[Path] = None,
    context_window: int = 1,
    adapt_config: Optional[OnlineAdaptationConfig] = None,
    finetune_fraction: float = 0.0,
    finetune_epochs: int = 0,
    finetune_lr: float = 1e-4,
    finetune_batch_size: int = 16,
) -> dict:
    mat_path, txt_path = resolve_subject_paths(subject_id, data_dir)
    raw = load_mouse_recording(mat_path)
    annotations, meta = parse_label_file(txt_path)
    sos = build_channel_filters(preprocess_cfg)
    filtered = apply_bandpass(raw, sos, real_time=True)
    standardized = exponential_moving_standardize(filtered, init_window=preprocess_cfg.samples_per_epoch())

    samples_per_unit = raw.shape[1] / max(meta["duration_units"], 1)
    labels = build_label_array(raw.shape[1], annotations, samples_per_unit)
    epochs, epoch_labels = slice_epochs_with_labels(standardized, labels, preprocess_cfg)
    if epochs.size == 0:
        print(f"[WARN] No valid epochs for {subject_id}")
        return

    total_epochs = len(epochs)
    if finetune_fraction > 0 and finetune_epochs > 0 and total_epochs > 1:
        prefix = int(total_epochs * finetune_fraction)
        prefix = max(1, min(prefix, total_epochs - 1))
        prefix_epochs = epochs[:prefix]
        prefix_labels = epoch_labels[:prefix]
        sequences, seq_labels = _prepare_sequences(
            prefix_epochs,
            prefix_labels,
            context_window,
        )
        print(
            f"[Finetune] Using first {prefix}/{total_epochs} epochs "
            f"({prefix / total_epochs:.1%}) for supervised update"
        )
        _finetune_model_on_prefix(
            model,
            sequences,
            seq_labels,
            device,
            epochs=finetune_epochs,
            batch_size=finetune_batch_size,
            lr=finetune_lr,
        )

    csv_rows: list[str] = []
    records: list[dict] = []
    label_buffer: list[int] = []
    pred_buffer: list[int] = []
    epoch_seconds = preprocess_cfg.epoch_seconds
    base_seconds = meta.get("base_seconds", 0)
    hop_seconds = preprocess_cfg.hop_length_samples() / preprocess_cfg.sample_rate
    adapt_state = None
    if adapt_config and adapt_config.enabled:
        adapt_params = _select_adaptation_params(model, adapt_config.param_scope)
        if adapt_params:
            optimizer = torch.optim.Adam(adapt_params, lr=adapt_config.lr)
            adapt_state = {
                "optimizer": optimizer,
                "params": adapt_params,
                "buffer": [],
                "updates": 0,
                "cfg": adapt_config,
            }
            print(
                f"[Adapt] Enabled online adaptation (scope={adapt_config.param_scope}, "
                f"lr={adapt_config.lr}, steps={adapt_config.steps})"
            )
        else:
            print("[Adapt] No parameters available for online adaptation; skipping.")

    buffer: list[np.ndarray] = []
    for idx, epoch in enumerate(epochs):
        buffer.append(epoch)
        if len(buffer) > context_window:
            buffer.pop(0)
        seq = buffer
        if len(seq) < context_window:
            pad = [seq[0]] * (context_window - len(seq))
            seq = pad + seq
        seq_arr = np.stack(seq, axis=0)
        target_label = int(epoch_labels[idx]) if idx < len(epoch_labels) else -1
        batch = torch.from_numpy(seq_arr[None, ...]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=-1)[0]
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())
        csv_rows.append(f"{idx},{target_label},{pred},{conf:.6f}")
        records.append(
            {
                "epoch_index": idx,
                "true": target_label,
                "pred": pred,
                "prob": conf,
            }
        )
        label_buffer.append(target_label)
        pred_buffer.append(pred)

        if adapt_state is not None:
            entry = torch.from_numpy(np.copy(seq_arr))
            adapt_state["buffer"].append(entry)
            if len(adapt_state["buffer"]) > adapt_config.buffer_size:
                adapt_state["buffer"].pop(0)
            update_every = max(1, adapt_config.update_every)
            if (idx + 1) % update_every == 0:
                batch_size = min(adapt_config.batch_size, len(adapt_state["buffer"]))
                if batch_size > 0:
                    adapt_batch = torch.stack(adapt_state["buffer"][-batch_size:], dim=0).to(device)
                    _run_online_adaptation(
                        model,
                        adapt_state["optimizer"],
                        adapt_state["params"],
                        adapt_batch,
                        adapt_config,
                    )
                    adapt_state["buffer"].clear()
                    adapt_state["updates"] += 1
                    print(
                        f"[Adapt] Update #{adapt_state['updates']} "
                        f"(batch={batch_size}, steps={adapt_config.steps})"
                    )
    if csv_output:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8") as fh:
            fh.write("epoch_index,true_label,pred_label,pred_prob\n")
            fh.write("\n".join(csv_rows))
    if txt_output:
        _write_txt_predictions(txt_output, records, base_seconds, epoch_seconds, hop_seconds)
    return {"labels": label_buffer, "preds": pred_buffer}
