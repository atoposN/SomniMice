from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

from llm_eeg.configs import PreprocessConfig
from src.utils import load_config
from .config import write_temp_config
from .finetune import FinetuneConfig, finetune
from .preprocess import preprocess_to_h5, stage_recordings


def run_finetune_pipeline_dir(
    recordings_dir: Path,
    preprocessed_dir: Path,
    config_path: Path,
    *,
    subjects: Sequence[str] | None = None,
    cache_dir: Path | None = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    device: Optional[str] = None,
    train_ratio: float = 1.0,
    eeg_key: Optional[str] = None,
    emg_key: Optional[str] = None,
    sample_rate: Optional[float] = None,
) -> Tuple[Path, Path]:
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    ft_settings = eval_cfg.get("finetune_settings", {})

    eeg_key = eeg_key or data_cfg.get("channels", {}).get("eeg", "EEG_P")
    emg_key = emg_key or data_cfg.get("channels", {}).get("emg", "EMG_diff")
    prep_cfg = PreprocessConfig(sample_rate=float(sample_rate) if sample_rate else PreprocessConfig().sample_rate)

    processed_subjects = preprocess_to_h5(
        recordings_dir=recordings_dir,
        output_dir=preprocessed_dir,
        subjects=subjects,
        eeg_key=eeg_key,
        emg_key=emg_key,
        cache_dir=cache_dir,
        sample_rate=float(prep_cfg.sample_rate),
    )

    batch_size = batch_size or int(train_cfg.get("batch_size", 64))
    epochs = epochs or int(eval_cfg.get("finetune_epochs", 10))
    base_lr = float(train_cfg.get("optimizer", {}).get("lr", 5e-4))
    lr_mult = float(ft_settings.get("lr_multiplier", 0.1))
    lr_val = float(lr) if lr is not None and lr > 0 else base_lr * lr_mult
    weight_decay_val = (
        float(weight_decay)
        if weight_decay is not None and weight_decay >= 0
        else float(train_cfg.get("optimizer", {}).get("weight_decay", 1e-2))
    )
    seq_len = int(data_cfg.get("sequence_length", 11))
    train_ratio = float(train_ratio)

    torch_device = (
        torch.device(device)
        if device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    ft_cfg = FinetuneConfig(
        config_path=config_path,
        preprocessed_dir=preprocessed_dir,
        subjects=processed_subjects,
        batch_size=batch_size,
        max_epochs=epochs,
        sequence_length=seq_len,
        sample_rate=int(round(prep_cfg.sample_rate)),
        device=torch_device,
        lr=lr_val,
        weight_decay=weight_decay_val,
        lr_multiplier=lr_mult,
        freeze_backbone=bool(ft_settings.get("freeze_backbone", False)),
        train_ratio=train_ratio,
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    return finetune(ft_cfg)


def run_finetune_pipeline_from_paths(
    mat_path: Path,
    txt_path: Path,
    *,
    base_config: Path,
    eeg_key: str,
    emg_key: str,
    sample_rate: float,
    finetune_epochs: int,
    device: Optional[str] = None,
    train_ratio: float = 1.0,
) -> Tuple[Path, Path]:
    mat_dir = mat_path if mat_path.is_dir() else mat_path.parent
    temp_root = mat_dir / f".tdt_temp_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    raw_dir = temp_root / "raw"
    preprocessed_dir = temp_root / "preprocessed"
    cache_dir = temp_root / "cache"
    temp_config = temp_root / "config.yaml"

    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        stage_recordings(mat_path, txt_path, raw_dir)
        write_temp_config(
            base_config,
            temp_config,
            eeg_key=eeg_key,
            emg_key=emg_key,
            sample_rate=sample_rate,
            finetune_epochs=finetune_epochs,
            train_ratio=train_ratio,
            use_zscore=False,
        )
        best_ckpt, norm_path = run_finetune_pipeline_dir(
            recordings_dir=raw_dir,
            preprocessed_dir=preprocessed_dir,
            config_path=temp_config,
            cache_dir=cache_dir,
            epochs=finetune_epochs,
            device=device,
            train_ratio=train_ratio,
            eeg_key=eeg_key,
            emg_key=emg_key,
            sample_rate=sample_rate,
        )
        best_ckpt, norm_path = _move_artifacts_to_mat_dir(best_ckpt, norm_path, mat_dir)
        return best_ckpt, norm_path
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


def _unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def _move_artifacts_to_mat_dir(
    best_ckpt: Path,
    norm_path: Path,
    mat_dir: Path,
) -> Tuple[Path, Path]:
    mat_dir.mkdir(parents=True, exist_ok=True)
    target_ckpt = _unique_target(mat_dir / best_ckpt.name)
    target_norm = _unique_target(mat_dir / norm_path.name)
    shutil.move(str(best_ckpt), target_ckpt)
    shutil.move(str(norm_path), target_norm)
    return target_ckpt, target_norm
