from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import datetime as dt
import os
import shutil

import h5py
import torch
import yaml

from llm_eeg.configs import PreprocessConfig
from llm_eeg.data import discover_subject_infos, load_subject_epochs
from finetune import FinetuneConfig, finetune
from src.utils import load_config


def preprocess_to_h5(
    recordings_dir: Path,
    output_dir: Path,
    subjects: Sequence[str] | None,
    eeg_key: str,
    emg_key: str,
    cache_dir: Path | None,
    sample_rate: float,
) -> List[str]:
    """Load MAT/TXT pairs, preprocess, and export per-subject H5 files."""
    prep_cfg = PreprocessConfig(sample_rate=sample_rate)
    infos = discover_subject_infos(recordings_dir, data_format="mat")
    if not infos:
        raise FileNotFoundError(f"No MAT/TXT pairs found under {recordings_dir}")

    target = set(subjects) if subjects else None
    output_dir.mkdir(parents=True, exist_ok=True)
    processed: List[str] = []
    for info in infos:
        if target and info.subject_id not in target:
            continue
        record = load_subject_epochs(
            info.subject_id,
            recordings_dir,
            prep_cfg,
            cache_dir=cache_dir,
            data_format="mat",
        )
        if record.epochs.shape[0] == 0:
            print(f"[Skip] {info.subject_id}: no usable epochs after preprocessing.")
            continue
        out_path = output_dir / f"{info.subject_id}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset(f"{eeg_key}_epochs", data=record.epochs[:, 0, :], compression="gzip")
            f.create_dataset(f"{emg_key}_epochs", data=record.epochs[:, 1, :], compression="gzip")
            f.create_dataset("labels", data=record.labels, compression="gzip")
        processed.append(info.subject_id)
        print(f"[OK] {info.subject_id} -> {out_path}")
    if not processed:
        raise RuntimeError("No subjects were processed; check inputs and label coverage.")
    return processed


def _list_files(path: Path, suffix: str) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob(f"*{suffix}"))
    if path.is_file():
        if path.suffix.lower() != suffix:
            raise ValueError(f"Expected *{suffix} file, got {path.name}")
        return [path]
    raise FileNotFoundError(f"Path not found: {path}")


def _match_pairs(mat_paths: Sequence[Path], txt_paths: Sequence[Path]) -> List[Tuple[Path, Path, str]]:
    txt_map = {p.stem.lower(): p for p in txt_paths}
    pairs = []
    missing = []
    for mat_path in mat_paths:
        stem = mat_path.stem
        txt = txt_map.get(stem.lower())
        if txt is None:
            missing.append(stem)
        else:
            pairs.append((mat_path, txt, stem))
    if missing:
        raise RuntimeError(f"Missing TXT labels for: {', '.join(missing)}")
    if not pairs:
        raise RuntimeError("No MAT/TXT pairs matched.")
    return pairs


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _write_temp_config(
    base_config: Path,
    output_path: Path,
    *,
    eeg_key: str,
    emg_key: str,
    sample_rate: float,
    finetune_epochs: int,
    train_ratio: float,
) -> None:
    cfg = load_config(base_config)
    cfg.setdefault("data", {})
    cfg["data"].setdefault("channels", {})
    cfg["data"]["channels"]["eeg"] = eeg_key
    cfg["data"]["channels"]["emg"] = emg_key
    cfg["data"]["sample_rate"] = float(sample_rate)
    cfg["data"]["train_ratio"] = float(train_ratio)
    cfg["data"]["use_zscore"] = False

    cfg.setdefault("model", {})
    cfg["model"].setdefault("stft", {})
    cfg["model"]["stft"]["sample_rate"] = float(sample_rate)

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["finetune_epochs"] = int(finetune_epochs)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


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
        mats = _list_files(mat_path, ".mat")
        txts = _list_files(txt_path, ".txt")
        pairs = _match_pairs(mats, txts)
        for mat_file, txt_file, stem in pairs:
            _link_or_copy(mat_file, raw_dir / f"{stem}.mat")
            _link_or_copy(txt_file, raw_dir / f"{stem}.txt")

        _write_temp_config(
            base_config,
            temp_config,
            eeg_key=eeg_key,
            emg_key=emg_key,
            sample_rate=sample_rate,
            finetune_epochs=finetune_epochs,
            train_ratio=train_ratio,
        )

        cfg = load_config(temp_config)
        train_cfg = cfg.get("training", {})
        eval_cfg = cfg.get("evaluation", {})
        ft_settings = eval_cfg.get("finetune_settings", {})

        processed_subjects = preprocess_to_h5(
            recordings_dir=raw_dir,
            output_dir=preprocessed_dir,
            subjects=None,
            eeg_key=eeg_key,
            emg_key=emg_key,
            cache_dir=cache_dir,
            sample_rate=sample_rate,
        )

        batch_size = int(train_cfg.get("batch_size", 64))
        epochs = int(eval_cfg.get("finetune_epochs", finetune_epochs))
        base_lr = float(train_cfg.get("optimizer", {}).get("lr", 5e-4))
        lr_mult = float(ft_settings.get("lr_multiplier", 0.1))
        lr_val = base_lr * lr_mult
        weight_decay_val = float(train_cfg.get("optimizer", {}).get("weight_decay", 1e-2))
        seq_len = int(cfg.get("data", {}).get("sequence_length", 11))

        torch_device = (
            torch.device(device)
            if device
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        ft_cfg = FinetuneConfig(
            config_path=temp_config,
            preprocessed_dir=preprocessed_dir,
            subjects=processed_subjects,
            batch_size=batch_size,
            max_epochs=epochs,
            sequence_length=seq_len,
            sample_rate=int(round(sample_rate)),
            device=torch_device,
            lr=lr_val,
            weight_decay=weight_decay_val,
            lr_multiplier=lr_mult,
            freeze_backbone=bool(ft_settings.get("freeze_backbone", False)),
            train_ratio=train_ratio,
            seed=int(cfg.get("experiment", {}).get("seed", 42)),
        )
        return finetune(ft_cfg)
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
