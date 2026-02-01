#!/usr/bin/env python
"""
python PRE.py --recordings-dir data/finetune/test20260105 --preprocessed-dir data/finetune/test20260105 --config training_config.yaml --device cuda
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
) -> List[str]:
    """Load MAT/TXT pairs, preprocess, and export per-subject H5 files."""
    prep_cfg = PreprocessConfig()
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: recordings (MAT/TXT) -> preprocess -> finetune best model."
    )
    p.add_argument("--recordings-dir", type=Path, default=Path("data/finetune"), help="Folder containing MAT/TXT pairs.")
    p.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=Path("data/finetune"),
        help="Output folder for generated .h5 files (also used for finetune input).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("training_config.yaml"),
        help="Training config (finetune settings).",
    )
    p.add_argument("--subjects", nargs="*", default=None, help="Optional subset of subject IDs to include.")
    p.add_argument("--cache-dir", type=Path, default=Path("cache_llm"), help="Cache for intermediate epochs.")
    p.add_argument("--epochs", type=int, default=None, help="Finetune epochs (override config).")
    p.add_argument("--batch-size", type=int, default=None, help="Finetune batch size (override config).")
    p.add_argument("--lr", type=float, default=-1.0, help="Override learning rate; <=0 uses config * lr_multiplier.")
    p.add_argument("--weight-decay", type=float, default=-1.0, help="Override weight decay; <0 uses config default.")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto).")
    p.add_argument("--train-ratio", type=float, default=None, help="Train split ratio within finetune set.")
    p.add_argument("--eeg-key", type=str, default=None, help="EEG channel key for H5 (default from config).")
    p.add_argument("--emg-key", type=str, default=None, help="EMG channel key for H5 (default from config).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    ft_settings = eval_cfg.get("finetune_settings", {})

    eeg_key = args.eeg_key or data_cfg.get("channels", {}).get("eeg", "EEG_P")
    emg_key = args.emg_key or data_cfg.get("channels", {}).get("emg", "EMG_diff")
    prep_cfg = PreprocessConfig()

    # Step 1: preprocess MAT/TXT -> .h5
    processed_subjects = preprocess_to_h5(
        recordings_dir=args.recordings_dir,
        output_dir=args.preprocessed_dir,
        subjects=args.subjects,
        eeg_key=eeg_key,
        emg_key=emg_key,
        cache_dir=args.cache_dir,
    )

    # Step 2: finetune using generated .h5 files
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 64))
    epochs = args.epochs or int(eval_cfg.get("finetune_epochs", 10))
    base_lr = float(train_cfg.get("optimizer", {}).get("lr", 5e-4))
    lr_mult = float(ft_settings.get("lr_multiplier", 0.1))
    lr = float(args.lr) if args.lr > 0 else base_lr * lr_mult
    weight_decay = float(args.weight_decay) if args.weight_decay >= 0 else float(
        train_cfg.get("optimizer", {}).get("weight_decay", 1e-2)
    )
    train_ratio = float(args.train_ratio) if args.train_ratio is not None else float(data_cfg.get("train_ratio", 0.8))
    seq_len = int(data_cfg.get("sequence_length", 11))

    sample_rate = float(prep_cfg.sample_rate)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    ft_cfg = FinetuneConfig(
        config_path=args.config,
        preprocessed_dir=args.preprocessed_dir,
        subjects=processed_subjects,
        batch_size=batch_size,
        max_epochs=epochs,
        sequence_length=seq_len,
        sample_rate=int(round(sample_rate)),
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        lr_multiplier=lr_mult,
        freeze_backbone=bool(ft_settings.get("freeze_backbone", False)),
        train_ratio=train_ratio,
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    best_ckpt, norm_path = finetune(ft_cfg)
    print(f"[Done] Best checkpoint: {best_ckpt}")
    print(f"[Done] Normalization stats: {norm_path}")
    print(
        "Update online_runtime.yaml -> model.checkpoint_path before online TDT. "
        "For CausalTransformer (LLM path), normalization_path is ignored."
    )


if __name__ == "__main__":
    main()
