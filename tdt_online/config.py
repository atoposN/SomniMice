from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from src.utils import load_config


def write_temp_config(
    base_config: Path,
    output_path: Path,
    *,
    eeg_key: str,
    emg_key: str,
    sample_rate: float,
    finetune_epochs: int,
    train_ratio: float = 1.0,
    use_zscore: bool = False,
) -> None:
    """Create a temporary config YAML with user-specified key parameters."""
    cfg = load_config(base_config)
    cfg.setdefault("data", {})
    cfg["data"].setdefault("channels", {})
    cfg["data"]["channels"]["eeg"] = eeg_key
    cfg["data"]["channels"]["emg"] = emg_key
    cfg["data"]["sample_rate"] = float(sample_rate)
    cfg["data"]["train_ratio"] = float(train_ratio)
    cfg["data"]["use_zscore"] = bool(use_zscore)

    cfg.setdefault("model", {})
    cfg["model"].setdefault("stft", {})
    cfg["model"]["stft"]["sample_rate"] = float(sample_rate)

    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["finetune_epochs"] = int(finetune_epochs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
