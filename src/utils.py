import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], destination: Path) -> None:
    """Persist resolved configuration for reproducibility."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def prepare_experiment_dir(config: Dict[str, Any], experiment_name: str | None = None) -> Path:
    """Create a timestamped experiment directory and mirror configuration."""
    exp_cfg = config.get("experiment", {})
    base_dir = Path(exp_cfg.get("output_dir", "outputs"))
    base_dir.mkdir(parents=True, exist_ok=True)

    if exp_cfg.get("timestamp", True):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = experiment_name or exp_cfg.get("name", "experiment")
        run_dir = base_dir / f"{name}_{stamp}"
    else:
        name = experiment_name or exp_cfg.get("name", "experiment")
        run_dir = base_dir / name

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    return run_dir


def get_logger(name: str, level: str | int = "INFO") -> logging.Logger:
    """Configure and return a module-level logger."""
    numeric_level = logging.getLevelName(level)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(numeric_level)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


def _cuda_memory_info(idx: int) -> Tuple[int, int]:
    device = torch.device(f"cuda:{idx}")
    if hasattr(torch.cuda, "mem_get_info"):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    else:
        with torch.cuda.device(device):
            total_bytes = torch.cuda.get_device_properties(idx).total_memory
            reserved = torch.cuda.memory_reserved(device)
            free_bytes = max(total_bytes - reserved, 0)
    return int(free_bytes), int(total_bytes)


def get_idle_gpu_indices(max_devices: int = 1, min_free_memory_ratio: float = 0.9) -> List[int]:
    """Return idle GPU indices, preferring higher-index devices first."""
    if not torch.cuda.is_available() or max_devices <= 0:
        return []

    max_devices = min(max_devices, 2)
    eligible: List[int] = []
    total_devices = torch.cuda.device_count()
    probe_devices = min(total_devices, 2)
    start = total_devices - probe_devices
    for idx in range(total_devices - 1, start - 1, -1):
        try:
            free_bytes, total_bytes = _cuda_memory_info(idx)
        except RuntimeError:
            continue
        ratio = free_bytes / max(total_bytes, 1)
        if ratio >= min_free_memory_ratio:
            eligible.append(idx)
        if len(eligible) >= max_devices:
            break
    return eligible


def _select_best_cuda_device() -> torch.device:
    """Return the CUDA device with the most available memory."""
    if torch.cuda.device_count() <= 1:
        return torch.device("cuda")

    best_idx = 0
    best_free_bytes = -1
    for idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{idx}")
        try:
            if hasattr(torch.cuda, "mem_get_info"):
                free_bytes, _ = torch.cuda.mem_get_info(device)
            else:
                # Fallback approximation using reserved memory if mem_get_info is not available.
                torch.cuda.set_device(device)
                total = torch.cuda.get_device_properties(idx).total_memory
                reserved = torch.cuda.memory_reserved(device)
                free_bytes = max(total - reserved, 0)
        except Exception:
            free_bytes = -1
        if free_bytes > best_free_bytes:
            best_idx = idx
            best_free_bytes = free_bytes
    torch.cuda.set_device(best_idx)
    return torch.device(f"cuda:{best_idx}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available computation device."""
    if prefer_gpu and torch.cuda.is_available():
        try:
            idle = get_idle_gpu_indices(max_devices=1)
            if idle:
                torch.cuda.set_device(idle[0])
                return torch.device(f"cuda:{idle[0]}")
            return torch.device("cpu")
        except Exception:
            return torch.device("cuda")
    if torch.backends.mps.is_available():  # macOS Metal backend
        return torch.device("mps")
    return torch.device("cpu")


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch to the desired device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(item, device) for item in batch)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    return batch


@dataclass(slots=True)
class EarlyStopping:
    """Simple early stopping helper that monitors improvements."""

    patience: int
    mode: str = "max"
    best_score: Optional[float] = None
    num_bad_epochs: int = 0
    best_state_dict: Optional[Dict[str, Any]] = None

    def step(self, score: float, model: torch.nn.Module) -> Tuple[bool, bool]:
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "max" and score > self.best_score:
            improved = True
        elif self.mode == "min" and score < self.best_score:
            improved = True

        if improved:
            self.best_score = score
            self.num_bad_epochs = 0
            self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.num_bad_epochs += 1

        stop = self.num_bad_epochs > self.patience
        return improved, stop


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
