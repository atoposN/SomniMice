from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple


@dataclass(slots=True)
class PreprocessConfig:
    """Shared signal preprocessing hyperparameters."""

    sample_rate: float = 305.1758
    epoch_seconds: float = 5.0
    filter_order: int = 4
    eeg_band: Tuple[float, float] = (0.5, 30.0)
    emg_band: Tuple[float, float] = (15.0, 100.0)

    def samples_per_epoch(self) -> int:
        return int(round(self.sample_rate * self.epoch_seconds))

    def hop_length_samples(self) -> int:
        return self.samples_per_epoch()


@dataclass(slots=True)
class STFTConfig:
    """STFT hyperparameters for spectrogram tokens."""

    sample_rate: float = 305.1758
    n_fft: int = 256
    hop_length: int = 128
    win_length: int = 256
    f_max: float = 40.0


@dataclass(slots=True)
class EEGTransformerConfig:
    """Lightweight Transformer classifier options."""

    num_classes: int = 3
    patch_size: int = 32
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    use_time_tokens: bool = True
    use_spec_tokens: bool = True
    stft: STFTConfig = field(default_factory=STFTConfig)
    context_window: int = 11
    conv_base_filters: int = 16
    conv_kernel_size: int = 7
    conv_blocks: int = 2


@dataclass(slots=True)
class TrainingConfig:
    """High-level training parameters."""

    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    train_subjects: Sequence[str] = ()
    val_subjects: Sequence[str] = ()
    log_interval: int = 20
    checkpoint_dir: Path = Path("checkpoints")
    class_weighting: bool = True
    balanced_sampling: bool = True
    ema_decay: float = 0.0
    early_stop_patience: int = 8
    data_format: str = "mat"
    w_wce: float = 1.0
    w_ct: float = 1.0
