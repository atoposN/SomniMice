"""
LLM-inspired EEG sleep staging toolkit.

This package provides:
- Unified preprocessing utilities (0.5–30 Hz bandpass, epoching, STFT).
- Dataset helpers for MAT/TXT mouse recordings.
- Lightweight Transformer/BERT-style classifiers for time, spectrogram and multimodal inputs.
- Training utilities to reach online-ready models.
"""

from . import configs, preprocessing, data, models, trainer, simulator

__all__ = ["configs", "preprocessing", "data", "models", "trainer", "simulator"]
