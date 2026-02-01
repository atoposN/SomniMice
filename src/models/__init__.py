"""Model registry for baseline and multimodal architectures."""

from . import baseline, conv_transformer, multimodal
from .factory import build_model

__all__ = ["baseline", "conv_transformer", "multimodal", "build_model"]
