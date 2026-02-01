from __future__ import annotations

"""Utilities for running online inference pipelines."""

from .synapse_predictor import (
    OnlineInferenceConfig,
    SynapseSleepStagePredictor,
)

__all__ = [
    "OnlineInferenceConfig",
    "SynapseSleepStagePredictor",
]
