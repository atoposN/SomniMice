from __future__ import annotations

import torch.nn as nn

from llm_eeg.configs import EEGTransformerConfig, STFTConfig
from llm_eeg.models import CausalTransformerClassifier


class CausalTransformerWrapper(nn.Module):
    """
    Adapter to use llm_eeg CausalTransformerClassifier within the existing factory/finetune pipeline.
    Expects batch["sequence"] shaped (B, W, C, T) where W == context window.
    """

    def __init__(self, model_cfg: dict) -> None:
        super().__init__()
        self.cfg = self._build_cfg(model_cfg)
        self.model = CausalTransformerClassifier(self.cfg)

    @staticmethod
    def _build_cfg(model_cfg: dict) -> EEGTransformerConfig:
        stft_cfg = model_cfg.get("stft", {})
        stft = STFTConfig(
            sample_rate=float(stft_cfg.get("sample_rate", 305.1758)),
            n_fft=int(stft_cfg.get("n_fft", 256)),
            hop_length=int(stft_cfg.get("hop_length", 128)),
            win_length=int(stft_cfg.get("win_length", 256)),
            f_max=float(stft_cfg.get("f_max", 40.0)),
        )
        # Allow CNN aliases for conv-related hyperparameters
        cnn_cfg = model_cfg.get("cnn", {})
        conv_base = int(model_cfg.get("conv_base_filters", cnn_cfg.get("base_filters", 16)))
        conv_kernel = int(model_cfg.get("conv_kernel_size", cnn_cfg.get("kernel_size", 7)))
        conv_blocks = int(model_cfg.get("conv_blocks", cnn_cfg.get("residual_blocks", 2)))

        # classifier num_classes can come from model.classifier.num_classes or model.num_classes
        classifier_cfg = model_cfg.get("classifier", {})
        num_classes = int(model_cfg.get("num_classes", classifier_cfg.get("num_classes", 3)))

        return EEGTransformerConfig(
            num_classes=num_classes,
            patch_size=int(model_cfg.get("patch_size", 32)),
            d_model=int(model_cfg.get("d_model", 128)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_time_tokens=bool(model_cfg.get("use_time_tokens", True)),
            use_spec_tokens=bool(model_cfg.get("use_spec_tokens", True)),
            stft=stft,
            context_window=int(model_cfg.get("context_window", model_cfg.get("sequence_length", 11))),
            conv_base_filters=conv_base,
            conv_kernel_size=conv_kernel,
            conv_blocks=conv_blocks,
        )

    def forward(self, batch):
        x = batch["sequence"] if isinstance(batch, dict) else batch
        return self.model(x)
