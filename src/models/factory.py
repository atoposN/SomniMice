from __future__ import annotations

from typing import Dict

import torch.nn as nn

from .baseline import BaselineModelConfig, BaselineSequenceModel
from .conv_transformer import ConvTransformerConfig, ConvTransformerSequenceModel
from .multimodal import (
    FeatureOnlyModel,
    FullMultimodalModel,
    MultimodalConfig,
    RawFeatureFusionModel,
    RawSpectrogramFusionModel,
    SpectrogramOnlyModel,
)
from .causal_transformer import CausalTransformerWrapper


MODEL_REGISTRY = {
    "baseline_sequence": BaselineSequenceModel,
    "raw_plus_features": RawFeatureFusionModel,
    "raw_plus_spectrogram": RawSpectrogramFusionModel,
    "full_multimodal": FullMultimodalModel,
    "features_only": FeatureOnlyModel,
    "spectrogram_only": SpectrogramOnlyModel,
    "cnn_simple": BaselineSequenceModel,
    "cnn_residual": BaselineSequenceModel,
    "cnn_attention": BaselineSequenceModel,
    "conv_transformer": ConvTransformerSequenceModel,
    "causal_transformer": CausalTransformerWrapper,
}


def build_model(config: Dict) -> nn.Module:
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name", "baseline_sequence")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    components_cfg = model_cfg.get("components", {})
    shared_kwargs = {
        "input_channels": model_cfg.get("input_channels", 2),
        "base_filters": model_cfg.get("cnn", {}).get("base_filters", 32),
        "kernel_size": model_cfg.get("cnn", {}).get("kernel_size", 7),
        "residual_blocks": model_cfg.get("cnn", {}).get("residual_blocks", 2),
        "reduction_ratio": model_cfg.get("attention", {}).get("reduction_ratio", 8),
        "lstm_hidden_size": model_cfg.get("lstm", {}).get("hidden_size", 128),
        "lstm_layers": model_cfg.get("lstm", {}).get("num_layers", 2),
        "lstm_dropout": model_cfg.get("lstm", {}).get("dropout", 0.2),
        "bidirectional": model_cfg.get("lstm", {}).get("bidirectional", True),
        "classifier_hidden_dims": tuple(model_cfg.get("classifier", {}).get("hidden_dims", [128])),
        "num_classes": model_cfg.get("classifier", {}).get("num_classes", 3),
        "sequence_length": config.get("data", {}).get("sequence_length", 11),
        "use_residual": components_cfg.get("residual", True),
        "use_attention": components_cfg.get("channel_attention", True),
        "use_lstm": components_cfg.get("lstm", True),
    }

    if model_name in {"baseline_sequence", "cnn_simple", "cnn_residual", "cnn_attention"}:
        cfg = BaselineModelConfig(**shared_kwargs)
        if model_name == "cnn_simple":
            cfg.use_residual = components_cfg.get("residual", False)
            cfg.use_attention = components_cfg.get("channel_attention", False)
            cfg.use_lstm = components_cfg.get("lstm", False)
        elif model_name == "cnn_residual":
            cfg.use_residual = True
            cfg.use_attention = components_cfg.get("channel_attention", False)
            cfg.use_lstm = components_cfg.get("lstm", False)
        elif model_name == "cnn_attention":
            cfg.use_residual = True
            cfg.use_attention = True
            cfg.use_lstm = components_cfg.get("lstm", False)
    else:
        if model_name == "conv_transformer":
            transformer_cfg = model_cfg.get("transformer", {})
            conv_cfg = model_cfg.get("cnn", {})
            cfg = ConvTransformerConfig(
                input_channels=shared_kwargs["input_channels"],
                base_filters=conv_cfg.get("base_filters", 16),
                kernel_size=conv_cfg.get("kernel_size", 7),
                conv_blocks=conv_cfg.get("residual_blocks", 2),
                embed_dim=transformer_cfg.get("embed_dim", 128),
                num_heads=transformer_cfg.get("num_heads", 4),
                ff_multiplier=transformer_cfg.get("ff_multiplier", 4.0),
                num_layers=transformer_cfg.get("num_layers", 2),
                dropout=transformer_cfg.get("dropout", 0.1),
                sequence_length=shared_kwargs["sequence_length"],
                classifier_hidden_dims=shared_kwargs["classifier_hidden_dims"],
                num_classes=shared_kwargs["num_classes"],
            )
        elif model_name == "causal_transformer":
            # Handled inside CausalTransformerWrapper using llm_eeg configs; no cfg object needed here
            cfg = model_cfg
        else:
            cfg = MultimodalConfig(
                **shared_kwargs,
                feature_dims=tuple(model_cfg.get("feature_encoder", {}).get("hidden_dims", [256, 128])),
                spec_base_channels=model_cfg.get("spectrogram_encoder", {}).get("base_channels", 16),
                fusion_hidden_dims=tuple(model_cfg.get("fusion", {}).get("hidden_dims", [256, 128])),
            )

    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(cfg)
