from __future__ import annotations

import copy
import logging
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from ..constants import STATE_NAMES
from ..models import build_model
from ..utils import get_logger, load_config
from .visualization import RealtimePlotter

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from llm_eeg.configs import PreprocessConfig

logger = get_logger(__name__)


@dataclass(slots=True)
class ChannelMapping:
    """Definition of how to assemble a logical channel from Synapse parameters."""

    name: str
    operation: str = "identity"
    sources: Tuple[str, ...] = tuple()
    weights: Tuple[float, ...] = tuple()
    scale: float = 1.0

    @classmethod
    def identity(cls, name: str, parameter: str, *, scale: float = 1.0) -> ChannelMapping:
        return cls(name=name, operation="identity", sources=(parameter,), scale=scale)

    @classmethod
    def from_config(cls, entry: Dict[str, Any]) -> ChannelMapping:
        name = entry.get("name")
        if not name:
            raise ValueError("Channel entry must include a 'name'.")

        scale = float(entry.get("scale", 1.0))
        operation_raw = entry.get("operation")
        sources_raw = entry.get("sources")
        parameter = entry.get("parameter")
        reference = entry.get("reference")
        weights_raw = entry.get("weights")

        sources: Tuple[str, ...] | None = None
        weights: Tuple[float, ...] | None = None

        if sources_raw is not None:
            sources = tuple(str(s) for s in sources_raw)
        if weights_raw is not None:
            weights = tuple(float(w) for w in weights_raw)

        if operation_raw:
            operation = operation_raw.strip().lower()
        else:
            if reference:
                operation = "diff"
                if not parameter:
                    raise ValueError(f"Channel '{name}' with reference requires 'parameter'.")
                sources = (str(parameter), str(reference))
            elif sources:
                operation = "weighted_sum"
            else:
                operation = "identity"
                if not parameter:
                    raise ValueError(f"Channel '{name}' requires 'parameter' when no sources are given.")
                sources = (str(parameter),)

        if operation in {"identity", "raw"}:
            if sources is None:
                if not parameter:
                    raise ValueError(f"Channel '{name}' requires a parameter for identity operation.")
                sources = (str(parameter),)
            return cls(name=name, operation="identity", sources=sources, scale=scale)

        if operation in {"diff", "difference", "subtract"}:
            if sources is None:
                if not (parameter and reference):
                    raise ValueError(
                        f"Channel '{name}' diff operation needs either 'sources' (>=2) or both 'parameter' and 'reference'."
                    )
                sources = (str(parameter), str(reference))
            if len(sources) < 2:
                raise ValueError(f"Channel '{name}' diff operation requires at least two sources.")
            return cls(name=name, operation="diff", sources=sources, scale=scale)

        if operation in {"weighted_sum", "linear_combination", "mix"}:
            if sources is None or len(sources) == 0:
                raise ValueError(f"Channel '{name}' weighted_sum operation requires 'sources'.")
            if weights and len(weights) != len(sources):
                raise ValueError(f"Channel '{name}' weights length must match number of sources.")
            if not weights:
                weights = tuple(1.0 for _ in sources)
            return cls(
                name=name,
                operation="weighted_sum",
                sources=sources,
                weights=weights,
                scale=scale,
            )

        raise ValueError(f"Unsupported channel operation '{operation}' for channel '{name}'.")

    @property
    def required_parameters(self) -> Tuple[str, ...]:
        return self.sources

    def compute(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.sources:
            raise ValueError(f"Channel '{self.name}' has no sources defined.")

        if self.operation == "identity":
            arr = data[self.sources[0]].astype(np.float32, copy=True)
        elif self.operation == "diff":
            arr = data[self.sources[0]].astype(np.float32, copy=True)
            for src in self.sources[1:]:
                arr -= data[src].astype(np.float32, copy=False)
        elif self.operation == "weighted_sum":
            arr = np.zeros_like(data[self.sources[0]], dtype=np.float32)
            weights = self.weights or tuple(1.0 for _ in self.sources)
            for src, weight in zip(self.sources, weights):
                arr += data[src].astype(np.float32, copy=False) * float(weight)
        else:  # pragma: no cover - defensive path
            raise ValueError(f"Unknown channel operation '{self.operation}'.")

        if self.scale != 1.0:
            arr *= float(self.scale)
        return arr


@dataclass(slots=True)
class PublishTarget:
    """Optional destination in Synapse to publish model predictions."""

    gizmo: str
    parameter: str
    as_text: bool = False


@dataclass(slots=True)
class OnlineInferenceConfig:
    """Runtime configuration for streaming online inference."""

    config_path: Path
    checkpoint_path: Path
    synapse_host: str = "localhost"
    synapse_mode: Optional[str] = None  # "Idle", "Preview", "Record"
    gizmo: str = ""
    channels: Sequence[ChannelMapping] = field(default_factory=list)
    publish_target: Optional[PublishTarget] = None
    # Original (incoming) sample rate from the device. Can be fractional (e.g. 305.1758).
    sample_rate: float = 256.0
    epoch_length_seconds: float = 5.0
    poll_interval_seconds: float = 0.5
    smoothing_window: int = 1
    min_confidence: float = 0.0
    plot_channels: Sequence[str] = field(default_factory=list)
    plot_window_seconds: float = 30.0
    plot_y_limits: Optional[Tuple[float, float]] = None
    # Optional per-channel z-score normalisation (means and stds length must match channels).
    normalization_means: Optional[Tuple[float, ...]] = None
    normalization_stds: Optional[Tuple[float, ...]] = None
    # Optional preprocessing overrides (e.g., bands, filter order) for LLM pipeline.
    preprocess: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("At least one channel mapping must be provided.")
        if not self.gizmo:
            raise ValueError("A Synapse gizmo name is required to read buffers.")


@dataclass(slots=True)
class ChannelBinding:
    parameter: str
    stream: Any
    channel_index: int


class _OnlineBandpassFilter:
    """Stateful causal bandpass filter matching the simulator's real-time path."""

    def __init__(self, sos_filters: Sequence[np.ndarray]):
        if not sos_filters:
            raise ValueError("At least one SOS filter is required for bandpass filtering.")
        self._filters = list(sos_filters)
        self._state = [np.zeros((sos.shape[0], 2), dtype=np.float32) for sos in self._filters]

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        if epoch.ndim != 2:
            raise ValueError("Expected epoch shape (channels, samples) for bandpass filtering.")
        filtered = np.empty_like(epoch, dtype=np.float32)
        for ch in range(epoch.shape[0]):
            idx = min(ch, len(self._filters) - 1)
            sos = self._filters[idx]
            zi = self._state[idx]
            out, zf = signal.sosfilt(sos, epoch[ch].astype(np.float32, copy=False), zi=zi)
            self._state[idx] = zf
            filtered[ch] = out
        return filtered


class _ExponentialMovingStandardizer:
    """Streaming exponential moving standardization mirroring TestOnline behaviour."""

    def __init__(self, alpha: float = 0.999, init_window: Optional[int] = None, eps: float = 1e-6):
        self.alpha = float(alpha)
        self.init_window = init_window
        self.eps = float(eps)
        self._mu: Optional[np.ndarray] = None
        self._var: Optional[np.ndarray] = None
        self._initialised = False

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        if epoch.ndim != 2:
            raise ValueError("Expected epoch shape (channels, samples) for standardization.")
        x = epoch.astype(np.float32, copy=False)
        channels, samples = x.shape
        if not self._initialised:
            window = x if self.init_window is None else x[:, : max(1, min(samples, int(self.init_window)))]
            self._mu = window.mean(axis=1, keepdims=True).astype(np.float32)
            self._var = np.clip(window.var(axis=1, keepdims=True).astype(np.float32), self.eps, None)
            self._initialised = True
        assert self._mu is not None and self._var is not None  # for type checkers
        mu = self._mu
        var = self._var
        out = np.empty_like(x, dtype=np.float32)
        for idx in range(samples):
            sample = x[:, idx : idx + 1]
            mu = (1.0 - self.alpha) * sample + self.alpha * mu
            var = (1.0 - self.alpha) * (sample - mu) ** 2 + self.alpha * var
            std = np.sqrt(np.maximum(var, self.eps))
            out[:, idx] = ((sample - mu) / std)[:, 0]
        self._mu = mu
        self._var = var
        return out


class _LLMPreprocessor:
    """Bandpass + exponential standardization pipeline aligned with TestOnline simulator."""

    def __init__(self, cfg: "PreprocessConfig"):
        from llm_eeg.preprocessing import build_channel_filters

        self._bandpass = _OnlineBandpassFilter(build_channel_filters(cfg))
        self._standardizer = _ExponentialMovingStandardizer(init_window=cfg.samples_per_epoch())

    def __call__(self, epoch: np.ndarray) -> np.ndarray:
        filtered = self._bandpass(epoch)
        return self._standardizer(filtered)


class SynapseSleepStagePredictor:
    """Stream EEG/EMG data via APIStreamer and feed it into the configured sleep staging model."""

    def __init__(self, cfg: OnlineInferenceConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device) if device else self._select_device()

        self.channels = list(cfg.channels)
        self._channel_names = [ch.name for ch in self.channels]
        self._base_parameters = sorted({param for ch in self.channels for param in ch.required_parameters})
        if not self._base_parameters:
            raise ValueError("No Synapse parameters configured for channel acquisition.")
        self._parameter_endpoints: Dict[str, Tuple[str, str]] = {
            param: self._resolve_endpoint(param, default=self.cfg.gizmo) for param in self._base_parameters
        }

        try:
            model_config = self._load_model_config(cfg.config_path)
        except Exception as exc:
            logger.warning(
                "Failed to load model config from %s (%s); falling back to minimal defaults.",
                cfg.config_path,
                exc,
            )
            model_config = {"data": {"sequence_length": 11, "sample_rate": cfg.sample_rate}, "model": {}}
        self.sequence_length = int(model_config.get("data", {}).get("sequence_length", 11))
        base_sample_rate = float(model_config.get("data", {}).get("sample_rate", cfg.sample_rate))
        model_name = str(model_config.get("model", {}).get("name", "")).lower()
        force_llm = model_name == "causal_transformer"

        self._use_llm_model = False
        self._preprocess_cfg: Optional["PreprocessConfig"] = None
        self.model, model_meta = self._load_model(model_config, cfg.checkpoint_path)
        if model_meta.get("sequence_length"):
            self.sequence_length = int(model_meta["sequence_length"])
        if model_meta.get("llm") or force_llm:
            if force_llm and not model_meta.get("llm"):
                logger.info(
                    "CausalTransformer config detected; enabling LLM preprocessing for compatibility "
                    "with TestOnline-style pipelines."
                )
            self._use_llm_model = True
            self._preprocess_cfg = self._build_preprocess_config(cfg, base_sample_rate)
            self._target_sample_rate = float(self._preprocess_cfg.sample_rate)
            self._llm_preprocessor = _LLMPreprocessor(self._preprocess_cfg)
        else:
            self._target_sample_rate = base_sample_rate
            self._llm_preprocessor = None

        self.epoch_samples = int(round(self._target_sample_rate * cfg.epoch_length_seconds))
        if self.epoch_samples <= 0:
            raise ValueError("Epoch length and sample rate must produce a positive sample count.")
        self.model.eval()

        if self._use_llm_model and self._preprocess_cfg is not None:
            logger.info(
                "Loaded CausalTransformer checkpoint | context_window=%d sample_rate=%.3f eeg_band=%s emg_band=%s",
                self.sequence_length,
                self._target_sample_rate,
                getattr(self._preprocess_cfg, "eeg_band", None),
                getattr(self._preprocess_cfg, "emg_band", None),
            )

        # Set up optional z-score normalisation
        self._norm_means: Optional[np.ndarray] = None
        self._norm_stds: Optional[np.ndarray] = None
        if (not self._use_llm_model) and cfg.normalization_means is not None and cfg.normalization_stds is not None:
            means = np.asarray(cfg.normalization_means, dtype=np.float32)
            stds = np.asarray(cfg.normalization_stds, dtype=np.float32)
            if means.shape[0] != len(self._channel_names) or stds.shape[0] != len(self._channel_names):
                logger.warning(
                    "Normalization stats length (%d/%d) does not match number of channels (%d). Ignoring.",
                    means.shape[0],
                    stds.shape[0],
                    len(self._channel_names),
                )
            else:
                # Avoid divide-by-zero at inference
                stds = np.where(stds == 0.0, 1.0, stds)
                self._norm_means = means
                self._norm_stds = stds

        self._buffer: Deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self._pred_history: Deque[int] = deque(maxlen=max(cfg.smoothing_window, 1))
        self._synthetic_epoch_index: int = -1
        self._stop_event = threading.Event()
        self._publish_disabled = False
        self._diagnostic_samples_logged = 0

        self._stream_history_seconds = max(
            cfg.epoch_length_seconds * (self.sequence_length + 2),
            cfg.plot_window_seconds + cfg.epoch_length_seconds,
            30.0,
        )
        self._streamers: Dict[str, Any] = {}
        self._bindings: Dict[str, ChannelBinding] = {}
        self._stream_bindings: Dict[Any, list[ChannelBinding]] = {}
        self._next_epoch_end: Optional[float] = None
        self._epoch_hook: Optional[Callable[[Dict[str, Any]], None]] = None
        self._last_epoch_bounds: Optional[Tuple[float, float]] = None
        self._initialise_streamers(cfg.synapse_host)

        first_streamer = next(iter(self._streamers.values()))
        self._synapse = first_streamer.syn
        self._maybe_set_mode(cfg.synapse_mode)

        self._plot_channels = self._determine_plot_channels()
        self._plotter: Optional[RealtimePlotter] = None
        if self._plot_channels:
            try:
                self._plotter = RealtimePlotter(
                    self._plot_channels,
                    cfg.sample_rate,
                    cfg.plot_window_seconds,
                    cfg.plot_y_limits,
                )
            except Exception:
                logger.warning("Failed to initialise real-time plotter; continuing without visualisation.", exc_info=True)

        logger.info(
            "Online predictor initialised | host=%s gizmos=%s channels=%s epoch_samples=%d sequence=%d device=%s",
            cfg.synapse_host,
            list(self._streamers.keys()),
            self._channel_names,
            self.epoch_samples,
            self.sequence_length,
            self.device,
        )

    # ------------------------------------------------------------------ Stream setup
    def _initialise_streamers(self, host: str) -> None:
        try:
            import tdt
        except ImportError as exc:  # pragma: no cover - dependency check
            raise ImportError(
                "The 'tdt' package is required to communicate with Synapse. "
                "Install it via 'pip install tdt'."
            ) from exc

        unique_gizmos: Dict[str, Dict[str, Any]] = {}
        for parameter, (gizmo, param_name) in self._parameter_endpoints.items():
            streamer = unique_gizmos.get(gizmo, {}).get("streamer")
            if streamer is None:
                # verbose=True to log polling latency stats from APIStreamer during runtime
                streamer = tdt.APIStreamer(
                    host=host,
                    gizmo=gizmo,
                    history_seconds=self._stream_history_seconds,
                    verbose=True,
                )
                self._streamers[gizmo] = streamer
                unique_gizmos[gizmo] = {"streamer": streamer, "bindings": []}
                logger.info(
                    "Started streamer for %s | nchan=%s fs=%.2f history=%d",
                    gizmo,
                    getattr(streamer, "nchan", "?"),
                    getattr(streamer, "fs", 0.0),
                    getattr(streamer, "data", np.zeros((1, 1), dtype=np.float32)).shape[-1],
                )
            index = self._resolve_stream_channel_index(param_name, streamer)
            binding = ChannelBinding(parameter=parameter, stream=streamer, channel_index=index)
            self._bindings[parameter] = binding
            unique_gizmos[gizmo]["bindings"].append(binding)
            logger.debug(
                "Mapped %s -> gizmo=%s channel=%d",
                parameter,
                gizmo,
                index + 1,
            )

        for gizmo_data in unique_gizmos.values():
            streamer = gizmo_data["streamer"]
            bindings = list(gizmo_data["bindings"])
            self._stream_bindings[streamer] = bindings

    @staticmethod
    def _resolve_stream_channel_index(parameter: str, streamer: Any) -> int:
        text = parameter.strip().lower()
        if text.startswith("data"):
            suffix = text[4:]
            if suffix.isdigit():
                idx = int(suffix) - 1
                if idx < 0:
                    raise ValueError(f"Invalid channel index derived from '{parameter}'.")
                nchan = getattr(streamer, "nchan", None)
                if isinstance(nchan, int) and nchan > 0 and idx >= nchan:
                    raise ValueError(f"Channel index {idx} out of range for streamer with {nchan} channels.")
                return idx
        raise ValueError(
            f"Parameter '{parameter}' is not a recognised streamer data channel (expected names like 'data1')."
        )

    def register_epoch_hook(self, func: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Register a callback invoked with epoch details after each inference step."""
        self._epoch_hook = func

    # ------------------------------------------------------------------ Runtime loop
    def run(self) -> None:
        logger.info("Starting APIStreamer inference loop.")
        try:
            while not self._stop_event.is_set():
                epoch = self._collect_epoch()
                if epoch is None:
                    time.sleep(self.cfg.poll_interval_seconds)
                    continue
                self._synthetic_epoch_index += 1
                self._handle_epoch(self._synthetic_epoch_index, epoch)
                time.sleep(max(self.cfg.poll_interval_seconds, 0.05))
        except KeyboardInterrupt:  # pragma: no cover - interactive path
            logger.info("Received KeyboardInterrupt. Stopping.")
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop_event.set()
        if self._plotter:
            self._plotter.close()

    def signal_stop(self) -> None:
        """Request the run loop to stop without touching UI resources (thread-safe)."""
        self._stop_event.set()

    def _collect_epoch(self) -> Optional[np.ndarray]:
        if not self._bindings:
            return None

        stream_payloads: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        for stream, bindings in self._stream_bindings.items():
            timestamps, data = stream.get_data()
            ts = np.asarray(timestamps, dtype=np.float64)
            values = np.asarray(data, dtype=np.float32)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if ts.size == 0 or values.shape[1] == 0:
                return None
            stream_payloads[stream] = (ts, values)

        latest_common = min(ts[-1] for ts, _ in stream_payloads.values())
        earliest_available = min(ts[0] for ts, _ in stream_payloads.values())

        if self._next_epoch_end is None:
            if latest_common - earliest_available < self.cfg.epoch_length_seconds:
                return None
            self._next_epoch_end = latest_common

        if latest_common + 1e-6 < self._next_epoch_end:
            return None

        start_time = self._next_epoch_end - self.cfg.epoch_length_seconds
        if any(ts[0] > start_time for ts, _ in stream_payloads.values()):
            return None

        target_times = np.linspace(
            start_time,
            self._next_epoch_end,
            self.epoch_samples,
            endpoint=False,
        )

        raw_channels: Dict[str, np.ndarray] = {}
        for parameter, binding in self._bindings.items():
            ts, values = stream_payloads[binding.stream]
            channel = values[binding.channel_index]
            resampled = np.interp(target_times, ts, channel).astype(np.float32)
            raw_channels[parameter] = resampled

        self._last_epoch_bounds = (start_time, self._next_epoch_end)
        self._next_epoch_end += self.cfg.epoch_length_seconds
        logical_channels = []
        for ch in self.channels:
            try:
                logical = ch.compute(raw_channels)
            except KeyError as exc:
                logger.error("Channel '%s' requires parameter '%s', which is unavailable.", ch.name, exc.args[0])
                return None
            logical_channels.append(logical)

        epoch = np.stack(logical_channels, axis=0)
        if self._llm_preprocessor is not None:
            epoch = self._llm_preprocessor(epoch)
        self._update_plot(epoch)
        return epoch

    def _update_plot(self, epoch: np.ndarray) -> None:
        if not self._plotter:
            return
        payload: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(self._channel_names):
            if name in self._plot_channels:
                payload[name] = epoch[idx]
        if payload:
            self._plotter.update(payload)

    # ------------------------------------------------------------------ Inference helpers
    def _build_preprocess_config(self, cfg: OnlineInferenceConfig, default_sample_rate: float) -> "PreprocessConfig":
        try:
            from llm_eeg.configs import PreprocessConfig
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "Loading CausalTransformer checkpoints requires the llm_eeg module to be available."
            ) from exc

        params = cfg.preprocess or {}
        sample_rate = float(params.get("sample_rate", default_sample_rate))
        epoch_seconds = float(params.get("epoch_seconds", cfg.epoch_length_seconds))
        filter_order = int(params.get("filter_order", 4))

        def _as_band(value: Any, fallback: Tuple[float, float]) -> Tuple[float, float]:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    return (float(value[0]), float(value[1]))
                except (TypeError, ValueError):
                    return fallback
            return fallback

        eeg_band = _as_band(params.get("eeg_band"), (0.5, 30.0))
        emg_band = _as_band(params.get("emg_band"), (15.0, 100.0))

        return PreprocessConfig(
            sample_rate=sample_rate,
            epoch_seconds=epoch_seconds,
            filter_order=filter_order,
            eeg_band=eeg_band,
            emg_band=emg_band,
        )

    def _apply_normalization(self, epoch: np.ndarray) -> np.ndarray:
        """
        Apply per-channel z-score normalisation to the incoming epoch if stats are available.
        epoch shape: (channels, samples)
        """
        if self._use_llm_model:
            return epoch
        if self._norm_means is None or self._norm_stds is None:
            return epoch
        try:
            return (epoch - self._norm_means[:, None]) / self._norm_stds[:, None]
        except Exception:
            logger.exception("Failed to apply normalization; continuing without.")
            return epoch

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # pragma: no cover - macOS path
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model_config(self, path: Path) -> Dict:
        config = load_config(path)
        config.setdefault("data", {})
        experiments = config.get("experiments", [])
        preferred = {"conv_transformer", "causal_transformer"}
        exp_cfg = next((exp for exp in experiments if exp.get("name") in preferred), None)
        if exp_cfg:
            name = exp_cfg.get("name", "conv_transformer")
            config["model"] = copy.deepcopy(exp_cfg.get("model", {"name": name}))
            config.setdefault("data", {})["sequence_length"] = exp_cfg.get(
                "sequence_length",
                config.get("data", {}).get("sequence_length", 11),
            )
        elif not config.get("model"):
            # Fallback to the first experiment definition if available
            if experiments:
                config["model"] = copy.deepcopy(experiments[0].get("model", {}))
                if "sequence_length" in experiments[0]:
                    config.setdefault("data", {})["sequence_length"] = experiments[0]["sequence_length"]
            else:
                raise ValueError(
                    "Configuration does not define a model. Provide model.name or add an experiment configuration."
                )
        return config

    def _load_model(self, config: Dict, checkpoint_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        meta: Dict[str, Any] = {}
        if isinstance(payload, dict) and "model_state" in payload and "config" in payload:
            try:
                from llm_eeg.configs import EEGTransformerConfig
                from llm_eeg.models import build_model as build_llm_model
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "Checkpoint appears to come from TestOnline (CausalTransformer). "
                    "Install or include the llm_eeg module to load it."
                ) from exc

            cfg_obj = payload["config"]
            if isinstance(cfg_obj, dict):
                cfg_obj = EEGTransformerConfig(**cfg_obj)
            elif not isinstance(cfg_obj, EEGTransformerConfig):
                cfg_obj = EEGTransformerConfig(**cfg_obj.__dict__)
            model = build_llm_model("CausalTransformer", cfg_obj)
            state_dict = payload["model_state"]
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            meta.update({"llm": True, "sequence_length": getattr(cfg_obj, "context_window", None)})
            return model, meta

        model = build_model(config)
        state_dict = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        model.load_state_dict(state_dict)
        model.to(self.device)
        meta["llm"] = False
        return model, meta

    def _handle_epoch(self, epoch_index: int, epoch: np.ndarray) -> None:
        if self._diagnostic_samples_logged < 5:
            stats = ", ".join(
                f"{name}:mean={float(np.mean(channel)):.6e} std={float(np.std(channel)):.6e}"
                for name, channel in zip(self._channel_names, epoch)
            )
            logger.info("Epoch %d channel stats | %s", epoch_index, stats)
            self._diagnostic_samples_logged += 1

        # Optional per-channel z-score normalisation using finetune stats (non-LLM path)
        epoch = self._apply_normalization(epoch)
        self._buffer.append(epoch)
        if not self._use_llm_model and len(self._buffer) < self.sequence_length:
            logger.debug(
                "Buffering epoch %d (buffer size %d/%d); waiting for enough context.",
                epoch_index,
                len(self._buffer),
                self.sequence_length,
            )
            return

        seq_list = list(self._buffer)
        if len(seq_list) < self.sequence_length:
            seq_list = [seq_list[0]] * (self.sequence_length - len(seq_list)) + seq_list
        sequence = np.stack(seq_list[-self.sequence_length :], axis=0)
        batch = torch.from_numpy(sequence).unsqueeze(0)  # (1, seq, channels, samples)
        batch = batch.to(self.device)
        with torch.no_grad():
            if self._use_llm_model:
                logits = self.model(batch)
            else:
                logits = self.model({"sequence": batch})
            probabilities = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probabilities))
        pred_prob = float(probabilities[pred_idx])

        self._pred_history.append(pred_idx)
        smoothed_idx = self._smooth_predictions()

        pred_name = STATE_NAMES.get(smoothed_idx, f"class_{smoothed_idx}")

        if self._epoch_hook:
            bounds = self._last_epoch_bounds or (None, None)
            payload = {
                "epoch_index": epoch_index,
                "start_time": bounds[0],
                "end_time": bounds[1],
                "epoch": epoch.copy(),
                "raw_prediction": pred_idx,
                "smoothed_prediction": smoothed_idx,
                "probabilities": probabilities.copy(),
                "confidence": pred_prob,
                "channel_names": list(self._channel_names),
            }
            try:
                self._epoch_hook(payload)
            except Exception:
                logger.exception("Epoch hook raised an exception.")

        logger.info(
            "Epoch %d | prediction=%s (idx=%d, prob=%.3f, raw=%d)",
            epoch_index,
            pred_name,
            smoothed_idx,
            pred_prob,
            pred_idx,
        )

        if pred_prob >= self.cfg.min_confidence:
            self._publish_prediction(smoothed_idx, pred_prob)
        else:
            logger.debug(
                "Prediction below confidence threshold %.2f (%.2f). Skipping publish.",
                self.cfg.min_confidence,
                pred_prob,
            )

    def _smooth_predictions(self) -> int:
        if len(self._pred_history) == 0:
            return 0
        if len(self._pred_history) == 1 or self.cfg.smoothing_window <= 1:
            return self._pred_history[-1]
        counts = np.bincount(list(self._pred_history))
        return int(np.argmax(counts))

    def _publish_prediction(self, idx: int, confidence: float) -> None:
        if not self.cfg.publish_target or self._publish_disabled:
            return
        target = self.cfg.publish_target
        try:
            if target.as_text:
                label = STATE_NAMES.get(idx, str(idx))
                payload = f"{label}:{confidence:.2f}"
                self._synapse.setParameterValue(target.gizmo, target.parameter, payload)
            else:
                self._synapse.setParameterValue(target.gizmo, target.parameter, int(idx))
        except Exception as exc:
            message = str(exc)
            if "404" in message or "Not found" in message:
                logger.warning(
                    "Publish target %s.%s not found (error: %s). Disabling further publish attempts.",
                    target.gizmo,
                    target.parameter,
                    message,
                )
                self._publish_disabled = True
                return
            logger.exception("Failed to publish prediction to %s.%s", target.gizmo, target.parameter)

    # ------------------------------------------------------------------ Utilities
    def _maybe_set_mode(self, mode: Optional[str]) -> None:
        if not mode:
            return
        mode_str = mode.lower()
        valid_modes = {"idle": 0, "preview": 2, "standby": 1, "record": 3}
        if mode_str not in valid_modes and mode_str not in {"0", "1", "2", "3"}:
            logger.warning("Requested Synapse mode '%s' not recognised. Skipping mode change.", mode)
            return

        current_mode = self._synapse.getMode()
        desired = int(mode) if mode.isdigit() else valid_modes[mode_str]
        if current_mode == desired:
            return
        logger.info("Switching Synapse mode from %s to %s", current_mode, desired)
        try:
            self._synapse.setMode(desired)
        except Exception:  # pragma: no cover - defensive path
            logger.exception("Failed to switch Synapse mode.")

    def _determine_plot_channels(self) -> list[str]:
        if self.cfg.plot_channels:
            preferred = [name for name in self.cfg.plot_channels if name in self._channel_names]
            if preferred:
                return preferred
        defaults = [name for name in ("eeg", "emg") if name in self._channel_names]
        if defaults:
            return defaults
        return list(self._channel_names[: min(len(self._channel_names), 2)])

    def _resolve_endpoint(self, raw: str, *, default: Optional[str] = None) -> Tuple[str, str]:
        if raw is None:
            raise ValueError("Parameter name cannot be None.")
        text = str(raw).strip()
        if not text:
            raise ValueError("Parameter name cannot be empty.")

        default_gizmo = (default or self.cfg.gizmo or "").strip()
        gizmo = default_gizmo
        parameter = text

        if "." in text:
            candidate_gizmo, candidate_param = text.split(".", 1)
            candidate_gizmo = candidate_gizmo.strip()
            candidate_param = candidate_param.strip()
            if candidate_gizmo:
                gizmo = candidate_gizmo
            parameter = candidate_param

        gizmo = gizmo.strip()
        parameter = parameter.strip()
        if not gizmo:
            raise ValueError(f"Unable to resolve Synapse gizmo for parameter '{raw}'.")
        if not parameter:
            raise ValueError(f"Unable to resolve Synapse parameter name from '{raw}'.")
        return gizmo, parameter


# ---------------------------------------------------------------------- Config helpers
def build_runtime_config(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    synapse_host: str = "localhost",
    synapse_mode: Optional[str] = None,
    gizmo: str,
    channel_params: Dict[str, Any],
    publish_target: Optional[Tuple[str, str, bool]] = None,
    sample_rate: int = 256,
    epoch_length_seconds: float = 5.0,
    poll_interval_seconds: float = 0.5,
    smoothing_window: int = 1,
    min_confidence: float = 0.0,
    plot_channels: Optional[Sequence[str]] = None,
    plot_window_seconds: float = 30.0,
    plot_y_limits: Optional[Tuple[float, float]] = None,
    normalization_means: Optional[Sequence[float]] = None,
    normalization_stds: Optional[Sequence[float]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
) -> OnlineInferenceConfig:
    channels = []
    for name, value in channel_params.items():
        if isinstance(value, str):
            channels.append(ChannelMapping.identity(name=name, parameter=value))
        elif isinstance(value, dict):
            entry = {"name": name, **value}
            channels.append(ChannelMapping.from_config(entry))
        else:
            raise ValueError(f"Unsupported channel specification for '{name}': {value}")

    fixed_y_limits = None
    if plot_y_limits is not None:
        if len(plot_y_limits) != 2:
            raise ValueError("plot_y_limits must contain exactly two numbers (ymin, ymax).")
        low = float(plot_y_limits[0])
        high = float(plot_y_limits[1])
        if low >= high:
            raise ValueError("plot_y_limits[0] must be smaller than plot_y_limits[1].")
        fixed_y_limits = (low, high)

    publish = None
    if publish_target:
        publish = PublishTarget(
            gizmo=publish_target[0],
            parameter=publish_target[1],
            as_text=bool(publish_target[2]) if len(publish_target) > 2 else False,
        )

    return OnlineInferenceConfig(
        config_path=Path(config_path),
        checkpoint_path=Path(checkpoint_path),
        synapse_host=synapse_host,
        synapse_mode=synapse_mode,
        gizmo=gizmo,
        channels=channels,
        publish_target=publish,
        sample_rate=sample_rate,
        epoch_length_seconds=epoch_length_seconds,
        poll_interval_seconds=poll_interval_seconds,
        smoothing_window=smoothing_window,
        min_confidence=min_confidence,
        plot_channels=list(plot_channels or []),
        plot_window_seconds=float(plot_window_seconds),
        plot_y_limits=fixed_y_limits,
        normalization_means=tuple(float(x) for x in normalization_means) if normalization_means else None,
        normalization_stds=tuple(float(x) for x in normalization_stds) if normalization_stds else None,
        preprocess=preprocess,
    )


def _parse_norm_array(value: Any) -> Optional[Tuple[float, ...]]:
    """
    Accepts a list/tuple of numbers, or a string path to a JSON file with keys
    'means' and 'stds'. When a path is provided here, we return None and let the
    corresponding counterpart load both arrays together.
    """
    if value is None:
        return None
    # If a string is provided, treat it as a JSON path and return None here.
    if isinstance(value, str):
        try:
            path = Path(value)
            if path.suffix.lower() in {".json", ".jsn"} and path.exists():
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)  # type: ignore[name-defined]
                arr = payload.get("means") or payload.get("stds") or payload.get("values")
                if isinstance(arr, (list, tuple)) and len(arr) > 0:
                    return tuple(float(x) for x in arr)
                return None
        except Exception:
            return None
        return None
    if isinstance(value, (list, tuple)):
        try:
            return tuple(float(x) for x in value)
        except (TypeError, ValueError):
            return None
    return None


def parse_runtime_config(path: Path) -> OnlineInferenceConfig:
    payload = load_config(path)
    model_cfg = payload.get("model", {})
    syn_cfg = payload.get("synapse", {})
    runtime_cfg = payload.get("runtime", {})

    publish_cfg = syn_cfg.get("publish")
    publish = None
    if publish_cfg:
        publish = PublishTarget(
            gizmo=publish_cfg["gizmo"],
            parameter=publish_cfg["parameter"],
            as_text=bool(publish_cfg.get("as_text", False)),
        )

    channels_cfg = syn_cfg.get("channels", {})
    channels: list[ChannelMapping] = []
    if isinstance(channels_cfg, dict):
        for name, value in channels_cfg.items():
            if isinstance(value, str):
                channels.append(ChannelMapping.identity(name=name, parameter=value))
            elif isinstance(value, dict):
                entry = {"name": name, **value}
                channels.append(ChannelMapping.from_config(entry))
            else:
                raise ValueError(f"Unsupported channel specification for '{name}': {value}")
    elif isinstance(channels_cfg, list):
        for entry in channels_cfg:
            if not isinstance(entry, dict):
                raise ValueError(f"Channel entries in list form must be dictionaries, got {entry!r}.")
            channels.append(ChannelMapping.from_config(entry))
    else:
        raise ValueError("synapse.channels must be a mapping or a list of channel definitions.")

    plot_channels = runtime_cfg.get("plot_channels", [])
    if isinstance(plot_channels, str):
        plot_channels = [plot_channels]
    elif not isinstance(plot_channels, Iterable):
        plot_channels = []

    plot_y_limits_raw = runtime_cfg.get("plot_y_limits")
    plot_y_limits: Optional[Tuple[float, float]] = None
    if isinstance(plot_y_limits_raw, (list, tuple)) and len(plot_y_limits_raw) == 2:
        try:
            low = float(plot_y_limits_raw[0])
            high = float(plot_y_limits_raw[1])
        except (TypeError, ValueError):
            plot_y_limits = None
        else:
            if low < high:
                plot_y_limits = (low, high)

    # ----- Normalization configuration (z-score) -----
    norm_means: Optional[Tuple[float, ...]] = None
    norm_stds: Optional[Tuple[float, ...]] = None
    norm_cfg = runtime_cfg.get("normalization") or runtime_cfg.get("zscore")
    norm_path = runtime_cfg.get("normalization_path")
    try:
        if isinstance(norm_cfg, dict):
            if isinstance(norm_cfg.get("path"), (str, Path)):
                with Path(norm_cfg["path"]).open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj.get("means"), (list, tuple)) and isinstance(obj.get("stds"), (list, tuple)):
                    norm_means = tuple(float(x) for x in obj["means"])
                    norm_stds = tuple(float(x) for x in obj["stds"])
            else:
                if isinstance(norm_cfg.get("means"), (list, tuple)):
                    norm_means = tuple(float(x) for x in norm_cfg["means"])
                if isinstance(norm_cfg.get("stds"), (list, tuple)):
                    norm_stds = tuple(float(x) for x in norm_cfg["stds"])
        if norm_path and (norm_means is None or norm_stds is None):
            with Path(norm_path).open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj.get("means"), (list, tuple)) and isinstance(obj.get("stds"), (list, tuple)):
                norm_means = tuple(float(x) for x in obj["means"])
                norm_stds = tuple(float(x) for x in obj["stds"])
    except Exception:
        norm_means = None
        norm_stds = None
    # Fallback: direct arrays
    if norm_means is None:
        norm_means = _parse_norm_array(runtime_cfg.get("normalization_means"))
    if norm_stds is None:
        norm_stds = _parse_norm_array(runtime_cfg.get("normalization_stds"))

    # Resolve checkpoint path; allow a directory that contains best_model_*.pt
    ckpt_raw = model_cfg.get("checkpoint_path")
    ckpt_path = Path(ckpt_raw)
    if ckpt_path.is_dir():
        candidates = sorted(
            list(ckpt_path.glob("best_model_*.pt")) + list(ckpt_path.glob("best_model.pt")) + list(ckpt_path.glob("*.pt")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            ckpt_path = candidates[0]
        else:
            raise FileNotFoundError(f"No checkpoint *.pt found in directory: {ckpt_path}")

    # If user provided a directory to look for normalization stats, pick the newest normalization_stats_*.json
    if norm_means is None or norm_stds is None:
        norm_dir_raw = runtime_cfg.get("normalization_dir")
        if isinstance(norm_dir_raw, (str, Path)):
            nd = Path(norm_dir_raw)
            if nd.is_dir():
                js = sorted(
                    list(nd.glob("normalization_stats_*.json")) + list(nd.glob("*.json")),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if js:
                    try:
                        with js[0].open("r", encoding="utf-8") as f:
                            obj = json.load(f)
                        if isinstance(obj.get("means"), (list, tuple)) and isinstance(obj.get("stds"), (list, tuple)):
                            norm_means = tuple(float(x) for x in obj["means"])
                            norm_stds = tuple(float(x) for x in obj["stds"])
                    except Exception:
                        pass

    cfg_path_raw = model_cfg.get("config_path") or model_cfg.get("config")
    if cfg_path_raw:
        config_path = Path(cfg_path_raw)
    else:
        # Fallback to a sibling config.yaml or a placeholder; constructor will handle missing files gracefully.
        config_path = ckpt_path.with_name("config.yaml") if ckpt_path else Path("config.yaml")

    preprocess_cfg = runtime_cfg.get("preprocess")
    if not isinstance(preprocess_cfg, dict):
        preprocess_cfg = None

    return OnlineInferenceConfig(
        config_path=config_path,
        checkpoint_path=ckpt_path,
        synapse_host=syn_cfg.get("host", "localhost"),
        synapse_mode=syn_cfg.get("mode"),
        gizmo=syn_cfg["gizmo"],
        channels=channels,
        publish_target=publish,
        sample_rate=float(runtime_cfg.get("sample_rate", model_cfg.get("sample_rate", 256))),
        epoch_length_seconds=float(
            runtime_cfg.get("epoch_length_seconds", model_cfg.get("epoch_length_seconds", 5.0))
        ),
        poll_interval_seconds=float(runtime_cfg.get("poll_interval_seconds", 0.5)),
        smoothing_window=int(runtime_cfg.get("smoothing_window", 1)),
        min_confidence=float(runtime_cfg.get("min_confidence", 0.0)),
        plot_channels=list(plot_channels),
        plot_window_seconds=float(runtime_cfg.get("plot_window_seconds", 30.0)),
        plot_y_limits=plot_y_limits,
        normalization_means=norm_means,
        normalization_stds=norm_stds,
        preprocess=preprocess_cfg,
    )


def main_from_args(args: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run conv_transformer online inference via TDT APIStreamer.")
    parser.add_argument("--runtime-config", type=Path, help="YAML file describing Synapse and runtime settings.")
    parser.add_argument("--config", type=Path, help="Path to training config YAML.", required=False)
    parser.add_argument("--checkpoint", type=Path, help="Path to model checkpoint (.pt).", required=False)
    parser.add_argument("--synapse-host", type=str, default="localhost")
    parser.add_argument("--synapse-mode", type=str, default=None)
    parser.add_argument("--gizmo", type=str, help="Name of Synapse gizmo exposing buffers.")
    parser.add_argument(
        "--channel",
        action="append",
        default=[],
        help="Channel mapping in format NAME=PARAMETER. Provide once per channel.",
    )
    parser.add_argument("--publish", type=str, default=None, help="Publish target in format GIZMO.PARAM[.text]")
    parser.add_argument("--sample-rate", type=int, default=256)
    parser.add_argument("--epoch-seconds", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--smoothing", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--plot-channel", action="append", dest="plot_channels", default=[])
    parser.add_argument("--plot-window", type=float, default=30.0)
    parser.add_argument(
        "--plot-y-limits",
        nargs=2,
        type=float,
        metavar=("YMIN", "YMAX"),
        help="Fix y-axis limits in the realtime plot (same units as channels).",
    )
    parser.add_argument("--device", type=str, default=None)

    parsed = parser.parse_args(args=args)

    if parsed.runtime_config:
        cfg = parse_runtime_config(parsed.runtime_config)
    else:
        if not parsed.config or not parsed.checkpoint or not parsed.gizmo:
            parser.error("Either --runtime-config or (--config, --checkpoint, --gizmo, --channel ...) must be provided.")
        channel_params: Dict[str, Any] = {}
        if not parsed.channel:
            parser.error("At least one --channel NAME=PARAMETER entry is required.")
        for entry in parsed.channel:
            if "=" not in entry:
                parser.error(f"Invalid channel mapping '{entry}'. Expected NAME=PARAMETER.")
            name, parameter = entry.split("=", 1)
            channel_params[name.strip()] = parameter.strip()

        publish_tuple = None
        if parsed.publish:
            parts = parsed.publish.split(".")
            if len(parts) < 2:
                parser.error("Publish target must be GIZMO.PARAM[.text].")
            as_text = len(parts) > 2 and parts[2].lower() == "text"
            publish_tuple = (parts[0], parts[1], as_text)

        cfg = build_runtime_config(
            config_path=parsed.config,
            checkpoint_path=parsed.checkpoint,
            synapse_host=parsed.synapse_host,
            synapse_mode=parsed.synapse_mode,
            gizmo=parsed.gizmo,
            channel_params=channel_params,
            publish_target=publish_tuple,
            sample_rate=parsed.sample_rate,
            epoch_length_seconds=parsed.epoch_seconds,
            poll_interval_seconds=parsed.poll_interval,
            smoothing_window=parsed.smoothing,
            min_confidence=parsed.min_confidence,
            plot_channels=parsed.plot_channels,
            plot_window_seconds=parsed.plot_window,
            plot_y_limits=tuple(parsed.plot_y_limits) if parsed.plot_y_limits else None,
        )

    predictor = SynapseSleepStagePredictor(cfg, device=parsed.device)
    predictor.run()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main_from_args()
