from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.constants import STATE_NAMES
from src.online.synapse_predictor import SynapseSleepStagePredictor, parse_runtime_config


class RecordingWriter:
    """Append predictions to CSV/TXT as epochs are produced."""

    def __init__(self, output_dir: Path, *, session_start: datetime, epoch_length: float) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._session_start = session_start
        self._epoch_length = float(epoch_length)
        self._first_stream_time: Optional[float] = None

        self._channel_names: Optional[List[str]] = None
        self._epoch_indices: List[int] = []
        self._start_times: List[Optional[float]] = []
        self._end_times: List[Optional[float]] = []
        self._raw_predictions: List[int] = []
        self._smoothed_predictions: List[int] = []
        self._confidences: List[float] = []

        # Prepare streaming outputs
        self._csv_path = self.output_dir / "predictions.csv"
        self._txt_path = self.output_dir / "predictions.txt"
        # Write headers once
        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch_index",
                    "start_time",
                    "end_time",
                    "raw_prediction",
                    "raw_label",
                    "smoothed_prediction",
                    "smoothed_label",
                    "confidence",
                    "probabilities",
                ]
            )
        with self._txt_path.open("w", encoding="utf-8") as f:
            f.write("abs_start_time,start_seconds,one,abs_end_time,end_seconds,zero,label_id,label_name\n")

    def handle_epoch(self, payload: Dict[str, np.ndarray]) -> None:
        with self._lock:
            if self._channel_names is None:
                self._channel_names = list(payload.get("channel_names", []))

            self._epoch_indices.append(int(payload["epoch_index"]))
            start_time = float(payload["start_time"]) if payload.get("start_time") is not None else None
            end_time = float(payload["end_time"]) if payload.get("end_time") is not None else None
            self._start_times.append(start_time)
            self._end_times.append(end_time)
            self._raw_predictions.append(int(payload["raw_prediction"]))
            self._smoothed_predictions.append(int(payload["smoothed_prediction"]))
            self._confidences.append(float(payload["confidence"]))

            # Determine stream-relative seconds
            if start_time is not None and self._first_stream_time is None:
                self._first_stream_time = start_time
            ref = self._first_stream_time if self._first_stream_time is not None else 0.0
            # Fallback to index-based timing if stream timestamps are not available
            idx = self._epoch_indices[-1]
            start_seconds = (start_time - ref) if start_time is not None else idx * self._epoch_length
            end_seconds = (end_time - ref) if end_time is not None else (idx + 1) * self._epoch_length

            # Compute absolute start/end datetimes based on session_start
            abs_start = self._session_start + timedelta(seconds=float(start_seconds))
            abs_end = self._session_start + timedelta(seconds=float(end_seconds))

            # Map label
            raw = self._raw_predictions[-1]
            smooth = self._smoothed_predictions[-1]
            smooth_label = STATE_NAMES.get(smooth, f"class_{smooth}")

            # Append CSV row (keep the current CSV format/version)
            with self._csv_path.open("a", newline="", encoding="utf-8") as f_csv:
                writer = csv.writer(f_csv)
                writer.writerow(
                    [
                        idx,
                        f"{start_time:.6f}" if start_time is not None else "",
                        f"{end_time:.6f}" if end_time is not None else "",
                        raw,
                        STATE_NAMES.get(raw, f"class_{raw}"),
                        smooth,
                        smooth_label,
                        f"{self._confidences[-1]:.6f}",
                        json.dumps(np.asarray(payload.get("probabilities", []), dtype=np.float32).tolist(), ensure_ascii=False),
                    ]
                )
                f_csv.flush()

            # Append TXT row with the requested 8 columns
            with self._txt_path.open("a", encoding="utf-8") as f_txt:
                f_txt.write(
                    f"{abs_start.isoformat()},"
                    f"{start_seconds:.3f},"
                    f"1,"
                    f"{abs_end.isoformat()},"
                    f"{end_seconds:.3f},"
                    f"0,"
                    f"{smooth},"
                    f"{smooth_label}\n"
                )
                f_txt.flush()

    def finalize(
        self,
        sample_rate: float,
        epoch_length: float,
        session_start: datetime,
        session_end: datetime,
    ) -> None:
        with self._lock:
            metadata = {
                "recorded_epochs": len(self._epoch_indices),
                "channel_names": self._channel_names,
                "sample_rate": float(sample_rate),
                "epoch_length_seconds": float(epoch_length),
                "session_start": session_start.isoformat(),
                "session_end": session_end.isoformat(),
                "duration_seconds": (session_end - session_start).total_seconds(),
                "outputs": {
                    "csv": str(self._csv_path),
                    "txt": str(self._txt_path),
                },
                "notes": "Raw epoch data is not saved in this mode; only labels are recorded.",
            }
            with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)


def create_predictor(
    runtime_config: Path,
    *,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
    on_epoch: Optional[Callable[[Dict[str, np.ndarray]], None]] = None,
    synapse_mode: Optional[str] = None,
    eeg_parameter: Optional[str] = None,
    emg_sources: Optional[Sequence[str]] = None,
    normalization_path: Optional[Path] = None,
) -> SynapseSleepStagePredictor:
    cfg = parse_runtime_config(runtime_config)
    if checkpoint_path:
        cfg.checkpoint_path = checkpoint_path
    if synapse_mode:
        cfg.synapse_mode = synapse_mode
    if eeg_parameter or emg_sources:
        cfg.channels = _override_channels(cfg.channels, eeg_parameter, emg_sources)
    if normalization_path:
        means, stds = _load_normalization_stats(normalization_path)
        if means is None or stds is None:
            raise ValueError(f"Invalid normalization stats file: {normalization_path}")
        cfg.normalization_means = means
        cfg.normalization_stds = stds
    predictor = SynapseSleepStagePredictor(cfg, device=device)
    if on_epoch:
        predictor.register_epoch_hook(on_epoch)
    return predictor


def _override_channels(
    channels: Sequence[ChannelMapping],
    eeg_parameter: Optional[str],
    emg_sources: Optional[Sequence[str]],
) -> List[ChannelMapping]:
    updated: List[ChannelMapping] = []
    eeg_done = False
    emg_done = False
    emg_list = [s.strip() for s in emg_sources] if emg_sources else []
    if emg_list and len(emg_list) < 2:
        raise ValueError("EMG sources must include at least two parameters.")

    for ch in channels:
        name = ch.name.lower()
        if name == "eeg" and eeg_parameter:
            updated.append(
                ChannelMapping(
                    name=ch.name,
                    operation="identity",
                    sources=(str(eeg_parameter),),
                    scale=ch.scale,
                )
            )
            eeg_done = True
        elif name == "emg" and emg_list:
            updated.append(
                ChannelMapping(
                    name=ch.name,
                    operation="diff",
                    sources=tuple(emg_list),
                    scale=ch.scale,
                )
            )
            emg_done = True
        else:
            updated.append(ch)

    if eeg_parameter and not eeg_done:
        updated.append(ChannelMapping.identity(name="eeg", parameter=str(eeg_parameter)))
    if emg_list and not emg_done:
        updated.append(ChannelMapping(name="emg", operation="diff", sources=tuple(emg_list)))

    return updated


def _load_normalization_stats(path: Path) -> Tuple[Optional[Tuple[float, ...]], Optional[Tuple[float, ...]]]:
    if not path.exists():
        return None, None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        means = payload.get("means")
        stds = payload.get("stds")
        if isinstance(means, (list, tuple)) and isinstance(stds, (list, tuple)):
            return tuple(float(x) for x in means), tuple(float(x) for x in stds)
    except Exception:
        return None, None
    return None, None


def run_online(
    runtime_config: Path,
    *,
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
    output_dir: Optional[Path] = None,
    duration_hours: Optional[float] = None,
    no_plot: bool = False,
) -> None:
    cfg = parse_runtime_config(runtime_config)
    if no_plot:
        cfg.plot_channels = []
    if checkpoint_path:
        cfg.checkpoint_path = checkpoint_path

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = output_dir or Path("data/OnlineRecord") / f"session_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "runtime_config_copy.yaml").write_text(
        runtime_config.read_text(encoding="utf-8"), encoding="utf-8"
    )

    predictor = SynapseSleepStagePredictor(cfg, device=device)
    session_start = datetime.now()
    recorder = RecordingWriter(output_dir, session_start=session_start, epoch_length=cfg.epoch_length_seconds)
    predictor.register_epoch_hook(recorder.handle_epoch)

    duration_seconds: Optional[float] = None
    if duration_hours and duration_hours > 0:
        duration_seconds = duration_hours * 3600.0

    timer: Optional[threading.Timer] = None

    try:
        if duration_seconds is not None:
            print(f"Recording for {duration_seconds / 3600.0:.2f} hours.")
            timer = threading.Timer(duration_seconds, predictor.signal_stop)
            timer.start()
        else:
            print("Recording... press Ctrl+C to stop.")

        predictor.run()
    except KeyboardInterrupt:
        print("Stopping recording (user interrupt).")
    finally:
        if timer:
            timer.cancel()
        predictor.stop()

    session_end = datetime.now()
    recorder.finalize(cfg.sample_rate, cfg.epoch_length_seconds, session_start, session_end)
    print(f"Recording saved to {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online inference while recording EEG/EMG epochs and predictions."
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("online_runtime.yaml"),
        help="Path to runtime configuration YAML (default: online_runtime.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store recordings (default: data/OnlineRecord/<timestamp>).",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        help="Optional duration (in hours). If omitted, run until interrupted.",
    )
    parser.add_argument("--device", type=str, default=None, help="Inference device (cpu/cuda).")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable realtime plotting during recording (useful for long sessions).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_online(
        args.runtime_config,
        checkpoint_path=None,
        device=args.device,
        output_dir=args.output_dir,
        duration_hours=args.duration_hours,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
