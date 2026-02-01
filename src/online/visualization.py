from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


def _safe_import_matplotlib() -> Optional["matplotlib.pyplot"]:
    try:
        import matplotlib
        import matplotlib.pyplot as plt  # type: ignore

        if matplotlib.get_backend() == "Agg":
            try:
                matplotlib.use("TkAgg")  # pragma: no cover - best effort backend switch
            except Exception:
                pass
        plt.ion()
        return plt
    except Exception:
        logger.warning("Matplotlib is unavailable; real-time plots disabled.", exc_info=True)
        return None


class RealtimePlotter:
    """Realtime plotting helper executed on the main (inference) thread."""

    def __init__(
        self,
        channel_names: Sequence[str],
        sample_rate: float,
        window_seconds: float,
        y_limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        plt = _safe_import_matplotlib()
        self._plt = plt
        self.enabled = plt is not None and len(channel_names) > 0
        self.sample_rate = float(sample_rate)
        self.channel_names = list(channel_names)
        self.window_seconds = max(float(window_seconds), 1.0)
        self.window_samples = max(int(round(self.sample_rate * self.window_seconds)), 1)
        self._history: Dict[str, np.ndarray] = {name: np.zeros(0, dtype=np.float32) for name in self.channel_names}
        self._lines: list = []
        self._axes: list = []
        self._y_limits = tuple(float(v) for v in y_limits) if y_limits is not None else None

        if not self.enabled:
            return

        assert plt is not None  # static type checking
        figure, axes = plt.subplots(
            len(self.channel_names),
            1,
            sharex=True,
            figsize=(12, 3 * len(self.channel_names)),
        )
        if len(self.channel_names) == 1:
            axes = [axes]
        self._axes = list(axes)
        for ax, name in zip(self._axes, self.channel_names):
            line, = ax.plot([], [], lw=1.0)
            ax.set_ylabel(name)
            self._lines.append(line)
        self._axes[-1].set_xlabel("Time (s, relative)")
        figure.tight_layout()
        if self._y_limits:
            for ax in self._axes:
                ax.set_ylim(*self._y_limits)
        self._figure = figure
        try:
            plt.show(block=False)
        except TypeError:
            plt.show()
        plt.pause(0.001)

    def close(self) -> None:
        if not self.enabled:
            return
        assert self._plt is not None
        self._plt.pause(0.001)

    def update(self, data: Dict[str, np.ndarray]) -> None:
        if not self.enabled:
            return
        assert self._plt is not None

        for idx, name in enumerate(self.channel_names):
            if name not in data:
                continue
            arr = np.asarray(data[name], dtype=np.float32)
            if arr.size == 0:
                continue
            history = np.concatenate([self._history[name], arr])
            if history.size > self.window_samples:
                history = history[-self.window_samples :]
            self._history[name] = history

            t = np.linspace(-history.size / self.sample_rate, 0, history.size, endpoint=False)
            self._lines[idx].set_data(t, history)
            self._axes[idx].set_xlim(t[0], t[-1] if history.size > 1 else 0.0)

            if self._y_limits:
                ymin, ymax = self._y_limits
            else:
                ymin = float(np.min(history))
                ymax = float(np.max(history))
                if np.isclose(ymin, ymax):
                    margin = 1.0 if np.isclose(ymax, 0.0) else abs(ymax) * 0.1
                    ymin, ymax = ymin - margin, ymax + margin
                else:
                    pad = (ymax - ymin) * 0.1
                    ymin -= pad
                    ymax += pad
            self._axes[idx].set_ylim(ymin, ymax)

        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()
        self._plt.pause(0.001)
