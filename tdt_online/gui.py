from __future__ import annotations

import contextlib
import queue
import shutil
import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from src.constants import STATE_NAMES
from src.utils import load_config
from .pipeline import run_finetune_pipeline_from_paths
from .preprocess import detect_mat_channels, detect_mat_sample_rate
from .online import create_predictor


class QueueWriter:
    def __init__(self, q: queue.Queue) -> None:
        self._q = q

    def write(self, text: str) -> None:
        if text:
            self._q.put(("log", text))

    def flush(self) -> None:
        return


class OnlineGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("TDT Online Sleep Staging")
        self.geometry("920x660")

        self._queue: queue.Queue = queue.Queue()
        self._predictor = None
        self._online_thread: Optional[threading.Thread] = None
        self._prepare_thread: Optional[threading.Thread] = None
        self._last_best_ckpt: Optional[Path] = None
        self._detected_channels: list[str] = []
        self._prepare_running = False

        self._build_ui()
        self.after(100, self._drain_queue)

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self.prepare_tab = ttk.Frame(notebook)
        self.online_tab = ttk.Frame(notebook)
        notebook.add(self.prepare_tab, text="Prepare / Finetune")
        notebook.add(self.online_tab, text="Online Predict")

        self._build_prepare_tab()
        self._build_online_tab()

    def _build_prepare_tab(self) -> None:
        pad = {"padx": 8, "pady": 6}
        frame = ttk.LabelFrame(self.prepare_tab, text="Input")
        frame.pack(fill="x", padx=8, pady=8)

        self.mat_path = tk.StringVar(value="")
        self.txt_path = tk.StringVar(value="")

        self._row_selector(frame, "MAT file:", self.mat_path, self._choose_mat, 0)
        self._row_selector(frame, "TXT file:", self.txt_path, self._choose_txt, 1)

        params = ttk.LabelFrame(self.prepare_tab, text="Key Parameters")
        params.pack(fill="x", padx=8, pady=8)
        self.eeg_channel = tk.StringVar(value="")
        self.emg_channel = tk.StringVar(value="")
        self.sample_rate = tk.StringVar(value="305.1758")
        self.finetune_epochs = tk.StringVar(value="10")
        self.device_var = tk.StringVar(value="auto")

        ttk.Label(params, text="EEG channel name").grid(row=0, column=0, sticky="w", **pad)
        self.eeg_channel_combo = ttk.Combobox(
            params, textvariable=self.eeg_channel, values=[], width=16, state="disabled"
        )
        self.eeg_channel_combo.grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(params, text="EMG channel name").grid(row=0, column=2, sticky="w", **pad)
        self.emg_channel_combo = ttk.Combobox(
            params, textvariable=self.emg_channel, values=[], width=16, state="disabled"
        )
        self.emg_channel_combo.grid(row=0, column=3, sticky="w", **pad)
        ttk.Label(params, text="Sample rate").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(params, textvariable=self.sample_rate, width=16).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(params, text="Finetune epochs").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(params, textvariable=self.finetune_epochs, width=16).grid(row=1, column=3, sticky="w", **pad)
        ttk.Label(params, text="Device").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(params, textvariable=self.device_var, values=["auto", "cpu", "cuda"], width=12).grid(
            row=2, column=1, sticky="w", **pad
        )

        actions = ttk.Frame(self.prepare_tab)
        actions.pack(fill="x", padx=8, pady=8)
        self.prepare_btn = ttk.Button(actions, text="Run Preprocess + Finetune", command=self._start_prepare)
        self.prepare_btn.pack(side="left")
        self.save_btn = ttk.Button(actions, text="Save Best Model As...", command=self._save_best, state="disabled")
        self.save_btn.pack(side="left", padx=10)

        status_frame = ttk.Frame(self.prepare_tab)
        status_frame.pack(fill="x", padx=8, pady=4)
        self.prepare_status = tk.StringVar(value="Idle")
        ttk.Label(status_frame, textvariable=self.prepare_status).pack(side="left")
        self.prepare_progress = ttk.Progressbar(status_frame, mode="indeterminate", length=240)
        self.prepare_progress.pack(side="left", padx=10)

        log_frame = ttk.LabelFrame(self.prepare_tab, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_text = ScrolledText(log_frame, height=12)
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _build_online_tab(self) -> None:
        pad = {"padx": 8, "pady": 6}
        frame = ttk.LabelFrame(self.online_tab, text="Runtime")
        frame.pack(fill="x", padx=8, pady=8)

        self.runtime_config = tk.StringVar(value=str(Path("online_runtime.yaml").resolve()))
        self.checkpoint_path = tk.StringVar(value="")
        self.online_device = tk.StringVar(value="auto")
        self.synapse_mode = tk.StringVar(value="record")
        self.eeg_parameter = tk.StringVar(value="")
        self.emg_source1 = tk.StringVar(value="")
        self.emg_source2 = tk.StringVar(value="")
        self.norm_path = tk.StringVar(value="")

        self._row_selector(frame, "Runtime config (YAML):", self.runtime_config, self._choose_runtime, 0)
        self._row_selector(frame, "Best model (checkpoint):", self.checkpoint_path, self._choose_checkpoint, 1)

        ttk.Label(frame, text="Device").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(frame, textvariable=self.online_device, values=["auto", "cpu", "cuda"], width=10).grid(
            row=2, column=1, sticky="w", **pad
        )

        ttk.Label(frame, text="Synapse mode").grid(row=3, column=0, sticky="w", **pad)
        ttk.Combobox(frame, textvariable=self.synapse_mode, values=["record", "preview"], width=10).grid(
            row=3, column=1, sticky="w", **pad
        )
        ttk.Label(frame, text="EEG parameter").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.eeg_parameter, width=40).grid(row=4, column=1, sticky="w", **pad)
        ttk.Label(frame, text="EMG source 1").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.emg_source1, width=40).grid(row=5, column=1, sticky="w", **pad)
        ttk.Label(frame, text="EMG source 2").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.emg_source2, width=40).grid(row=6, column=1, sticky="w", **pad)
        self._row_selector(frame, "Normalization JSON:", self.norm_path, self._choose_norm_path, 7)

        self._load_runtime_defaults(Path(self.runtime_config.get()))

        actions = ttk.Frame(self.online_tab)
        actions.pack(fill="x", padx=8, pady=8)
        self.start_btn = ttk.Button(actions, text="Start Online", command=self._start_online)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="Stop", command=self._stop_online, state="disabled")
        self.stop_btn.pack(side="left", padx=10)

        status = ttk.LabelFrame(self.online_tab, text="Live Output (5s/epoch)")
        status.pack(fill="both", expand=True, padx=8, pady=8)
        self.pred_label = ttk.Label(status, text="Prediction: --", font=("Segoe UI", 14))
        self.pred_label.pack(anchor="w", padx=8, pady=6)
        self.conf_label = ttk.Label(status, text="Confidence: --")
        self.conf_label.pack(anchor="w", padx=8, pady=6)
        self.epoch_label = ttk.Label(status, text="Epoch: --")
        self.epoch_label.pack(anchor="w", padx=8, pady=6)

    def _row_selector(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        cmd,
        row: int,
    ) -> None:
        pad = {"padx": 8, "pady": 6}
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(parent, textvariable=var, width=70).grid(row=row, column=1, sticky="w", **pad)
        ttk.Button(parent, text="Browse", command=cmd).grid(row=row, column=2, sticky="w", **pad)

    def _choose_mat(self) -> None:
        path = filedialog.askopenfilename(title="Select MAT file", filetypes=[("MAT", "*.mat")])
        if path:
            self.mat_path.set(path)
            self._load_channels_from_mat(Path(path))

    def _choose_txt(self) -> None:
        path = filedialog.askopenfilename(title="Select TXT file", filetypes=[("TXT", "*.txt")])
        if path:
            self.txt_path.set(path)

    def _choose_runtime(self) -> None:
        path = filedialog.askopenfilename(title="Select runtime config", filetypes=[("YAML", "*.yaml *.yml")])
        if path:
            self.runtime_config.set(path)
            self._load_runtime_defaults(Path(path))

    def _choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(title="Select checkpoint", filetypes=[("PyTorch", "*.pt")])
        if path:
            self.checkpoint_path.set(path)

    def _choose_norm_path(self) -> None:
        path = filedialog.askopenfilename(title="Select normalization JSON", filetypes=[("JSON", "*.json")])
        if path:
            self.norm_path.set(path)

    def _load_runtime_defaults(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            payload = load_config(path)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        syn_cfg = payload.get("synapse", {})
        mode = syn_cfg.get("mode")
        if isinstance(mode, str) and mode.strip():
            self.synapse_mode.set(mode.strip().lower())

        channels = syn_cfg.get("channels", {})
        eeg_param = None
        emg_sources: list[str] = []
        if isinstance(channels, list):
            for entry in channels:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "")).strip().lower()
                if name == "eeg":
                    eeg_param = entry.get("parameter")
                    if not eeg_param:
                        sources = entry.get("sources") or []
                        if isinstance(sources, (list, tuple)) and sources:
                            eeg_param = sources[0]
                elif name == "emg":
                    sources = entry.get("sources") or []
                    if isinstance(sources, str):
                        emg_sources = [sources]
                    elif isinstance(sources, (list, tuple)):
                        emg_sources = [str(s) for s in sources]
        elif isinstance(channels, dict):
            eeg_entry = channels.get("eeg")
            if isinstance(eeg_entry, str):
                eeg_param = eeg_entry
            elif isinstance(eeg_entry, dict):
                eeg_param = eeg_entry.get("parameter")
                if not eeg_param:
                    sources = eeg_entry.get("sources") or []
                    if isinstance(sources, (list, tuple)) and sources:
                        eeg_param = sources[0]

            emg_entry = channels.get("emg")
            if isinstance(emg_entry, dict):
                sources = emg_entry.get("sources") or []
                if isinstance(sources, str):
                    emg_sources = [sources]
                elif isinstance(sources, (list, tuple)):
                    emg_sources = [str(s) for s in sources]

        if eeg_param:
            self.eeg_parameter.set(str(eeg_param))
        if emg_sources:
            self.emg_source1.set(emg_sources[0] if len(emg_sources) > 0 else "")
            self.emg_source2.set(emg_sources[1] if len(emg_sources) > 1 else "")

        runtime_cfg = payload.get("runtime", {})
        norm_path = runtime_cfg.get("normalization_path")
        if isinstance(norm_path, str) and norm_path.strip():
            self.norm_path.set(norm_path.strip())

    def _load_channels_from_mat(self, mat_path: Path) -> None:
        try:
            channels = detect_mat_channels(mat_path)
        except Exception as exc:
            messagebox.showerror("MAT", f"Failed to read channels: {exc}")
            self._detected_channels = []
            self.eeg_channel_combo.configure(values=[], state="disabled")
            self.emg_channel_combo.configure(values=[], state="disabled")
            return
        if not channels:
            messagebox.showerror("MAT", "No signal channels found in MAT file.")
            self._detected_channels = []
            self.eeg_channel_combo.configure(values=[], state="disabled")
            self.emg_channel_combo.configure(values=[], state="disabled")
            return

        self._detected_channels = channels
        self.eeg_channel_combo.configure(values=channels, state="readonly")
        self.emg_channel_combo.configure(values=channels, state="readonly")

        eeg_guess = self._guess_channel(channels, ["eeg_p", "eeg"])
        emg_guess = self._guess_channel(channels, ["emg_diff", "emg"])
        if eeg_guess and emg_guess and eeg_guess == emg_guess and len(channels) > 1:
            for name in channels:
                if name != eeg_guess:
                    emg_guess = name
                    break
        if eeg_guess:
            self.eeg_channel.set(eeg_guess)
        if emg_guess:
            self.emg_channel.set(emg_guess)

        try:
            sample_rate = detect_mat_sample_rate(mat_path)
        except Exception:
            sample_rate = None
        if sample_rate:
            self.sample_rate.set(str(sample_rate))

    def _start_prepare(self) -> None:
        if self._prepare_thread and self._prepare_thread.is_alive():
            messagebox.showinfo("Prepare", "Pipeline is already running.")
            return

        mat_path = self.mat_path.get().strip()
        txt_path = self.txt_path.get().strip()
        if not mat_path or not txt_path:
            messagebox.showerror("Prepare", "Please provide both MAT and TXT paths.")
            return
        if not Path(mat_path).is_file() or not Path(txt_path).is_file():
            messagebox.showerror("Prepare", "Please select MAT/TXT files (folders are not supported).")
            return
        if not self._detected_channels:
            self._load_channels_from_mat(Path(mat_path))
            if not self._detected_channels:
                return

        sample_rate = self._parse_float(self.sample_rate.get())
        finetune_epochs = self._parse_int(self.finetune_epochs.get())
        if sample_rate is None or finetune_epochs is None:
            messagebox.showerror("Prepare", "Sample rate and finetune epochs must be valid numbers.")
            return
        if not self.eeg_channel.get().strip() or not self.emg_channel.get().strip():
            messagebox.showerror("Prepare", "Please load MAT channels and select EEG/EMG.")
            return

        self.prepare_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.prepare_status.set("Running...")
        self._prepare_running = True
        self.prepare_progress.start(10)
        self.log_text.delete("1.0", tk.END)

        def task() -> None:
            writer = QueueWriter(self._queue)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    best_ckpt, norm_path = run_finetune_pipeline_from_paths(
                        mat_path=Path(mat_path),
                        txt_path=Path(txt_path),
                        base_config=Path("training_config.yaml"),
                        eeg_key=self.eeg_channel.get().strip(),
                        emg_key=self.emg_channel.get().strip(),
                        sample_rate=float(sample_rate),
                        finetune_epochs=int(finetune_epochs),
                        device=None if self.device_var.get() == "auto" else self.device_var.get(),
                        train_ratio=1.0,
                    )
                self._queue.put(("prepare_done", (best_ckpt, norm_path)))
            except Exception as exc:
                self._queue.put(("error", f"Prepare failed: {exc}"))
            finally:
                self._queue.put(("prepare_finished", None))

        self._prepare_thread = threading.Thread(target=task, daemon=True)
        self._prepare_thread.start()

    def _save_best(self) -> None:
        if not self._last_best_ckpt:
            return
        path = filedialog.asksaveasfilename(
            title="Save best model",
            defaultextension=".pt",
            filetypes=[("PyTorch", "*.pt")],
        )
        if not path:
            return
        target = Path(path)
        shutil.copy2(self._last_best_ckpt, target)
        messagebox.showinfo("Saved", f"Best model saved to:\n{target}")

    def _start_online(self) -> None:
        if self._online_thread and self._online_thread.is_alive():
            messagebox.showinfo("Online", "Online prediction already running.")
            return
        runtime_path = Path(self.runtime_config.get())
        if not runtime_path.exists():
            messagebox.showerror("Online", "Runtime config not found.")
            return
        ckpt_path = self.checkpoint_path.get().strip()
        if ckpt_path and not Path(ckpt_path).exists():
            messagebox.showerror("Online", "Checkpoint not found.")
            return

        eeg_param = self.eeg_parameter.get().strip()
        emg1 = self.emg_source1.get().strip()
        emg2 = self.emg_source2.get().strip()
        if not eeg_param:
            messagebox.showerror("Online", "Please provide EEG parameter.")
            return
        if not emg1 or not emg2:
            messagebox.showerror("Online", "Please provide two EMG sources.")
            return
        norm_path = self.norm_path.get().strip()
        if norm_path and not Path(norm_path).exists():
            messagebox.showerror("Online", "Normalization JSON not found.")
            return

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        def task() -> None:
            try:
                device = None if self.online_device.get() == "auto" else self.online_device.get()
                mode = self.synapse_mode.get().strip().lower()
                if mode not in {"record", "preview"}:
                    mode = None

                def hook(payload: dict) -> None:
                    self._queue.put(("pred", payload))

                predictor = create_predictor(
                    runtime_config=runtime_path,
                    checkpoint_path=Path(ckpt_path) if ckpt_path else None,
                    device=device,
                    on_epoch=hook,
                    synapse_mode=mode,
                    eeg_parameter=eeg_param,
                    emg_sources=[emg1, emg2],
                    normalization_path=Path(norm_path) if norm_path else None,
                )
                self._predictor = predictor
                predictor.run()
            except Exception as exc:
                self._queue.put(("error", f"Online failed: {exc}"))
            finally:
                self._queue.put(("online_finished", None))

        self._online_thread = threading.Thread(target=task, daemon=True)
        self._online_thread.start()

    def _stop_online(self) -> None:
        if self._predictor:
            self._predictor.signal_stop()

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self.log_text.insert(tk.END, payload)
                    self.log_text.see(tk.END)
                elif kind == "error":
                    messagebox.showerror("Error", payload)
                    self.prepare_status.set("Failed")
                elif kind == "prepare_done":
                    best_ckpt, norm_path = payload
                    self._last_best_ckpt = Path(best_ckpt)
                    self.save_btn.configure(state="normal")
                    self.checkpoint_path.set(str(best_ckpt))
                    self.log_text.insert(tk.END, f"\n[Done] Best checkpoint: {best_ckpt}\n")
                    self.log_text.insert(tk.END, f"[Done] Normalization stats: {norm_path}\n")
                    self.log_text.see(tk.END)
                    self.prepare_status.set("Done")
                elif kind == "prepare_finished":
                    self.prepare_btn.configure(state="normal")
                    if self._prepare_running:
                        self.prepare_progress.stop()
                        self._prepare_running = False
                    if self.prepare_status.get() == "Running...":
                        self.prepare_status.set("Idle")
                elif kind == "pred":
                    self._update_prediction(payload)
                elif kind == "online_finished":
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _update_prediction(self, payload: dict) -> None:
        idx = payload.get("epoch_index")
        pred = payload.get("smoothed_prediction")
        conf = payload.get("confidence")
        label = STATE_NAMES.get(int(pred), f"class_{pred}") if pred is not None else "--"
        self.pred_label.configure(text=f"Prediction: {label}")
        if conf is not None:
            self.conf_label.configure(text=f"Confidence: {float(conf):.3f}")
        if idx is not None:
            self.epoch_label.configure(text=f"Epoch: {int(idx)}")

    @staticmethod
    def _guess_channel(candidates: list[str], preferred: list[str]) -> str:
        lowered = [(name, name.lower()) for name in candidates]
        for pref in preferred:
            for name, name_lower in lowered:
                if name_lower == pref:
                    return name
        for pref in preferred:
            for name, name_lower in lowered:
                if pref in name_lower:
                    return name
        return candidates[0] if candidates else ""

    @staticmethod
    def _parse_int(text: str) -> Optional[int]:
        text = text.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None

    @staticmethod
    def _parse_float(text: str) -> Optional[float]:
        text = text.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None


def main() -> None:
    app = OnlineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
