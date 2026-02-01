from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import os
import shutil

import h5py
import numpy as np
import scipy.io as sio

from llm_eeg.configs import PreprocessConfig
from llm_eeg.data import discover_subject_infos, load_subject_epochs


def preprocess_to_h5(
    recordings_dir: Path,
    output_dir: Path,
    subjects: Sequence[str] | None,
    eeg_key: str,
    emg_key: str,
    cache_dir: Path | None,
    sample_rate: float,
) -> List[str]:
    """Load MAT/TXT pairs, preprocess, and export per-subject H5 files."""
    prep_cfg = PreprocessConfig(sample_rate=sample_rate)
    infos = discover_subject_infos(recordings_dir, data_format="mat")
    if not infos:
        raise FileNotFoundError(f"No MAT/TXT pairs found under {recordings_dir}")

    target = set(subjects) if subjects else None
    output_dir.mkdir(parents=True, exist_ok=True)
    processed: List[str] = []
    for info in infos:
        if target and info.subject_id not in target:
            continue
        record = load_subject_epochs(
            info.subject_id,
            recordings_dir,
            prep_cfg,
            cache_dir=cache_dir,
            data_format="mat",
            eeg_key=eeg_key,
            emg_key=emg_key,
        )
        if record.epochs.shape[0] == 0:
            print(f"[Skip] {info.subject_id}: no usable epochs after preprocessing.")
            continue
        out_path = output_dir / f"{info.subject_id}.h5"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as f:
            f.create_dataset(f"{eeg_key}_epochs", data=record.epochs[:, 0, :], compression="gzip")
            f.create_dataset(f"{emg_key}_epochs", data=record.epochs[:, 1, :], compression="gzip")
            f.create_dataset("labels", data=record.labels, compression="gzip")
        processed.append(info.subject_id)
        print(f"[OK] {info.subject_id} -> {out_path}")
    if not processed:
        raise RuntimeError("No subjects were processed; check inputs and label coverage.")
    return processed


def list_files(path: Path, suffix: str) -> List[Path]:
    if path.is_dir():
        return sorted(path.glob(f"*{suffix}"))
    if path.is_file():
        if path.suffix.lower() != suffix:
            raise ValueError(f"Expected *{suffix} file, got {path.name}")
        return [path]
    raise FileNotFoundError(f"Path not found: {path}")


def _resolve_mat_key(mat_dict: Dict) -> str:
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError("MAT file does not contain any usable keys.")
    if len(keys) == 1:
        return keys[0]
    for candidate in keys:
        if candidate.lower().startswith("mouse"):
            return candidate
    return keys[0]


def _collect_channel_keys(entry) -> List[str]:
    if isinstance(entry, dict):
        items = entry.items()
    elif isinstance(entry, np.ndarray) and entry.dtype.names:
        items = ((name, entry[name]) for name in entry.dtype.names)
    else:
        return []

    candidates: List[str] = []
    for key, value in items:
        if key.startswith("__"):
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if not np.issubdtype(arr.dtype, np.number):
            continue
        if arr.size < 10:
            continue
        candidates.append(key)
    return sorted({c for c in candidates})


def _get_entry_value(entry, key: str):
    if isinstance(entry, dict):
        return entry.get(key)
    if isinstance(entry, np.ndarray) and entry.dtype.names:
        if key in entry.dtype.names:
            return entry[key]
    return None


def _extract_sample_rate(entry) -> Optional[float]:
    for key in ("sf", "fs", "sample_rate", "sampling_rate", "srate"):
        value = _get_entry_value(entry, key)
        if value is None:
            continue
        try:
            values = np.asarray(value, dtype=np.float32).ravel()
        except Exception:
            continue
        for val in values:
            if val > 0:
                return float(val)
    return None


def detect_mat_channels(mat_path: Path) -> List[str]:
    mat = sio.loadmat(mat_path, simplify_cells=True)
    if not mat:
        return []
    try:
        key = mat_path.stem if mat_path.stem in mat else _resolve_mat_key(mat)
        entry = mat[key]
    except Exception:
        entry = None

    channels = _collect_channel_keys(entry) if entry is not None else []
    if not channels:
        channels = _collect_channel_keys(mat)
    return channels


def detect_mat_sample_rate(mat_path: Path, default: Optional[float] = None) -> Optional[float]:
    mat = sio.loadmat(mat_path, simplify_cells=True)
    if not mat:
        return None
    try:
        key = mat_path.stem if mat_path.stem in mat else _resolve_mat_key(mat)
        entry = mat[key]
    except Exception:
        entry = None

    if entry is not None:
        value = _extract_sample_rate(entry)
        if value:
            return value
    value = _extract_sample_rate(mat)
    if value:
        return value
    return default


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def stage_recordings(mat_path: Path, txt_path: Path, dest_dir: Path) -> List[str]:
    if not mat_path.is_file():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    if not txt_path.is_file():
        raise FileNotFoundError(f"TXT file not found: {txt_path}")
    mats = list_files(mat_path, ".mat")
    txts = list_files(txt_path, ".txt")
    if mats[0].stem.lower() != txts[0].stem.lower():
        raise RuntimeError("MAT/TXT filenames must match for single-file mode.")
    pairs = [(mats[0], txts[0], mats[0].stem)]

    dest_dir.mkdir(parents=True, exist_ok=True)
    staged: List[str] = []
    for mat_file, txt_file, stem in pairs:
        link_or_copy(mat_file, dest_dir / f"{stem}.mat")
        link_or_copy(txt_file, dest_dir / f"{stem}.txt")
        staged.append(stem)
    return staged
