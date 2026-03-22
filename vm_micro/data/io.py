"""vm_micro.data.io
~~~~~~~~~~~~~~~~~~
Unified signal reader for FLAC/WAV (airborne) and HDF5 (structure-borne).

All callers should use :func:`read_signal_auto`, which returns a normalised
dict regardless of source format.  Lower-level readers are also exported for
use in the splitter.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# ─────────────────────────────────────────────────────────────────────────────
# Format detection
# ─────────────────────────────────────────────────────────────────────────────

_AUDIO_SUFFIXES = {".flac", ".wav", ".ogg", ".aiff", ".aif"}
_HDF5_SUFFIXES  = {".h5", ".hdf5"}


def get_input_kind(path: str | Path) -> str:
    """Return ``'audio'`` or ``'hdf5'`` based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in _AUDIO_SUFFIXES:
        return "audio"
    if suffix in _HDF5_SUFFIXES:
        return "hdf5"
    raise ValueError(f"Unsupported file extension: {suffix!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Audio (FLAC / WAV)
# ─────────────────────────────────────────────────────────────────────────────

def _resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    factor = gcd(sr_in, sr_out)
    return resample_poly(
        y.astype(np.float32, copy=False),
        up=sr_out // factor,
        down=sr_in // factor,
    ).astype(np.float32)


def read_audio_mono(
    path: str | Path,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int]:
    """Read FLAC/WAV, downmix to mono float32, optionally resample."""
    path = Path(path)
    y, sr = sf.read(path, always_2d=True)
    y = y.astype(np.float32).mean(axis=1)
    if target_sr is not None and target_sr != sr:
        y = _resample(y, int(sr), int(target_sr))
        sr = target_sr
    return y.astype(np.float32, copy=False), int(sr)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 measurements (structure-borne)
# ─────────────────────────────────────────────────────────────────────────────

def _infer_sr_from_time_vector(
    time_vector: np.ndarray,
) -> tuple[int, float, float]:
    """Infer sample rate and jitter from a time vector."""
    tv = np.asarray(time_vector, dtype=np.float64)
    if tv.ndim != 1 or len(tv) < 2:
        raise ValueError("time_vector must be 1-D with ≥ 2 elements")

    dt = np.diff(tv)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError("Could not infer sample rate: no valid positive diffs")

    dt_median = float(np.median(dt))
    sr = int(round(1.0 / dt_median))
    jitter = float(np.max(np.abs(dt - dt_median)) / dt_median) if len(dt) else 0.0
    return sr, dt_median, jitter


def read_measurement_h5(
    path: str | Path,
    *,
    data_key: str = "measurement/data",
    time_key: str = "measurement/time_vector",
    center_signal: bool = True,
    target_sr: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray, dict[str, Any]]:
    """Read one HDF5 measurement file.

    Returns
    -------
    y : float32 1-D array
    sr : int
    time_vector : float64 1-D array
    meta : dict  (dt_median_s, relative_time_jitter, data_key, time_key)
    """
    path = Path(path)
    with h5py.File(path, "r") as fh:
        data = fh[data_key][:]
        time_vector = fh[time_key][:]

    y = np.asarray(data, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(
            f"Expected 1-D dataset at {data_key!r}, got shape {y.shape}"
        )

    sr, dt_median, jitter = _infer_sr_from_time_vector(time_vector)

    if center_signal:
        y = y - float(np.mean(y))

    tv = np.asarray(time_vector, dtype=np.float64)
    if target_sr is not None and target_sr != sr:
        y = _resample(y, sr, int(target_sr))
        tv = np.arange(len(y), dtype=np.float64) / float(target_sr)
        sr = int(target_sr)
        dt_median = 1.0 / float(sr)
        jitter = 0.0

    meta: dict[str, Any] = {
        "dt_median_s": float(dt_median),
        "relative_time_jitter": float(jitter),
        "data_key": data_key,
        "time_key": time_key,
    }
    return y.astype(np.float32, copy=False), int(sr), tv, meta


# ─────────────────────────────────────────────────────────────────────────────
# Unified reader
# ─────────────────────────────────────────────────────────────────────────────

def read_signal_auto(
    path: str | Path,
    *,
    target_sr: int | None = None,
    h5_data_key: str = "measurement/data",
    h5_time_key: str = "measurement/time_vector",
    center_h5_signal: bool = True,
) -> dict[str, Any]:
    """Read either format and return a unified signal dict.

    Keys
    ----
    path, stem, input_kind, y (float32), sr (int),
    time_vector (float64), duration_s (float),
    dt_median_s (float), relative_time_jitter (float).
    """
    path = Path(path)
    kind = get_input_kind(path)

    if kind == "audio":
        y, sr = read_audio_mono(path, target_sr=target_sr)
        time_vector = np.arange(len(y), dtype=np.float64) / float(sr)
        meta: dict[str, Any] = {"dt_median_s": 1.0 / float(sr), "relative_time_jitter": 0.0}
    else:
        y, sr, time_vector, meta = read_measurement_h5(
            path,
            data_key=h5_data_key,
            time_key=h5_time_key,
            center_signal=center_h5_signal,
            target_sr=target_sr,
        )

    if len(time_vector) > 1:
        duration_s = float(time_vector[-1] - time_vector[0]) + float(
            np.median(np.diff(time_vector))
        )
    else:
        duration_s = float(len(y) / sr)

    return {
        "path": path,
        "stem": path.stem,
        "input_kind": kind,
        "y": y.astype(np.float32, copy=False),
        "sr": int(sr),
        "time_vector": np.asarray(time_vector, dtype=np.float64),
        "duration_s": float(duration_s),
        **meta,
    }
