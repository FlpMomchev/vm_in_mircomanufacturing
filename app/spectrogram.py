from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.figure import Figure
from scipy import signal

SPECTROGRAM_CONFIG = {
    "airborne": {
        "nperseg": 4096,
        "noverlap": 3072,
        "max_freq_hz": 12000,
        "n_mels": 128,
        "fmin_hz": 150,
        "top_db": 70,
    },
    "structure": {
        "nperseg": 8192,
        "noverlap": 6144,
        "max_freq_hz": 12000,
        "n_mels": 128,
        "fmin_hz": 150,
        "top_db": 70,
    },
}


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    if np.ndim(y) > 1:
        y = np.mean(y, axis=1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    return y, int(sr)


def _load_h5(path: Path) -> tuple[np.ndarray, int]:
    with h5py.File(str(path), "r") as fh:
        y = np.asarray(fh["measurement/data"][:], dtype=np.float32).reshape(-1)
        t = np.asarray(fh["measurement/time_vector"][:], dtype=np.float64).reshape(-1)

    if len(t) < 2:
        raise ValueError(f"Invalid time_vector in {path}")

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        raise ValueError(f"Could not infer sample rate from {path}")

    sr = int(round(1.0 / float(np.median(dt))))
    return y, sr


def load_signal(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()

    if suffix in {".flac", ".wav"}:
        return _load_audio(path)
    if suffix in {".h5", ".hdf5"}:
        return _load_h5(path)

    raise ValueError(f"Unsupported spectrogram file type: {suffix}")


def _hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    freqs_hz: np.ndarray,
    sr: int,
    n_mels: int,
    fmin_hz: float,
    fmax_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    fmax_hz = min(float(fmax_hz), sr / 2.0)
    fmin_hz = max(0.0, float(fmin_hz))

    mel_min = _hz_to_mel(fmin_hz)
    mel_max = _hz_to_mel(fmax_hz)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    fb = np.zeros((n_mels, len(freqs_hz)), dtype=np.float32)

    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        if center <= left or right <= center:
            continue

        left_slope = (freqs_hz - left) / (center - left)
        right_slope = (right - freqs_hz) / (right - center)
        fb[i] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    # Slaney-style normalization for more balanced mel-band energy
    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-12)
    fb *= enorm[:, None]

    mel_centers_hz = hz_points[1:-1]
    return fb, mel_centers_hz


def _compute_logmel(
    y: np.ndarray,
    sr: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(int(cfg["nperseg"]), max(256, len(y)))
    noverlap = min(int(cfg["noverlap"]), max(128, len(y) // 2))

    freqs_hz, times_s, sxx = signal.spectrogram(
        y,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
        detrend="constant",
    )

    power_spec = sxx**2

    fmin_hz = float(cfg["fmin_hz"])
    fmax_hz = min(float(cfg["max_freq_hz"]), sr / 2.0)

    band_mask = (freqs_hz >= fmin_hz) & (freqs_hz <= fmax_hz)
    freqs_band_hz = freqs_hz[band_mask]
    power_band = power_spec[band_mask, :]

    if len(freqs_band_hz) < 4:
        raise ValueError("Too few frequency bins available for log-mel rendering.")

    n_freq_bins = len(freqs_band_hz)
    n_mels_eff = max(4, min(int(cfg["n_mels"]), n_freq_bins // 2))

    mel_fb, mel_centers_hz = _build_mel_filterbank(
        freqs_hz=freqs_band_hz,
        sr=sr,
        n_mels=n_mels_eff,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )

    mel_power = mel_fb @ power_band
    mel_power = np.maximum(mel_power, 1e-12)

    logmel_db = 10.0 * np.log10(mel_power)
    peak_db = float(np.max(logmel_db))
    floor_db = peak_db - float(cfg["top_db"])
    logmel_db = np.clip(logmel_db, floor_db, peak_db)

    return mel_centers_hz, times_s, logmel_db


def make_spectrogram_figure(
    file_path: str | Path,
    modality: str,
) -> Figure:
    modality = str(modality).strip().lower()
    if modality not in SPECTROGRAM_CONFIG:
        raise ValueError(f"Unsupported modality: {modality}")

    cfg = SPECTROGRAM_CONFIG[modality]
    y, sr = load_signal(file_path)

    mel_freqs_hz, times_s, logmel_db = _compute_logmel(y=y, sr=sr, cfg=cfg)

    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=160)

    mesh = ax.pcolormesh(
        times_s, mel_freqs_hz, logmel_db, shading="gouraud", cmap="magma", rasterized=True
    )

    ax.set_xlabel("Time [s]", fontsize=11)
    ax.set_ylabel("Frequency [Hz]", fontsize=11)

    ax.set_ylim(float(cfg["fmin_hz"]), min(float(cfg["max_freq_hz"]), sr / 2.0))
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Log-mel power [dB]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    return fig
