"""vm_micro.features.core
~~~~~~~~~~~~~~~~~~~~~~~~
Signal feature extraction functions shared by both airborne and structure-borne
pipelines.

Every function accepts a 1-D float32 ``y`` array and an integer ``sr``, and
returns a flat ``dict[str, float]``.  The dicts are merged by the modality-
specific extractors into a single feature row.

Airborne-only families (machining proxies) are gated by a flag in the config
and call :func:`compute_machining_features`.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import pywt
import scipy.signal as sig
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view

# ─────────────────────────────────────────────────────────────────────────────
# Time-domain features
# ─────────────────────────────────────────────────────────────────────────────

def compute_time_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Basic time-domain statistics."""
    y = np.asarray(y, dtype=np.float64)
    rms    = float(np.sqrt(np.mean(y ** 2)))
    peak   = float(np.max(np.abs(y)))
    crest  = float(peak / rms) if rms > 0 else 0.0
    kurt   = float(stats.kurtosis(y, fisher=True))
    skew   = float(stats.skew(y))
    zcr    = float(np.sum(np.diff(np.sign(y)) != 0) / max(1, len(y) - 1))

    return {
        "time_rms":           rms,
        "time_peak":          peak,
        "time_mean_abs":      float(np.mean(np.abs(y))),
        "time_std":           float(np.std(y)),
        "time_crest_factor":  crest,
        "time_kurtosis":      kurt,
        "time_skewness":      skew,
        "time_zcr":           zcr,
        "time_energy":        float(np.sum(y ** 2)),
        "time_p2p":           float(np.ptp(y)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain features
# ─────────────────────────────────────────────────────────────────────────────

def compute_frequency_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """FFT-based spectral features."""
    y = np.asarray(y, dtype=np.float64)
    n  = len(y)
    if n < 4:
        return {}

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag   = np.abs(np.fft.rfft(y * np.hanning(n)))
    power = mag ** 2
    total = float(np.sum(power)) + 1e-18

    centroid  = float(np.sum(freqs * power) / total)
    spread    = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total))
    rolloff_q = 0.85
    cumsum    = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumsum, rolloff_q * float(cumsum[-1])))
    rolloff   = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Spectral flatness (geometric / arithmetic mean of power)
    log_mag   = np.log(mag + 1e-18)
    flatness  = float(np.exp(np.mean(log_mag)) / (np.mean(mag) + 1e-18))

    # Peak frequency
    peak_idx  = int(np.argmax(mag))
    peak_freq = float(freqs[peak_idx])

    return {
        "freq_centroid_hz":  centroid,
        "freq_spread_hz":    spread,
        "freq_rolloff_hz":   rolloff,
        "freq_flatness":     flatness,
        "freq_peak_hz":      peak_freq,
        "freq_peak_mag":     float(mag[peak_idx]),
        "freq_entropy":      _spectral_entropy(power),
        "freq_total_power":  total,
    }


def _spectral_entropy(power: np.ndarray) -> float:
    total = float(np.sum(power)) + 1e-18
    p = power / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ─────────────────────────────────────────────────────────────────────────────
# Band-power features
# ─────────────────────────────────────────────────────────────────────────────

def compute_band_power_features(
    y: np.ndarray,
    sr: int,
    bands: list[tuple[float, float]] | None = None,
) -> dict[str, float]:
    """RMS power in configurable frequency bands."""
    if bands is None:
        nyq = sr / 2.0
        bands = [
            (100.0,   1000.0),
            (1000.0,  5000.0),
            (5000.0,  20000.0),
            (20000.0, min(nyq - 1, 96000.0)),
        ]

    y  = np.asarray(y, dtype=np.float64)
    n  = len(y)
    if n < 4:
        return {}

    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag   = np.abs(np.fft.rfft(y * np.hanning(n)))

    out: dict[str, float] = {}
    for lo, hi in bands:
        mask  = (freqs >= lo) & (freqs < hi)
        rms   = float(np.sqrt(np.mean(mag[mask] ** 2))) if mask.any() else 0.0
        label = f"band_{int(lo)}_{int(hi)}_rms"
        out[label] = rms

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Machining-environment proxies  (airborne only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_machining_features(
    y: np.ndarray,
    sr: int,
    env_sr: int = 24000,
    hf_sr: int  = 64000,
    hf_proxy: str = "roughness",
) -> dict[str, float]:
    """Roughness, AE-proxy, and envelope-based machining condition features.

    These are *airborne-specific* and should not be used for structure-borne.
    """
    y = np.asarray(y, dtype=np.float64)

    # Amplitude envelope (low-rate) – captures tool-wear trend
    env = _amplitude_envelope(y, sr, target_sr=env_sr)
    env_feats = {
        "mach_env_mean":  float(np.mean(env)),
        "mach_env_std":   float(np.std(env)),
        "mach_env_range": float(np.ptp(env)),
        "mach_env_rms":   float(np.sqrt(np.mean(env ** 2))),
    }

    # High-frequency proxy (roughness or acoustic emission)
    hf = _hf_band_signal(y, sr, target_sr=hf_sr)
    if hf_proxy == "roughness":
        hf_val = float(np.mean(np.abs(hf)))
        hf_rms  = float(np.sqrt(np.mean(hf ** 2)))
        hf_feats = {"mach_roughness_mean": hf_val, "mach_roughness_rms": hf_rms}
    else:  # "ae"
        hf_feats = {
            "mach_ae_mean": float(np.mean(np.abs(hf))),
            "mach_ae_rms":  float(np.sqrt(np.mean(hf ** 2))),
        }

    return {**env_feats, **hf_feats}


def _amplitude_envelope(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Full-wave rectified + low-pass envelope, downsampled."""
    rectified = np.abs(y)
    cutoff = target_sr / 2.0
    sos = sig.butter(4, cutoff / (sr / 2.0), btype="low", output="sos")
    env = sig.sosfiltfilt(sos, rectified)
    factor = max(1, int(round(sr / target_sr)))
    return env[::factor].astype(np.float64)


def _hf_band_signal(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """High-pass filtered signal, downsampled to *target_sr*."""
    cutoff = target_sr / 4.0
    sos = sig.butter(4, cutoff / (sr / 2.0), btype="high", output="sos")
    hf = sig.sosfiltfilt(sos, y)
    factor = max(1, int(round(sr / target_sr)))
    return hf[::factor].astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical features
# ─────────────────────────────────────────────────────────────────────────────

def compute_statistical_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Higher-order statistics and percentile features."""
    y = np.asarray(y, dtype=np.float64)
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct = np.percentile(y, percentiles)
    out: dict[str, float] = {f"stat_p{p}": float(v) for p, v in zip(percentiles, pct)}
    out["stat_iqr"]       = float(pct[4] - pct[2])  # 75th - 25th
    out["stat_median_abs_dev"] = float(np.median(np.abs(y - np.median(y))))
    out["stat_sample_entropy"] = _sample_entropy(y[:2048])  # limit length
    return out


def _sample_entropy(y: np.ndarray, m: int = 2, r_frac: float = 0.2) -> float:
    """Approximate sample entropy (Richman & Moorman, 2000)."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < m + 2:
        return 0.0
    r = r_frac * float(np.std(y)) + 1e-18
    try:
        templates_m  = sliding_window_view(y, m)
        templates_m1 = sliding_window_view(y, m + 1)
        diff_m  = np.max(np.abs(templates_m[:, None] - templates_m[None, :]), axis=-1)
        diff_m1 = np.max(np.abs(templates_m1[:, None] - templates_m1[None, :]), axis=-1)
        np.fill_diagonal(diff_m,  r + 1)
        np.fill_diagonal(diff_m1, r + 1)
        B = float(np.sum(diff_m  <= r))
        A = float(np.sum(diff_m1 <= r))
        if B == 0:
            return 0.0
        return float(-np.log(A / B))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Short-time statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_short_time_features(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 10.0,
    hop_ms: float = 5.0,
) -> dict[str, float]:
    """Mean / std of per-frame RMS, crest factor, and ZCR."""
    y = np.asarray(y, dtype=np.float64)
    frame_n = max(1, int(round(sr * frame_ms / 1000.0)))
    hop_n   = max(1, int(round(sr * hop_ms  / 1000.0)))

    if len(y) < frame_n:
        return {}

    try:
        frames = sliding_window_view(y, frame_n)[::hop_n]
    except Exception:
        return {}

    rms    = np.sqrt(np.mean(frames ** 2, axis=1))
    peaks  = np.max(np.abs(frames), axis=1)
    crest  = np.where(rms > 0, peaks / rms, 0.0)
    zcr    = np.sum(np.diff(np.sign(frames), axis=1) != 0, axis=1) / float(frame_n - 1)

    return {
        "st_rms_mean":   float(np.mean(rms)),
        "st_rms_std":    float(np.std(rms)),
        "st_crest_mean": float(np.mean(crest)),
        "st_crest_std":  float(np.std(crest)),
        "st_zcr_mean":   float(np.mean(zcr)),
        "st_zcr_std":    float(np.std(zcr)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DWT features
# ─────────────────────────────────────────────────────────────────────────────

def compute_dwt_features(
    y: np.ndarray,
    sr: int,
    wavelet: str = "db8",
    max_level: int = 8,
) -> dict[str, float]:
    """DWT per-level energy, RMS, median, std."""
    y = np.asarray(y, dtype=np.float64)
    wavelet_obj = pywt.Wavelet(wavelet)
    actual_level = min(max_level, pywt.dwt_max_level(len(y), wavelet_obj.dec_len))
    if actual_level < 1:
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = pywt.wavedec(y, wavelet, level=actual_level)

    out: dict[str, float] = {}
    for lvl, c in enumerate(coeffs, start=1):
        prefix = f"dwt_l{lvl}"
        out[f"{prefix}_rms"]    = float(np.sqrt(np.mean(c ** 2)))
        out[f"{prefix}_energy"] = float(np.sum(c ** 2))
        out[f"{prefix}_median"] = float(np.median(c))
        out[f"{prefix}_std"]    = float(np.std(c))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CWT features
# ─────────────────────────────────────────────────────────────────────────────

def compute_cwt_features(
    y: np.ndarray,
    sr: int,
    wavelet: str = "morl",
    num_scales: int = 64,
    fmin: float = 200.0,
    fmax: float | None = None,
) -> dict[str, float]:
    """CWT-based features: per-scale and global energy/RMS statistics."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 16:
        return {}

    if fmax is None:
        fmax = float(sr) / 2.0

    scales = _auto_scales(sr, wavelet, fmin, min(fmax, sr / 2.0 - 1), num_scales)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs, _ = pywt.cwt(y, scales, wavelet, sampling_period=1.0 / sr)

    abs_c = np.abs(coeffs)  # shape: (num_scales, time)

    out: dict[str, float] = {}
    scale_energy = np.sum(abs_c ** 2, axis=1)
    total_energy = float(np.sum(scale_energy)) + 1e-18
    for i, e in enumerate(scale_energy):
        out[f"cwt_scale{i:03d}_energy_frac"] = float(e / total_energy)
        out[f"cwt_scale{i:03d}_rms"]         = float(np.sqrt(np.mean(abs_c[i] ** 2)))

    out["cwt_global_max"]   = float(np.max(abs_c))
    out["cwt_global_mean"]  = float(np.mean(abs_c))
    out["cwt_global_std"]   = float(np.std(abs_c))
    out["cwt_peak_scale"]   = int(np.argmax(scale_energy))
    out["cwt_energy_spread"] = float(np.std(scale_energy) / (np.mean(scale_energy) + 1e-18))
    return out


@lru_cache(maxsize=32)
def _auto_scales(
    sr: int,
    wavelet_name: str,
    fmin: float,
    fmax: float,
    num_scales: int,
) -> np.ndarray:
    """Log-spaced CWT scales for a frequency range."""
    wavelet = pywt.ContinuousWavelet(wavelet_name)
    fc = wavelet.center_frequency

    # Newer PyWavelets may return None; fall back to FFT-based estimate
    if fc is None or fc == 0:
        psi, x = wavelet.wavefun(level=10)
        psi = np.asarray(psi, dtype=np.float64)
        domain = float(x[-1] - x[0])
        idx = int(np.argmax(np.abs(np.fft.rfft(psi)[1:])) + 1)
        fc = float(idx / domain) if domain > 0 else 1.0

    dt = 1.0 / sr
    scale_min = fc / (fmax * dt)
    scale_max = fc / (fmin * dt)
    return np.logspace(np.log10(scale_min), np.log10(scale_max), num_scales)
