"""vm_micro.features.core
~~~~~~~~~~~~~~~~~~~~~~~~
Every function returns a flat dict[str, float].  The airborne and
structure-borne extractors call these directly.

Families
--------
time            : amplitude stats, crest/shape/impulse factors, TKEO,
                  Hilbert envelope, percentiles, ZCR, waveform length
frequency       : spectral centroid/spread/flatness/entropy/slope/crest,
                  rolloff (85 % + 95 %), peak frequencies, spectral decrease
band_power      : per-band RMS power and dB via bandpass filter,
                  band ratios (all bands from original extraction.py)
machining       : tpf, modulation depth, force proxy, temporal stability,
                  autocorrelation, roughness/AE proxy
timefrequency   : STFT-based spectral flux, variation, temporal centroid,
                  per-band energy CV, centroid/bandwidth/rolloff modulation
statistical     : mean, std, MAD, range, threshold counts
short_time      : per-frame RMS/crest/kurtosis/skewness/TKEO/ZCR stats
dwt             : per-level energy ratio, RMS, entropy; grouped HF/MF/LF
cwt             : complex Morlet scalogram; per physical-band stats + ridge
"""

from __future__ import annotations

import math
import re
import warnings
from functools import lru_cache

import numpy as np
import pywt
import scipy.signal as sig
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import fft, fftfreq
from scipy.signal import decimate as _decimate

#
# Machining proxy rate defaults
# Plain literals used only as function-signature fallbacks.
# Real values always come from configs/airborne.yaml and are passed explicitly
# by airborne.py.  Do NOT import these; use the cfg values instead.
#
_MACHINING_ENV_SR_DEFAULT = 24_000
_MACHINING_HF_SR_DEFAULT = 64_000
_MACHINING_HF_PROXY_DEFAULT = "roughness"  # "roughness" | "ae"


#
# Shared low-level helpers
#


def _resample_if_needed(x: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if sr <= target_sr:
        return np.asarray(x, dtype=np.float64), sr
    g = math.gcd(int(sr), int(target_sr))
    up, down = target_sr // g, sr // g
    return sig.resample_poly(np.asarray(x, dtype=np.float64), up, down).astype(
        np.float64
    ), target_sr


@lru_cache(maxsize=256)
def _bandpass_sos(sr: int, f_low: float, f_high: float, order: int = 4):
    f_high = min(float(f_high), sr / 2 - 100.0)
    if f_low <= 0 or f_low >= f_high:
        raise ValueError(f"Invalid band: {f_low}{f_high} @ sr={sr}")
    return sig.butter(order, [float(f_low), float(f_high)], btype="bandpass", fs=sr, output="sos")


def _bandpass(x: np.ndarray, sr: int, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    sos = _bandpass_sos(int(sr), float(f_low), float(f_high), int(order))
    return sig.sosfilt(sos, np.asarray(x, dtype=np.float64))


#
# 1. Time-domain features
#


def compute_time_features(y: np.ndarray, sr: int) -> dict[str, float]:
    x = np.asarray(y, dtype=np.float64)
    feat: dict[str, float] = {}

    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))

    feat["rms"] = rms
    feat["peak_amplitude"] = peak
    feat["mean_amplitude"] = mean_abs
    feat["waveform_len_per_s"] = float(np.sum(np.abs(np.diff(x))) * sr / (len(x) + 1e-12))
    feat["crest_factor"] = peak / (rms + 1e-10)
    feat["shape_factor"] = rms / (mean_abs + 1e-10)
    feat["impulse_factor"] = peak / (mean_abs + 1e-10)

    sqrtamp = np.mean(np.sqrt(np.abs(x))) ** 2
    feat["clearance_factor"] = peak / (sqrtamp + 1e-10)

    feat["kurtosis"] = float(stats.kurtosis(x, fisher=True))
    feat["skewness"] = float(stats.skew(x))

    # TKEO
    if len(x) >= 3:
        tkeo = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
        feat["tkeo_mean"] = float(np.mean(tkeo))
        feat["tkeo_std"] = float(np.std(tkeo))
    else:
        feat["tkeo_mean"] = 0.0
        feat["tkeo_std"] = 0.0

    energy = float(np.sum(x**2))
    feat["energy"] = energy
    feat["log_energy"] = float(np.log(energy + 1e-10))

    zc = np.sum(np.diff(np.sign(x)) != 0)
    feat["zcr"] = zc / (2 * len(x))
    feat["zcr_hz"] = feat["zcr"] * sr

    # Hilbert envelope
    envelope = np.abs(sig.hilbert(x))
    env_max = float(np.max(envelope))
    feat["envelope_mean"] = float(np.mean(envelope))
    feat["envelope_std"] = float(np.std(envelope))
    feat["envelope_max"] = env_max
    feat["envelope_dynamic_range_db"] = 20 * np.log10(
        env_max / (float(np.percentile(envelope, 10)) + 1e-10)
    )

    feat["waveform_length"] = float(np.sum(np.abs(np.diff(x))))

    abs_x = np.abs(x)
    for p in [10, 25, 50, 75, 90, 95, 99]:
        feat[f"percentile_{p}"] = float(np.percentile(abs_x, p))

    feat["iqr"] = feat["percentile_75"] - feat["percentile_25"]
    feat["dynamic_range_db"] = 20 * np.log10(
        feat["percentile_99"] / (feat["percentile_10"] + 1e-10)
    )
    return feat


#
# 2. Frequency-domain features
#


def compute_frequency_features(y: np.ndarray, sr: int, nfft: int = 8192) -> dict[str, float]:
    x = np.asarray(y, dtype=np.float64)
    feat: dict[str, float] = {}

    X = fft(x, n=nfft)
    freqs = fftfreq(nfft, 1.0 / sr)
    pos = freqs >= 0
    freqs = freqs[pos]
    mag = np.abs(X[pos])
    psd = mag**2 / len(x)
    psd_n = psd / (np.sum(psd) + 1e-10)

    centroid = float(np.sum(freqs * psd_n))
    feat["spectral_centroid"] = centroid
    feat["spectral_spread"] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_n)))
    feat["spectral_bandwidth"] = feat["spectral_spread"]

    cumsum = np.cumsum(psd_n)
    idx85 = np.where(cumsum >= 0.85)[0]
    idx95 = np.where(cumsum >= 0.95)[0]
    feat["spectral_rolloff_85"] = float(freqs[idx85[0]] if len(idx85) else freqs[-1])
    feat["spectral_rolloff_95"] = float(freqs[idx95[0]] if len(idx95) else freqs[-1])

    feat["spectral_skewness"] = float(stats.skew(psd_n))
    feat["spectral_kurtosis"] = float(stats.kurtosis(psd_n, fisher=True))

    gm = np.exp(np.mean(np.log(psd + 1e-10)))
    am = np.mean(psd)
    feat["spectral_flatness"] = float(gm / (am + 1e-10))

    p_ent = psd_n + 1e-10
    feat["spectral_entropy"] = float(-np.sum(p_ent * np.log2(p_ent)) / np.log2(len(p_ent)))

    feat["spectral_crest"] = float(np.max(psd) / (np.mean(psd) + 1e-10))

    # Spectral slope / intercept (log-frequency linear regression)
    fl = np.log(freqs[1:] + 1)
    pdb = 10 * np.log10(psd[1:] + 1e-10)
    c = np.polyfit(fl, pdb, 1)
    feat["spectral_slope"] = float(c[0])
    feat["spectral_intercept"] = float(c[1])

    # Peak frequencies
    peaks, _ = sig.find_peaks(psd, height=np.max(psd) * 0.1)
    if len(peaks):
        order = np.argsort(psd[peaks])[::-1]
        for i in range(min(5, len(order))):
            idx = peaks[order[i]]
            feat[f"peak_freq_{i + 1}"] = float(freqs[idx])
            feat[f"peak_mag_{i + 1}"] = float(psd[idx])
    else:
        for i in range(1, 6):
            feat[f"peak_freq_{i}"] = 0.0
            feat[f"peak_mag_{i}"] = 0.0

    if len(psd) > 1:
        k = np.arange(1, len(psd))
        feat["spectral_decrease"] = float(
            np.sum((psd[1:] - psd[0]) / k) / (np.sum(psd[1:]) + 1e-10)
        )
    else:
        feat["spectral_decrease"] = 0.0

    return feat


#
# 3. Band-power features
#

_BANDS = {
    "very_low": (100, 2500),
    "low": (2500, 7500),
    "mid": (7500, 10000),
    "high": (10000, 12500),
    "very_high": (12500, 90000),
    "sub_500": (100, 500),
    "band_500_1.5k": (500, 1500),
    "band_1.5k_2.5k": (1500, 2500),
    "band_2.5k_5k": (2500, 5000),
    "band_5k_7.5k": (5000, 7500),
    "band_12.5k_20k": (12500, 20000),
    "band_20k_32k": (20000, 32000),
}


def compute_band_power_features(
    y: np.ndarray, sr: int, bands: list[tuple[float, float]] | None = None
) -> dict[str, float]:
    x = np.asarray(y, dtype=np.float64)
    feat: dict[str, float] = {}

    band_dict = _BANDS if bands is None else {f"{int(lo)}_{int(hi)}": (lo, hi) for lo, hi in bands}

    powers: dict[str, float] = {}
    for name, (flo, fhi) in band_dict.items():
        if flo >= sr / 2:
            powers[name] = 0.0
            feat[f"band_power_{name}"] = 0.0
            feat[f"band_power_{name}_db"] = -100.0
            continue
        try:
            xf = _bandpass(x, sr, flo, min(fhi, sr / 2 - 100.0))
            p = float(np.sqrt(np.mean(xf**2)))
        except Exception:
            p = 0.0
        powers[name] = p
        feat[f"band_power_{name}"] = p
        feat[f"band_power_{name}_db"] = 20 * np.log10(p + 1e-10)

    total = sum(powers.values()) + 1e-10
    for name, p in powers.items():
        feat[f"band_ratio_{name}"] = p / total

    feat["ratio_mid_to_verylow"] = powers.get("mid", 0.0) / (powers.get("very_low", 0.0) + 1e-10)
    feat["ratio_high_to_verylow"] = powers.get("high", 0.0) / (powers.get("very_low", 0.0) + 1e-10)
    feat["ratio_high_to_low"] = powers.get("high", 0.0) / (powers.get("low", 0.0) + 1e-10)
    feat["ratio_mid_to_low"] = powers.get("mid", 0.0) / (powers.get("low", 0.0) + 1e-10)
    feat["ratio_veryhigh_to_mid"] = powers.get("very_high", 0.0) / (powers.get("mid", 0.0) + 1e-10)

    return feat


#
# 4. Machining-specific features  (airborne only)
#


def compute_machining_features(
    y: np.ndarray,
    sr: int,
    env_sr: int = _MACHINING_ENV_SR_DEFAULT,
    hf_sr: int = _MACHINING_HF_SR_DEFAULT,
    hf_proxy: str = _MACHINING_HF_PROXY_DEFAULT,
) -> dict[str, float]:
    feat: dict[str, float] = {
        "tpf_hz": 0.0,
        "modulation_depth_0.1k_2.5kHz": 0.0,
        "force_proxy_peak_1k_5kHz": 0.0,
        "force_proxy_variation_1k_5kHz": 0.0,
        "temporal_stability": 0.0,
        "autocorr_10ms": 0.0,
        "roughness_proxy_rms_5k_12.5kHz": 0.0,
        "roughness_spectral_centroid": 0.0,
        "roughness_spectral_entropy": 0.0,
        "acoustic_emission_rms": 0.0,
    }

    x = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 32:
        return feat

    # Envelope block (low-rate copy)
    x_env, sr_env = _resample_if_needed(x, sr, env_sr)
    analytic = sig.hilbert(x_env)
    envelope = np.abs(analytic)
    env_det = sig.detrend(envelope)

    nfft_env = int(min(8192, 2 ** math.ceil(math.log2(max(256, len(env_det))))))
    env_fft = np.abs(fft(env_det, n=nfft_env))
    env_freqs = fftfreq(nfft_env, 1.0 / sr_env)[: nfft_env // 2]
    env_mag = env_fft[: nfft_env // 2]

    if len(env_freqs) > 0:
        f_hi = min(5000.0, float(env_freqs[-1]))
        band = (env_freqs >= 50.0) & (env_freqs <= f_hi)
        if np.any(band):
            idx = int(np.argmax(env_mag[band]))
            feat["tpf_hz"] = float(env_freqs[band][idx])

    feat["modulation_depth_0.1k_2.5kHz"] = float(np.std(envelope) / (np.mean(envelope) + 1e-10))

    if sr_env > 10000:
        x_force = _bandpass(x_env, sr_env, 1000, 5000)
        feat["force_proxy_peak_1k_5kHz"] = float(np.max(np.abs(x_force)))
        from scipy.ndimage import uniform_filter1d

        w = max(8, int(0.01 * sr_env))
        mrms = np.sqrt(uniform_filter1d(x_force**2, size=w, mode="nearest"))
        feat["force_proxy_variation_1k_5kHz"] = float(np.std(mrms) / (np.mean(mrms) + 1e-10))

    n_seg = min(10, max(2, len(x_env) // max(1, int(0.02 * sr_env))))
    if n_seg >= 2:
        parts = np.array_split(x_env, n_seg)
        seg_rms = np.array([np.sqrt(np.mean(p**2)) for p in parts])
        feat["temporal_stability"] = float(np.std(seg_rms) / (np.mean(seg_rms) + 1e-10))

    lag = min(int(0.01 * sr_env), len(x_env) - 1)
    if lag > 0:
        feat["autocorr_10ms"] = float(
            np.dot(x_env[:-lag], x_env[lag:]) / (np.dot(x_env, x_env) + 1e-10)
        )

    # HF proxy block
    x_hf, sr_hf = _resample_if_needed(x, sr, hf_sr)

    if hf_proxy == "ae":
        f_hi = min(30000.0, sr_hf / 2 - 100.0)
        if f_hi > 12500.0:
            xb = _bandpass(x_hf, sr_hf, 12500.0, f_hi)
            feat["acoustic_emission_rms"] = float(np.sqrt(np.mean(xb**2)))
    else:
        f_lo, f_hi = 5000.0, min(12500.0, sr_hf / 2 - 100.0)
        if f_hi > f_lo:
            xb = _bandpass(x_hf, sr_hf, f_lo, f_hi)
            feat["roughness_proxy_rms_5k_12.5kHz"] = float(np.sqrt(np.mean(xb**2)))

            nfft_hf = int(min(8192, 2 ** math.ceil(math.log2(max(256, len(xb))))))
            spec = np.abs(np.fft.rfft(xb, n=nfft_hf)) ** 2
            freqs_r = np.fft.rfftfreq(nfft_hf, d=1.0 / sr_hf)
            bm = (freqs_r >= f_lo) & (freqs_r <= f_hi)
            bp, bf = spec[bm], freqs_r[bm]

            if bp.size > 1:
                ps = float(np.sum(bp))
                if ps > 1e-20:
                    p = bp / ps
                    feat["roughness_spectral_centroid"] = float(np.sum(bf * p))
                    feat["roughness_spectral_entropy"] = float(-np.sum(p * np.log2(p + 1e-12)))

    return feat


#
# 5. Time-frequency features  (STFT-based)
#


def compute_timefrequency_features(
    y: np.ndarray, sr: int, nperseg: int = 8192, hop_length: int = 2048
) -> dict[str, float]:
    x = np.asarray(y, dtype=np.float64)
    feat: dict[str, float] = {}

    freqs, times, Zxx = sig.stft(x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop_length)
    mag = np.abs(Zxx)
    pwr = mag**2

    # Dominant frequency over time
    if pwr.shape[1] > 0:
        dom_idx = np.argmax(pwr, axis=0)
        dom_f = freqs[dom_idx]
        feat["spec_dom_freq_hz"] = float(np.mean(dom_f))
        feat["spec_dom_freq_std_hz"] = float(np.std(dom_f))

        gm = np.exp(np.mean(np.log(pwr + 1e-10), axis=0))
        am = np.mean(pwr + 1e-10, axis=0)
        tonal = 1.0 - gm / (am + 1e-10)
        feat["spec_tonalness_mean"] = float(np.mean(tonal))
    else:
        feat["spec_dom_freq_hz"] = 0.0
        feat["spec_dom_freq_std_hz"] = 0.0
        feat["spec_tonalness_mean"] = 0.0

    # Spectral flux
    flux = np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0))
    feat["spectral_flux_mean"] = float(np.mean(flux))
    feat["spectral_flux_std"] = float(np.std(flux))
    feat["spectral_flux_max"] = float(np.max(flux)) if flux.size else 0.0

    # Spectral variation
    sv = np.std(mag, axis=1) / (np.mean(mag, axis=1) + 1e-10)
    feat["spectral_variation_mean"] = float(np.mean(sv))
    feat["spectral_variation_median"] = float(np.median(sv))

    # Temporal centroid
    energy_env = np.sum(pwr, axis=0)
    energy_norm = energy_env / (np.sum(energy_env) + 1e-10)
    feat["temporal_centroid"] = float(np.sum(times * energy_norm))

    # Per-band time-frequency statistics
    tf_bands = {
        "very_low": (100, 2500),
        "low": (2500, 7500),
        "mid": (7500, 10000),
        "high": (10000, 12500),
        "very_high": (12500, 90000),
    }
    total_e_time = np.sum(pwr, axis=0) + 1e-10
    for bname, (flo, fhi) in tf_bands.items():
        bm = (freqs >= flo) & (freqs <= fhi)
        if np.any(bm):
            be = np.sum(pwr[bm, :], axis=0)
            feat[f"tf_band_ratio_{bname}_mean"] = float(np.mean(be / total_e_time))
            feat[f"{bname}_band_energy_mean"] = float(np.mean(be))
            feat[f"{bname}_band_energy_std"] = float(np.std(be))
            feat[f"{bname}_band_energy_cv"] = float(np.std(be) / (np.mean(be) + 1e-10))
        else:
            feat[f"tf_band_ratio_{bname}_mean"] = 0.0
            feat[f"{bname}_band_energy_mean"] = 0.0
            feat[f"{bname}_band_energy_std"] = 0.0
            feat[f"{bname}_band_energy_cv"] = 0.0

    # Spectral centroid modulation over time
    pwr_n = pwr / (np.sum(pwr, axis=0, keepdims=True) + 1e-10)
    cot = np.sum(freqs[:, np.newaxis] * pwr_n, axis=0)
    feat["centroid_modulation_mean"] = float(np.mean(cot))
    feat["centroid_modulation_std"] = float(np.std(cot))
    feat["centroid_modulation_range"] = float(np.ptp(cot))

    # Bandwidth modulation
    bwt = np.sqrt(np.sum(((freqs[:, np.newaxis] - cot[np.newaxis, :]) ** 2) * pwr_n, axis=0))
    feat["bandwidth_modulation_mean"] = float(np.mean(bwt))
    feat["bandwidth_modulation_std"] = float(np.std(bwt))

    # Rolloff modulation
    rolloff_vals = []
    for t in range(pwr.shape[1]):
        cs = np.cumsum(pwr[:, t])
        ri = np.where(cs >= 0.85 * cs[-1])[0]
        if len(ri):
            rolloff_vals.append(freqs[ri[0]])
    if rolloff_vals:
        ra = np.array(rolloff_vals)
        feat["rolloff_modulation_mean"] = float(np.mean(ra))
        feat["rolloff_modulation_std"] = float(np.std(ra))
    else:
        feat["rolloff_modulation_mean"] = 0.0
        feat["rolloff_modulation_std"] = 0.0

    return feat


#
# 6. Statistical features
#


def compute_statistical_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """General statistical features (original extraction.py)."""
    x = np.asarray(y, dtype=np.float64)
    abs_x = np.abs(x)
    feat: dict[str, float] = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "variance": float(np.var(x)),
        "abs_mean": float(np.mean(abs_x)),
        "abs_std": float(np.std(abs_x)),
        "mad": float(np.median(np.abs(x - np.median(x)))),
        "range": float(np.max(x) - np.min(x)),
        "abs_range": float(np.max(abs_x) - np.min(abs_x)),
    }
    thr = 0.1 * np.max(abs_x) if abs_x.size else 0.0
    cnt = int(np.sum(abs_x > thr))
    feat["count_above_threshold"] = float(cnt)
    feat["ratio_above_threshold"] = float(cnt / len(x))
    feat["stat_iqr"] = float(np.subtract(*np.percentile(x, [75, 25])))
    feat["stat_median_abs_dev"] = feat["mad"]

    return feat


#
# 7. Short-time statistics
#


def compute_short_time_features(
    y: np.ndarray, sr: int, frame_ms: float = 2.0, hop_ms: float | None = None, overlap: float = 0.5
) -> dict[str, float]:
    x = np.asarray(y, dtype=np.float64).ravel()
    feat: dict[str, float] = {}

    frame_len = max(64, int(frame_ms * 1e-3 * sr))
    if hop_ms is not None:
        hop_len = max(1, int(hop_ms * 1e-3 * sr))
    else:
        hop_len = max(1, int(frame_len * (1.0 - overlap)))

    if len(x) < frame_len:
        frames = x[np.newaxis, :]
    else:
        windows = sliding_window_view(x, frame_len)
        frames = windows[::hop_len]
        if frames.size == 0:
            frames = x[np.newaxis, :]

    frame_rms = np.sqrt(np.mean(frames**2, axis=1))
    frame_peak = np.max(np.abs(frames), axis=1)
    frame_crest = frame_peak / (frame_rms + 1e-10)
    frame_kurt = np.asarray(stats.kurtosis(frames, axis=1, fisher=True), dtype=np.float64)
    frame_skew = np.asarray(stats.skew(frames, axis=1), dtype=np.float64)

    if frames.shape[1] >= 3:
        frame_tkeo = np.mean(np.abs(frames[:, 1:-1] ** 2 - frames[:, :-2] * frames[:, 2:]), axis=1)
    else:
        frame_tkeo = np.zeros(frames.shape[0], dtype=np.float64)

    frame_zcr = np.sum(np.diff(np.sign(frames), axis=1) != 0, axis=1) / (2 * frames.shape[1]) * sr

    def _agg(arr: np.ndarray, name: str) -> None:
        feat[f"st_{name}_mean"] = float(np.mean(arr))
        feat[f"st_{name}_std"] = float(np.std(arr))
        feat[f"st_{name}_max"] = float(np.max(arr))
        feat[f"st_{name}_range"] = float(np.ptp(arr))

    _agg(frame_rms, "rms")
    _agg(frame_crest, "crest_factor")
    _agg(frame_kurt, "kurtosis")
    _agg(frame_skew, "skewness")
    _agg(frame_tkeo, "tkeo")
    _agg(frame_zcr, "zcr_hz")

    if len(frame_rms) > 1:
        t = np.linspace(0, 1, len(frame_rms))
        feat["st_rms_trend"] = float(np.polyfit(t, frame_rms, 1)[0])
    else:
        feat["st_rms_trend"] = 0.0

    feat["st_kurt_impulsive_ratio"] = float(np.mean(frame_kurt > 3.0))
    return feat


#
# 8. DWT features
#


def compute_dwt_features(
    y: np.ndarray, sr: int, wavelet: str = "db4", max_level: int = 8
) -> dict[str, float]:
    x = np.asarray(y, dtype=np.float64).ravel()
    feat: dict[str, float] = {}

    # Zero-fill schema if signal too short
    if x.size < 256 or not np.isfinite(x).any():
        for k in range(1, max_level + 1):
            feat[f"dwt_cD{k}_energy_ratio"] = 0.0
            feat[f"dwt_cD{k}_rms"] = 0.0
            feat[f"dwt_cD{k}_entropy"] = 0.0
        feat.update(
            {
                "dwt_total_energy": 0.0,
                "dwt_entropy_mean": 0.0,
                "dwt_energy_ratio_hf": 0.0,
                "dwt_energy_ratio_mf": 0.0,
                "dwt_energy_ratio_lf": 0.0,
            }
        )
        return feat

    w = pywt.Wavelet(wavelet)
    level = int(max(1, min(max_level, pywt.dwt_max_level(x.size, w.dec_len))))
    coeffs = pywt.wavedec(x, wavelet, level=level)
    eps = 1e-10
    energies = [float(np.sum(c**2)) for c in coeffs]
    total = float(np.sum(energies)) + eps
    feat["dwt_total_energy"] = total - eps

    entropies, ratios = [], []
    for k, cD in enumerate(coeffs[1:][::-1], start=1):
        e = float(np.sum(cD**2))
        r = e / total
        ratios.append(r)
        feat[f"dwt_cD{k}_energy_ratio"] = r
        feat[f"dwt_cD{k}_rms"] = float(np.sqrt(np.mean(cD**2)))
        p = (cD**2) / (e + eps)
        ent = float(-np.sum(p * np.log2(p + eps)))
        feat[f"dwt_cD{k}_entropy"] = ent
        entropies.append(ent)

    feat["dwt_entropy_mean"] = float(np.mean(entropies)) if entropies else 0.0

    hf = float(np.sum(ratios[: min(2, len(ratios))]))
    lf = float(np.sum(ratios[max(0, len(ratios) - 2) :]))
    mf = float(max(0.0, sum(ratios) - hf - lf))
    feat["dwt_energy_ratio_hf"] = hf
    feat["dwt_energy_ratio_mf"] = mf
    feat["dwt_energy_ratio_lf"] = lf
    return feat


#
# 9. CWT features  (complex Morlet, physical band grouping)
#


def compute_cwt_features(
    y: np.ndarray,
    sr: int,
    wavelet: str = "cmor1.5-1.0",
    n_scales: int = 32,
    ds_factor: int = 4,
    fmin: float = 200.0,
    fmax: float | None = None,
    num_scales: int | None = None,
    cwt_fmin: float | None = None,
) -> dict[str, float]:
    if num_scales is not None:
        n_scales = num_scales
    if cwt_fmin is not None:
        fmin = cwt_fmin

    x = np.asarray(y, dtype=np.float64).ravel()
    feat: dict[str, float] = {}

    x_ds = _decimate(x, ds_factor, ftype="fir", zero_phase=True)
    sr_ds = sr // ds_factor
    f_max = (sr_ds / 2.0 * 0.95) if fmax is None else min(float(fmax), sr_ds / 2.0 * 0.95)

    wav_obj = pywt.ContinuousWavelet(wavelet)
    fc = pywt.central_frequency(wav_obj)
    freqs_hz = np.logspace(np.log10(max(fmin, 1.0)), np.log10(f_max), n_scales)
    scales = fc * sr_ds / freqs_hz

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs, _ = pywt.cwt(x_ds, scales, wavelet, sampling_period=1.0 / sr_ds)
    power = np.abs(coeffs) ** 2
    total = np.sum(power) + 1e-10

    phys_bands = {
        "structural": (200, 2500),
        "cutting_force": (2500, 7500),
        "surface_ae": (7500, 20000),
        "ultrasonic": (20000, f_max),
    }

    for bname, (flo, fhi) in phys_bands.items():
        fhi = min(fhi, f_max)
        if flo >= fhi:
            continue
        bm = (freqs_hz >= flo) & (freqs_hz < fhi)
        if not np.any(bm):
            continue
        bp = power[bm, :]
        feat[f"cwt_{bname}_rms"] = float(np.sqrt(np.mean(bp)))
        feat[f"cwt_{bname}_mean_power"] = float(np.mean(bp))
        feat[f"cwt_{bname}_max_power"] = float(np.max(bp))
        feat[f"cwt_{bname}_std_power"] = float(np.std(bp))
        feat[f"cwt_{bname}_rel_energy"] = float(np.sum(bp)) / total
        p_n = bp / (np.sum(bp) + 1e-10)
        p_f = p_n.ravel()
        p_f = p_f[p_f > 0]
        feat[f"cwt_{bname}_entropy"] = float(-np.sum(p_f * np.log2(p_f)))

    e_struct = feat.get("cwt_structural_rel_energy", 0.0)
    e_cf = feat.get("cwt_cutting_force_rel_energy", 0.0)
    e_surf = feat.get("cwt_surface_ae_rel_energy", 0.0)
    e_ultra = feat.get("cwt_ultrasonic_rel_energy", 0.0)
    feat["cwt_ratio_ultrasonic_to_structural"] = e_ultra / (e_struct + 1e-10)
    feat["cwt_ratio_cuttingforce_to_structural"] = e_cf / (e_struct + 1e-10)
    feat["cwt_ratio_high_to_low"] = (e_surf + e_ultra) / (e_struct + e_cf + 1e-10)

    ridge = freqs_hz[np.argmax(power, axis=0)]
    feat["cwt_ridge_freq_mean"] = float(np.mean(ridge))
    feat["cwt_ridge_freq_std"] = float(np.std(ridge))
    feat["cwt_ridge_freq_range"] = float(np.ptp(ridge))
    feat["cwt_ridge_stability"] = float(np.std(ridge) / (np.mean(ridge) + 1e-10))

    tp = np.sum(power, axis=0)
    feat["cwt_temporal_power_mean"] = float(np.mean(tp))
    feat["cwt_temporal_power_std"] = float(np.std(tp))
    feat["cwt_temporal_power_max"] = float(np.max(tp))
    feat["cwt_temporal_power_skew"] = float(stats.skew(tp))
    feat["cwt_temporal_power_kurt"] = float(stats.kurtosis(tp, fisher=True))
    t_n = np.linspace(0, 1, len(tp))
    tp_n = tp / (np.sum(tp) + 1e-10)
    feat["cwt_temporal_centroid"] = float(np.sum(t_n * tp_n))

    p_g = power.ravel() / total
    p_g = p_g[p_g > 0]
    feat["cwt_scalogram_entropy"] = float(-np.sum(p_g * np.log2(p_g)))
    feat["cwt_scalogram_kurtosis"] = float(stats.kurtosis(power.ravel(), fisher=True))
    return feat


#
# 10. Geometry / mic-position features
#


def _load_hole_pos_map(xlsx_path: str | None) -> dict[str, tuple[float, float]]:
    if xlsx_path is None:
        return {}
    from pathlib import Path

    p = Path(xlsx_path)
    if not p.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        p = project_root / p
    if not p.exists():
        warnings.warn(f"Grid xlsx not found: {p}")
        return {}

    import pandas as pd

    df_pos = pd.read_excel(p)
    req = {"HoleID", "X_mm", "Y_mm"}
    if not req.issubset(set(df_pos.columns)):
        return {}
    m: dict[str, tuple[float, float]] = {}
    for _, r in df_pos.dropna(subset=["HoleID", "X_mm", "Y_mm"]).iterrows():
        hid = str(r["HoleID"]).strip().upper()
        if not re.fullmatch(r"[A-Z]\d+", hid):
            continue
        m[hid] = (float(r["X_mm"]), float(r["Y_mm"]))
    return m


@lru_cache(maxsize=1)
def _get_hole_pos_map(xlsx_path: str | None) -> dict[str, tuple[float, float]]:
    """Cached loader  reads the Excel file at most once per process."""
    return _load_hole_pos_map(xlsx_path)


def _parse_hole_id(record_name: str) -> str | None:
    for pat in [r"hole_([A-Z]\d+)", r"__([A-Z]\d+)__", r"\b([A-Z]\d+)\b"]:
        m = re.search(pat, record_name, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def compute_geometry_features(
    record_name: str,
    grid_xlsx_path: str | None = None,
    mic_x_mm: float = 0.0,
    mic_y_mm: float = 0.0,
) -> dict[str, float]:
    feat: dict[str, float] = {}
    hole_id = _parse_hole_id(record_name)

    hole_pos = _get_hole_pos_map(grid_xlsx_path) if grid_xlsx_path else {}

    if hole_id is not None and hole_id in hole_pos:
        hx, hy = hole_pos[hole_id]
        dx = float(hx - mic_x_mm)
        dy = float(hy - mic_y_mm)
        r_mm = float(math.hypot(dx, dy))
        feat["mic_dx_mm"] = dx
        feat["mic_dy_mm"] = dy
        feat["mic_r_mm"] = r_mm
        feat["mic_log_r"] = float(np.log(max(r_mm, 1e-6)))
        feat["mic_angle_rad"] = float(math.atan2(dy, dx))
    else:
        feat["mic_dx_mm"] = float("nan")
        feat["mic_dy_mm"] = float("nan")
        feat["mic_r_mm"] = float("nan")
        feat["mic_log_r"] = float("nan")
        feat["mic_angle_rad"] = float("nan")

    return feat
