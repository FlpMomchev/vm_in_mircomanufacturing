"""Tests for vm_micro.features — feature extraction correctness."""

from __future__ import annotations

import numpy as np
import pytest

from vm_micro.features.core import (
    compute_band_power_features,
    compute_cwt_features,
    compute_dwt_features,
    compute_frequency_features,
    compute_machining_features,
    compute_short_time_features,
    compute_statistical_features,
    compute_time_features,
)

SR = 48_000
RNG = np.random.default_rng(0)


def _white_noise(n: int = SR) -> np.ndarray:
    return RNG.standard_normal(n).astype(np.float64)


def _sine(freq: float = 1000.0, n: int = SR) -> np.ndarray:
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * freq * t)


# ─────────────────────────────────────────────────────────────────────────────
# All functions return only finite floats
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (compute_time_features, {}),
        (compute_frequency_features, {}),
        (compute_statistical_features, {}),
        (compute_short_time_features, {"frame_ms": 10.0, "hop_ms": 5.0}),
        (compute_machining_features, {"env_sr": 1000, "hf_sr": 4000}),
    ],
)
def test_returns_finite_floats(fn, kwargs):
    y = _white_noise()
    result = fn(y, SR, **kwargs)
    assert isinstance(result, dict)
    assert len(result) > 0
    for k, v in result.items():
        assert np.isfinite(v), f"{fn.__name__}[{k}] = {v} is not finite"


def test_dwt_features_finite():
    y = _white_noise()
    result = compute_dwt_features(y, SR, wavelet="db8", max_level=6)
    assert len(result) > 0
    for k, v in result.items():
        assert np.isfinite(v), f"DWT[{k}] = {v}"


def test_cwt_features_finite():
    # Use a short signal for speed
    y = _white_noise(SR // 4)
    result = compute_cwt_features(y, SR, wavelet="morl", num_scales=8, fmin=200.0, fmax=4000.0)
    assert len(result) > 0
    for k, v in result.items():
        assert np.isfinite(float(v)), f"CWT[{k}] = {v}"


def test_band_power_features():
    # A 1 kHz sine should have most power in the 500–2000 Hz band
    y = _sine(freq=1000.0)
    bands = [(100.0, 500.0), (500.0, 2000.0), (2000.0, 10000.0)]
    result = compute_band_power_features(y, SR, bands=bands)
    assert result["band_500_2000_rms"] > result["band_100_500_rms"]
    assert result["band_500_2000_rms"] > result["band_2000_10000_rms"]
    # Band ratios should exist and sum to ~1
    ratio_sum = sum(v for k, v in result.items() if k.endswith("_ratio"))
    assert ratio_sum == pytest.approx(1.0, abs=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# New feature families — basic sanity checks
# ─────────────────────────────────────────────────────────────────────────────


def test_hjorth_parameters():
    y = _white_noise()
    result = compute_time_features(y, SR)
    assert "time_hjorth_activity" in result
    assert "time_hjorth_mobility" in result
    assert "time_hjorth_complexity" in result
    assert result["time_hjorth_activity"] > 0
    assert result["time_hjorth_mobility"] > 0
    # Energy per sample should be close to time_rms^2
    assert result["time_energy_ps"] == pytest.approx(result["time_rms"] ** 2, rel=0.01)


def test_spectral_shape_features():
    y = _white_noise()
    result = compute_frequency_features(y, SR)
    assert "freq_slope" in result
    assert "freq_decrease" in result
    assert "freq_skewness" in result
    assert "freq_kurtosis" in result
    for k in ["freq_slope", "freq_decrease", "freq_skewness", "freq_kurtosis"]:
        assert np.isfinite(result[k]), f"{k} = {result[k]}"


def test_dwt_energy_ratios():
    y = _white_noise()
    result = compute_dwt_features(y, SR, wavelet="db8", max_level=6)
    ratio_keys = [k for k in result if k.endswith("_energy_ratio")]
    assert len(ratio_keys) > 0
    ratio_sum = sum(result[k] for k in ratio_keys)
    assert ratio_sum == pytest.approx(1.0, abs=0.01)
    # Normalised energy should also exist
    normed_keys = [k for k in result if k.endswith("_energy_normed")]
    assert len(normed_keys) == len(ratio_keys)


def test_short_time_expanded():
    y = _white_noise()
    result = compute_short_time_features(y, SR, frame_ms=10.0, hop_ms=5.0)
    # Should have 5 quantities × 6 stats = 30 features
    assert len(result) >= 30
    # Slope and IQR keys should exist
    assert "st_rms_slope" in result
    assert "st_rms_iqr" in result
    assert "st_centroid_mean" in result
    assert "st_energy_ps_med" in result


def test_complexity_measures():
    y = _white_noise()
    result = compute_statistical_features(y, SR)
    assert "stat_perm_entropy" in result
    assert "stat_lempel_ziv" in result
    # Permutation entropy of white noise should be close to 1 (normalised)
    assert result["stat_perm_entropy"] > 0.9
    # Lempel-Ziv of white noise should be high (close to 1)
    assert result["stat_lempel_ziv"] > 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Time features — known analytical values
# ─────────────────────────────────────────────────────────────────────────────


def test_time_rms_sine():
    y = _sine(freq=440.0, n=SR * 2)
    result = compute_time_features(y, SR)
    # RMS of a unit sine = 1/sqrt(2)
    assert result["time_rms"] == pytest.approx(1.0 / np.sqrt(2), rel=0.01)


def test_time_zero_crossings_sine():
    f = 100.0
    y = _sine(freq=f, n=SR)
    result = compute_time_features(y, SR)
    # Expect ~2*f zero-crossings per second (normalised by n-1)
    expected_zcr = 2 * f / (SR - 1)
    assert result["time_zcr"] == pytest.approx(expected_zcr, rel=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Short signal edge cases
# ─────────────────────────────────────────────────────────────────────────────


def test_very_short_signal_doesnt_crash():
    y = np.array([0.1, -0.1, 0.2], dtype=np.float64)
    # These may return empty dicts for very short signals — that's acceptable
    compute_time_features(y, SR)
    compute_frequency_features(y, SR)
    compute_short_time_features(y, SR)
    compute_dwt_features(y, SR)
