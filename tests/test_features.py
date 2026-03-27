"""Tests for vm_micro.features  feature extraction correctness."""

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


def _find_band_key(result: dict[str, float], low: int, high: int) -> str:
    low_s = str(low)
    high_s = str(high)
    candidates = [k for k in result if low_s in k and high_s in k and not k.endswith("_ratio")]
    assert candidates, f"No band key found for {low}-{high} Hz in keys={list(result.keys())}"

    for suffix in ("_rms", "_power", "_db"):
        preferred = [k for k in candidates if k.endswith(suffix)]
        if preferred:
            return preferred[0]
    return candidates[0]


#
# All functions return only finite floats
#


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
    y = _white_noise(SR // 4)
    result = compute_cwt_features(y, SR, wavelet="morl", num_scales=8, fmin=200.0, fmax=4000.0)
    assert len(result) > 0
    for k, v in result.items():
        assert np.isfinite(float(v)), f"CWT[{k}] = {v}"


def test_band_power_features():
    y = _sine(freq=1000.0)
    bands = [(100.0, 500.0), (500.0, 2000.0), (2000.0, 10000.0)]
    result = compute_band_power_features(y, SR, bands=bands)

    low_key = _find_band_key(result, 100, 500)
    mid_key = _find_band_key(result, 500, 2000)
    high_key = _find_band_key(result, 2000, 10000)

    assert np.isfinite(result[low_key])
    assert np.isfinite(result[mid_key])
    assert np.isfinite(result[high_key])

    assert result[mid_key] > result[low_key]
    assert result[mid_key] > result[high_key]


#
# Feature families  current API sanity checks
#


def test_time_feature_family_current_keys():
    y = _white_noise()
    result = compute_time_features(y, SR)

    assert "rms" in result
    assert "crest_factor" in result
    assert "shape_factor" in result
    assert "impulse_factor" in result
    assert "clearance_factor" in result
    assert "energy" in result
    assert "zcr" in result
    assert "zcr_hz" in result

    assert result["rms"] > 0
    assert result["crest_factor"] > 0
    assert result["shape_factor"] > 0
    assert result["energy"] > 0


def test_spectral_shape_features():
    y = _white_noise()
    result = compute_frequency_features(y, SR)

    assert "spectral_slope" in result
    assert "spectral_decrease" in result
    assert "spectral_skewness" in result
    assert "spectral_kurtosis" in result

    for k in ["spectral_slope", "spectral_decrease", "spectral_skewness", "spectral_kurtosis"]:
        assert np.isfinite(result[k]), f"{k} = {result[k]}"


def test_dwt_energy_ratios():
    y = _white_noise()
    result = compute_dwt_features(y, SR, wavelet="db8", max_level=6)

    ratio_keys = [k for k in result if k.endswith("_energy_ratio")]
    assert len(ratio_keys) > 0

    ratio_sum = sum(result[k] for k in ratio_keys)
    assert ratio_sum == pytest.approx(1.0, abs=0.02)


def test_short_time_expanded():
    y = _white_noise()
    result = compute_short_time_features(y, SR, frame_ms=10.0, hop_ms=5.0)

    assert len(result) >= 26
    assert "st_rms_mean" in result
    assert "st_rms_std" in result
    assert "st_rms_trend" in result
    assert "st_kurt_impulsive_ratio" in result
    assert "st_zcr_hz_mean" in result


def test_complexity_measures():
    y = _white_noise()
    result = compute_statistical_features(y, SR)

    assert "stat_iqr" in result
    assert "stat_median_abs_dev" in result
    assert "ratio_above_threshold" in result

    assert np.isfinite(result["stat_iqr"])
    assert np.isfinite(result["stat_median_abs_dev"])
    assert 0.0 <= result["ratio_above_threshold"] <= 1.0


#
# Time features  known analytical values
#


def test_time_rms_sine():
    y = _sine(freq=440.0, n=SR * 2)
    result = compute_time_features(y, SR)
    assert result["rms"] == pytest.approx(1.0 / np.sqrt(2), rel=0.01)


def test_time_zero_crossings_sine():
    f = 100.0
    y = _sine(freq=f, n=SR)
    result = compute_time_features(y, SR)

    assert result["zcr_hz"] == pytest.approx(f, rel=0.05)


#
# Short signal edge cases
#


def test_very_short_signal_doesnt_crash():
    y = np.array([0.1, -0.1, 0.2], dtype=np.float64)
    compute_time_features(y, SR)
    compute_frequency_features(y, SR)
    compute_short_time_features(y, SR)
    compute_dwt_features(y, SR)
