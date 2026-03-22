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

@pytest.mark.parametrize("fn,kwargs", [
    (compute_time_features,       {}),
    (compute_frequency_features,  {}),
    (compute_statistical_features,{}),
    (compute_short_time_features, {"frame_ms": 10.0, "hop_ms": 5.0}),
    (compute_machining_features,  {"env_sr": 1000, "hf_sr": 4000}),
])
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
    result = compute_cwt_features(y, SR, wavelet="morl", num_scales=8,
                                  fmin=200.0, fmax=4000.0)
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
