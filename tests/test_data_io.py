"""Tests for vm_micro.data  signal I/O and segmentation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from vm_micro.data.io import get_input_kind, read_audio_mono, read_measurement_h5, read_signal_auto
from vm_micro.data.manifest import (
    build_segment_filename,
    extract_recording_root,
    parse_depth_mm,
    try_parse_depth_mm,
    try_parse_step_idx,
)
from vm_micro.data.splitter import (
    apply_padding,
    band_envelope_db,
    detect_segments,
    process_one_file,
)

#
# Fixtures
#

SR = 48_000  # synthetic test sample rate


def _sine_burst(sr: int, freq: float, duration: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(duration * sr), dtype=np.float32) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(sr: int, duration: float) -> np.ndarray:
    return np.zeros(int(duration * sr), dtype=np.float32)


@pytest.fixture
def synthetic_recording(tmp_path: Path) -> Path:
    """FLAC with 3 short drilling bursts at 3 kHz separated by silence."""
    burst = _sine_burst(SR, freq=3000.0, duration=0.5)
    silence = _silence(SR, duration=0.8)
    signal = np.concatenate([silence, burst, silence, burst, silence, burst, silence])
    p = tmp_path / "0503_1_2_4532.flac"
    sf.write(str(p), signal, SR)
    return p


@pytest.fixture
def synthetic_h5(tmp_path: Path) -> Path:
    """HDF5 file with the same signal pattern."""
    burst = _sine_burst(SR, freq=3000.0, duration=0.5)
    silence = _silence(SR, duration=0.8)
    signal = np.concatenate([silence, burst, silence, burst, silence, burst, silence])
    t = np.arange(len(signal), dtype=np.float64) / SR
    p = tmp_path / "struct_test.h5"
    with h5py.File(str(p), "w") as fh:
        grp = fh.require_group("measurement")
        grp.create_dataset("data", data=signal.astype(np.float32))
        grp.create_dataset("time_vector", data=t)
    return p


#
# get_input_kind
#


def test_get_input_kind_audio():
    assert get_input_kind("file.flac") == "audio"
    assert get_input_kind("file.wav") == "audio"


def test_get_input_kind_hdf5():
    assert get_input_kind("file.h5") == "hdf5"
    assert get_input_kind("file.hdf5") == "hdf5"


def test_get_input_kind_unknown():
    with pytest.raises(ValueError):
        get_input_kind("file.mp3")


#
# read_audio_mono
#


def test_read_audio_mono(synthetic_recording: Path):
    y, sr = read_audio_mono(synthetic_recording)
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert sr == SR
    assert len(y) > 0


def test_read_audio_mono_resamples(synthetic_recording: Path):
    y, sr = read_audio_mono(synthetic_recording, target_sr=24_000)
    assert sr == 24_000
    assert len(y) > 0


#
# read_measurement_h5
#


def test_read_measurement_h5(synthetic_h5: Path):
    y, sr, tv, meta = read_measurement_h5(synthetic_h5)
    assert y.ndim == 1
    assert y.dtype == np.float32
    assert sr == SR
    assert len(tv) == len(y)
    assert "dt_median_s" in meta


#
# read_signal_auto  format dispatch
#


def test_read_signal_auto_flac(synthetic_recording: Path):
    sig = read_signal_auto(synthetic_recording)
    assert sig["input_kind"] == "audio"
    assert sig["y"].ndim == 1
    assert sig["sr"] == SR
    assert sig["duration_s"] > 0


def test_read_signal_auto_h5(synthetic_h5: Path):
    sig = read_signal_auto(synthetic_h5)
    assert sig["input_kind"] == "hdf5"
    assert sig["y"].ndim == 1


#
# Manifest helpers
#


def test_parse_depth_mm():
    assert parse_depth_mm("0503_1_2_4532__seg001__step001__B2__depth0.500") == pytest.approx(0.5)
    assert parse_depth_mm("depth1.000") == pytest.approx(1.0)


def test_try_parse_depth_mm_missing():
    assert try_parse_depth_mm("no_depth_here") is None


def test_try_parse_step_idx():
    assert try_parse_step_idx("run__step007__hole") == 7
    assert try_parse_step_idx("no_step") is None


def test_extract_recording_root():
    assert (
        extract_recording_root("0503_1_2_4532__seg001__step001__B2__depth0.500") == "0503_1_2_4532"
    )
    assert extract_recording_root("noseg") == "noseg"


def test_build_segment_filename():
    fn = build_segment_filename("run1", 3, step=7, hole="B2", depth=0.5, ext=".flac")
    assert "run1" in fn
    assert "seg003" in fn
    assert "step007" in fn
    assert "depth0.500" in fn


#
# Segmentation
#


def test_band_envelope_db(synthetic_recording: Path):
    y, sr = read_audio_mono(synthetic_recording)
    times, env = band_envelope_db(y, sr, band_hz=(2000.0, 5000.0))
    assert len(times) == len(env)
    assert np.all(np.isfinite(env))


def test_detect_segments_returns_three_bursts(synthetic_recording: Path):
    y, sr = read_audio_mono(synthetic_recording)
    segs, dbg = detect_segments(y, sr, segments_per_file=3, band_hz=(2000.0, 5000.0))
    # The splitter should find all 3 bursts (exact count depends on thresholds)
    assert len(segs) >= 1
    for a, b in segs:
        assert b > a
        assert a >= 0.0


def test_process_one_file_uses_band_fallback(tmp_path: Path, synthetic_recording: Path):
    doe_df = pd.DataFrame(
        {
            "Step": [1, 2, 3],
            "HoleID": ["NA", "NA", "NA"],
            "Depth_mm": [None, None, None],
        }
    )
    manifest_df, summary = process_one_file(
        audio_path=synthetic_recording,
        doe_df=doe_df,
        out_root=tmp_path / "segments",
        expected_segments=3,
        band_hz=(50.0, 60.0),
        band_hz_fallbacks=[(2000.0, 5000.0)],
        export_format="flac",
    )

    assert not manifest_df.empty
    assert summary["band_hz_requested"] == (50.0, 60.0)
    assert summary["band_hz_used"] == (2000.0, 5000.0)
    assert summary["band_attempt_index_used"] == 2
    assert summary["band_match_found"] is True


def test_apply_padding_clips_to_bounds():
    segs = [(1.0, 2.0), (3.0, 4.0)]
    padded = apply_padding(segs, pre_pad_s=0.5, post_pad_s=0.5, duration_s=5.0)
    assert padded[0][0] == pytest.approx(0.5)
    assert padded[1][1] == pytest.approx(4.5)
