from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import decimate, resample_poly
from torch.utils.data import Dataset

from vm_micro.data.io import get_input_kind, read_measurement_h5

from .config import TrainConfig


@dataclass
class WindowRecord:
    file_id: int
    path: str
    start_target: int
    frames_target: int
    depth_mm: float
    class_idx: int


class AudioCache:
    """Optional in-memory cache for resampled mono audio signals."""

    def __init__(self, enable: bool = True) -> None:
        self.enable = enable
        self.store: dict[str, np.ndarray] = {}

    def get(self, path: str) -> np.ndarray | None:
        if not self.enable:
            return None
        return self.store.get(path)

    def put(self, path: str, value: np.ndarray) -> None:
        if self.enable:
            self.store[path] = value


def _safe_resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float32, copy=False)
    factor = gcd(sr_in, sr_out)
    return resample_poly(y, up=sr_out // factor, down=sr_in // factor).astype(np.float32)


def _load_flac(path: str, target_sr: int) -> np.ndarray:
    """Load a FLAC/WAV file, downmix to mono, resample, peak-normalise."""
    audio, sr = sf.read(path, always_2d=False)
    if np.ndim(audio) > 1:
        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros((1,), dtype=np.float32)

    audio = _safe_resample(audio, int(sr), target_sr)
    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def _load_h5(
    path: str,
    target_sr: int,
    data_key: str = "measurement/data",
    time_key: str = "measurement/time_vector",
) -> np.ndarray:
    """Load an HDF5 measurement file, decimate then resample to target_sr.

    Strategy
    --------
    Structure-borne native SR is ~3.125 MHz  far too high to resample
    directly to 48 kHz in one step (ratio 65:1). A two-stage approach is applied:
      1. Decimate by the largest factor of 10 that keeps the intermediate
         rate above 2  target_sr (anti-alias filter applied automatically
         by scipy.signal.decimate).
      2. Resample the intermediate signal to exactly target_sr using
         resample_poly.
    This avoids large intermediate arrays and numerical precision issues.
    """
    y, sr_native, _tv, _meta = read_measurement_h5(
        path,
        data_key=data_key,
        time_key=time_key,
        center_signal=True,
    )

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    sr = int(sr_native)

    # Stage 1: integer decimation
    while sr // 10 >= 2 * target_sr:
        y = decimate(y, 10, ftype="iir", zero_phase=True)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        sr = sr // 10

    # Stage 2: polyphase resample to exact target_sr
    y = _safe_resample(y.astype(np.float32), int(sr), target_sr)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if not np.isfinite(peak) or peak <= 0.0:
        return np.zeros(max(1, y.size), dtype=np.float32)

    y = y / peak
    return y.astype(np.float32, copy=False)


class WaveformWindowDataset(Dataset):
    """Materialises sliding waveform windows from file-level metadata.

    Supports both FLAC/WAV (airborne) and HDF5 (structure-borne) files
    transparently  format is detected per-file from the extension.
    """

    def __init__(
        self,
        file_df: pd.DataFrame,
        cfg: TrainConfig,
        training: bool,
        h5_data_key: str = "measurement/data",
        h5_time_key: str = "measurement/time_vector",
    ) -> None:
        self.file_df = file_df.reset_index(drop=True).copy()
        self.cfg = cfg
        self.training = training
        self.h5_data_key = h5_data_key
        self.h5_time_key = h5_time_key
        self.cache = AudioCache(enable=cfg.cache_audio)
        self.window_records = self._build_window_records()

        if not self.window_records:
            raise ValueError("No waveform windows were created from the provided files.")

    def _build_window_records(self) -> list[WindowRecord]:
        win_frames = self.cfg.signal_num_samples()
        hop_frames = max(1, int(round(self.cfg.window_hop_sec * self.cfg.sample_rate)))

        records: list[WindowRecord] = []

        for _, row in self.file_df.iterrows():
            duration_sec = float(row.get("duration_sec", 0.0))
            total_frames = max(1, int(round(duration_sec * self.cfg.sample_rate)))

            starts = list(range(0, max(1, total_frames - win_frames + 1), hop_frames))
            if not starts or len(starts) < self.cfg.min_windows_per_file:
                starts = [0]

            if (
                self.training
                and self.cfg.max_windows_per_file_train is not None
                and len(starts) > self.cfg.max_windows_per_file_train
            ):
                rng = np.random.default_rng(self.cfg.seed + int(row["file_id"]))
                starts = sorted(
                    rng.choice(
                        starts,
                        size=self.cfg.max_windows_per_file_train,
                        replace=False,
                    ).tolist()
                )

            for start in starts:
                records.append(
                    WindowRecord(
                        file_id=int(row["file_id"]),
                        path=str(row["path"]),
                        start_target=int(start),
                        frames_target=int(win_frames),
                        depth_mm=float(row["depth_mm"]),
                        class_idx=int(row.get("class_idx", -1)),
                    )
                )

        return records

    def __len__(self) -> int:
        return len(self.window_records)

    def _load_full_audio_resampled(self, path: str) -> np.ndarray:
        """Load and cache the full resampled signal for one file.

        Dispatches to the correct reader based on file extension.
        """
        cached = self.cache.get(path)
        if cached is not None:
            return cached

        kind = get_input_kind(path)

        if kind == "audio":
            audio = _load_flac(path, self.cfg.sample_rate)
        else:  # hdf5
            audio = _load_h5(
                path,
                self.cfg.sample_rate,
                data_key=self.h5_data_key,
                time_key=self.h5_time_key,
            )

        self.cache.put(path, audio)
        return audio

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.window_records[idx]
        waveform_full = self._load_full_audio_resampled(record.path)
        waveform = waveform_full[record.start_target : record.start_target + record.frames_target]

        if waveform.size < record.frames_target:
            waveform = np.pad(waveform, (0, record.frames_target - waveform.size))

        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        waveform_tensor = torch.from_numpy(waveform.astype(np.float32, copy=False))

        if self.cfg.task == "classification":
            target = torch.tensor(record.class_idx, dtype=torch.long)
        else:
            target = torch.tensor(record.depth_mm, dtype=torch.float32)

        return {
            "waveform": waveform_tensor,
            "y": target,
            "file_id": torch.tensor(record.file_id, dtype=torch.long),
            "depth_mm": torch.tensor(record.depth_mm, dtype=torch.float32),
            "class_idx": torch.tensor(record.class_idx, dtype=torch.long),
            "window_start_target": torch.tensor(record.start_target, dtype=torch.long),
            "path": record.path,
        }


def collate_waveforms(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "waveform": torch.stack([item["waveform"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "file_id": torch.stack([item["file_id"] for item in batch], dim=0),
        "depth_mm": torch.stack([item["depth_mm"] for item in batch], dim=0),
        "class_idx": torch.stack([item["class_idx"] for item in batch], dim=0),
        "window_start_target": torch.stack([item["window_start_target"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
    }
    return out
