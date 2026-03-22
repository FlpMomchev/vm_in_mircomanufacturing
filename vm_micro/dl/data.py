from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import soundfile as sf
import torch
from torch.utils.data import Dataset

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


class WaveformWindowDataset(Dataset):
    """Materializes sliding waveform windows from file-level metadata."""

    def __init__(self, file_df: pd.DataFrame, cfg: TrainConfig, training: bool) -> None:
        self.file_df = file_df.reset_index(drop=True).copy()
        self.cfg = cfg
        self.training = training
        self.cache = AudioCache(enable=cfg.cache_audio)
        self.window_records = self._build_window_records()

        if not self.window_records:
            raise ValueError("No waveform windows were created from the provided files.")

    def _build_window_records(self) -> list[WindowRecord]:
        win_frames = self.cfg.signal_num_samples()
        hop_frames = int(round(self.cfg.window_hop_sec * self.cfg.sample_rate))
        hop_frames = max(1, hop_frames)

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
        cached = self.cache.get(path)
        if cached is not None:
            return cached

        audio, sr = sf.read(path, always_2d=False)
        if np.ndim(audio) > 1:
            audio = np.mean(audio, axis=1)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            audio = np.zeros((1,), dtype=np.float32)

        audio = _safe_resample(audio, int(sr), self.cfg.sample_rate)

        peak = float(np.max(np.abs(audio))) if audio.size else 1.0
        if peak > 0:
            audio = audio / peak

        audio = audio.astype(np.float32, copy=False)
        self.cache.put(path, audio)
        return audio

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.window_records[idx]
        waveform_full = self._load_full_audio_resampled(record.path)
        waveform = waveform_full[
            record.start_target : record.start_target + record.frames_target
        ]

        if waveform.size < record.frames_target:
            waveform = np.pad(waveform, (0, record.frames_target - waveform.size))

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
        }


def collate_waveforms(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["waveform", "y", "file_id", "depth_mm", "class_idx", "window_start_target"]
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}
