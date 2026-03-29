"""vm_micro.features.airborne
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Airborne acoustic feature extractor.

Reads segmented FLAC files, computes all feature families configured in
``configs/airborne.yaml``, and assembles a tidy row per segment.

CLI entry point: ``vm-extract-air``  (see scripts/extract_airborne.py).
"""

from __future__ import annotations

import math
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm.auto import tqdm

from ..data.manifest import extract_recording_root, try_parse_depth_mm, try_parse_step_idx
from ..utils import get_logger
from .core import (
    compute_band_power_features,
    compute_cwt_features,
    compute_dwt_features,
    compute_frequency_features,
    compute_geometry_features,
    compute_machining_features,
    compute_short_time_features,
    compute_statistical_features,
    compute_time_features,
    compute_timefrequency_features,
)

logger = get_logger(__name__)

_DEFAULT_BANDS = [(100.0, 1000.0), (1000.0, 5000.0), (5000.0, 20000.0), (20000.0, 96000.0)]
DEFAULT_FILE_GLOB = "**/*.flac"
DEFAULT_N_WORKERS = 6
DEFAULT_TARGET_SR = 192_000
DEFAULT_STFT_WINDOW = "hann"
DEFAULT_DWT_PADDING_MODE = "symmetric"
DEFAULT_AIRBORNE_FAMILIES: dict[str, bool] = {
    "time": True,
    "frequency": True,
    "band_power": True,
    "machining": True,
    "statistical": True,
    "short_time": True,
    "dwt": True,
    "cwt": True,
    "timefrequency": True,
}


def _safe_resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float64, copy=False)
    factor = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(
        y.astype(np.float64, copy=False),
        up=int(sr_out) // factor,
        down=int(sr_in) // factor,
    ).astype(np.float64, copy=False)


def _resolve_feature_families(cfg: dict[str, Any]) -> dict[str, bool]:
    raw = cfg.get("feature_families", {})
    if not isinstance(raw, dict):
        raw = {}
    out: dict[str, bool] = {}
    for key, default in DEFAULT_AIRBORNE_FAMILIES.items():
        out[key] = bool(raw.get(key, default))
    return out


def _resolve_band_power_bands(cfg: dict[str, Any]) -> list[tuple[float, float]]:
    raw_bands = cfg.get("band_power_bands", _DEFAULT_BANDS)
    bands: list[tuple[float, float]] = []
    if isinstance(raw_bands, (list, tuple)):
        for item in raw_bands:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                lo = float(item[0])
                hi = float(item[1])
                if hi > lo:
                    bands.append((lo, hi))
    if not bands:
        bands = list(_DEFAULT_BANDS)
    return bands


def resolve_effective_airborne_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return effective airborne extraction settings for provenance/replay."""
    effective: dict[str, Any] = {
        "target_sr": (
            int(round(float(cfg["target_sr"]))) if cfg.get("target_sr", None) is not None else None
        ),
        "mic_x_mm": float(cfg.get("mic_x_mm", -257.0)),
        "mic_y_mm": float(cfg.get("mic_y_mm", -40.0)),
        "band_power_bands": [[lo, hi] for lo, hi in _resolve_band_power_bands(cfg)],
        "machining_env_sr": int(cfg.get("machining_env_sr", 24_000)),
        "machining_hf_sr": int(cfg.get("machining_hf_sr", 64_000)),
        "machining_hf_proxy": str(cfg.get("machining_hf_proxy", "roughness")),
        "nperseg": int(cfg.get("nperseg", 8192)),
        "hop_length": int(cfg.get("hop_length", 2048)),
        "stft_window": str(cfg.get("stft_window", DEFAULT_STFT_WINDOW)),
        "dwt_wavelet": str(cfg.get("dwt_wavelet", "db8")),
        "dwt_max_level": int(cfg.get("dwt_max_level", 8)),
        "dwt_padding_mode": str(cfg.get("dwt_padding_mode", DEFAULT_DWT_PADDING_MODE)),
        "cwt_wavelet": str(cfg.get("cwt_wavelet", "morl")),
        "cwt_num_scales": int(cfg.get("cwt_num_scales", 64)),
        "cwt_fmin": float(cfg.get("cwt_fmin", 200.0)),
        "cwt_fmax": float(cfg.get("cwt_fmax", 20000.0)),
        "short_time_frame_ms": float(cfg.get("short_time_frame_ms", 10.0)),
        "short_time_hop_ms": float(cfg.get("short_time_hop_ms", 5.0)),
        "skip_start_s": float(cfg.get("skip_start_s", 0.0)),
        "skip_end_s": float(cfg.get("skip_end_s", 0.0)),
        "feature_families": _resolve_feature_families(cfg),
        "file_glob": str(cfg.get("file_glob", DEFAULT_FILE_GLOB)),
        "n_workers": int(cfg.get("n_workers", DEFAULT_N_WORKERS)),
    }
    return effective


def extract_one_file(
    path: Path,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        audio, sr_native = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        y = audio.astype(np.float64, copy=False)

        target_sr_raw = cfg.get("target_sr", DEFAULT_TARGET_SR)
        if target_sr_raw is not None:
            target_sr = int(round(float(target_sr_raw)))
            if target_sr <= 0:
                raise ValueError(f"target_sr must be > 0, got {target_sr}.")
        else:
            target_sr = int(sr_native)

        sr_used = int(sr_native)
        if sr_used != target_sr:
            y = _safe_resample(y, sr_used, target_sr)
            sr_used = int(target_sr)

        skip_s = float(cfg.get("skip_start_s", 0.0))
        end_s = float(cfg.get("skip_end_s", 0.0))
        if skip_s > 0:
            y = y[int(round(skip_s * sr_used)) :]
        if end_s > 0:
            trim = int(round(end_s * sr_used))
            if trim > 0:
                y = y[:-trim]

        if len(y) < 16:
            return None

        y64 = y.astype(np.float64, copy=False)
        families = _resolve_feature_families(cfg)
        feats: dict[str, float] = {}

        _rec_root = extract_recording_root(path.stem)

        _grid_xlsx = (
            "docs/doe/Design_of_Experiment_7353.xlsx"
            if _rec_root == "0503_4_1_7353"
            else "docs/doe/Design_of_Experiment.xlsx"
        )

        feats.update(
            compute_geometry_features(
                record_name=path.stem,
                grid_xlsx_path=_grid_xlsx,
                mic_x_mm=cfg.get("mic_x_mm", -257.0),
                mic_y_mm=cfg.get("mic_y_mm", -40.0),
            )
        )

        if families.get("time", True):
            feats.update(compute_time_features(y64, sr_used))
        if families.get("frequency", True):
            feats.update(compute_frequency_features(y64, sr_used))
        if families.get("band_power", True):
            bands = _resolve_band_power_bands(cfg)
            feats.update(compute_band_power_features(y64, sr_used, bands=bands))
        if families.get("machining", True):
            feats.update(
                compute_machining_features(
                    y64,
                    sr_used,
                    env_sr=int(cfg.get("machining_env_sr", 24000)),
                    hf_sr=int(cfg.get("machining_hf_sr", 64000)),
                    hf_proxy=str(cfg.get("machining_hf_proxy", "roughness")),
                )
            )
        if families.get("statistical", True):
            feats.update(compute_statistical_features(y64, sr_used))
        if families.get("short_time", True):
            feats.update(
                compute_short_time_features(
                    y64,
                    sr_used,
                    frame_ms=float(cfg.get("short_time_frame_ms", 10.0)),
                    hop_ms=float(cfg.get("short_time_hop_ms", 5.0)),
                )
            )
        if families.get("dwt", True):
            feats.update(
                compute_dwt_features(
                    y64,
                    sr_used,
                    wavelet=str(cfg.get("dwt_wavelet", "db8")),
                    max_level=int(cfg.get("dwt_max_level", 8)),
                    padding_mode=str(cfg.get("dwt_padding_mode", DEFAULT_DWT_PADDING_MODE)),
                )
            )
        if families.get("cwt", True):
            feats.update(
                compute_cwt_features(
                    y64,
                    sr_used,
                    wavelet=str(cfg.get("cwt_wavelet", "morl")),
                    num_scales=int(cfg.get("cwt_num_scales", 64)),
                    fmin=float(cfg.get("cwt_fmin", 200.0)),
                    fmax=float(cfg.get("cwt_fmax", 20000.0)),
                )
            )
        if families.get("timefrequency", True):
            feats.update(
                compute_timefrequency_features(
                    y64,
                    sr_used,
                    nperseg=int(cfg.get("nperseg", 8192)),
                    hop_length=int(cfg.get("hop_length", 2048)),
                    stft_window=str(cfg.get("stft_window", DEFAULT_STFT_WINDOW)),
                )
            )

        stem = path.stem
        meta: dict[str, Any] = {
            "modality": "airborne",
            "record_name": stem,
            "recording_root": extract_recording_root(stem),
            "depth_mm": try_parse_depth_mm(stem),
            "step_idx": try_parse_step_idx(stem),
            "duration_s": float(len(y) / sr_used),
            "sr_hz": int(sr_used),
            "sr_hz_native": int(sr_native),
            "sr_hz_used": int(sr_used),
            "ds_rate": float(sr_native) / float(sr_used),
            "file_path": str(path),
        }
        return {**meta, **feats}

    except Exception:
        logger.warning("Failed to extract features from %s:\n%s", path, traceback.format_exc())
        return None


def extract_airborne(
    segments_dir: str | Path,
    cfg: dict[str, Any],
    out_csv: str | Path | None = None,
    file_glob: str | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    resolved_glob = str(file_glob or cfg.get("file_glob", DEFAULT_FILE_GLOB))
    resolved_workers = int(
        n_workers if n_workers is not None else cfg.get("n_workers", DEFAULT_N_WORKERS)
    )

    segments_dir = Path(segments_dir)
    paths = sorted(segments_dir.glob(resolved_glob))
    if not paths:
        raise FileNotFoundError(f"No files matching {resolved_glob!r} under {segments_dir}")

    logger.info("Airborne extraction: %d files (workers=%d)", len(paths), resolved_workers)

    rows: list[dict[str, Any]] = []

    if resolved_workers <= 1:
        for p in tqdm(paths, desc="airborne features"):
            row = extract_one_file(p, cfg)
            if row is not None:
                rows.append(row)
    else:
        try:
            with ProcessPoolExecutor(max_workers=resolved_workers) as exe:
                futures = {exe.submit(extract_one_file, p, cfg): p for p in paths}
                for fut in tqdm(
                    as_completed(futures), total=len(futures), desc="airborne features"
                ):
                    row = fut.result()
                    if row is not None:
                        rows.append(row)
        except Exception as exc:
            logger.warning(
                "Multiprocessing unavailable for airborne extraction (%s). "
                "Falling back to single-worker mode.",
                exc,
            )
            rows = []
            for p in tqdm(paths, desc="airborne features [fallback]"):
                row = extract_one_file(p, cfg)
                if row is not None:
                    rows.append(row)

    if not rows:
        raise RuntimeError("Feature extraction produced no rows. Check input files and config.")

    df = pd.DataFrame(rows)
    df = df.sort_values("record_name").reset_index(drop=True)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.info("Saved airborne features to %s  (%d rows  %d cols)", out_csv, *df.shape)

    return df
