"""vm_micro.features.airborne
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Airborne acoustic feature extractor.

Reads segmented FLAC files, computes all feature families configured in
``configs/airborne.yaml``, and assembles a tidy row per segment.

CLI entry point: ``vm-extract-air``  (see scripts/extract_airborne.py).
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from ..data.manifest import parse_depth_mm, try_parse_depth_mm, try_parse_step_idx, extract_recording_root
from ..utils import get_logger
from .core import (
    compute_band_power_features,
    compute_cwt_features,
    compute_dwt_features,
    compute_frequency_features,
    compute_machining_features,
    compute_short_time_features,
    compute_statistical_features,
    compute_time_features,
)

logger = get_logger(__name__)

_DEFAULT_BANDS = [
    (100.0,   1000.0),
    (1000.0,  5000.0),
    (5000.0,  20000.0),
    (20000.0, 96000.0),
]


def extract_one_file(
    path: Path,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract all features for a single segmented FLAC file.

    Returns a flat dict (one row for the feature DataFrame), or ``None`` on error.
    """
    try:
        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        y = audio.astype(np.float32)

        skip_s = float(cfg.get("skip_start_s", 0.0))
        end_s  = float(cfg.get("skip_end_s",   0.0))
        if skip_s > 0:
            y = y[int(round(skip_s * sr)):]
        if end_s > 0:
            trim = int(round(end_s * sr))
            if trim > 0:
                y = y[:-trim]

        if len(y) < 16:
            return None

        y64 = y.astype(np.float64)
        families: dict[str, bool] = cfg.get("feature_families", {})
        feats: dict[str, float] = {}

        if families.get("time",         True): feats.update(compute_time_features(y64, sr))
        if families.get("frequency",    True): feats.update(compute_frequency_features(y64, sr))
        if families.get("band_power",   True):
            bands = [tuple(b) for b in cfg.get("band_power_bands", _DEFAULT_BANDS)]
            feats.update(compute_band_power_features(y64, sr, bands=bands))
        if families.get("machining",    True):
            feats.update(compute_machining_features(
                y64, sr,
                env_sr=int(cfg.get("machining_env_sr",   24000)),
                hf_sr =int(cfg.get("machining_hf_sr",    64000)),
                hf_proxy=str(cfg.get("machining_hf_proxy", "roughness")),
            ))
        if families.get("statistical",  True): feats.update(compute_statistical_features(y64, sr))
        if families.get("short_time",   True):
            feats.update(compute_short_time_features(
                y64, sr,
                frame_ms=float(cfg.get("short_time_frame_ms", 10.0)),
                hop_ms  =float(cfg.get("short_time_hop_ms",    5.0)),
            ))
        if families.get("dwt",          True):
            feats.update(compute_dwt_features(
                y64, sr,
                wavelet  =str(cfg.get("dwt_wavelet",    "db8")),
                max_level=int(cfg.get("dwt_max_level",   8)),
            ))
        if families.get("cwt",          True):
            feats.update(compute_cwt_features(
                y64, sr,
                wavelet   =str(cfg.get("cwt_wavelet",    "morl")),
                num_scales=int(cfg.get("cwt_num_scales",  64)),
                fmin      =float(cfg.get("cwt_fmin",     200.0)),
                fmax      =float(cfg.get("cwt_fmax",   20000.0)),
            ))

        stem = path.stem
        meta: dict[str, Any] = {
            "modality":        "airborne",
            "record_name":     stem,
            "recording_root":  extract_recording_root(stem),
            "depth_mm":        try_parse_depth_mm(stem),
            "step_idx":        try_parse_step_idx(stem),
            "duration_s":      float(len(y) / sr),
            "sr_hz":           int(sr),
            "file_path":       str(path),
        }
        return {**meta, **feats}

    except Exception:
        logger.warning("Failed to extract features from %s:\n%s", path, traceback.format_exc())
        return None


def extract_airborne(
    segments_dir: str | Path,
    cfg: dict[str, Any],
    out_csv: str | Path | None = None,
    file_glob: str = "**/*.flac",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Extract airborne features from all segments under *segments_dir*.

    Parameters
    ----------
    segments_dir : Path to the root of segmented audio files.
    cfg          : Loaded airborne config dict.
    out_csv      : Optional path to save the feature DataFrame as CSV.
    file_glob    : Glob pattern relative to *segments_dir*.
    n_workers    : Number of parallel worker processes.

    Returns
    -------
    pd.DataFrame with one row per segment, all features + metadata columns.
    """
    segments_dir = Path(segments_dir)
    paths = sorted(segments_dir.glob(file_glob))
    if not paths:
        raise FileNotFoundError(f"No files matching {file_glob!r} under {segments_dir}")

    logger.info("Airborne extraction: %d files (workers=%d)", len(paths), n_workers)

    rows: list[dict[str, Any]] = []

    if n_workers <= 1:
        for p in tqdm(paths, desc="airborne features"):
            row = extract_one_file(p, cfg)
            if row is not None:
                rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {exe.submit(extract_one_file, p, cfg): p for p in paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="airborne features"):
                row = fut.result()
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
        logger.info("Saved airborne features to %s  (%d rows × %d cols)", out_csv, *df.shape)

    return df
