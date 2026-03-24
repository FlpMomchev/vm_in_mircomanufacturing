"""vm_micro.features.structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Structure-borne acoustic feature extractor.

Reads segmented HDF5 files (produced by the splitter from raw structure-borne
measurements), applies decimation to reduce the very-high native sample rate
(3.125 MHz → ~3 kHz), then computes the same feature families as the airborne
extractor — minus the airborne-specific machining proxies.

Supports two extractor versions selectable via ``extractor`` in config:
- ``"v1"`` (default): core.py functions on single-step decimated signal.
- ``"extensive"``: StructureBorneFeatureExtractorV2 with fixed-window analysis,
  cascade decimation to ~48.8 kHz, WPD, cepstral, and normalised features.

Key differences from airborne:
- Input format: HDF5 (``measurement/data`` + ``measurement/time_vector``)
- Native SR: 3_125_000 Hz; decimated with ``ds_rate`` before feature extraction
- No machining-proxy features (``families.machining = false`` in config)
- CWT frequency range is adapted to the lower post-decimation Nyquist

CLI entry point: ``vm-extract-struct``  (see scripts/extract_structure.py).
"""

from __future__ import annotations

import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import decimate
from tqdm.auto import tqdm

from ..data.io import read_measurement_h5
from ..data.manifest import extract_recording_root, try_parse_depth_mm, try_parse_step_idx
from ..utils import get_logger
from .core import (
    compute_band_power_features,
    compute_cwt_features,
    compute_dwt_features,
    compute_frequency_features,
    compute_short_time_features,
    compute_statistical_features,
    compute_time_features,
)

logger = get_logger(__name__)


def extract_one_file(
    path: Path,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract features from a single segmented HDF5 file.

    Returns a flat dict (one feature row), or ``None`` on error.
    """
    try:
        h5_data_key = str(cfg.get("h5_data_key", "measurement/data"))
        h5_time_key = str(cfg.get("h5_time_key", "measurement/time_vector"))
        ds_rate = int(cfg.get("ds_rate", 1000))

        y_raw, sr_native, _tv, _meta = read_measurement_h5(
            path,
            data_key=h5_data_key,
            time_key=h5_time_key,
            center_signal=True,
        )

        # Decimate to working sample rate
        # scipy.signal.decimate applies an anti-alias filter before downsampling
        y = decimate(y_raw.astype(np.float64), ds_rate, ftype="iir", zero_phase=True)
        sr = max(1, sr_native // ds_rate)

        if len(y) < 16:
            return None

        families: dict[str, bool] = cfg.get("feature_families", {})
        feats: dict[str, float] = {}

        if families.get("time", True):
            feats.update(compute_time_features(y, sr))
        if families.get("frequency", True):
            feats.update(compute_frequency_features(y, sr))
        if families.get("band_power", True):
            nyq = sr / 2.0
            bands = [(10.0, 100.0), (100.0, 500.0), (500.0, min(nyq - 1, 1500.0))]
            feats.update(compute_band_power_features(y, sr, bands=bands))
        if families.get("statistical", True):
            feats.update(compute_statistical_features(y, sr))
        if families.get("short_time", True):
            feats.update(
                compute_short_time_features(
                    y,
                    sr,
                    frame_ms=float(cfg.get("short_time_frame_ms", 10.0)),
                    hop_ms=float(cfg.get("short_time_hop_ms", 5.0)),
                )
            )
        if families.get("dwt", True):
            feats.update(
                compute_dwt_features(
                    y,
                    sr,
                    wavelet=str(cfg.get("dwt_wavelet", "sym8")),
                    max_level=int(cfg.get("dwt_level", 5)),
                )
            )
        if families.get("cwt", True):
            feats.update(
                compute_cwt_features(
                    y,
                    sr,
                    wavelet=str(cfg.get("cwt_wavelet", "morl")),
                    num_scales=int(cfg.get("cwt_num_scales", 64)),
                    fmin=float(cfg.get("cwt_fmin", 50.0)),
                    fmax=float(cfg.get("cwt_fmax", 1562.0)),
                )
            )
        # machining proxy intentionally omitted for structure-borne

        stem = path.stem
        meta: dict[str, Any] = {
            "modality": "structure",
            "record_name": stem,
            "recording_root": extract_recording_root(stem),
            "depth_mm": try_parse_depth_mm(stem),
            "step_idx": try_parse_step_idx(stem),
            "duration_s": float(len(y) / sr),
            "sr_hz_native": int(cfg.get("native_sr", 3_125_000)),
            "sr_hz_used": int(sr),
            "ds_rate": int(ds_rate),
            "file_path": str(path),
        }
        return {**meta, **feats}

    except Exception:
        logger.warning("Failed to extract features from %s:\n%s", path, traceback.format_exc())
        return None


def _extract_one_file_extensive(
    path: Path,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract features using StructureBorneFeatureExtractorV2.

    Reads the HDF5 the same way as the standard extractor, but passes
    the *raw* signal (before decimation) to the extensive extractor,
    which handles its own cascade decimation and windowed feature
    extraction internally.
    """
    try:
        from .structure_extensive import StructureBorneFeatureExtractorExtensive

        h5_data_key = str(cfg.get("h5_data_key", "measurement/data"))
        h5_time_key = str(cfg.get("h5_time_key", "measurement/time_vector"))

        y_raw, sr_native, _tv, _meta = read_measurement_h5(
            path,
            data_key=h5_data_key,
            time_key=h5_time_key,
            center_signal=True,
        )

        if len(y_raw) < 64:
            return None

        # Build extensive extractor with config overrides
        ext_kwargs: dict[str, Any] = {}
        if "ext_ds_stages" in cfg:
            ext_kwargs["ds_stages"] = list(cfg["ext_ds_stages"])
        if "ext_window_s" in cfg:
            ext_kwargs["window_s"] = float(cfg["ext_window_s"])
        if "ext_hop_s" in cfg:
            ext_kwargs["hop_s"] = float(cfg["ext_hop_s"])
        if "ext_wpd_level" in cfg:
            ext_kwargs["wpd_level"] = int(cfg["ext_wpd_level"])
        if "ext_cwt_n_scales" in cfg:
            ext_kwargs["cwt_n_scales"] = int(cfg["ext_cwt_n_scales"])

        ext = StructureBorneFeatureExtractorExtensive(fs_native=sr_native, **ext_kwargs)
        feats = ext.extract(y_raw.astype(np.float64))

        stem = path.stem
        meta: dict[str, Any] = {
            "modality": "structure",
            "record_name": stem,
            "recording_root": extract_recording_root(stem),
            "depth_mm": try_parse_depth_mm(stem),
            "step_idx": try_parse_step_idx(stem),
            "duration_s": float(len(y_raw) / sr_native),
            "sr_hz_native": int(sr_native),
            "sr_hz_used": int(ext.sr_hz_used),
            "ds_rate": int(ext.ds_rate),
            "file_path": str(path),
        }
        return {**meta, **dict(feats)}

    except Exception:
        logger.warning("Extensive extraction failed for %s:\n%s", path, traceback.format_exc())
        return None


def extract_structure(
    segments_dir: str | Path,
    cfg: dict[str, Any],
    out_csv: str | Path | None = None,
    file_glob: str = "**/*.h5",
    n_workers: int = 4,
) -> pd.DataFrame:
    """Extract structure-borne features from all segments under *segments_dir*.

    Parameters
    ----------
    segments_dir : Path to root of segmented HDF5 files.
    cfg          : Loaded structure config dict.
    out_csv      : Optional path to save the feature DataFrame as CSV.
    file_glob    : Glob pattern relative to *segments_dir*.
    n_workers    : Number of parallel worker processes.

    Returns
    -------
    pd.DataFrame with one row per segment.
    """
    segments_dir = Path(segments_dir)
    paths = sorted(segments_dir.glob(file_glob))
    if not paths:
        raise FileNotFoundError(f"No files matching {file_glob!r} under {segments_dir}")

    version = str(cfg.get("extractor", "v1")).lower()
    extract_fn = _extract_one_file_extensive if version == "extensive" else extract_one_file
    label = f"structure features ({version})"

    logger.info(
        "Structure-borne extraction [%s]: %d files (workers=%d)",
        version,
        len(paths),
        n_workers,
    )

    rows: list[dict[str, Any]] = []

    if n_workers <= 1:
        for p in tqdm(paths, desc=label):
            row = extract_fn(p, cfg)
            if row is not None:
                rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futures = {exe.submit(extract_fn, p, cfg): p for p in paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=label):
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
        logger.info(
            "Saved structure-borne features to %s  (%d rows × %d cols)",
            out_csv,
            *df.shape,
        )

    return df
