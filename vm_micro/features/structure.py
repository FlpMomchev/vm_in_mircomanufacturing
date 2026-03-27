"""
V1/default structure feature extractor.
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
    compute_geometry_features,
    compute_short_time_features,
    compute_statistical_features,
    compute_time_features,
    compute_timefrequency_features,
)

logger = get_logger(__name__)
DEFAULT_FILE_GLOB = "**/*.h5"
DEFAULT_N_WORKERS = 6
DEFAULT_DS_RATE = 62.5
DEFAULT_NPERSEG = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_NATIVE_SR = 3_125_000


def extract_one_file(path: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    try:
        h5_data_key = str(cfg.get("h5_data_key", "measurement/data"))
        h5_time_key = str(cfg.get("h5_time_key", "measurement/time_vector"))
        ds_rate = float(cfg.get("ds_rate", DEFAULT_DS_RATE))
        if ds_rate <= 0:
            raise ValueError(f"Invalid ds_rate={ds_rate}. Must be > 0.")

        y_raw, sr_native, _tv, _meta = read_measurement_h5(
            path, data_key=h5_data_key, time_key=h5_time_key, center_signal=True
        )

        # Cascade decimation  single-step with factor
        y = y_raw.astype(np.float64)
        sr = int(sr_native)
        target_sr = max(1, int(round(float(sr_native) / ds_rate)))
        while sr // 10 >= 2 * target_sr:
            y = decimate(y, 10, ftype="iir", zero_phase=True)
            sr = sr // 10
        remaining = sr // target_sr
        if remaining > 1:
            y = decimate(y, remaining, ftype="iir", zero_phase=True)
            sr = sr // remaining

        if len(y) < 16:
            return None

        families: dict[str, bool] = cfg.get("feature_families", {})
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
                    y, sr, frame_ms=float(cfg.get("short_time_frame_ms", 10.0))
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
        if families.get("timefrequency", True):
            feats.update(
                compute_timefrequency_features(
                    y,
                    sr,
                    nperseg=int(cfg.get("nperseg", DEFAULT_NPERSEG)),
                    hop_length=int(cfg.get("hop_length", DEFAULT_HOP_LENGTH)),
                )
            )

        stem = path.stem
        meta: dict[str, Any] = {
            "modality": "structure",
            "record_name": stem,
            "recording_root": extract_recording_root(stem),
            "depth_mm": try_parse_depth_mm(stem),
            "step_idx": try_parse_step_idx(stem),
            "duration_s": float(len(y) / sr),
            "sr_hz_native": int(cfg.get("native_sr", DEFAULT_NATIVE_SR)),
            "sr_hz_used": int(sr),
            "ds_rate": float(ds_rate),
            "file_path": str(path),
        }
        return {**meta, **feats}

    except Exception:
        logger.warning("Failed to extract features from %s:\n%s", path, traceback.format_exc())
        return None


def _extract_one_file_extensive(path: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
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

        # Geometry features (mandatory  not gated by feature_families)
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
            )
        )

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

    version = str(cfg.get("extractor", "v1")).lower()
    extract_fn = _extract_one_file_extensive if version == "extensive" else extract_one_file
    label = f"structure features ({version})"

    logger.info(
        "Structure-borne extraction [%s]: %d files (workers=%d)",
        version,
        len(paths),
        resolved_workers,
    )

    rows: list[dict[str, Any]] = []

    if resolved_workers <= 1:
        for p in tqdm(paths, desc=label):
            row = extract_fn(p, cfg)
            if row is not None:
                rows.append(row)
    else:
        try:
            with ProcessPoolExecutor(max_workers=resolved_workers) as exe:
                futures = {exe.submit(extract_fn, p, cfg): p for p in paths}
                for fut in tqdm(as_completed(futures), total=len(futures), desc=label):
                    row = fut.result()
                    if row is not None:
                        rows.append(row)
        except Exception as exc:
            logger.warning(
                "Multiprocessing unavailable for structure extraction (%s). "
                "Falling back to single-worker mode.",
                exc,
            )
            rows = []
            for p in tqdm(paths, desc=f"{label} [fallback]"):
                row = extract_fn(p, cfg)
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
            "Saved structure-borne features to %s  (%d rows  %d cols)",
            out_csv,
            *df.shape,
        )

    return df
