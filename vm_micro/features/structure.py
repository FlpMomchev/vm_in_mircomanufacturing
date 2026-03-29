"""
Structure-borne feature extraction (v1 + v2/extensive).
"""

from __future__ import annotations

import math
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import decimate, resample_poly
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
    compute_machining_features,
    compute_short_time_features,
    compute_statistical_features,
    compute_time_features,
    compute_timefrequency_features,
)

logger = get_logger(__name__)
DEFAULT_FILE_GLOB = "**/*.h5"
DEFAULT_N_WORKERS = 6
DEFAULT_DS_RATE_V1 = 62.5
DEFAULT_DS_RATE_V2 = 64.0
DEFAULT_NPERSEG = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_STFT_WINDOW = "hann"
DEFAULT_DWT_PADDING_MODE = "symmetric"
DEFAULT_STRUCTURE_FAMILIES: dict[str, bool] = {
    "time": True,
    "frequency": True,
    "band_power": True,
    "machining": False,
    "statistical": True,
    "short_time": True,
    "dwt": True,
    "cwt": True,
    "timefrequency": True,
}
_EXTRACTOR_ALIASES = {
    "v1": "v1",
    "v2": "v2",
    "extensive": "v2",
}


def _normalise_extractor_version(raw: Any) -> str:
    key = str(raw if raw is not None else "v1").strip().lower()
    return _EXTRACTOR_ALIASES.get(key, "v1")


def _resolve_ds_rate(cfg: dict[str, Any], *, version: str) -> float:
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported extractor version {version!r}.")

    fallback = DEFAULT_DS_RATE_V1 if version == "v1" else DEFAULT_DS_RATE_V2
    value: Any = None

    # New config style (preferred)
    specific_key = f"ds_rate_{version}"
    if specific_key in cfg:
        value = cfg.get(specific_key)

    # Optional map-style config: ds_rate: {v1: ..., v2: ...}
    if value is None and isinstance(cfg.get("ds_rate"), dict):
        ds_map = cfg["ds_rate"]
        value = ds_map.get(version)
        if value is None and version == "v2":
            # Backward-compatible alias key
            value = ds_map.get("extensive")

    # Legacy scalar config: ds_rate: 62.5 (v1 only).
    # v2 used ext_ds_stages historically, so keep scalar ds_rate from affecting v2.
    if value is None and version == "v1" and not isinstance(cfg.get("ds_rate"), dict):
        value = cfg.get("ds_rate")

    if value is None:
        value = fallback

    ds_rate = float(value)
    if not math.isfinite(ds_rate) or ds_rate <= 0.0:
        raise ValueError(f"Invalid ds_rate for {version}: {ds_rate}. Must be finite and > 0.")

    return ds_rate


def _factorise_decimation(total_factor: int) -> list[int]:
    if total_factor < 1:
        raise ValueError(f"total_factor must be >= 1, got {total_factor}.")
    if total_factor == 1:
        return []

    remainder = int(total_factor)
    stages: list[int] = []
    for candidate in (10, 9, 8, 7, 6, 5, 4, 3, 2):
        while remainder % candidate == 0 and remainder > 1:
            stages.append(candidate)
            remainder //= candidate

    if remainder > 1:
        stages.append(remainder)
    return stages


def _safe_resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float64, copy=False)
    factor = math.gcd(int(sr_in), int(sr_out))
    return resample_poly(
        y.astype(np.float64, copy=False),
        up=int(sr_out) // factor,
        down=int(sr_in) // factor,
    ).astype(np.float64, copy=False)


def _downsample_to_target_sr(
    y_raw: np.ndarray,
    *,
    sr_native: int,
    target_sr_hz: int,
) -> tuple[np.ndarray, int]:
    if target_sr_hz <= 0:
        raise ValueError(f"target_sr_hz must be > 0, got {target_sr_hz}.")

    y = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    sr = int(sr_native)
    target = int(target_sr_hz)

    # Keep anti-aliasing stable by reducing huge native SR in coarse x10 stages first.
    while sr // 10 >= 2 * target:
        y = decimate(y, 10, ftype="iir", zero_phase=True)
        sr = sr // 10

    if sr != target:
        y = _safe_resample(y, sr, target)
        sr = target

    return y, sr


def _resolve_v2_ds_stages(cfg: dict[str, Any]) -> list[int]:
    if "ext_ds_stages" in cfg and cfg["ext_ds_stages"] is not None:
        stages_raw = cfg["ext_ds_stages"]
        if not isinstance(stages_raw, (list, tuple)) or len(stages_raw) == 0:
            raise ValueError("ext_ds_stages must be a non-empty list of integers.")
        stages = [int(x) for x in stages_raw]
        if any(s < 1 for s in stages):
            raise ValueError(f"Invalid ext_ds_stages={stages}. Each stage must be >= 1.")
        return stages

    ds_rate_v2 = _resolve_ds_rate(cfg, version="v2")
    ds_rate_v2_int = int(round(ds_rate_v2))
    if not math.isclose(ds_rate_v2, float(ds_rate_v2_int), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"ds_rate_v2 must be an integer-compatible factor, got {ds_rate_v2}. "
            "Use ext_ds_stages for custom non-standard cascades."
        )
    return _factorise_decimation(ds_rate_v2_int)


def _resolve_feature_families(cfg: dict[str, Any]) -> dict[str, bool]:
    raw = cfg.get("feature_families", {})
    if not isinstance(raw, dict):
        raw = {}
    out: dict[str, bool] = {}
    for key, default in DEFAULT_STRUCTURE_FAMILIES.items():
        out[key] = bool(raw.get(key, default))
    return out


def resolve_effective_structure_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return effective structure extraction settings for provenance/replay."""
    version = _normalise_extractor_version(cfg.get("extractor", "v1"))
    effective: dict[str, Any] = {
        "extractor": version,
        "h5_data_key": str(cfg.get("h5_data_key", "measurement/data")),
        "h5_time_key": str(cfg.get("h5_time_key", "measurement/time_vector")),
        "file_glob": str(cfg.get("file_glob", DEFAULT_FILE_GLOB)),
        "n_workers": int(cfg.get("n_workers", DEFAULT_N_WORKERS)),
    }
    if cfg.get("target_sr_hz", None) is not None:
        effective["target_sr_hz"] = int(round(float(cfg["target_sr_hz"])))

    if version == "v1":
        effective.update(
            {
                "ds_rate_v1": float(_resolve_ds_rate(cfg, version="v1")),
                "stft_window": str(cfg.get("stft_window", DEFAULT_STFT_WINDOW)),
                "nperseg": int(cfg.get("nperseg", DEFAULT_NPERSEG)),
                "hop_length": int(cfg.get("hop_length", DEFAULT_HOP_LENGTH)),
                "short_time_frame_ms": float(cfg.get("short_time_frame_ms", 10.0)),
                "cwt_wavelet": str(cfg.get("cwt_wavelet", "morl")),
                "cwt_num_scales": int(cfg.get("cwt_num_scales", 64)),
                "cwt_fmin": float(cfg.get("cwt_fmin", 50.0)),
                "cwt_fmax": float(cfg.get("cwt_fmax", 1562.0)),
                "dwt_wavelet": str(cfg.get("dwt_wavelet", "sym8")),
                "dwt_level": int(cfg.get("dwt_level", 5)),
                "dwt_padding_mode": str(cfg.get("dwt_padding_mode", DEFAULT_DWT_PADDING_MODE)),
                "feature_families": _resolve_feature_families(cfg),
                "machining_env_sr": int(cfg.get("machining_env_sr", 24_000)),
                "machining_hf_sr": int(cfg.get("machining_hf_sr", 64_000)),
                "machining_hf_proxy": str(cfg.get("machining_hf_proxy", "roughness")),
            }
        )
    else:
        effective.update(
            {
                "ds_rate_v2": float(_resolve_ds_rate(cfg, version="v2")),
                "ext_ds_stages": _resolve_v2_ds_stages(cfg),
            }
        )
        ext_map: dict[str, tuple[str, type]] = {
            "ext_window_s": ("ext_window_s", float),
            "ext_hop_s": ("ext_hop_s", float),
            "ext_wpd_wavelet": ("ext_wpd_wavelet", str),
            "ext_wpd_level": ("ext_wpd_level", int),
            "ext_n_mfcc": ("ext_n_mfcc", int),
            "ext_n_filters": ("ext_n_filters", int),
            "ext_f_min_cepstral": ("ext_f_min_cepstral", float),
            "ext_cwt_wavelet": ("ext_cwt_wavelet", str),
            "ext_cwt_n_scales": ("ext_cwt_n_scales", int),
            "ext_cwt_f_min": ("ext_cwt_f_min", float),
            "ext_complexity_n_samples": ("ext_complexity_n_samples", int),
            "ext_sampen_m": ("ext_sampen_m", int),
            "ext_sampen_r_factor": ("ext_sampen_r_factor", float),
            "ext_permen_order": ("ext_permen_order", int),
            "ext_permen_delay": ("ext_permen_delay", int),
        }
        for cfg_key, (out_key, caster) in ext_map.items():
            if cfg_key in cfg and cfg[cfg_key] is not None:
                effective[out_key] = caster(cfg[cfg_key])
        if "ext_bands" in cfg and cfg["ext_bands"] is not None:
            raw_bands = cfg["ext_bands"]
            if isinstance(raw_bands, (list, tuple)):
                bands: list[list[float]] = []
                for item in raw_bands:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        bands.append([float(item[0]), float(item[1])])
                if bands:
                    effective["ext_bands"] = bands

    return effective


def extract_one_file(path: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    try:
        h5_data_key = str(cfg.get("h5_data_key", "measurement/data"))
        h5_time_key = str(cfg.get("h5_time_key", "measurement/time_vector"))

        y_raw, sr_native, _tv, _meta = read_measurement_h5(
            path, data_key=h5_data_key, time_key=h5_time_key, center_signal=True
        )

        target_sr_raw = cfg.get("target_sr_hz", None)
        if target_sr_raw is not None:
            target_sr = int(round(float(target_sr_raw)))
            y, sr = _downsample_to_target_sr(
                y_raw,
                sr_native=int(sr_native),
                target_sr_hz=target_sr,
            )
            ds_rate = float(sr_native) / float(sr)
        else:
            ds_rate = _resolve_ds_rate(cfg, version="v1")
            # Backward-compatible v1 path: decimation-only strategy.
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
            feats.update(compute_time_features(y, sr))
        if families.get("frequency", True):
            feats.update(compute_frequency_features(y, sr))
        if families.get("band_power", True):
            nyq = sr / 2.0
            bands = [(10.0, 100.0), (100.0, 500.0), (500.0, min(nyq - 1, 1500.0))]
            feats.update(compute_band_power_features(y, sr, bands=bands))
        if families.get("machining", False):
            feats.update(
                compute_machining_features(
                    y,
                    sr,
                    env_sr=int(cfg.get("machining_env_sr", 24_000)),
                    hf_sr=int(cfg.get("machining_hf_sr", 64_000)),
                    hf_proxy=str(cfg.get("machining_hf_proxy", "roughness")),
                )
            )
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
                    padding_mode=str(cfg.get("dwt_padding_mode", DEFAULT_DWT_PADDING_MODE)),
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
                    stft_window=str(cfg.get("stft_window", DEFAULT_STFT_WINDOW)),
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
            "sr_hz_native": int(sr_native),
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
        target_sr_raw = cfg.get("target_sr_hz", None)

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
        ext_key_map: dict[str, tuple[str, type]] = {
            "ext_window_s": ("window_s", float),
            "ext_hop_s": ("hop_s", float),
            "ext_wpd_wavelet": ("wpd_wavelet", str),
            "ext_wpd_level": ("wpd_level", int),
            "ext_n_mfcc": ("n_mfcc", int),
            "ext_n_filters": ("n_filters", int),
            "ext_f_min_cepstral": ("f_min_cepstral", float),
            "ext_cwt_wavelet": ("cwt_wavelet", str),
            "ext_cwt_n_scales": ("cwt_n_scales", int),
            "ext_cwt_f_min": ("cwt_f_min", float),
            "ext_complexity_n_samples": ("complexity_n_samples", int),
            "ext_sampen_m": ("sampen_m", int),
            "ext_sampen_r_factor": ("sampen_r_factor", float),
            "ext_permen_order": ("permen_order", int),
            "ext_permen_delay": ("permen_delay", int),
        }
        for cfg_key, (ext_key, caster) in ext_key_map.items():
            if cfg_key in cfg and cfg[cfg_key] is not None:
                ext_kwargs[ext_key] = caster(cfg[cfg_key])
        if "ext_bands" in cfg and cfg["ext_bands"] is not None:
            raw_bands = cfg["ext_bands"]
            if isinstance(raw_bands, (list, tuple)):
                bands: list[tuple[float, float]] = []
                for item in raw_bands:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        bands.append((float(item[0]), float(item[1])))
                if bands:
                    ext_kwargs["bands"] = bands

        y_for_ext = y_raw.astype(np.float64, copy=False)

        if target_sr_raw is not None:
            target_sr = int(round(float(target_sr_raw)))
            if target_sr <= 0:
                raise ValueError(f"target_sr_hz must be > 0, got {target_sr}.")
            # Keep v2 runtime aligned with training behavior by preferring
            # integer cascade decimation (same filter family as training),
            # with an adaptive total factor inferred from native SR.
            adaptive_factor = int(round(float(sr_native) / float(target_sr)))
            adaptive_factor = max(1, adaptive_factor)
            ds_stages = _factorise_decimation(adaptive_factor)
            ext = StructureBorneFeatureExtractorExtensive(
                fs_native=sr_native,
                ds_stages=ds_stages,
                **ext_kwargs,
            )
            ds_rate_meta: float | int = float(sr_native) / float(ext.sr_hz_used)
        else:
            ds_stages = _resolve_v2_ds_stages(cfg)
            ext = StructureBorneFeatureExtractorExtensive(
                fs_native=sr_native,
                ds_stages=ds_stages,
                **ext_kwargs,
            )
            ds_rate_meta = int(ext.ds_rate)

        feats = ext.extract(y_for_ext)

        # Geometry features (mandatory, not gated by feature_families)
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
            "ds_rate": ds_rate_meta,
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

    configured_version = str(cfg.get("extractor", "v1")).strip().lower()
    version = _normalise_extractor_version(configured_version)
    if configured_version not in _EXTRACTOR_ALIASES:
        logger.warning("Unknown extractor '%s'; falling back to 'v1'.", configured_version)

    extract_fn = _extract_one_file_extensive if version == "v2" else extract_one_file
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
