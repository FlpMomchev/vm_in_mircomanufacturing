"""vm-predict-fused - One-shot fused inference from new raw recordings.

This script is designed as the first backend-friendly entrypoint:
- watches a raw input folder (scan on demand)
- splits whole recordings into segments
- runs per-modality inference (classical + DL)
- fuses predictions (intra + inter)
- writes one tidy output folder:
  - modality folders with CSV artifacts (+ segment manifests)
  - `final/` with batch-level JSON reports and final fused CSV

Config-first: see ``configs/fusion.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd
import soundfile as sf

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.classical.inference import infer_classical
from vm_micro.data.manifest import load_doe, try_parse_depth_mm
from vm_micro.data.splitter import process_one_file
from vm_micro.features.airborne import (
    DEFAULT_FILE_GLOB as AIRBORNE_DEFAULT_FILE_GLOB,
)
from vm_micro.features.airborne import (
    DEFAULT_N_WORKERS as AIRBORNE_DEFAULT_N_WORKERS,
)
from vm_micro.features.airborne import (
    extract_airborne,
)
from vm_micro.features.structure import (
    DEFAULT_FILE_GLOB as STRUCTURE_DEFAULT_FILE_GLOB,
)
from vm_micro.features.structure import (
    DEFAULT_N_WORKERS as STRUCTURE_DEFAULT_N_WORKERS,
)
from vm_micro.features.structure import (
    extract_structure,
)
from vm_micro.fusion.fuser import (
    PredictionBundle,
    bundle_batch_metrics,
    fuse_intra_modality,
    fuse_modalities,
)
from vm_micro.utils import apply_overrides, get_logger, load_config

logger = get_logger(__name__)


@dataclass
class RawFile:
    path: Path
    stem: str
    mtime_ns: int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-predict-fused",
        description="Run fused prediction on new raw recordings (split->infer->fuse).",
    )
    p.add_argument("--config", default="configs/fusion.yaml")
    p.add_argument("--out-dir", default=None, help="Override run.out_dir for this execution.")
    p.add_argument(
        "--only",
        default="both",
        choices=["both", "airborne", "structure"],
        help="Limit processing to one modality.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-process files even if present in the state file.",
    )
    p.add_argument(
        "override", nargs="*", help="YAML config overrides, e.g. --models.airborne.dl.device=cpu"
    )
    return p


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "processed": {}}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def _scan_raw(raw_dir: Path, file_glob: str) -> list[RawFile]:
    paths = sorted(raw_dir.glob(file_glob))
    out: list[RawFile] = []
    for p in paths:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        out.append(RawFile(path=p, stem=p.stem, mtime_ns=int(st.st_mtime_ns)))
    return out


def _dummy_doe(expected_segments: int) -> pd.DataFrame:
    # Use Step 1..N so segment stems always include a step index.
    return pd.DataFrame(
        {
            "Step": list(range(1, expected_segments + 1)),
            "HoleID": ["NA"] * expected_segments,
            "Depth_mm": [None] * expected_segments,
        }
    )


def _expected_map(cfg: dict[str, Any]) -> tuple[dict[str, int], pd.DataFrame | None]:
    exp = cfg.get("splitting", {}).get("expected_segments", {})
    default = int(exp.get("default", 25))
    map_xlsx = exp.get("map_xlsx", None)
    doe_sheet = str(exp.get("doe_sheet", "DOE_run_order"))

    mapping: dict[str, int] = {}
    doe_df: pd.DataFrame | None = None

    if map_xlsx:
        p = Path(str(map_xlsx))
        if p.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
            logger.warning(
                "splitting.expected_segments.map_xlsx must be an Excel file. Got %s, ignoring.",
                map_xlsx,
            )
        else:
            try:
                doe_df = load_doe(map_xlsx, sheet_name=doe_sheet)
            except Exception as exc:
                logger.warning(
                    "Failed to load DOE workbook %s (sheet=%s): %s. "
                    "Falling back to configured default=%d.",
                    map_xlsx,
                    doe_sheet,
                    exc,
                    default,
                )
            else:
                doe_rows = int(len(doe_df))
                if doe_rows <= 0:
                    logger.warning(
                        "DOE workbook %s (sheet=%s) is empty. "
                        "Falling back to configured default=%d.",
                        map_xlsx,
                        doe_sheet,
                        default,
                    )
                else:
                    if default != doe_rows:
                        logger.info(
                            "Overriding splitting.expected_segments.default=%d with DOE row count=%d "
                            "from %s (sheet=%s).",
                            default,
                            doe_rows,
                            map_xlsx,
                            doe_sheet,
                        )
                    default = doe_rows

    mapping["_default"] = default
    return mapping, doe_df


def _resolve_expected(mapping: dict[str, int], stem: str) -> int:
    return int(mapping.get(stem, mapping.get("_default", 25)))


def _doe_for_file(doe_template: pd.DataFrame | None, expected_segments: int) -> pd.DataFrame:
    expected_segments = int(expected_segments)
    if doe_template is None:
        return _dummy_doe(expected_segments)

    doe_base = doe_template.reset_index(drop=True).copy()
    n_rows = len(doe_base)
    if expected_segments <= n_rows:
        return doe_base.iloc[:expected_segments].reset_index(drop=True).copy()

    extra_n = expected_segments - n_rows
    extras = pd.DataFrame(
        {
            "Step": list(range(n_rows + 1, expected_segments + 1)),
            "HoleID": ["NA"] * extra_n,
            "Depth_mm": [None] * extra_n,
        }
    )
    return pd.concat([doe_base, extras], ignore_index=True, sort=False)


def _ensure_feature_cols(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    out = df.copy()
    missing = [c for c in required if c not in out.columns]
    for c in missing:
        out[c] = np.nan
    return out


def _extract_for_roots(
    *,
    roots: set[str],
    segments_root: Path,
    extractor: Any,
    cfg: dict[str, Any],
    file_glob: str | None,
    n_workers: int | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in sorted(roots):
        root_dir = segments_root / root
        if not root_dir.exists():
            logger.warning(
                "Segments folder missing for root %s under %s; skipping.", root, segments_root
            )
            continue
        frames.append(
            extractor(root_dir, cfg, out_csv=None, file_glob=file_glob, n_workers=n_workers)
        )

    if not frames:
        raise FileNotFoundError(
            f"No segment folders found under {segments_root} for roots={sorted(roots)}"
        )

    return pd.concat(frames, ignore_index=True)


def _h5_file_info(path: Path, data_key: str, time_key: str) -> dict[str, Any]:
    with h5py.File(str(path), "r") as fh:
        n_samples = int(fh[data_key].shape[0])
        tv = np.asarray(fh[time_key][:], dtype=np.float64).reshape(-1)
    dt = np.diff(tv)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    dt_median = float(np.median(dt)) if len(dt) else 1.0
    sr = int(round(1.0 / dt_median))
    return {
        "sample_rate_native": sr,
        "duration_sec": float(n_samples / max(1, sr)),
        "frames_native": n_samples,
        "channels": 1,
    }


def _build_unlabeled_file_df(
    data_dir: Path,
    file_glob: str,
    *,
    h5_data_key: str,
    h5_time_key: str,
    include_recording_roots: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted(data_dir.glob(file_glob)):
        if p.is_dir():
            continue
        recording_root = p.stem.split("__seg")[0] if "__seg" in p.stem else p.stem
        if include_recording_roots is not None and recording_root not in include_recording_roots:
            continue
        if p.suffix.lower() in {".flac", ".wav"}:
            info = sf.info(str(p))
            meta = {
                "sample_rate_native": int(info.samplerate),
                "duration_sec": float(info.duration),
                "frames_native": int(info.frames),
                "channels": int(info.channels),
            }
        else:
            meta = _h5_file_info(p, h5_data_key, h5_time_key)

        rows.append(
            {
                "file_id": len(rows),
                "path": str(p.resolve()),
                "record_name": p.stem,
                "stem": p.stem,
                "file_name": p.name,
                # If depth is encoded in the filename (e.g. "__depth0.750"),
                # retain it so per-run MAE can be computed for fusion weights.
                "depth_mm": try_parse_depth_mm(p.stem),
                "class_idx": -1,
                "recording_root": recording_root,
                "parent_dir": p.parent.name,
                "file_group_id": p.stem,
                **meta,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No files found under {data_dir} matching {file_glob}")
    return pd.DataFrame(rows)


def _infer_dl_on_segments(
    *,
    model_dir: Path,
    segments_dir: Path,
    file_glob: str | None,
    device: str,
    batch_size: int | None,
    out_csv: Path,
    h5_data_key: str,
    h5_time_key: str,
    include_recording_roots: set[str] | None = None,
) -> pd.DataFrame:
    import torch

    from vm_micro.dl.config import TrainConfig
    from vm_micro.dl.data import WaveformWindowDataset
    from vm_micro.dl.engine import aggregate_file_predictions, make_loader, predict_loader
    from vm_micro.dl.models import DepthModel
    from vm_micro.dl.utils import attach_step_idx_if_possible, choose_device, read_label_mapping

    final_dir = model_dir / "final_model"
    if final_dir.exists() and (final_dir / "best_model.pt").exists():
        model_dir = final_dir
    elif not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"No best_model.pt found under {model_dir}")

    with open(model_dir / "config.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cfg = TrainConfig.from_json_dict(payload)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    effective_file_glob = str(file_glob or cfg.file_glob)
    cfg.data_dir = str(segments_dir)
    cfg.file_glob = effective_file_glob

    dev = choose_device(device if device != "auto" else cfg.device)
    logger.info("DL inference device: %s", dev)

    file_df = _build_unlabeled_file_df(
        segments_dir,
        effective_file_glob,
        h5_data_key=h5_data_key,
        h5_time_key=h5_time_key,
        include_recording_roots=include_recording_roots,
    )
    file_df = attach_step_idx_if_possible(file_df)

    label_map_path = model_dir / "label_mapping.json"
    class_to_depth = read_label_mapping(label_map_path) if label_map_path.exists() else None
    out_dim = (
        (max(class_to_depth.keys()) + 1) if (cfg.task == "classification" and class_to_depth) else 1
    )

    model = DepthModel(cfg, out_dim=out_dim).to(dev)
    state = torch.load(model_dir / "best_model.pt", map_location=dev)
    model.load_state_dict(state)
    model.eval()

    ds = WaveformWindowDataset(
        file_df, cfg, training=False, h5_data_key=h5_data_key, h5_time_key=h5_time_key
    )
    loader = make_loader(ds, cfg, shuffle=False)

    try:
        window_df = predict_loader(model, loader, dev, cfg)
    except PermissionError as exc:
        logger.warning(
            "DL DataLoader multiprocessing unavailable (%s). Retrying with num_workers=0.",
            exc,
        )
        cfg.num_workers = 0
        loader = make_loader(ds, cfg, shuffle=False)
        window_df = predict_loader(model, loader, dev, cfg)
    file_pred = aggregate_file_predictions(window_df, file_df, cfg, class_to_depth)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_pred.to_csv(out_csv, index=False)
    return file_pred


def _bundle_from_pred_csv(
    csv_path: Path,
    modality: str,
    *,
    fusion_mae: float,
    fusion_mae_source: str,
) -> PredictionBundle:
    df = pd.read_csv(csv_path)
    sigma = df["sigma"].to_numpy() if "sigma" in df.columns else np.zeros(len(df), dtype=np.float64)
    y_true, y_true_col = _extract_y_true_from_prediction_df(df)

    if "record_name" not in df.columns:
        raise KeyError(f"Expected 'record_name' column in {csv_path}, got {list(df.columns)}")
    fusion_keys = df["record_name"].astype(str).map(_fusion_key_from_record_name)

    metadata: dict[str, Any] = {
        "reference_mae_source": fusion_mae_source,
        "record_key": "step+hole (parsed from record_name)",
    }
    if y_true_col is not None:
        metadata["batch_ground_truth_column"] = y_true_col

    return PredictionBundle(
        modality=modality,
        record_names=fusion_keys.to_numpy(),
        y_pred=df["y_pred"].to_numpy(dtype=np.float64),
        sigma=sigma,
        validation_mae=float(fusion_mae),
        y_true=y_true,
        metadata=metadata,
    )


def _extract_y_true_from_prediction_df(df: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    for col in ("depth_mm", "y_true_depth", "y_true"):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(vals).any():
            return vals, col
    return None, None


def _fusion_key_from_record_name(record_name: str) -> str:
    """Modality-agnostic key for inter-fusion alignment (based on DOE identity).

    Segment filenames are canonical:
      {stem}__seg{NNN}__step{SSS}__{HOLE}__depth{D.DDD}

    Raw stems can differ between modalities; (step, hole) should match.
    """
    s = str(record_name)
    parts = s.split("__")
    step_idx: int | None = None
    hole: str | None = None
    for i, token in enumerate(parts):
        if token.lower().startswith("step"):
            try:
                step_idx = int(token[4:])
            except Exception:
                step_idx = None
            if i + 1 < len(parts):
                hole = parts[i + 1]
            break

    if step_idx is None and hole is None:
        return s
    if step_idx is None:
        return f"hole={hole}"
    if hole is None:
        return f"step={step_idx:03d}"
    return f"step={step_idx:03d}__hole={hole}"


def _classical_model_root_from_bundle_path(bundle_path: Path) -> Path:
    p = Path(bundle_path)
    if p.parent.name == "final_model":
        return p.parent.parent
    return p.parent


def _dl_model_root(model_dir: Path) -> Path:
    p = Path(model_dir)
    if p.name == "final_model":
        return p.parent
    if (p / "final_model").exists():
        return p
    return p


def _config_mae_fallback(cfg_node: dict[str, Any]) -> float | None:
    raw = cfg_node.get("fusion_mae_fallback", cfg_node.get("validation_mae", None))
    if raw is None:
        return None
    return float(raw)


def _read_classical_single_holdout_mae(bundle_path: Path) -> float:
    model_root = _classical_model_root_from_bundle_path(bundle_path)
    metrics_path = model_root / "final_model" / "test_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing classical test metrics: {metrics_path}")

    df = pd.read_csv(metrics_path)
    if df.empty:
        raise ValueError(f"Empty test metrics file: {metrics_path}")
    if "holdout_mae_raw" not in df.columns:
        raise KeyError(f"'holdout_mae_raw' column missing in {metrics_path}")

    maes = pd.to_numeric(df["holdout_mae_raw"], errors="coerce").to_numpy(dtype=np.float64)
    maes = maes[np.isfinite(maes)]
    if maes.size == 0:
        raise ValueError(f"No finite holdout_mae_raw values in {metrics_path}")

    mae = float(maes[0])
    logger.info(
        "Using classical single-model holdout raw MAE %.6f from %s",
        mae,
        metrics_path,
    )
    return mae


def _read_classical_ensemble_holdout_mae(bundle_path: Path) -> float:
    model_root = _classical_model_root_from_bundle_path(bundle_path)
    pred_path = model_root / "ensemble_test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing ensemble holdout predictions: {pred_path}")

    df = pd.read_csv(pred_path)
    if "y_pred_raw" not in df.columns:
        raise KeyError(f"'y_pred_raw' column missing in {pred_path}")

    if "y_true" in df.columns:
        target_col = "y_true"
    elif "depth_mm" in df.columns:
        target_col = "depth_mm"
    else:
        raise KeyError(f"Neither 'y_true' nor 'depth_mm' column found in {pred_path}")

    y_true = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
    y_pred_raw = pd.to_numeric(df["y_pred_raw"], errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred_raw)
    if not np.any(mask):
        raise ValueError(f"No finite (y_true, y_pred_raw) pairs in {pred_path}")

    mae = float(np.mean(np.abs(y_true[mask] - y_pred_raw[mask])))
    logger.info(
        "Using classical ensemble holdout raw MAE %.6f computed from %s (%s vs y_pred_raw)",
        mae,
        pred_path,
        target_col,
    )
    return mae


def _read_classical_fusion_mae(
    bundle_path: Path,
    bundle: dict[str, Any],
) -> float:
    if "model" in bundle:
        return _read_classical_single_holdout_mae(bundle_path)
    if "members" in bundle:
        return _read_classical_ensemble_holdout_mae(bundle_path)
    raise KeyError(
        "Unsupported classical bundle format. Expected 'model' or 'members'. "
        f"Available keys: {list(bundle.keys())}"
    )


def _read_dl_fusion_mae_from_summary(model_dir: Path) -> tuple[float, str]:
    model_root = _dl_model_root(model_dir)
    summary_path = model_root / "repeat_metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing DL repeat summary: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    source_key = (
        "mean_test_mae" if payload.get("mean_test_mae", None) is not None else "mean_val_mae"
    )
    mae = payload.get(source_key, None)
    if mae is None:
        raise KeyError(f"Neither 'mean_test_mae' nor 'mean_val_mae' found in {summary_path}")
    mae = float(mae)
    if not np.isfinite(mae):
        raise ValueError(f"Non-finite {source_key} in {summary_path}: {mae}")

    logger.info(
        "Using DL fusion MAE %.6f from %s (%s)",
        mae,
        summary_path,
        source_key,
    )
    return mae, source_key


def _resolve_classical_fusion_mae(
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
    cfg_mae_fallback: float | None,
) -> tuple[float, str]:
    try:
        mae = _read_classical_fusion_mae(bundle_path, bundle)
        if "model" in bundle:
            return mae, "final_model/test_metrics.csv.holdout_mae_raw"
        if "members" in bundle:
            return mae, "ensemble_test_predictions.csv.y_pred_raw"
        return mae, "bundle"
    except Exception as exc:
        if cfg_mae_fallback is not None:
            logger.warning(
                "Falling back to config classical fusion_mae_fallback=%s because MAE lookup failed: %s",
                cfg_mae_fallback,
                exc,
            )
            return float(cfg_mae_fallback), "config.fusion_mae_fallback"
        raise


def _resolve_dl_fusion_mae(
    *,
    model_dir: Path,
    cfg_mae_fallback: float | None,
) -> tuple[float, str]:
    try:
        mae, source_key = _read_dl_fusion_mae_from_summary(model_dir)
        return mae, f"repeat_metrics_summary.json.{source_key}"
    except Exception as exc:
        if cfg_mae_fallback is not None:
            logger.warning(
                "Falling back to config DL fusion_mae_fallback=%s because MAE lookup failed: %s",
                cfg_mae_fallback,
                exc,
            )
            return float(cfg_mae_fallback), "config.fusion_mae_fallback"
        raise


def _safe_rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _quality_entry(
    bundle: PredictionBundle, *, predictions_csv: Path, run_dir: Path
) -> dict[str, Any]:
    return {
        "predictions_csv": _safe_rel(predictions_csv, run_dir),
        **bundle_batch_metrics(bundle),
    }


def _save_bundle_predictions_csv(bundle: PredictionBundle, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    bundle.to_dataframe().to_csv(out_csv, index=False)
    return out_csv


def _copy_split_debug_plots_to_run_dir(
    summary: dict[str, Any], run_split_dir: Path
) -> dict[str, str]:
    """Copy splitter debug plots next to a run-specific split manifest."""
    run_split_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for summary_key in ("debug_core", "debug_padded"):
        raw_src = summary.get(summary_key)
        if not raw_src:
            continue
        src = Path(str(raw_src))
        if not src.exists():
            logger.warning("Missing split debug plot (%s): %s", summary_key, src)
            continue
        dst = run_split_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied[summary_key] = str(dst)
    return copied


def _resolve_local_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.is_absolute() else Path.cwd() / p


def _normalise_structure_extractor(raw: Any) -> str:
    key = str(raw if raw is not None else "v1").strip().lower()
    if key == "extensive":
        return "v2"
    if key in {"v1", "v2"}:
        return key
    return "v1"


def _resolve_classical_training_features_csv_path(bundle_path: Path) -> tuple[Path | None, str]:
    model_root = _classical_model_root_from_bundle_path(bundle_path)
    run_cfg = _safe_read_json(model_root / "run_config.json")
    best_meta = _safe_read_json(model_root / "final_model" / "best_model_metadata.json")

    candidates: list[tuple[str, Path | None]] = [
        (
            "best_model_metadata.features_csv",
            _resolve_local_path(str(best_meta.get("features_csv")))
            if best_meta and best_meta.get("features_csv")
            else None,
        ),
        (
            "run_config.features_csv",
            _resolve_local_path(str(run_cfg.get("features_csv")))
            if run_cfg and run_cfg.get("features_csv")
            else None,
        ),
    ]

    seen: set[Path] = set()
    for source_name, maybe_path in candidates:
        if maybe_path is None:
            continue
        path = maybe_path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return path, source_name

    return None, "unresolved"


def _effective_cfg_from_sidecar_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    eff = payload.get("effective_extraction_config")
    return eff if isinstance(eff, dict) else None


def _load_effective_cfg_from_sidecar_path(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _safe_read_json(path)
    return _effective_cfg_from_sidecar_payload(payload)


def _infer_structure_extractor_from_feature_cols(feature_cols: list[str]) -> tuple[str | None, str]:
    if not feature_cols:
        return None, "no feature columns available"

    v1_score = 0
    v2_score = 0
    for col in feature_cols:
        c = str(col).strip().lower()
        if c.startswith(("wpd_", "mfcc_", "dmfcc_", "ddmfcc_", "td_", "ss_", "br_", "tf_", "cx_")):
            v2_score += 3
        if re.match(r"^cwt_s\d+", c):
            v2_score += 2

        if c.startswith(
            ("dwt_", "spectral_", "st_", "peak_freq_", "peak_mag_", "band_power_", "ratio_")
        ):
            v1_score += 2
        if c in {"crest_factor", "waveform_length", "percentile_95"}:
            v1_score += 1

    if v2_score > v1_score:
        return "v2", f"feature_cols heuristic (v2_score={v2_score}, v1_score={v1_score})"
    if v1_score > v2_score:
        return "v1", f"feature_cols heuristic (v1_score={v1_score}, v2_score={v2_score})"
    return None, f"feature_cols heuristic inconclusive (v1_score={v1_score}, v2_score={v2_score})"


def _resolve_structure_runtime_extraction_cfg(
    *,
    bundle_path: Path,
    bundle_obj: dict[str, Any],
    base_cfg: dict[str, Any],
    feature_cols: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cfg = dict(base_cfg)
    warnings: list[str] = []
    info: dict[str, Any] = {
        "training_extraction_config_source": None,
        "training_sr_hz": None,
        "training_sr_source": None,
        "extractor_source": None,
    }

    model_root = _classical_model_root_from_bundle_path(bundle_path)
    run_cfg = _safe_read_json(model_root / "run_config.json")
    best_meta = _safe_read_json(model_root / "final_model" / "best_model_metadata.json")
    training_features_csv, training_features_source = _resolve_classical_training_features_csv_path(
        bundle_path
    )

    extraction_candidates: list[tuple[str, dict[str, Any] | None]] = [
        (
            "bundle.feature_extraction_config",
            bundle_obj.get("feature_extraction_config")
            if isinstance(bundle_obj.get("feature_extraction_config"), dict)
            else None,
        ),
        (
            "bundle.feature_extraction_sidecar.effective_extraction_config",
            _effective_cfg_from_sidecar_payload(
                bundle_obj.get("feature_extraction_sidecar")
                if isinstance(bundle_obj.get("feature_extraction_sidecar"), dict)
                else None
            ),
        ),
        (
            "best_model_metadata.feature_extraction_config",
            best_meta.get("feature_extraction_config")
            if isinstance(best_meta, dict)
            and isinstance(best_meta.get("feature_extraction_config"), dict)
            else None,
        ),
        (
            "run_config.feature_extraction_config",
            run_cfg.get("feature_extraction_config")
            if isinstance(run_cfg, dict)
            and isinstance(run_cfg.get("feature_extraction_config"), dict)
            else None,
        ),
    ]

    sidecar_paths: list[tuple[str, Path | None]] = [
        (
            "best_model_metadata.feature_extraction_sidecar_path",
            _resolve_local_path(str(best_meta.get("feature_extraction_sidecar_path")))
            if isinstance(best_meta, dict) and best_meta.get("feature_extraction_sidecar_path")
            else None,
        ),
        (
            "run_config.feature_extraction_sidecar_path",
            _resolve_local_path(str(run_cfg.get("feature_extraction_sidecar_path")))
            if isinstance(run_cfg, dict) and run_cfg.get("feature_extraction_sidecar_path")
            else None,
        ),
        (
            f"{training_features_source}.extractor_config_sidecar",
            Path(str(training_features_csv) + ".extractor_config.json")
            if training_features_csv is not None
            else None,
        ),
    ]

    seen_sidecar_paths: set[Path] = set()
    for source_name, maybe_path in sidecar_paths:
        if maybe_path is None:
            continue
        sidecar_path = maybe_path.resolve()
        if sidecar_path in seen_sidecar_paths:
            continue
        seen_sidecar_paths.add(sidecar_path)
        extraction_candidates.append(
            (
                f"{source_name}::{sidecar_path}",
                _load_effective_cfg_from_sidecar_path(sidecar_path),
            )
        )

    for source_name, cfg_candidate in extraction_candidates:
        if isinstance(cfg_candidate, dict) and cfg_candidate:
            cfg.update(cfg_candidate)
            info["training_extraction_config_source"] = source_name
            break

    if info["training_extraction_config_source"] is None:
        warnings.append(
            "No structure extraction config sidecar found in training artifacts; "
            "falling back to current configs/structure.yaml plus feature-signature inference."
        )

    if info["training_extraction_config_source"] is None:
        inferred_extractor, infer_source = _infer_structure_extractor_from_feature_cols(
            feature_cols
        )
        if inferred_extractor is not None:
            cfg["extractor"] = inferred_extractor
            info["extractor_source"] = infer_source
        else:
            cfg["extractor"] = _normalise_structure_extractor(cfg.get("extractor", "v1"))
            warnings.append(
                "Could not infer structure extractor version from model features; "
                "falling back to configs/structure.yaml default."
            )
            info["extractor_source"] = infer_source
    else:
        cfg["extractor"] = _normalise_structure_extractor(cfg.get("extractor", "v1"))
        info["extractor_source"] = (
            f"training config ({info['training_extraction_config_source']})"
            if info["training_extraction_config_source"] is not None
            else "configs/structure.yaml"
        )

    training_sr_hz, training_sr_source = _resolve_classical_training_sr_hz(bundle_path)
    if training_sr_hz is not None:
        cfg["target_sr_hz"] = int(training_sr_hz)
        info["training_sr_hz"] = int(training_sr_hz)
        info["training_sr_source"] = training_sr_source
    else:
        warnings.append(
            "Could not resolve structure training sr_hz_used from training features; "
            "runtime extraction will use config ds_rate settings."
        )
        info["training_sr_source"] = "unresolved"

    return cfg, info, warnings


def _resolve_airborne_runtime_extraction_cfg(
    *,
    bundle_path: Path,
    bundle_obj: dict[str, Any],
    base_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    cfg = dict(base_cfg)
    warnings: list[str] = []
    info: dict[str, Any] = {
        "training_extraction_config_source": None,
        "training_sr_hz": None,
        "training_sr_source": None,
    }

    model_root = _classical_model_root_from_bundle_path(bundle_path)
    run_cfg = _safe_read_json(model_root / "run_config.json")
    best_meta = _safe_read_json(model_root / "final_model" / "best_model_metadata.json")
    training_features_csv, training_features_source = _resolve_classical_training_features_csv_path(
        bundle_path
    )

    extraction_candidates: list[tuple[str, dict[str, Any] | None]] = [
        (
            "bundle.feature_extraction_config",
            bundle_obj.get("feature_extraction_config")
            if isinstance(bundle_obj.get("feature_extraction_config"), dict)
            else None,
        ),
        (
            "bundle.feature_extraction_sidecar.effective_extraction_config",
            _effective_cfg_from_sidecar_payload(
                bundle_obj.get("feature_extraction_sidecar")
                if isinstance(bundle_obj.get("feature_extraction_sidecar"), dict)
                else None
            ),
        ),
        (
            "best_model_metadata.feature_extraction_config",
            best_meta.get("feature_extraction_config")
            if isinstance(best_meta, dict)
            and isinstance(best_meta.get("feature_extraction_config"), dict)
            else None,
        ),
        (
            "run_config.feature_extraction_config",
            run_cfg.get("feature_extraction_config")
            if isinstance(run_cfg, dict)
            and isinstance(run_cfg.get("feature_extraction_config"), dict)
            else None,
        ),
    ]

    sidecar_paths: list[tuple[str, Path | None]] = [
        (
            "best_model_metadata.feature_extraction_sidecar_path",
            _resolve_local_path(str(best_meta.get("feature_extraction_sidecar_path")))
            if isinstance(best_meta, dict) and best_meta.get("feature_extraction_sidecar_path")
            else None,
        ),
        (
            "run_config.feature_extraction_sidecar_path",
            _resolve_local_path(str(run_cfg.get("feature_extraction_sidecar_path")))
            if isinstance(run_cfg, dict) and run_cfg.get("feature_extraction_sidecar_path")
            else None,
        ),
        (
            f"{training_features_source}.extractor_config_sidecar",
            Path(str(training_features_csv) + ".extractor_config.json")
            if training_features_csv is not None
            else None,
        ),
    ]

    seen_sidecar_paths: set[Path] = set()
    for source_name, maybe_path in sidecar_paths:
        if maybe_path is None:
            continue
        sidecar_path = maybe_path.resolve()
        if sidecar_path in seen_sidecar_paths:
            continue
        seen_sidecar_paths.add(sidecar_path)
        extraction_candidates.append(
            (
                f"{source_name}::{sidecar_path}",
                _load_effective_cfg_from_sidecar_path(sidecar_path),
            )
        )

    for source_name, cfg_candidate in extraction_candidates:
        if isinstance(cfg_candidate, dict) and cfg_candidate:
            cfg.update(cfg_candidate)
            info["training_extraction_config_source"] = source_name
            break

    if info["training_extraction_config_source"] is None:
        warnings.append(
            "No airborne extraction config sidecar found in training artifacts; "
            "falling back to current configs/airborne.yaml for non-SR extractor settings."
        )

    training_sr_hz, training_sr_source = _resolve_classical_training_sr_hz(bundle_path)
    if training_sr_hz is not None:
        cfg["target_sr"] = int(training_sr_hz)
        info["training_sr_hz"] = int(training_sr_hz)
        info["training_sr_source"] = training_sr_source
    else:
        warnings.append(
            "Could not resolve airborne training sr_hz from training features; "
            "runtime extraction will use configs/airborne.yaml target_sr."
        )
        info["training_sr_source"] = "unresolved"

    return cfg, info, warnings


def _read_training_sr_mode_hz(features_csv: Path) -> int | None:
    if not features_csv.exists():
        return None
    for col in ("sr_hz_used", "sr_hz"):
        try:
            series = pd.read_csv(features_csv, usecols=[col])[col]
        except ValueError:
            continue
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rounded = np.rint(vals).astype(np.int64)
        mode = pd.Series(rounded).mode()
        if not mode.empty and int(mode.iloc[0]) > 0:
            return int(mode.iloc[0])
    return None


def _resolve_classical_training_sr_hz(bundle_path: Path) -> tuple[int | None, str]:
    path, source_name = _resolve_classical_training_features_csv_path(bundle_path)
    if path is not None:
        sr_hz = _read_training_sr_mode_hz(path)
        if sr_hz is not None:
            return sr_hz, f"{source_name}::{path}"
    return None, "unresolved"


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _numeric_col_stats(df: pd.DataFrame, col: str) -> dict[str, float] | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return None
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def _sampling_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for c in ("sr_hz", "sr_hz_native", "sr_hz_used", "ds_rate", "duration_s"):
        st = _numeric_col_stats(df, c)
        if st is not None:
            stats[c] = st
    return stats


def _bundle_pred_diagnostics(bundle: PredictionBundle) -> dict[str, Any]:
    if bundle.y_true is None:
        return {
            "bias_mm": None,
            "corr_true_pred": None,
            "slope_pred_vs_true": None,
            "intercept_pred_vs_true": None,
            "mean_pred_mm": float(np.mean(bundle.y_pred)) if len(bundle.y_pred) else None,
            "mean_true_mm": None,
        }

    y_true = np.asarray(bundle.y_true, dtype=np.float64)
    y_pred = np.asarray(bundle.y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return {
            "bias_mm": None,
            "corr_true_pred": None,
            "slope_pred_vs_true": None,
            "intercept_pred_vs_true": None,
            "mean_pred_mm": None,
            "mean_true_mm": None,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    bias = float(np.mean(yp - yt))
    mean_pred = float(np.mean(yp))
    mean_true = float(np.mean(yt))
    if len(yt) < 2:
        return {
            "bias_mm": bias,
            "corr_true_pred": None,
            "slope_pred_vs_true": None,
            "intercept_pred_vs_true": None,
            "mean_pred_mm": mean_pred,
            "mean_true_mm": mean_true,
        }
    corr = float(np.corrcoef(yt, yp)[0, 1])
    slope, intercept = np.polyfit(yt, yp, 1)
    return {
        "bias_mm": bias,
        "corr_true_pred": corr,
        "slope_pred_vs_true": float(slope),
        "intercept_pred_vs_true": float(intercept),
        "mean_pred_mm": mean_pred,
        "mean_true_mm": mean_true,
    }


def _snapped_mae_mm(bundle: PredictionBundle, step_mm: float | None) -> float | None:
    if bundle.y_true is None or step_mm is None or step_mm <= 0:
        return None
    y_true = np.asarray(bundle.y_true, dtype=np.float64)
    y_pred = np.asarray(bundle.y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return None
    yp = y_pred[mask]
    yt = y_true[mask]
    yp_snap = np.round(yp / step_mm) * step_mm
    return float(np.mean(np.abs(yp_snap - yt)))


def _apples_to_apples_entry(
    *,
    reference_mae_raw_mm: float | None,
    new_batch_mae_raw_mm: float | None,
    diagnostics: dict[str, Any],
    reference_mae_snapped_mm: float | None = None,
    new_batch_mae_snapped_mm: float | None = None,
) -> dict[str, Any]:
    raw_mae_delta_mm = None
    raw_mae_ratio = None
    if reference_mae_raw_mm is not None and new_batch_mae_raw_mm is not None:
        ref = float(reference_mae_raw_mm)
        new = float(new_batch_mae_raw_mm)
        raw_mae_delta_mm = new - ref
        if ref > 0:
            raw_mae_ratio = new / ref

    return {
        "reference_mae_raw_mm": reference_mae_raw_mm,
        "reference_mae_snapped_mm": reference_mae_snapped_mm,
        "new_batch_mae_raw_mm": new_batch_mae_raw_mm,
        "new_batch_mae_snapped_mm": new_batch_mae_snapped_mm,
        "raw_mae_delta_mm": raw_mae_delta_mm,
        "raw_mae_ratio": raw_mae_ratio,
        "diagnostics": diagnostics,
    }


def _bundle_feature_cols(bundle_obj: dict[str, Any] | None) -> list[str]:
    if not bundle_obj:
        return []
    if "feature_cols" in bundle_obj:
        cols = bundle_obj.get("feature_cols")
        return [str(c) for c in cols] if isinstance(cols, list) else []
    members = bundle_obj.get("members")
    if isinstance(members, list) and members and isinstance(members[0], dict):
        cols = members[0].get("feature_cols")
        return [str(c) for c in cols] if isinstance(cols, list) else []
    return []


def _read_training_reference_snapped_mae_mm(classical_model_root: Path) -> float | None:
    p = classical_model_root / "final_model" / "ensemble_test_metrics.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty:
            return None
        for col in ("ensemble_mae", "holdout_mae"):
            if col in df.columns:
                val = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
                val = val[np.isfinite(val)]
                if val.size:
                    return float(val[0])
    except Exception:
        return None
    return None


def _classical_setup_snapshot(
    *,
    model_key: str,
    bundle_path: Path | None,
    bundle_obj: dict[str, Any] | None,
    reference_mae_source: str,
    reference_mae_mm: float | None,
    runtime_pred_csv: Path,
    runtime_features_csv: Path | None,
    runtime_extraction_replay: dict[str, Any] | None,
    run_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    feature_cols = _bundle_feature_cols(bundle_obj)
    model_root = _classical_model_root_from_bundle_path(bundle_path) if bundle_path else None
    run_cfg_path = model_root / "run_config.json" if model_root else None
    meta_path = model_root / "final_model" / "best_model_metadata.json" if model_root else None
    run_cfg = _safe_read_json(run_cfg_path) if run_cfg_path else None
    best_meta = _safe_read_json(meta_path) if meta_path else None

    features_csv_train: Path | None = None
    if best_meta and best_meta.get("features_csv"):
        features_csv_train = _resolve_local_path(str(best_meta["features_csv"]))
    elif run_cfg and run_cfg.get("features_csv"):
        features_csv_train = _resolve_local_path(str(run_cfg["features_csv"]))

    training_extraction_config: dict[str, Any] | None = None
    training_extraction_config_source: str | None = None
    if bundle_obj and isinstance(bundle_obj.get("feature_extraction_config"), dict):
        training_extraction_config = dict(bundle_obj["feature_extraction_config"])
        training_extraction_config_source = "bundle.feature_extraction_config"
    elif best_meta and isinstance(best_meta.get("feature_extraction_config"), dict):
        training_extraction_config = dict(best_meta["feature_extraction_config"])
        training_extraction_config_source = "best_model_metadata.feature_extraction_config"
    elif run_cfg and isinstance(run_cfg.get("feature_extraction_config"), dict):
        training_extraction_config = dict(run_cfg["feature_extraction_config"])
        training_extraction_config_source = "run_config.feature_extraction_config"
    else:
        sidecar_candidates: list[tuple[str, Path | None]] = [
            (
                "best_model_metadata.feature_extraction_sidecar_path",
                _resolve_local_path(str(best_meta.get("feature_extraction_sidecar_path")))
                if best_meta and best_meta.get("feature_extraction_sidecar_path")
                else None,
            ),
            (
                "run_config.feature_extraction_sidecar_path",
                _resolve_local_path(str(run_cfg.get("feature_extraction_sidecar_path")))
                if run_cfg and run_cfg.get("feature_extraction_sidecar_path")
                else None,
            ),
            (
                "features_csv.extractor_config_sidecar",
                Path(str(features_csv_train) + ".extractor_config.json")
                if features_csv_train is not None
                else None,
            ),
        ]
        seen_sidecars: set[Path] = set()
        for source_name, maybe_path in sidecar_candidates:
            if maybe_path is None:
                continue
            path = maybe_path.resolve()
            if path in seen_sidecars:
                continue
            seen_sidecars.add(path)
            payload = _safe_read_json(path)
            effective_cfg = _effective_cfg_from_sidecar_payload(payload)
            if effective_cfg:
                training_extraction_config = dict(effective_cfg)
                training_extraction_config_source = f"{source_name}::{path}"
                break

    train_sampling: dict[str, dict[str, float]] = {}
    missing_in_training_source: list[str] = []
    if features_csv_train and features_csv_train.exists():
        try:
            df_train = pd.read_csv(features_csv_train)
            train_sampling = _sampling_stats(df_train)
            train_cols = set(df_train.columns)
            missing_in_training_source = [c for c in feature_cols if c not in train_cols]
            if missing_in_training_source:
                warnings.append(
                    f"{model_key}: training feature source is missing {len(missing_in_training_source)} "
                    f"required model feature(s)."
                )
        except Exception:
            warnings.append(f"{model_key}: failed to read training feature source CSV.")

    runtime_sampling: dict[str, dict[str, float]] = {}
    try:
        df_runtime = pd.read_csv(runtime_pred_csv)
        runtime_sampling = _sampling_stats(df_runtime)
    except Exception:
        warnings.append(f"{model_key}: failed to read runtime predictions CSV for sampling stats.")

    runtime_feature_missing_required: list[str] = []
    runtime_feature_all_nan_required: list[str] = []
    runtime_feature_cols_total: int | None = None
    if runtime_features_csv is not None and Path(runtime_features_csv).exists():
        try:
            df_runtime_feat = pd.read_csv(runtime_features_csv)
            runtime_feature_cols_total = int(len(df_runtime_feat.columns))
            runtime_cols = set(df_runtime_feat.columns)
            runtime_feature_missing_required = [c for c in feature_cols if c not in runtime_cols]
            for c in feature_cols:
                if c not in runtime_cols:
                    continue
                vals = pd.to_numeric(df_runtime_feat[c], errors="coerce").to_numpy(dtype=np.float64)
                if not np.isfinite(vals).any():
                    runtime_feature_all_nan_required.append(c)
            if runtime_feature_missing_required:
                warnings.append(
                    f"{model_key}: runtime features are missing {len(runtime_feature_missing_required)} "
                    f"required model feature(s)."
                )
            if runtime_feature_all_nan_required:
                warnings.append(
                    f"{model_key}: runtime features contain {len(runtime_feature_all_nan_required)} "
                    f"required model feature(s) that are all-NaN."
                )
        except Exception:
            warnings.append(
                f"{model_key}: failed to read runtime features CSV for feature coverage."
            )

    # Sampling mismatch checks on common columns.
    for c in sorted(set(train_sampling.keys()) & set(runtime_sampling.keys())):
        tr_mean = train_sampling[c]["mean"]
        rt_mean = runtime_sampling[c]["mean"]
        if not np.isfinite(tr_mean) or not np.isfinite(rt_mean):
            continue
        if abs(tr_mean - rt_mean) > max(1e-6, 1e-3 * max(abs(tr_mean), abs(rt_mean), 1.0)):
            warnings.append(
                f"{model_key}: sampling column {c} mean mismatch (train={tr_mean}, runtime={rt_mean})."
            )

    snapshot = {
        "model_kind": "classical",
        "bundle_path": _safe_rel(bundle_path, run_dir) if bundle_path else None,
        "bundle_path_sha256": _sha256_file(bundle_path) if bundle_path else None,
        "reference_mae_raw_mm": reference_mae_mm,
        "reference_mae_source": reference_mae_source,
        "bundle_defaults": {
            "snap_predictions": bundle_obj.get("snap_predictions") if bundle_obj else None,
            "doe_step_mm": bundle_obj.get("doe_step_mm") if bundle_obj else None,
            "target_col": bundle_obj.get("target_col") if bundle_obj else None,
            "group_col": bundle_obj.get("group_col") if bundle_obj else None,
        },
        "model_feature_signature": {
            "n_features": int(len(feature_cols)),
            "sha256": hashlib.sha256("|".join(feature_cols).encode("utf-8")).hexdigest()
            if feature_cols
            else None,
        },
        "training_artifacts": {
            "run_config_path": _safe_rel(run_cfg_path, run_dir) if run_cfg_path else None,
            "best_model_metadata_path": _safe_rel(meta_path, run_dir) if meta_path else None,
            "features_csv_path": _safe_rel(features_csv_train, run_dir)
            if features_csv_train
            else None,
            "features_csv_sha256": _sha256_file(features_csv_train),
            "features_sampling_stats": train_sampling,
            "features_missing_required_model_cols": missing_in_training_source,
            "feature_extraction_config_source": training_extraction_config_source,
            "feature_extraction_config": training_extraction_config,
            "best_model_metadata": best_meta,
        },
        "runtime_artifacts": {
            "predictions_csv_path": _safe_rel(runtime_pred_csv, run_dir),
            "predictions_csv_sampling_stats": runtime_sampling,
            "features_csv_path": _safe_rel(runtime_features_csv, run_dir)
            if runtime_features_csv is not None
            else None,
            "features_columns_total": runtime_feature_cols_total,
            "features_required_missing_model_cols": runtime_feature_missing_required,
            "features_required_all_nan_model_cols": runtime_feature_all_nan_required,
            "extraction_replay": runtime_extraction_replay,
        },
    }
    return snapshot, warnings


def _dl_setup_snapshot(
    *,
    model_key: str,
    model_dir: Path | None,
    reference_mae_source: str,
    reference_mae_mm: float | None,
    run_dir: Path,
) -> dict[str, Any]:
    if model_dir is None:
        return {
            "model_kind": "dl",
            "model_dir": None,
            "reference_mae_raw_mm": reference_mae_mm,
            "reference_mae_source": reference_mae_source,
        }
    model_root = _dl_model_root(model_dir)
    cfg_path = (
        model_root / "final_model" / "config.json"
        if (model_root / "final_model" / "config.json").exists()
        else model_root / "config.json"
    )
    cfg_payload = _safe_read_json(cfg_path)
    file_table_path = (
        model_root / "final_model" / "file_table.csv"
        if (model_root / "final_model" / "file_table.csv").exists()
        else model_root / "file_table.csv"
    )
    file_table_stats: dict[str, dict[str, float]] = {}
    if file_table_path.exists():
        try:
            df = pd.read_csv(file_table_path)
            file_table_stats = _sampling_stats(df)
        except Exception:
            file_table_stats = {}
    return {
        "model_kind": "dl",
        "model_dir": _safe_rel(model_dir, run_dir),
        "config_path": _safe_rel(cfg_path, run_dir),
        "config_sha256": _sha256_file(cfg_path),
        "reference_mae_raw_mm": reference_mae_mm,
        "reference_mae_source": reference_mae_source,
        "training_file_table_path": _safe_rel(file_table_path, run_dir)
        if file_table_path.exists()
        else None,
        "training_file_table_sampling_stats": file_table_stats,
        "model_config": cfg_payload,
    }


def _model_setup_lock_entry(snapshot: dict[str, Any]) -> dict[str, Any]:
    kind = str(snapshot.get("model_kind", ""))
    if kind == "classical":
        sig = snapshot.get("model_feature_signature", {})
        if not isinstance(sig, dict):
            sig = {}
        return {
            "model_kind": "classical",
            "bundle_path": snapshot.get("bundle_path"),
            "bundle_path_sha256": snapshot.get("bundle_path_sha256"),
            "model_feature_signature_n_features": sig.get("n_features"),
            "model_feature_signature_sha256": sig.get("sha256"),
            "reference_mae_raw_mm": snapshot.get("reference_mae_raw_mm"),
            "reference_mae_source": snapshot.get("reference_mae_source"),
        }
    if kind == "dl":
        return {
            "model_kind": "dl",
            "model_dir": snapshot.get("model_dir"),
            "config_path": snapshot.get("config_path"),
            "config_sha256": snapshot.get("config_sha256"),
            "reference_mae_raw_mm": snapshot.get("reference_mae_raw_mm"),
            "reference_mae_source": snapshot.get("reference_mae_source"),
        }
    return {"model_kind": kind}


def _build_model_setup_lock_payload(setup_audit: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    models_obj = setup_audit.get("models", {})
    models: dict[str, Any] = {}
    if isinstance(models_obj, dict):
        for model_key in sorted(models_obj.keys()):
            model_snapshot = models_obj.get(model_key)
            if isinstance(model_snapshot, dict):
                models[model_key] = _model_setup_lock_entry(model_snapshot)

    run_cfg = setup_audit.get("run_config_snapshot", {})
    if not isinstance(run_cfg, dict):
        run_cfg = {}

    return {
        "description": "Canonical lock snapshot of model artifacts used for fused prediction runs.",
        "run_name": run_dir.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "final_prediction_config": run_cfg.get("final_prediction_config"),
        "models": models,
    }


def _compare_model_setup_locks(
    previous_lock: dict[str, Any] | None,
    current_lock: dict[str, Any],
) -> list[str]:
    if not isinstance(previous_lock, dict):
        return []

    warnings: list[str] = []
    prev_models = previous_lock.get("models", {})
    cur_models = current_lock.get("models", {})
    if not isinstance(prev_models, dict):
        prev_models = {}
    if not isinstance(cur_models, dict):
        cur_models = {}

    for key in sorted(set(prev_models.keys()) | set(cur_models.keys())):
        if key not in prev_models:
            warnings.append(f"setup-lock drift: model '{key}' added vs previous latest lock.")
            continue
        if key not in cur_models:
            warnings.append(f"setup-lock drift: model '{key}' missing vs previous latest lock.")
            continue

        prev_entry = prev_models[key]
        cur_entry = cur_models[key]
        if not isinstance(prev_entry, dict) or not isinstance(cur_entry, dict):
            continue

        for field in (
            "bundle_path_sha256",
            "config_sha256",
            "model_feature_signature_sha256",
            "model_feature_signature_n_features",
        ):
            prev_val = prev_entry.get(field)
            cur_val = cur_entry.get(field)
            if prev_val is not None and cur_val is not None and str(prev_val) != str(cur_val):
                warnings.append(
                    f"setup-lock drift: {key} field '{field}' changed "
                    f"(prev={prev_val}, current={cur_val})."
                )
    return warnings


def _persist_model_setup_lock(
    *,
    setup_audit: dict[str, Any],
    run_dir: Path,
    final_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    lock_dir = final_dir / "model_setup_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)

    run_lock_path = lock_dir / f"{run_dir.name}_setup_lock.json"
    latest_lock_path = lock_dir / "LATEST_setup_lock.json"

    lock_payload = _build_model_setup_lock_payload(setup_audit, run_dir)
    previous_lock = _safe_read_json(latest_lock_path)
    drift_warnings = _compare_model_setup_locks(previous_lock, lock_payload)

    with open(run_lock_path, "w", encoding="utf-8") as fh:
        json.dump(lock_payload, fh, indent=2)
    with open(latest_lock_path, "w", encoding="utf-8") as fh:
        json.dump(lock_payload, fh, indent=2)

    return (
        {
            "run_lock": _safe_rel(run_lock_path, run_dir),
            "latest_lock": _safe_rel(latest_lock_path, run_dir),
        },
        drift_warnings,
    )


def main() -> None:
    args = build_parser().parse_args()

    # This project pins modern scientific stack (see pyproject.toml).
    # Many stored Joblib bundles are not unpicklable on NumPy < 2.
    import numpy as _np

    if sys.version_info < (3, 13) or int(_np.__version__.split(".", 1)[0]) < 2:
        raise RuntimeError(
            "Environment mismatch for vm-predict-fused.\n"
            f"- Detected Python {sys.version_info.major}.{sys.version_info.minor}\n"
            f"- Detected NumPy {_np.__version__}\n\n"
            "This repo targets Python >= 3.13 and NumPy >= 2.0 (see pyproject.toml).\n"
            "Re-run with the correct project environment/venv."
        )

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    if args.out_dir:
        cfg.setdefault("run", {})["out_dir"] = args.out_dir

    run_out_root = Path(cfg.get("run", {}).get("out_dir", "data/fusion_results"))
    state_path = Path(
        cfg.get("run", {}).get("state_json", "data/fusion_results/final_prediction_state.json")
    )
    run_tag = str(cfg.get("run", {}).get("tag", "manual"))
    run_dir = run_out_root / f"{_now_tag()}__{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(state_path)
    processed: dict[str, Any] = state.setdefault("processed", {})
    exp_map, doe_template = _expected_map(cfg)

    only_air = args.only in {"both", "airborne"}
    only_str = args.only in {"both", "structure"}

    # Collect worklist
    work_air: list[RawFile] = []
    work_str: list[RawFile] = []

    if only_air and bool(cfg.get("inputs", {}).get("airborne", {}).get("enabled", True)):
        air_raw_dir = Path(cfg["inputs"]["airborne"]["raw_dir"])
        air_glob = str(cfg["inputs"]["airborne"].get("file_glob", "**/*.flac"))
        for rf in _scan_raw(air_raw_dir, air_glob):
            key = str(rf.path.resolve())
            if (
                (not args.force)
                and key in processed
                and processed[key].get("mtime_ns") == rf.mtime_ns
            ):
                continue
            work_air.append(rf)

    if only_str and bool(cfg.get("inputs", {}).get("structure", {}).get("enabled", True)):
        st_raw_dir = Path(cfg["inputs"]["structure"]["raw_dir"])
        st_glob = str(cfg["inputs"]["structure"].get("file_glob", "**/*.h5"))
        for rf in _scan_raw(st_raw_dir, st_glob):
            key = str(rf.path.resolve())
            if (
                (not args.force)
                and key in processed
                and processed[key].get("mtime_ns") == rf.mtime_ns
            ):
                continue
            work_str.append(rf)

    if not work_air and not work_str:
        print("No new raw files detected (or all already processed).")
        return

    # Restrict inference to segments generated from this run's raw-file stems.
    air_roots = {rf.stem for rf in work_air}
    str_roots = {rf.stem for rf in work_str}

    # Splitting
    split_cfg = cfg.get("splitting", {})
    common = split_cfg.get("common", {})
    band_hz = (float(common.get("band_low_hz", 2000.0)), float(common.get("band_high_hz", 5000.0)))
    common_band_fallbacks = common.get("band_fallbacks_hz", None)

    seg_root_air = Path(
        split_cfg.get("segments_root", {}).get(
            "airborne", "data/raw_data_extracted_splits/air/live"
        )
    )
    seg_root_str = Path(
        split_cfg.get("segments_root", {}).get(
            "structure", "data/raw_data_extracted_splits/structure/live"
        )
    )

    # Split airborne
    seg_dir_air = None
    if work_air:
        air_split = split_cfg.get("airborne", {})
        air_pre_pad_s = float(air_split.get("pre_pad_s", common.get("pre_pad_s", 0.20)))
        air_post_pad_s = float(air_split.get("post_pad_s", common.get("post_pad_s", 0.25)))
        air_band_fallbacks = air_split.get("band_fallbacks_hz", common_band_fallbacks)
        logger.info(
            "Airborne split padding resolved to pre_pad_s=%.3f, post_pad_s=%.3f",
            air_pre_pad_s,
            air_post_pad_s,
        )
        for rf in work_air:
            target_seg_dir = seg_root_air / rf.stem
            if target_seg_dir.exists():
                shutil.rmtree(target_seg_dir)
            expected = _resolve_expected(exp_map, rf.stem)
            doe_df = _doe_for_file(doe_template, expected)
            manifest_df, summary = process_one_file(
                rf.path,
                doe_df,
                seg_root_air,
                expected_segments=expected,
                pre_pad_s=air_pre_pad_s,
                post_pad_s=air_post_pad_s,
                band_hz=band_hz,
                band_hz_fallbacks=air_band_fallbacks,
                export_format=str(air_split.get("export_format", "flac")),
                target_sr=air_split.get("target_sr", None),
            )
            out_manifest = run_dir / "airborne" / rf.stem / "segments_manifest.csv"
            out_manifest.parent.mkdir(parents=True, exist_ok=True)
            copied_debug = _copy_split_debug_plots_to_run_dir(summary, out_manifest.parent)
            manifest_df.to_csv(out_manifest, index=False)
            processed[str(rf.path.resolve())] = {
                "mtime_ns": rf.mtime_ns,
                "modality": "airborne",
                "stem": rf.stem,
            }
            logger.info(
                "Split airborne %s -> %s (segments=%d, copied_debug=%d)",
                rf.path.name,
                summary["out_dir"],
                summary["exported_segments_final"],
                len(copied_debug),
            )
        seg_dir_air = seg_root_air

    # Split structure
    seg_dir_str = None
    if work_str:
        st_split = split_cfg.get("structure", {})
        st_pre_pad_s = float(st_split.get("pre_pad_s", common.get("pre_pad_s", 0.20)))
        st_post_pad_s = float(st_split.get("post_pad_s", common.get("post_pad_s", 0.25)))
        st_band_fallbacks = st_split.get("band_fallbacks_hz", common_band_fallbacks)
        logger.info(
            "Structure split padding resolved to pre_pad_s=%.3f, post_pad_s=%.3f",
            st_pre_pad_s,
            st_post_pad_s,
        )
        for rf in work_str:
            target_seg_dir = seg_root_str / rf.stem
            if target_seg_dir.exists():
                shutil.rmtree(target_seg_dir)
            expected = _resolve_expected(exp_map, rf.stem)
            doe_df = _doe_for_file(doe_template, expected)
            manifest_df, summary = process_one_file(
                rf.path,
                doe_df,
                seg_root_str,
                expected_segments=expected,
                pre_pad_s=st_pre_pad_s,
                post_pad_s=st_post_pad_s,
                band_hz=band_hz,
                band_hz_fallbacks=st_band_fallbacks,
                export_format=str(st_split.get("export_format", "h5")),
                h5_data_key=str(st_split.get("h5_data_key", "measurement/data")),
                h5_time_key=str(st_split.get("h5_time_key", "measurement/time_vector")),
                target_sr=st_split.get("target_sr", None),
            )
            out_manifest = run_dir / "structure" / rf.stem / "segments_manifest.csv"
            out_manifest.parent.mkdir(parents=True, exist_ok=True)
            copied_debug = _copy_split_debug_plots_to_run_dir(summary, out_manifest.parent)
            manifest_df.to_csv(out_manifest, index=False)
            processed[str(rf.path.resolve())] = {
                "mtime_ns": rf.mtime_ns,
                "modality": "structure",
                "stem": rf.stem,
            }
            logger.info(
                "Split structure %s -> %s (segments=%d, copied_debug=%d)",
                rf.path.name,
                summary["out_dir"],
                summary["exported_segments_final"],
                len(copied_debug),
            )
        seg_dir_str = seg_root_str

    _save_state(state_path, state)

    # Inference + fusion
    fused_modality_bundles: list[PredictionBundle] = []
    batch_quality: dict[str, Any] = {
        "description": (
            "Performance measured on this run's raw batch using labels available in the "
            "prediction CSVs (depth_mm / y_true_depth / y_true)."
        ),
        "metrics_definition": {
            "mae_mm": "Mean absolute error in mm on the current batch.",
            "rmse_mm": "Root mean squared error in mm on the current batch.",
        },
        "models": {},
        "modality_fusions": {},
        "final_fusion": None,
    }
    setup_audit: dict[str, Any] = {
        "description": (
            "Snapshot of model + data setup used by this run, plus compatibility checks "
            "between training artifacts and current-batch inference setup."
        ),
        "models": {},
        "warnings": [],
    }
    apples_to_apples: dict[str, Any] = {
        "description": (
            "Raw-mm apples-to-apples comparison between training reference metrics "
            "(from each model's own artifacts) and current-batch metrics."
        ),
        "models": {},
        "modality_fusions": {},
        "final_fusion": None,
    }
    fusion_min_weight = float(cfg.get("models", {}).get("fusion", {}).get("min_weight", 0.05))

    # Airborne
    air_models_cfg = cfg.get("models", {}).get("airborne", {})
    air_any_enabled = bool(air_models_cfg.get("classical", {}).get("enabled", True)) or bool(
        air_models_cfg.get("dl", {}).get("enabled", True)
    )
    if seg_dir_air is not None and air_any_enabled:
        air_cfg = cfg["models"]["airborne"]

        # Classical (features)
        cls_cfg = air_cfg.get("classical", {})
        air_classical_csv = run_dir / "airborne" / "classical_predictions.csv"
        air_features_csv = run_dir / "airborne" / "features_airborne.csv"
        air_fusion_csv_path = run_dir / "airborne" / "fusion_predictions.csv"
        air_runtime_extraction_replay: dict[str, Any] | None = None
        air_classical_fusion_mae: float | None = None
        air_classical_fusion_mae_source = ""
        air_classical_bundle_path: Path | None = None
        air_classical_bundle_obj: dict[str, Any] | None = None
        if cls_cfg.get("enabled", True) and cls_cfg.get("bundle_path"):
            bundle_path = Path(cls_cfg["bundle_path"])
            bundle = joblib.load(bundle_path)
            air_classical_bundle_path = bundle_path
            air_classical_bundle_obj = bundle
            air_classical_fusion_mae, air_classical_fusion_mae_source = (
                _resolve_classical_fusion_mae(
                    bundle_path=bundle_path,
                    bundle=bundle,
                    cfg_mae_fallback=_config_mae_fallback(cls_cfg),
                )
            )
            feat_cols = None
            if "feature_cols" in bundle:
                feat_cols = list(bundle["feature_cols"])
            elif "members" in bundle and bundle["members"]:
                feat_cols = list(bundle["members"][0]["feature_cols"])

            feat_cfg_raw = load_config("configs/airborne.yaml")
            feat_cfg_base = feat_cfg_raw.get("classical", feat_cfg_raw)
            feat_cfg, air_runtime_extraction_replay, air_cfg_replay_warnings = (
                _resolve_airborne_runtime_extraction_cfg(
                    bundle_path=bundle_path,
                    bundle_obj=bundle,
                    base_cfg=feat_cfg_base,
                )
            )
            logger.info(
                "Airborne classical runtime extraction config source=%s | target_sr=%s",
                air_runtime_extraction_replay.get("training_extraction_config_source")
                if air_runtime_extraction_replay
                else None,
                feat_cfg.get("target_sr"),
            )
            for warn_msg in air_cfg_replay_warnings:
                logger.warning("%s", warn_msg)
            setup_audit["warnings"].extend(air_cfg_replay_warnings)
            features_df = _extract_for_roots(
                roots=air_roots,
                segments_root=seg_dir_air,
                extractor=extract_airborne,
                cfg=feat_cfg,
                file_glob=str(feat_cfg.get("file_glob", AIRBORNE_DEFAULT_FILE_GLOB)),
                n_workers=int(feat_cfg.get("n_workers", AIRBORNE_DEFAULT_N_WORKERS)),
            )
            if feat_cols is not None:
                features_df = _ensure_feature_cols(features_df, feat_cols)
            air_features_csv.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(air_features_csv, index=False)

            infer_classical(
                bundle_path=bundle_path,
                features_csv=air_features_csv,
                out_csv=air_classical_csv,
                snap_predictions=cls_cfg.get("snap_predictions", None),
                doe_step_mm=cls_cfg.get("snap_step_mm", None),
            )

        # DL
        dl_cfg = air_cfg.get("dl", {})
        air_dl_csv = run_dir / "airborne" / "dl_predictions.csv"
        air_dl_fusion_mae: float | None = None
        air_dl_fusion_mae_source = ""
        air_dl_model_dir: Path | None = None
        if dl_cfg.get("enabled", True) and dl_cfg.get("model_dir"):
            air_dl_model_dir = Path(dl_cfg["model_dir"])
            air_dl_fusion_mae, air_dl_fusion_mae_source = _resolve_dl_fusion_mae(
                model_dir=air_dl_model_dir,
                cfg_mae_fallback=_config_mae_fallback(dl_cfg),
            )
            _infer_dl_on_segments(
                model_dir=air_dl_model_dir,
                segments_dir=seg_dir_air,
                file_glob=None,
                device=str(dl_cfg.get("device", "auto")),
                batch_size=dl_cfg.get("batch_size", None),
                out_csv=air_dl_csv,
                h5_data_key="measurement/data",
                h5_time_key="measurement/time_vector",
                include_recording_roots=air_roots,
            )

        # Fuse intra-modality if both exist
        if air_classical_csv.exists() and air_dl_csv.exists():
            if air_classical_fusion_mae is None or air_dl_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for airborne intra-fusion.")
            air_cls_bundle = _bundle_from_pred_csv(
                air_classical_csv,
                "airborne_classical",
                fusion_mae=air_classical_fusion_mae,
                fusion_mae_source=air_classical_fusion_mae_source,
            )
            batch_quality["models"]["airborne_classical"] = _quality_entry(
                air_cls_bundle, predictions_csv=air_classical_csv, run_dir=run_dir
            )
            air_cls_ref_snap = (
                _read_training_reference_snapped_mae_mm(
                    _classical_model_root_from_bundle_path(air_classical_bundle_path)
                )
                if air_classical_bundle_path is not None
                else None
            )
            air_cls_step = (
                float(air_classical_bundle_obj.get("doe_step_mm"))
                if air_classical_bundle_obj
                and air_classical_bundle_obj.get("doe_step_mm") is not None
                else 0.1
            )
            air_cls_raw = batch_quality["models"]["airborne_classical"]["mae_mm"]
            apples_to_apples["models"]["airborne_classical"] = _apples_to_apples_entry(
                reference_mae_raw_mm=air_classical_fusion_mae,
                reference_mae_snapped_mm=air_cls_ref_snap,
                new_batch_mae_raw_mm=air_cls_raw,
                new_batch_mae_snapped_mm=_snapped_mae_mm(air_cls_bundle, air_cls_step),
                diagnostics=_bundle_pred_diagnostics(air_cls_bundle),
            )
            cls_setup, cls_warn = _classical_setup_snapshot(
                model_key="airborne_classical",
                bundle_path=air_classical_bundle_path,
                bundle_obj=air_classical_bundle_obj,
                reference_mae_source=air_classical_fusion_mae_source,
                reference_mae_mm=air_classical_fusion_mae,
                runtime_pred_csv=air_classical_csv,
                runtime_features_csv=air_features_csv,
                runtime_extraction_replay=air_runtime_extraction_replay,
                run_dir=run_dir,
            )
            setup_audit["models"]["airborne_classical"] = cls_setup
            setup_audit["warnings"].extend(cls_warn)
            air_dl_bundle = _bundle_from_pred_csv(
                air_dl_csv,
                "airborne_dl",
                fusion_mae=air_dl_fusion_mae,
                fusion_mae_source=air_dl_fusion_mae_source,
            )
            batch_quality["models"]["airborne_dl"] = _quality_entry(
                air_dl_bundle, predictions_csv=air_dl_csv, run_dir=run_dir
            )
            air_dl_raw = batch_quality["models"]["airborne_dl"]["mae_mm"]
            apples_to_apples["models"]["airborne_dl"] = _apples_to_apples_entry(
                reference_mae_raw_mm=air_dl_fusion_mae,
                new_batch_mae_raw_mm=air_dl_raw,
                new_batch_mae_snapped_mm=_snapped_mae_mm(air_dl_bundle, 0.1),
                diagnostics=_bundle_pred_diagnostics(air_dl_bundle),
            )
            setup_audit["models"]["airborne_dl"] = _dl_setup_snapshot(
                model_key="airborne_dl",
                model_dir=air_dl_model_dir,
                reference_mae_source=air_dl_fusion_mae_source,
                reference_mae_mm=air_dl_fusion_mae,
                run_dir=run_dir,
            )
            air_fused = fuse_intra_modality(
                air_cls_bundle,
                air_dl_bundle,
                modality_name="airborne_ensemble",
                min_weight=fusion_min_weight,
            )
            air_fusion_csv = _save_bundle_predictions_csv(
                air_fused,
                air_fusion_csv_path,
            )
            batch_quality["modality_fusions"]["airborne_ensemble"] = _quality_entry(
                air_fused,
                predictions_csv=air_fusion_csv,
                run_dir=run_dir,
            )
            fused_modality_bundles.append(air_fused)
        elif air_dl_csv.exists() and not air_classical_csv.exists():
            # Allow single-model airborne modality when classical inference is unavailable.
            if air_dl_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for airborne DL bundle.")
            air_dl_bundle = _bundle_from_pred_csv(
                air_dl_csv,
                "airborne_dl",
                fusion_mae=air_dl_fusion_mae,
                fusion_mae_source=air_dl_fusion_mae_source,
            )
            batch_quality["models"]["airborne_dl"] = _quality_entry(
                air_dl_bundle, predictions_csv=air_dl_csv, run_dir=run_dir
            )
            air_dl_raw = batch_quality["models"]["airborne_dl"]["mae_mm"]
            apples_to_apples["models"]["airborne_dl"] = _apples_to_apples_entry(
                reference_mae_raw_mm=air_dl_fusion_mae,
                new_batch_mae_raw_mm=air_dl_raw,
                new_batch_mae_snapped_mm=_snapped_mae_mm(air_dl_bundle, 0.1),
                diagnostics=_bundle_pred_diagnostics(air_dl_bundle),
            )
            setup_audit["models"]["airborne_dl"] = _dl_setup_snapshot(
                model_key="airborne_dl",
                model_dir=air_dl_model_dir,
                reference_mae_source=air_dl_fusion_mae_source,
                reference_mae_mm=air_dl_fusion_mae,
                run_dir=run_dir,
            )
            b = _bundle_from_pred_csv(
                air_dl_csv,
                "airborne_ensemble",
                fusion_mae=air_dl_fusion_mae,
                fusion_mae_source=air_dl_fusion_mae_source,
            )
            air_fusion_csv = _save_bundle_predictions_csv(
                b,
                air_fusion_csv_path,
            )
            batch_quality["modality_fusions"]["airborne_ensemble"] = _quality_entry(
                b,
                predictions_csv=air_fusion_csv,
                run_dir=run_dir,
            )
            fused_modality_bundles.append(b)
        elif air_classical_csv.exists() and not air_dl_csv.exists():
            if air_classical_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for airborne classical bundle.")
            air_cls_bundle = _bundle_from_pred_csv(
                air_classical_csv,
                "airborne_classical",
                fusion_mae=air_classical_fusion_mae,
                fusion_mae_source=air_classical_fusion_mae_source,
            )
            batch_quality["models"]["airborne_classical"] = _quality_entry(
                air_cls_bundle, predictions_csv=air_classical_csv, run_dir=run_dir
            )
            air_cls_ref_snap = (
                _read_training_reference_snapped_mae_mm(
                    _classical_model_root_from_bundle_path(air_classical_bundle_path)
                )
                if air_classical_bundle_path is not None
                else None
            )
            air_cls_step = (
                float(air_classical_bundle_obj.get("doe_step_mm"))
                if air_classical_bundle_obj
                and air_classical_bundle_obj.get("doe_step_mm") is not None
                else 0.1
            )
            air_cls_raw = batch_quality["models"]["airborne_classical"]["mae_mm"]
            apples_to_apples["models"]["airborne_classical"] = _apples_to_apples_entry(
                reference_mae_raw_mm=air_classical_fusion_mae,
                reference_mae_snapped_mm=air_cls_ref_snap,
                new_batch_mae_raw_mm=air_cls_raw,
                new_batch_mae_snapped_mm=_snapped_mae_mm(air_cls_bundle, air_cls_step),
                diagnostics=_bundle_pred_diagnostics(air_cls_bundle),
            )
            cls_setup, cls_warn = _classical_setup_snapshot(
                model_key="airborne_classical",
                bundle_path=air_classical_bundle_path,
                bundle_obj=air_classical_bundle_obj,
                reference_mae_source=air_classical_fusion_mae_source,
                reference_mae_mm=air_classical_fusion_mae,
                runtime_pred_csv=air_classical_csv,
                runtime_features_csv=air_features_csv,
                runtime_extraction_replay=air_runtime_extraction_replay,
                run_dir=run_dir,
            )
            setup_audit["models"]["airborne_classical"] = cls_setup
            setup_audit["warnings"].extend(cls_warn)
            b = _bundle_from_pred_csv(
                air_classical_csv,
                "airborne_ensemble",
                fusion_mae=air_classical_fusion_mae,
                fusion_mae_source=air_classical_fusion_mae_source,
            )
            air_fusion_csv = _save_bundle_predictions_csv(
                b,
                air_fusion_csv_path,
            )
            batch_quality["modality_fusions"]["airborne_ensemble"] = _quality_entry(
                b,
                predictions_csv=air_fusion_csv,
                run_dir=run_dir,
            )
            fused_modality_bundles.append(b)

    # Structure (optional, mirrors airborne)
    st_models_cfg = cfg.get("models", {}).get("structure", {})
    st_any_enabled = bool(st_models_cfg.get("classical", {}).get("enabled", False)) or bool(
        st_models_cfg.get("dl", {}).get("enabled", False)
    )
    if seg_dir_str is not None and st_any_enabled:
        st_cfg = cfg["models"]["structure"]

        cls_cfg = st_cfg.get("classical", {})
        dl_cfg = st_cfg.get("dl", {})

        st_classical_csv = run_dir / "structure" / "classical_predictions.csv"
        st_features_csv = run_dir / "structure" / "features_structure.csv"
        st_dl_csv = run_dir / "structure" / "dl_predictions.csv"
        st_fusion_csv_path = run_dir / "structure" / "fusion_predictions.csv"
        st_runtime_extraction_replay: dict[str, Any] | None = None
        st_classical_fusion_mae: float | None = None
        st_classical_fusion_mae_source = ""
        st_classical_bundle_path: Path | None = None
        st_classical_bundle_obj: dict[str, Any] | None = None
        st_dl_fusion_mae: float | None = None
        st_dl_fusion_mae_source = ""
        st_dl_model_dir: Path | None = None

        if cls_cfg.get("enabled", False) and cls_cfg.get("bundle_path"):
            bundle_path = Path(cls_cfg["bundle_path"])
            bundle = joblib.load(bundle_path)
            st_classical_bundle_path = bundle_path
            st_classical_bundle_obj = bundle
            st_classical_fusion_mae, st_classical_fusion_mae_source = _resolve_classical_fusion_mae(
                bundle_path=bundle_path,
                bundle=bundle,
                cfg_mae_fallback=_config_mae_fallback(cls_cfg),
            )
            feat_cols = None
            if "feature_cols" in bundle:
                feat_cols = list(bundle["feature_cols"])
            elif "members" in bundle and bundle["members"]:
                feat_cols = list(bundle["members"][0]["feature_cols"])

            feat_cfg_raw = load_config("configs/structure.yaml")
            feat_cfg_base = feat_cfg_raw.get("classical", feat_cfg_raw)
            feat_cfg, st_runtime_extraction_replay, cfg_replay_warnings = (
                _resolve_structure_runtime_extraction_cfg(
                    bundle_path=bundle_path,
                    bundle_obj=bundle,
                    base_cfg=feat_cfg_base,
                    feature_cols=feat_cols or [],
                )
            )
            logger.info(
                "Structure classical runtime extraction config source=%s | extractor=%s | "
                "target_sr_hz=%s",
                st_runtime_extraction_replay.get("training_extraction_config_source")
                if st_runtime_extraction_replay
                else None,
                feat_cfg.get("extractor"),
                feat_cfg.get("target_sr_hz"),
            )
            for warn_msg in cfg_replay_warnings:
                logger.warning("%s", warn_msg)
            setup_audit["warnings"].extend(cfg_replay_warnings)
            features_df = _extract_for_roots(
                roots=str_roots,
                segments_root=seg_dir_str,
                extractor=extract_structure,
                cfg=feat_cfg,
                file_glob=str(feat_cfg.get("file_glob", STRUCTURE_DEFAULT_FILE_GLOB)),
                n_workers=int(feat_cfg.get("n_workers", STRUCTURE_DEFAULT_N_WORKERS)),
            )
            if feat_cols is not None:
                features_df = _ensure_feature_cols(features_df, feat_cols)
            st_features_csv.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(st_features_csv, index=False)

            infer_classical(
                bundle_path=bundle_path,
                features_csv=st_features_csv,
                out_csv=st_classical_csv,
                snap_predictions=cls_cfg.get("snap_predictions", None),
                doe_step_mm=cls_cfg.get("snap_step_mm", None),
            )

        if dl_cfg.get("enabled", False) and dl_cfg.get("model_dir"):
            st_dl_model_dir = Path(dl_cfg["model_dir"])
            st_dl_fusion_mae, st_dl_fusion_mae_source = _resolve_dl_fusion_mae(
                model_dir=st_dl_model_dir,
                cfg_mae_fallback=_config_mae_fallback(dl_cfg),
            )
            st_split = cfg.get("splitting", {}).get("structure", {})
            _infer_dl_on_segments(
                model_dir=st_dl_model_dir,
                segments_dir=seg_dir_str,
                file_glob=None,
                device=str(dl_cfg.get("device", "auto")),
                batch_size=dl_cfg.get("batch_size", None),
                out_csv=st_dl_csv,
                h5_data_key=str(st_split.get("h5_data_key", "measurement/data")),
                h5_time_key=str(st_split.get("h5_time_key", "measurement/time_vector")),
                include_recording_roots=str_roots,
            )

        if st_classical_csv.exists() and st_dl_csv.exists():
            if st_classical_fusion_mae is None or st_dl_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for structure intra-fusion.")
            st_cls_bundle = _bundle_from_pred_csv(
                st_classical_csv,
                "structure_classical",
                fusion_mae=st_classical_fusion_mae,
                fusion_mae_source=st_classical_fusion_mae_source,
            )
            batch_quality["models"]["structure_classical"] = _quality_entry(
                st_cls_bundle, predictions_csv=st_classical_csv, run_dir=run_dir
            )
            st_cls_ref_snap = (
                _read_training_reference_snapped_mae_mm(
                    _classical_model_root_from_bundle_path(st_classical_bundle_path)
                )
                if st_classical_bundle_path is not None
                else None
            )
            st_cls_step = (
                float(st_classical_bundle_obj.get("doe_step_mm"))
                if st_classical_bundle_obj
                and st_classical_bundle_obj.get("doe_step_mm") is not None
                else 0.1
            )
            st_cls_raw = batch_quality["models"]["structure_classical"]["mae_mm"]
            apples_to_apples["models"]["structure_classical"] = _apples_to_apples_entry(
                reference_mae_raw_mm=st_classical_fusion_mae,
                reference_mae_snapped_mm=st_cls_ref_snap,
                new_batch_mae_raw_mm=st_cls_raw,
                new_batch_mae_snapped_mm=_snapped_mae_mm(st_cls_bundle, st_cls_step),
                diagnostics=_bundle_pred_diagnostics(st_cls_bundle),
            )
            cls_setup, cls_warn = _classical_setup_snapshot(
                model_key="structure_classical",
                bundle_path=st_classical_bundle_path,
                bundle_obj=st_classical_bundle_obj,
                reference_mae_source=st_classical_fusion_mae_source,
                reference_mae_mm=st_classical_fusion_mae,
                runtime_pred_csv=st_classical_csv,
                runtime_features_csv=st_features_csv,
                runtime_extraction_replay=st_runtime_extraction_replay,
                run_dir=run_dir,
            )
            setup_audit["models"]["structure_classical"] = cls_setup
            setup_audit["warnings"].extend(cls_warn)
            st_dl_bundle = _bundle_from_pred_csv(
                st_dl_csv,
                "structure_dl",
                fusion_mae=st_dl_fusion_mae,
                fusion_mae_source=st_dl_fusion_mae_source,
            )
            batch_quality["models"]["structure_dl"] = _quality_entry(
                st_dl_bundle, predictions_csv=st_dl_csv, run_dir=run_dir
            )
            st_dl_raw = batch_quality["models"]["structure_dl"]["mae_mm"]
            apples_to_apples["models"]["structure_dl"] = _apples_to_apples_entry(
                reference_mae_raw_mm=st_dl_fusion_mae,
                new_batch_mae_raw_mm=st_dl_raw,
                new_batch_mae_snapped_mm=_snapped_mae_mm(st_dl_bundle, 0.1),
                diagnostics=_bundle_pred_diagnostics(st_dl_bundle),
            )
            setup_audit["models"]["structure_dl"] = _dl_setup_snapshot(
                model_key="structure_dl",
                model_dir=st_dl_model_dir,
                reference_mae_source=st_dl_fusion_mae_source,
                reference_mae_mm=st_dl_fusion_mae,
                run_dir=run_dir,
            )
            st_fused = fuse_intra_modality(
                st_cls_bundle,
                st_dl_bundle,
                modality_name="structure_ensemble",
                min_weight=fusion_min_weight,
            )
            st_fusion_csv = _save_bundle_predictions_csv(
                st_fused,
                st_fusion_csv_path,
            )
            batch_quality["modality_fusions"]["structure_ensemble"] = _quality_entry(
                st_fused,
                predictions_csv=st_fusion_csv,
                run_dir=run_dir,
            )
            fused_modality_bundles.append(st_fused)
        elif st_dl_csv.exists() and not st_classical_csv.exists():
            # Allow single-model structure modality for now
            if st_dl_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for structure DL bundle.")
            st_dl_bundle = _bundle_from_pred_csv(
                st_dl_csv,
                "structure_dl",
                fusion_mae=st_dl_fusion_mae,
                fusion_mae_source=st_dl_fusion_mae_source,
            )
            batch_quality["models"]["structure_dl"] = _quality_entry(
                st_dl_bundle, predictions_csv=st_dl_csv, run_dir=run_dir
            )
            st_dl_raw = batch_quality["models"]["structure_dl"]["mae_mm"]
            apples_to_apples["models"]["structure_dl"] = _apples_to_apples_entry(
                reference_mae_raw_mm=st_dl_fusion_mae,
                new_batch_mae_raw_mm=st_dl_raw,
                new_batch_mae_snapped_mm=_snapped_mae_mm(st_dl_bundle, 0.1),
                diagnostics=_bundle_pred_diagnostics(st_dl_bundle),
            )
            setup_audit["models"]["structure_dl"] = _dl_setup_snapshot(
                model_key="structure_dl",
                model_dir=st_dl_model_dir,
                reference_mae_source=st_dl_fusion_mae_source,
                reference_mae_mm=st_dl_fusion_mae,
                run_dir=run_dir,
            )
            b = _bundle_from_pred_csv(
                st_dl_csv,
                "structure_ensemble",
                fusion_mae=st_dl_fusion_mae,
                fusion_mae_source=st_dl_fusion_mae_source,
            )
            st_fusion_csv = _save_bundle_predictions_csv(
                b,
                st_fusion_csv_path,
            )
            batch_quality["modality_fusions"]["structure_ensemble"] = _quality_entry(
                b,
                predictions_csv=st_fusion_csv,
                run_dir=run_dir,
            )
            fused_modality_bundles.append(b)
        elif st_classical_csv.exists() and not st_dl_csv.exists():
            if st_classical_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for structure classical bundle.")
            st_cls_bundle = _bundle_from_pred_csv(
                st_classical_csv,
                "structure_classical",
                fusion_mae=st_classical_fusion_mae,
                fusion_mae_source=st_classical_fusion_mae_source,
            )
            batch_quality["models"]["structure_classical"] = _quality_entry(
                st_cls_bundle, predictions_csv=st_classical_csv, run_dir=run_dir
            )
            st_cls_ref_snap = (
                _read_training_reference_snapped_mae_mm(
                    _classical_model_root_from_bundle_path(st_classical_bundle_path)
                )
                if st_classical_bundle_path is not None
                else None
            )
            st_cls_step = (
                float(st_classical_bundle_obj.get("doe_step_mm"))
                if st_classical_bundle_obj
                and st_classical_bundle_obj.get("doe_step_mm") is not None
                else 0.1
            )
            st_cls_raw = batch_quality["models"]["structure_classical"]["mae_mm"]
            apples_to_apples["models"]["structure_classical"] = _apples_to_apples_entry(
                reference_mae_raw_mm=st_classical_fusion_mae,
                reference_mae_snapped_mm=st_cls_ref_snap,
                new_batch_mae_raw_mm=st_cls_raw,
                new_batch_mae_snapped_mm=_snapped_mae_mm(st_cls_bundle, st_cls_step),
                diagnostics=_bundle_pred_diagnostics(st_cls_bundle),
            )
            cls_setup, cls_warn = _classical_setup_snapshot(
                model_key="structure_classical",
                bundle_path=st_classical_bundle_path,
                bundle_obj=st_classical_bundle_obj,
                reference_mae_source=st_classical_fusion_mae_source,
                reference_mae_mm=st_classical_fusion_mae,
                runtime_pred_csv=st_classical_csv,
                runtime_features_csv=st_features_csv,
                runtime_extraction_replay=st_runtime_extraction_replay,
                run_dir=run_dir,
            )
            setup_audit["models"]["structure_classical"] = cls_setup
            setup_audit["warnings"].extend(cls_warn)
            b = _bundle_from_pred_csv(
                st_classical_csv,
                "structure_ensemble",
                fusion_mae=st_classical_fusion_mae,
                fusion_mae_source=st_classical_fusion_mae_source,
            )
            st_fusion_csv = _save_bundle_predictions_csv(
                b,
                st_fusion_csv_path,
            )
            batch_quality["modality_fusions"]["structure_ensemble"] = _quality_entry(
                b,
                predictions_csv=st_fusion_csv,
                run_dir=run_dir,
            )
            fused_modality_bundles.append(b)

    if not fused_modality_bundles:
        print(f"Run complete, but no fused modality bundles were produced. Outputs: {run_dir}")
        return

    final = fuse_modalities(
        *fused_modality_bundles,
        min_weight=fusion_min_weight,
    )
    final_dir = run_dir / "final"
    final_predictions_csv = _save_bundle_predictions_csv(
        final,
        final_dir / "final_predictions.csv",
    )
    final_quality = _quality_entry(
        final,
        predictions_csv=final_predictions_csv,
        run_dir=run_dir,
    )
    setup_audit["run_config_snapshot"] = {
        "final_prediction_config": str(Path(args.config)),
        "only": args.only,
        "force": bool(args.force),
        "splitting": cfg.get("splitting", {}),
        "models": cfg.get("models", {}),
    }

    for mb in fused_modality_bundles:
        q = batch_quality["modality_fusions"].get(mb.modality, None)
        if q is None:
            continue
        raw_mae = q.get("mae_mm", None)
        apples_to_apples["modality_fusions"][mb.modality] = _apples_to_apples_entry(
            reference_mae_raw_mm=float(mb.validation_mae),
            new_batch_mae_raw_mm=raw_mae,
            diagnostics=_bundle_pred_diagnostics(mb),
        )

    batch_quality["final_fusion"] = final_quality
    apples_to_apples["final_fusion"] = _apples_to_apples_entry(
        reference_mae_raw_mm=float(final.validation_mae),
        new_batch_mae_raw_mm=final_quality.get("mae_mm", None),
        diagnostics=_bundle_pred_diagnostics(final),
    )

    lock_artifacts, lock_warnings = _persist_model_setup_lock(
        setup_audit=setup_audit,
        run_dir=run_dir,
        final_dir=final_dir,
    )
    setup_audit["setup_lock_artifacts"] = lock_artifacts
    setup_audit["warnings"].extend(lock_warnings)
    setup_audit["warnings"] = sorted(set(str(w) for w in setup_audit["warnings"]))

    with open(final_dir / "batch_quality_report.json", "w", encoding="utf-8") as fh:
        json.dump(batch_quality, fh, indent=2)
    with open(final_dir / "setup_audit.json", "w", encoding="utf-8") as fh:
        json.dump(setup_audit, fh, indent=2)
    with open(final_dir / "apples_to_apples_report.json", "w", encoding="utf-8") as fh:
        json.dump(apples_to_apples, fh, indent=2)
    if final_quality.get("mae_mm", None) is not None:
        print(f"Final batch MAE: {float(final_quality['mae_mm']):.6f} mm")
    print(f"Batch quality report: {final_dir / 'batch_quality_report.json'}")
    print(f"Setup audit report: {final_dir / 'setup_audit.json'}")
    print(f"Apples-to-apples report: {final_dir / 'apples_to_apples_report.json'}")
    print(f"Model setup lock (run): {lock_artifacts.get('run_lock')}")
    print(f"Model setup lock (latest): {lock_artifacts.get('latest_lock')}")
    print(f"Final fused predictions: {len(final.y_pred)} rows  {final_predictions_csv}")


if __name__ == "__main__":
    main()
