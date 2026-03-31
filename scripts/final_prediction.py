"""vm-predict-fused -- One-shot fused inference from new raw recordings.

Workflow: scan raw folder -> split -> extract features -> infer (classical + DL)
-> fuse intra-modality -> fuse inter-modality -> write output folder.

Config-driven: see ``configs/fusion.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
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
from vm_micro.data.manifest import extract_recording_root, load_doe, try_parse_depth_mm
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-predict-fused",
        description="Run fused prediction on new raw recordings (split->infer->fuse).",
    )
    p.add_argument("--config", default="configs/fusion.yaml")
    p.add_argument(
        "--airborne-input-path",
        default=None,
        help="Needed for the dashboard app. Deafult fallback is config.",
    )
    p.add_argument(
        "--structure-input-path",
        default=None,
        help="Needed for the dashboard app. Deafult fallback is config.",
    )

    p.add_argument("--out-dir", default=None, help="Override run.out_dir.")
    p.add_argument(
        "--only",
        default="both",
        choices=["both", "airborne", "structure"],
        help="Limit processing to one modality.",
    )
    p.add_argument("--force", action="store_true", help="Re-process already-processed files.")
    p.add_argument(
        "mode",
        choices=["batch", "single"],
        help="'batch' splits then infers; 'single' skips splitting.",
    )
    p.add_argument(
        "--actual-depth-mm",
        type=float,
        default=None,
        help="Optional ground-truth depth (mm) for MAE/RMSE reporting.",
    )
    p.add_argument("override", nargs="*", help="YAML config overrides.")
    return p


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RawFile:
    path: Path
    stem: str
    mtime_ns: int


@dataclass
class ModalityResult:
    """Outputs from processing a single modality."""

    fused_bundle: PredictionBundle | None = None
    fused_long_bundle: PredictionBundle | None = None
    model_quality: dict[str, Any] = field(default_factory=dict)
    fusion_quality: dict[str, Any] = field(default_factory=dict)
    model_apples: dict[str, Any] = field(default_factory=dict)
    model_setups: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scanning and DOE
# ---------------------------------------------------------------------------


def _scan_raw(raw_dir: Path, file_glob: str) -> list[RawFile]:
    out: list[RawFile] = []
    for p in sorted(raw_dir.glob(file_glob)):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        out.append(RawFile(path=p, stem=p.stem, mtime_ns=int(st.st_mtime_ns)))
    return out


def _dummy_doe(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Step": list(range(1, n + 1)),
            "HoleID": ["NA"] * n,
            "Depth_mm": [None] * n,
        }
    )


def _expected_map(cfg: dict[str, Any]) -> tuple[dict[str, int], pd.DataFrame | None]:
    exp = cfg.get("splitting", {}).get("expected_segments", {})
    default = int(exp.get("default", 25))
    map_xlsx = exp.get("map_xlsx")
    doe_sheet = str(exp.get("doe_sheet", "DOE_run_order"))
    doe_df: pd.DataFrame | None = None

    if map_xlsx:
        p = Path(str(map_xlsx))
        if p.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
            logger.warning("map_xlsx must be Excel. Got %s, ignoring.", map_xlsx)
        else:
            try:
                doe_df = load_doe(map_xlsx, sheet_name=doe_sheet)
            except Exception as exc:
                logger.warning("Failed to load DOE %s: %s. default=%d.", map_xlsx, exc, default)
            else:
                if len(doe_df) > 0:
                    default = len(doe_df)

    return {"_default": default}, doe_df


def _resolve_expected(mapping: dict[str, int], stem: str) -> int:
    return int(mapping.get(stem, mapping.get("_default", 25)))


def _doe_for_file(doe_template: pd.DataFrame | None, n: int) -> pd.DataFrame:
    n = int(n)
    if doe_template is None:
        return _dummy_doe(n)
    base = doe_template.reset_index(drop=True).copy()
    if n <= len(base):
        return base.iloc[:n].reset_index(drop=True).copy()
    extra = pd.DataFrame(
        {
            "Step": list(range(len(base) + 1, n + 1)),
            "HoleID": ["NA"] * (n - len(base)),
            "Depth_mm": [None] * (n - len(base)),
        }
    )
    return pd.concat([base, extra], ignore_index=True, sort=False)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _ensure_feature_cols(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in required:
        if c not in out.columns:
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
    grouped_by_root: bool = True,
) -> pd.DataFrame:
    if grouped_by_root:
        frames: list[pd.DataFrame] = []
        for root in sorted(roots):
            root_dir = segments_root / root
            if not root_dir.exists():
                logger.warning("Segments missing for %s; skipping.", root)
                continue
            frames.append(
                extractor(root_dir, cfg, out_csv=None, file_glob=file_glob, n_workers=n_workers)
            )
        if not frames:
            raise FileNotFoundError(
                f"No segment folders under {segments_root} for roots={sorted(roots)}"
            )
        return pd.concat(frames, ignore_index=True)

    if not segments_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {segments_root}")
    df = extractor(segments_root, cfg, out_csv=None, file_glob=file_glob, n_workers=n_workers)
    if "record_name" not in df.columns and "recording_root" not in df.columns:
        raise KeyError("Feature extraction must produce 'recording_root' or 'record_name'.")

    for col in ("record_name", "recording_root"):
        if col not in df.columns:
            continue
        out = df[df[col].astype(str).isin(roots)].copy()
        if not out.empty:
            return out.reset_index(drop=True)
    if "record_name" in df.columns:
        out = df[df["record_name"].astype(str).map(extract_recording_root).isin(roots)].copy()
        if not out.empty:
            return out.reset_index(drop=True)

    raise FileNotFoundError(f"No files under {segments_root} for roots={sorted(roots)}")


# ---------------------------------------------------------------------------
# DL inference
# ---------------------------------------------------------------------------


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
    parse_depth_from_filename: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted(data_dir.glob(file_glob)):
        if p.is_dir():
            continue
        recording_root = p.stem.split("__seg")[0] if "__seg" in p.stem else p.stem
        if include_recording_roots is not None and (
            recording_root not in include_recording_roots and p.stem not in include_recording_roots
        ):
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
                "depth_mm": try_parse_depth_mm(p.stem)
                if parse_depth_from_filename
                else float("nan"),
                "class_idx": -1,
                "recording_root": recording_root,
                "parent_dir": p.parent.name,
                "file_group_id": p.stem,
                **meta,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No files under {data_dir} matching {file_glob}")
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
    parse_depth_from_filename: bool = True,
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
        raise FileNotFoundError(f"No best_model.pt under {model_dir}")

    with open(model_dir / "config.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cfg = TrainConfig.from_json_dict(payload)
    if batch_size is not None:
        cfg.batch_size = int(batch_size)
    effective_glob = str(file_glob or cfg.file_glob)
    cfg.data_dir = str(segments_dir)
    cfg.file_glob = effective_glob

    dev = choose_device(device if device != "auto" else cfg.device)
    logger.info("DL inference device: %s", dev)

    file_df = _build_unlabeled_file_df(
        segments_dir,
        effective_glob,
        h5_data_key=h5_data_key,
        h5_time_key=h5_time_key,
        include_recording_roots=include_recording_roots,
        parse_depth_from_filename=parse_depth_from_filename,
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
        file_df,
        cfg,
        training=False,
        h5_data_key=h5_data_key,
        h5_time_key=h5_time_key,
    )
    loader = make_loader(ds, cfg, shuffle=False)
    try:
        window_df = predict_loader(model, loader, dev, cfg)
    except PermissionError as exc:
        logger.warning("Multiprocessing unavailable (%s). Retrying num_workers=0.", exc)
        cfg.num_workers = 0
        loader = make_loader(ds, cfg, shuffle=False)
        window_df = predict_loader(model, loader, dev, cfg)

    file_pred = aggregate_file_predictions(window_df, file_df, cfg, class_to_depth)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    file_pred.to_csv(out_csv, index=False)
    return file_pred


# ---------------------------------------------------------------------------
# Record-key helpers
# ---------------------------------------------------------------------------


def _fusion_key_from_record_name(record_name: str) -> str:
    """Modality-agnostic key for inter-fusion alignment (step + hole from DOE)."""
    parts = str(record_name).split("__")
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
        return str(record_name)
    if step_idx is None:
        return f"hole={hole}"
    if hole is None:
        return f"step={step_idx:03d}"
    return f"step={step_idx:03d}__hole={hole}"


def _canonical_long_record_keys(record_names: np.ndarray) -> np.ndarray:
    """Canonical cross-modality keys for long fusion (run-index + step/hole)."""
    names = [str(x) for x in np.asarray(record_names).tolist()]
    roots = [extract_recording_root(n) for n in names]
    root_rank = {root: i + 1 for i, root in enumerate(sorted(set(roots)))}
    base_keys = [
        f"run={root_rank[root]:03d}__{_fusion_key_from_record_name(name)}"
        for name, root in zip(names, roots)
    ]
    seen: dict[str, int] = {}
    out: list[str] = []
    for base in base_keys:
        idx = seen.get(base, 0) + 1
        seen[base] = idx
        out.append(base if idx == 1 else f"{base}__occ{idx:03d}")
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Prediction bundle I/O
# ---------------------------------------------------------------------------


def _extract_y_true(df: pd.DataFrame) -> tuple[np.ndarray | None, str | None]:
    for col in ("depth_mm", "y_true_depth", "y_true"):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(vals).any():
            return vals, col
    return None, None


def _bundle_from_pred_csv(
    csv_path: Path,
    modality: str,
    *,
    fusion_mae: float,
    fusion_mae_source: str,
    record_key_mode: str = "fusion",
) -> PredictionBundle:
    df = pd.read_csv(csv_path)
    has_real_sigma = "sigma" in df.columns
    sigma = (
        df["sigma"].to_numpy(dtype=np.float64)
        if has_real_sigma
        else np.zeros(len(df), dtype=np.float64)
    )
    y_true, y_true_col = _extract_y_true(df)

    if "record_name" not in df.columns:
        raise KeyError(f"Expected 'record_name' in {csv_path}, got {list(df.columns)}")

    if record_key_mode == "fusion":
        record_keys = df["record_name"].astype(str).map(_fusion_key_from_record_name)
        record_key_desc = "step+hole (parsed from record_name)"
    elif record_key_mode == "record":
        record_keys = df["record_name"].astype(str)
        record_key_desc = "record_name (full)"
    else:
        raise ValueError(f"Unsupported record_key_mode={record_key_mode!r}")

    metadata: dict[str, Any] = {
        "reference_mae_source": fusion_mae_source,
        "record_key": record_key_desc,
        "has_real_sigma": bool(has_real_sigma),
    }
    if y_true_col is not None:
        metadata["batch_ground_truth_column"] = y_true_col

    return PredictionBundle(
        modality=modality,
        record_names=record_keys.to_numpy(),
        y_pred=df["y_pred"].to_numpy(dtype=np.float64),
        sigma=sigma,
        validation_mae=float(fusion_mae),
        y_true=y_true,
        metadata=metadata,
    )


def _confidence_labels(sigma_vals: np.ndarray) -> np.ndarray:
    sig = np.asarray(sigma_vals, dtype=np.float64)
    labels = np.full(sig.shape, None, dtype=object)
    valid = np.isfinite(sig)
    labels[valid & (sig < 0.05)] = "high"
    labels[valid & (sig >= 0.05) & (sig < 0.15)] = "medium"
    labels[valid & (sig >= 0.15)] = "low"
    return labels


def _save_bundle_predictions_csv(bundle: PredictionBundle, out_csv: Path) -> Path:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = bundle.to_dataframe()
    y_pred = pd.to_numeric(df.get("y_pred", pd.Series(dtype=np.float64)), errors="coerce")

    y_true_series: pd.Series | None = None
    for col in ("y_true", "depth_mm", "y_true_depth"):
        if col not in df.columns:
            continue
        cand = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(cand.to_numpy(dtype=np.float64)).any():
            y_true_series = cand
            break

    if y_true_series is not None:
        df["y_true"] = y_true_series
        residual = y_pred - y_true_series
        df["residual_mm"] = residual
        df["abs_residual_mm"] = np.abs(residual)
    else:
        df = df.drop(
            columns=[c for c in ("y_true", "residual_mm", "abs_residual_mm") if c in df.columns],
        )

    has_real_sigma = bool(bundle.metadata.get("has_real_sigma", False))
    if has_real_sigma and "sigma" in df.columns:
        sigma_vals = pd.to_numeric(df["sigma"], errors="coerce").to_numpy(dtype=np.float64)
        df["confidence_label"] = _confidence_labels(sigma_vals)
        if y_true_series is not None and "abs_residual_mm" in df.columns:
            abs_res = pd.to_numeric(
                df["abs_residual_mm"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            valid = np.isfinite(abs_res) & np.isfinite(sigma_vals) & (sigma_vals > 0)
            z_abs = np.full(len(df), np.nan, dtype=np.float64)
            z_abs[valid] = abs_res[valid] / sigma_vals[valid]
            within_1 = np.full(len(df), None, dtype=object)
            within_2 = np.full(len(df), None, dtype=object)
            within_1[valid] = abs_res[valid] <= sigma_vals[valid]
            within_2[valid] = abs_res[valid] <= (2.0 * sigma_vals[valid])
            df["z_abs"] = z_abs
            df["within_1sigma"] = within_1
            df["within_2sigma"] = within_2
    else:
        df = df.drop(
            columns=[
                c
                for c in ("z_abs", "within_1sigma", "within_2sigma", "confidence_label")
                if c in df.columns
            ],
        )

    if bundle.modality == "final_fusion":
        for col in ("sigma_airborne_mm", "sigma_structure_mm", "sigma_between_modalities_mm"):
            vals = bundle.metadata.get(col)
            if isinstance(vals, list) and len(vals) == len(df):
                df[col] = vals

    # Drop redundant columns
    if "residual_mm" in df.columns and "residual" in df.columns:
        df = df.drop(columns=["residual"])
    if "y_true_depth" in df.columns and ("depth_mm" in df.columns or "y_true" in df.columns):
        df = df.drop(columns=["y_true_depth"])

    df.to_csv(out_csv, index=False)
    return out_csv


def _cleanup_single_model_predictions_csv(predictions_csv: Path) -> None:
    """Normalise a single-model predictions CSV: drop sigma, add y_true/residual_mm."""
    if not predictions_csv.exists():
        return
    df = pd.read_csv(predictions_csv)
    if df.empty:
        return

    df = df.drop(columns=["sigma"], errors="ignore")

    y_true_series: pd.Series | None = None
    for col in ("y_true", "depth_mm", "y_true_depth"):
        if col not in df.columns:
            continue
        cand = pd.to_numeric(df[col], errors="coerce")
        if np.isfinite(cand.to_numpy(dtype=np.float64)).any():
            y_true_series = cand
            break

    if y_true_series is not None and "y_pred" in df.columns:
        y_pred = pd.to_numeric(df["y_pred"], errors="coerce")
        residual = y_pred - y_true_series
        base = df.drop(
            columns=["y_true", "residual_mm", "abs_residual_mm", "y_true_depth"],
            errors="ignore",
        )

        extra = pd.DataFrame(
            {
                "y_true": y_true_series.to_numpy(dtype=np.float64),
                "residual_mm": residual.to_numpy(dtype=np.float64),
                "abs_residual_mm": np.abs(residual.to_numpy(dtype=np.float64)),
            },
            index=base.index,
        )

        df = pd.concat([base, extra], axis=1)
    else:
        df = df.drop(
            columns=[
                c
                for c in ("y_true", "y_true_depth", "residual_mm", "abs_residual_mm")
                if c in df.columns
            ],
        )

    df = df.drop(
        columns=[
            c
            for c in ("z_abs", "within_1sigma", "within_2sigma", "confidence_label", "residual")
            if c in df.columns
        ],
    )
    df.to_csv(predictions_csv, index=False)


def _attach_actual_depth_mm_to_predictions_csv(
    predictions_csv: Path,
    actual_depth_mm: float | None,
) -> None:
    if actual_depth_mm is None or not predictions_csv.exists():
        return
    df = pd.read_csv(predictions_csv)
    if df.empty:
        return
    df = df.copy()
    depth = float(actual_depth_mm)
    df["depth_mm"] = depth
    if "residual" in df.columns and "y_pred" in df.columns:
        df["residual"] = depth - pd.to_numeric(df["y_pred"], errors="coerce")
    df.to_csv(predictions_csv, index=False)


def _strip_ground_truth(predictions_csv: Path) -> None:
    if not predictions_csv.exists():
        return
    df = pd.read_csv(predictions_csv)
    cols = [
        c
        for c in (
            "depth_mm",
            "y_true_depth",
            "y_true",
            "residual",
            "residual_mm",
            "abs_residual_mm",
            "z_abs",
            "within_1sigma",
            "within_2sigma",
        )
        if c in df.columns
    ]
    if cols:
        df.drop(columns=cols).to_csv(predictions_csv, index=False)


# ---------------------------------------------------------------------------
# MAE resolution
# ---------------------------------------------------------------------------


def _classical_model_root(bundle_path: Path) -> Path:
    return (
        bundle_path.parent.parent
        if bundle_path.parent.name == "final_model"
        else bundle_path.parent
    )


def _dl_model_root(model_dir: Path) -> Path:
    return model_dir.parent if model_dir.name == "final_model" else model_dir


def _config_mae_fallback(cfg_node: dict[str, Any]) -> float | None:
    raw = cfg_node.get("fusion_mae_fallback", cfg_node.get("validation_mae"))
    return float(raw) if raw is not None else None


def _read_classical_fusion_mae(bundle_path: Path, bundle: dict[str, Any]) -> float:
    model_root = _classical_model_root(bundle_path)
    if "model" in bundle:
        metrics_path = model_root / "final_model" / "test_metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing: {metrics_path}")
        df = pd.read_csv(metrics_path)
        if "holdout_mae_raw" not in df.columns:
            raise KeyError(f"'holdout_mae_raw' missing in {metrics_path}")
        maes = pd.to_numeric(df["holdout_mae_raw"], errors="coerce").to_numpy(dtype=np.float64)
        maes = maes[np.isfinite(maes)]
        if maes.size == 0:
            raise ValueError(f"No finite holdout_mae_raw in {metrics_path}")
        return float(maes[0])

    if "members" in bundle:
        pred_path = model_root / "ensemble_test_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing: {pred_path}")
        df = pd.read_csv(pred_path)
        if "y_pred_raw" not in df.columns:
            raise KeyError(f"'y_pred_raw' missing in {pred_path}")
        target_col = "y_true" if "y_true" in df.columns else "depth_mm"
        if target_col not in df.columns:
            raise KeyError(f"No target column in {pred_path}")
        yt = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=np.float64)
        yp = pd.to_numeric(df["y_pred_raw"], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(yt) & np.isfinite(yp)
        if not np.any(mask):
            raise ValueError(f"No finite pairs in {pred_path}")
        return float(np.mean(np.abs(yt[mask] - yp[mask])))

    raise KeyError(f"Unsupported bundle format: {list(bundle.keys())}")


def _resolve_classical_mae(
    *,
    bundle_path: Path,
    bundle: dict[str, Any],
    cfg_fallback: float | None,
) -> tuple[float, str]:
    try:
        return _read_classical_fusion_mae(bundle_path, bundle), "holdout_artifacts"
    except Exception as exc:
        if cfg_fallback is not None:
            logger.warning("Classical MAE fallback=%s: %s", cfg_fallback, exc)
            return float(cfg_fallback), "config.fusion_mae_fallback"
        raise


def _resolve_dl_mae(
    *,
    model_dir: Path,
    cfg_fallback: float | None,
) -> tuple[float, str]:
    try:
        root = _dl_model_root(model_dir)
        summary_path = root / "repeat_metrics_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        key = "mean_test_mae" if payload.get("mean_test_mae") is not None else "mean_val_mae"
        mae = payload.get(key)
        if mae is None:
            raise KeyError(f"No MAE key in {summary_path}")
        mae = float(mae)
        if not np.isfinite(mae):
            raise ValueError(f"Non-finite {key} in {summary_path}")
        return mae, f"repeat_metrics_summary.json.{key}"
    except Exception as exc:
        if cfg_fallback is not None:
            logger.warning("DL MAE fallback=%s: %s", cfg_fallback, exc)
            return float(cfg_fallback), "config.fusion_mae_fallback"
        raise


# ---------------------------------------------------------------------------
# Extraction config resolution
# ---------------------------------------------------------------------------


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _normalise_structure_extractor(raw: Any) -> str:
    key = str(raw if raw is not None else "v1").strip().lower()
    return "v2" if key == "extensive" else (key if key in {"v1", "v2"} else "v1")


def _infer_structure_extractor(feature_cols: list[str]) -> str | None:
    if not feature_cols:
        return None
    v1 = v2 = 0
    for col in feature_cols:
        c = str(col).strip().lower()
        if c.startswith(("wpd_", "mfcc_", "dmfcc_", "ddmfcc_", "td_", "ss_", "br_", "tf_", "cx_")):
            v2 += 3
        if re.match(r"^cwt_s\d+", c):
            v2 += 2
        if c.startswith(
            ("dwt_", "spectral_", "st_", "peak_freq_", "peak_mag_", "band_power_", "ratio_")
        ):
            v1 += 2
        if c in {"crest_factor", "waveform_length", "percentile_95"}:
            v1 += 1
    if v2 > v1:
        return "v2"
    if v1 > v2:
        return "v1"
    return None


def _resolve_training_features_csv(bundle_path: Path) -> Path | None:
    model_root = _classical_model_root(bundle_path)
    for cfg_path in (
        model_root / "final_model" / "best_model_metadata.json",
        model_root / "run_config.json",
    ):
        payload = _safe_read_json(cfg_path)
        if payload and payload.get("features_csv"):
            p_str = str(payload["features_csv"])
            p = Path(p_str) if Path(p_str).is_absolute() else Path.cwd() / p_str
            if p.exists():
                return p
    return None


def _read_training_sr_hz(features_csv: Path) -> int | None:
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
        mode = pd.Series(np.rint(vals).astype(np.int64)).mode()
        if not mode.empty and int(mode.iloc[0]) > 0:
            return int(mode.iloc[0])
    return None


def _resolve_extraction_cfg(
    *,
    bundle_path: Path,
    bundle_obj: dict[str, Any],
    base_cfg: dict[str, Any],
    modality: str,
    feature_cols: list[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Resolve runtime extraction config from training artifacts."""
    cfg = dict(base_cfg)
    warnings: list[str] = []
    info: dict[str, Any] = {"training_extraction_config_source": None}

    model_root = _classical_model_root(bundle_path)
    for source, candidate in [
        (
            "bundle",
            bundle_obj.get("feature_extraction_config")
            if isinstance(bundle_obj.get("feature_extraction_config"), dict)
            else None,
        ),
        (
            "best_model_metadata",
            (_safe_read_json(model_root / "final_model" / "best_model_metadata.json") or {}).get(
                "feature_extraction_config"
            ),
        ),
        (
            "run_config",
            (_safe_read_json(model_root / "run_config.json") or {}).get(
                "feature_extraction_config"
            ),
        ),
    ]:
        if isinstance(candidate, dict) and candidate:
            cfg.update(candidate)
            info["training_extraction_config_source"] = source
            break

    if info["training_extraction_config_source"] is None:
        warnings.append(f"No extraction config sidecar for {modality}; using current YAML config.")

    if modality == "structure":
        if info["training_extraction_config_source"] is None:
            inferred = _infer_structure_extractor(feature_cols)
            cfg["extractor"] = inferred or _normalise_structure_extractor(cfg.get("extractor"))
        else:
            cfg["extractor"] = _normalise_structure_extractor(cfg.get("extractor"))

    training_csv = _resolve_training_features_csv(bundle_path)
    if training_csv:
        sr_hz = _read_training_sr_hz(training_csv)
        if sr_hz is not None:
            sr_key = "target_sr_hz" if modality == "structure" else "target_sr"
            cfg[sr_key] = int(sr_hz)
            info["training_sr_hz"] = int(sr_hz)

    return cfg, info, warnings


# ---------------------------------------------------------------------------
# Quality / metrics helpers
# ---------------------------------------------------------------------------


def _safe_rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _quality_entry(
    bundle: PredictionBundle,
    *,
    predictions_csv: Path,
    run_dir: Path,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "predictions_csv": _safe_rel(predictions_csv, run_dir),
        **bundle_batch_metrics(bundle),
    }
    has_real_sigma = bool(bundle.metadata.get("has_real_sigma", False))
    if not has_real_sigma:
        return out

    sigma_vals = np.asarray(bundle.sigma, dtype=np.float64)
    sigma_valid = sigma_vals[np.isfinite(sigma_vals)]
    if len(sigma_valid):
        out["sigma_mean_mm"] = float(np.mean(sigma_valid))

    if bundle.y_true is not None:
        yp = np.asarray(bundle.y_pred, dtype=np.float64)
        yt = np.asarray(bundle.y_true, dtype=np.float64)
        sg = np.asarray(bundle.sigma, dtype=np.float64)
        mask = np.isfinite(yp) & np.isfinite(yt) & np.isfinite(sg) & (sg > 0)
        if np.any(mask):
            abs_res = np.abs(yp[mask] - yt[mask])
            out["coverage_1sigma"] = float(np.mean(abs_res <= sg[mask]))
            out["coverage_2sigma"] = float(np.mean(abs_res <= (2.0 * sg[mask])))
    return out


def _bundle_diagnostics(bundle: PredictionBundle) -> dict[str, Any]:
    yp = np.asarray(bundle.y_pred, dtype=np.float64)
    if bundle.y_true is None:
        return {"bias_mm": None, "mean_pred_mm": float(np.mean(yp)) if len(yp) else None}
    yt = np.asarray(bundle.y_true, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        return {"bias_mm": None, "mean_pred_mm": None}
    out: dict[str, Any] = {"bias_mm": float(np.mean(yp[mask] - yt[mask]))}
    if mask.sum() >= 2:
        out["corr_true_pred"] = float(np.corrcoef(yt[mask], yp[mask])[0, 1])
    return out


def _apples_entry(
    *,
    ref_mae: float | None,
    batch_mae: float | None,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    delta = (batch_mae - ref_mae) if ref_mae is not None and batch_mae is not None else None
    ratio = (batch_mae / ref_mae) if delta is not None and ref_mae and ref_mae > 0 else None
    return {
        "reference_mae_raw_mm": ref_mae,
        "new_batch_mae_raw_mm": batch_mae,
        "raw_mae_delta_mm": delta,
        "raw_mae_ratio": ratio,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Setup audit / lock
# ---------------------------------------------------------------------------


def _sha256_file(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundle_feature_cols(bundle_obj: dict[str, Any] | None) -> list[str]:
    if not bundle_obj:
        return []
    if "feature_cols" in bundle_obj:
        cols = bundle_obj["feature_cols"]
        return [str(c) for c in cols] if isinstance(cols, list) else []
    members = bundle_obj.get("members")
    if isinstance(members, list) and members and isinstance(members[0], dict):
        cols = members[0].get("feature_cols")
        return [str(c) for c in cols] if isinstance(cols, list) else []
    return []


def _classical_setup_snapshot(
    *,
    model_key: str,
    bundle_path: Path,
    bundle_obj: dict[str, Any],
    reference_mae_source: str,
    reference_mae_mm: float | None,
    run_dir: Path,
) -> dict[str, Any]:
    feature_cols = _bundle_feature_cols(bundle_obj)
    return {
        "model_kind": "classical",
        "bundle_path": _safe_rel(bundle_path, run_dir),
        "bundle_path_sha256": _sha256_file(bundle_path),
        "reference_mae_raw_mm": reference_mae_mm,
        "reference_mae_source": reference_mae_source,
        "model_feature_signature": {
            "n_features": len(feature_cols),
            "sha256": hashlib.sha256("|".join(feature_cols).encode()).hexdigest()
            if feature_cols
            else None,
        },
    }


def _dl_setup_snapshot(
    *,
    model_dir: Path | None,
    reference_mae_source: str,
    reference_mae_mm: float | None,
    run_dir: Path,
) -> dict[str, Any]:
    if model_dir is None:
        return {"model_kind": "dl", "model_dir": None, "reference_mae_raw_mm": reference_mae_mm}
    root = _dl_model_root(model_dir)
    cfg_path = root / "final_model" / "config.json"
    if not cfg_path.exists():
        cfg_path = root / "config.json"
    return {
        "model_kind": "dl",
        "model_dir": _safe_rel(model_dir, run_dir),
        "config_sha256": _sha256_file(cfg_path),
        "reference_mae_raw_mm": reference_mae_mm,
        "reference_mae_source": reference_mae_source,
    }


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

    # Build lock payload
    models_obj = setup_audit.get("models", {})
    lock_models: dict[str, Any] = {}
    if isinstance(models_obj, dict):
        for key in sorted(models_obj):
            snap = models_obj[key]
            if not isinstance(snap, dict):
                continue
            entry: dict[str, Any] = {
                k: snap[k]
                for k in (
                    "model_kind",
                    "bundle_path",
                    "bundle_path_sha256",
                    "config_sha256",
                    "model_dir",
                    "reference_mae_raw_mm",
                    "reference_mae_source",
                )
                if k in snap and snap[k] is not None
            }
            sig = snap.get("model_feature_signature", {})
            if isinstance(sig, dict):
                entry["model_feature_signature_n_features"] = sig.get("n_features")
                entry["model_feature_signature_sha256"] = sig.get("sha256")
            lock_models[key] = entry

    run_cfg = setup_audit.get("run_config_snapshot", {})
    lock_payload = {
        "description": "Model artifact lock for fused prediction runs.",
        "run_name": run_dir.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(run_dir),
        "final_prediction_config": (
            run_cfg.get("final_prediction_config") if isinstance(run_cfg, dict) else None
        ),
        "models": lock_models,
    }

    # Drift check
    previous_lock = _safe_read_json(latest_lock_path)
    drift_warnings: list[str] = []
    if isinstance(previous_lock, dict):
        prev_m = previous_lock.get("models", {})
        cur_m = lock_payload["models"]
        for k in sorted(set(prev_m) | set(cur_m)):
            if k not in prev_m:
                drift_warnings.append(f"setup-lock drift: model '{k}' added.")
            elif k not in cur_m:
                drift_warnings.append(f"setup-lock drift: model '{k}' missing.")
            else:
                for f in ("bundle_path_sha256", "config_sha256", "model_feature_signature_sha256"):
                    pv, cv = prev_m[k].get(f), cur_m[k].get(f)
                    if pv is not None and cv is not None and str(pv) != str(cv):
                        drift_warnings.append(f"setup-lock drift: {k}.{f} changed.")

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


# ---------------------------------------------------------------------------
# Copy debug plots
# ---------------------------------------------------------------------------


def _copy_split_debug_plots_to_run_dir(
    summary: dict[str, Any],
    run_split_dir: Path,
) -> dict[str, str]:
    run_split_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key in ("debug_core", "debug_padded"):
        raw_src = summary.get(key)
        if not raw_src:
            continue
        src = Path(str(raw_src))
        if not src.exists():
            logger.warning("Missing split debug plot (%s): %s", key, src)
            continue
        dst = run_split_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        copied[key] = str(dst)
    return copied


# ---------------------------------------------------------------------------
# Single-prediction report
# ---------------------------------------------------------------------------


def _single_prediction_report_payload(
    *,
    run_dir: Path,
    final_predictions_csv: Path,
    final_quality: dict[str, Any],
    batch_quality: dict[str, Any],
    actual_depth_mm: float | None,
) -> dict[str, Any]:
    final_df = pd.read_csv(final_predictions_csv)
    pred_cols = [
        c
        for c in (
            "record_name",
            "y_pred",
            "sigma",
            "confidence_label",
            "depth_mm",
            "y_true",
            "residual_mm",
            "abs_residual_mm",
            "z_abs",
            "within_1sigma",
            "within_2sigma",
            "sigma_airborne_mm",
            "sigma_structure_mm",
            "sigma_between_modalities_mm",
        )
        if c in final_df
    ]

    y_pred = pd.to_numeric(
        final_df.get("y_pred", pd.Series(dtype=float)),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    y_pred = y_pred[np.isfinite(y_pred)]

    return {
        "description": "Single-mode prediction report.",
        "mode": "single",
        "actual_depth_mm": actual_depth_mm,
        "models": batch_quality.get("models", {}),
        "modality_fusions": batch_quality.get("modality_fusions", {}),
        "final_prediction": {
            **final_quality,
            "prediction_summary": {
                "n_predictions": len(final_df),
                "y_pred_mean_mm": float(np.mean(y_pred)) if len(y_pred) else None,
                "y_pred_min_mm": float(np.min(y_pred)) if len(y_pred) else None,
                "y_pred_max_mm": float(np.max(y_pred)) if len(y_pred) else None,
            },
            "predictions": final_df[pred_cols].to_dict(orient="records"),
            "predictions_csv": _safe_rel(final_predictions_csv, run_dir),
        },
    }


# ---------------------------------------------------------------------------
# Modality defaults
# ---------------------------------------------------------------------------

_MODALITY_DEFAULTS = {
    "airborne": {
        "extractor": extract_airborne,
        "default_file_glob": AIRBORNE_DEFAULT_FILE_GLOB,
        "default_n_workers": AIRBORNE_DEFAULT_N_WORKERS,
        "config_yaml": "configs/airborne.yaml",
        "h5_data_key": "measurement/data",
        "h5_time_key": "measurement/time_vector",
        "default_enabled": True,
    },
    "structure": {
        "extractor": extract_structure,
        "default_file_glob": STRUCTURE_DEFAULT_FILE_GLOB,
        "default_n_workers": STRUCTURE_DEFAULT_N_WORKERS,
        "config_yaml": "configs/structure.yaml",
        "h5_data_key": "measurement/data",
        "h5_time_key": "measurement/time_vector",
        "default_enabled": False,
    },
}


# ---------------------------------------------------------------------------
# Generic modality processing
# ---------------------------------------------------------------------------


def _process_modality(
    *,
    modality: str,
    seg_dir: Path,
    roots: set[str],
    models_cfg: dict[str, Any],
    split_cfg: dict[str, Any],
    run_dir: Path,
    mode: str,
    actual_depth_mm: float | None,
    fusion_min_weight: float,
) -> ModalityResult:
    """Process classical + DL inference and intra-fusion for one modality."""
    result = ModalityResult()
    defaults = _MODALITY_DEFAULTS[modality]
    mod_dir = run_dir / modality
    cls_cfg = models_cfg.get("classical", {})
    dl_cfg = models_cfg.get("dl", {})
    is_batch = mode == "batch"
    default_enabled = defaults["default_enabled"]

    h5_data_key = str(split_cfg.get("h5_data_key", defaults["h5_data_key"]))
    h5_time_key = str(split_cfg.get("h5_time_key", defaults["h5_time_key"]))

    classical_csv = mod_dir / "classical_predictions.csv"
    dl_csv = mod_dir / "dl_predictions.csv"
    features_csv = mod_dir / f"features_{modality}.csv"
    cls_mae: float | None = None
    cls_mae_source = ""
    cls_bundle_path: Path | None = None
    cls_bundle_obj: dict[str, Any] | None = None
    dl_mae: float | None = None
    dl_mae_source = ""
    dl_model_dir: Path | None = None

    # --- Classical inference ---
    if cls_cfg.get("enabled", default_enabled) and cls_cfg.get("bundle_path"):
        bundle_path = Path(cls_cfg["bundle_path"])
        bundle = joblib.load(bundle_path)
        cls_bundle_path = bundle_path
        cls_bundle_obj = bundle
        cls_mae, cls_mae_source = _resolve_classical_mae(
            bundle_path=bundle_path,
            bundle=bundle,
            cfg_fallback=_config_mae_fallback(cls_cfg),
        )
        feat_cols = _bundle_feature_cols(bundle)
        feat_cfg_raw = load_config(defaults["config_yaml"])
        feat_cfg_base = feat_cfg_raw.get("classical", feat_cfg_raw)
        feat_cfg, _info, cfg_warnings = _resolve_extraction_cfg(
            bundle_path=bundle_path,
            bundle_obj=bundle,
            base_cfg=feat_cfg_base,
            modality=modality,
            feature_cols=feat_cols,
        )
        result.warnings.extend(cfg_warnings)

        features_df = _extract_for_roots(
            roots=roots,
            segments_root=seg_dir,
            extractor=defaults["extractor"],
            cfg=feat_cfg,
            file_glob=str(feat_cfg.get("file_glob", defaults["default_file_glob"])),
            n_workers=int(feat_cfg.get("n_workers", defaults["default_n_workers"])),
            grouped_by_root=is_batch,
        )
        if not is_batch and "depth_mm" in features_df.columns:
            features_df = features_df.drop(columns=["depth_mm"], errors="ignore")
        if feat_cols:
            features_df = _ensure_feature_cols(features_df, feat_cols)
        features_csv.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(features_csv, index=False)

        infer_classical(
            bundle_path=bundle_path,
            features_csv=features_csv,
            out_csv=classical_csv,
            snap_predictions=cls_cfg.get("snap_predictions"),
            doe_step_mm=cls_cfg.get("snap_step_mm"),
        )
        if not is_batch and actual_depth_mm is None:
            _strip_ground_truth(classical_csv)
        else:
            _attach_actual_depth_mm_to_predictions_csv(classical_csv, actual_depth_mm)
        _cleanup_single_model_predictions_csv(classical_csv)

    # --- DL inference ---
    if dl_cfg.get("enabled", default_enabled) and dl_cfg.get("model_dir"):
        dl_model_dir = Path(dl_cfg["model_dir"])
        dl_mae, dl_mae_source = _resolve_dl_mae(
            model_dir=dl_model_dir,
            cfg_fallback=_config_mae_fallback(dl_cfg),
        )
        _infer_dl_on_segments(
            model_dir=dl_model_dir,
            segments_dir=seg_dir,
            file_glob=None,
            device=str(dl_cfg.get("device", "auto")),
            batch_size=dl_cfg.get("batch_size"),
            out_csv=dl_csv,
            h5_data_key=h5_data_key,
            h5_time_key=h5_time_key,
            include_recording_roots=roots,
            parse_depth_from_filename=is_batch,
        )
        if not is_batch and actual_depth_mm is None:
            _strip_ground_truth(dl_csv)
        else:
            _attach_actual_depth_mm_to_predictions_csv(dl_csv, actual_depth_mm)
        _cleanup_single_model_predictions_csv(dl_csv)

    # --- Intra-modality fusion ---
    has_cls = classical_csv.exists()
    has_dl = dl_csv.exists()
    if not has_cls and not has_dl:
        return result

    ensemble_name = f"{modality}_ensemble"

    # Build bundles for available models
    bundles_short: list[PredictionBundle] = []
    bundles_long: list[PredictionBundle] = []

    if has_cls:
        if cls_mae is None:
            raise RuntimeError(f"Missing MAE for {modality} classical.")
        b = _bundle_from_pred_csv(
            classical_csv,
            f"{modality}_classical",
            fusion_mae=cls_mae,
            fusion_mae_source=cls_mae_source,
        )
        bundles_short.append(b)
        bundles_long.append(
            _bundle_from_pred_csv(
                classical_csv,
                f"{modality}_classical",
                fusion_mae=cls_mae,
                fusion_mae_source=cls_mae_source,
                record_key_mode="record",
            )
        )
        result.model_quality[f"{modality}_classical"] = _quality_entry(
            b,
            predictions_csv=classical_csv,
            run_dir=run_dir,
        )
        result.model_apples[f"{modality}_classical"] = _apples_entry(
            ref_mae=cls_mae,
            batch_mae=result.model_quality[f"{modality}_classical"]["mae_mm"],
            diagnostics=_bundle_diagnostics(b),
        )
        if cls_bundle_path:
            result.model_setups[f"{modality}_classical"] = _classical_setup_snapshot(
                model_key=f"{modality}_classical",
                bundle_path=cls_bundle_path,
                bundle_obj=cls_bundle_obj,
                reference_mae_source=cls_mae_source,
                reference_mae_mm=cls_mae,
                run_dir=run_dir,
            )

    if has_dl:
        if dl_mae is None:
            raise RuntimeError(f"Missing MAE for {modality} DL.")
        b = _bundle_from_pred_csv(
            dl_csv,
            f"{modality}_dl",
            fusion_mae=dl_mae,
            fusion_mae_source=dl_mae_source,
        )
        bundles_short.append(b)
        bundles_long.append(
            _bundle_from_pred_csv(
                dl_csv,
                f"{modality}_dl",
                fusion_mae=dl_mae,
                fusion_mae_source=dl_mae_source,
                record_key_mode="record",
            )
        )
        result.model_quality[f"{modality}_dl"] = _quality_entry(
            b,
            predictions_csv=dl_csv,
            run_dir=run_dir,
        )
        result.model_apples[f"{modality}_dl"] = _apples_entry(
            ref_mae=dl_mae,
            batch_mae=result.model_quality[f"{modality}_dl"]["mae_mm"],
            diagnostics=_bundle_diagnostics(b),
        )
        result.model_setups[f"{modality}_dl"] = _dl_setup_snapshot(
            model_dir=dl_model_dir,
            reference_mae_source=dl_mae_source,
            reference_mae_mm=dl_mae,
            run_dir=run_dir,
        )

    # Fuse or pass-through
    if len(bundles_short) == 2:
        fused_short = fuse_intra_modality(
            bundles_short[0],
            bundles_short[1],
            ensemble_name,
            min_weight=fusion_min_weight,
        )
        fused_long = fuse_intra_modality(
            bundles_long[0],
            bundles_long[1],
            ensemble_name,
            min_weight=fusion_min_weight,
        )
    else:
        src = bundles_short[0]
        fused_short = PredictionBundle(
            modality=ensemble_name,
            record_names=src.record_names.copy(),
            y_pred=src.y_pred.copy(),
            sigma=np.zeros_like(src.y_pred, dtype=np.float64),
            validation_mae=src.validation_mae,
            y_true=src.y_true.copy() if src.y_true is not None else None,
            metadata={**src.metadata, "has_real_sigma": False},
        )
        src_l = bundles_long[0]
        fused_long = PredictionBundle(
            modality=ensemble_name,
            record_names=src_l.record_names.copy(),
            y_pred=src_l.y_pred.copy(),
            sigma=np.zeros_like(src_l.y_pred, dtype=np.float64),
            validation_mae=src_l.validation_mae,
            y_true=src_l.y_true.copy() if src_l.y_true is not None else None,
            metadata={**src_l.metadata, "has_real_sigma": False},
        )

    fusion_csv = _save_bundle_predictions_csv(
        fused_short,
        mod_dir / "fusion_predictions.csv",
    )
    _save_bundle_predictions_csv(fused_long, mod_dir / "fusion_predictions_long.csv")
    result.fusion_quality[ensemble_name] = _quality_entry(
        fused_short,
        predictions_csv=fusion_csv,
        run_dir=run_dir,
    )
    result.fused_bundle = fused_short
    result.fused_long_bundle = fused_long
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    import numpy as _np

    if sys.version_info < (3, 13) or int(_np.__version__.split(".", 1)[0]) < 2:
        raise RuntimeError(
            f"Environment mismatch: Python {sys.version_info.major}."
            f"{sys.version_info.minor}, NumPy {_np.__version__}. "
            "Requires Python >= 3.13 and NumPy >= 2.0."
        )

    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)
    if args.out_dir:
        cfg.setdefault("run", {})["out_dir"] = args.out_dir

    if args.airborne_input_path or args.structure_input_path:
        if args.airborne_input_path:
            airborne_path = Path(args.airborne_input_path).expanduser().resolve()
            if not airborne_path.is_file():
                raise FileNotFoundError(f"--airborne-input-path does not exist: {airborne_path}")
            if airborne_path.suffix.lower() not in {".flac", ".wav"}:
                raise ValueError(f"Unsupported airborne suffix: {airborne_path.suffix.lower()}")
            cfg["inputs"]["airborne"]["enabled"] = True
            cfg["inputs"]["airborne"]["raw_dir"] = str(airborne_path.parent)
            cfg["inputs"]["airborne"]["file_glob"] = airborne_path.name
        else:
            cfg["inputs"]["airborne"]["enabled"] = False

        if args.structure_input_path:
            structure_path = Path(args.structure_input_path).expanduser().resolve()
            if not structure_path.is_file():
                raise FileNotFoundError(f"--structure-input-path does not exist: {structure_path}")
            if structure_path.suffix.lower() not in {".h5", ".hdf5"}:
                raise ValueError(f"Unsupported structure suffix: {structure_path.suffix.lower()}")
            cfg["inputs"]["structure"]["enabled"] = True
            cfg["inputs"]["structure"]["raw_dir"] = str(structure_path.parent)
            cfg["inputs"]["structure"]["file_glob"] = structure_path.name
        else:
            cfg["inputs"]["structure"]["enabled"] = False

    run_out_root = Path(cfg.get("run", {}).get("out_dir", "data/fusion_results"))
    state_path = Path(
        cfg.get("run", {}).get(
            "state_json",
            "data/fusion_results/final_prediction_state.json",
        )
    )
    run_tag = str(cfg.get("run", {}).get("tag", "manual"))
    run_dir = run_out_root / f"{_now_tag()}__{run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    state = _load_state(state_path)
    processed: dict[str, Any] = state.setdefault("processed", {})
    exp_map, doe_template = _expected_map(cfg)

    # Collect per-modality worklists
    modality_work: dict[str, tuple[list[RawFile], Path]] = {}
    for mod in ("airborne", "structure"):
        if args.only not in {"both", mod}:
            continue
        inp = cfg.get("inputs", {}).get(mod, {})
        if not inp.get("enabled", True):
            continue
        raw_dir = Path(inp["raw_dir"])
        glob = str(inp.get("file_glob", "**/*.flac" if mod == "airborne" else "**/*.h5"))
        work = [
            rf
            for rf in _scan_raw(raw_dir, glob)
            if args.force
            or str(rf.path.resolve()) not in processed
            or processed[str(rf.path.resolve())].get("mtime_ns") != rf.mtime_ns
        ]
        if work:
            modality_work[mod] = (work, raw_dir)

    if not modality_work:
        print("No new raw files detected (or all already processed).")
        return

    # Splitting
    seg_dirs: dict[str, Path] = {}
    roots_map: dict[str, set[str]] = {}

    if args.mode == "batch":
        split_cfg = cfg.get("splitting", {})
        common = split_cfg.get("common", {})
        band_hz = (
            float(common.get("band_low_hz", 2000.0)),
            float(common.get("band_high_hz", 5000.0)),
        )
        common_fallbacks = common.get("band_fallbacks_hz")

        for mod, (work, _) in modality_work.items():
            mod_split = split_cfg.get(mod, {})
            seg_root = Path(
                split_cfg.get("segments_root", {}).get(
                    mod,
                    f"data/raw_data_extracted_splits/{mod[:3]}/live",
                )
            )
            pre_pad = float(mod_split.get("pre_pad_s", common.get("pre_pad_s", 0.20)))
            post_pad = float(mod_split.get("post_pad_s", common.get("post_pad_s", 0.25)))
            fallbacks = mod_split.get("band_fallbacks_hz", common_fallbacks)

            for rf in work:
                target = seg_root / rf.stem
                if target.exists():
                    shutil.rmtree(target)
                expected = _resolve_expected(exp_map, rf.stem)
                doe_df = _doe_for_file(doe_template, expected)
                manifest_df, summary = process_one_file(
                    rf.path,
                    doe_df,
                    seg_root,
                    expected_segments=expected,
                    pre_pad_s=pre_pad,
                    post_pad_s=post_pad,
                    band_hz=band_hz,
                    band_hz_fallbacks=fallbacks,
                    export_format=str(
                        mod_split.get(
                            "export_format",
                            "flac" if mod == "airborne" else "h5",
                        )
                    ),
                    h5_data_key=str(mod_split.get("h5_data_key", "measurement/data")),
                    h5_time_key=str(mod_split.get("h5_time_key", "measurement/time_vector")),
                    target_sr=mod_split.get("target_sr"),
                )
                manifest_out = run_dir / mod / rf.stem / "segments_manifest.csv"
                manifest_out.parent.mkdir(parents=True, exist_ok=True)
                _copy_split_debug_plots_to_run_dir(summary, manifest_out.parent)
                manifest_df.to_csv(manifest_out, index=False)
                processed[str(rf.path.resolve())] = {
                    "mtime_ns": rf.mtime_ns,
                    "modality": mod,
                    "mode": "batch",
                    "stem": rf.stem,
                }
                logger.info(
                    "Split %s %s -> %d segments",
                    mod,
                    rf.path.name,
                    summary["exported_segments_final"],
                )
            seg_dirs[mod] = seg_root
            roots_map[mod] = {rf.stem for rf in work}
    else:
        for mod, (work, raw_dir) in modality_work.items():
            seg_dirs[mod] = raw_dir
            roots_map[mod] = {rf.stem for rf in work}
            for rf in work:
                processed[str(rf.path.resolve())] = {
                    "mtime_ns": rf.mtime_ns,
                    "modality": mod,
                    "mode": "single",
                    "stem": rf.stem,
                }

    _save_state(state_path, state)

    # Per-modality inference + fusion
    fused_bundles: list[PredictionBundle] = []
    fused_long_bundles: list[PredictionBundle] = []
    batch_quality: dict[str, Any] = {
        "description": "Performance on this run's batch.",
        "models": {},
        "modality_fusions": {},
        "final_fusion": None,
    }
    setup_audit: dict[str, Any] = {
        "description": "Model + data setup snapshot.",
        "models": {},
        "warnings": [],
    }
    apples: dict[str, Any] = {
        "description": "Reference vs batch MAE comparison.",
        "models": {},
        "modality_fusions": {},
        "final_fusion": None,
    }
    fusion_min_weight = float(
        cfg.get("models", {}).get("fusion", {}).get("min_weight", 0.05),
    )

    for mod in ("airborne", "structure"):
        if mod not in seg_dirs:
            continue
        mod_cfg = cfg.get("models", {}).get(mod, {})
        default_en = _MODALITY_DEFAULTS[mod]["default_enabled"]
        if not (
            mod_cfg.get("classical", {}).get("enabled", default_en)
            or mod_cfg.get("dl", {}).get("enabled", default_en)
        ):
            continue

        result = _process_modality(
            modality=mod,
            seg_dir=seg_dirs[mod],
            roots=roots_map[mod],
            models_cfg=mod_cfg,
            split_cfg=cfg.get("splitting", {}).get(mod, {}),
            run_dir=run_dir,
            mode=args.mode,
            actual_depth_mm=args.actual_depth_mm,
            fusion_min_weight=fusion_min_weight,
        )
        batch_quality["models"].update(result.model_quality)
        batch_quality["modality_fusions"].update(result.fusion_quality)
        setup_audit["models"].update(result.model_setups)
        setup_audit["warnings"].extend(result.warnings)
        apples["models"].update(result.model_apples)

        if result.fused_bundle is not None:
            fused_bundles.append(result.fused_bundle)
            q = result.fusion_quality.get(result.fused_bundle.modality, {})
            apples["modality_fusions"][result.fused_bundle.modality] = _apples_entry(
                ref_mae=float(result.fused_bundle.validation_mae),
                batch_mae=q.get("mae_mm"),
                diagnostics=_bundle_diagnostics(result.fused_bundle),
            )
        if result.fused_long_bundle is not None:
            fused_long_bundles.append(result.fused_long_bundle)

    if not fused_bundles:
        print(f"No fused modality bundles produced. Outputs: {run_dir}")
        return

    # Inter-modality fusion
    final = fuse_modalities(*fused_bundles, min_weight=fusion_min_weight)
    final_dir = run_dir / "final"
    final_csv = _save_bundle_predictions_csv(final, final_dir / "final_predictions.csv")

    final_long_csv: Path | None = None
    if fused_long_bundles:
        long_for_fusion = (
            [
                PredictionBundle(
                    modality=b.modality,
                    record_names=_canonical_long_record_keys(b.record_names),
                    y_pred=b.y_pred.copy(),
                    sigma=b.sigma.copy(),
                    validation_mae=float(b.validation_mae),
                    y_true=b.y_true.copy() if b.y_true is not None else None,
                    metadata=dict(b.metadata),
                )
                for b in fused_long_bundles
            ]
            if len(fused_long_bundles) > 1
            else fused_long_bundles
        )
        final_long = fuse_modalities(*long_for_fusion, min_weight=fusion_min_weight)
        final_long_csv = _save_bundle_predictions_csv(
            final_long,
            final_dir / "final_predictions_long.csv",
        )

    final_quality = _quality_entry(final, predictions_csv=final_csv, run_dir=run_dir)
    batch_quality["final_fusion"] = final_quality
    apples["final_fusion"] = _apples_entry(
        ref_mae=float(final.validation_mae),
        batch_mae=final_quality.get("mae_mm"),
        diagnostics=_bundle_diagnostics(final),
    )

    setup_audit["run_config_snapshot"] = {
        "final_prediction_config": str(Path(args.config)),
        "mode": args.mode,
        "only": args.only,
        "force": bool(args.force),
        "actual_depth_mm": args.actual_depth_mm,
    }

    lock_artifacts, lock_warnings = _persist_model_setup_lock(
        setup_audit=setup_audit,
        run_dir=run_dir,
        final_dir=final_dir,
    )
    setup_audit["setup_lock_artifacts"] = lock_artifacts
    setup_audit["warnings"].extend(lock_warnings)
    setup_audit["warnings"] = sorted(set(str(w) for w in setup_audit["warnings"]))

    with open(final_dir / "setup_audit.json", "w", encoding="utf-8") as fh:
        json.dump(setup_audit, fh, indent=2)

    if args.mode == "single":
        report = _single_prediction_report_payload(
            run_dir=run_dir,
            final_predictions_csv=final_csv,
            final_quality=final_quality,
            batch_quality=batch_quality,
            actual_depth_mm=args.actual_depth_mm,
        )
        with open(final_dir / "single_prediction_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        if final_quality.get("mae_mm") is not None:
            print(f"Final single MAE: {float(final_quality['mae_mm']):.6f} mm")
        print(f"Report: {final_dir / 'single_prediction_report.json'}")
    else:
        with open(final_dir / "batch_quality_report.json", "w", encoding="utf-8") as fh:
            json.dump(batch_quality, fh, indent=2)
        with open(final_dir / "apples_to_apples_report.json", "w", encoding="utf-8") as fh:
            json.dump(apples, fh, indent=2)
        if final_quality.get("mae_mm") is not None:
            print(f"Final batch MAE: {float(final_quality['mae_mm']):.6f} mm")
        print(f"Reports: {final_dir}")

    print(f"Setup audit: {final_dir / 'setup_audit.json'}")
    print(f"Final predictions: {len(final.y_pred)} rows  {final_csv}")
    if final_long_csv is not None:
        print(f"Final (long): {len(pd.read_csv(final_long_csv))} rows  {final_long_csv}")


if __name__ == "__main__":
    main()
