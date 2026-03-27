"""vm-predict-fused - One-shot fused inference from new raw recordings.

This script is designed as the first backend-friendly entrypoint:
- watches a raw input folder (scan on demand)
- splits whole recordings into segments
- runs per-modality inference (classical + DL)
- fuses predictions (intra + inter)
- writes one tidy output folder with CSV + JSON artefacts

Config-first: see ``configs/predict_fused.yaml``.
"""

from __future__ import annotations

import argparse
import json
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
    fuse_intra_modality,
    fuse_modalities,
    save_fusion_report,
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
    p.add_argument("--config", default="configs/predict_fused.yaml")
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

    if "record_name" not in df.columns:
        raise KeyError(f"Expected 'record_name' column in {csv_path}, got {list(df.columns)}")
    fusion_keys = df["record_name"].astype(str).map(_fusion_key_from_record_name)

    return PredictionBundle(
        modality=modality,
        record_names=fusion_keys.to_numpy(),
        y_pred=df["y_pred"].to_numpy(dtype=np.float64),
        sigma=sigma,
        validation_mae=float(fusion_mae),
        y_true=None,
        metadata={
            "validation_mae_source": fusion_mae_source,
            "record_key": "step+hole (parsed from record_name)",
        },
    )


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
        if "model" in bundle:
            return _read_classical_fusion_mae(
                bundle_path, bundle
            ), "final_model/test_metrics.csv.holdout_mae_raw"
        if "members" in bundle:
            return _read_classical_fusion_mae(
                bundle_path, bundle
            ), "ensemble_test_predictions.csv.y_pred_raw"
        return _read_classical_fusion_mae(bundle_path, bundle), "bundle"
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


def main() -> None:
    args = build_parser().parse_args()

    # This project pins modern scientific stack (see pyproject.toml).
    # Many stored Joblib bundles are not unpickleable on NumPy < 2.
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

    run_out_root = Path(cfg.get("run", {}).get("out_dir", "models/fusion/live_runs"))
    state_path = Path(
        cfg.get("run", {}).get("state_json", "models/fusion/predict_fused_state.json")
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

    #  splitting
    split_cfg = cfg.get("splitting", {})
    common = split_cfg.get("common", {})
    band_hz = (float(common.get("band_low_hz", 2000.0)), float(common.get("band_high_hz", 5000.0)))
    common_band_fallbacks = common.get("band_fallbacks_hz", None)
    pre_pad_s = float(common.get("pre_pad_s", 0.20))
    post_pad_s = float(common.get("post_pad_s", 0.25))

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
        air_band_fallbacks = air_split.get("band_fallbacks_hz", common_band_fallbacks)
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
                pre_pad_s=pre_pad_s,
                post_pad_s=post_pad_s,
                band_hz=band_hz,
                band_hz_fallbacks=air_band_fallbacks,
                export_format=str(air_split.get("export_format", "flac")),
                target_sr=air_split.get("target_sr", None),
            )
            out_manifest = run_dir / "airborne" / rf.stem / "segments_manifest.csv"
            out_manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest_df.to_csv(out_manifest, index=False)
            processed[str(rf.path.resolve())] = {
                "mtime_ns": rf.mtime_ns,
                "modality": "airborne",
                "stem": rf.stem,
            }
            logger.info(
                "Split airborne %s -> %s (segments=%d)",
                rf.path.name,
                summary["out_dir"],
                summary["exported_segments_final"],
            )
        seg_dir_air = seg_root_air

    # Split structure
    seg_dir_str = None
    if work_str:
        st_split = split_cfg.get("structure", {})
        st_band_fallbacks = st_split.get("band_fallbacks_hz", common_band_fallbacks)
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
                pre_pad_s=pre_pad_s,
                post_pad_s=post_pad_s,
                band_hz=band_hz,
                band_hz_fallbacks=st_band_fallbacks,
                export_format=str(st_split.get("export_format", "h5")),
                h5_data_key=str(st_split.get("h5_data_key", "measurement/data")),
                h5_time_key=str(st_split.get("h5_time_key", "measurement/time_vector")),
                target_sr=st_split.get("target_sr", None),
            )
            out_manifest = run_dir / "structure" / rf.stem / "segments_manifest.csv"
            out_manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest_df.to_csv(out_manifest, index=False)
            processed[str(rf.path.resolve())] = {
                "mtime_ns": rf.mtime_ns,
                "modality": "structure",
                "stem": rf.stem,
            }
            logger.info(
                "Split structure %s -> %s (segments=%d)",
                rf.path.name,
                summary["out_dir"],
                summary["exported_segments_final"],
            )
        seg_dir_str = seg_root_str

    _save_state(state_path, state)

    #  inference + fusion
    fused_modality_bundles: list[PredictionBundle] = []

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
        air_classical_fusion_mae: float | None = None
        air_classical_fusion_mae_source = ""
        if cls_cfg.get("enabled", True) and cls_cfg.get("bundle_path"):
            bundle_path = Path(cls_cfg["bundle_path"])
            bundle = joblib.load(bundle_path)
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

            feat_cfg = load_config("configs/airborne.yaml")
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
            tmp_features_csv = run_dir / "airborne" / "features_airborne.csv"
            tmp_features_csv.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(tmp_features_csv, index=False)

            infer_classical(
                bundle_path=bundle_path,
                features_csv=tmp_features_csv,
                out_csv=air_classical_csv,
                snap_predictions=cls_cfg.get("snap_predictions", None),
                doe_step_mm=cls_cfg.get("snap_step_mm", None),
            )

        # DL
        dl_cfg = air_cfg.get("dl", {})
        air_dl_csv = run_dir / "airborne" / "dl_predictions.csv"
        air_dl_fusion_mae: float | None = None
        air_dl_fusion_mae_source = ""
        if dl_cfg.get("enabled", True) and dl_cfg.get("model_dir"):
            air_dl_fusion_mae, air_dl_fusion_mae_source = _resolve_dl_fusion_mae(
                model_dir=Path(dl_cfg["model_dir"]),
                cfg_mae_fallback=_config_mae_fallback(dl_cfg),
            )
            _infer_dl_on_segments(
                model_dir=Path(dl_cfg["model_dir"]),
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
            air_dl_bundle = _bundle_from_pred_csv(
                air_dl_csv,
                "airborne_dl",
                fusion_mae=air_dl_fusion_mae,
                fusion_mae_source=air_dl_fusion_mae_source,
            )
            air_fused = fuse_intra_modality(
                air_cls_bundle,
                air_dl_bundle,
                modality_name="airborne_ensemble",
                min_weight=float(cfg.get("models", {}).get("fusion", {}).get("min_weight", 0.05)),
            )
            save_fusion_report(air_fused, run_dir / "airborne" / "fusion")
            fused_modality_bundles.append(air_fused)
        elif air_dl_csv.exists() and not air_classical_csv.exists():
            # Allow single-model airborne modality when classical inference is unavailable.
            if air_dl_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for airborne DL bundle.")
            b = _bundle_from_pred_csv(
                air_dl_csv,
                "airborne_ensemble",
                fusion_mae=air_dl_fusion_mae,
                fusion_mae_source=air_dl_fusion_mae_source,
            )
            save_fusion_report(b, run_dir / "airborne" / "fusion")
            fused_modality_bundles.append(b)
        elif air_classical_csv.exists() and not air_dl_csv.exists():
            if air_classical_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for airborne classical bundle.")
            b = _bundle_from_pred_csv(
                air_classical_csv,
                "airborne_ensemble",
                fusion_mae=air_classical_fusion_mae,
                fusion_mae_source=air_classical_fusion_mae_source,
            )
            save_fusion_report(b, run_dir / "airborne" / "fusion")
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
        st_dl_csv = run_dir / "structure" / "dl_predictions.csv"
        st_classical_fusion_mae: float | None = None
        st_classical_fusion_mae_source = ""
        st_dl_fusion_mae: float | None = None
        st_dl_fusion_mae_source = ""

        if cls_cfg.get("enabled", False) and cls_cfg.get("bundle_path"):
            bundle_path = Path(cls_cfg["bundle_path"])
            bundle = joblib.load(bundle_path)
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

            feat_cfg = load_config("configs/structure.yaml")
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
            tmp_features_csv = run_dir / "structure" / "features_structure.csv"
            tmp_features_csv.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(tmp_features_csv, index=False)

            infer_classical(
                bundle_path=bundle_path,
                features_csv=tmp_features_csv,
                out_csv=st_classical_csv,
                snap_predictions=cls_cfg.get("snap_predictions", None),
                doe_step_mm=cls_cfg.get("snap_step_mm", None),
            )

        if dl_cfg.get("enabled", False) and dl_cfg.get("model_dir"):
            st_dl_fusion_mae, st_dl_fusion_mae_source = _resolve_dl_fusion_mae(
                model_dir=Path(dl_cfg["model_dir"]),
                cfg_mae_fallback=_config_mae_fallback(dl_cfg),
            )
            st_split = cfg.get("splitting", {}).get("structure", {})
            _infer_dl_on_segments(
                model_dir=Path(dl_cfg["model_dir"]),
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
            st_dl_bundle = _bundle_from_pred_csv(
                st_dl_csv,
                "structure_dl",
                fusion_mae=st_dl_fusion_mae,
                fusion_mae_source=st_dl_fusion_mae_source,
            )
            st_fused = fuse_intra_modality(
                st_cls_bundle,
                st_dl_bundle,
                modality_name="structure_ensemble",
                min_weight=float(cfg.get("models", {}).get("fusion", {}).get("min_weight", 0.05)),
            )
            save_fusion_report(st_fused, run_dir / "structure" / "fusion")
            fused_modality_bundles.append(st_fused)
        elif st_dl_csv.exists() and not st_classical_csv.exists():
            # Allow single-model structure modality for now
            if st_dl_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for structure DL bundle.")
            b = _bundle_from_pred_csv(
                st_dl_csv,
                "structure_ensemble",
                fusion_mae=st_dl_fusion_mae,
                fusion_mae_source=st_dl_fusion_mae_source,
            )
            save_fusion_report(b, run_dir / "structure" / "fusion")
            fused_modality_bundles.append(b)
        elif st_classical_csv.exists() and not st_dl_csv.exists():
            if st_classical_fusion_mae is None:
                raise RuntimeError("Missing resolved MAE for structure classical bundle.")
            b = _bundle_from_pred_csv(
                st_classical_csv,
                "structure_ensemble",
                fusion_mae=st_classical_fusion_mae,
                fusion_mae_source=st_classical_fusion_mae_source,
            )
            save_fusion_report(b, run_dir / "structure" / "fusion")
            fused_modality_bundles.append(b)

    if not fused_modality_bundles:
        print(f"Run complete, but no fused modality bundles were produced. Outputs: {run_dir}")
        return

    final = fuse_modalities(
        *fused_modality_bundles,
        min_weight=float(cfg.get("models", {}).get("fusion", {}).get("min_weight", 0.05)),
    )
    save_fusion_report(final, run_dir / "final")
    final.to_dataframe().to_csv(run_dir / "final_predictions.csv", index=False)
    print(f"Final fused predictions: {len(final.y_pred)} rows  {run_dir / 'final_predictions.csv'}")


if __name__ == "__main__":
    main()
