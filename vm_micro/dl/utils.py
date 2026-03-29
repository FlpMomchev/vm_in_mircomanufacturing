from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from vm_micro.data.io import get_input_kind

DEPTH_RE = re.compile(r"depth\s*([0-9]+(?:[.,][0-9]+)?)", flags=re.IGNORECASE)
STEP_RE = re.compile(r"__step\s*([0-9]+)__", flags=re.IGNORECASE)


def parse_depth_mm(name: str) -> float:
    match = DEPTH_RE.search(name)
    if not match:
        raise ValueError(f"Could not parse depth from filename: {name}")
    return float(match.group(1).replace(",", "."))


def parse_step_idx(stem: str) -> int:
    match = STEP_RE.search(stem)
    if not match:
        raise ValueError(f"Could not parse step index from filename: {stem}")
    return int(match.group(1))


def try_parse_step_idx(stem: str) -> int | None:
    try:
        return parse_step_idx(stem)
    except Exception:
        return None


def extract_recording_root(stem: str) -> str:
    return stem.split("__seg")[0] if "__seg" in stem else stem


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def choose_device(device: str = "auto") -> str:
    import torch

    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _h5_file_info(
    file_path: Path,
    data_key: str = "measurement/data",
    time_key: str = "measurement/time_vector",
) -> dict[str, Any]:
    """Read duration and sample rate from an HDF5 measurement file without
    loading the full signal into memory."""
    with h5py.File(str(file_path), "r") as fh:
        n_samples = fh[data_key].shape[0]
        time_vector = fh[time_key][:]

    dt = np.diff(time_vector)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    dt_median = float(np.median(dt)) if len(dt) else 1.0
    sr = int(round(1.0 / dt_median))
    duration_sec = float(n_samples / sr)

    return {
        "sample_rate_native": sr,
        "duration_sec": duration_sec,
        "frames_native": n_samples,
        "channels": 1,
    }


def build_file_table(
    data_dir: str | Path,
    file_glob: str = "**/*.flac",
    h5_data_key: str = "measurement/data",
    h5_time_key: str = "measurement/time_vector",
) -> pd.DataFrame:
    """Scan a directory and build the file-level metadata table.

    Supports both FLAC/WAV (airborne) and HDF5 (structure-borne) files.
    The file_glob determines which format is scanned; use ``**/*.flac``
    for airborne or ``**/*.h5`` for structure-borne.

    Metadata is used for splitting and analysis only; it is never fed
    into the model.
    """
    data_dir = Path(data_dir)
    rows: list[dict[str, Any]] = []

    for file_path in sorted(data_dir.glob(file_glob)):
        kind = get_input_kind(file_path)

        if kind == "audio":
            info = sf.info(str(file_path))
            file_meta = {
                "sample_rate_native": int(info.samplerate),
                "duration_sec": float(info.duration),
                "frames_native": int(info.frames),
                "channels": int(info.channels),
            }
        else:  # hdf5
            file_meta = _h5_file_info(file_path, h5_data_key, h5_time_key)

        rows.append(
            {
                "file_id": len(rows),
                "path": str(file_path.resolve()),
                "record_name": file_path.stem,
                "stem": file_path.stem,
                "file_name": file_path.name,
                "input_kind": kind,
                "depth_mm": parse_depth_mm(file_path.stem),
                "recording_root": extract_recording_root(file_path.stem),
                "parent_dir": file_path.parent.name,
                "file_group_id": file_path.stem,
                **file_meta,
            }
        )

    if not rows:
        raise FileNotFoundError(f"No files found in {data_dir} matching {file_glob}")

    return pd.DataFrame(rows)


def add_class_labels(
    file_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[float, int], dict[int, float]]:
    depths = sorted(file_df["depth_mm"].unique().tolist())
    depth_to_class = {depth: idx for idx, depth in enumerate(depths)}
    class_to_depth = {idx: depth for depth, idx in depth_to_class.items()}

    out = file_df.copy()
    out["class_idx"] = out["depth_mm"].map(depth_to_class).astype(int)
    return out, depth_to_class, class_to_depth


def attach_step_idx_if_possible(file_df: pd.DataFrame) -> pd.DataFrame:
    out = file_df.copy()
    out["step_idx"] = [try_parse_step_idx(stem) for stem in out["stem"].astype(str)]

    if out["step_idx"].isna().all():
        return out.drop(columns=["step_idx"])

    out["step_idx"] = out["step_idx"].astype("Int64")
    return out


def resolve_split_group_id(
    file_df: pd.DataFrame,
    evaluation_unit: str,
    group_mode: str,
) -> pd.Series:
    if evaluation_unit == "file":
        return file_df["file_group_id"].astype(str)

    if group_mode not in file_df.columns:
        raise ValueError(f"Missing group column for grouped evaluation: {group_mode}")

    return file_df[group_mode].astype(str)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def round_to_step(values: np.ndarray, step: float) -> np.ndarray:
    return np.round(values / step) * step


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_confusion_matrix_csv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    out_path: str | Path,
) -> None:
    label_ids = np.arange(len(labels))
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    frame = pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    frame.to_csv(out_path, index=True)


def write_label_mapping(class_to_depth: dict[int, float], out_path: str | Path) -> None:
    payload = [
        {"class_idx": int(class_idx), "depth_mm": float(depth_mm)}
        for class_idx, depth_mm in sorted(class_to_depth.items())
    ]
    dump_json(payload, out_path)


def read_label_mapping(path: str | Path) -> dict[int, float]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    return {int(row["class_idx"]): float(row["depth_mm"]) for row in rows}
