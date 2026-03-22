"""vm_micro.classical.inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Load a trained classical ML bundle and run inference on a feature CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..utils import get_logger

logger = get_logger(__name__)


def infer_classical(
    bundle_path: str | Path,
    features_csv: str | Path,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Run inference with a classical ML bundle.

    Parameters
    ----------
    bundle_path  : Path to ``best_model_bundle.joblib``.
    features_csv : CSV with feature columns matching the bundle's ``feature_cols``.
    out_csv      : If provided, save predictions CSV here.

    Returns
    -------
    DataFrame with columns: record_name, depth_mm (true), y_pred, residual.
    """
    bundle: dict[str, Any] = joblib.load(bundle_path)
    model      = bundle["model"]
    feat_cols  = bundle["feature_cols"]
    modality   = bundle.get("modality", "unknown")
    holdout_mae = float(bundle.get("holdout_mae", float("nan")))
    sigma_pred  = float(bundle.get("sigma_pred",  float("nan")))
    sigma_mae   = float(bundle.get("sigma_mae",   float("nan")))

    df = pd.read_csv(features_csv)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Features CSV is missing columns: {missing[:5]}")

    X    = df[feat_cols].to_numpy()
    pred = model.predict(X)

    out = df[["record_name"]].copy()
    if "depth_mm" in df.columns:
        out["depth_mm"] = df["depth_mm"].values
    out["y_pred"]    = pred
    out["modality"]  = modality
    out["holdout_mae"] = holdout_mae
    out["sigma_pred"]  = sigma_pred
    out["sigma_mae"]   = sigma_mae

    if "depth_mm" in out.columns:
        out["residual"] = out["y_pred"] - out["depth_mm"]

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        logger.info("Saved inference results to %s", out_csv)

    return out
