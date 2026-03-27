"""vm_micro.classical.inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Load a trained classical ML bundle and run inference on a feature CSV.
Supports both single-model and ensemble bundles.
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
    *,
    snap_predictions: bool | None = None,
    doe_step_mm: float | None = None,
) -> pd.DataFrame:
    """Run inference with a classical ML bundle."""

    bundle: dict[str, Any] = joblib.load(bundle_path)
    df = pd.read_csv(features_csv)

    # ----------------------------
    # Single-model bundle
    # ----------------------------
    if "model" in bundle:
        model = bundle["model"]
        feat_cols = bundle["feature_cols"]

        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required feature columns: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

        X = df[feat_cols]
        y_pred = model.predict(X)

    # ----------------------------
    # Ensemble bundle
    # ----------------------------
    elif "members" in bundle:
        members = bundle["members"]
        if not members:
            raise ValueError("Ensemble bundle contains no members.")

        member_preds = []

        for i, member in enumerate(members):
            if not isinstance(member, dict):
                raise TypeError(f"Ensemble member {i} is not a dict. Got: {type(member)}")

            if "model" not in member:
                raise KeyError(
                    f"Ensemble member {i} does not contain 'model'. "
                    f"Available keys: {list(member.keys())}"
                )

            if "feature_cols" not in member:
                raise KeyError(
                    f"Ensemble member {i} does not contain 'feature_cols'. "
                    f"Available keys: {list(member.keys())}"
                )

            model = member["model"]
            feat_cols = member["feature_cols"]

            missing = [c for c in feat_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing {len(missing)} required feature columns for ensemble member {i}: "
                    f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
                )

            X = df[feat_cols]
            pred = np.asarray(model.predict(X), dtype=float)
            member_preds.append(pred)

        y_pred = np.mean(np.column_stack(member_preds), axis=1)

        # Optional snapping to DOE grid
        effective_snap = (
            bundle.get("snap_predictions", False) if snap_predictions is None else snap_predictions
        )
        effective_step = bundle.get("doe_step_mm", None) if doe_step_mm is None else doe_step_mm
        if effective_snap:
            if effective_step is not None and effective_step > 0:
                y_pred = np.round(y_pred / effective_step) * effective_step

    else:
        raise KeyError(f"Unsupported bundle format. Available keys: {list(bundle.keys())}")

    out = df.copy()
    out["y_pred"] = y_pred

    if "depth_mm" in out.columns:
        out["residual"] = out["depth_mm"] - out["y_pred"]
        mae = np.mean(np.abs(out["residual"]))
        logger.info("MAE vs ground truth: %.4f mm", mae)

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        logger.info("Saved inference results to %s", out_csv)

    return out
