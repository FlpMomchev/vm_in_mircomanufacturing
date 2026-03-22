"""vm_micro.classical.trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Grouped holdout + nested grouped CV classical ML trainer.

Supports: RF, ET, HistGBT, XGB, LGB, CatBoost, GPR, Ridge, ElasticNet,
          SVR (optional), KernelRidge (optional).

Persists:
  outputs/features/<run_tag>/final_model/
    best_model_bundle.joblib   <- estimator + scaler + metadata
    best_model_metadata.json
    best_model_feature_importance.csv
    holdout_metrics.csv
    repeat_metrics.csv
    fixed_grouped_split_assignment.csv

CLI entry point: ``vm-train-cls``  (scripts/train_classical.py).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

from ..utils import get_logger

logger = get_logger(__name__)

_META_COLS = {
    "modality", "record_name", "recording_root", "depth_mm",
    "step_idx", "duration_s", "sr_hz", "sr_hz_native", "sr_hz_used",
    "ds_rate", "file_path", "run_id", "batch_id",
    "exclude_from_dl_training",  # manifest flag
}
TARGET_COL = "depth_mm"
GROUP_COL  = "recording_root"


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2":   float(r2_score(y_true, y_pred)),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
    }


def _build_models(skip_slow: bool = False) -> dict[str, Any]:
    """Return a dict of model_name → sklearn-compatible estimator."""
    models: dict[str, Any] = {
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
        "ExtraTrees": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
        "HistGBT": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", HistGradientBoostingRegressor(max_iter=500, random_state=42)),
        ]),
        "Ridge": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", Ridge(alpha=1.0)),
        ]),
        "ElasticNet": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", ElasticNet(max_iter=5000, random_state=42)),
        ]),
        "GPR": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                n_restarts_optimizer=3, random_state=42,
            )),
        ]),
    }

    if _HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", XGBRegressor(
                n_estimators=500, learning_rate=0.05, tree_method="hist",
                device="cpu", n_jobs=1, random_state=42, verbosity=0,
            )),
        ])
    if _HAS_LGB:
        models["LightGBM"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", LGBMRegressor(
                n_estimators=500, learning_rate=0.05,
                device="cpu", n_jobs=1, random_state=42, verbose=-1,
            )),
        ])
    if _HAS_CAT:
        models["CatBoost"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", CatBoostRegressor(
                iterations=500, learning_rate=0.05, task_type="CPU",
                random_seed=42, verbose=0, eval_metric="MAE",
            )),
        ])

    return models


def _feature_cols(df: pd.DataFrame) -> list[str]:
    non_feat = _META_COLS | {TARGET_COL}
    return [c for c in df.columns if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])]


def train_classical(
    features_csv: str | Path,
    out_dir: str | Path,
    *,
    holdout_runs: list[str] | None = None,
    train_fraction: float = 0.70,
    n_cv_folds: int = 5,
    skip_slow_models: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Train all classical ML models with grouped holdout + nested CV.

    Parameters
    ----------
    features_csv    : Path to the selected-features CSV.
    out_dir         : Output directory for artefacts.
    holdout_runs    : Optional list of ``recording_root`` values to hold out
                      completely (e.g. the 2 unseen DL runs).
    train_fraction  : Fraction of non-holdout data used for training.
    n_cv_folds      : Number of grouped CV folds.
    skip_slow_models: Skip SVR and KernelRidge (O(n²-n³), no GPU).
    seed            : Random seed.

    Returns
    -------
    dict with keys: best_model_name, holdout_metrics, repeat_metrics_df, out_dir.
    """
    out_dir = Path(out_dir)
    final_dir = out_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    feat_cols = _feature_cols(df)
    logger.info("Loaded %d rows × %d features from %s", len(df), len(feat_cols), features_csv)

    # Holdout split (completely unseen groups)
    if holdout_runs:
        mask_ho = df[GROUP_COL].isin(holdout_runs)
        df_holdout = df[mask_ho].copy()
        df_train   = df[~mask_ho].copy()
        logger.info("Holdout: %d rows (%s). Train pool: %d rows.", len(df_holdout), holdout_runs, len(df_train))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=1 - train_fraction, random_state=seed)
        train_idx, hold_idx = next(gss.split(df, groups=df[GROUP_COL]))
        df_train   = df.iloc[train_idx].copy()
        df_holdout = df.iloc[hold_idx].copy()

    df_train.to_csv(out_dir / "fixed_grouped_split_assignment.csv", index=False)

    X_train = df_train[feat_cols].to_numpy()
    y_train = df_train[TARGET_COL].to_numpy(dtype=np.float64)
    groups  = df_train[GROUP_COL].to_numpy()
    X_hold  = df_holdout[feat_cols].to_numpy()
    y_hold  = df_holdout[TARGET_COL].to_numpy(dtype=np.float64)

    models = _build_models(skip_slow=skip_slow_models)

    # Nested grouped CV to compare models
    cv = GroupKFold(n_splits=n_cv_folds)
    cv_results: dict[str, list[float]] = {name: [] for name in models}

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train, groups)):
        xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        xva, yva = X_train[va_idx], y_train[va_idx]
        for name, pipe in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe_copy = clone(pipe)
                pipe_copy.fit(xtr, ytr)
                pred = pipe_copy.predict(xva)
            cv_results[name].append(float(mean_absolute_error(yva, pred)))
        logger.info("CV fold %d/%d done.", fold_i + 1, n_cv_folds)

    repeat_df = pd.DataFrame(cv_results)
    repeat_summary = repeat_df.describe().T[["mean", "std"]]
    best_model_name = str(repeat_summary["mean"].idxmin())
    logger.info("Best model by CV MAE: %s", best_model_name)

    # Retrain best model on full training set
    best_pipe = clone(models[best_model_name])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_pipe.fit(X_train, y_train)

    # Holdout evaluation
    hold_pred = best_pipe.predict(X_hold)
    hold_metrics = _regression_metrics(y_hold, hold_pred)
    hold_metrics["n_holdout"] = int(len(y_hold))
    logger.info("Holdout metrics: %s", hold_metrics)

    # Uncertainty (ensemble spread across CV folds)
    fold_preds = []
    for tr_idx, _ in cv.split(X_train, y_train, groups):
        xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        p = clone(models[best_model_name])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.fit(xtr, ytr)
        fold_preds.append(p.predict(X_hold))

    fold_preds_arr = np.stack(fold_preds, axis=1)  # (n_hold, n_folds)
    sigma_pred = float(np.std(fold_preds_arr, axis=1).mean())
    residuals  = hold_pred - y_hold
    sigma_mae  = float(np.std(residuals))
    total_uncertainty = float(np.sqrt(sigma_pred ** 2 + sigma_mae ** 2))

    # Persist artefacts
    bundle = {
        "model": best_pipe,
        "feature_cols": feat_cols,
        "best_model_name": best_model_name,
        "holdout_mae": hold_metrics["mae"],
        "sigma_pred": sigma_pred,
        "sigma_mae": sigma_mae,
        "total_uncertainty": total_uncertainty,
        "modality": "airborne_classical",  # overridden by caller for structure
    }
    joblib.dump(bundle, final_dir / "best_model_bundle.joblib")

    metadata = {
        "best_model_name":   best_model_name,
        "features_csv":      str(features_csv),
        "n_features":        len(feat_cols),
        "n_train":           int(len(y_train)),
        "n_holdout":         int(len(y_hold)),
        "holdout_runs":      holdout_runs or [],
        "holdout_metrics":   hold_metrics,
        "sigma_pred":        sigma_pred,
        "sigma_mae":         sigma_mae,
        "total_uncertainty": total_uncertainty,
        "cv_mean_mae":       float(repeat_summary.loc[best_model_name, "mean"]),
    }
    with open(final_dir / "best_model_metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)

    repeat_df.to_csv(out_dir / "repeat_metrics.csv", index=False)
    pd.DataFrame([hold_metrics]).to_csv(final_dir / "holdout_metrics.csv", index=False)

    # Feature importance (if available)
    _try_save_feature_importance(best_pipe, feat_cols, final_dir)

    logger.info("Artefacts saved to %s", final_dir)
    return {
        "best_model_name":   best_model_name,
        "holdout_metrics":   hold_metrics,
        "repeat_metrics_df": repeat_df,
        "out_dir":           str(out_dir),
        "bundle_path":       str(final_dir / "best_model_bundle.joblib"),
        "total_uncertainty": total_uncertainty,
    }


def _try_save_feature_importance(pipe: Pipeline, feat_cols: list[str], out_dir: Path) -> None:
    try:
        mdl = pipe["mdl"]  # type: ignore[index]
        if hasattr(mdl, "feature_importances_"):
            imp = mdl.feature_importances_
        elif hasattr(mdl, "coef_"):
            imp = np.abs(mdl.coef_)
        else:
            return
        pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values(
            "importance", ascending=False
        ).to_csv(out_dir / "best_model_feature_importance.csv", index=False)
    except Exception:
        pass
