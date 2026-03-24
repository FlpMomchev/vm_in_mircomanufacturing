"""vm_micro.classical.trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Grouped holdout + nested grouped CV classical ML trainer.

This module mirrors the stable search / selection logic used in the newer
standalone trainer while keeping the package-facing API for ``vm-train-cls``.

Highlights
----------
- grouped train / holdout split by ``recording_root``
- nested grouped CV for model comparison
- configurable search presets with centralised param-grid configs
- optional GPU acceleration for XGBoost / LightGBM only
- optional top-N ensemble persisted for inference
- CSV + Joblib artefacts compatible with the package workflow
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
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_META_COLS = {
    "modality",
    "record_name",
    "recording_root",
    "depth_mm",
    "step_idx",
    "duration_s",
    "sr_hz",
    "sr_hz_native",
    "sr_hz_used",
    "ds_rate",
    "file_path",
    "run_id",
    "batch_id",
    "exclude_from_dl_training",
}
TARGET_COL = "depth_mm"
GROUP_COL = "recording_root"
RECORD_COL = "record_name"
DOE_STEP_MM = 0.10
TARGET_MAE_MM = 0.05
DEFAULT_SEARCH_N_JOBS = -1
GPU_MODEL_NAMES: frozenset[str] = frozenset({"xgboost", "lightgbm"})
AVAILABLE_MODEL_NAMES: tuple[str, ...] = (
    "ridge",
    "elasticnet",
    "svr",
    "kernel_ridge",
    "random_forest",
    "extra_trees",
    "hist_gb",
    "xgboost",
    "lightgbm",
    "catboost",
    "gaussian_process",
)
SEARCH_KIND_BY_PRESET: dict[str, str] = {
    "fast": "grid",
    "balanced": "grid",
    "exhaustive": "random",
}

PARAM_GRID_CONFIGS: dict[str, dict[str, list[Any]]] = {
    "fast": {
        "ridge_alpha": [0.01, 0.1, 1.0, 10.0],
        "en_alpha": [0.001, 0.01, 0.1],
        "en_l1": [0.2, 0.5, 0.8],
        "svr_linear_c": [0.1, 1.0, 10.0],
        "svr_epsilon": [0.01, 0.03, 0.05],
        "svr_rbf_c": [1.0, 10.0, 50.0],
        "svr_gamma": ["scale", 0.01, 0.1],
        "kr_alpha": [0.01, 0.1, 1.0],
        "kr_gamma": [0.01, 0.1],
        "forest_estimators": [300, 600],
        "forest_depth": [None, 24],
        "forest_leaf": [2, 4],
        "forest_max_features": [1.0],
        "hist_lr": [0.03, 0.1],
        "hist_depth": [None, 5],
        "hist_iter": [300, 600],
        "hist_leaf": [10, 20],
        "xgb_estimators": [300],
        "xgb_depth": [3, 5],
        "xgb_lr": [0.03, 0.1],
        "xgb_subsample": [0.8, 1.0],
        "xgb_colsample": [0.8, 1.0],
        "xgb_mcw": [1, 3],
        "xgb_reg_alpha": [0.0, 0.1],
        "lgb_estimators": [300],
        "lgb_leaves": [15, 31],
        "lgb_depth": [-1, 6],
        "lgb_lr": [0.03, 0.1],
        "lgb_mcs": [10, 20],
        "lgb_reg_alpha": [0.0, 0.1],
        "cat_depth": [4, 6],
        "cat_lr": [0.03, 0.1],
        "cat_iter": [300, 800],
        "cat_l2": [3, 7],
        "cat_rs": [1.0],
        "cat_bt": [1.0],
        "gpr_alpha": [1e-6, 1e-4],
    },
    "balanced": {
        "ridge_alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0],
        "en_alpha": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        "en_l1": [0.1, 0.3, 0.5, 0.7, 0.9],
        "svr_linear_c": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        "svr_epsilon": [0.005, 0.01, 0.02, 0.03],
        "svr_rbf_c": [0.3, 1.0, 3.0, 10.0, 30.0],
        "svr_gamma": ["scale", 0.003, 0.01, 0.03, 0.1],
        "kr_alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "kr_gamma": [0.003, 0.01, 0.03, 0.1],
        "forest_estimators": [300, 600, 900],
        "forest_depth": [None, 12, 16, 24],
        "forest_leaf": [1, 2, 4],
        "forest_max_features": [1.0, "sqrt"],
        "hist_lr": [0.02, 0.03, 0.05],
        "hist_depth": [None, 3, 5],
        "hist_iter": [300, 600, 900],
        "hist_leaf": [5, 10, 20],
        "xgb_estimators": [300, 600],
        "xgb_depth": [3, 5],
        "xgb_lr": [0.02, 0.03, 0.05],
        "xgb_subsample": [0.8, 1.0],
        "xgb_colsample": [0.8, 1.0],
        "xgb_mcw": [1, 3, 5],
        "xgb_reg_alpha": [0.0, 0.05, 0.1],
        "lgb_estimators": [300, 600],
        "lgb_leaves": [15, 31, 63],
        "lgb_depth": [-1, 4, 6],
        "lgb_lr": [0.02, 0.03, 0.05],
        "lgb_mcs": [5, 10, 20],
        "lgb_reg_alpha": [0.0, 0.05, 0.1],
        "cat_depth": [4, 6, 8],
        "cat_lr": [0.02, 0.03, 0.05],
        "cat_iter": [300, 600, 1000],
        "cat_l2": [1, 3, 5],
        "cat_rs": [0.5, 1.0],
        "cat_bt": [0.5, 1.0],
        "gpr_alpha": [1e-8, 1e-6, 1e-4],
    },
    "exhaustive": {
        "ridge_alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        "en_alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        "en_l1": [0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95],
        "svr_linear_c": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        "svr_epsilon": [0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        "svr_rbf_c": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        "svr_gamma": ["scale", 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        "kr_alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "kr_gamma": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        "forest_estimators": [300, 600, 900],
        "forest_depth": [None, 8, 12, 16, 24],
        "forest_leaf": [1, 2, 4, 8],
        "forest_max_features": [1.0, "sqrt"],
        "hist_lr": [0.01, 0.03, 0.05, 0.1],
        "hist_depth": [None, 3, 5, 8],
        "hist_iter": [200, 400, 800],
        "hist_leaf": [5, 10, 20, 30],
        "xgb_estimators": [200, 400, 800],
        "xgb_depth": [3, 5, 7],
        "xgb_lr": [0.02, 0.03, 0.05, 0.1],
        "xgb_subsample": [0.7, 0.85, 1.0],
        "xgb_colsample": [0.7, 0.85, 1.0],
        "xgb_mcw": [1, 3, 5, 8],
        "xgb_reg_alpha": [0.0, 0.05, 0.1, 0.5, 1.0],
        "lgb_estimators": [200, 400, 800],
        "lgb_leaves": [15, 31, 63],
        "lgb_depth": [-1, 4, 6, 8],
        "lgb_lr": [0.02, 0.03, 0.05, 0.1],
        "lgb_mcs": [5, 10, 20, 40],
        "lgb_reg_alpha": [0.0, 0.05, 0.1, 0.5, 1.0],
        "cat_depth": [4, 6, 8, 10],
        "cat_lr": [0.02, 0.03, 0.05, 0.1],
        "cat_iter": [300, 600, 1000],
        "cat_l2": [1, 3, 5, 9],
        "cat_rs": [0.5, 1.0, 2.0],
        "cat_bt": [0.5, 1.0, 2.0],
        "gpr_alpha": [1e-8, 1e-6, 1e-4, 1e-3],
    },
}


def _normalise_preset(preset: str) -> str:
    preset_norm = preset.strip().lower()
    if preset_norm not in PARAM_GRID_CONFIGS:
        raise ValueError(
            f"Unknown preset '{preset}'. Expected one of {sorted(PARAM_GRID_CONFIGS)}."
        )
    return preset_norm


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
    }


def _feature_cols(df: pd.DataFrame) -> list[str]:
    non_feat = _META_COLS | {TARGET_COL}
    return [c for c in df.columns if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])]


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _snap_to_grid(y_pred: np.ndarray, step: float) -> np.ndarray:
    return np.round(y_pred / step) * step


def make_model_specs(
    *,
    preset: str,
    random_state: int,
    use_cuda: bool = False,
    include_gpr: bool = False,
) -> dict[str, dict[str, Any]]:
    cfg = PARAM_GRID_CONFIGS[_normalise_preset(preset)]

    xgb_device = {"tree_method": "hist", "device": "cuda"} if use_cuda else {"tree_method": "hist"}
    lgb_device = {"device_type": "gpu"} if use_cuda else {}
    cat_cpu = {"thread_count": -1}

    gpr_kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=1.0
    ) + WhiteKernel(noise_level=1e-3)

    specs: dict[str, dict[str, Any]] = {
        "ridge": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            "param_grid": {"model__alpha": cfg["ridge_alpha"]},
        },
        "elasticnet": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(max_iter=50_000, random_state=random_state)),
                ]
            ),
            "param_grid": {
                "model__alpha": cfg["en_alpha"],
                "model__l1_ratio": cfg["en_l1"],
            },
        },
        "svr": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVR()),
                ]
            ),
            "param_grid": [
                {
                    "model__kernel": ["linear"],
                    "model__C": cfg["svr_linear_c"],
                    "model__epsilon": cfg["svr_epsilon"],
                },
                {
                    "model__kernel": ["rbf"],
                    "model__C": cfg["svr_rbf_c"],
                    "model__epsilon": cfg["svr_epsilon"],
                    "model__gamma": cfg["svr_gamma"],
                },
            ],
        },
        "kernel_ridge": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", KernelRidge()),
                ]
            ),
            "param_grid": [
                {"model__kernel": ["linear"], "model__alpha": cfg["kr_alpha"]},
                {
                    "model__kernel": ["rbf"],
                    "model__alpha": cfg["kr_alpha"],
                    "model__gamma": cfg["kr_gamma"],
                },
            ],
        },
        "random_forest": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestRegressor(random_state=random_state, n_jobs=1)),
                ]
            ),
            "param_grid": {
                "model__n_estimators": cfg["forest_estimators"],
                "model__max_depth": cfg["forest_depth"],
                "model__min_samples_leaf": cfg["forest_leaf"],
                "model__max_features": cfg["forest_max_features"],
                "model__criterion": ["squared_error", "absolute_error"],
            },
        },
        "extra_trees": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", ExtraTreesRegressor(random_state=random_state, n_jobs=1)),
                ]
            ),
            "param_grid": {
                "model__n_estimators": cfg["forest_estimators"],
                "model__max_depth": cfg["forest_depth"],
                "model__min_samples_leaf": cfg["forest_leaf"],
                "model__max_features": cfg["forest_max_features"],
                "model__criterion": ["squared_error", "absolute_error"],
            },
        },
        "hist_gb": {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingRegressor(
                            random_state=random_state,
                            loss="absolute_error",
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__learning_rate": cfg["hist_lr"],
                "model__max_depth": cfg["hist_depth"],
                "model__max_iter": cfg["hist_iter"],
                "model__min_samples_leaf": cfg["hist_leaf"],
            },
        },
    }

    if _HAS_XGB:
        specs["xgboost"] = {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:absoluteerror",
                            eval_metric="mae",
                            random_state=random_state,
                            n_jobs=1,
                            verbosity=0,
                            **xgb_device,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__n_estimators": cfg["xgb_estimators"],
                "model__max_depth": cfg["xgb_depth"],
                "model__learning_rate": cfg["xgb_lr"],
                "model__subsample": cfg["xgb_subsample"],
                "model__colsample_bytree": cfg["xgb_colsample"],
                "model__min_child_weight": cfg["xgb_mcw"],
                "model__reg_alpha": cfg["xgb_reg_alpha"],
            },
        }

    if _HAS_LGB:
        specs["lightgbm"] = {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        LGBMRegressor(
                            objective="regression_l1",
                            random_state=random_state,
                            n_jobs=1,
                            verbose=-1,
                            **lgb_device,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__n_estimators": cfg["lgb_estimators"],
                "model__num_leaves": cfg["lgb_leaves"],
                "model__max_depth": cfg["lgb_depth"],
                "model__learning_rate": cfg["lgb_lr"],
                "model__min_child_samples": cfg["lgb_mcs"],
                "model__reg_alpha": cfg["lgb_reg_alpha"],
            },
        }

    if _HAS_CAT:
        specs["catboost"] = {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        CatBoostRegressor(
                            loss_function="MAE",
                            eval_metric="MAE",
                            verbose=0,
                            random_seed=random_state,
                            task_type="CPU",
                            allow_writing_files=False,
                            **cat_cpu,
                        ),
                    ),
                ]
            ),
            "param_grid": {
                "model__depth": cfg["cat_depth"],
                "model__learning_rate": cfg["cat_lr"],
                "model__iterations": cfg["cat_iter"],
                "model__l2_leaf_reg": cfg["cat_l2"],
                "model__random_strength": cfg["cat_rs"],
                "model__bagging_temperature": cfg["cat_bt"],
            },
        }

    if include_gpr:
        specs["gaussian_process"] = {
            "pipeline": Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GaussianProcessRegressor(
                            kernel=gpr_kernel,
                            alpha=1e-6,
                            normalize_y=True,
                            n_restarts_optimizer=0,
                        ),
                    ),
                ]
            ),
            "param_grid": {"model__alpha": cfg["gpr_alpha"]},
        }

    return specs


def _resolve_model_specs(
    *,
    preset: str,
    random_state: int,
    use_cuda: bool,
    include_gpr: bool,
    skip_slow_models: bool,
    requested_models: list[str] | None,
) -> dict[str, dict[str, Any]]:
    specs = make_model_specs(
        preset=preset,
        random_state=random_state,
        use_cuda=use_cuda,
        include_gpr=include_gpr,
    )

    if skip_slow_models:
        specs.pop("svr", None)
        specs.pop("kernel_ridge", None)

    if requested_models is None:
        return specs

    requested_models = list(dict.fromkeys(requested_models))
    unavailable: list[str] = []
    for name in requested_models:
        if name not in specs:
            if name == "gaussian_process" and not include_gpr:
                raise ValueError(
                    "Requested model 'gaussian_process' is unavailable. "
                    "Use --include-gpr together with --model gaussian_process."
                )
            if name in {"svr", "kernel_ridge"} and skip_slow_models:
                raise ValueError(f"Requested model '{name}' was removed by --skip-slow-models.")
            unavailable.append(name)

    if unavailable:
        raise ValueError(
            f"Requested model(s) not available: {unavailable}. Enabled models: {sorted(specs)}"
        )

    return {name: specs[name] for name in requested_models}


def _make_search(
    model_spec: dict[str, Any],
    inner_cv: GroupKFold,
    search_n_jobs: int,
    preset: str,
    n_iter: int,
    random_state: int,
    is_gpu_model: bool = False,
) -> GridSearchCV | RandomizedSearchCV:
    effective_n_jobs = 1 if is_gpu_model else search_n_jobs
    preset_norm = _normalise_preset(preset)

    common = dict(
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=effective_n_jobs,
        refit=True,
        verbose=0,
        error_score="raise",
        return_train_score=False,
    )

    if SEARCH_KIND_BY_PRESET[preset_norm] == "random":
        return RandomizedSearchCV(
            estimator=clone(model_spec["pipeline"]),
            param_distributions=model_spec["param_grid"],
            n_iter=n_iter,
            random_state=random_state,
            **common,
        )

    return GridSearchCV(
        estimator=clone(model_spec["pipeline"]),
        param_grid=model_spec["param_grid"],
        **common,
    )


def _grouped_train_holdout_split(
    df: pd.DataFrame,
    *,
    holdout_runs: list[str] | None,
    train_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if holdout_runs:
        mask_holdout = df[GROUP_COL].astype(str).isin([str(x) for x in holdout_runs])
        holdout_df = df.loc[mask_holdout].copy()
        train_df = df.loc[~mask_holdout].copy()
        if holdout_df.empty:
            raise ValueError("None of the provided --holdout-runs matched the CSV.")
        if train_df.empty:
            raise ValueError("Holdout split removed all rows from training.")
        return train_df, holdout_df

    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=1 - train_fraction,
        random_state=random_state,
    )
    train_idx, hold_idx = next(gss.split(df, groups=df[GROUP_COL]))
    return df.iloc[train_idx].copy(), df.iloc[hold_idx].copy()


def _summarize_predictions(pred_df: pd.DataFrame) -> dict[str, float]:
    residual = pred_df["y_true"] - pred_df["y_pred"]
    abs_error = np.abs(residual)
    return {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "r2": float(r2_score(pred_df["y_true"], pred_df["y_pred"])),
        "mean_signed_error": float(np.mean(pred_df["y_pred"] - pred_df["y_true"])),
        "bias_mean": float(np.mean(residual)),
        "abs_error_p50": float(np.quantile(abs_error, 0.50)),
        "abs_error_p90": float(np.quantile(abs_error, 0.90)),
        "abs_error_p95": float(np.quantile(abs_error, 0.95)),
    }


def evaluate_model_nested_cv(
    train_df: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
    model_spec: dict[str, Any],
    *,
    outer_max_splits: int,
    inner_max_splits: int,
    search_n_jobs: int,
    preset: str,
    n_iter: int,
    random_state: int,
    use_cuda: bool,
) -> dict[str, pd.DataFrame]:
    data = (
        train_df[feature_names + [TARGET_COL, GROUP_COL]]
        .dropna(subset=[TARGET_COL, GROUP_COL])
        .copy()
    )
    X = data[feature_names]
    y = data[TARGET_COL].to_numpy(dtype=np.float64)
    groups = data[GROUP_COL].to_numpy()

    n_groups = pd.Series(groups).nunique()
    if n_groups < 3:
        raise ValueError(f"Need >=3 groups for outer CV, got {n_groups}")

    outer_cv = GroupKFold(n_splits=min(outer_max_splits, n_groups))
    is_gpu = use_cuda and model_name in GPU_MODEL_NAMES

    fold_rows: list[dict[str, Any]] = []
    search_rows: list[pd.DataFrame] = []

    for fold_idx, (outer_train_idx, outer_valid_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        Xtr = X.iloc[outer_train_idx]
        Xva = X.iloc[outer_valid_idx]
        ytr = y[outer_train_idx]
        yva = y[outer_valid_idx]
        gtr = groups[outer_train_idx]
        gva = groups[outer_valid_idx]

        n_inner_groups = pd.Series(gtr).nunique()
        if n_inner_groups < 2:
            raise ValueError(f"Need >=2 groups for inner CV, got {n_inner_groups}")

        inner_cv = GroupKFold(n_splits=min(inner_max_splits, n_inner_groups))
        search = _make_search(
            model_spec,
            inner_cv,
            search_n_jobs,
            preset,
            n_iter,
            random_state,
            is_gpu_model=is_gpu,
        )
        search.fit(Xtr, ytr, groups=gtr)
        y_pred = search.best_estimator_.predict(Xva)

        fold_metrics = _regression_metrics(yva, y_pred)
        fold_rows.append(
            {
                "model_name": model_name,
                "fold": fold_idx,
                "n_features": len(feature_names),
                "n_train_rows": int(len(outer_train_idx)),
                "n_valid_rows": int(len(outer_valid_idx)),
                "n_train_runs": int(pd.Series(gtr).nunique()),
                "n_valid_runs": int(pd.Series(gva).nunique()),
                "inner_best_neg_mae": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                **fold_metrics,
            }
        )

        search_df = pd.DataFrame(
            {
                "rank_test_score": search.cv_results_["rank_test_score"],
                "mean_test_score": search.cv_results_["mean_test_score"],
                "std_test_score": search.cv_results_["std_test_score"],
                "params": [json.dumps(p, sort_keys=True) for p in search.cv_results_["params"]],
            }
        )
        search_df["model_name"] = model_name
        search_df["outer_fold"] = fold_idx
        search_rows.append(search_df)

        logger.info(
            "Nested CV | %s | fold %d | MAE=%.4f | RMSE=%.4f | R2=%.4f",
            model_name,
            fold_idx,
            fold_metrics["mae"],
            fold_metrics["rmse"],
            fold_metrics["r2"],
        )

    return {
        "fold_results": pd.DataFrame(fold_rows),
        "search_results": pd.concat(search_rows, ignore_index=True),
    }


def fit_model_on_train(
    train_df: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
    model_spec: dict[str, Any],
    *,
    inner_max_splits: int,
    search_n_jobs: int,
    preset: str,
    n_iter: int,
    random_state: int,
    use_cuda: bool,
) -> GridSearchCV | RandomizedSearchCV:
    data = (
        train_df[feature_names + [TARGET_COL, GROUP_COL]]
        .dropna(subset=[TARGET_COL, GROUP_COL])
        .copy()
    )
    X = data[feature_names]
    y = data[TARGET_COL].to_numpy(dtype=np.float64)
    groups = data[GROUP_COL].to_numpy()

    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        raise ValueError(f"Need >=2 groups for final inner CV, got {n_groups}")

    inner_cv = GroupKFold(n_splits=min(inner_max_splits, n_groups))
    search = _make_search(
        model_spec,
        inner_cv,
        search_n_jobs,
        preset,
        n_iter,
        random_state,
        is_gpu_model=use_cuda and model_name in GPU_MODEL_NAMES,
    )
    search.fit(X, y, groups=groups)
    return search


def evaluate_on_holdout(
    fitted_model: Any,
    holdout_df: pd.DataFrame,
    feature_names: list[str],
    *,
    target_mae: float,
    doe_step: float,
    snap_predictions: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    present_cols = [
        c for c in feature_names + [TARGET_COL, GROUP_COL, RECORD_COL] if c in holdout_df.columns
    ]
    data = holdout_df[present_cols].dropna(subset=[TARGET_COL, GROUP_COL]).copy()

    y_true = data[TARGET_COL].to_numpy(dtype=np.float64)
    y_pred_raw = fitted_model.predict(data[feature_names])
    y_pred = (
        _snap_to_grid(y_pred_raw, doe_step) if snap_predictions and doe_step > 0 else y_pred_raw
    )

    pred_df = data.copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred_raw"] = y_pred_raw
    pred_df["y_pred"] = y_pred
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df["sq_error"] = np.square(pred_df["residual"])

    metrics = {
        "holdout_mae": float(mean_absolute_error(y_true, y_pred)),
        "holdout_mae_raw": float(mean_absolute_error(y_true, y_pred_raw)),
        "holdout_rmse": float(_safe_rmse(y_true, y_pred)),
        "holdout_r2": float(r2_score(y_true, y_pred)),
        "n_holdout_rows": int(len(pred_df)),
        "n_holdout_runs": int(pred_df[GROUP_COL].nunique()),
        "goal_mae_threshold_mm": float(target_mae),
        "goal_met": bool(mean_absolute_error(y_true, y_pred) < target_mae),
        "doe_step_mm": float(doe_step),
        "mae_as_fraction_of_doe_step": float(mean_absolute_error(y_true, y_pred) / doe_step)
        if doe_step > 0
        else np.nan,
        "snapping_applied": bool(snap_predictions),
        **_summarize_predictions(pred_df),
    }
    return pred_df, metrics


def grouped_oof_predictions(
    estimator_template: Any,
    data_df: pd.DataFrame,
    feature_names: list[str],
    *,
    n_splits_max: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    present_cols = [
        c for c in feature_names + [TARGET_COL, GROUP_COL, RECORD_COL] if c in data_df.columns
    ]
    data = data_df[present_cols].dropna(subset=[TARGET_COL, GROUP_COL]).copy()
    X = data[feature_names]
    y = data[TARGET_COL].to_numpy(dtype=np.float64)
    groups = data[GROUP_COL].to_numpy()

    n_splits = min(n_splits_max, pd.Series(groups).nunique())
    if n_splits < 2:
        raise ValueError("Need >=2 groups for grouped OOF predictions.")

    rows: list[pd.DataFrame] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(
        GroupKFold(n_splits=n_splits).split(X, y, groups), start=1
    ):
        est = clone(estimator_template)
        est.fit(X.iloc[tr_idx], y[tr_idx])
        chunk = data.iloc[va_idx].copy()
        chunk["cv_fold"] = fold_idx
        chunk["y_true"] = y[va_idx]
        chunk["y_pred"] = est.predict(X.iloc[va_idx])
        rows.append(chunk)

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df["sq_error"] = np.square(pred_df["residual"])

    metrics = {
        "post_selection_grouped_oof_mae": float(
            mean_absolute_error(pred_df["y_true"], pred_df["y_pred"])
        ),
        "post_selection_grouped_oof_rmse": float(_safe_rmse(pred_df["y_true"], pred_df["y_pred"])),
        "post_selection_grouped_oof_r2": float(r2_score(pred_df["y_true"], pred_df["y_pred"])),
        "n_rows": int(len(pred_df)),
        "n_runs": int(pred_df[GROUP_COL].nunique()),
        "n_folds": int(n_splits),
    }
    return pred_df, metrics


def build_ensemble_predictions(
    top_rows: pd.DataFrame,
    *,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    feature_names: list[str],
    preset: str,
    random_state: int,
    use_cuda: bool,
    include_gpr: bool,
    skip_slow_models: bool,
    inner_max_splits: int,
    search_n_jobs: int,
    n_iter: int,
    doe_step: float,
    snap_predictions: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    member_predictions: list[np.ndarray] = []
    fitted_members: list[Any] = []
    member_info: list[dict[str, Any]] = []

    holdout_eval = (
        holdout_df[feature_names + [TARGET_COL, GROUP_COL]]
        .dropna(subset=[TARGET_COL, GROUP_COL])
        .copy()
    )
    y_true = holdout_eval[TARGET_COL].to_numpy(dtype=np.float64)

    specs = _resolve_model_specs(
        preset=preset,
        random_state=random_state,
        use_cuda=use_cuda,
        include_gpr=include_gpr,
        skip_slow_models=skip_slow_models,
        requested_models=None,
    )

    for _, row in top_rows.iterrows():
        model_name = str(row["model_name"])
        search = fit_model_on_train(
            train_df,
            feature_names,
            model_name,
            specs[model_name],
            inner_max_splits=inner_max_splits,
            search_n_jobs=search_n_jobs,
            preset=preset,
            n_iter=n_iter,
            random_state=random_state,
            use_cuda=use_cuda,
        )
        fitted = search.best_estimator_
        preds = fitted.predict(holdout_eval[feature_names])
        member_predictions.append(preds)
        fitted_members.append(fitted)
        member_info.append(
            {
                "model_name": model_name,
                "nested_cv_mae": float(row["mae_mean"]),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        logger.info("Ensemble member fitted: %s", model_name)

    y_pred_raw = np.mean(np.stack(member_predictions, axis=1), axis=1)
    y_pred = (
        _snap_to_grid(y_pred_raw, doe_step) if snap_predictions and doe_step > 0 else y_pred_raw
    )

    pred_df = holdout_eval[
        [c for c in [GROUP_COL, RECORD_COL, TARGET_COL] if c in holdout_eval.columns]
    ].copy()
    pred_df["y_true"] = y_true
    pred_df["y_pred_raw"] = y_pred_raw
    pred_df["y_pred"] = y_pred
    pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
    pred_df["abs_error"] = np.abs(pred_df["residual"])
    pred_df["sq_error"] = np.square(pred_df["residual"])

    metrics = {
        "ensemble_n_members": int(len(member_predictions)),
        "ensemble_mae": float(mean_absolute_error(y_true, y_pred)),
        "ensemble_rmse": float(_safe_rmse(y_true, y_pred)),
        "ensemble_r2": float(r2_score(y_true, y_pred)),
        "snapping_applied": bool(snap_predictions),
        "members": member_info,
        "fitted_members": fitted_members,
    }
    return pred_df, metrics


def _try_save_feature_importance(pipe: Pipeline, feat_cols: list[str], out_dir: Path) -> None:
    try:
        mdl = pipe.named_steps["model"]
        if hasattr(mdl, "feature_importances_"):
            imp = mdl.feature_importances_
        elif hasattr(mdl, "coef_"):
            imp = np.abs(np.ravel(mdl.coef_))
        else:
            return
        pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values(
            "importance", ascending=False
        ).to_csv(out_dir / "best_model_feature_importance.csv", index=False)
    except Exception as exc:  # pragma: no cover - best effort only
        logger.warning("Could not save feature importance: %s", exc)


def train_classical(
    features_csv: str | Path,
    out_dir: str | Path,
    *,
    holdout_runs: list[str] | None = None,
    train_fraction: float = 0.70,
    outer_max_splits: int = 5,
    inner_max_splits: int = 4,
    preset: str = "balanced",
    n_iter: int = 40,
    search_n_jobs: int = DEFAULT_SEARCH_N_JOBS,
    requested_models: list[str] | None = None,
    include_gpr: bool = False,
    skip_slow_models: bool = False,
    use_cuda: bool = False,
    ensemble_top_n: int = 1,
    snap_predictions: bool = False,
    target_mae: float = TARGET_MAE_MM,
    doe_step: float = DOE_STEP_MM,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train classical ML models with grouped holdout + nested grouped CV.

    Parameters are intentionally aligned with the newer standalone trainer,
    but this function operates on a single selected-features CSV.
    """
    preset = _normalise_preset(preset)
    out_dir = Path(out_dir)
    final_dir = out_dir / "final_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    if ensemble_top_n < 1:
        raise ValueError("ensemble_top_n must be >= 1")

    df = pd.read_csv(features_csv)
    if GROUP_COL not in df.columns:
        raise ValueError(f"Expected '{GROUP_COL}' column in features CSV.")
    feat_cols = _feature_cols(df)
    if not feat_cols:
        raise ValueError("No numeric feature columns found in features CSV.")

    logger.info("Loaded %d rows × %d features from %s", len(df), len(feat_cols), features_csv)
    logger.info(
        "Preset=%s | n_iter=%d | use_cuda=%s | ensemble_top_n=%d | skip_slow=%s",
        preset,
        n_iter,
        use_cuda,
        ensemble_top_n,
        skip_slow_models,
    )

    train_df, holdout_df = _grouped_train_holdout_split(
        df,
        holdout_runs=holdout_runs,
        train_fraction=train_fraction,
        random_state=random_state,
    )
    train_runs = sorted(train_df[GROUP_COL].astype(str).unique().tolist())
    holdout_runs_actual = sorted(holdout_df[GROUP_COL].astype(str).unique().tolist())
    logger.info(
        "Split: train=%d rows / %d runs | holdout=%d rows / %d runs",
        len(train_df),
        len(train_runs),
        len(holdout_df),
        len(holdout_runs_actual),
    )

    split_assign_cols = [GROUP_COL, TARGET_COL] + ([RECORD_COL] if RECORD_COL in df.columns else [])
    split_assign = df[split_assign_cols].copy()
    split_assign["split"] = "train"
    holdout_index = set(holdout_df.index.tolist())
    split_assign.loc[split_assign.index.isin(holdout_index), "split"] = "holdout"
    split_assign.to_csv(
        out_dir / "fixed_grouped_split_assignment.csv", index=True, index_label="row_index"
    )

    model_specs = _resolve_model_specs(
        preset=preset,
        random_state=random_state,
        use_cuda=use_cuda,
        include_gpr=include_gpr,
        skip_slow_models=skip_slow_models,
        requested_models=requested_models,
    )

    all_fold_results: list[pd.DataFrame] = []
    all_search_results: list[pd.DataFrame] = []
    for model_name, model_spec in model_specs.items():
        logger.info("Nested CV: evaluating %s", model_name)
        result_pack = evaluate_model_nested_cv(
            train_df,
            feat_cols,
            model_name,
            model_spec,
            outer_max_splits=outer_max_splits,
            inner_max_splits=inner_max_splits,
            search_n_jobs=search_n_jobs,
            preset=preset,
            n_iter=n_iter,
            random_state=random_state,
            use_cuda=use_cuda,
        )
        all_fold_results.append(result_pack["fold_results"])
        all_search_results.append(result_pack["search_results"])

    results_df = pd.concat(all_fold_results, ignore_index=True)
    inner_search_df = pd.concat(all_search_results, ignore_index=True)
    results_df.to_csv(out_dir / "nested_groupkfold_raw_results.csv", index=False)
    inner_search_df.to_csv(out_dir / "nested_inner_search_results.csv", index=False)

    summary = (
        results_df.groupby(["model_name", "n_features"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            inner_best_neg_mae_mean=("inner_best_neg_mae", "mean"),
        )
        .sort_values(["mae_mean", "rmse_mean", "r2_mean"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
    summary["goal_target_mae_mm"] = target_mae
    summary["goal_met_nested_mean"] = summary["mae_mean"] < target_mae
    summary["mae_fraction_of_doe_step"] = summary["mae_mean"] / doe_step
    summary.to_csv(out_dir / "nested_groupkfold_summary.csv", index=False)

    best_row = summary.iloc[0]
    best_model_name = str(best_row["model_name"])
    logger.info("Winner: %s with nested CV MAE %.4f", best_model_name, float(best_row["mae_mean"]))

    final_search = fit_model_on_train(
        train_df,
        feat_cols,
        best_model_name,
        model_specs[best_model_name],
        inner_max_splits=inner_max_splits,
        search_n_jobs=search_n_jobs,
        preset=preset,
        n_iter=n_iter,
        random_state=random_state,
        use_cuda=use_cuda,
    )
    best_estimator = final_search.best_estimator_
    best_params = final_search.best_params_

    pd.DataFrame(final_search.cv_results_).assign(
        params_json=lambda d: d["params"].apply(lambda x: json.dumps(x, sort_keys=True))
    ).to_csv(out_dir / "best_model_final_train_cv_results.csv", index=False)

    train_oof_pred_df, train_oof_metrics = grouped_oof_predictions(
        best_estimator,
        train_df,
        feat_cols,
        n_splits_max=outer_max_splits,
    )
    train_oof_pred_df.to_csv(
        out_dir / "train_post_selection_grouped_oof_predictions.csv", index=False
    )
    pd.DataFrame([train_oof_metrics]).to_csv(
        out_dir / "train_post_selection_grouped_oof_metrics.csv", index=False
    )

    holdout_pred_df, holdout_metrics = evaluate_on_holdout(
        best_estimator,
        holdout_df,
        feat_cols,
        target_mae=target_mae,
        doe_step=doe_step,
        snap_predictions=snap_predictions,
    )
    holdout_pred_df.to_csv(out_dir / "holdout_predictions.csv", index=False)
    pd.DataFrame([holdout_metrics]).to_csv(final_dir / "holdout_metrics.csv", index=False)

    fold_pred_matrix: list[np.ndarray] = []
    X_train = train_df[feat_cols].to_numpy(dtype=np.float64)
    y_train = train_df[TARGET_COL].to_numpy(dtype=np.float64)
    groups = train_df[GROUP_COL].to_numpy()
    X_hold = holdout_df[feat_cols].to_numpy(dtype=np.float64)
    outer_cv = GroupKFold(n_splits=min(outer_max_splits, pd.Series(groups).nunique()))
    for tr_idx, _ in outer_cv.split(X_train, y_train, groups):
        refit_search = fit_model_on_train(
            train_df.iloc[tr_idx].copy(),
            feat_cols,
            best_model_name,
            model_specs[best_model_name],
            inner_max_splits=inner_max_splits,
            search_n_jobs=search_n_jobs,
            preset=preset,
            n_iter=n_iter,
            random_state=random_state,
            use_cuda=use_cuda,
        )
        fold_pred_matrix.append(refit_search.best_estimator_.predict(X_hold))

    fold_preds_arr = np.stack(fold_pred_matrix, axis=1)
    sigma_pred = float(np.std(fold_preds_arr, axis=1).mean())
    sigma_mae = float(np.std(holdout_pred_df["residual"].to_numpy(dtype=np.float64)))
    total_uncertainty = float(np.sqrt(sigma_pred**2 + sigma_mae**2))

    ensemble_metrics_json: dict[str, Any] | None = None
    if ensemble_top_n > 1:
        ensemble_top_n = min(ensemble_top_n, len(summary))
        logger.info("Building ensemble from top-%d models", ensemble_top_n)
        ensemble_pred_df, ensemble_metrics = build_ensemble_predictions(
            summary.head(ensemble_top_n),
            train_df=train_df,
            holdout_df=holdout_df,
            feature_names=feat_cols,
            preset=preset,
            random_state=random_state,
            use_cuda=use_cuda,
            include_gpr=include_gpr,
            skip_slow_models=skip_slow_models,
            inner_max_splits=inner_max_splits,
            search_n_jobs=search_n_jobs,
            n_iter=n_iter,
            doe_step=doe_step,
            snap_predictions=snap_predictions,
        )
        ensemble_pred_df.to_csv(out_dir / "ensemble_holdout_predictions.csv", index=False)
        ensemble_members = ensemble_metrics.pop("fitted_members")
        ensemble_metrics_json = {k: v for k, v in ensemble_metrics.items()}
        pd.DataFrame([ensemble_metrics_json]).to_csv(
            final_dir / "ensemble_holdout_metrics.csv", index=False
        )
        joblib.dump(
            {
                "members": [
                    {
                        "model": fitted_member,
                        "model_name": info["model_name"],
                        "feature_cols": feat_cols,
                        "selected_features": feat_cols,
                        "nested_cv_mae": info["nested_cv_mae"],
                    }
                    for fitted_member, info in zip(
                        ensemble_members, ensemble_metrics_json["members"]
                    )
                ],
                "doe_step_mm": doe_step,
                "snap_predictions": snap_predictions,
                "target_col": TARGET_COL,
                "group_col": GROUP_COL,
                "record_col": RECORD_COL,
                "ensemble_metrics": ensemble_metrics_json,
            },
            final_dir / "ensemble_model_bundle.joblib",
        )

    bundle = {
        "model": best_estimator,
        "feature_cols": feat_cols,
        "selected_features": feat_cols,
        "best_model_name": best_model_name,
        "best_params": best_params,
        "holdout_mae": holdout_metrics["holdout_mae"],
        "sigma_pred": sigma_pred,
        "sigma_mae": sigma_mae,
        "total_uncertainty": total_uncertainty,
        "modality": "airborne_classical",
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
        "record_col": RECORD_COL,
        "train_run_ids": train_runs,
        "holdout_run_ids": holdout_runs_actual,
        "snap_predictions": snap_predictions,
        "doe_step_mm": doe_step,
        "nested_cv_winner_row": best_row.to_dict(),
        "holdout_metrics": holdout_metrics,
        "post_selection_grouped_oof_metrics": train_oof_metrics,
        "ensemble_metrics": ensemble_metrics_json,
        "use_cuda": use_cuda,
    }
    bundle_path = final_dir / "best_model_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    metadata = {
        "best_model_name": best_model_name,
        "features_csv": str(features_csv),
        "n_features": len(feat_cols),
        "n_train_rows": int(len(train_df)),
        "n_holdout_rows": int(len(holdout_df)),
        "n_train_runs": int(len(train_runs)),
        "n_holdout_runs": int(len(holdout_runs_actual)),
        "holdout_runs": holdout_runs_actual,
        "preset": preset,
        "n_iter": int(n_iter),
        "search_n_jobs": int(search_n_jobs),
        "outer_max_splits": int(outer_max_splits),
        "inner_max_splits": int(inner_max_splits),
        "requested_models": requested_models or [],
        "include_gpr": bool(include_gpr),
        "skip_slow_models": bool(skip_slow_models),
        "use_cuda": bool(use_cuda),
        "ensemble_top_n": int(ensemble_top_n),
        "snap_predictions": bool(snap_predictions),
        "holdout_metrics": holdout_metrics,
        "sigma_pred": sigma_pred,
        "sigma_mae": sigma_mae,
        "total_uncertainty": total_uncertainty,
        "best_params": best_params,
        "cv_winner": best_row.to_dict(),
        "post_selection_grouped_oof_metrics": train_oof_metrics,
        "ensemble_metrics": ensemble_metrics_json,
    }
    with open(final_dir / "best_model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    run_config = {
        "features_csv": str(features_csv),
        "out_dir": str(out_dir),
        "holdout_runs": holdout_runs or [],
        "train_fraction": float(train_fraction),
        "outer_max_splits": int(outer_max_splits),
        "inner_max_splits": int(inner_max_splits),
        "preset": preset,
        "n_iter": int(n_iter),
        "search_n_jobs": int(search_n_jobs),
        "requested_models": requested_models or [],
        "include_gpr": bool(include_gpr),
        "skip_slow_models": bool(skip_slow_models),
        "use_cuda": bool(use_cuda),
        "ensemble_top_n": int(ensemble_top_n),
        "snap_predictions": bool(snap_predictions),
        "target_mae": float(target_mae),
        "doe_step": float(doe_step),
        "random_state": int(random_state),
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    _try_save_feature_importance(best_estimator, feat_cols, final_dir)
    logger.info("Artefacts saved to %s", final_dir)

    return {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "holdout_metrics": holdout_metrics,
        "summary_df": summary,
        "nested_results_df": results_df,
        "inner_search_df": inner_search_df,
        "repeat_metrics_df": results_df,
        "train_oof_metrics": train_oof_metrics,
        "out_dir": str(out_dir),
        "bundle_path": str(bundle_path),
        "total_uncertainty": total_uncertainty,
        "ensemble_metrics": ensemble_metrics_json,
        "train_run_ids": train_runs,
        "holdout_run_ids": holdout_runs_actual,
    }
