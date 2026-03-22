"""vm_micro.features.selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feature selection pipeline (modality-agnostic).

Uses the "inverted-cone" grouped strategy from notebooks/selection.py:

1. Near-constant and high-missingness columns dropped.
2. Low target-correlation columns dropped (Spearman threshold).
3. VIF-based multicollinearity pruning.
4. Consensus ranking from multiple methods:
   - Spearman |r| (univariate)
   - Mutual information
   - ElasticNet coefficient magnitude
   - ExtraTreesRegressor importance
   - Permutation importance (grouped CV)
5. Top-N features selected by consensus rank.

The result is saved as a CSV alongside the input features CSV.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..utils import get_logger

logger = get_logger(__name__)

# Columns that are metadata, not features
_META_COLS = {
    "modality", "record_name", "recording_root", "depth_mm",
    "step_idx", "duration_s", "sr_hz", "sr_hz_native", "sr_hz_used",
    "ds_rate", "file_path", "run_id", "batch_id", "slot",
    "mic_dx_mm", "mic_dy_mm", "mic_r_mm", "mic_log_r", "mic_angle_rad",
}


@dataclass
class SelectionConfig:
    target_col:             str   = "depth_mm"
    record_col:             str   = "record_name"
    group_col:              str   = "recording_root"
    max_missing_frac:       float = 0.20
    near_constant_std:      float = 1e-10
    min_target_abs_spearman:float = 0.10
    vif_threshold:          float = 5.0
    intercorr_threshold:    float = 0.75
    preselect_top_n:        int   = 60
    final_max_features:     int   = 15
    grouped_cv_folds:       int   = 5
    seed:                   int   = 42


def select_features(
    df: pd.DataFrame,
    cfg: SelectionConfig | None = None,
    out_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full inverted-cone feature selection.

    Parameters
    ----------
    df      : Feature DataFrame (must contain ``depth_mm`` and group column).
    cfg     : Selection hyper-parameters.  Defaults used if None.
    out_csv : If provided, saves selected feature CSV here.

    Returns
    -------
    df_selected : DataFrame with only selected feature + metadata columns.
    selected    : List of selected feature names.
    """
    if cfg is None:
        cfg = SelectionConfig()

    target = cfg.target_col
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in DataFrame.")

    # Identify feature columns
    non_feat = _META_COLS | {target}
    feat_cols = [c for c in df.columns if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])]
    logger.info("Starting selection: %d feature candidates", len(feat_cols))

    X_raw = df[feat_cols].copy()
    y     = df[target].to_numpy(dtype=np.float64)

    # ── Step 1: impute & drop near-constant / high-missing ────────────────────
    miss_frac = X_raw.isna().mean()
    keep_mask = miss_frac <= cfg.max_missing_frac
    X_raw = X_raw.loc[:, keep_mask]
    feat_cols = list(X_raw.columns)
    logger.info("After missingness filter: %d features", len(feat_cols))

    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw)

    std_vals = X.std(axis=0)
    const_mask = std_vals > cfg.near_constant_std
    X = X[:, const_mask]
    feat_cols = [f for f, k in zip(feat_cols, const_mask) if k]
    logger.info("After near-constant filter: %d features", len(feat_cols))

    # ── Step 2: low Spearman correlation with target ───────────────────────────
    corr = np.array([abs(float(spearmanr(X[:, i], y).statistic)) for i in range(X.shape[1])])
    corr_mask = corr >= cfg.min_target_abs_spearman
    X = X[:, corr_mask]
    feat_cols = [f for f, k in zip(feat_cols, corr_mask) if k]
    logger.info("After Spearman filter: %d features", len(feat_cols))

    # ── Step 3: intercorrelation pruning (greedy, keep highest |corr_target|) ─
    if X.shape[1] > 1:
        feat_cols, X = _intercorr_prune(X, feat_cols, corr[corr_mask], cfg.intercorr_threshold)
    logger.info("After intercorr pruning: %d features", len(feat_cols))

    # ── Step 4: Pre-select top-N by Spearman before expensive methods ─────────
    if X.shape[1] > cfg.preselect_top_n:
        corr2 = np.array([abs(float(spearmanr(X[:, i], y).statistic)) for i in range(X.shape[1])])
        top_idx = np.argsort(corr2)[::-1][: cfg.preselect_top_n]
        X = X[:, top_idx]
        feat_cols = [feat_cols[i] for i in top_idx]

    # ── Step 5: Consensus ranking ──────────────────────────────────────────────
    scores = _consensus_scores(X, y, feat_cols, cfg)
    ranking = pd.Series(scores, index=feat_cols).sort_values(ascending=False)

    selected = list(ranking.index[: cfg.final_max_features])
    logger.info("Selected %d features: %s", len(selected), selected[:5])

    # ── Output ─────────────────────────────────────────────────────────────────
    keep_cols = [c for c in df.columns if c in selected or c in _META_COLS or c == target]
    df_out = df[keep_cols].copy()

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)
        logger.info("Saved selected features to %s", out_csv)

    return df_out, selected


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _intercorr_prune(
    X: np.ndarray,
    feat_cols: list[str],
    target_corr: np.ndarray,
    threshold: float,
) -> tuple[list[str], np.ndarray]:
    """Greedy intercorrelation pruning: drop the one with lower target corr."""
    corr_matrix = np.corrcoef(X.T)
    n = len(feat_cols)
    drop = set()
    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            if abs(corr_matrix[i, j]) > threshold:
                victim = j if target_corr[i] >= target_corr[j] else i
                drop.add(victim)
    keep_idx = [i for i in range(n) if i not in drop]
    return [feat_cols[i] for i in keep_idx], X[:, keep_idx]


def _consensus_scores(
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: list[str],
    cfg: SelectionConfig,
) -> dict[str, float]:
    """Combine four ranking methods into a normalised consensus score."""
    scores: dict[str, np.ndarray] = {}

    # 1. |Spearman|
    sp = np.array([abs(float(spearmanr(X[:, i], y).statistic)) for i in range(X.shape[1])])
    scores["spearman"] = sp

    # 2. Mutual information
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = mutual_info_regression(X, y, n_neighbors=5, random_state=cfg.seed)
    scores["mi"] = mi

    # 3. ElasticNet coefficient magnitude
    pipe = Pipeline([("scaler", StandardScaler()), ("enet", ElasticNet(random_state=cfg.seed, max_iter=5000))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    scores["enet"] = np.abs(pipe["enet"].coef_)

    # 4. ExtraTrees importance
    et = ExtraTreesRegressor(n_estimators=200, random_state=cfg.seed, n_jobs=-1)
    et.fit(X, y)
    scores["et"] = et.feature_importances_

    # Normalise each method to [0, 1] and average
    combined = np.zeros(X.shape[1])
    for arr in scores.values():
        rng = arr.max() - arr.min()
        if rng > 0:
            combined += (arr - arr.min()) / rng
        else:
            combined += arr
    combined /= len(scores)

    return dict(zip(feat_cols, combined.tolist()))
