"""vm_micro.features.selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Feature selection pipeline (modality-agnostic).

Uses the "inverted-cone" grouped strategy.

1. Near-constant and high-missingness columns dropped.
2. Low target-correlation columns dropped (Spearman threshold).
2b. (Optional) Partial-correlation filter: drop features whose
    Spearman correlation with the target does not survive after
    controlling for a confound column (e.g. duration_s).
    Enabled via ``min_partial_r`` in ``SelectionConfig``.
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
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils import get_logger

logger = get_logger(__name__)

# Columns that are metadata, not features
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
    "slot",
    "mic_dx_mm",
    "mic_dy_mm",
    "mic_r_mm",
    "mic_log_r",
    "mic_angle_rad",
}


@dataclass
class SelectionConfig:
    target_col: str = "depth_mm"
    record_col: str = "record_name"
    group_col: str = "recording_root"
    max_missing_frac: float = 0.20
    near_constant_std: float = 1e-10
    min_target_abs_spearman: float = 0.10
    min_partial_r: float | None = None
    partial_control_col: str = "duration_s"
    vif_threshold: float = 5.0
    intercorr_threshold: float = 0.75
    preselect_top_n: int = 60
    final_max_features: int | None = None
    grouped_cv_folds: int = 5
    seed: int = 42


def _feature_count_sweep(
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: list[str],
    ranking: pd.Series,
    groups: np.ndarray,
    candidates: list[int] | None = None,
    n_folds: int = 5,
    seed: int = 42,
    plot_path: str | Path | None = None,
) -> tuple[int, pd.DataFrame]:
    if candidates is None:
        max_n = min(len(feat_cols), 30)
        candidates = sorted(set([3, 5, 8, 10, 15, 20, max_n]) & set(range(1, max_n + 1)))

    ranked_features = list(ranking.index)
    feat_to_idx = {f: i for i, f in enumerate(feat_cols)}

    models = {
        "ridge": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "extratrees": lambda: ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1),
    }

    rows: list[dict] = []
    gkf = GroupKFold(n_splits=n_folds)

    for n_feat in candidates:
        top_feats = ranked_features[:n_feat]
        col_idx = [feat_to_idx[f] for f in top_feats]
        X_sub = X[:, col_idx]

        for model_name, model_fn in models.items():
            fold_maes = []
            for train_idx, val_idx in gkf.split(X_sub, y, groups):
                model = model_fn()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_sub[train_idx], y[train_idx])
                preds = model.predict(X_sub[val_idx])
                fold_maes.append(float(np.mean(np.abs(y[val_idx] - preds))))

            rows.append(
                {
                    "n_features": n_feat,
                    "model": model_name,
                    "mae_mean": float(np.mean(fold_maes)),
                    "mae_std": float(np.std(fold_maes)),
                    "mae_se": float(np.std(fold_maes) / np.sqrt(len(fold_maes))),
                }
            )

    sweep_df = pd.DataFrame(rows)

    # 1-SE rule: rule on best model only
    # the smallest n_features whose mean MAE is within 1 SE of that min
    best_model = sweep_df.groupby("model")["mae_mean"].min().idxmin()
    model_df = sweep_df[sweep_df["model"] == best_model]
    best_row = model_df.loc[model_df["mae_mean"].idxmin()]
    threshold = best_row["mae_mean"] + best_row["mae_se"]
    eligible = model_df[model_df["mae_mean"] <= threshold]
    best_n = int(eligible["n_features"].min())

    logger.info(
        "Feature count sweep: global min MAE=%.4f at n=%d, 1-SE threshold=%.4f  recommended n=%d",
        best_row["mae_mean"],
        int(best_row["n_features"]),
        threshold,
        best_n,
    )

    if plot_path is not None:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model_name, grp in sweep_df.groupby("model"):
            grp = grp.sort_values("n_features")
            ax.errorbar(
                grp["n_features"],
                grp["mae_mean"],
                yerr=grp["mae_se"],
                marker="o",
                capsize=3,
                label=model_name,
            )
        ax.axvline(best_n, color="gray", linestyle="--", alpha=0.5, label=f"selected n={best_n}")
        ax.set_xlabel("Number of selected features")
        ax.set_ylabel("Grouped nested CV MAE")
        ax.legend()
        ax.set_xticks(candidates)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info("Saved feature count sweep plot to %s", plot_path)

    return best_n, sweep_df


def select_features(
    df: pd.DataFrame,
    cfg: SelectionConfig | None = None,
    out_csv: str | Path | None = None,
    sweep_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if cfg is None:
        cfg = SelectionConfig()

    target = cfg.target_col
    if target not in df.columns:
        raise ValueError(f"Target column {target!r} not found in DataFrame.")

    # Identify feature columns
    non_feat = _META_COLS | {target}
    feat_cols = [
        c for c in df.columns if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])
    ]
    logger.info("Starting selection: %d feature candidates", len(feat_cols))

    X_raw = df[feat_cols].copy()
    y = df[target].to_numpy(dtype=np.float64)

    #  Step 1: impute & drop near-constant / high-missing
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

    #  Step 2: low Spearman correlation with target
    corr = np.array([abs(float(spearmanr(X[:, i], y).statistic)) for i in range(X.shape[1])])
    corr_mask = corr >= cfg.min_target_abs_spearman
    X = X[:, corr_mask]
    feat_cols = [f for f, k in zip(feat_cols, corr_mask) if k]
    logger.info("After Spearman filter: %d features", len(feat_cols))

    #  Step 2b (optional): partial-correlation filter
    if cfg.min_partial_r is not None:
        ctrl_col = cfg.partial_control_col
        if ctrl_col not in df.columns:
            logger.warning(
                "Partial-correlation control column %r not found  skipping filter.", ctrl_col
            )
        else:
            z = df[ctrl_col].to_numpy(dtype=np.float64)
            partial_r, X, feat_cols = _partial_corr_filter(X, y, z, feat_cols, cfg.min_partial_r)
            logger.info(
                "After partial-r filter (|r|%.2f, controlling for %s): %d features",
                cfg.min_partial_r,
                ctrl_col,
                len(feat_cols),
            )

    #  Step 3: intercorrelation pruning (greedy, keep highest |corr_target|)
    # Recompute target correlations for the current feature set (may have
    # changed after the optional partial-r filter in step 2b).
    corr_current = np.array(
        [abs(float(spearmanr(X[:, i], y).statistic)) for i in range(X.shape[1])]
    )
    if X.shape[1] > 1:
        feat_cols, X = _intercorr_prune(X, feat_cols, corr_current, cfg.intercorr_threshold)
    logger.info("After intercorr pruning: %d features", len(feat_cols))

    #  Step 4: Pre-select top-N by Spearman before expensive methods
    if X.shape[1] > cfg.preselect_top_n:
        corr2 = np.array([abs(float(spearmanr(X[:, i], y).statistic)) for i in range(X.shape[1])])
        top_idx = np.argsort(corr2)[::-1][: cfg.preselect_top_n]
        X = X[:, top_idx]
        feat_cols = [feat_cols[i] for i in top_idx]

    #  Step 5: Consensus ranking
    scores = _consensus_scores(X, y, feat_cols, cfg)
    ranking = pd.Series(scores, index=feat_cols).sort_values(ascending=False)

    selected = list(ranking.index[: cfg.final_max_features])
    logger.info("Selected %d features: %s", len(selected), selected[:5])

    #  Step 6: Feature count sweep (always runs for diagnostics)
    groups = df[cfg.group_col].astype(str).to_numpy()

    if sweep_dir is not None:
        sweep_dir = Path(sweep_dir)
        sweep_dir.mkdir(parents=True, exist_ok=True)
        plot_path = sweep_dir / "nested_cv_mae.png"
    else:
        plot_path = None

    recommended_n, sweep_df = _feature_count_sweep(
        X=X,
        y=y,
        feat_cols=feat_cols,
        ranking=ranking,
        groups=groups,
        n_folds=cfg.grouped_cv_folds,
        seed=cfg.seed,
        plot_path=plot_path,
    )

    if sweep_dir is not None:
        sweep_df.to_csv(sweep_dir / "feature_count_sweep.csv", index=False)

    #  Output
    keep_cols = [c for c in df.columns if c in selected or c in _META_COLS or c == target]
    df_out = df[keep_cols].copy()

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)
        logger.info("Saved selected features to %s", out_csv)

    return df_out, selected


#
# Helpers
#


def _partial_corr_filter(
    X: np.ndarray, y: np.ndarray, z: np.ndarray, feat_cols: list[str], threshold: float
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from scipy.stats import rankdata

    y_rank = rankdata(y)
    z_rank = rankdata(z).reshape(-1, 1)

    # Residualise y on z (rank space)
    coef_y = np.linalg.lstsq(z_rank, y_rank, rcond=None)[0]
    res_y = y_rank - z_rank @ coef_y

    partial_r = np.empty(X.shape[1], dtype=np.float64)
    for i in range(X.shape[1]):
        x_rank = rankdata(X[:, i])
        coef_x = np.linalg.lstsq(z_rank, x_rank, rcond=None)[0]
        res_x = x_rank - z_rank @ coef_x
        partial_r[i] = np.corrcoef(res_x, res_y)[0, 1]

    mask = np.abs(partial_r) >= threshold
    return partial_r[mask], X[:, mask], [f for f, k in zip(feat_cols, mask) if k]


def _intercorr_prune(
    X: np.ndarray,
    feat_cols: list[str],
    target_corr: np.ndarray,
    threshold: float,
) -> tuple[list[str], np.ndarray]:
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
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("enet", ElasticNet(random_state=cfg.seed, max_iter=5000))]
    )
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
