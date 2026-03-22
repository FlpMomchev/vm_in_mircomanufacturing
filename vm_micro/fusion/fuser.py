"""vm_micro.fusion.fuser
~~~~~~~~~~~~~~~~~~~~~~~~
Self-contained fusion layer.

Architecture
------------
Stage 1 – intra-modality fusion
    airborne_feat  + airborne_dl  → airborne_ensemble
    structure_feat + structure_dl → structure_ensemble  (future)

Stage 2 – inter-modality fusion
    airborne_ensemble + structure_ensemble → final prediction

Each stage uses **inverse-validation-MAE weighted averaging**:
    w_i  = 1 / (MAE_i + ε)          (raw weight)
    w̃_i  = max(w_i, min_weight)      (floor to avoid zero weight)
    w̃_i  = w̃_i / sum(w̃_j)           (normalise)

Uncertainty is propagated as:
    σ_out = sqrt( sum_i  (w̃_i * σ_i)² )

Interface contract
------------------
Every upstream model produces a :class:`PredictionBundle`.  The fuser
consumes a *list* of bundles and returns a single merged bundle.

To swap out the fusion strategy later, only this file needs to change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PredictionBundle:
    """Standardised prediction output from any upstream model.

    Parameters
    ----------
    modality        : One of ``'airborne_classical'``, ``'airborne_dl'``,
                      ``'structure_classical'``, ``'structure_dl'``, or a
                      merged name like ``'airborne_ensemble'``.
    record_names    : 1-D array of segment / file identifiers.
    y_pred          : 1-D float array of depth predictions (mm).
    sigma           : Per-prediction uncertainty (std, mm).  Set to scalar
                      if homoscedastic.
    validation_mae  : Scalar MAE on a held-out validation set.  Used for
                      weighting in the fusion.
    y_true          : Optional ground-truth labels (mm); used for evaluation.
    class_probs     : Optional (N, C) probability matrix (classification).
    metadata        : Free-form dict for diagnostics.
    """
    modality:       str
    record_names:   np.ndarray
    y_pred:         np.ndarray
    sigma:          np.ndarray
    validation_mae: float
    y_true:         np.ndarray | None = None
    class_probs:    np.ndarray | None = None
    metadata:       dict[str, Any]    = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.record_names = np.asarray(self.record_names)
        self.y_pred       = np.asarray(self.y_pred, dtype=np.float64)
        self.sigma        = np.broadcast_to(
            np.asarray(self.sigma, dtype=np.float64), self.y_pred.shape
        ).copy()
        if self.y_true is not None:
            self.y_true = np.asarray(self.y_true, dtype=np.float64)

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "record_name": self.record_names,
            "y_pred":      self.y_pred,
            "sigma":       self.sigma,
            "modality":    self.modality,
            "validation_mae": self.validation_mae,
        })
        if self.y_true is not None:
            df["depth_mm"] = self.y_true
            df["residual"] = self.y_pred - self.y_true
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Core fusion functions
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-9


def _inverse_mae_weights(maes: np.ndarray, min_weight: float = 0.05) -> np.ndarray:
    """Compute normalised inverse-MAE weights with a floor."""
    raw = 1.0 / (np.asarray(maes, dtype=np.float64) + _EPS)
    raw = np.maximum(raw, min_weight)
    return raw / raw.sum()


def _align_records(
    bundles: list[PredictionBundle],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Align multiple bundles to the intersection of their record names.

    Returns
    -------
    common_records : sorted array of shared record names
    preds_list     : list of aligned y_pred arrays
    sigmas_list    : list of aligned sigma arrays
    """
    common = set(bundles[0].record_names.tolist())
    for b in bundles[1:]:
        common &= set(b.record_names.tolist())
    common_records = np.array(sorted(common))

    preds_list:  list[np.ndarray] = []
    sigmas_list: list[np.ndarray] = []
    for b in bundles:
        idx = {r: i for i, r in enumerate(b.record_names)}
        ii  = [idx[r] for r in common_records]
        preds_list.append(b.y_pred[ii])
        sigmas_list.append(b.sigma[ii])

    return common_records, preds_list, sigmas_list


def _fuse(
    bundles: list[PredictionBundle],
    fused_modality: str,
    min_weight: float = 0.05,
) -> PredictionBundle:
    """Weighted average fusion of N bundles with Gaussian uncertainty propagation."""
    if not bundles:
        raise ValueError("Need at least one bundle to fuse.")
    if len(bundles) == 1:
        b = bundles[0]
        return PredictionBundle(
            modality=fused_modality,
            record_names=b.record_names.copy(),
            y_pred=b.y_pred.copy(),
            sigma=b.sigma.copy(),
            validation_mae=b.validation_mae,
            y_true=b.y_true.copy() if b.y_true is not None else None,
            metadata={"source_modalities": [b.modality]},
        )

    maes   = np.array([b.validation_mae for b in bundles])
    weights = _inverse_mae_weights(maes, min_weight=min_weight)

    common, preds_list, sigmas_list = _align_records(bundles)

    y_fused = sum(w * p for w, p in zip(weights, preds_list))  # type: ignore[arg-type]
    sigma_fused = np.sqrt(sum((w * s) ** 2 for w, s in zip(weights, sigmas_list)))  # type: ignore[arg-type]
    mae_fused = float(np.sum(weights * maes))

    # Propagate ground truth if all bundles have it
    y_true_fused: np.ndarray | None = None
    if all(b.y_true is not None for b in bundles):
        _, trues, _ = _align_records([
            PredictionBundle(b.modality, b.record_names, b.y_true, b.sigma, b.validation_mae)  # type: ignore[arg-type]
            for b in bundles
        ])
        y_true_fused = trues[0]  # they should all be the same

    return PredictionBundle(
        modality=fused_modality,
        record_names=common,
        y_pred=np.asarray(y_fused),
        sigma=np.asarray(sigma_fused),
        validation_mae=mae_fused,
        y_true=y_true_fused,
        metadata={
            "source_modalities": [b.modality for b in bundles],
            "weights":           weights.tolist(),
            "source_maes":       maes.tolist(),
        },
    )


def fuse_intra_modality(
    classical_bundle: PredictionBundle,
    dl_bundle: PredictionBundle,
    modality_name: str,
    min_weight: float = 0.05,
) -> PredictionBundle:
    """Stage 1: fuse classical + DL bundles for a single modality.

    Example::

        airborne = fuse_intra_modality(air_cls, air_dl, "airborne_ensemble")
    """
    return _fuse([classical_bundle, dl_bundle], fused_modality=modality_name, min_weight=min_weight)


def fuse_modalities(
    *modality_bundles: PredictionBundle,
    min_weight: float = 0.05,
) -> PredictionBundle:
    """Stage 2: fuse any number of modality-level bundles into a final prediction.

    Example::

        final = fuse_modalities(airborne_ensemble, structure_ensemble)
    """
    return _fuse(list(modality_bundles), fused_modality="final_fusion", min_weight=min_weight)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load bundles from disk and fuse
# ─────────────────────────────────────────────────────────────────────────────

def load_bundle_from_csv(
    csv_path: str | Path,
    modality: str,
    validation_mae: float,
) -> PredictionBundle:
    """Build a :class:`PredictionBundle` from an inference CSV.

    The CSV is expected to have at least: ``record_name``, ``y_pred``.
    Optional: ``depth_mm`` (y_true), ``sigma``.
    """
    df = pd.read_csv(csv_path)
    sigma = df["sigma"].to_numpy() if "sigma" in df.columns else np.zeros(len(df))
    y_true = df["depth_mm"].to_numpy() if "depth_mm" in df.columns else None
    return PredictionBundle(
        modality=modality,
        record_names=df["record_name"].to_numpy(),
        y_pred=df["y_pred"].to_numpy(),
        sigma=sigma,
        validation_mae=validation_mae,
        y_true=y_true,
    )


def save_fusion_report(
    bundle: PredictionBundle,
    out_dir: str | Path,
) -> None:
    """Save the fused predictions and a JSON metadata report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle.to_dataframe().to_csv(out_dir / "fusion_predictions.csv", index=False)

    report = {
        "modality": bundle.modality,
        "n_predictions": int(len(bundle.y_pred)),
        "validation_mae_fused": float(bundle.validation_mae),
        "mean_sigma": float(np.mean(bundle.sigma)),
        "metadata": bundle.metadata,
    }
    if bundle.y_true is not None:
        residuals = bundle.y_pred - bundle.y_true
        report["holdout_mae"]   = float(np.mean(np.abs(residuals)))
        report["holdout_rmse"]  = float(np.sqrt(np.mean(residuals ** 2)))
        report["mean_uncertainty"] = float(np.mean(bundle.sigma))

    with open(out_dir / "fusion_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
