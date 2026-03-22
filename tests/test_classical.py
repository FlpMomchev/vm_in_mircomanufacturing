"""Tests for vm_micro.classical — training and inference round-trip."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vm_micro.classical.trainer import train_classical
from vm_micro.classical.inference import infer_classical


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic feature dataset
# ─────────────────────────────────────────────────────────────────────────────

_DEPTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_RUNS   = [f"run_{i:02d}" for i in range(10)]
_N_PER  = 5   # segments per run × depth level


def _make_synthetic_features(rng: np.random.Generator) -> pd.DataFrame:
    """Create a minimal feature DataFrame that looks like real extraction output."""
    rows = []
    for run_id, run in enumerate(_RUNS):
        for step, depth in enumerate(_DEPTHS):
            for seg in range(_N_PER):
                stem = f"{run}__seg{step * _N_PER + seg + 1:03d}__step{step:03d}__B1__depth{depth:.3f}"
                row = {
                    "record_name":     stem,
                    "recording_root":  run,
                    "depth_mm":        depth,
                    "modality":        "airborne",
                    # synthetic features — correlated with depth
                    "feat_a": depth * 2.0 + rng.normal(0, 0.1),
                    "feat_b": depth ** 2   + rng.normal(0, 0.05),
                    "feat_c": -depth       + rng.normal(0, 0.15),
                    "feat_d": rng.normal(0, 1.0),  # noise feature
                }
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def features_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("features")
    rng = np.random.default_rng(42)
    df  = _make_synthetic_features(rng)
    p   = tmp / "features_selected.csv"
    df.to_csv(p, index=False)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_bundle(features_csv: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("classical_out")
    result = train_classical(
        features_csv=features_csv,
        out_dir=out,
        holdout_runs=["run_08", "run_09"],
        train_fraction=0.70,
        n_cv_folds=3,
        skip_slow_models=True,
        seed=42,
    )
    return Path(result["bundle_path"])


def test_train_produces_bundle(trained_bundle: Path):
    assert trained_bundle.exists()


def test_train_produces_metadata(trained_bundle: Path):
    meta_path = trained_bundle.parent / "best_model_metadata.json"
    assert meta_path.exists()
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert "best_model_name" in meta
    assert meta["holdout_metrics"]["mae"] < 1.0  # sanity: depth range is 0.1–1.0 mm


def test_train_uncertainty_positive(trained_bundle: Path):
    meta_path = trained_bundle.parent / "best_model_metadata.json"
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert meta["total_uncertainty"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def test_inference_round_trip(
    trained_bundle: Path,
    features_csv: Path,
    tmp_path: Path,
):
    out_csv = tmp_path / "predictions.csv"
    df = infer_classical(trained_bundle, features_csv, out_csv=out_csv)

    assert out_csv.exists()
    assert "y_pred" in df.columns
    assert len(df) > 0
    assert df["y_pred"].between(0.0, 1.5).all(), "Predictions outside plausible range"


def test_inference_mae_reasonable(
    trained_bundle: Path,
    features_csv: Path,
    tmp_path: Path,
):
    """Training data has strong signal; MAE should be well below 0.5 mm."""
    df = infer_classical(trained_bundle, features_csv)
    if "depth_mm" in df.columns:
        mae = float(np.mean(np.abs(df["y_pred"] - df["depth_mm"])))
        assert mae < 0.5
