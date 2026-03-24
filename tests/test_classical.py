"""Tests for vm_micro.classical — training and inference round-trip.

This version targets the newer classical trainer API with:
- nested grouped CV
- search preset / n_iter
- explicit inner / outer split control
- optional ensemble persistence
- CUDA flag in the public interface

The test keeps the search intentionally small so CI does not become a hostage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vm_micro.classical.inference import infer_classical
from vm_micro.classical.trainer import train_classical

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic feature dataset
# ─────────────────────────────────────────────────────────────────────────────

_DEPTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_RUNS = [f"run_{i:02d}" for i in range(10)]
_N_PER = 5  # segments per run × depth level


def _make_synthetic_features(rng: np.random.Generator) -> pd.DataFrame:
    """Create a minimal grouped feature DataFrame resembling selected features."""
    rows = []
    for run_id, run in enumerate(_RUNS):
        run_shift = rng.normal(0.0, 0.01)  # tiny run-specific offset
        for step, depth in enumerate(_DEPTHS):
            for seg in range(_N_PER):
                stem = (
                    f"{run}__seg{step * _N_PER + seg + 1:03d}__step{step:03d}__B1__depth{depth:.3f}"
                )
                rows.append(
                    {
                        "record_name": stem,
                        "recording_root": run,
                        "depth_mm": depth,
                        "modality": "airborne",
                        # signal-like features with clear relationship to depth
                        "feat_a": 2.0 * depth + run_shift + rng.normal(0, 0.03),
                        "feat_b": depth**2 + rng.normal(0, 0.02),
                        "feat_c": -0.8 * depth + rng.normal(0, 0.03),
                        "feat_d": np.sin(depth * np.pi) + rng.normal(0, 0.02),
                        "feat_noise": rng.normal(0, 1.0),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def features_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("features")
    rng = np.random.default_rng(42)
    df = _make_synthetic_features(rng)
    p = tmp / "features_selected.csv"
    df.to_csv(p, index=False)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _extract_metric(metrics: dict, *keys: str) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    raise AssertionError(f"Could not find any of {keys} in metrics: {list(metrics.keys())}")


def _extract_holdout_mae(metrics: dict) -> float:
    return _extract_metric(metrics, "holdout_mae", "mae")


def _extract_holdout_r2(metrics: dict) -> float:
    return _extract_metric(metrics, "holdout_r2", "r2")


def _extract_ensemble_mae(metrics: dict) -> float:
    return _extract_metric(metrics, "ensemble_mae", "mae")


def _extract_ensemble_r2(metrics: dict) -> float:
    return _extract_metric(metrics, "ensemble_r2", "r2")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trained_result(
    features_csv: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict:
    out = tmp_path_factory.mktemp("classical_out")

    # Keep this intentionally small but real:
    # - fast preset
    # - tiny inner/outer CV
    # - restricted models
    # - top-2 ensemble
    result = train_classical(
        features_csv=features_csv,
        out_dir=out,
        holdout_runs=["run_08", "run_09"],
        train_fraction=0.70,
        outer_max_splits=3,
        inner_max_splits=2,
        preset="fast",
        n_iter=4,
        search_n_jobs=1,
        requested_models=["ridge", "extra_trees"],
        include_gpr=False,
        skip_slow_models=True,
        use_cuda=False,
        ensemble_top_n=2,
        snap_predictions=False,
        target_mae=0.05,
        doe_step=0.10,
        random_state=42,
    )
    return result


@pytest.fixture(scope="module")
def trained_bundle(trained_result: dict) -> Path:
    return Path(trained_result["bundle_path"])


def test_train_produces_bundle(trained_bundle: Path):
    assert trained_bundle.exists()
    assert trained_bundle.name.endswith(".joblib")


def test_train_result_has_expected_keys(trained_result: dict):
    assert "best_model_name" in trained_result
    assert "holdout_metrics" in trained_result
    assert "bundle_path" in trained_result
    assert "out_dir" in trained_result
    assert "total_uncertainty" in trained_result


def test_train_produces_metadata(trained_bundle: Path):
    meta_path = trained_bundle.parent / "best_model_metadata.json"
    assert meta_path.exists()

    meta = _read_json(meta_path)
    assert "best_model_name" in meta
    assert "holdout_metrics" in meta

    mae = _extract_holdout_mae(meta["holdout_metrics"])
    assert mae < 0.5  # generous sanity bound for synthetic data


def test_train_uncertainty_positive(trained_result: dict, trained_bundle: Path):
    assert float(trained_result["total_uncertainty"]) >= 0.0

    meta_path = trained_bundle.parent / "best_model_metadata.json"
    meta = _read_json(meta_path)
    assert float(meta["total_uncertainty"]) >= 0.0


def test_train_produces_nested_cv_summary(trained_result: dict):
    out_dir = Path(trained_result["out_dir"])

    # Newer implementation should save nested_groupkfold_summary.csv.
    # If the internal filename changes again, humans can enjoy updating one line.
    summary_path = out_dir / "nested_groupkfold_summary.csv"
    assert summary_path.exists()

    df = pd.read_csv(summary_path)
    assert len(df) > 0


def test_train_records_holdout_runs_if_available(trained_result: dict):
    # Only assert this if the newer result schema exposes it.
    holdout_ids = trained_result.get("holdout_run_ids")
    if holdout_ids is not None:
        assert set(holdout_ids) == {"run_08", "run_09"}


def test_train_ensemble_metrics_present_when_enabled(trained_result: dict):
    ens = trained_result.get("ensemble_metrics")
    assert ens is not None, "Expected ensemble_metrics when ensemble_top_n > 1"

    mae = _extract_ensemble_mae(ens)
    assert mae < 0.5

    assert "members" in ens
    assert "ensemble_n_members" in ens
    assert int(ens["ensemble_n_members"]) >= 1


def test_train_holdout_quality_reasonable(trained_result: dict):
    metrics = trained_result["holdout_metrics"]
    mae = _extract_holdout_mae(metrics)
    r2 = _extract_holdout_r2(metrics)

    assert mae < 0.5
    assert r2 > 0.0


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
    assert len(df) > 0
    assert "y_pred" in df.columns
    assert df["y_pred"].between(0.0, 1.5).all(), "Predictions outside plausible range"


def test_inference_mae_reasonable(
    trained_bundle: Path,
    features_csv: Path,
):
    """Synthetic data has strong signal; inference should remain sensible."""
    df = infer_classical(trained_bundle, features_csv)

    assert "y_pred" in df.columns
    assert "depth_mm" in df.columns

    mae = float(np.mean(np.abs(df["y_pred"] - df["depth_mm"])))
    assert mae < 0.5
