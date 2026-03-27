"""Tests for vm_micro.classical  training and inference round-trip.

This version is written against the current classical trainer behavior:
- grouped internal split + optional external holdout
- nested CV summary persisted to disk
- validation / test / holdout metrics
- optional ensemble persistence
- bundle + metadata round-trip through inference
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vm_micro.classical.inference import infer_classical
from vm_micro.classical.trainer import train_classical

#
# Synthetic feature dataset
#

_DEPTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
_RUNS = [f"run_{i:02d}" for i in range(10)]
_N_PER = 5


def _make_synthetic_features(rng: np.random.Generator) -> pd.DataFrame:
    """Create a grouped synthetic feature set with clear signal."""
    rows = []
    for run_id, run in enumerate(_RUNS):
        run_shift = rng.normal(0.0, 0.01)
        run_scale = 1.0 + rng.normal(0.0, 0.01)

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
                        "feat_a": run_scale * (2.0 * depth) + run_shift + rng.normal(0, 0.03),
                        "feat_b": depth**2 + rng.normal(0, 0.02),
                        "feat_c": -0.8 * depth + rng.normal(0, 0.03),
                        "feat_d": np.sin(depth * np.pi) + rng.normal(0, 0.02),
                        "feat_e": np.cos(depth * np.pi * 0.5) + rng.normal(0, 0.02),
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


#
# Helpers
#


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _first_existing(parent: Path, *names: str) -> Path:
    for name in names:
        p = parent / name
        if p.exists():
            return p
    raise AssertionError(f"None of the expected files exist in {parent}: {names}")


def _extract_metric(metrics: dict, *keys: str) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    raise AssertionError(f"Could not find any of {keys} in metrics keys={list(metrics.keys())}")


def _extract_mae(metrics: dict) -> float:
    return _extract_metric(
        metrics,
        "holdout_mae",
        "validation_mae",
        "test_mae",
        "ensemble_mae",
        "mae",
    )


def _extract_r2(metrics: dict) -> float:
    return _extract_metric(
        metrics,
        "holdout_r2",
        "validation_r2",
        "test_r2",
        "ensemble_r2",
        "r2",
    )


def _bundle_path(result: dict) -> Path:
    for key in ("bundle_path", "best_bundle_path"):
        if key in result and result[key]:
            return Path(result[key])
    raise AssertionError(f"No bundle path key found in result keys={list(result.keys())}")


def _out_dir(result: dict) -> Path:
    if "out_dir" in result and result["out_dir"]:
        return Path(result["out_dir"])
    raise AssertionError(f"No out_dir found in result keys={list(result.keys())}")


#
# Training
#


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
    return _bundle_path(trained_result)


def test_train_produces_bundle(trained_bundle: Path):
    assert trained_bundle.exists()
    assert trained_bundle.suffix == ".joblib"


def test_train_result_has_expected_keys(trained_result: dict):
    expected_any = {
        "best_model_name",
        "validation_metrics",
        "test_metrics",
        "holdout_metrics",
        "total_uncertainty",
    }
    missing = [k for k in expected_any if k not in trained_result]
    assert not missing, f"Missing expected keys: {missing}"

    assert _bundle_path(trained_result).exists()
    assert _out_dir(trained_result).exists()


def test_train_produces_metadata(trained_bundle: Path):
    meta_path = _first_existing(
        trained_bundle.parent,
        "best_model_metadata.json",
        "metadata.json",
    )
    meta = _read_json(meta_path)

    assert any(k in meta for k in ("best_model_name", "model_name"))
    assert "holdout_metrics" in meta
    assert "validation_metrics" in meta
    assert "test_metrics" in meta

    mae = _extract_mae(meta["holdout_metrics"])
    assert mae < 0.5


def test_train_uncertainty_positive(trained_result: dict, trained_bundle: Path):
    assert float(trained_result["total_uncertainty"]) >= 0.0

    meta_path = _first_existing(
        trained_bundle.parent,
        "best_model_metadata.json",
        "metadata.json",
    )
    meta = _read_json(meta_path)
    assert "total_uncertainty" in meta
    assert float(meta["total_uncertainty"]) >= 0.0


def test_train_produces_nested_cv_summary(trained_result: dict):
    out_dir = _out_dir(trained_result)
    summary_path = _first_existing(
        out_dir,
        "nested_groupkfold_summary.csv",
        "nested_cv_summary.csv",
    )

    df = pd.read_csv(summary_path)
    assert len(df) > 0
    assert any(col in df.columns for col in ("model_name", "model", "name"))


def test_train_records_holdout_runs_if_available(trained_result: dict):
    holdout_ids = trained_result.get("holdout_run_ids")
    if holdout_ids is not None:
        assert set(holdout_ids) == {"run_08", "run_09"}


def test_train_ensemble_metrics_present_when_enabled(trained_result: dict):
    ens = trained_result.get("ensemble_metrics")
    assert ens is not None, "Expected ensemble_metrics when ensemble_top_n > 1"

    mae = _extract_mae(ens)
    r2 = _extract_r2(ens)

    assert mae < 0.5
    assert r2 > 0.0

    assert "members" in ens
    assert "ensemble_n_members" in ens
    assert int(ens["ensemble_n_members"]) == 2
    assert len(ens["members"]) == 2

    for member in ens["members"]:
        assert "model_name" in member
        assert "nested_cv_mae" in member
        assert np.isfinite(float(member["nested_cv_mae"]))


def test_train_produces_internal_split_artifacts(trained_result: dict):
    out_dir = _out_dir(trained_result)

    assert _first_existing(
        out_dir,
        "fixed_grouped_split_assignment.csv",
        "split_assignments.csv",
    ).exists()

    assert _first_existing(
        out_dir,
        "validation_model_summary.csv",
        "validation_summary.csv",
    ).exists()

    assert _first_existing(
        out_dir,
        "validation_predictions.csv",
        "val_predictions.csv",
    ).exists()

    assert _first_existing(
        out_dir,
        "test_predictions.csv",
        "internal_test_predictions.csv",
    ).exists()


def test_validation_and_test_metrics_present(trained_result: dict):
    val_metrics = trained_result["validation_metrics"]
    test_metrics = trained_result["test_metrics"]

    assert _extract_mae(val_metrics) < 0.5
    assert _extract_mae(test_metrics) < 0.5


def test_train_holdout_quality_reasonable(trained_result: dict):
    metrics = trained_result["holdout_metrics"]
    mae = _extract_mae(metrics)
    r2 = _extract_r2(metrics)

    assert mae < 0.5
    assert r2 > 0.0


#
# Inference
#


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
    assert df["y_pred"].between(0.0, 1.5).all()


def test_inference_mae_reasonable(
    trained_bundle: Path,
    features_csv: Path,
):
    df = infer_classical(trained_bundle, features_csv)

    assert "y_pred" in df.columns
    assert "depth_mm" in df.columns

    mae = float(np.mean(np.abs(df["y_pred"] - df["depth_mm"])))
    assert mae < 0.5
