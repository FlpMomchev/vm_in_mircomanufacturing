"""Tests for scripts.final_prediction helper persistence behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.final_prediction import (
    _copy_split_debug_plots_to_run_dir,
    _persist_model_setup_lock,
    _save_bundle_predictions_csv,
)
from vm_micro.fusion.fuser import PredictionBundle


def _example_setup_audit() -> dict[str, object]:
    return {
        "models": {
            "airborne_dl": {
                "model_kind": "dl",
                "model_dir": "models/dl/air/reg/example",
                "config_path": "models/dl/air/reg/example/final_model/config.json",
                "config_sha256": "abc123",
                "reference_mae_raw_mm": 0.0123,
                "reference_mae_source": "final_model/test_metrics.csv.holdout_mae_raw",
            }
        },
        "run_config_snapshot": {
            "final_prediction_config": "configs/fusion.yaml",
        },
    }


def test_save_bundle_predictions_csv_writes_parent_dirs(tmp_path: Path) -> None:
    bundle = PredictionBundle(
        modality="airborne_ensemble",
        record_names=np.array(["r1", "r2"]),
        y_pred=np.array([0.2, 0.3], dtype=np.float64),
        sigma=np.array([0.01, 0.02], dtype=np.float64),
        validation_mae=0.01,
        y_true=np.array([0.21, 0.29], dtype=np.float64),
    )

    out_csv = tmp_path / "airborne" / "fusion_predictions.csv"
    result = _save_bundle_predictions_csv(bundle, out_csv)

    assert result == out_csv
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert list(df.columns) == [
        "record_name",
        "y_pred",
        "sigma",
        "modality",
        "validation_mae",
        "depth_mm",
        "residual",
    ]
    assert len(df) == 2


def test_persist_model_setup_lock_writes_lock_files_only(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260329_120000__manual"
    final_dir = run_dir / "final"
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts, warnings = _persist_model_setup_lock(
        setup_audit=_example_setup_audit(),
        run_dir=run_dir,
        final_dir=final_dir,
    )

    assert warnings == []
    assert set(artifacts.keys()) == {"run_lock", "latest_lock"}

    run_lock = run_dir / artifacts["run_lock"]
    latest_lock = run_dir / artifacts["latest_lock"]
    assert run_lock.exists()
    assert latest_lock.exists()
    assert not list((final_dir / "model_setup_locks").glob("*setup_audit.json"))

    payload = json.loads(run_lock.read_text(encoding="utf-8"))
    assert payload["run_name"] == run_dir.name
    assert payload["final_prediction_config"] == "configs/fusion.yaml"


def test_copy_split_debug_plots_to_run_dir_copies_expected_files(tmp_path: Path) -> None:
    split_out_dir = tmp_path / "splits" / "rec1"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    core_src = split_out_dir / "rec1__debug__core.png"
    padded_src = split_out_dir / "rec1__debug__padded.png"
    core_src.write_bytes(b"core-bytes")
    padded_src.write_bytes(b"padded-bytes")

    run_split_dir = tmp_path / "fusion" / "airborne" / "rec1"
    copied = _copy_split_debug_plots_to_run_dir(
        {"debug_core": str(core_src), "debug_padded": str(padded_src)},
        run_split_dir,
    )

    core_dst = run_split_dir / core_src.name
    padded_dst = run_split_dir / padded_src.name
    assert core_dst.exists()
    assert padded_dst.exists()
    assert core_dst.read_bytes() == b"core-bytes"
    assert padded_dst.read_bytes() == b"padded-bytes"
    assert copied["debug_core"] == str(core_dst)
    assert copied["debug_padded"] == str(padded_dst)


def test_copy_split_debug_plots_to_run_dir_skips_missing_sources(tmp_path: Path) -> None:
    run_split_dir = tmp_path / "fusion" / "structure" / "rec2"
    copied = _copy_split_debug_plots_to_run_dir(
        {
            "debug_core": str(tmp_path / "missing__debug__core.png"),
            "debug_padded": None,
        },
        run_split_dir,
    )
    assert copied == {}
