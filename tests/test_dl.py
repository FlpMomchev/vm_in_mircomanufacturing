"""Tests for vm_micro.dl and vm-train-dl helper behavior."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from vm_micro.dl.config import TrainConfig
from vm_micro.dl.data import WaveformWindowDataset, _load_h5, collate_waveforms
from vm_micro.dl.engine import aggregate_file_predictions, evaluate_file_level, predict_loader


def _load_train_dl_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "train_dl.py"
    spec = importlib.util.spec_from_file_location("train_dl_script_module", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_dl_script():
    return _load_train_dl_script()


@pytest.fixture
def tiny_flac(tmp_path: Path) -> tuple[Path, int, float]:
    sr = 48_000
    duration = 0.6
    t = np.arange(int(duration * sr), dtype=np.float64) / sr
    y = (0.25 * np.sin(2 * np.pi * 1200.0 * t)).astype(np.float32)
    path = tmp_path / "run01__seg001__step001__B1__depth0.500.flac"
    sf.write(str(path), y, sr)
    return path, sr, duration


@pytest.fixture
def tiny_h5(tmp_path: Path) -> tuple[Path, int, float]:
    sr = 96_000
    duration = 0.5
    t = np.arange(int(duration * sr), dtype=np.float64) / sr
    y = (0.35 * np.sin(2 * np.pi * 2500.0 * t)).astype(np.float32)
    path = tmp_path / "run02__seg001__step001__B2__depth0.300.h5"
    with h5py.File(path, "w") as fh:
        grp = fh.require_group("measurement")
        grp.create_dataset("data", data=y)
        grp.create_dataset("time_vector", data=t)
    return path, sr, duration


def _reg_file_df(path: Path, duration_sec: float, depth_mm: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "file_id": 0,
                "path": str(path.resolve()),
                "record_name": path.stem,
                "split_group_id": path.stem,
                "split": "train",
                "depth_mm": depth_mm,
                "recording_root": path.stem.split("__seg")[0],
                "parent_dir": path.parent.name,
                "step_idx": 1,
                "duration_sec": duration_sec,
                "class_idx": 0,
            }
        ]
    )


def test_resolve_dl_section_for_combined_config(train_dl_script):
    cfg_raw = {
        "modality": "airborne",
        "classical": {"target_sr": 192000},
        "dl": {"epochs": 12, "task": "regression"},
    }
    out = train_dl_script._resolve_dl_section(cfg_raw, "dummy.yaml")
    assert out["epochs"] == 12
    assert out["task"] == "regression"


def test_resolve_dl_section_raises_for_missing_dl(train_dl_script):
    cfg_raw = {"modality": "airborne", "classical": {"target_sr": 192000}}
    with pytest.raises(ValueError, match="contains 'classical' but no 'dl' section"):
        train_dl_script._resolve_dl_section(cfg_raw, "dummy.yaml")


def test_build_cfg_reads_combined_dl_section(tmp_path: Path, train_dl_script):
    cfg_path = tmp_path / "airborne.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'modality: "airborne"',
                "classical:",
                "  target_sr: 192000",
                "dl:",
                '  task: "regression"',
                '  file_glob: "**/*.flac"',
                "  sample_rate: 44100",
                "  epochs: 3",
                '  model_type: "spec_resnet"',
            ]
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        data_dir=str(tmp_path / "air" / "live"),
        output_dir=str(tmp_path / "out"),
        file_glob=None,
        task=None,
        feature_type=None,
        model_type=None,
        device=None,
        exclude_runs=None,
        config=str(cfg_path),
        modality=None,
        skip_final_model=False,
        final_only=False,
        override=[],
    )
    cfg = train_dl_script._build_cfg(args)

    assert isinstance(cfg, TrainConfig)
    assert cfg.sample_rate == 44100
    assert cfg.task == "regression"
    assert cfg.model_type == "spec_resnet"
    assert cfg.data_dir == args.data_dir
    assert cfg.output_dir == args.output_dir


def test_build_cfg_requires_modality_or_config_when_not_inferable(train_dl_script):
    args = argparse.Namespace(
        data_dir="data/raw_data_extracted_splits/unknown",
        output_dir="models/dl/tmp",
        file_glob=None,
        task=None,
        feature_type=None,
        model_type=None,
        device=None,
        exclude_runs=None,
        config=None,
        modality=None,
        skip_final_model=False,
        final_only=False,
        override=[],
    )
    with pytest.raises(ValueError, match="Could not infer modality"):
        train_dl_script._build_cfg(args)


def test_waveform_window_dataset_regression_shapes(
    tiny_flac: tuple[Path, int, float], tmp_path: Path
):
    path, sr, duration = tiny_flac
    file_df = _reg_file_df(path, duration)
    cfg = TrainConfig(
        data_dir=str(path.parent),
        output_dir=str(tmp_path / "dl_out"),
        task="regression",
        sample_rate=sr,
        window_sec=0.10,
        window_hop_sec=0.05,
        cache_audio=False,
        num_workers=0,
        batch_size=2,
    )

    ds = WaveformWindowDataset(file_df, cfg, training=False)
    assert len(ds) >= 1

    item = ds[0]
    assert item["waveform"].shape[0] == cfg.signal_num_samples()
    assert item["y"].dtype == torch.float32

    batch = collate_waveforms([item, item])
    assert batch["waveform"].shape == (2, cfg.signal_num_samples())
    assert batch["file_id"].shape[0] == 2


def test_waveform_window_dataset_training_caps_windows(
    tiny_flac: tuple[Path, int, float], tmp_path: Path
):
    path, sr, duration = tiny_flac
    file_df = _reg_file_df(path, duration)
    cfg = TrainConfig(
        data_dir=str(path.parent),
        output_dir=str(tmp_path / "dl_out"),
        task="regression",
        sample_rate=sr,
        window_sec=0.08,
        window_hop_sec=0.02,
        max_windows_per_file_train=3,
        cache_audio=False,
        num_workers=0,
        batch_size=2,
        seed=123,
    )

    ds = WaveformWindowDataset(file_df, cfg, training=True)
    assert len(ds.window_records) == 3
    starts = [rec.start_target for rec in ds.window_records]
    assert starts == sorted(starts)


def test_load_h5_resamples_and_normalizes(tiny_h5: tuple[Path, int, float]):
    path, _sr_native, duration = tiny_h5
    target_sr = 24_000

    y = _load_h5(str(path), target_sr=target_sr)
    assert y.dtype == np.float32
    assert np.isfinite(y).all()
    assert np.max(np.abs(y)) <= 1.0001
    assert abs(len(y) - int(round(duration * target_sr))) <= 2


def test_aggregate_file_predictions_regression_uses_window_median():
    cfg = TrainConfig(data_dir=".", output_dir=".", task="regression")
    pred_df = pd.DataFrame(
        [
            {"file_id": 0, "y_pred_depth_window": 0.10, "y_true_depth": 0.20},
            {"file_id": 0, "y_pred_depth_window": 0.50, "y_true_depth": 0.20},
            {"file_id": 0, "y_pred_depth_window": 0.20, "y_true_depth": 0.20},
        ]
    )
    file_df = pd.DataFrame(
        [
            {
                "file_id": 0,
                "record_name": "run01__seg001__step001__B1__depth0.200",
                "split_group_id": "run01",
                "split": "test",
                "depth_mm": 0.2,
                "recording_root": "run01",
                "parent_dir": "live",
                "step_idx": 1,
            }
        ]
    )

    agg = aggregate_file_predictions(pred_df, file_df, cfg)
    assert len(agg) == 1
    assert agg.loc[0, "y_pred"] == pytest.approx(0.20)
    assert agg.loc[0, "y_true_depth"] == pytest.approx(0.20)


def test_aggregate_file_predictions_classification_requires_mapping():
    cfg = TrainConfig(data_dir=".", output_dir=".", task="classification")
    pred_df = pd.DataFrame(
        [
            {"file_id": 0, "p_0": 0.2, "p_1": 0.8, "y_true_class": 1, "y_true_depth": 0.2},
            {"file_id": 0, "p_0": 0.3, "p_1": 0.7, "y_true_class": 1, "y_true_depth": 0.2},
        ]
    )
    file_df = pd.DataFrame(
        [
            {
                "file_id": 0,
                "record_name": "run01__seg001__step001__B1__depth0.200",
                "split_group_id": "run01",
                "split": "test",
                "depth_mm": 0.2,
                "recording_root": "run01",
                "parent_dir": "live",
                "step_idx": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="requires class_to_depth"):
        aggregate_file_predictions(pred_df, file_df, cfg, class_to_depth=None)

    out = aggregate_file_predictions(pred_df, file_df, cfg, class_to_depth={0: 0.1, 1: 0.2})
    assert len(out) == 1
    assert int(out.loc[0, "y_pred_class"]) == 1
    assert out.loc[0, "y_pred"] == pytest.approx(0.2)


def test_predict_loader_regression_raises_on_nonfinite_predictions():
    class BadRegressionModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D401
            return {"regression": torch.tensor([0.2, float("nan")], dtype=torch.float32)}

    cfg = TrainConfig(data_dir=".", output_dir=".", task="regression")
    loader = [
        {
            "waveform": torch.zeros((2, 32), dtype=torch.float32),
            "y": torch.zeros((2,), dtype=torch.float32),
            "file_id": torch.tensor([0, 1], dtype=torch.long),
            "depth_mm": torch.tensor([0.2, 0.3], dtype=torch.float32),
            "class_idx": torch.tensor([-1, -1], dtype=torch.long),
            "window_start_target": torch.tensor([0, 16], dtype=torch.long),
            "path": ["a.flac", "b.flac"],
        }
    ]

    with pytest.raises(ValueError, match="Non-finite regression predictions detected"):
        predict_loader(BadRegressionModel(), loader, device="cpu", cfg=cfg)


def test_evaluate_file_level_regression_contains_current_metrics():
    cfg = TrainConfig(data_dir=".", output_dir=".", task="regression", rounding_step_mm=0.1)
    df = pd.DataFrame(
        {
            "y_true_depth": [0.2, 0.4, 0.6],
            "y_pred": [0.22, 0.39, 0.58],
        }
    )
    metrics = evaluate_file_level(df, cfg)
    for key in ("mae", "rmse", "r2", "rounded_step_accuracy", "mean_signed_error"):
        assert key in metrics
        assert np.isfinite(float(metrics[key]))
