"""vm-train-dl - Train the deep-learning depth prediction model.

Config-driven: all hyperparameters, split fractions, repeat count, and
final-model behaviour live in the selected modality config
(configs/airborne.yaml or configs/structure.yaml, under the dl section).
The CLI stays minimal.

Usage
-----
Full run (repeated experiment + final model)::

    vm-train-dl `
        --data-dir    data/raw_data_extracted_splits/air/live `
        --output-dir  models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL `
        --task        regression

Structure-borne HDF5::

    vm-train-dl `
        --data-dir    data/raw_data_extracted_splits/structure/live `
        --output-dir  models/dl/structure/reg/structure_spec_resnet_reg_96k_retry `
        --file-glob   "**/*.h5" `
        --task        regression `
        --model-type  spec_resnet

Skip the repeated experiment (already done), train final model only::

    vm-train-dl `
        --data-dir   data/raw_data_extracted_splits/air/live `
        --output-dir models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL `
        --final-only

Run repeated experiment but skip the final model::

    vm-train-dl `
        --data-dir   data/raw_data_extracted_splits/air/live `
        --output-dir models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL `
        --skip-final-model

Config overrides (positional key=value pairs)::

    vm-train-dl --data-dir ... --output-dir ... epochs=30 lr=5e-4 n_repeats=3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import torch

from vm_micro.dl.config import TrainConfig
from vm_micro.dl.training import (
    choose_final_training_epochs,
    fit_final_model_all_files,
    fit_repeated_experiment,
    make_main_split_builder,
)
from vm_micro.dl.utils import (
    add_class_labels,
    attach_step_idx_if_possible,
    build_file_table,
    choose_device,
    dump_json,
    write_label_mapping,
)
from vm_micro.utils import apply_overrides, get_logger, load_config

logger = get_logger(__name__)


#
# Parser  intentionally minimal; config does the heavy lifting
#


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-train-dl",
        description="Train the DL depth prediction model. "
        "Hyperparameters are read from the selected modality config (dl section).",
    )
    # Required
    p.add_argument(
        "--data-dir", required=True, help="Root of segmented audio files (FLAC or HDF5)."
    )
    p.add_argument("--output-dir", required=True, help="Output directory for all artefacts.")

    # Format
    p.add_argument(
        "--file-glob",
        default=None,
        help="Glob pattern for audio files. "
        "Defaults to config file_glob; if missing there, "
        "auto-detected from --data-dir.",
    )

    # Task / architecture overrides (convenience; also settable via config)
    p.add_argument(
        "--task",
        choices=["classification", "regression"],
        default=None,
        help="Override task from config.",
    )
    p.add_argument(
        "--feature-type",
        choices=["logmel", "cwt", "linear_spec"],
        default=None,
        help="Override frontend from config.",
    )
    p.add_argument(
        "--model-type",
        choices=["small_cnn", "spec_resnet", "hybrid_spec_transformer"],
        default=None,
        help="Override model architecture from config.",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override compute device from config.",
    )

    # Exclude runs
    p.add_argument(
        "--exclude-runs",
        nargs="*",
        default=None,
        help="recording_root values to exclude from training "
        "(e.g. runs reserved for classical ML holdout).",
    )

    # Config path
    p.add_argument(
        "--config",
        default=None,
        help="Path to modality config (airborne.yaml/structure.yaml). "
        "Defaults to the modality inferred from --data-dir.",
    )
    p.add_argument(
        "--modality",
        choices=["airborne", "structure"],
        default=None,
        help="Modality to select default config when --config is omitted.",
    )

    # Final-model control
    p.add_argument(
        "--skip-final-model",
        action="store_true",
        help="Run the repeated experiment only; skip fit_final_model_all_files.",
    )
    p.add_argument(
        "--final-only",
        action="store_true",
        help="Skip the repeated experiment; run fit_final_model_all_files "
        "using final_epochs from config (or --override final_epochs=N).",
    )

    # Free-form config overrides
    p.add_argument(
        "override", nargs="*", help="Key=value config overrides, e.g. epochs=30 lr=5e-4 n_repeats=3"
    )

    return p


#
# Helpers
#


def _auto_file_glob(data_dir: str) -> str:
    """Infer file glob from directory contents."""
    p = Path(data_dir)
    if any(p.rglob("*.h5")):
        logger.info("Auto-detected HDF5 files  using **/*.h5")
        return "**/*.h5"
    logger.info("Defaulting to **/*.flac")
    return "**/*.flac"


def _infer_modality_from_data_dir(data_dir: str) -> str | None:
    tokens = [tok for tok in Path(data_dir).as_posix().lower().split("/") if tok]
    if any(tok in {"air", "airborne"} for tok in tokens):
        return "airborne"
    if any(tok in {"structure", "struct"} for tok in tokens):
        return "structure"
    return None


def _default_modality_config_path(modality: str | None) -> str | None:
    if modality == "airborne":
        return "configs/airborne.yaml"
    if modality == "structure":
        return "configs/structure.yaml"
    return None


def _resolve_dl_section(cfg_raw: dict[str, Any], config_path: str) -> dict[str, Any]:
    """Resolve DL config from combined modality config or flat legacy config."""
    if "dl" in cfg_raw:
        dl_cfg = cfg_raw["dl"]
        if not isinstance(dl_cfg, dict):
            raise TypeError(
                f"Invalid 'dl' section in {config_path}: expected dict, got {type(dl_cfg).__name__}"
            )
        return dl_cfg

    if "classical" in cfg_raw and "dl" not in cfg_raw:
        raise ValueError(
            f"Config {config_path} contains 'classical' but no 'dl' section. "
            "Expected combined modality config."
        )

    return cfg_raw


def _build_cfg(args: argparse.Namespace) -> TrainConfig:
    """Load YAML, apply CLI overrides, and build TrainConfig."""
    inferred_modality = args.modality or _infer_modality_from_data_dir(args.data_dir)
    config_path = args.config or _default_modality_config_path(inferred_modality)
    if config_path is None:
        raise ValueError(
            "Could not infer modality from --data-dir. Pass --modality or --config explicitly."
        )

    logger.info("Using DL config from %s", config_path)
    cfg_raw = load_config(config_path)
    cfg_dict = _resolve_dl_section(cfg_raw, config_path)

    # CLI flag overrides (only when explicitly set)
    if args.task is not None:
        cfg_dict["task"] = args.task
    if args.feature_type is not None:
        cfg_dict["feature_type"] = args.feature_type
    if args.model_type is not None:
        cfg_dict["model_type"] = args.model_type
    if args.device is not None:
        cfg_dict["device"] = args.device

    cfg_dict["data_dir"] = args.data_dir
    cfg_dict["output_dir"] = args.output_dir

    # Free-form key=value overrides
    if args.override:
        cfg_dict = apply_overrides(cfg_dict, args.override)

    # Resolve file_glob after all config/override layers.
    if args.file_glob is not None:
        cfg_dict["file_glob"] = args.file_glob
    elif not cfg_dict.get("file_glob"):
        cfg_dict["file_glob"] = _auto_file_glob(args.data_dir)

    # Build TrainConfig  only pass fields it knows about
    valid = TrainConfig.__dataclass_fields__.keys()  # type: ignore[attr-defined]
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in valid})

    return cfg


#
# Main
#


def main() -> None:
    args = build_parser().parse_args()
    cfg = _build_cfg(args)

    device = choose_device(cfg.device)
    logger.info("Device: %s", device)
    if device != "cuda":
        torch.set_num_threads(1)

    #  Build file table
    file_df = build_file_table(args.data_dir, cfg.file_glob)
    file_df = attach_step_idx_if_possible(file_df)

    #  Exclude runs
    if args.exclude_runs:
        before = len(file_df)
        file_df = file_df[~file_df["recording_root"].isin(args.exclude_runs)].copy()
        logger.info("Excluded %d files from runs %s", before - len(file_df), args.exclude_runs)

    #  Class labels
    file_df, depth_to_class, class_to_depth = add_class_labels(file_df)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save label mapping at the experiment root
    write_label_mapping(class_to_depth, out_dir / "label_mapping.json")
    dump_json(
        {
            "depth_to_class": {str(k): v for k, v in depth_to_class.items()},
            "class_to_depth": {str(k): v for k, v in class_to_depth.items()},
        },
        out_dir / "label_mapping_full.json",
    )

    #  Split builder  reads all split params from config
    split_builder = make_main_split_builder(
        split_strategy=cfg.split_strategy,
        evaluation_unit=cfg.evaluation_unit,
        group_mode=cfg.group_mode,
        train_fraction=float(cfg.train_fraction),
        val_fraction=float(cfg.val_fraction),
        test_fraction=float(cfg.test_fraction),
    )

    n_repeats = int(cfg.n_repeats)
    run_final = bool(cfg.run_final_model)
    final_epochs_cfg = int(cfg.final_epochs) if cfg.final_epochs is not None else None

    #  Repeated experiment
    experiment = None

    if not args.final_only:
        logger.info(
            "Starting repeated experiment: task=%s  frontend=%s  model=%s  "
            "repeats=%d  evaluation_unit=%s",
            cfg.task,
            cfg.feature_type,
            cfg.model_type,
            n_repeats,
            cfg.evaluation_unit,
        )
        experiment = fit_repeated_experiment(
            cfg=cfg,
            file_df=file_df,
            root_out_dir=out_dir,
            n_repeats=n_repeats,
            split_builder=split_builder,
            class_to_depth=class_to_depth,
            device=device,
        )
        logger.info("Repeated experiment complete.")
    else:
        logger.info("--final-only: skipping repeated experiment.")

    #  Final model
    skip_final = args.skip_final_model or not run_final

    if not skip_final:
        if experiment is not None:
            n_final = choose_final_training_epochs(
                experiment.summary_df,
                explicit_epochs=final_epochs_cfg,
            )
        else:
            # --final-only: must have explicit epochs
            if final_epochs_cfg is None:
                raise ValueError(
                    "--final-only requires either final_epochs set in the selected "
                    "modality config (dl section) "
                    "or a 'final_epochs=N' override."
                )
            n_final = final_epochs_cfg

        logger.info("Training final model on ALL files for %d epochs.", n_final)
        fit_final_model_all_files(
            cfg=cfg,
            file_df=file_df,
            root_out_dir=out_dir,
            class_to_depth=class_to_depth,
            device=device,
            final_epochs=n_final,
        )
        logger.info("Final model saved to %s/final_model/", out_dir)
    else:
        logger.info("Skipping final model.")

    #  Summary
    if experiment is not None and not experiment.summary_df.empty:
        df = experiment.summary_df
        print("\n=== Repeated experiment summary ===")
        metric_cols = [c for c in df.columns if c.startswith("test_") or c.startswith("val_")]
        for col in metric_cols:
            print(f"  {col:35s}: {df[col].mean():.4f}  {df[col].std():.4f}")


if __name__ == "__main__":
    main()
