"""vm-train-dl — Train the deep-learning depth model.

Usage::

    # Classification (default)
    vm-train-dl \\
        --data-dir    all_outputs \\
        --output-dir  outputs/dl/hybrid_spec_transformer_cls \\
        --config      configs/dl.yaml \\
        --task        classification \\
        --model-type  hybrid_spec_transformer

    # Regression
    vm-train-dl \\
        --data-dir    all_outputs \\
        --output-dir  outputs/dl/hybrid_spec_transformer_reg \\
        --task        regression

Config overrides via positional args::

    vm-train-dl --data-dir ... --output-dir ... epochs=30 lr=5e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import pandas as pd
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
)
from vm_micro.utils import load_config, apply_overrides, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-train-dl",
        description="Train the DL depth prediction model (classification + regression).",
    )
    p.add_argument("--data-dir",    required=True,
                   help="Root of segmented audio files.")
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--config",      default="configs/dl.yaml")
    p.add_argument("--file-glob",   default="**/*.flac")
    p.add_argument("--task",        default="classification",
                   choices=["classification", "regression"])
    p.add_argument("--feature-type",default="logmel",
                   choices=["logmel", "cwt"])
    p.add_argument("--model-type",  default="hybrid_spec_transformer",
                   choices=["small_cnn", "spec_resnet", "hybrid_spec_transformer"])
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cpu", "cuda"])
    # allow --exclude-runs to honour the manifest flag
    p.add_argument("--exclude-runs", nargs="*", default=None,
                   help="recording_root values to exclude from DL training "
                        "(matches exclude_from_dl_training flag in manifest).")
    p.add_argument("override", nargs="*",
                   help="YAML config overrides, e.g. epochs=30 lr=5e-4")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Load + merge config
    cfg_dict = load_config(args.config)
    if args.override:
        cfg_dict = apply_overrides(cfg_dict, args.override)

    # CLI flags override config file
    cfg_dict["task"]         = args.task
    cfg_dict["feature_type"] = args.feature_type
    cfg_dict["model_type"]   = args.model_type
    cfg_dict["data_dir"]     = args.data_dir
    cfg_dict["output_dir"]   = args.output_dir
    cfg_dict["file_glob"]    = args.file_glob
    cfg_dict["device"]       = args.device

    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items()
                         if hasattr(TrainConfig, k) or
                         k in TrainConfig.__dataclass_fields__})  # type: ignore[attr-defined]

    device = choose_device(cfg.device)
    logger.info("Using device: %s", device)

    # Build file table
    file_df = build_file_table(args.data_dir, args.file_glob)
    file_df = attach_step_idx_if_possible(file_df)

    # Honour exclude-runs flag
    if args.exclude_runs:
        before = len(file_df)
        file_df = file_df[~file_df["recording_root"].isin(args.exclude_runs)].copy()
        logger.info("Excluded %d files from excluded runs %s", before - len(file_df), args.exclude_runs)

    file_df, depth_to_class, class_to_depth = add_class_labels(file_df)
    dump_json({"depth_to_class": {str(k): v for k, v in depth_to_class.items()},
               "class_to_depth": {str(k): v for k, v in class_to_depth.items()}},
              Path(args.output_dir) / "label_mapping.json")

    split_builder = make_main_split_builder(cfg)

    # Repeated CV experiment
    logger.info("Starting repeated experiment (%s, %s, %s)",
                cfg.task, cfg.feature_type, cfg.model_type)
    experiment = fit_repeated_experiment(
        cfg=cfg,
        file_df=file_df,
        split_builder=split_builder,
        class_to_depth=class_to_depth,
        device=device,
    )
    logger.info("Repeated experiment done.")

    # Final model on all data
    n_final = choose_final_training_epochs(experiment.summary_df, cfg)
    logger.info("Training final model for %d epochs on all files.", n_final)
    fit_final_model_all_files(
        cfg=cfg,
        file_df=file_df,
        n_epochs=n_final,
        class_to_depth=class_to_depth,
        device=device,
    )
    logger.info("Final model saved to %s/final_model/", args.output_dir)


if __name__ == "__main__":
    main()
