"""vm-train-cls — Train classical ML models on selected features.

Usage:

    vm-train-cls \\
        --features-csv outputs/features/airborne/features_selected.csv \\
        --out-dir      outputs/features/airborne \\
        --holdout-runs 0303_3_1_8881 0503_7_2_9976

The ``--holdout-runs`` flag accepts one or more ``recording_root`` values
(plate run IDs) to exclude completely from training, matching the DL framework
convention of keeping 2 runs fully unseen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.classical.trainer import train_classical
from vm_micro.utils import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-train-cls",
        description="Train classical ML models (RF, XGB, LGB, CatBoost, GPR …) with grouped CV.",
    )
    p.add_argument("--features-csv",    required=True,
                   help="Selected-features CSV produced by vm-select.")
    p.add_argument("--out-dir",         required=True,
                   help="Output directory (will contain final_model/ subfolder).")
    p.add_argument("--holdout-runs",    nargs="*", default=None,
                   help="recording_root values to hold out (e.g. the 2 DL-unseen runs).")
    p.add_argument("--train-fraction",  type=float, default=0.70)
    p.add_argument("--cv-folds",        type=int,   default=5)
    p.add_argument("--skip-slow",       action="store_true",
                   help="Skip SVR and KernelRidge (O(n²-n³)).")
    p.add_argument("--seed",            type=int,   default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()

    result = train_classical(
        features_csv   =args.features_csv,
        out_dir        =args.out_dir,
        holdout_runs   =args.holdout_runs,
        train_fraction =args.train_fraction,
        n_cv_folds     =args.cv_folds,
        skip_slow_models=args.skip_slow,
        seed           =args.seed,
    )

    print("\n=== Training complete ===")
    print(f"Best model    : {result['best_model_name']}")
    print(f"Holdout MAE   : {result['holdout_metrics']['mae']:.4f} mm")
    print(f"Holdout RMSE  : {result['holdout_metrics']['rmse']:.4f} mm")
    print(f"Holdout R²    : {result['holdout_metrics']['r2']:.4f}")
    print(f"Total uncertainty (σ): {result['total_uncertainty']:.4f} mm")
    print(f"Bundle saved to: {result['bundle_path']}")


if __name__ == "__main__":
    main()
