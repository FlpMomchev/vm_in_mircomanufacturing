"""vm-train-cls — Train classical ML models on selected features.

Usage:

    vm-train-cls \
        --features-csv outputs/features/airborne/features_selected.csv \
        --out-dir      outputs/features/airborne \
        --holdout-runs 0303_3_1_8881 0503_7_2_9976 \
        --preset balanced \
        --inner-max-splits 4 \
        --n-iter 40 \
        --use-cuda \
        --ensemble-top-n 3

The ``--holdout-runs`` flag accepts one or more ``recording_root`` values
(plate run IDs) to exclude completely from training, matching the DL framework
convention of keeping fully unseen runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.classical.trainer import AVAILABLE_MODEL_NAMES, train_classical
from vm_micro.utils import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-train-cls",
        description=(
            "Train classical ML models with grouped holdout, nested grouped CV, "
            "optional CUDA boosting, and optional top-N ensemble persistence."
        ),
    )
    p.add_argument(
        "--features-csv", required=True, help="Selected-features CSV produced by vm-select."
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (contains final_model/ and CSV artefacts).",
    )
    p.add_argument(
        "--holdout-runs",
        nargs="*",
        default=None,
        help="recording_root values to hold out completely (e.g. the unseen DL runs).",
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Used only when --holdout-runs is omitted. Fraction of groups kept for training.",
    )
    p.add_argument(
        "--preset",
        default="balanced",
        choices=["fast", "balanced", "exhaustive"],
        help=(
            "Hyperparameter search preset. 'fast' uses small grids. 'balanced' and 'exhaustive' "
            "use RandomizedSearchCV."
        ),
    )
    p.add_argument(
        "--n-iter",
        type=int,
        default=160,
        help="RandomizedSearchCV iterations for balanced / exhaustive presets.",
    )
    p.add_argument(
        "--search-n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for CPU-side inner search. GPU models are forced to n_jobs=1.",
    )
    p.add_argument(
        "--outer-max-splits",
        dest="outer_max_splits",
        type=int,
        default=5,
        help="Maximum number of outer grouped CV folds.",
    )
    p.add_argument(
        "--inner-max-splits",
        type=int,
        default=4,
        help="Maximum number of inner grouped CV folds used for hyperparameter search.",
    )
    p.add_argument(
        "--model",
        dest="model",
        nargs="+",
        default=None,
        choices=AVAILABLE_MODEL_NAMES,
        help="Restrict training to one or more specific model names.",
    )
    p.add_argument(
        "--include-gpr",
        action="store_true",
        help="Include GaussianProcessRegressor in the comparison.",
    )
    p.add_argument(
        "--skip-slow",
        dest="skip_slow",
        action="store_true",
        help="Exclude SVR and KernelRidge.",
    )
    p.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use GPU for XGBoost and LightGBM when available. CatBoost stays on CPU.",
    )
    p.add_argument(
        "--ensemble-top-n",
        type=int,
        default=1,
        help="Persist an ensemble made from the top-N nested-CV models. 1 disables the ensemble.",
    )
    p.add_argument(
        "--snap-predictions",
        action="store_true",
        help="Snap final predictions to the nearest DOE step on holdout / ensemble evaluation.",
    )
    p.add_argument(
        "--target-mae",
        type=float,
        default=0.05,
        help="Reference MAE goal used in summaries and metadata.",
    )
    p.add_argument(
        "--doe-step",
        type=float,
        default=0.10,
        help="DOE step size in mm, used for summary ratios and optional snapping.",
    )
    p.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=42,
        help="Random seed / random_state.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    log_msg = "Starting vm-train-cls | preset=%s"
    log_args = [args.preset]

    if args.preset == "exhaustive":
        log_msg += " | n_iter=%d"
        log_args.append(args.n_iter)

    log_msg += " | use_cuda=%s | ensemble_top_n=%d"
    log_args.extend([args.use_cuda, args.ensemble_top_n])

    logger.info(log_msg, *log_args)

    result = train_classical(
        features_csv=args.features_csv,
        out_dir=args.out_dir,
        holdout_runs=args.holdout_runs,
        train_fraction=args.train_fraction,
        outer_max_splits=args.outer_max_splits,
        inner_max_splits=args.inner_max_splits,
        preset=args.preset,
        n_iter=args.n_iter,
        search_n_jobs=args.search_n_jobs,
        requested_models=args.model,
        include_gpr=args.include_gpr,
        skip_slow_models=args.skip_slow,
        use_cuda=args.use_cuda,
        ensemble_top_n=args.ensemble_top_n,
        snap_predictions=args.snap_predictions,
        target_mae=args.target_mae,
        doe_step=args.doe_step,
        random_state=args.seed,
    )

    holdout = result["holdout_metrics"]
    print("\n=== Training complete ===")
    print(f"Best model       : {result['best_model_name']}")
    print(f"Holdout MAE      : {holdout['holdout_mae']:.4f} mm")
    print(f"Holdout RMSE     : {holdout['holdout_rmse']:.4f} mm")
    print(f"Holdout R²       : {holdout['holdout_r2']:.4f}")
    print(f"Total uncertainty: {result['total_uncertainty']:.4f} mm")
    if result.get("ensemble_metrics"):
        ens = result["ensemble_metrics"]
        print(
            f"Ensemble (top-{ens['ensemble_n_members']}) : "
            f"MAE={ens['ensemble_mae']:.4f} mm | RMSE={ens['ensemble_rmse']:.4f} mm | R²={ens['ensemble_r2']:.4f}"
        )
    print(f"Holdout runs     : {', '.join(result['holdout_run_ids'])}")
    print(f"Bundle saved to  : {result['bundle_path']}")
    print(f"Output dir       : {result['out_dir']}")

    summary_path = Path(result["out_dir"]) / "nested_groupkfold_summary.csv"
    if summary_path.exists():
        print(f"Summary CSV      : {summary_path}")

    metadata_preview = {
        "best_model_name": result["best_model_name"],
        "holdout_metrics": holdout,
        "ensemble_metrics": result.get("ensemble_metrics"),
    }
    logger.info("Run summary: %s", json.dumps(metadata_preview, default=str))


if __name__ == "__main__":
    main()
