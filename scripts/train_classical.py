"""vm-train-cls - Train classical ML models on selected features.

Usage::

    vm-train-cls `
        --features-csv data/features/airborne/features_selected.csv `
        --out-dir      models/features/air/final_models_fast `
        --holdout-runs 0303_3_1_8881 0503_7_2_9976 `
        --val-fraction 0.15 `
        --test-fraction 0.15 `
        --preset balanced `
        --inner-max-splits 4 `
        --n-iter 40 `
        --use-cuda `
        --ensemble-top-n 3

The training pool is always split internally into grouped train / val / test.
Optional external testing can be provided either by excluding `--holdout-runs`
from the main CSV or by passing a separate `--external-holdout-csv`.
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
            "Train classical ML models with grouped internal train/val/test splits, "
            "optional external holdouts, nested grouped CV, optional CUDA boosting, "
            "and optional top-N ensemble persistence."
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
        help=(
            "recording_root values to exclude from the internal train/val/test pool and use "
            "only for external holdout evaluation."
        ),
    )
    p.add_argument(
        "--external-holdout-csv",
        default=None,
        help=("Optional second features CSV used only for final external holdout evaluation."),
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help=(
            "Internal train fraction. If omitted, it is inferred as 1 - val_fraction - test_fraction."
        ),
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Internal validation fraction (grouped split).",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Internal test fraction (grouped split).",
    )
    p.add_argument(
        "--preset",
        default="balanced",
        choices=["fast", "balanced", "exhaustive"],
        help=(
            "Hyperparameter search preset. 'fast' uses small grids. 'balanced' and 'fast' use GridSearchCV."
            "'exhaustive' uses RandomizedSearchCV."
        ),
    )
    p.add_argument(
        "--n-iter",
        type=int,
        default=160,
        help="RandomizedSearchCV iterations for exhaustive presets.",
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
        default=7,
        help="Maximum number of outer grouped CV folds.",
    )
    p.add_argument(
        "--inner-max-splits",
        type=int,
        default=5,
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
        external_holdout_csv=args.external_holdout_csv,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
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

    validation = result["validation_metrics"]
    test_metrics = result["test_metrics"]
    print("\n=== Training complete ===")
    print(f"Best model          : {result['best_model_name']}")
    print(f"Validation MAE      : {validation['holdout_mae']:.4f} mm")
    print(f"Validation RMSE     : {validation['holdout_rmse']:.4f} mm")
    print(f"Validation R       : {validation['holdout_r2']:.4f}")
    print(f"Internal test MAE   : {test_metrics['holdout_mae']:.4f} mm")
    print(f"Internal test RMSE  : {test_metrics['holdout_rmse']:.4f} mm")
    print(f"Internal test R    : {test_metrics['holdout_r2']:.4f}")
    ext = result.get("external_holdout_metrics")
    if ext:
        print(f"External holdout MAE: {ext['holdout_mae']:.4f} mm")
        print(f"External holdout RMSE: {ext['holdout_rmse']:.4f} mm")
        print(f"External holdout R : {ext['holdout_r2']:.4f}")
    print(f"Total uncertainty   : {result['total_uncertainty']:.4f} mm")
    if result.get("ensemble_metrics"):
        ens = result["ensemble_metrics"]
        print(
            f"Ensemble (top-{ens['ensemble_n_members']})  : "
            f"MAE={ens['ensemble_mae']:.4f} mm | RMSE={ens['ensemble_rmse']:.4f} mm | R={ens['ensemble_r2']:.4f}"
        )
    print(
        f"External holdout runs: {', '.join(result['holdout_run_ids']) if result['holdout_run_ids'] else '-'}"
    )
    print(f"Bundle saved to     : {result['bundle_path']}")
    print(f"Output dir          : {result['out_dir']}")

    summary_path = Path(result["out_dir"]) / "nested_groupkfold_summary.csv"
    if summary_path.exists():
        print(f"Summary CSV      : {summary_path}")

    metadata_preview = {
        "best_model_name": result["best_model_name"],
        "validation_metrics": validation,
        "test_metrics": test_metrics,
        "external_holdout_metrics": result.get("external_holdout_metrics"),
        "ensemble_metrics": result.get("ensemble_metrics"),
    }
    logger.info("Run summary: %s", json.dumps(metadata_preview, default=str))


if __name__ == "__main__":
    main()
