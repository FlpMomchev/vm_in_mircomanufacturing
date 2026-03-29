"""vm-infer - Run inference with a trained model (classical or DL).

Usage - classical::

    vm-infer classical `
        --bundle   models/features/air/final_models_fast_top3/final_model/ensemble_model_bundle.joblib `
        --features data/features/airborne/features_selected.csv `
        --out-csv  models/features/air/inference_predictions.csv

Usage - DL::

    vm-infer dl `
        --model-dir models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL `
        --data-dir  data/raw_data_extracted_splits/air/live `
        --out-csv   models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/inference_predictions.csv `
        --task      regression
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import torch

from vm_micro.utils import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vm-infer", description="Run depth prediction inference.")
    sub = p.add_subparsers(dest="mode", required=True)

    # Classical
    cp = sub.add_parser("classical", help="Classical ML inference.")
    cp.add_argument("--bundle", required=True, help="Path to best_model_bundle.joblib.")
    cp.add_argument(
        "--features", required=True, help="Feature CSV (must contain same columns as training)."
    )
    cp.add_argument("--out-csv", default=None)
    cp.add_argument(
        "--no-snap",
        action="store_true",
        help="Disable snapping of predictions to the DOE grid (useful when half-steps exist).",
    )
    cp.add_argument(
        "--snap-step",
        type=float,
        default=None,
        help="Override DOE step (mm) used for snapping, e.g. 0.05 for half-steps.",
    )

    # DL
    dp = sub.add_parser("dl", help="DL model inference.")
    dp.add_argument("--model-dir", required=True, help="Training output dir or final_model dir.")
    dp.add_argument(
        "--data-dir", required=True, help="Directory of audio files to run inference on."
    )
    dp.add_argument(
        "--file-glob",
        default=None,
        help="Glob override for inference data. Defaults to value stored in model config.",
    )
    dp.add_argument("--out-csv", default=None)
    dp.add_argument(
        "--task",
        default=None,
        choices=["classification", "regression"],
        help="Override task from config.",
    )
    dp.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device override. Defaults to value stored in model config.",
    )
    dp.add_argument("--batch-size", type=int, default=None)

    return p


def _infer_classical(args: argparse.Namespace) -> None:
    from vm_micro.classical.inference import infer_classical

    out_csv = args.out_csv or str(Path(args.bundle).parent / "inference_predictions.csv")
    df = infer_classical(
        args.bundle,
        args.features,
        out_csv=out_csv,
        snap_predictions=(False if args.no_snap else None),
        doe_step_mm=args.snap_step,
    )
    print(f"Predictions: {len(df)} rows {out_csv}")
    if "depth_mm" in df.columns:
        import numpy as np

        mae = float(np.mean(np.abs(df["y_pred"] - df["depth_mm"])))
        print(f"MAE vs ground truth: {mae:.4f} mm")


def _infer_dl(args: argparse.Namespace) -> None:
    from vm_micro.dl.config import TrainConfig
    from vm_micro.dl.data import WaveformWindowDataset
    from vm_micro.dl.engine import aggregate_file_predictions, make_loader, predict_loader
    from vm_micro.dl.models import DepthModel
    from vm_micro.dl.utils import (
        add_class_labels,
        attach_step_idx_if_possible,
        build_file_table,
        choose_device,
        read_label_mapping,
    )

    model_dir = Path(args.model_dir)
    # Resolve final_model subfolder
    final_dir = model_dir / "final_model"
    if final_dir.exists() and (final_dir / "best_model.pt").exists():
        model_dir = final_dir
    elif not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"No best_model.pt found under {args.model_dir}")

    with open(model_dir / "config.json", "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cfg = TrainConfig.from_json_dict(payload)

    if args.task:
        cfg.task = args.task
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    effective_file_glob = args.file_glob or cfg.file_glob
    selected_device = args.device if args.device is not None else (cfg.device or "auto")

    cfg.data_dir = args.data_dir
    cfg.file_glob = effective_file_glob

    device = choose_device(selected_device)
    logger.info("DL inference on device: %s", device)

    file_df = build_file_table(args.data_dir, effective_file_glob)
    file_df = attach_step_idx_if_possible(file_df)

    label_map_path = model_dir / "label_mapping.json"
    if label_map_path.exists():
        class_to_depth = read_label_mapping(label_map_path)
    else:
        _, _, class_to_depth = add_class_labels(file_df)

    file_df, _, _ = add_class_labels(file_df)

    out_dim = max(class_to_depth.keys()) + 1 if cfg.task == "classification" else 1
    model = DepthModel(cfg, out_dim=out_dim).to(device)
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    ds = WaveformWindowDataset(file_df, cfg, training=False)
    loader = make_loader(ds, cfg, shuffle=False)

    window_df = predict_loader(model, loader, device, cfg)
    file_pred = aggregate_file_predictions(window_df, file_df, cfg, class_to_depth)

    out_csv = args.out_csv or str(model_dir / "inference_predictions.csv")
    file_pred.to_csv(out_csv, index=False)
    print(f"Predictions: {len(file_pred)} files {out_csv}")

    if "y_true_depth" in file_pred.columns:
        import numpy as np

        if "y_pred_depth" in file_pred.columns:
            pred_col = "y_pred_depth"
        elif "y_pred" in file_pred.columns:
            pred_col = "y_pred"
        else:
            raise KeyError(
                f"No prediction column found in DL inference output: {list(file_pred.columns)}"
            )

        mae = float(np.mean(np.abs(file_pred[pred_col] - file_pred["y_true_depth"])))
        print(f"MAE vs ground truth: {mae:.4f} mm")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "classical":
        _infer_classical(args)
    elif args.mode == "dl":
        _infer_dl(args)


if __name__ == "__main__":
    main()
