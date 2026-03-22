"""vm-infer — Run inference with a trained model (classical or DL).

Usage — classical::

    vm-infer classical \\
        --bundle   outputs/features/airborne/final_model/best_model_bundle.joblib \\
        --features outputs/features/airborne/features_selected.csv \\
        --out-csv  outputs/features/airborne/inference_predictions.csv

Usage — DL::

    vm-infer dl \\
        --model-dir outputs/dl/hybrid_spec_transformer_cls \\
        --data-dir  unseen \\
        --out-csv   outputs/dl/hybrid_spec_transformer_cls/inference_predictions.csv \\
        --task      classification
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
    p = argparse.ArgumentParser(prog="vm-infer",
                                description="Run depth prediction inference.")
    sub = p.add_subparsers(dest="mode", required=True)

    # ── classical ─────────────────────────────────────────────────────────────
    cp = sub.add_parser("classical", help="Classical ML inference.")
    cp.add_argument("--bundle",   required=True,
                    help="Path to best_model_bundle.joblib.")
    cp.add_argument("--features", required=True,
                    help="Feature CSV (must contain same columns as training).")
    cp.add_argument("--out-csv",  default=None)

    # ── DL ────────────────────────────────────────────────────────────────────
    dp = sub.add_parser("dl", help="DL model inference.")
    dp.add_argument("--model-dir",  required=True,
                    help="Training output dir or final_model dir.")
    dp.add_argument("--data-dir",   required=True,
                    help="Directory of audio files to run inference on.")
    dp.add_argument("--file-glob",  default="**/*.flac")
    dp.add_argument("--out-csv",    default=None)
    dp.add_argument("--task",       default=None,
                    choices=["classification", "regression"],
                    help="Override task from config.")
    dp.add_argument("--device",     default="auto",
                    choices=["auto", "cpu", "cuda"])
    dp.add_argument("--batch-size", type=int, default=None)

    return p


def _infer_classical(args: argparse.Namespace) -> None:
    from vm_micro.classical.inference import infer_classical

    out_csv = args.out_csv or str(Path(args.bundle).parent / "inference_predictions.csv")
    df = infer_classical(args.bundle, args.features, out_csv=out_csv)
    print(f"Predictions: {len(df)} rows → {out_csv}")
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
        add_class_labels, attach_step_idx_if_possible,
        build_file_table, choose_device, read_label_mapping,
    )
    import pandas as pd

    model_dir = Path(args.model_dir)
    # Resolve final_model subfolder
    final_dir = model_dir / "final_model"
    if final_dir.exists() and (final_dir / "best_model.pt").exists():
        model_dir = final_dir
    elif not (model_dir / "best_model.pt").exists():
        raise FileNotFoundError(f"No best_model.pt found under {args.model_dir}")

    with open(model_dir / "config.json") as fh:
        payload = json.load(fh)
    cfg = TrainConfig.from_json_dict(payload)

    if args.task:
        cfg.task = args.task
    if args.batch_size:
        cfg.batch_size = args.batch_size
    cfg.data_dir  = args.data_dir
    cfg.file_glob = args.file_glob

    device = choose_device(args.device if args.device != "auto" else cfg.device)
    logger.info("DL inference on device: %s", device)

    file_df = build_file_table(args.data_dir, args.file_glob)
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

    ds     = WaveformWindowDataset(file_df, cfg, training=False)
    loader = make_loader(ds, cfg, shuffle=False)

    window_df = predict_loader(model, loader, device, cfg)
    file_pred = aggregate_file_predictions(window_df, file_df, cfg, class_to_depth)

    out_csv = args.out_csv or str(model_dir / "inference_predictions.csv")
    file_pred.to_csv(out_csv, index=False)
    print(f"Predictions: {len(file_pred)} files → {out_csv}")

    if "y_true_depth" in file_pred.columns:
        import numpy as np
        mae = float(np.mean(np.abs(file_pred["y_pred_depth"] - file_pred["y_true_depth"])))
        print(f"MAE vs ground truth: {mae:.4f} mm")


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "classical":
        _infer_classical(args)
    elif args.mode == "dl":
        _infer_dl(args)


if __name__ == "__main__":
    main()
