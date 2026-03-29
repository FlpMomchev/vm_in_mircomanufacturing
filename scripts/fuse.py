"""vm-fuse - Fuse predictions from multiple models into a single output.

Stage 1 - intra-modality (airborne classical + DL)::

    vm-fuse intra `
        --classical-csv models/features/air/inference_predictions.csv `
        --classical-mae 0.042 `
        --dl-csv        models/dl/air/reg/air_spec_resnet_reg_BEST_MODEL/inference_predictions.csv `
        --dl-mae        0.038 `
        --modality      airborne_ensemble `
        --out-dir       models/fusion/airborne

Stage 2 - inter-modality (airborne + structure-borne)::

    vm-fuse inter `
        --bundle-csvs `
            models/fusion/airborne/fusion_predictions.csv:0.040:airborne_ensemble `
            models/fusion/structure/fusion_predictions.csv:0.055:structure_ensemble `
        --out-dir models/fusion/final

Each `--bundle-csvs` entry is `<csv_path>:<reference_mae>:<modality>`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.fusion.fuser import (
    PredictionBundle,
    fuse_intra_modality,
    fuse_modalities,
    load_bundle_from_csv,
    save_fusion_report,
)
from vm_micro.utils import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-fuse", description="Fuse model predictions with inverse-MAE weighting."
    )
    sub = p.add_subparsers(dest="stage", required=True)

    #  intra
    ip = sub.add_parser("intra", help="Stage 1: fuse classical + DL for one modality.")
    ip.add_argument("--classical-csv", required=True)
    ip.add_argument("--classical-mae", required=True, type=float)
    ip.add_argument("--dl-csv", required=True)
    ip.add_argument("--dl-mae", required=True, type=float)
    ip.add_argument("--classical-modality", default="airborne_classical")
    ip.add_argument("--dl-modality", default="airborne_dl")
    ip.add_argument(
        "--modality", default="airborne_ensemble", help="Name for the fused output modality."
    )
    ip.add_argument("--out-dir", required=True)
    ip.add_argument("--min-weight", type=float, default=0.05)

    #  inter
    fp = sub.add_parser("inter", help="Stage 2: fuse multiple modality ensembles.")
    fp.add_argument(
        "--bundle-csvs", nargs="+", required=True, help="<csv>:<mae>:<modality> triples."
    )
    fp.add_argument("--out-dir", required=True)
    fp.add_argument("--min-weight", type=float, default=0.05)

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.stage == "intra":
        cls_bundle = load_bundle_from_csv(
            args.classical_csv, args.classical_modality, args.classical_mae
        )
        dl_bundle = load_bundle_from_csv(args.dl_csv, args.dl_modality, args.dl_mae)
        fused = fuse_intra_modality(
            cls_bundle,
            dl_bundle,
            modality_name=args.modality,
            min_weight=args.min_weight,
        )
        save_fusion_report(fused, args.out_dir)
        _print_summary(fused, args.out_dir)

    elif args.stage == "inter":
        bundles: list[PredictionBundle] = []
        for token in args.bundle_csvs:
            parts = token.split(":")
            if len(parts) != 3:
                raise ValueError(f"Expected <csv>:<mae>:<modality>, got: {token!r}")
            csv_path, mae_str, modality = parts
            bundles.append(load_bundle_from_csv(csv_path, modality, float(mae_str)))

        fused = fuse_modalities(*bundles, min_weight=args.min_weight)
        save_fusion_report(fused, args.out_dir)
        _print_summary(fused, args.out_dir)


def _print_summary(bundle: PredictionBundle, out_dir: str) -> None:
    print(f"\n=== Fusion complete: {bundle.modality} ===")
    print(f"Predictions  : {len(bundle.y_pred)}")
    print(f"Weights      : {[f'{w:.3f}' for w in bundle.metadata.get('weights', [])]}")
    print(f"Reference MAE: {bundle.validation_mae:.4f} mm  (used for weighting)")
    print(f"Mean sigma   : {bundle.sigma.mean():.4f} mm")
    if bundle.y_true is not None:
        import numpy as np

        mae = float(np.mean(np.abs(bundle.y_pred - bundle.y_true)))
        print(f"Batch MAE    : {mae:.4f} mm  (vs provided ground truth)")
    print(f"Saved to     : {out_dir}/")


if __name__ == "__main__":
    main()
