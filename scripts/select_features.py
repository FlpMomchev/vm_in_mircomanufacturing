"""vm-select  Run the inverted-cone feature selection pipeline.

Usage::

    vm-select `
        --features-csv data/features/airborne/features.csv `
        --out-csv      data/features/airborne/features_selected.csv `
        --final-n      20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import pandas as pd

from vm_micro.features.selection import SelectionConfig, select_features
from vm_micro.utils import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-select",
        description="Inverted-cone feature selection (Spearman, MI, ElasticNet, ExtraTrees).",
    )
    p.add_argument("--features-csv", required=True)
    p.add_argument("--out-csv", default=None)
    p.add_argument("--target-col", default="depth_mm")
    p.add_argument("--group-col", default="recording_root")
    p.add_argument("--final-n", type=int, default=None)
    p.add_argument("--preselect-n", type=int, default=60)
    p.add_argument("--min-spearman", type=float, default=0.10)
    p.add_argument(
        "--min-partial-r",
        type=float,
        default=None,
        help="If set, keep only features whose |partial Spearman r| with the "
        "target (controlling for --partial-control-col) meets this threshold. "
        "E.g. 0.15 to filter out duration-proxy features.",
    )
    p.add_argument(
        "--partial-control-col",
        default="duration_s",
        help="Column to control for in the partial-correlation filter (default: duration_s).",
    )
    p.add_argument("--vif-threshold", type=float, default=5.0)
    p.add_argument("--intercorr", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_parser().parse_args()

    df = pd.read_csv(args.features_csv)
    logger.info("Loaded %d rows  %d cols from %s", *df.shape, args.features_csv)

    cfg = SelectionConfig(
        target_col=args.target_col,
        group_col=args.group_col,
        final_max_features=args.final_n,
        preselect_top_n=args.preselect_n,
        min_target_abs_spearman=args.min_spearman,
        min_partial_r=args.min_partial_r,
        partial_control_col=args.partial_control_col,
        vif_threshold=args.vif_threshold,
        intercorr_threshold=args.intercorr,
        seed=args.seed,
    )

    out_csv = args.out_csv or args.features_csv.replace(".csv", "_selected.csv")
    out_dir_sweep = Path(out_csv).parent / "FS_validation"
    df_sel, selected = select_features(df, cfg, out_csv=out_csv, sweep_dir=out_dir_sweep)

    print(f"\nSelected {len(selected)} features:")
    for f in selected:
        print(f"  {f}")
    print(f"\nSaved to: {out_csv}")


if __name__ == "__main__":
    main()
