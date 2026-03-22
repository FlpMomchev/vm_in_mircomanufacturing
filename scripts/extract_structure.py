"""vm-extract-struct — Extract features from segmented structure-borne HDF5 files.

Usage::

    vm-extract-struct \\
        --segments-dir all_outputs/structure \\
        --config       configs/structure.yaml \\
        --out-csv      outputs/structure/features.csv \\
        --workers      4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.features.structure import extract_structure
from vm_micro.utils import load_config, apply_overrides, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-extract-struct",
        description="Extract structure-borne features from segmented HDF5 files.",
    )
    p.add_argument("--segments-dir", required=True,
                   help="Root directory of segmented HDF5 files.")
    p.add_argument("--config",       default="configs/structure.yaml")
    p.add_argument("--out-csv",      default=None)
    p.add_argument("--file-glob",    default="**/*.h5")
    p.add_argument("--workers",      type=int, default=4)
    p.add_argument("override", nargs="*",
                   help="YAML config overrides, e.g. --ds_rate=500")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg  = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    out_csv = args.out_csv or "outputs/structure/features.csv"

    df = extract_structure(
        segments_dir=args.segments_dir,
        cfg=cfg,
        out_csv=out_csv,
        file_glob=args.file_glob,
        n_workers=args.workers,
    )
    print(f"Extracted {len(df)} rows × {len(df.columns)} columns → {out_csv}")


if __name__ == "__main__":
    main()
