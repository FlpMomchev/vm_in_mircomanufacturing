"""vm-extract-struct  Extract features from segmented structure-borne HDF5 files.

Usage::

    vm-extract-struct `
        --segments-dir data/raw_data_extracted_splits/structure/live `
        --config       configs/structure.yaml `
        --out-csv      data/features/structure/features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.features.structure import extract_structure
from vm_micro.utils import apply_overrides, get_logger, load_config

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-extract-struct",
        description="Extract structure-borne features from segmented HDF5 files.",
    )
    p.add_argument("--segments-dir", required=True, help="Root directory of segmented HDF5 files.")
    p.add_argument("--config", default="configs/structure.yaml")
    p.add_argument("--out-csv", default=None)
    p.add_argument(
        "--file-glob",
        default=None,
        help="Glob pattern override. Defaults to config file_glob (fallback: **/*.h5).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker-count override. Defaults to config n_workers.",
    )
    p.add_argument(
        "--extractor",
        default=None,
        choices=["v1", "extensive"],
        help="Feature extractor version: v1 (core.py-based, default) or "
        "extensive (windowed, higher SR, WPD/MFCC). Overrides config.",
    )
    p.add_argument("override", nargs="*", help="YAML config overrides, e.g. --ds_rate=500")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    if args.extractor is not None:
        cfg["extractor"] = args.extractor

    out_csv = args.out_csv or "data/features/structure/features.csv"

    df = extract_structure(
        segments_dir=args.segments_dir,
        cfg=cfg,
        out_csv=out_csv,
        file_glob=args.file_glob,
        n_workers=args.workers,
    )
    print(f"Extracted {len(df)} rows  {len(df.columns)} columns  {out_csv}")


if __name__ == "__main__":
    main()
