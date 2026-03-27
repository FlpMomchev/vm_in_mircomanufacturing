"""vm-extract-air  Extract features from segmented airborne FLAC files.

Usage::

    vm-extract-air `
        --segments-dir data/raw_data_extracted_splits/air/live `
        --config       configs/airborne.yaml `
        --out-csv      data/features/airborne/features.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.features.airborne import extract_airborne
from vm_micro.utils import apply_overrides, get_logger, load_config

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-extract-air",
        description="Extract airborne acoustic features from segmented FLAC files.",
    )
    p.add_argument("--segments-dir", required=True, help="Root directory of segmented audio files.")
    p.add_argument(
        "--config", default="configs/airborne.yaml", help="Path to airborne.yaml config."
    )
    p.add_argument(
        "--out-csv",
        default=None,
        help="Output CSV path.  Defaults to data/features/airborne/features.csv.",
    )
    p.add_argument(
        "--file-glob",
        default=None,
        help="Glob pattern override. Defaults to config file_glob (fallback: **/*.flac).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker-count override. Defaults to config n_workers.",
    )
    p.add_argument("override", nargs="*", help="YAML config overrides, e.g. --dwt_wavelet=db4")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    out_csv = args.out_csv or "data/features/airborne/features.csv"

    df = extract_airborne(
        segments_dir=args.segments_dir,
        cfg=cfg,
        out_csv=out_csv,
        file_glob=args.file_glob,
        n_workers=args.workers,
    )
    print(f"Extracted {len(df)} rows  {len(df.columns)} columns  {out_csv}")


if __name__ == "__main__":
    main()
