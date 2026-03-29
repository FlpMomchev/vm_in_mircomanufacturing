"""vm-extract-struct  Extract features from segmented structure-borne HDF5 files.

Usage::

    vm-extract-struct `
        --segments-dir data/raw_data_extracted_splits/structure/live `
        --config       configs/structure.yaml `
        --out-csv      data/features/structure/features.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.features.structure import extract_structure, resolve_effective_structure_config
from vm_micro.utils import apply_overrides, get_logger, load_config

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-extract-struct",
        description="Extract structure-borne features from segmented HDF5 files.",
    )
    p.add_argument("--segments-dir", required=True, help="Root directory of segmented HDF5 files.")
    p.add_argument(
        "--config",
        default="configs/structure.yaml",
        help="Path to structure.yaml config (uses classical section if present).",
    )
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
        choices=["v1", "v2", "extensive"],
        help="Feature extractor version: v1 (core.py-based) or v2 "
        "(windowed, higher SR, WPD/MFCC). 'extensive' is kept as v2 alias. "
        "Overrides config.",
    )
    p.add_argument(
        "override",
        nargs="*",
        help="YAML config overrides, e.g. --ds_rate_v1=62.5 --ds_rate_v2=64",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg_all = load_config(args.config)
    if "classical" in cfg_all:
        if not isinstance(cfg_all["classical"], dict):
            raise TypeError(
                f"Invalid 'classical' section in {args.config}: "
                f"expected dict, got {type(cfg_all['classical']).__name__}"
            )
        cfg = cfg_all["classical"]
    else:
        cfg = cfg_all

    if args.override:
        cfg = apply_overrides(cfg, args.override)

    if args.extractor is not None:
        cfg["extractor"] = args.extractor

    out_csv = args.out_csv or "data/features/structure/features.csv"
    out_csv_path = Path(out_csv)

    df = extract_structure(
        segments_dir=args.segments_dir,
        cfg=cfg,
        out_csv=out_csv_path,
        file_glob=args.file_glob,
        n_workers=args.workers,
    )
    sidecar_path = Path(str(out_csv_path) + ".extractor_config.json")
    sampling_stats: dict[str, dict[str, float]] = {}
    for col in ("sr_hz_native", "sr_hz_used", "ds_rate", "duration_s"):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        sampling_stats[col] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
        }
    payload = {
        "schema_version": 1,
        "effective_extraction_config": resolve_effective_structure_config(cfg),
        "source_config_path": str(args.config),
        "cli_overrides": list(args.override or []),
        "rows": int(len(df)),
        "sampling_stats": sampling_stats,
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved structure extraction sidecar to %s", sidecar_path)
    print(f"Extracted {len(df)} rows  {len(df.columns)} columns  {out_csv}")


if __name__ == "__main__":
    main()
