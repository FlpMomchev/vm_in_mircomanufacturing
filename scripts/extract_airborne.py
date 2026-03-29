"""vm-extract-air  Extract features from segmented airborne FLAC files.

Usage::

    vm-extract-air `
        --segments-dir data/raw_data_extracted_splits/air/live `
        --config       configs/airborne.yaml `
        --out-csv      data/features/airborne/features.csv
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

from vm_micro.features.airborne import extract_airborne, resolve_effective_airborne_config
from vm_micro.utils import apply_overrides, get_logger, load_config

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vm-extract-air",
        description="Extract airborne acoustic features from segmented FLAC files.",
    )
    p.add_argument("--segments-dir", required=True, help="Root directory of segmented audio files.")
    p.add_argument(
        "--config",
        default="configs/airborne.yaml",
        help="Path to airborne.yaml config (uses classical section if present).",
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

    out_csv = args.out_csv or "data/features/airborne/features.csv"
    out_csv_path = Path(out_csv)

    df = extract_airborne(
        segments_dir=args.segments_dir,
        cfg=cfg,
        out_csv=out_csv_path,
        file_glob=args.file_glob,
        n_workers=args.workers,
    )
    sidecar_path = Path(str(out_csv_path) + ".extractor_config.json")
    sampling_stats: dict[str, dict[str, float]] = {}
    for col in ("sr_hz_native", "sr_hz_used", "sr_hz", "ds_rate", "duration_s"):
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
        "effective_extraction_config": resolve_effective_airborne_config(cfg),
        "source_config_path": str(args.config),
        "cli_overrides": list(args.override or []),
        "rows": int(len(df)),
        "sampling_stats": sampling_stats,
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Saved airborne extraction sidecar to %s", sidecar_path)
    print(f"Extracted {len(df)} rows  {len(df.columns)} columns  {out_csv}")


if __name__ == "__main__":
    main()
