"""vm-split - CLI entry point for audio/HDF5 segmentation.

Usage examples
--------------
Single file::

    vm-split single `
        --input data/raw_data/air_borne/0503_1_2_4532.flac `
        --out-dir data/raw_data_extracted_splits/air/0503_1_2_4532 `
        --segments-per-file 49 `
        --band-low 2000 --band-high 5000

Batch with preset::

    vm-split batch --preset normalBand

Batch with explicit args::

    vm-split batch `
        --doe-xlsx docs/doe/Design_of_Experiment.xlsx `
        --input-dir data/raw_data/air_borne `
        --input-glob "*.flac" `
        --out-root data/raw_data_extracted_splits/air `
        --expected-map-csv data/manifests/expected_segments.csv `
        --band-low 2000 --band-high 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when run directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from vm_micro.data.manifest import load_doe, load_expected_map_csv
from vm_micro.data.splitter import process_batch, process_one_file
from vm_micro.utils import get_logger, load_config
from vm_micro.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)
_PRESETS_PATH = PROJECT_ROOT / "configs" / "split_presets.yaml"


#
# Parser
#


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--band-low", type=float, default=2000.0, help="Lower bound of detection band (Hz)."
    )
    p.add_argument(
        "--band-high", type=float, default=5000.0, help="Upper bound of detection band (Hz)."
    )
    p.add_argument(
        "--band-fallback",
        nargs=2,
        type=float,
        action="append",
        metavar=("LOW_HZ", "HIGH_HZ"),
        default=None,
        help=(
            "Optional fallback detection band [low high] in Hz. "
            "Repeat this flag to provide multiple fallback bands."
        ),
    )
    p.add_argument("--pre-pad-s", type=float, default=0.20)
    p.add_argument("--post-pad-s", type=float, default=0.25)
    p.add_argument("--export-format", choices=["auto", "flac", "wav", "h5", "npz"], default="auto")
    p.add_argument("--target-sr", type=int, default=None)
    p.add_argument("--h5-data-key", type=str, default="measurement/data")
    p.add_argument("--h5-time-key", type=str, default="measurement/time_vector")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vm-split",
        description="Split long recordings (FLAC or HDF5) into per-hole segments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # single
    sp = sub.add_parser("single", help="Split one file.")
    _common_args(sp)
    sp.add_argument("--input", required=True)
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--segments-per-file", required=True, type=int)
    sp.add_argument(
        "--doe-xlsx",
        default=None,
        help="If provided, map segments to DOE and use canonical filenames.",
    )
    sp.add_argument("--doe-sheet", default="DOE_run_order")
    sp.add_argument("--save-manifest", action="store_true")

    # batch
    bp = sub.add_parser("batch", help="Batch split with DOE mapping.")
    _common_args(bp)
    bp.add_argument("--preset", nargs="+", default=None, choices=sorted(BATCH_PRESETS) or None)
    bp.add_argument("--doe-xlsx", default=None)
    bp.add_argument("--doe-sheet", default="DOE_run_order")
    bp.add_argument("--input-dir", default=None)
    bp.add_argument("--input-glob", default=None)
    bp.add_argument("--out-root", default=None)
    bp.add_argument("--expected-map-csv", default=None)
    bp.add_argument("--show-config", action="store_true")

    return parser


#
# Handlers
#


def _load_presets() -> dict:
    if not _PRESETS_PATH.exists():
        return {}
    raw = load_config(_PRESETS_PATH)
    # normalise keys to uppercase to match old BATCH_PRESETS convention internally
    return {name: {k.upper(): v for k, v in cfg.items()} for name, cfg in raw.items()}


BATCH_PRESETS = _load_presets()


def _run_single(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    doe_df = None
    if args.doe_xlsx:
        doe_df = load_doe(args.doe_xlsx, sheet_name=args.doe_sheet)

    if doe_df is None:
        # Minimal DOE with no labels; filenames will use "NA".
        import pandas as pd

        doe_df = pd.DataFrame(
            {
                "Step": range(args.segments_per_file),
                "HoleID": ["NA"] * args.segments_per_file,
                "Depth_mm": [None] * args.segments_per_file,
            }
        )

    manifest_df, summary = process_one_file(
        input_path,
        doe_df,
        out_dir.parent,
        expected_segments=args.segments_per_file,
        pre_pad_s=args.pre_pad_s,
        post_pad_s=args.post_pad_s,
        band_hz=(args.band_low, args.band_high),
        band_hz_fallbacks=args.band_fallback,
        export_format=args.export_format,
        h5_data_key=args.h5_data_key,
        h5_time_key=args.h5_time_key,
        target_sr=args.target_sr,
    )
    print(summary)
    if args.save_manifest:
        mp = out_dir / "segments_manifest.csv"
        manifest_df.to_csv(mp, index=False)
        print(f"Manifest saved: {mp}")


def _run_batch(args: argparse.Namespace) -> None:
    configs = _resolve_batch_configs(args)
    if args.show_config:
        for name, cfg in configs:
            print(
                f"\n[{name}] band={cfg['BAND_HZ']} glob={cfg['INPUT_GLOB']}"
                f" runs={len(cfg['EXPECTED_MAP'])} "
                f"fallbacks={cfg.get('BAND_HZ_FALLBACKS', [])}"
            )

    for name, cfg in configs:
        logger.info("=== Batch preset: %s ===", name)
        doe_df = load_doe(cfg["DOE_XLSX"], sheet_name=cfg.get("DOE_SHEET", "DOE_run_order"))
        in_dir = Path(cfg["INPUT_DIR"])
        out_root = Path(cfg["OUT_ROOT"])
        paths = sorted(in_dir.glob(cfg.get("INPUT_GLOB", "*.flac")))
        logger.info("Found %d input files", len(paths))

        manifest, summary = process_batch(
            paths,
            doe_df,
            out_root,
            expected_map=cfg["EXPECTED_MAP"],
            pre_pad_s=float(cfg.get("PRE_PAD_S", 0.20)),
            post_pad_s=float(cfg.get("POST_PAD_S", 0.25)),
            band_hz=tuple(cfg.get("BAND_HZ", (args.band_low, args.band_high))),
            band_hz_fallbacks=cfg.get("BAND_HZ_FALLBACKS", args.band_fallback),
            export_format=args.export_format,
            h5_data_key=args.h5_data_key,
            h5_time_key=args.h5_time_key,
            target_sr=args.target_sr,
        )
        print(summary.to_string())
        print(f"\nManifest {out_root / f'manifest_{name}.csv'}")


def _resolve_batch_configs(args: argparse.Namespace) -> list[tuple[str, dict]]:
    if args.preset:
        configs = []
        for name in args.preset:
            cfg = dict(BATCH_PRESETS[name])
            if args.doe_xlsx:
                cfg["DOE_XLSX"] = args.doe_xlsx
            if args.doe_sheet:
                cfg["DOE_SHEET"] = args.doe_sheet
            if args.input_dir:
                cfg["INPUT_DIR"] = args.input_dir
            if args.input_glob:
                cfg["INPUT_GLOB"] = args.input_glob
            if args.out_root:
                cfg["OUT_ROOT"] = args.out_root
            if args.expected_map_csv:
                cfg["EXPECTED_MAP"] = load_expected_map_csv(args.expected_map_csv)
            if args.band_fallback:
                cfg["BAND_HZ_FALLBACKS"] = args.band_fallback
            configs.append((name, cfg))
        return configs

    required = {
        "doe_xlsx": args.doe_xlsx,
        "input_dir": args.input_dir,
        "input_glob": args.input_glob,
        "out_root": args.out_root,
        "expected_map_csv": args.expected_map_csv,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(
            f"Without --preset, it must be provided: {missing}. "
            "Or use --preset to select a named preset."
        )
    cfg = {
        "DOE_XLSX": args.doe_xlsx,
        "DOE_SHEET": args.doe_sheet,
        "INPUT_DIR": args.input_dir,
        "INPUT_GLOB": args.input_glob,
        "OUT_ROOT": args.out_root,
        "BAND_HZ": (args.band_low, args.band_high),
        "BAND_HZ_FALLBACKS": args.band_fallback,
        "PRE_PAD_S": args.pre_pad_s,
        "POST_PAD_S": args.post_pad_s,
        "EXPECTED_MAP": load_expected_map_csv(args.expected_map_csv),
    }
    return [("custom", cfg)]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "single":
        _run_single(args)
    elif args.command == "batch":
        _run_batch(args)


if __name__ == "__main__":
    main()
