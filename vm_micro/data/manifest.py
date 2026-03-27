"""vm_micro.data.manifest
~~~~~~~~~~~~~~~~~~~~~~~~
DOE loading and segment manifest utilities.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

#
# DOE
#


def load_doe(xlsx_path: str | Path, sheet_name: str = "DOE_run_order") -> pd.DataFrame:
    """Load the DOE run-order sheet from the Excel manifest."""
    xlsx_path = Path(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if "Step" in df.columns:
        df["Step"] = pd.to_numeric(df["Step"], errors="coerce").astype("Int64")
    return df


#
# Segment manifest
#


def load_manifest(csv_path: str | Path) -> pd.DataFrame:
    """Load a segments manifest CSV produced by the splitter."""
    return pd.read_csv(csv_path)


def load_expected_map_csv(csv_path: str | Path) -> dict[str, int]:
    """Load a CSV with columns ``stem, expected_segments`` into a dict."""
    df = pd.read_csv(csv_path)
    required = {"stem", "expected_segments"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {sorted(required)}")
    df["expected_segments"] = pd.to_numeric(df["expected_segments"], errors="raise").astype(int)
    return dict(zip(df["stem"].astype(str), df["expected_segments"].astype(int)))


def map_segments_to_doe(doe_df: pd.DataFrame, n_segments: int) -> pd.DataFrame:
    """Return a DOE slice for the first *n_segments* rows.

    Extra segments beyond the DOE length are marked ``status='extra_segment'``.
    """
    n_doe = len(doe_df)
    n_use = min(n_segments, n_doe)
    mapped = doe_df.iloc[:n_use].copy()
    mapped["doe_row_index0"] = np.arange(n_use, dtype=int)
    mapped["status"] = "ok"

    if n_segments > n_doe:
        extra_n = n_segments - n_doe
        extras = pd.DataFrame(
            {
                "Step": [pd.NA] * extra_n,
                "HoleID": [pd.NA] * extra_n,
                "Depth_mm": [pd.NA] * extra_n,
                "doe_row_index0": np.arange(n_doe, n_doe + extra_n, dtype=int),
                "status": ["extra_segment"] * extra_n,
            }
        )
        mapped = pd.concat([mapped, extras], ignore_index=True)

    return mapped


#
# Filename helpers (shared with splitter and feature extractor)
#

_SAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_DEPTH_RE = re.compile(r"depth([0-9]+(?:[.,][0-9]+)?)", flags=re.IGNORECASE)
_STEP_RE = re.compile(r"__step([0-9]+)__", flags=re.IGNORECASE)


def safe_slug(s: str) -> str:
    s = str(s).strip()
    s = _SAFE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "NA"


def fmt_float_for_name(x: float | int | None, decimals: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    s = f"{float(x):.{decimals}f}".rstrip(".")
    return s.replace("-", "m")


def parse_depth_mm(name: str) -> float:
    """Parse depth label from a segment filename stem."""
    m = _DEPTH_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse depth from filename: {name!r}")
    return float(m.group(1).replace(",", "."))


def try_parse_depth_mm(name: str) -> float | None:
    try:
        return parse_depth_mm(name)
    except ValueError:
        return None


def parse_step_idx(stem: str) -> int:
    m = _STEP_RE.search(stem)
    if not m:
        raise ValueError(f"Cannot parse step index from: {stem!r}")
    return int(m.group(1))


def try_parse_step_idx(stem: str) -> int | None:
    try:
        return parse_step_idx(stem)
    except ValueError:
        return None


def extract_recording_root(stem: str) -> str:
    """Return the plate-run ID from a segment stem (everything before ``__seg``)."""
    return stem.split("__seg")[0] if "__seg" in stem else stem


def build_segment_filename(
    stem: str, seg_idx: int, step: object, hole: object, depth: object, ext: str
) -> str:
    """Construct the canonical segment filename.

    Format: ``{stem}__seg{NNN}__step{SSS}__{HOLE}__depth{D.DDD}{ext}``
    """
    step_s = f"{int(step):03d}" if step is not None and str(step) != "<NA>" else "NA"
    hole_s = safe_slug(str(hole)) if hole is not None and str(hole) != "<NA>" else "NA"
    depth_s = fmt_float_for_name(depth, decimals=3)
    return f"{stem}__seg{seg_idx:03d}__step{step_s}__{hole_s}__depth{depth_s}{ext}"
