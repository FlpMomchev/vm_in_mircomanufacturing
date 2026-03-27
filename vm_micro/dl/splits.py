from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .utils import resolve_split_group_id


@dataclass
class SplitSpec:
    split_strategy: str
    evaluation_unit: str
    group_mode: str
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PositionHoldoutSpec:
    holdout_steps: list[int]
    val_fraction_within_remaining: float = 0.20
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "holdout_steps": [int(step) for step in self.holdout_steps],
            "val_fraction_within_remaining": float(self.val_fraction_within_remaining),
            "seed": int(self.seed),
        }


def _normalize_fractions(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[float, float, float]:
    values = np.asarray([train_fraction, val_fraction, test_fraction], dtype=float)
    values = values / values.sum()
    return tuple(float(value) for value in values.tolist())


def _simple_stratified_file_split(
    file_df: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    train_fraction, val_fraction, test_fraction = _normalize_fractions(
        train_fraction,
        val_fraction,
        test_fraction,
    )

    idx = np.arange(len(file_df))
    y = file_df["class_idx"].to_numpy()

    out = file_df.copy()
    out["split"] = "unused"

    if test_fraction > 0.0:
        test_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=seed,
        )
        train_val_idx, test_idx = next(test_split.split(idx, y))
    else:
        train_val_idx = idx
        test_idx = np.array([], dtype=int)

    if val_fraction > 0.0:
        rel_val_fraction = val_fraction / max(1e-12, 1.0 - test_fraction)
        val_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=rel_val_fraction,
            random_state=seed + 1,
        )
        train_rel, val_rel = next(val_split.split(np.arange(len(train_val_idx)), y[train_val_idx]))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
    else:
        train_idx = train_val_idx
        val_idx = np.array([], dtype=int)

    split_col = out.columns.get_loc("split")
    out.iloc[train_idx, split_col] = "train"
    out.iloc[val_idx, split_col] = "val"
    out.iloc[test_idx, split_col] = "test"

    return out.loc[out["split"].isin(["train", "val", "test"])].reset_index(drop=True)


def _greedy_group_stratified_split(
    file_df: pd.DataFrame,
    split_group_id: pd.Series,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    train_fraction, val_fraction, test_fraction = _normalize_fractions(
        train_fraction,
        val_fraction,
        test_fraction,
    )

    split_names = ["train", "val", "test"]
    split_fracs = np.asarray([train_fraction, val_fraction, test_fraction], dtype=float)

    tmp = file_df.copy()
    tmp["split_group_id"] = split_group_id.astype(str)
    class_ids = sorted(tmp["class_idx"].unique().tolist())

    rng = np.random.default_rng(seed)
    group_ids = tmp["split_group_id"].drop_duplicates().tolist()
    rng.shuffle(group_ids)
    random_order = {group_id: idx for idx, group_id in enumerate(group_ids)}

    grouped_items = []
    for group_id, group_df in tmp.groupby("split_group_id"):
        counts = (
            group_df["class_idx"]
            .value_counts()
            .reindex(class_ids, fill_value=0)
            .to_numpy(dtype=float)
        )
        grouped_items.append(
            {
                "group_id": str(group_id),
                "counts": counts,
                "n_files": int(len(group_df)),
                "sort_key": (-int(len(group_df)), random_order[str(group_id)]),
            }
        )

    grouped_items.sort(key=lambda item: item["sort_key"])

    total_counts = np.sum([item["counts"] for item in grouped_items], axis=0)
    target_counts = np.outer(split_fracs, total_counts)
    target_n_files = split_fracs * float(len(tmp))

    current_counts = np.zeros_like(target_counts)
    current_n_files = np.zeros(3, dtype=float)
    assignments: dict[str, str] = {}

    for item in grouped_items:
        best_split_idx = 0
        best_score = None

        for split_idx in range(3):
            candidate_counts = current_counts.copy()
            candidate_counts[split_idx] += item["counts"]

            candidate_n_files = current_n_files.copy()
            candidate_n_files[split_idx] += item["n_files"]

            score = float(
                np.sum((candidate_counts - target_counts) ** 2)
                + 0.25 * np.sum((candidate_n_files - target_n_files) ** 2)
            )

            if best_score is None or score < best_score:
                best_score = score
                best_split_idx = split_idx

        current_counts[best_split_idx] += item["counts"]
        current_n_files[best_split_idx] += item["n_files"]
        assignments[item["group_id"]] = split_names[best_split_idx]

    out = tmp.copy()
    out["split"] = out["split_group_id"].map(assignments)
    return out.reset_index(drop=True)


def build_main_split_assignments(
    file_df: pd.DataFrame,
    split_strategy: str,
    evaluation_unit: str,
    group_mode: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, SplitSpec]:
    if split_strategy != "stratified_random":
        raise ValueError(f"Unsupported split_strategy: {split_strategy}")

    split_group_id = resolve_split_group_id(
        file_df,
        evaluation_unit=evaluation_unit,
        group_mode=group_mode,
    )

    file_df = file_df.copy()
    file_df["split_group_id"] = split_group_id.astype(str)

    if evaluation_unit == "file":
        split_df = _simple_stratified_file_split(
            file_df=file_df,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    else:
        split_df = _greedy_group_stratified_split(
            file_df=file_df,
            split_group_id=split_group_id,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    spec = SplitSpec(
        split_strategy=split_strategy,
        evaluation_unit=evaluation_unit,
        group_mode=group_mode,
        train_fraction=float(train_fraction),
        val_fraction=float(val_fraction),
        test_fraction=float(test_fraction),
        seed=int(seed),
    )
    return split_df, spec


def build_position_holdout_split_assignments(
    file_df: pd.DataFrame,
    spec: PositionHoldoutSpec,
) -> pd.DataFrame:
    if "step_idx" not in file_df.columns:
        raise ValueError("Position holdout requires parsable '__stepXX__' indices.")

    holdout_steps = {int(step) for step in spec.holdout_steps}

    out = file_df.copy()
    out["split_group_id"] = out["file_group_id"].astype(str)
    out["split"] = "unused"

    is_test = out["step_idx"].astype("Int64").isin(list(holdout_steps))
    test_df = out.loc[is_test].copy()
    remain_df = out.loc[~is_test].copy()

    if test_df.empty:
        raise ValueError("No files matched the requested position holdout steps.")
    if remain_df.empty:
        raise ValueError("No files remain for training after the position holdout.")

    remain_split = _simple_stratified_file_split(
        file_df=remain_df,
        train_fraction=1.0 - spec.val_fraction_within_remaining,
        val_fraction=spec.val_fraction_within_remaining,
        test_fraction=0.0,
        seed=spec.seed,
    )
    remain_split.loc[remain_split["split"] == "test", "split"] = "val"
    test_df["split"] = "test"

    return pd.concat([remain_split, test_df], ignore_index=True).reset_index(drop=True)


def _overlap_summary(split_df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in split_df.columns:
        return {"column_present": False}

    sets = {
        split_name: set(
            split_df.loc[split_df["split"] == split_name, column].dropna().astype(str).tolist()
        )
        for split_name in ["train", "val", "test"]
    }

    return {
        "column_present": True,
        "column": column,
        "n_unique_train": int(len(sets["train"])),
        "n_unique_val": int(len(sets["val"])),
        "n_unique_test": int(len(sets["test"])),
        "n_shared_train_val": int(len(sets["train"] & sets["val"])),
        "n_shared_train_test": int(len(sets["train"] & sets["test"])),
        "n_shared_val_test": int(len(sets["val"] & sets["test"])),
        "n_shared_all_three": int(len(sets["train"] & sets["val"] & sets["test"])),
    }


def summarize_split(split_df: pd.DataFrame) -> dict[str, Any]:
    depth_counts = (
        split_df.groupby(["split", "depth_mm"])
        .size()
        .reset_index(name="n_files")
        .sort_values(["split", "depth_mm"])
        .reset_index(drop=True)
    )

    summary: dict[str, Any] = {
        "n_total_files_used": int(len(split_df)),
        "depth_counts": depth_counts,
        "split_group_overlap": _overlap_summary(split_df, "split_group_id"),
    }

    if "step_idx" in split_df.columns:
        step_counts = (
            split_df.groupby(["split", "step_idx", "depth_mm"])
            .size()
            .reset_index(name="n_files")
            .sort_values(["split", "step_idx", "depth_mm"])
            .reset_index(drop=True)
        )
        summary["step_counts"] = step_counts
        summary["step_overlap"] = _overlap_summary(split_df, "step_idx")

    return summary


def save_split_summary(summary: dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(summary.get("depth_counts"), pd.DataFrame):
        summary["depth_counts"].to_csv(out_dir / "split_depth_counts.csv", index=False)

    if isinstance(summary.get("step_counts"), pd.DataFrame):
        summary["step_counts"].to_csv(out_dir / "split_step_counts.csv", index=False)

    payload = {
        "n_total_files_used": summary.get("n_total_files_used"),
        "split_group_overlap": summary.get("split_group_overlap"),
        "step_overlap": summary.get("step_overlap"),
    }

    with open(out_dir / "split_overlap_summary.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
