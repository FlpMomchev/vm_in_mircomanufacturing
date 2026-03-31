from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .db import DashboardDB
from .settings import AppSettings, load_settings


def _normalize_scalar(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except Exception:
            return value

    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else None


def _read_csv_rows(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty:
        return []

    records = df.to_dict(orient="records")
    return [{str(k): _normalize_scalar(v) for k, v in row.items()} for row in records]


def _path_or_none(raw: Any) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve(strict=False)


def _require_existing(path: Path, *, kind: str, run_id: int) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Run {run_id}: missing {kind}: {path}")
    return path


def _record_key_from_step_hole(step: Any, hole: Any) -> str | None:
    if step is None or hole is None:
        return None
    try:
        step_int = int(float(step))
    except Exception:
        return None

    hole_str = str(hole).strip()
    if not hole_str:
        return None

    return f"step={step_int:03d}__hole={hole_str}"


def _resolve_backend_run_dir(requested_output_dir: Path) -> Path:
    requested_output_dir = requested_output_dir.expanduser().resolve()

    if (requested_output_dir / "final").is_dir():
        return requested_output_dir

    candidates = [
        child
        for child in requested_output_dir.iterdir()
        if child.is_dir() and (child / "final").is_dir()
    ]

    if not candidates:
        raise FileNotFoundError(f"Could not resolve backend run dir under: {requested_output_dir}")

    candidates.sort(key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=True)
    return candidates[0].resolve()


def _collect_debug_pngs(modality_dir: Path) -> dict[str, list[str]]:
    core: list[str] = []
    padded: list[str] = []

    if not modality_dir.exists():
        return {"debug_core_paths": core, "debug_padded_paths": padded}

    for path in sorted(modality_dir.rglob("*.png")):
        name = path.name.lower()
        if "padded" in name:
            padded.append(str(path.resolve()))
        elif "core" in name:
            core.append(str(path.resolve()))

    return {
        "debug_core_paths": core,
        "debug_padded_paths": padded,
    }


class RunParser:
    """Read backend outputs and normalize them for the future dashboard UI."""

    def __init__(self, settings: AppSettings, db: DashboardDB) -> None:
        self.settings = settings
        self.db = db

    def parse_run(self, run_id: int) -> dict[str, Any]:
        run_row = self.db.get_run(run_id)
        if run_row is None:
            raise ValueError(f"Unknown run_id: {run_id}")

        raw_output_dir = run_row.get("output_dir")
        if not raw_output_dir:
            raise ValueError(f"Run {run_id} has no output_dir yet.")

        requested_output_dir = _path_or_none(raw_output_dir)
        if requested_output_dir is None:
            raise ValueError(f"Run {run_id} has invalid output_dir: {raw_output_dir!r}")

        _require_existing(requested_output_dir, kind="output_dir", run_id=run_id)
        resolved_run_dir = _resolve_backend_run_dir(requested_output_dir)
        final_dir = _require_existing(
            resolved_run_dir / "final",
            kind="final directory",
            run_id=run_id,
        )

        mode = str(run_row["mode"])

        _require_existing(
            final_dir / "final_predictions.csv",
            kind="final_predictions.csv",
            run_id=run_id,
        )

        if mode == "single":
            _require_existing(
                final_dir / "single_prediction_report.json",
                kind="single_prediction_report.json",
                run_id=run_id,
            )
        else:
            _require_existing(
                final_dir / "batch_quality_report.json",
                kind="batch_quality_report.json",
                run_id=run_id,
            )

        setup_audit = _read_json(final_dir / "setup_audit.json") or {}
        single_report = _read_json(final_dir / "single_prediction_report.json")
        batch_quality = _read_json(final_dir / "batch_quality_report.json")
        apples_to_apples = _read_json(final_dir / "apples_to_apples_report.json")

        final_predictions = _read_csv_rows(final_dir / "final_predictions.csv") or []
        final_predictions_long = _read_csv_rows(final_dir / "final_predictions_long.csv")

        modalities = {
            "airborne": self._parse_modality(
                resolved_run_dir / "airborne",
                raw_source_file_path=run_row.get("airborne_file_path"),
            ),
            "structure": self._parse_modality(
                resolved_run_dir / "structure",
                raw_source_file_path=run_row.get("structure_file_path"),
            ),
        }
        available_modalities = [name for name, payload in modalities.items() if payload["present"]]

        artifacts = {
            "debug_core_paths": [
                *modalities["airborne"]["debug_core_paths"],
                *modalities["structure"]["debug_core_paths"],
            ],
            "debug_padded_paths": [
                *modalities["airborne"]["debug_padded_paths"],
                *modalities["structure"]["debug_padded_paths"],
            ],
        }

        summary = self._build_summary(
            mode=mode,
            run_row=run_row,
            single_report=single_report,
            batch_quality=batch_quality,
            apples_to_apples=apples_to_apples,
        )

        return {
            "run": {
                "db_run_id": int(run_row["id"]),
                "mode": mode,
                "status": str(run_row["status"]),
                "actual_depth_mm": run_row.get("actual_depth_mm"),
                "requested_output_dir": str(requested_output_dir),
                "resolved_run_dir": str(resolved_run_dir),
                "final_dir": str(final_dir),
                "available_modalities": available_modalities,
                "created_at": run_row.get("created_at_utc_plus_2", run_row.get("created_at_utc")),
                "started_at": run_row.get("started_at_utc_plus_2", run_row.get("started_at_utc")),
                "finished_at": run_row.get(
                    "finished_at_utc_plus_2", run_row.get("finished_at_utc")
                ),
                "airborne_file_path": run_row.get("airborne_file_path"),
                "airborne_file_name": run_row.get("airborne_file_name"),
                "structure_file_path": run_row.get("structure_file_path"),
                "structure_file_name": run_row.get("structure_file_name"),
            },
            "summary": summary,
            "final_predictions": final_predictions,
            "final_predictions_long": final_predictions_long,
            "modalities": modalities,
            "artifacts": artifacts,
            "audit": setup_audit,
            "reports": {
                "single_prediction_report": single_report,
                "batch_quality_report": batch_quality,
                "apples_to_apples_report": apples_to_apples,
            },
            "report_paths": {
                "setup_audit_json": str((final_dir / "setup_audit.json").resolve()),
                "final_predictions_csv": str((final_dir / "final_predictions.csv").resolve()),
                "final_predictions_long_csv": str(
                    (final_dir / "final_predictions_long.csv").resolve()
                )
                if (final_dir / "final_predictions_long.csv").exists()
                else None,
                "single_prediction_report_json": str(
                    (final_dir / "single_prediction_report.json").resolve()
                )
                if (final_dir / "single_prediction_report.json").exists()
                else None,
                "batch_quality_report_json": str(
                    (final_dir / "batch_quality_report.json").resolve()
                )
                if (final_dir / "batch_quality_report.json").exists()
                else None,
                "apples_to_apples_report_json": str(
                    (final_dir / "apples_to_apples_report.json").resolve()
                )
                if (final_dir / "apples_to_apples_report.json").exists()
                else None,
            },
        }

    def parse_latest_succeeded_run(self) -> dict[str, Any]:
        for row in self.db.list_runs(limit=self.settings.history_limit):
            if row["status"] == "succeeded":
                return self.parse_run(int(row["id"]))
        raise RuntimeError("No succeeded run is available.")

    def list_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        rows = self.db.list_runs(limit=limit or self.settings.history_limit)
        history: list[dict[str, Any]] = []

        for row in rows:
            history.append(
                {
                    "id": int(row["id"]),
                    "mode": str(row["mode"]),
                    "status": str(row["status"]),
                    "actual_depth_mm": row.get("actual_depth_mm"),
                    "created_at": row.get("created_at_utc_plus_2", row.get("created_at_utc")),
                    "started_at": row.get("started_at_utc_plus_2", row.get("started_at_utc")),
                    "finished_at": row.get("finished_at_utc_plus_2", row.get("finished_at_utc")),
                    "output_dir": row.get("output_dir"),
                    "airborne_file_name": row.get("airborne_file_name"),
                    "structure_file_name": row.get("structure_file_name"),
                }
            )

        return history

    def _resolve_segment_output_path(self, modality_dir: Path, raw_value: Any) -> str | None:
        if raw_value is None:
            return None

        text = str(raw_value).strip()
        if not text:
            return None

        rel_path = Path(text)

        if rel_path.is_absolute():
            return str(rel_path.resolve())

        modality = modality_dir.name.lower()
        split_tag = "air" if modality == "airborne" else "structure"
        split_root = self.settings.data_root / "raw_data_extracted_splits" / split_tag / "live"

        candidates = [(modality_dir / rel_path).resolve(), (split_root / rel_path).resolve()]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return str(candidates[-1])

    def _parse_modality(
        self, modality_dir: Path, *, raw_source_file_path: str | None = None
    ) -> dict[str, Any]:
        present = modality_dir.exists() and modality_dir.is_dir()
        debug_payload = _collect_debug_pngs(modality_dir)

        if not present:
            return {
                "present": False,
                "path": None,
                "raw_source_file_path": raw_source_file_path,
                "classical_predictions_csv": None,
                "classical_predictions": None,
                "dl_predictions_csv": None,
                "dl_predictions": None,
                "fusion_predictions_csv": None,
                "fusion_predictions": None,
                "fusion_predictions_long_csv": None,
                "fusion_predictions_long": None,
                "features_csv_path": None,
                "segment_file_map": {},
                "manifest_csv_paths": [],
                **debug_payload,
            }

        features_candidates = sorted(modality_dir.glob("features_*.csv"))
        features_csv_path = str(features_candidates[0].resolve()) if features_candidates else None

        manifest_csv_paths = [
            str(path.resolve()) for path in sorted(modality_dir.rglob("segments_manifest.csv"))
        ]

        segment_file_map: dict[str, str] = {}
        manifest_raw_source: str | None = None

        for manifest_path_str in manifest_csv_paths:
            manifest_path = Path(manifest_path_str)
            manifest_df = pd.read_csv(manifest_path)
            if manifest_df.empty:
                continue

            if manifest_raw_source is None and "input_file" in manifest_df.columns:
                first_input = manifest_df["input_file"].iloc[0]
                if isinstance(first_input, str) and first_input.strip():
                    resolved_first_input = _path_or_none(first_input)
                    manifest_raw_source = (
                        str(resolved_first_input) if resolved_first_input is not None else None
                    )

            for _, row in manifest_df.iterrows():
                key = _record_key_from_step_hole(row.get("Step"), row.get("HoleID"))
                if key is None:
                    continue

                output_path = self._resolve_segment_output_path(
                    modality_dir,
                    row.get("output_path"),
                )
                if output_path is not None:
                    segment_file_map[key] = output_path

        return {
            "present": True,
            "path": str(modality_dir.resolve()),
            "raw_source_file_path": raw_source_file_path or manifest_raw_source,
            "classical_predictions_csv": str((modality_dir / "classical_predictions.csv").resolve())
            if (modality_dir / "classical_predictions.csv").exists()
            else None,
            "classical_predictions": _read_csv_rows(modality_dir / "classical_predictions.csv"),
            "dl_predictions_csv": str((modality_dir / "dl_predictions.csv").resolve())
            if (modality_dir / "dl_predictions.csv").exists()
            else None,
            "dl_predictions": _read_csv_rows(modality_dir / "dl_predictions.csv"),
            "fusion_predictions_csv": str((modality_dir / "fusion_predictions.csv").resolve())
            if (modality_dir / "fusion_predictions.csv").exists()
            else None,
            "fusion_predictions": _read_csv_rows(modality_dir / "fusion_predictions.csv"),
            "fusion_predictions_long_csv": str(
                (modality_dir / "fusion_predictions_long.csv").resolve()
            )
            if (modality_dir / "fusion_predictions_long.csv").exists()
            else None,
            "fusion_predictions_long": _read_csv_rows(modality_dir / "fusion_predictions_long.csv"),
            "features_csv_path": features_csv_path,
            "segment_file_map": segment_file_map,
            "manifest_csv_paths": manifest_csv_paths,
            **debug_payload,
        }

    def _build_summary(
        self,
        *,
        mode: str,
        run_row: dict[str, Any],
        single_report: dict[str, Any] | None,
        batch_quality: dict[str, Any] | None,
        apples_to_apples: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if mode == "single":
            report = single_report or {}
            final_prediction = report.get("final_prediction", {})
            return {
                "mode": "single",
                "has_ground_truth": bool(final_prediction.get("has_ground_truth")),
                "actual_depth_mm": report.get("actual_depth_mm"),
                "models": report.get("models", {}),
                "modality_fusions": report.get("modality_fusions", {}),
                "final_prediction": final_prediction,
                "apples_to_apples": None,
            }

        final_fusion = (batch_quality or {}).get("final_fusion", {})
        return {
            "mode": "batch",
            "has_ground_truth": bool(final_fusion.get("has_ground_truth")),
            "actual_depth_mm": run_row.get("actual_depth_mm"),
            "models": (batch_quality or {}).get("models", {}),
            "modality_fusions": (batch_quality or {}).get("modality_fusions", {}),
            "final_fusion": final_fusion,
            "apples_to_apples": apples_to_apples,
        }


if __name__ == "__main__":
    settings = load_settings()
    db = DashboardDB(settings.db_path)
    db.init()

    parser = RunParser(settings=settings, db=db)
    payload = parser.parse_latest_succeeded_run()

    print("run keys:", list(payload["run"].keys()))
    print("summary keys:", list(payload["summary"].keys()))
    print("modalities:", list(payload["modalities"].keys()))
    print("history rows:", len(parser.list_history()))
