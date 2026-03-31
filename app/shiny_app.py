from __future__ import annotations

import asyncio
import concurrent.futures
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from shiny import App, reactive, render, ui

from app.db import DashboardDB
from app.parser import RunParser
from app.runner import FinalPredictionRunner
from app.settings import load_settings
from app.spectrogram import make_spectrogram_figure

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

settings = load_settings()
db = DashboardDB(settings.db_path)
db.init()

pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _parse_actual_depth_mm(raw: str) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    return float(text)


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


def _metrics_dict_to_df(metrics: dict[str, Any] | None) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name, payload in metrics.items():
        row: dict[str, Any] = {"name": name}
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(value, (dict, list)):
                    continue
                row[str(key)] = _normalize_scalar(value)
        else:
            row["value"] = _normalize_scalar(payload)
        rows.append(row)

    return pd.DataFrame(rows)


def _rows_to_df(rows: list[dict[str, Any]] | None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _fmt_detected_file(row: dict[str, Any] | None, label: str) -> str:
    if row is None:
        return f"{label}: none"

    lines = [
        f"{label}: {row.get('file_name')}",
        f"status: {row.get('status')}",
        f"detected_at: {row.get('detected_at_utc_plus_2')}",
        f"path: {row.get('file_path')}",
    ]
    return "\n".join(lines)


def _fmt_active_run(row: dict[str, Any] | None) -> str:
    if row is None:
        return "No active run."

    lines = [
        f"run_id: {row.get('id')}",
        f"mode: {row.get('mode')}",
        f"status: {row.get('status')}",
        f"created_at: {row.get('created_at_utc_plus_2')}",
        f"started_at: {row.get('started_at_utc_plus_2')}",
        f"airborne: {row.get('airborne_file_name')}",
        f"structure: {row.get('structure_file_name')}",
    ]
    return "\n".join(lines)


def _fmt_latest_success(row: dict[str, Any] | None) -> str:
    if row is None:
        return "No completed run yet."

    lines = [
        f"run_id: {row.get('id')}",
        f"mode: {row.get('mode')}",
        f"status: {row.get('status')}",
        f"finished_at: {row.get('finished_at_utc_plus_2')}",
        f"output_dir: {row.get('output_dir')}",
    ]
    return "\n".join(lines)


def _read_history_rows() -> list[dict[str, Any]]:
    parser_db = DashboardDB(settings.db_path)
    parser_db.init()
    parser = RunParser(settings=settings, db=parser_db)
    return parser.list_history(limit=settings.history_limit)


def _history_run_label(row: dict[str, Any]) -> str:
    run_id = row.get("id")
    finished_at = row.get("finished_at") or "-"
    mode = row.get("mode") or "-"
    status = row.get("status") or "-"
    airborne = row.get("airborne_file_name") or "-"
    return f"{run_id} | {finished_at} | {mode} | {status} | {airborne}"


def _history_run_choices(rows: list[dict[str, Any]]) -> dict[str, str]:
    choices: dict[str, str] = {}
    for row in rows:
        if row.get("status") != "succeeded":
            continue
        run_id = row.get("id")
        if run_id is None:
            continue
        choices[str(run_id)] = _history_run_label(row)
    return choices


def _current_display_text(
    selected_history_run_id: int | None,
    latest_success: dict[str, Any] | None,
) -> str:
    if selected_history_run_id is not None:
        return f"displaying historical run_id={selected_history_run_id}"

    if latest_success is None:
        return "displaying latest succeeded run: none available"

    return f"displaying latest succeeded run_id={latest_success.get('id')}"


def _latest_succeeded_run(db: DashboardDB) -> dict[str, Any] | None:
    for row in db.list_runs(limit=settings.history_limit):
        if row["status"] == "succeeded":
            return row
    return None


def _read_shell_snapshot() -> dict[str, Any]:
    latest_files = db.get_latest_detected_files()
    airborne = latest_files.get("airborne")
    structure = latest_files.get("structure")
    active_run = db.get_active_run()
    latest_success = _latest_succeeded_run(db)

    ready = (
        active_run is None
        and airborne is not None
        and structure is not None
        and airborne.get("status") == "ready"
        and structure.get("status") == "ready"
    )

    return {
        "airborne": airborne,
        "structure": structure,
        "active_run": active_run,
        "latest_success": latest_success,
        "ready": ready,
    }


def _run_latest_detected(mode: str, actual_depth_mm: float | None) -> dict[str, Any]:
    runner_db = DashboardDB(settings.db_path)
    runner_db.init()
    runner = FinalPredictionRunner(settings=settings, db=runner_db)
    result = runner.run_latest_detected(mode=mode, actual_depth_mm=actual_depth_mm)

    parser_db = DashboardDB(settings.db_path)
    parser_db.init()
    parser = RunParser(settings=settings, db=parser_db)
    parsed = parser.parse_run(result.run_id)

    return {
        "run_id": result.run_id,
        "mode": mode,
        "output_dir": str(result.output_dir) if result.output_dir else None,
        "resolved_run_dir": parsed["run"]["resolved_run_dir"],
    }


def _read_latest_dashboard_payload() -> dict[str, Any] | None:
    parser_db = DashboardDB(settings.db_path)
    parser_db.init()
    parser = RunParser(settings=settings, db=parser_db)
    try:
        return parser.parse_latest_succeeded_run()
    except Exception:
        return None


def _existing_path_or_none(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser().resolve(strict=False)
    return str(path) if path.exists() else None


def _final_summary_text(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "No succeeded run available yet."

    summary = payload["summary"]
    run = payload["run"]

    if summary["mode"] == "single":
        final_prediction = summary.get("final_prediction", {})
        preds = final_prediction.get("predictions", []) or []
        first = preds[0] if preds else {}

        lines = [
            f"mode: {summary.get('mode')}",
            f"actual_depth_mm: {summary.get('actual_depth_mm')}",
            f"has_ground_truth: {summary.get('has_ground_truth')}",
            f"available_modalities: {', '.join(run.get('available_modalities', []))}",
            f"y_pred_mm: {first.get('y_pred')}",
            f"sigma_mm: {first.get('sigma')}",
            f"confidence_label: {first.get('confidence_label')}",
            f"mae_mm: {final_prediction.get('mae_mm')}",
            f"rmse_mm: {final_prediction.get('rmse_mm')}",
            f"sigma_mean_mm: {final_prediction.get('sigma_mean_mm')}",
        ]
        return "\n".join(lines)

    final_fusion = summary.get("final_fusion", {})
    lines = [
        f"mode: {summary.get('mode')}",
        f"actual_depth_mm: {summary.get('actual_depth_mm')}",
        f"has_ground_truth: {summary.get('has_ground_truth')}",
        f"available_modalities: {', '.join(run.get('available_modalities', []))}",
        f"mae_mm: {final_fusion.get('mae_mm')}",
        f"rmse_mm: {final_fusion.get('rmse_mm')}",
        f"sigma_mean_mm: {final_fusion.get('sigma_mean_mm')}",
        f"coverage_1sigma: {final_fusion.get('coverage_1sigma')}",
        f"coverage_2sigma: {final_fusion.get('coverage_2sigma')}",
    ]
    return "\n".join(lines)


def _uncertainty_text(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "No uncertainty payload available yet."

    summary = payload["summary"]

    if summary["mode"] == "single":
        final_prediction = summary.get("final_prediction", {})
        preds = final_prediction.get("predictions", []) or []
        first = preds[0] if preds else {}

        lines = [
            f"sigma_mm: {first.get('sigma')}",
            f"sigma_airborne_mm: {first.get('sigma_airborne_mm')}",
            f"sigma_structure_mm: {first.get('sigma_structure_mm')}",
            f"sigma_between_modalities_mm: {first.get('sigma_between_modalities_mm')}",
            f"z_abs: {first.get('z_abs')}",
            f"within_1sigma: {first.get('within_1sigma')}",
            f"within_2sigma: {first.get('within_2sigma')}",
            f"coverage_1sigma: {final_prediction.get('coverage_1sigma')}",
            f"coverage_2sigma: {final_prediction.get('coverage_2sigma')}",
        ]
        return "\n".join(lines)

    final_fusion = summary.get("final_fusion", {})
    lines = [
        f"sigma_mean_mm: {final_fusion.get('sigma_mean_mm')}",
        f"coverage_1sigma: {final_fusion.get('coverage_1sigma')}",
        f"coverage_2sigma: {final_fusion.get('coverage_2sigma')}",
        f"n_with_ground_truth: {final_fusion.get('n_with_ground_truth')}",
    ]
    return "\n".join(lines)


def _run_metadata_text(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "No parsed run metadata available yet."

    run = payload["run"]
    report_paths = payload["report_paths"]

    lines = [
        f"db_run_id: {run.get('db_run_id')}",
        f"mode: {run.get('mode')}",
        f"status: {run.get('status')}",
        f"requested_output_dir: {run.get('requested_output_dir')}",
        f"resolved_run_dir: {run.get('resolved_run_dir')}",
        f"final_dir: {run.get('final_dir')}",
        f"available_modalities: {', '.join(run.get('available_modalities', []))}",
        f"setup_audit_json: {report_paths.get('setup_audit_json')}",
        f"final_predictions_csv: {report_paths.get('final_predictions_csv')}",
    ]
    return "\n".join(lines)


def _audit_warnings_text(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "No audit available yet."

    warnings = payload.get("audit", {}).get("warnings", []) or []
    if not warnings:
        return "No setup warnings."
    return "\n".join(str(w) for w in warnings)


def _segment_choices(payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return []

    rows = payload.get("final_predictions", []) or []
    out: list[str] = []
    for row in rows:
        key = row.get("record_name")
        if isinstance(key, str) and key.strip():
            out.append(key)
    return out


def _effective_selected_key(
    payload: dict[str, Any] | None,
    requested_key: str | None,
) -> str | None:
    choices = _segment_choices(payload)
    if not choices:
        return None
    if requested_key in choices:
        return requested_key
    return choices[0]


def _resolve_spectrogram_source(
    payload: dict[str, Any] | None,
    modality: str,
    selected_key: str | None,
) -> str | None:
    if payload is None:
        return None

    mode = payload["run"]["mode"]
    mod_payload = payload["modalities"].get(modality, {})
    if not mod_payload.get("present"):
        return None

    if mode == "single":
        return _existing_path_or_none(mod_payload.get("raw_source_file_path"))

    segment_map = mod_payload.get("segment_file_map", {}) or {}
    if selected_key and selected_key in segment_map:
        return _existing_path_or_none(segment_map[selected_key])

    if segment_map:
        first_key = sorted(segment_map.keys())[0]
        return _existing_path_or_none(segment_map[first_key])

    return None


def _artifact_status_text(payload: dict[str, Any] | None, selected_key: str | None) -> str:
    if payload is None:
        return "No parsed run available yet."

    run = payload["run"]
    air = payload["modalities"]["airborne"]
    st = payload["modalities"]["structure"]

    air_src = _resolve_spectrogram_source(payload, "airborne", selected_key)
    st_src = _resolve_spectrogram_source(payload, "structure", selected_key)

    lines = [
        f"mode: {run.get('mode')}",
        f"selected_key: {selected_key}",
        f"airborne_debug_core_count: {len(air.get('debug_core_paths', []))}",
        f"airborne_debug_padded_count: {len(air.get('debug_padded_paths', []))}",
        f"structure_debug_core_count: {len(st.get('debug_core_paths', []))}",
        f"structure_debug_padded_count: {len(st.get('debug_padded_paths', []))}",
        f"airborne_spectrogram_source: {air_src}",
        f"structure_spectrogram_source: {st_src}",
    ]
    return "\n".join(lines)


def _first_or_none(values: list[str] | None) -> str | None:
    if not values:
        return None
    return values[0]


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Run control"),
        ui.input_text(
            "actual_depth_mm",
            "Actual depth (optional, mm)",
            placeholder="e.g. 0.2",
        ),
        ui.input_task_button("run_single_btn", "Run Single"),
        ui.input_task_button("run_batch_btn", "Run Batch"),
        ui.hr(),
        ui.h5("Readiness"),
        ui.output_text_verbatim("readiness_text"),
        width=320,
    ),
    ui.h3("VM dashboard"),
    ui.p("Pre-final version without styling and comprehensive plot analysis."),
    ui.navset_tab(
        ui.nav_panel(
            "Control",
            ui.h4("Latest airborne file"),
            ui.output_text_verbatim("latest_airborne_text", placeholder=True),
            ui.h4("Latest structure file"),
            ui.output_text_verbatim("latest_structure_text", placeholder=True),
            ui.h4("Current run"),
            ui.output_text_verbatim("active_run_text", placeholder=True),
            ui.h4("Latest succeeded run"),
            ui.output_text_verbatim("latest_success_text", placeholder=True),
            ui.h4("Last launch result"),
            ui.output_text_verbatim("launch_result_text", placeholder=True),
            ui.output_text_verbatim("dashboard_error_text", placeholder=True),
        ),
        ui.nav_panel(
            "Summary",
            ui.h4("Final summary"),
            ui.output_text_verbatim("final_summary_text", placeholder=True),
            ui.h4("Uncertainty"),
            ui.output_text_verbatim("uncertainty_text", placeholder=True),
        ),
        ui.nav_panel(
            "Models",
            ui.h4("Model metrics"),
            ui.output_data_frame("models_table"),
            ui.h4("Modality fusion metrics"),
            ui.output_data_frame("modality_fusions_table"),
        ),
        ui.nav_panel(
            "Final predictions",
            ui.h4("Final predictions table"),
            ui.output_data_frame("final_predictions_table"),
        ),
        ui.nav_panel(
            "Artifacts",
            ui.input_select(
                "spectrogram_key",
                "Segment / prediction key",
                choices=[],
                selected=None,
            ),
            ui.h4("Artifact status"),
            ui.output_text_verbatim("artifact_status_text", placeholder=True),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Airborne spectrogram"),
                    ui.output_plot("airborne_spectrogram_plot"),
                ),
                ui.card(
                    ui.card_header("Structure spectrogram"),
                    ui.output_plot("structure_spectrogram_plot"),
                ),
            ),
            ui.card(
                ui.card_header("Airborne debug core"),
                ui.output_image("airborne_debug_core_image"),
            ),
            ui.card(
                ui.card_header("Structure debug core"),
                ui.output_image("structure_debug_core_image"),
            ),
            ui.card(
                ui.card_header("Airborne debug padded"),
                ui.output_image("airborne_debug_padded_image"),
            ),
            ui.card(
                ui.card_header("Structure debug padded"),
                ui.output_image("structure_debug_padded_image"),
            ),
        ),
        ui.nav_panel(
            "History",
            ui.h4("Display target"),
            ui.output_text_verbatim("current_display_text", placeholder=True),
            ui.input_select(
                "history_run_id",
                "Succeeded run",
                choices={},
                selected=None,
            ),
            ui.layout_columns(
                ui.input_action_button("load_history_run_btn", "Load selected run"),
                ui.input_action_button("follow_latest_btn", "Follow latest"),
            ),
            ui.h4("Recent runs"),
            ui.output_data_frame("history_table"),
        ),
        ui.nav_panel(
            "Audit",
            ui.h4("Run metadata"),
            ui.output_text_verbatim("run_metadata_text", placeholder=True),
            ui.h4("Setup warnings"),
            ui.output_text_verbatim("audit_warnings_text", placeholder=True),
        ),
    ),
    title="VM dashboard",
)


def server(input, output, session):
    launch_note = reactive.value("No run launched from the UI yet.")
    parsed_payload_cache = reactive.value(None)
    parsed_payload_run_id = reactive.value(None)
    selected_history_run_id = reactive.value(None)
    history_choices_cache = reactive.value({})
    ui_error_message = reactive.value("")

    @reactive.calc
    def shell_snapshot() -> dict[str, Any]:
        reactive.invalidate_later(2)
        return _read_shell_snapshot()

    @reactive.calc
    def history_rows() -> list[dict[str, Any]]:
        shell_snapshot()
        return _read_history_rows()

    @reactive.effect
    def _refresh_history_choices():
        rows = history_rows()
        choices = _history_run_choices(rows)
        previous = history_choices_cache.get()

        if choices == previous:
            return

        current_selected = selected_history_run_id.get()
        selected_value = (
            str(current_selected)
            if current_selected is not None and str(current_selected) in choices
            else None
        )

        ui.update_select(
            "history_run_id", choices=choices, selected=selected_value, session=session
        )
        history_choices_cache.set(dict(choices))

    @reactive.effect
    def _refresh_parsed_payload_when_run_changes():
        snap = shell_snapshot()
        latest_success = snap["latest_success"]
        selected_run_id = selected_history_run_id.get()

        target_run_id: int | None
        if selected_run_id is not None:
            target_run_id = int(selected_run_id)
        elif latest_success is not None:
            target_run_id = int(latest_success["id"])
        else:
            target_run_id = None

        if target_run_id is None:
            parsed_payload_cache.set(None)
            parsed_payload_run_id.set(None)
            ui_error_message.set("")
            return

        if parsed_payload_run_id.get() == target_run_id:
            return

        parser_db = DashboardDB(settings.db_path)
        parser_db.init()
        parser = RunParser(settings=settings, db=parser_db)

        try:
            payload = parser.parse_run(target_run_id)
        except Exception as exc:
            parsed_payload_cache.set(None)
            parsed_payload_run_id.set(None)
            ui_error_message.set(f"Failed to parse run {target_run_id}: {exc}")
            return

        parsed_payload_cache.set(payload)
        parsed_payload_run_id.set(target_run_id)
        ui_error_message.set("")

    @reactive.calc
    def latest_dashboard_payload() -> dict[str, Any] | None:
        return parsed_payload_cache.get()

    segment_choices_cache = reactive.value([])

    @reactive.effect
    def _refresh_segment_choices_on_new_run():
        run_id = parsed_payload_run_id.get()
        payload = latest_dashboard_payload()
        previous = segment_choices_cache.get()

        if run_id is None or payload is None:
            if previous:
                ui.update_select(
                    "spectrogram_key",
                    choices=[],
                    selected=None,
                    session=session,
                )
                segment_choices_cache.set([])
            return

        choices = _segment_choices(payload)

        if choices == previous:
            return

        selected = choices[0] if choices else None

        ui.update_select(
            "spectrogram_key",
            choices=choices,
            selected=selected,
            session=session,
        )
        segment_choices_cache.set(list(choices))

    @ui.bind_task_button(button_id="run_single_btn")
    @reactive.extended_task
    async def run_single_task(actual_depth_mm: float | None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            pool,
            _run_latest_detected,
            "single",
            actual_depth_mm,
        )

    @ui.bind_task_button(button_id="run_batch_btn")
    @reactive.extended_task
    async def run_batch_task(actual_depth_mm: float | None):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            pool,
            _run_latest_detected,
            "batch",
            actual_depth_mm,
        )

    @reactive.effect
    @reactive.event(input.run_single_btn)
    def _launch_single():
        try:
            actual_depth_mm = _parse_actual_depth_mm(input.actual_depth_mm())
            ui_error_message.set("")
            launch_note.set("Launching single run...")
            run_single_task(actual_depth_mm)
        except Exception as exc:
            launch_note.set(f"Single run not started: {exc}")

    @reactive.effect
    @reactive.event(input.run_batch_btn)
    def _launch_batch():
        try:
            actual_depth_mm = _parse_actual_depth_mm(input.actual_depth_mm())
            ui_error_message.set("")
            launch_note.set("Launching batch run...")
            run_batch_task(actual_depth_mm)
        except Exception as exc:
            launch_note.set(f"Batch run not started: {exc}")

    @reactive.effect
    @reactive.event(input.load_history_run_btn)
    def _load_selected_history_run():
        raw_value = input.history_run_id()
        if raw_value is None:
            return
        try:
            selected_history_run_id.set(int(str(raw_value).strip()))
        except ValueError:
            return

    @reactive.effect
    @reactive.event(input.follow_latest_btn)
    def _follow_latest():
        selected_history_run_id.set(None)

    @render.text
    def readiness_text():
        snap = shell_snapshot()
        airborne = snap["airborne"]
        structure = snap["structure"]
        active_run = snap["active_run"]

        lines = [
            f"ready_to_run: {snap['ready']}",
            f"airborne_present: {airborne is not None}",
            f"structure_present: {structure is not None}",
            f"active_run: {'yes' if active_run is not None else 'no'}",
        ]

        if airborne is not None:
            lines.append(f"airborne_status: {airborne.get('status')}")
        if structure is not None:
            lines.append(f"structure_status: {structure.get('status')}")

        return "\n".join(lines)

    @render.text
    def latest_airborne_text():
        snap = shell_snapshot()
        return _fmt_detected_file(snap["airborne"], "airborne")

    @render.text
    def latest_structure_text():
        snap = shell_snapshot()
        return _fmt_detected_file(snap["structure"], "structure")

    @render.text
    def active_run_text():
        snap = shell_snapshot()
        return _fmt_active_run(snap["active_run"])

    @render.text
    def latest_success_text():
        snap = shell_snapshot()
        return _fmt_latest_success(snap["latest_success"])

    @render.text
    def dashboard_error_text():
        message = ui_error_message.get().strip()
        return message if message else "No dashboard errors."

    @render.text
    def launch_result_text():
        try:
            result = run_single_task.result()
            if result is not None:
                return (
                    f"single run finished\n"
                    f"run_id: {result['run_id']}\n"
                    f"resolved_run_dir: {result['resolved_run_dir']}"
                )
        except Exception as exc:
            return f"single run failed: {exc}"

        try:
            result = run_batch_task.result()
            if result is not None:
                return (
                    f"batch run finished\n"
                    f"run_id: {result['run_id']}\n"
                    f"resolved_run_dir: {result['resolved_run_dir']}"
                )
        except Exception as exc:
            return f"batch run failed: {exc}"

        return launch_note.get()

    @render.text
    def final_summary_text():
        return _final_summary_text(latest_dashboard_payload())

    @render.text
    def uncertainty_text():
        return _uncertainty_text(latest_dashboard_payload())

    @render.text
    def current_display_text():
        snap = shell_snapshot()
        return _current_display_text(
            selected_history_run_id=selected_history_run_id.get(),
            latest_success=snap["latest_success"],
        )

    @render.data_frame
    def history_table():
        rows = history_rows()
        df = pd.DataFrame(rows)
        return render.DataGrid(df, width="100%")

    @render.data_frame
    def models_table():
        payload = latest_dashboard_payload()
        if payload is None:
            return render.DataGrid(pd.DataFrame())

        df = _metrics_dict_to_df(payload["summary"].get("models"))
        return render.DataGrid(df, width="100%")

    @render.data_frame
    def modality_fusions_table():
        payload = latest_dashboard_payload()
        if payload is None:
            return render.DataGrid(pd.DataFrame())

        df = _metrics_dict_to_df(payload["summary"].get("modality_fusions"))
        return render.DataGrid(df, width="100%")

    @render.data_frame
    def final_predictions_table():
        payload = latest_dashboard_payload()
        if payload is None:
            return render.DataGrid(pd.DataFrame())

        df = _rows_to_df(payload.get("final_predictions"))
        return render.DataGrid(df, width="100%")

    @render.text
    def artifact_status_text():
        payload = latest_dashboard_payload()
        selected_key = _effective_selected_key(payload, input.spectrogram_key())
        return _artifact_status_text(payload, selected_key)

    @render.plot
    def airborne_spectrogram_plot():
        payload = latest_dashboard_payload()
        selected_key = _effective_selected_key(payload, input.spectrogram_key())
        path = _resolve_spectrogram_source(
            payload,
            "airborne",
            selected_key,
        )
        if path is None:
            return None
        try:
            return make_spectrogram_figure(path, "airborne")
        except Exception:
            return None

    @render.plot
    def structure_spectrogram_plot():
        payload = latest_dashboard_payload()
        selected_key = _effective_selected_key(payload, input.spectrogram_key())
        path = _resolve_spectrogram_source(
            payload,
            "structure",
            selected_key,
        )
        if path is None:
            return None
        try:
            return make_spectrogram_figure(path, "structure")
        except Exception:
            return None

    @render.image
    def airborne_debug_core_image():
        payload = latest_dashboard_payload()
        if payload is None:
            return None
        path = _existing_path_or_none(
            _first_or_none(payload["modalities"]["airborne"]["debug_core_paths"])
        )
        if path is None:
            return None
        return {"src": path, "width": "100%", "height": "auto"}

    @render.image
    def airborne_debug_padded_image():
        payload = latest_dashboard_payload()
        if payload is None:
            return None
        path = _existing_path_or_none(
            _first_or_none(payload["modalities"]["airborne"]["debug_padded_paths"])
        )
        if path is None:
            return None
        return {"src": path, "width": "100%", "height": "auto"}

    @render.image
    def structure_debug_core_image():
        payload = latest_dashboard_payload()
        if payload is None:
            return None
        path = _existing_path_or_none(
            _first_or_none(payload["modalities"]["structure"]["debug_core_paths"])
        )
        if path is None:
            return None
        return {"src": path, "width": "100%", "height": "auto"}

    @render.image
    def structure_debug_padded_image():
        payload = latest_dashboard_payload()
        if payload is None:
            return None
        path = _existing_path_or_none(
            _first_or_none(payload["modalities"]["structure"]["debug_padded_paths"])
        )
        if path is None:
            return None
        return {"src": path, "width": "100%", "height": "auto"}

    @render.text
    def run_metadata_text():
        return _run_metadata_text(latest_dashboard_payload())

    @render.text
    def audit_warnings_text():
        return _audit_warnings_text(latest_dashboard_payload())


app = App(app_ui, server)
app.on_shutdown(pool.shutdown)
