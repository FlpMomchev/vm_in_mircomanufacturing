from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .db import DashboardDB
from .settings import AppSettings, load_settings

RUN_MODES = {"single", "batch"}


@dataclass(frozen=True, slots=True)
class RunRequest:
    mode: str
    actual_depth_mm: float | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: int
    return_code: int
    stdout_log_path: Path
    stderr_log_path: Path
    output_dir: Path | None
    command_text: str


class FinalPredictionRunner:
    """Thin wrapper around scripts/final_prediction.py.

    Responsibilities:
    - read the latest detected file from the DB
    - create a run row
    - build the CLI command
    - execute the backend script
    - capture stdout/stderr to log files
    - mark the run as succeeded / failed in SQLite

    Responsibilities intentionally NOT here:
    - file watching
    - output parsing
    - dashboard UI
    """

    def __init__(self, settings: AppSettings, db: DashboardDB) -> None:
        self.settings = settings
        self.db = db

    def run_latest_detected(self, mode: str, actual_depth_mm: float | None = None) -> RunResult:
        request = RunRequest(mode=mode, actual_depth_mm=actual_depth_mm)
        self._validate_request(request)

        airborne = self.db.get_latest_detected_file("airborne")
        structure = self.db.get_latest_detected_file("structure")

        if airborne is None:
            raise RuntimeError("No latest airborne file is available in the database.")
        if structure is None:
            raise RuntimeError("No latest structure file is available in the database.")

        if airborne["status"] != "ready":
            raise RuntimeError(
                f"Latest airborne file is not ready for processing (status={airborne['status']})."
            )
        if structure["status"] != "ready":
            raise RuntimeError(
                f"Latest structure file is not ready for processing (status={structure['status']})."
            )

        active_run = self.db.get_active_run()
        if active_run is not None:
            raise RuntimeError(f"Another run is already active (run_id={active_run['id']}).")

        airborne_path = Path(str(airborne["file_path"])).expanduser().resolve()
        structure_path = Path(str(structure["file_path"])).expanduser().resolve()

        if not airborne_path.is_file():
            raise FileNotFoundError(f"Airborne file no longer exists: {airborne_path}")
        if not structure_path.is_file():
            raise FileNotFoundError(f"Structure file no longer exists: {structure_path}")

        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_row = self.db.create_run(
            airborne_detected_file_id=int(airborne["id"]),
            structure_detected_file_id=int(structure["id"]),
            mode=request.mode,
            actual_depth_mm=request.actual_depth_mm,
            command_text=None,
        )
        run_id = int(run_row["id"])

        output_dir = self._build_output_dir(request.mode, run_tag)
        stdout_log_path, stderr_log_path = self._build_log_paths(mode=request.mode, run_tag=run_tag)

        command = self._build_command(
            airborne_path=airborne_path,
            structure_path=structure_path,
            mode=request.mode,
            output_dir=output_dir,
            actual_depth_mm=request.actual_depth_mm,
        )
        command_text = subprocess.list2cmdline([str(part) for part in command])

        self.db.start_run(
            run_id=run_id,
            command_text=command_text,
            stdout_log_path=str(stdout_log_path),
            stderr_log_path=str(stderr_log_path),
        )

        try:
            completed = self._execute_command(
                command=command, stdout_log_path=stdout_log_path, stderr_log_path=stderr_log_path
            )

            if completed.returncode != 0:
                error_message = (
                    f"final_prediction.py failed with return code "
                    f"{completed.returncode}. See stderr log: {stderr_log_path}"
                )
                self.db.finish_run_failure(run_id=run_id, error_message=error_message)
                raise RuntimeError(error_message)

            self.db.finish_run_success(run_id=run_id, output_dir=output_dir)

            return RunResult(
                run_id=run_id,
                return_code=completed.returncode,
                stdout_log_path=stdout_log_path,
                stderr_log_path=stderr_log_path,
                output_dir=output_dir,
                command_text=command_text,
            )
        except Exception as exc:
            run_row = self.db.get_run(run_id)
            if run_row is not None and run_row["status"] not in {"succeeded", "failed"}:
                self.db.finish_run_failure(run_id=run_id, error_message=str(exc))
            raise

    def _validate_request(self, request: RunRequest) -> None:
        if request.mode not in RUN_MODES:
            raise ValueError(f"Invalid mode: {request.mode}")

        if request.actual_depth_mm is not None:
            float(request.actual_depth_mm)

    def _build_output_dir(self, mode: str, run_tag: str) -> Path:
        folder_name = f"dashboard__{mode}__{run_tag}"
        out_dir = self.settings.results_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir.resolve()

    def _build_log_paths(self, mode: str, run_tag: str) -> tuple[Path, Path]:
        base_name = f"{mode}__{run_tag}"
        stdout_path = self.settings.logs_dir / f"{base_name}__stdout.log"
        stderr_path = self.settings.logs_dir / f"{base_name}__stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        return stdout_path.resolve(), stderr_path.resolve()

    def _build_command(
        self,
        airborne_path: Path,
        structure_path: Path,
        mode: str,
        output_dir: Path,
        actual_depth_mm: float | None,
    ) -> list[str]:
        cmd = [
            str(self.settings.python_executable),
            str(self.settings.final_prediction_script),
            "--out-dir",
            str(output_dir),
            "--airborne-input-path",
            str(airborne_path),
            "--structure-input-path",
            str(structure_path),
            mode,
        ]

        if actual_depth_mm is not None:
            cmd.extend(["--actual-depth-mm", str(actual_depth_mm)])

        return cmd

    def _execute_command(
        self, command: list[str], stdout_log_path: Path, stderr_log_path: Path
    ) -> subprocess.CompletedProcess[str]:
        with (
            stdout_log_path.open("w", encoding="utf-8") as stdout_fh,
            stderr_log_path.open("w", encoding="utf-8") as stderr_fh,
        ):
            return subprocess.run(
                command,
                cwd=str(self.settings.repo_root),
                text=True,
                stdout=stdout_fh,
                stderr=stderr_fh,
                timeout=self.settings.run_timeout_sec,
                check=False,
            )


def preview_latest_command(mode: str, actual_depth_mm: float | None = None) -> str:
    settings = load_settings()
    db = DashboardDB(settings.db_path)
    db.init()

    airborne = db.get_latest_detected_file("airborne")
    structure = db.get_latest_detected_file("structure")

    if airborne is None:
        raise RuntimeError("No latest airborne file is available in the database.")
    if structure is None:
        raise RuntimeError("No latest structure file is available in the database.")

    runner = FinalPredictionRunner(settings=settings, db=db)
    run_tag = "preview"
    output_dir = runner._build_output_dir(mode=mode, run_tag=run_tag)
    command = runner._build_command(
        airborne_path=Path(str(airborne["file_path"])).expanduser().resolve(),
        structure_path=Path(str(structure["file_path"])).expanduser().resolve(),
        mode=mode,
        output_dir=output_dir,
        actual_depth_mm=actual_depth_mm,
    )
    return subprocess.list2cmdline(command)


if __name__ == "__main__":
    settings = load_settings()
    db = DashboardDB(settings.db_path)
    db.init()

    print("Latest airborne file:", db.get_latest_detected_file("airborne"))
    print("Latest structure file:", db.get_latest_detected_file("structure"))
    print("Latest detected files:", db.get_latest_detected_files())
    print("Preview single:", preview_latest_command(mode="single"))
    print("Preview batch :", preview_latest_command(mode="batch"))
