from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default.resolve()
    return Path(value).expanduser().resolve()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(frozen=True, slots=True)
class AppSettings:
    repo_root: Path
    data_root: Path
    watch_dir: Path
    results_root: Path
    app_state_dir: Path
    logs_dir: Path
    db_path: Path
    python_executable: Path
    final_prediction_script: Path
    allowed_extensions: tuple[str, ...]
    run_timeout_sec: int
    history_limit: int

    def ensure_directories(self) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        (self.watch_dir / "airborne").mkdir(parents=True, exist_ok=True)
        (self.watch_dir / "structure").mkdir(parents=True, exist_ok=True)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.app_state_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        problems: list[str] = []

        if not self.repo_root.exists():
            problems.append(f"repo_root does not exist: {self.repo_root}")

        if not self.final_prediction_script.is_file():
            problems.append(f"final_prediction.py not found: {self.final_prediction_script}")

        if not self.python_executable.is_file():
            problems.append(f"python_executable not found: {self.python_executable}")

        if problems:
            raise FileNotFoundError("\n".join(problems))


def load_settings() -> AppSettings:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = _env_path("VM_DASH_DATA_ROOT", repo_root / "data")

    settings = AppSettings(
        repo_root=repo_root,
        data_root=data_root,
        watch_dir=_env_path("VM_DASH_WATCH_DIR", data_root / "incoming_raw"),
        results_root=_env_path("VM_DASH_RESULTS_ROOT", data_root / "fusion_results"),
        app_state_dir=_env_path("VM_DASH_STATE_DIR", data_root / "dashboard"),
        logs_dir=_env_path("VM_DASH_LOGS_DIR", data_root / "dashboard" / "logs"),
        db_path=_env_path("VM_DASH_DB_PATH", data_root / "dashboard" / "dashboard.sqlite3"),
        python_executable=_env_path("VM_DASH_PYTHON", Path(sys.executable)),
        final_prediction_script=_env_path(
            "VM_DASH_FINAL_PREDICTION_SCRIPT", repo_root / "scripts" / "final_prediction.py"
        ),
        allowed_extensions=(".flac", ".wav", ".h5", ".hdf5"),
        run_timeout_sec=_env_int("VM_DASH_RUN_TIMEOUT_SEC", 6 * 60 * 60),
        history_limit=_env_int("VM_DASH_HISTORY_LIMIT", 50),
    )

    settings.ensure_directories()
    settings.validate()

    return settings
