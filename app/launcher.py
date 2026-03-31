from __future__ import annotations

import argparse
import shutil
import subprocess

from .db import DashboardDB
from .settings import load_settings
from .watcher import LatestFileWatcher, bootstrap_latest_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the VM dashboard with watcher bootstrap.")
    parser.add_argument(
        "--fresh", action="store_true", help="Reset dashboard DB and logs before launch."
    )
    parser.add_argument(
        "--purge-results",
        action="store_true",
        help="Also remove dashboard-visible fusion results on fresh start.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for the Shiny app.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the Shiny app.")
    parser.add_argument(
        "--reload", action="store_true", help="Use Shiny reload mode for development only."
    )
    return parser.parse_args()


def _reset_dashboard_state(settings, *, purge_results: bool) -> None:
    if settings.db_path.exists():
        settings.db_path.unlink()

    if settings.logs_dir.exists():
        shutil.rmtree(settings.logs_dir, ignore_errors=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)

    if settings.app_state_dir.exists():
        for child in settings.app_state_dir.iterdir():
            if child == settings.logs_dir or child == settings.db_path:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)

    if purge_results and settings.results_root.exists():
        shutil.rmtree(settings.results_root, ignore_errors=True)
        settings.results_root.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = _parse_args()
    settings = load_settings()

    if args.fresh:
        _reset_dashboard_state(settings, purge_results=args.purge_results)

    settings.ensure_directories()
    settings.validate()

    db = DashboardDB(settings.db_path)
    db.init()

    bootstrap_latest_files(settings=settings, db=db)

    watcher = LatestFileWatcher(settings=settings, db=db)
    watcher.start()

    cmd = [
        str(settings.python_executable),
        "-m",
        "shiny",
        "run",
        str(settings.repo_root / "app" / "shiny_app.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(settings.repo_root),
            check=False,
        )
        return int(completed.returncode)
    finally:
        watcher.stop()


if __name__ == "__main__":
    raise SystemExit(main())
