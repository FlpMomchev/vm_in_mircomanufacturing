from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Iterable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .db import DashboardDB
from .settings import AppSettings, load_settings

logger = logging.getLogger(__name__)


def _normalize_extensions(values: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        item = value.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        normalized.append(item)
    return tuple(dict.fromkeys(normalized))


def _infer_modality_from_path(path: Path) -> str | None:
    parent = path.parent.name.lower()
    if parent == "airborne":
        return "airborne"
    if parent == "structure":
        return "structure"
    return None


def _iter_existing_candidate_files(
    root: Path,
    allowed_extensions: tuple[str, ...],
) -> list[Path]:
    if not root.exists():
        return []

    candidates: list[Path] = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed_extensions:
            continue
        candidates.append(path)

    return candidates


def bootstrap_latest_files(settings: AppSettings, db: DashboardDB) -> None:
    for modality in ("airborne", "structure"):
        modality_dir = settings.watch_dir / modality
        candidates = _iter_existing_candidate_files(
            modality_dir,
            settings.allowed_extensions,
        )
        if not candidates:
            continue

        latest = max(candidates, key=lambda p: p.stat().st_mtime_ns)
        db.upsert_detected_file(latest, modality=modality)
        logger.info(
            "Bootstrapped latest file | modality=%s | file=%s",
            modality,
            latest.name,
        )


class LatestFileEventHandler(FileSystemEventHandler):
    """Watch a folder and keep only the newest relevant file marked as latest."""

    def __init__(
        self, db: DashboardDB, allowed_extensions: Iterable[str], settle_time_sec: float = 0.75
    ) -> None:
        super().__init__()
        self.db = db
        self.allowed_extensions = _normalize_extensions(allowed_extensions)
        self.settle_time_sec = float(settle_time_sec)
        self._lock = threading.Lock()

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def _handle_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        src_path = getattr(event, "src_path", None)
        dest_path = getattr(event, "dest_path", None)

        candidate = dest_path or src_path
        if not candidate:
            return

        path = Path(candidate).expanduser().resolve()

        if path.suffix.lower() not in self.allowed_extensions:
            return

        modality = _infer_modality_from_path(path)
        if modality is None:
            return

        with self._lock:
            if not self._wait_until_readable(path):
                logger.debug("Skipped unreadable or unstable file: %s", path)
                return

            try:
                row = self.db.upsert_detected_file(path, modality=modality)
                logger.info(
                    "Latest detected file updated | modality=%s | id=%s | name=%s | status=%s",
                    row["modality"],
                    row["id"],
                    row["file_name"],
                    row["status"],
                )
            except FileNotFoundError:
                logger.debug("File disappeared before registration: %s", path)
            except Exception:
                logger.exception("Failed to register detected file: %s", path)

    def _wait_until_readable(self, path: Path) -> bool:
        """Wait briefly until the file exists and its size stops changing."""
        deadline = time.monotonic() + max(self.settle_time_sec, 0.0)
        last_size: int | None = None

        while time.monotonic() <= deadline:
            try:
                if not path.is_file():
                    time.sleep(0.1)
                    continue

                current_size = path.stat().st_size

                if last_size is not None and current_size == last_size:
                    with path.open("rb"):
                        return True

                last_size = current_size
            except OSError:
                pass

            time.sleep(0.1)

        try:
            if path.is_file():
                with path.open("rb"):
                    return True
        except OSError:
            return False

        return False


class LatestFileWatcher:
    """Small wrapper around watchdog for the dashboard app."""

    def __init__(self, settings: AppSettings, db: DashboardDB) -> None:
        self.settings = settings
        self.db = db
        self.handler = LatestFileEventHandler(db=db, allowed_extensions=settings.allowed_extensions)
        self.observer = Observer()
        self._started = False

    def start(self) -> None:
        if self._started:
            return

        self.settings.watch_dir.mkdir(parents=True, exist_ok=True)
        self.observer.schedule(self.handler, str(self.settings.watch_dir), recursive=True)
        self.observer.start()
        self._started = True

        logger.info("Watching folder: %s", self.settings.watch_dir)
        logger.info("Allowed extensions: %s", ", ".join(self.settings.allowed_extensions))

    def stop(self) -> None:
        if not self._started:
            return

        self.observer.stop()
        self.observer.join(timeout=5.0)
        self._started = False
        logger.info("Watcher stopped")

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Watcher interrupted by user")
        finally:
            self.stop()


def run_latest_file_watcher() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    settings = load_settings()
    db = DashboardDB(settings.db_path)
    db.init()

    bootstrap_latest_files(settings=settings, db=db)

    watcher = LatestFileWatcher(settings=settings, db=db)
    watcher.run_forever()


if __name__ == "__main__":
    run_latest_file_watcher()
