from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

FILE_STATUSES = {"ready", "queued", "running", "processed", "failed", "ignored"}

RUN_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}

RUN_MODES = {"single", "batch"}
FILE_MODALITIES = {"airborne", "structure"}


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS detected_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    modality TEXT NOT NULL
        CHECK (modality IN ('airborne', 'structure')),
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_suffix TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    file_mtime_ns INTEGER NOT NULL,
    detected_at_utc_plus_2 TEXT NOT NULL,
    last_seen_at_utc_plus_2 TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ready'
        CHECK (status IN ('ready', 'queued', 'running', 'processed', 'failed', 'ignored')),
    is_latest INTEGER NOT NULL DEFAULT 0
        CHECK (is_latest IN (0, 1)),
    notes TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_detected_files_latest_per_modality
    ON detected_files(modality)
    WHERE is_latest = 1;

CREATE INDEX IF NOT EXISTS idx_detected_files_modality
    ON detected_files(modality);

CREATE INDEX IF NOT EXISTS idx_detected_files_status
    ON detected_files(status);

CREATE INDEX IF NOT EXISTS idx_detected_files_detected_at
    ON detected_files(detected_at_utc_plus_2 DESC);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at_utc_plus_2 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    airborne_detected_file_id INTEGER NOT NULL,
    structure_detected_file_id INTEGER NOT NULL,
    mode TEXT NOT NULL
        CHECK (mode IN ('single', 'batch')),
    actual_depth_mm REAL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    command_text TEXT,
    output_dir TEXT,
    stdout_log_path TEXT,
    stderr_log_path TEXT,
    error_message TEXT,
    created_at_utc_plus_2 TEXT NOT NULL,
    started_at_utc_plus_2 TEXT,
    finished_at_utc_plus_2 TEXT,
    updated_at_utc_plus_2 TEXT NOT NULL,
    FOREIGN KEY (airborne_detected_file_id) REFERENCES detected_files(id) ON DELETE RESTRICT,
    FOREIGN KEY (structure_detected_file_id) REFERENCES detected_files(id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs(status);

CREATE INDEX IF NOT EXISTS idx_runs_created_at
    ON runs(created_at_utc_plus_2 DESC);

CREATE INDEX IF NOT EXISTS idx_runs_airborne_detected_file_id
    ON runs(airborne_detected_file_id);

CREATE INDEX IF NOT EXISTS idx_runs_structure_detected_file_id
    ON runs(structure_detected_file_id);

"""


def utc_plus_2_now_iso() -> str:
    return (
        datetime.now(timezone(timedelta(hours=2)))
        .replace(microsecond=0)
        .isoformat()
        .replace("+02:00", "")
    )


class DashboardDB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row

        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")

            yield conn
            conn.commit()

        except Exception:
            conn.rollback()

            raise

        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def upsert_detected_file(self, file_path: str | Path, modality: str) -> dict[str, Any]:
        if modality not in FILE_MODALITIES:
            raise ValueError(f"Invalid modality: {modality}")

        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Detected file does not exist: {path}")

        stat = path.stat()
        now = utc_plus_2_now_iso()

        payload = {
            "modality": modality,
            "file_path": str(path),
            "file_name": path.name,
            "file_suffix": path.suffix.lower(),
            "file_size_bytes": int(stat.st_size),
            "file_mtime_ns": int(stat.st_mtime_ns),
            "now": now,
        }

        with self.connect() as conn:
            conn.execute(
                "UPDATE detected_files SET is_latest = 0 WHERE modality = ? AND is_latest = 1",
                (modality,),
            )

            conn.execute(
                """
                INSERT INTO detected_files (
                    modality,
                    file_path,
                    file_name,
                    file_suffix,
                    file_size_bytes,
                    file_mtime_ns,
                    detected_at_utc_plus_2,
                    last_seen_at_utc_plus_2,
                    status,
                    is_latest,
                    notes
                )
                VALUES (
                    :modality,
                    :file_path,
                    :file_name,
                    :file_suffix,
                    :file_size_bytes,
                    :file_mtime_ns,
                    :now,
                    :now,
                    'ready',
                    1,
                    NULL
                )
                ON CONFLICT(file_path) DO UPDATE SET
                    modality = excluded.modality,
                    file_name = excluded.file_name,
                    file_suffix = excluded.file_suffix,
                    file_size_bytes = excluded.file_size_bytes,
                    file_mtime_ns = excluded.file_mtime_ns,
                    last_seen_at_utc_plus_2 = excluded.last_seen_at_utc_plus_2,
                    status = CASE
                        WHEN detected_files.status IN ('queued', 'running')
                            THEN detected_files.status
                        ELSE 'ready'
                    END,
                    is_latest = 1
                """,
                payload,
            )

            row = conn.execute(
                "SELECT * FROM detected_files WHERE file_path = ?", (str(path),)
            ).fetchone()

        return dict(row)

    def get_latest_detected_file(self, modality: str) -> dict[str, Any] | None:
        if modality not in FILE_MODALITIES:
            raise ValueError(f"Invalid modality: {modality}")

        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM detected_files
                WHERE modality = ? AND is_latest = 1
                LIMIT 1
                """,
                (modality,),
            ).fetchone()

        return dict(row) if row is not None else None

    def get_latest_detected_files(self) -> dict[str, dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM detected_files
                WHERE is_latest = 1
                ORDER BY modality
                """
            ).fetchall()

        return {str(row["modality"]): dict(row) for row in rows}

    def get_detected_file(self, detected_file_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM detected_files WHERE id = ?", (detected_file_id,)
            ).fetchone()

        return dict(row) if row is not None else None

    def list_detected_files(
        self, limit: int = 50, modality: str | None = None
    ) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if modality is None:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM detected_files
                    ORDER BY detected_at_utc_plus_2 DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                if modality not in FILE_MODALITIES:
                    raise ValueError(f"Invalid modality: {modality}")
                rows = conn.execute(
                    """
                    SELECT *
                    FROM detected_files
                    WHERE modality = ?
                    ORDER BY detected_at_utc_plus_2 DESC, id DESC
                    LIMIT ?
                    """,
                    (modality, limit),
                ).fetchall()

        return [dict(row) for row in rows]

    def mark_detected_file_status(
        self,
        detected_file_id: int,
        status: str,
        notes: str | None = None,
    ) -> None:
        if status not in FILE_STATUSES:
            raise ValueError(f"Invalid detected file status: {status}")

        with self.connect() as conn:
            conn.execute(
                """
                UPDATE detected_files
                SET status = ?, notes = ?
                WHERE id = ?
                """,
                (status, notes, detected_file_id),
            )

    def create_run(
        self,
        airborne_detected_file_id: int,
        structure_detected_file_id: int,
        mode: str,
        actual_depth_mm: float | None = None,
        command_text: str | None = None,
    ) -> dict[str, Any]:
        if mode not in RUN_MODES:
            raise ValueError(f"Invalid run mode: {mode}")

        if self.get_detected_file(airborne_detected_file_id) is None:
            raise ValueError(f"Unknown airborne_detected_file_id: {airborne_detected_file_id}")

        if self.get_detected_file(structure_detected_file_id) is None:
            raise ValueError(f"Unknown structure_detected_file_id: {structure_detected_file_id}")

        now = utc_plus_2_now_iso()

        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (
                    airborne_detected_file_id,
                    structure_detected_file_id,
                    mode,
                    actual_depth_mm,
                    status,
                    command_text,
                    output_dir,
                    stdout_log_path,
                    stderr_log_path,
                    error_message,
                    created_at_utc_plus_2,
                    started_at_utc_plus_2,
                    finished_at_utc_plus_2,
                    updated_at_utc_plus_2
                )
                VALUES (?, ?, ?, ?, 'queued', ?, NULL, NULL, NULL, NULL, ?, NULL, NULL, ?)
                """,
                (
                    airborne_detected_file_id,
                    structure_detected_file_id,
                    mode,
                    actual_depth_mm,
                    command_text,
                    now,
                    now,
                ),
            )
            run_id = int(cursor.lastrowid)

            conn.execute(
                "UPDATE detected_files SET status = 'queued' WHERE id IN (?, ?)",
                (airborne_detected_file_id, structure_detected_file_id),
            )

            self._set_app_state_in_conn(conn, "current_run_id", str(run_id), now)

            row = conn.execute(
                """
                SELECT
                    r.*,
                    a.file_path AS airborne_file_path,
                    a.file_name AS airborne_file_name,
                    s.file_path AS structure_file_path,
                    s.file_name AS structure_file_name
                FROM runs AS r
                JOIN detected_files AS a
                    ON a.id = r.airborne_detected_file_id
                JOIN detected_files AS s
                    ON s.id = r.structure_detected_file_id
                WHERE r.id = ?
                """,
                (run_id,),
            ).fetchone()

        return dict(row)

    def start_run(
        self,
        run_id: int,
        command_text: str | None = None,
        stdout_log_path: str | None = None,
        stderr_log_path: str | None = None,
    ) -> None:
        now = utc_plus_2_now_iso()

        with self.connect() as conn:
            run_row = conn.execute(
                """
                SELECT airborne_detected_file_id, structure_detected_file_id
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise ValueError(f"Unknown run_id: {run_id}")

            conn.execute(
                """
                UPDATE runs
                SET status = 'running',
                    command_text = COALESCE(?, command_text),
                    stdout_log_path = COALESCE(?, stdout_log_path),
                    stderr_log_path = COALESCE(?, stderr_log_path),
                    started_at_utc_plus_2 = COALESCE(started_at_utc_plus_2, ?),
                    updated_at_utc_plus_2 = ?
                WHERE id = ?
                """,
                (command_text, stdout_log_path, stderr_log_path, now, now, run_id),
            )

            conn.execute(
                "UPDATE detected_files SET status = 'running' WHERE id IN (?, ?)",
                (
                    int(run_row["airborne_detected_file_id"]),
                    int(run_row["structure_detected_file_id"]),
                ),
            )

            self._set_app_state_in_conn(conn, "current_run_id", str(run_id), now)

    def finish_run_success(self, run_id: int, output_dir: str | Path | None) -> None:
        now = utc_plus_2_now_iso()
        output_dir_str = str(Path(output_dir).expanduser().resolve()) if output_dir else None

        with self.connect() as conn:
            run_row = conn.execute(
                """
                SELECT airborne_detected_file_id, structure_detected_file_id
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise ValueError(f"Unknown run_id: {run_id}")

            conn.execute(
                """
                UPDATE runs
                SET status = 'succeeded',
                    output_dir = ?,
                    finished_at_utc_plus_2 = ?,
                    updated_at_utc_plus_2 = ?
                WHERE id = ?
                """,
                (output_dir_str, now, now, run_id),
            )

            conn.execute(
                "UPDATE detected_files SET status = 'processed' WHERE id IN (?, ?)",
                (
                    int(run_row["airborne_detected_file_id"]),
                    int(run_row["structure_detected_file_id"]),
                ),
            )

            self._set_app_state_in_conn(conn, "last_completed_run_id", str(run_id), now)
            self._set_app_state_in_conn(conn, "current_run_id", "", now)

    def finish_run_failure(self, run_id: int, error_message: str) -> None:
        now = utc_plus_2_now_iso()

        with self.connect() as conn:
            run_row = conn.execute(
                """
                SELECT airborne_detected_file_id, structure_detected_file_id
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise ValueError(f"Unknown run_id: {run_id}")

            conn.execute(
                """
                UPDATE runs
                SET status = 'failed',
                    error_message = ?,
                    finished_at_utc_plus_2 = ?,
                    updated_at_utc_plus_2 = ?
                WHERE id = ?
                """,
                (error_message, now, now, run_id),
            )

            conn.execute(
                "UPDATE detected_files SET status = 'failed' WHERE id IN (?, ?)",
                (
                    int(run_row["airborne_detected_file_id"]),
                    int(run_row["structure_detected_file_id"]),
                ),
            )

            self._set_app_state_in_conn(conn, "current_run_id", "", now)

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    r.*,
                    a.file_path AS airborne_file_path,
                    a.file_name AS airborne_file_name,
                    s.file_path AS structure_file_path,
                    s.file_name AS structure_file_name
                FROM runs AS r
                JOIN detected_files AS a
                    ON a.id = r.airborne_detected_file_id
                JOIN detected_files AS s
                    ON s.id = r.structure_detected_file_id
                WHERE r.id = ?
                """,
                (run_id,),
            ).fetchone()

        return dict(row) if row is not None else None

    def get_active_run(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    r.*,
                    a.file_path AS airborne_file_path,
                    a.file_name AS airborne_file_name,
                    s.file_path AS structure_file_path,
                    s.file_name AS structure_file_name
                FROM runs AS r
                JOIN detected_files AS a
                    ON a.id = r.airborne_detected_file_id
                JOIN detected_files AS s
                    ON s.id = r.structure_detected_file_id
                WHERE r.status IN ('queued', 'running')
                ORDER BY r.created_at_utc_plus_2 DESC, r.id DESC
                LIMIT 1
                """
            ).fetchone()

        return dict(row) if row is not None else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    r.*,
                    a.file_path AS airborne_file_path,
                    a.file_name AS airborne_file_name,
                    s.file_path AS structure_file_path,
                    s.file_name AS structure_file_name
                FROM runs AS r
                JOIN detected_files AS a
                    ON a.id = r.airborne_detected_file_id
                JOIN detected_files AS s
                    ON s.id = r.structure_detected_file_id
                ORDER BY r.created_at_utc_plus_2 DESC, r.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_app_state(self, key: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()

        return None if row is None else str(row["value"])

    def set_app_state(self, key: str, value: str) -> None:
        now = utc_plus_2_now_iso()
        with self.connect() as conn:
            self._set_app_state_in_conn(conn, key, value, now)

    @staticmethod
    def _set_app_state_in_conn(
        conn: sqlite3.Connection,
        key: str,
        value: str,
        now: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO app_state (key, value, updated_at_utc_plus_2)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at_utc_plus_2 = excluded.updated_at_utc_plus_2
            """,
            (key, value, now),
        )
