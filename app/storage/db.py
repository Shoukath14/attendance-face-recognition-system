"""SQLite connection factory, WAL mode, migrations, and helpers."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("attendance.db")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_path: str | Path, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection for the calling thread."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    """Create tables from schema.sql if they don't exist."""
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_connection(db_path)
    try:
        conn.executescript(schema)
        conn.commit()
        logger.info("Database initialized at %s", db_path)
    finally:
        conn.close()

