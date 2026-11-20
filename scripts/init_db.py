"""Initialize the SQLite database from schema.sql."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import load_config
from app.logging_setup import setup_logging
from app.storage.db import init_db


def main() -> None:
    setup_logging()
    cfg = load_config("config.yaml")
    db_path = cfg.paths.db
    init_db(db_path)
    print(f"Database initialized at: {db_path}")


if __name__ == "__main__":
    main()

