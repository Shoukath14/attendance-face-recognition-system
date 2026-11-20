PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no       TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    thumbnail_path TEXT,
    active        INTEGER NOT NULL DEFAULT 1,
    consent_at    TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    model_name  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model_name);

CREATE TABLE IF NOT EXISTS sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    start_time       TEXT NOT NULL,
    late_after_min   INTEGER NOT NULL DEFAULT 10,
    end_time         TEXT,
    status           TEXT NOT NULL DEFAULT 'scheduled'
);

CREATE TABLE IF NOT EXISTS attendance (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    student_id   INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    marked_at    TEXT NOT NULL,
    status       TEXT NOT NULL,
    confidence   REAL,
    source       TEXT NOT NULL DEFAULT 'auto',
    note         TEXT,
    UNIQUE (session_id, student_id)
);

CREATE TABLE IF NOT EXISTS review_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
    snapshot_path   TEXT NOT NULL,
    reason          TEXT NOT NULL,
    top_candidates  TEXT,
    created_at      TEXT NOT NULL,
    resolved        INTEGER NOT NULL DEFAULT 0,
    resolution      TEXT
);

