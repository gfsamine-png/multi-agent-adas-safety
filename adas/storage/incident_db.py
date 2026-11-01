from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

KINDS = ("collision", "lane_departure")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT '',
    simulated  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    timestamp_s REAL NOT NULL,
    kind        TEXT NOT NULL,
    level       TEXT NOT NULL,
    track_id    INTEGER,
    ttc_s       REAL,
    offset      REAL,
    reason      TEXT NOT NULL,
    can_id      INTEGER,
    can_data    TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_run ON alerts(run_id, timestamp_s);
"""


class IncidentDb:
    """Small SQLite log of runs and the alerts raised in them."""

    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def start_run(self, source: str, note: str = "", simulated: bool = False) -> int:
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self.conn.execute(
            "INSERT INTO runs (source, created_at, note, simulated) VALUES (?, ?, ?, ?)",
            (source, created, note, int(simulated)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_alert(
        self,
        run_id: int,
        timestamp_s: float,
        kind: str,
        level: str,
        reason: str,
        track_id: int | None = None,
        ttc_s: float | None = None,
        offset: float | None = None,
        can_id: int | None = None,
        can_data: str | None = None,
    ) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown alert kind: {kind}")
        cur = self.conn.execute(
            "INSERT INTO alerts (run_id, timestamp_s, kind, level, track_id, ttc_s,"
            " offset, reason, can_id, can_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, timestamp_s, kind, level, track_id, ttc_s, offset, reason, can_id, can_data),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def runs(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]

    def alerts(self, run_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM alerts WHERE run_id = ? ORDER BY timestamp_s, id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()