from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class AppDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT,
                    created_at TEXT NOT NULL,
                    display_time TEXT,
                    speaker TEXT,
                    text TEXT NOT NULL,
                    draft_text TEXT,
                    engine TEXT,
                    language TEXT,
                    elapsed_ms REAL
                );

                CREATE INDEX IF NOT EXISTS idx_transcripts_created_at
                ON transcripts(created_at);

                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    title TEXT,
                    summary TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    transcript_count INTEGER,
                    source_from_id INTEGER,
                    source_to_id INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_summaries_created_at
                ON summaries(created_at);

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    title TEXT,
                    detail TEXT,
                    people_count INTEGER,
                    summary_id INTEGER,
                    FOREIGN KEY(summary_id) REFERENCES summaries(id)
                );

                CREATE INDEX IF NOT EXISTS idx_events_created_at
                ON events(created_at);
                """
            )

    def add_transcript(self, record: dict[str, Any]) -> int:
        text = str(record.get("text") or "").strip()
        if not text:
            return 0

        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO transcripts (
                    record_id, created_at, display_time, speaker, text, draft_text,
                    engine, language, elapsed_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("id"),
                    datetime.now().isoformat(timespec="seconds"),
                    record.get("time"),
                    record.get("speaker"),
                    text,
                    record.get("draft_text"),
                    record.get("engine"),
                    record.get("language"),
                    record.get("elapsed_ms"),
                ),
            )
            return int(cur.lastrowid)

    def list_transcripts(self, limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 1000))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, record_id, created_at, display_time, speaker, text, draft_text,
                       engine, language, elapsed_ms
                FROM transcripts
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_transcripts_for_summary(self, limit: int = 80) -> list[dict[str, Any]]:
        rows = self.list_transcripts(limit)
        return list(reversed(rows))

    def search_transcripts(self, query: str, limit: int = 200) -> list[dict[str, Any]]:
        query = query.strip()
        limit = max(1, min(int(limit), 1000))
        if not query:
            return self.list_transcripts(limit)

        pattern = f"%{query}%"
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, record_id, created_at, display_time, speaker, text, draft_text,
                       engine, language, elapsed_ms
                FROM transcripts
                WHERE text LIKE ? OR draft_text LIKE ? OR speaker LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (pattern, pattern, pattern, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_transcripts(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM transcripts")

    def add_summary(
        self,
        *,
        title: str,
        summary: str,
        provider: str,
        model: str,
        transcript_count: int,
        source_from_id: int | None,
        source_to_id: int | None,
    ) -> int:
        summary = summary.strip()
        if not summary:
            return 0

        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO summaries (
                    created_at, title, summary, provider, model, transcript_count,
                    source_from_id, source_to_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    title,
                    summary,
                    provider,
                    model,
                    transcript_count,
                    source_from_id,
                    source_to_id,
                ),
            )
            return int(cur.lastrowid)

    def list_summaries(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, title, summary, provider, model, transcript_count,
                       source_from_id, source_to_id
                FROM summaries
                WHERE TRIM(COALESCE(summary, '')) != ''
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_event(
        self,
        *,
        event_type: str,
        title: str,
        detail: str = "",
        people_count: int | None = None,
        summary_id: int | None = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO events (
                    created_at, event_type, title, detail, people_count, summary_id
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    event_type,
                    title,
                    detail,
                    people_count,
                    summary_id,
                ),
            )
            return int(cur.lastrowid)

    def list_events(self, limit: int = 80) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, created_at, event_type, title, detail, people_count, summary_id
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_events(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM events")
