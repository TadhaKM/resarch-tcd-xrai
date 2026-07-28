"""Low-level SQLite access: schema, connections, and CRUD for people/memory_notes.

Safe to call from a background thread (the voice loop does): every call opens
its own short-lived connection rather than sharing one across threads, WAL
mode plus a busy_timeout absorb transient cross-connection lock contention,
and a process-wide lock serializes writes on top of that. Every function
catches sqlite3.Error and degrades (logs + returns an empty/no-op result)
instead of raising -- a lost memory note shouldn't crash the conversation.
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

import numpy as np

from config import MODELS

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    MODELS.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MODELS.db_path, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit/rollback via the `with conn:` transaction
    context, and always close it -- sqlite3.Connection's own context manager
    protocol commits/rolls back but does *not* close the connection."""
    conn = _connect()
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create tables/index if they don't exist yet. Safe to call repeatedly."""
    with _write_lock, _connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_notes (
                id INTEGER PRIMARY KEY,
                person_id INTEGER REFERENCES people(id),
                note TEXT,
                created_at TIMESTAMP
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_notes_person_id ON memory_notes(person_id)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS people_embeddings (
                person_id INTEGER PRIMARY KEY REFERENCES people(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP
            )
            """
        )


def ensure_person(person_id: int, name: Optional[str] = None) -> None:
    """Lazily create a people row so memory_notes' FK reference is valid."""
    try:
        with _write_lock, _connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO people (id, name, created_at) VALUES (?, ?, ?)",
                (person_id, name, _now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to ensure person row for person_id=%s", person_id)


def create_person(name: Optional[str] = None) -> int:
    """Create and return a new person id."""
    try:
        with _write_lock, _connection() as conn:
            cur = conn.execute(
                "INSERT INTO people (name, created_at) VALUES (?, ?)",
                (name, _now()),
            )
            return int(cur.lastrowid)
    except sqlite3.Error:
        logger.exception("Failed to create person row")
        raise


def list_people() -> list[dict[str, object]]:
    """Return enrolled people and lightweight counts for management UIs/CLI."""
    try:
        with _connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    p.id,
                    p.name,
                    p.created_at,
                    CASE WHEN e.person_id IS NULL THEN 0 ELSE 1 END AS has_embedding,
                    COUNT(n.id) AS note_count
                FROM people p
                LEFT JOIN people_embeddings e ON e.person_id = p.id
                LEFT JOIN memory_notes n ON n.person_id = p.id
                GROUP BY p.id, p.name, p.created_at, e.person_id
                ORDER BY p.id ASC
                """
            ).fetchall()
        return [
            {
                "person_id": row[0],
                "name": row[1],
                "created_at": row[2],
                "has_embedding": bool(row[3]),
                "note_count": row[4],
            }
            for row in rows
        ]
    except sqlite3.Error:
        logger.exception("Failed to list people")
        return []


def delete_person(person_id: int) -> None:
    """Delete a person and their embeddings + notes."""
    try:
        with _write_lock, _connection() as conn:
            conn.execute("DELETE FROM people_embeddings WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM memory_notes WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
    except sqlite3.Error:
        logger.exception("Failed to delete person_id=%s", person_id)


def save_embedding(person_id: int, embedding: np.ndarray) -> None:
    """Create or replace a person's face embedding."""
    try:
        ensure_person(person_id)
        normalized = np.asarray(embedding, dtype=np.float32)
        with _write_lock, _connection() as conn:
            conn.execute(
                """
                INSERT INTO people_embeddings (person_id, embedding, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    embedding = excluded.embedding,
                    created_at = excluded.created_at
                """,
                (person_id, normalized.tobytes(), _now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to save embedding for person_id=%s", person_id)


def get_embeddings() -> list[tuple[int, np.ndarray]]:
    """Return all enrolled face embeddings as float32 vectors."""
    try:
        with _connection() as conn:
            rows = conn.execute(
                "SELECT person_id, embedding FROM people_embeddings ORDER BY person_id ASC"
            ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32).copy()) for row in rows]
    except sqlite3.Error:
        logger.exception("Failed to read face embeddings")
        return []


def add_note(person_id: int, note: str) -> None:
    """Insert one memory note. Logs and swallows failures rather than raising."""
    try:
        ensure_person(person_id)
        with _write_lock, _connection() as conn:
            conn.execute(
                "INSERT INTO memory_notes (person_id, note, created_at) VALUES (?, ?, ?)",
                (person_id, note, _now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to add memory note for person_id=%s", person_id)


def get_notes(person_id: int) -> list[str]:
    """Return all notes for a person, oldest first. Empty list on failure."""
    try:
        with _connection() as conn:
            rows = conn.execute(
                "SELECT note FROM memory_notes WHERE person_id = ? ORDER BY created_at ASC, id ASC",
                (person_id,),
            ).fetchall()
        return [row[0] for row in rows]
    except sqlite3.Error:
        logger.exception("Failed to read memory notes for person_id=%s", person_id)
        return []


def count_notes(person_id: int) -> int:
    """Return how many notes a person has. 0 on failure (never blocks a turn on a read error)."""
    try:
        with _connection() as conn:
            (count,) = conn.execute(
                "SELECT COUNT(*) FROM memory_notes WHERE person_id = ?", (person_id,)
            ).fetchone()
        return count
    except sqlite3.Error:
        logger.exception("Failed to count memory notes for person_id=%s", person_id)
        return 0


def replace_notes_with_profile(person_id: int, profile_text: str) -> None:
    """Delete this person's existing notes and replace them with one consolidated row."""
    try:
        with _write_lock, _connection() as conn:
            conn.execute("DELETE FROM memory_notes WHERE person_id = ?", (person_id,))
            conn.execute(
                "INSERT INTO memory_notes (person_id, note, created_at) VALUES (?, ?, ?)",
                (person_id, profile_text, _now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to consolidate memory notes for person_id=%s", person_id)


init_db()
