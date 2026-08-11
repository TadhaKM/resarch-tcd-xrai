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
from pathlib import Path
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


#: Dated copies of the database to keep. This file is the only record of every
#: name the robot knows, it is gitignored, and nothing else in the project
#: writes it anywhere else -- so a laptop reimaged between one open day and the
#: next takes every visitor with it. A week of daily copies is enough to notice
#: and recover from that, and costs about a megabyte.
_BACKUPS_TO_KEEP = 7


def backup_db() -> Optional[Path]:
    """Keep a dated copy of the database. At most one a day; oldest pruned.

    Uses sqlite3's own backup API rather than copying the file, because the
    voice loop may be mid-write: a plain copy of a WAL database can land
    without its journal and restore as a corrupt file, which is the one
    outcome worse than having no backup at all.
    """
    source = MODELS.db_path
    if not source.exists():
        return None
    folder = source.parent / "backups"
    target = folder / f"{source.stem}-{datetime.now(timezone.utc):%Y%m%d}.db"
    if target.exists():
        return target
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with _connection() as conn, sqlite3.connect(target) as copy:
            conn.backup(copy)
        for old in sorted(folder.glob(f"{source.stem}-*.db"))[:-_BACKUPS_TO_KEEP]:
            old.unlink(missing_ok=True)
        logger.info("Database backed up to %s", target.name)
        return target
    except (sqlite3.Error, OSError):
        # Never fatal. A robot that will not start because it could not write a
        # backup is worse than one running without today's copy.
        logger.exception("Could not back up the database")
        return None


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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_people_embeddings_person "
            "ON people_embeddings(person_id)"
        )

        # Migration. The table used to key on person_id, which allowed exactly
        # one face per person: whatever angle and light they happened to be in
        # when they gave their name, kept forever. Nothing deleted their name --
        # recognition simply stopped matching them, which from the visitor's
        # side is the robot forgetting who they are. Rebuilt in place, keeping
        # the embeddings already stored.
        existing = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='people_embeddings'"
        ).fetchone()
        if existing and "person_id INTEGER PRIMARY KEY" in existing[0]:
            logger.info("Migrating people_embeddings to allow several faces per person.")
            conn.execute("ALTER TABLE people_embeddings RENAME TO people_embeddings_v1")
            conn.execute(
                """
                CREATE TABLE people_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP
                )
                """
            )
            conn.execute(
                "INSERT INTO people_embeddings (person_id, embedding, created_at) "
                "SELECT person_id, embedding, created_at FROM people_embeddings_v1"
            )
            conn.execute("DROP TABLE people_embeddings_v1")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_people_embeddings_person "
                "ON people_embeddings(person_id)"
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


def rename_person(person_id: int, name: str) -> None:
    """Correct the name on an existing person, keeping their face and notes.

    Names arrive by speech-to-text, and speech-to-text is worst at exactly the
    proper nouns this stores -- one visitor was enrolled as "Telaget". Until
    this existed the only cure was editing the database by hand, which is not
    something the person standing in front of the robot can do, so a misheard
    name was permanent and the robot greeted them by it forever.

    Deliberately an UPDATE rather than delete-and-re-enrol: the embedding and
    any notes belong to the same visitor and should survive the spelling being
    fixed.
    """
    try:
        with _write_lock, _connection() as conn:
            conn.execute("UPDATE people SET name = ? WHERE id = ?", (name, person_id))
    except sqlite3.Error:
        logger.exception("Failed to rename person %s", person_id)
        raise


def get_person_name(person_id: int) -> Optional[str]:
    """Return a person's name, or None if unknown or never named."""
    try:
        with _connection() as conn:
            row = conn.execute(
                "SELECT name FROM people WHERE id = ?", (person_id,)
            ).fetchone()
    except sqlite3.Error:
        # Greeting someone is not worth failing a turn over.
        logger.exception("Failed to read person name")
        return None
    return row[0] if row and row[0] else None


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
                    EXISTS(SELECT 1 FROM people_embeddings e WHERE e.person_id = p.id)
                        AS has_embedding,
                    COUNT(n.id) AS note_count
                FROM people p
                LEFT JOIN memory_notes n ON n.person_id = p.id
                GROUP BY p.id, p.name, p.created_at
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


#: Faces kept per person. A face recognised from one stored angle is a face
#: recognised from one angle; several is what lets somebody be known standing
#: where they happen to stand. Bounded because every stored face is compared
#: against on every detection, and because a cap is what stops a slow drift of
#: near-identical crops crowding out the varied ones.
MAX_EMBEDDINGS_PER_PERSON = 12


def add_embedding(person_id: int, embedding: np.ndarray) -> None:
    """Remember another view of this person's face.

    Adds rather than replaces. The previous version overwrote, so a person had
    exactly one stored face for as long as they existed and being recognised
    depended on standing the way they stood the day they gave their name.
    """
    try:
        ensure_person(person_id)
        normalized = np.asarray(embedding, dtype=np.float32)
        with _write_lock, _connection() as conn:
            conn.execute(
                "INSERT INTO people_embeddings (person_id, embedding, created_at) "
                "VALUES (?, ?, ?)",
                (person_id, normalized.tobytes(), _now()),
            )
            # The newest views, so what survives is how they look now -- plus
            # the very first one, which is never dropped. That anchor matters
            # over a long gap: a busy afternoon in front of the robot adds a
            # dozen views of one angle and would otherwise evict the face it
            # was enrolled with, so somebody returning a year later would be
            # matched only against how they stood last March.
            conn.execute(
                """
                DELETE FROM people_embeddings
                WHERE person_id = ?
                  AND id <> (SELECT MIN(id) FROM people_embeddings WHERE person_id = ?)
                  AND id NOT IN (
                      SELECT id FROM people_embeddings
                      WHERE person_id = ? ORDER BY id DESC LIMIT ?
                  )
                """,
                (person_id, person_id, person_id, MAX_EMBEDDINGS_PER_PERSON - 1),
            )
    except sqlite3.Error:
        logger.exception("Failed to save embedding for person_id=%s", person_id)


def count_embeddings(person_id: int) -> int:
    """How many views of this person's face are stored."""
    try:
        with _connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM people_embeddings WHERE person_id = ?", (person_id,)
            ).fetchone()
    except sqlite3.Error:
        logger.exception("Failed to count embeddings for person_id=%s", person_id)
        return 0
    return int(row[0]) if row else 0


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
# Taken at import, which is once per run of the robot. After init_db so the
# copy has the current schema, and deliberately not on a timer: what this
# guards against is the file being lost between one open day and the next,
# not a write going wrong mid-session.
backup_db()
