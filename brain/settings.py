"""Small operator choices that must survive a restart.

The robot restarts a lot -- every network drop relaunches it -- and until this
existed, every restart silently undid the operator's voice pick and left the
robot back on the config default. A choice somebody made deliberately from the
dashboard should still be in force the next morning.

One key-value table, and deliberately nothing more. Anything with structure
(features, the dashboard layout) has its own module with its own validation;
this is for single small strings whose only failure mode is absence. Follows
brain/qa_cache.py exactly: borrows db's lock and connection (a second lock
would not serialize against db.py's writes), carries _available so a failed
CREATE TABLE degrades every call to its default instead of raising out of an
import, and never raises.
"""

import logging
import sqlite3
from typing import Optional

from brain import db

logger = logging.getLogger(__name__)

#: Guard against a runaway caller treating this as a real store.
_MAX_VALUE_CHARS = 200

_available = True


def get(key: str, default: str = "") -> str:
    """The saved value, or `default` when unset or unavailable."""
    if not _available:
        return default
    try:
        with db._connection() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
    except sqlite3.Error:
        logger.exception("Failed to read setting %r", key)
        return default
    return row[0] if row and isinstance(row[0], str) else default


def put(key: str, value: str) -> bool:
    """Save one choice. False (and a log line) rather than ever raising."""
    if not _available:
        return False
    try:
        with db._write_lock, db._connection() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, (value or "")[:_MAX_VALUE_CHARS], db._now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to save setting %r", key)
        return False
    return True


def _init_settings() -> None:
    """Create the table if it does not exist. Safe to call repeatedly."""
    with db._write_lock, db._connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TIMESTAMP
            )
            """
        )


try:
    _init_settings()
except (sqlite3.Error, OSError):
    _available = False
    logger.exception("Settings unavailable; operator choices will not survive a restart")
