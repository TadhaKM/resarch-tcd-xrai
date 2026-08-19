"""Structured logging for human-robot interaction research, behind consent.

Professor Berry researches how people respond to immersive technology and to
human-robot interaction, and this robot has been generating exactly that data
every day and throwing it away. This makes it recordable -- but only
deliberately, only with the participant's spoken agreement, and only in a
shape an ethics committee would recognise.

READ THIS BEFORE TURNING IT ON
Nothing here is a substitute for ethics approval. Trinity requires research
involving human participants to be approved before data collection begins, and
this module does not and cannot grant that. What it does is make the technical
side match what an approved protocol would require, so that the gap between
"we have approval" and "we can collect" is a switch rather than a project:

  - OFF by default, and off again on every restart. A study that is running
    because somebody forgot to turn it off is the failure mode this is designed
    against, so the state is deliberately NOT persisted.
  - Nothing is recorded until the participant has heard what is being recorded
    and said yes, out loud, in this session.
  - A participant id is a random token minted per session. Names, face ids and
    person ids never enter this table -- the whole point is that a transcript
    here cannot be walked back to a person, even by whoever holds the database.
  - What IS recorded: the condition (which persona), what was said to the
    robot, what it said back, and the timing. That is what an HRI study needs
    and it is the least that will do.
  - Withdrawal deletes the session outright, which is why the id exists.

The A/B condition is the persona, because that is the manipulable variable this
robot already has: the same question answered as Friendly, Professional or
Consultant, with the wording, the voice and the posture all shifting together.
"""

import logging
import secrets
import sqlite3
import time
from typing import Optional

from brain import db

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2000

_available = True

#: Deliberately module state, not a database row: a study must never still be
#: running tomorrow because nobody remembered to stop it. Restarting the robot
#: -- which happens on every network drop -- turns this off.
_running = False
_session: Optional[str] = None
_consented = False
_condition = ""


def running() -> bool:
    return _running


def status() -> dict:
    """What the dashboard shows, and what the consent demo checks."""
    return {
        "running": _running,
        "consented": _consented,
        "session": _session or "",
        "condition": _condition,
        "available": _available,
    }


def start(condition: str = "") -> dict:
    """Begin a study session. Records NOTHING until consent is given.

    A fresh random id every time, so two participants are never merged and a
    withdrawal never takes somebody else's data with it.
    """
    global _running, _session, _consented, _condition
    _running = True
    _consented = False
    _condition = (condition or "").strip()[:40]
    _session = secrets.token_hex(8)
    logger.info("Study session %s started (condition %r); awaiting consent.",
                _session, _condition)
    return status()


def consent(given: bool) -> dict:
    """Record the participant's answer. Only True opens the recording."""
    global _consented, _running
    _consented = bool(given)
    if not given:
        # Declined is not merely "do not record" -- it ends the session, so a
        # later turn cannot quietly start recording the same person.
        logger.info("Study consent declined for %s; session ended.", _session)
        _running = False
    return status()


def stop() -> dict:
    global _running, _consented, _session
    _running = False
    _consented = False
    _session = None
    return status()


def record(
    said: str,
    replied: str,
    latency_s: float = 0.0,
    *,
    persona: str = "",
    first_word_s: float = 0.0,
    backend: str = "",
) -> None:
    """One exchange, if and only if a consented session is running.

    persona is the manipulated variable; condition is whatever free-text label
    the operator typed when arming. Both are kept because they answer different
    questions -- "which manner was this" and "which arm of my design was this".
    """
    if not (_available and _running and _consented and _session):
        return
    try:
        with db._write_lock, db._connection() as conn:
            conn.execute(
                """
                INSERT INTO study_turns
                    (session, condition, persona, said, replied,
                     latency_s, first_word_s, backend, at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (_session, _condition, persona or "",
                 (said or "")[:MAX_TEXT_CHARS], (replied or "")[:MAX_TEXT_CHARS],
                 float(latency_s), float(first_word_s), backend or "", db._now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to record a study turn")


def withdraw(session: Optional[str] = None) -> int:
    """Delete a session's data outright. Returns how many turns went.

    The participant's right, and it has to be one action: a study that can only
    "mark as withdrawn" is one where somebody's data is still there.
    """
    target = session or _session
    if not (_available and target):
        return 0
    try:
        with db._write_lock, db._connection() as conn:
            gone = conn.execute(
                "DELETE FROM study_turns WHERE session = ?", (target,)
            ).rowcount
    except sqlite3.Error:
        logger.exception("Failed to withdraw session %s", target)
        return 0
    logger.info("Withdrew study session %s (%d turn(s) deleted).", target, gone)
    return int(gone or 0)


def summary() -> dict:
    """Turn counts per condition, for the dashboard. No text."""
    if not _available:
        return {"sessions": 0, "turns": 0, "by_condition": {}}
    try:
        with db._connection() as conn:
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session) FROM study_turns"
            ).fetchone()
            turns = conn.execute("SELECT COUNT(*) FROM study_turns").fetchone()
            by = dict(conn.execute(
                "SELECT condition, COUNT(*) FROM study_turns GROUP BY condition"
            ))
    except sqlite3.Error:
        logger.exception("Failed to summarise study data")
        return {"sessions": 0, "turns": 0, "by_condition": {}}
    return {
        "sessions": int(sessions[0] if sessions else 0),
        "turns": int(turns[0] if turns else 0),
        "by_condition": {str(k or "default"): int(v) for k, v in by.items()},
    }


def _init_study() -> None:
    """Create the table if it does not exist. Safe to call repeatedly.

    No person_id, no name, no face. That absence is the design -- see the
    module docstring.
    """
    with db._write_lock, db._connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS study_turns (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session      TEXT NOT NULL,
                condition    TEXT NOT NULL DEFAULT '',
                persona      TEXT NOT NULL DEFAULT '',
                said         TEXT NOT NULL DEFAULT '',
                replied      TEXT NOT NULL DEFAULT '',
                latency_s    REAL NOT NULL DEFAULT 0,
                first_word_s REAL NOT NULL DEFAULT 0,
                backend      TEXT NOT NULL DEFAULT '',
                at           TIMESTAMP
            )
            """
        )
        # Columns added after the first sessions were recorded. CREATE TABLE IF
        # NOT EXISTS silently does nothing to a table that already exists, so a
        # database from before this change keeps the old shape and every insert
        # naming a new column fails -- which, in a module whose contract is
        # "never raise", would look like a study that simply records nothing.
        #
        # persona:      the docstring above has always said the condition IS the
        #               persona, but only the free-text box was ever stored, so
        #               the manipulated variable was missing from the data.
        # first_word_s: latency_s covers ctx.reply(), which does not return
        #               until the robot has finished SPEAKING -- so a long
        #               answer was recorded as a slow robot. Both are kept: one
        #               is responsiveness, the other is turn length.
        # backend:      Anthropic and the local model differ enough in latency
        #               that a wifi drop mid-study changes the condition without
        #               anybody choosing to.
        have = {row[1] for row in conn.execute("PRAGMA table_info(study_turns)")}
        for column, ddl in (
            ("persona", "persona TEXT NOT NULL DEFAULT ''"),
            ("first_word_s", "first_word_s REAL NOT NULL DEFAULT 0"),
            ("backend", "backend TEXT NOT NULL DEFAULT ''"),
        ):
            if column not in have:
                conn.execute(f"ALTER TABLE study_turns ADD COLUMN {ddl}")
                logger.info("study_turns: added %s column", column)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_study_session ON study_turns(session)")


try:
    _init_study()
except (sqlite3.Error, OSError):
    _available = False
    logger.exception("Study logging unavailable; research mode will refuse to start")
