"""Structured logging for human-robot interaction research.

Professor Berry researches how people respond to immersive technology and to
human-robot interaction, and this robot has been generating exactly that data
every day and throwing it away. This makes it recordable -- but only
deliberately, only by a named operator, and only in a shape an ethics
committee would recognise.

CONSENT IS TAKEN OUTSIDE THIS ROBOT. Arming a session marks it consented; the
robot does not ask anybody out loud. That is the Hub's decision and a normal
one -- consent in HRI studies is usually taken on paper before the session --
but it means the operator carries it. start() records who armed the session
against every turn precisely because the spoken record is gone, and a
participant can still end the whole thing by saying so (see demos/study.py).

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
_operator = ""


def running() -> bool:
    return _running


def status() -> dict:
    """What the dashboard shows, and what the study demo checks."""
    return {
        "running": _running,
        "consented": _consented,
        "session": _session or "",
        "condition": _condition,
        "operator": _operator,
        "available": _available,
    }


def start(condition: str = "", operator: str = "") -> dict:
    """Begin a study session. Arming it IS the consent.

    A fresh random id every time, so two participants are never merged and a
    withdrawal never takes somebody else's data with it.

    WHY ARMING IS CONSENT. The robot used to read a notice aloud and wait for a
    spoken yes. The Hub's decision is that the researcher switching this on has
    already taken consent the way HRI studies normally take it -- on paper,
    before the session -- and that fifty seconds of notice before the first
    real exchange cost more than it bought.

    The cost of that decision is that the database no longer evidences consent,
    so `operator` exists to replace it: whoever armed the session is recorded
    against every turn in it. That is the trail an ethics reviewer asks for,
    and it is the only thing here that ties data to somebody accountable for
    it. It is a free-text name because this dashboard has no accounts -- it is
    an audit note, not an authentication.
    """
    global _running, _session, _consented, _condition, _operator
    _running = True
    _consented = True
    _condition = (condition or "").strip()[:40]
    _operator = (operator or "").strip()[:60]
    _session = secrets.token_hex(8)
    logger.info("Study session %s recording (condition %r, armed by %r).",
                _session, _condition, _operator or "unnamed")
    return status()


def consent(given: bool) -> dict:
    """Withdraw consent, or restore it.

    Arming a session now grants consent, so the caller that matters is the
    withdrawal path -- somebody in the room saying "stop recording" or "I do
    not consent". That still ends the session outright rather than muting it.
    """
    global _consented, _running
    _consented = bool(given)
    if not given:
        # Declined is not merely "do not record" -- it ends the session, so a
        # later turn cannot quietly start recording the same person.
        logger.info("Study consent declined for %s; session ended.", _session)
        _running = False
    return status()


def stop() -> dict:
    global _running, _consented, _session, _operator
    _running = False
    _consented = False
    _session = None
    _operator = ""
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
                    (session, condition, persona, operator, said, replied,
                     latency_s, first_word_s, backend, at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (_session, _condition, persona or "", _operator,
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


def sessions() -> list[dict]:
    """One row per recorded session, newest first, for the dashboard list.

    Counts and timings only -- no transcript text. The list is something an
    operator leaves open in a browser on a shared desk during an open day, so
    what was actually said is one deliberate click away (transcript()) rather
    than on screen by default.
    """
    if not _available:
        return []
    try:
        with db._connection() as conn:
            rows = conn.execute(
                """
                SELECT session,
                       MAX(condition)      AS condition,
                       MAX(operator)       AS operator,
                       COUNT(*)            AS turns,
                       MIN(at)             AS started,
                       MAX(at)             AS ended,
                       AVG(first_word_s)   AS avg_first_word_s,
                       AVG(latency_s)      AS avg_latency_s,
                       GROUP_CONCAT(DISTINCT persona) AS personas,
                       GROUP_CONCAT(DISTINCT backend) AS backends
                FROM study_turns
                GROUP BY session
                ORDER BY MIN(at) DESC
                """
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Failed to list study sessions")
        return []
    out = []
    for r in rows:
        out.append({
            "session": r[0], "condition": r[1] or "", "operator": r[2] or "",
            "turns": int(r[3] or 0), "started": str(r[4] or ""), "ended": str(r[5] or ""),
            "avg_first_word_s": round(float(r[6] or 0.0), 2),
            "avg_latency_s": round(float(r[7] or 0.0), 2),
            "personas": [p for p in (r[8] or "").split(",") if p],
            "backends": [b for b in (r[9] or "").split(",") if b],
            # True while this is the session currently being recorded into, so
            # the dashboard can say so rather than showing it as finished.
            "live": bool(_running and _session == r[0]),
        })
    return out


def transcript(session: str) -> list[dict]:
    """Every turn of one session, in order. The text, for export."""
    if not (_available and session):
        return []
    try:
        with db._connection() as conn:
            rows = conn.execute(
                """
                SELECT at, said, replied, persona, condition, operator,
                       latency_s, first_word_s, backend
                FROM study_turns WHERE session = ? ORDER BY id
                """,
                (session,),
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Failed to read session %s", session)
        return []
    return [{
        "at": str(r[0] or ""), "said": r[1] or "", "replied": r[2] or "",
        "persona": r[3] or "", "condition": r[4] or "", "operator": r[5] or "",
        "latency_s": float(r[6] or 0.0), "first_word_s": float(r[7] or 0.0),
        "backend": r[8] or "",
    } for r in rows]


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
            # Who armed the session. Since arming is now the consent, this is
            # the only record of a person taking responsibility for collecting
            # the data -- see start().
            ("operator", "operator TEXT NOT NULL DEFAULT ''"),
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
