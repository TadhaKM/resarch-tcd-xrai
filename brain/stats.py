"""What actually happened during a visit, counted as it happens.

The Hub runs open days and school visits and then has nothing to say about
them beyond an impression. Every fact needed for a debrief already passes
through this process -- turns taken, demos run, faces seen, questions asked --
and none of it was being kept. This keeps it, per day, in a shape a member of
staff can read off the dashboard.

WHAT IS COUNTED AND WHAT IS NOT
Counters and question text, keyed by day. Deliberately NOT: who said what, or
any link between a question and a person. The transcript already shows the
current session to whoever is standing at the dashboard, which is a different
thing from a permanent record of what named visitors asked -- and this table
outlives the session, gets backed up nightly, and would be the first thing to
regret. brain/memory.py is where per-person material belongs, with its own
rules; this is aggregate only.

Same shape as brain/settings.py and brain/qa_cache.py: borrows db's lock and
connection, carries _available, never raises. A dashboard panel is not worth
failing a visit over.
"""

import logging
import sqlite3
from typing import Optional

from brain import db

logger = logging.getLogger(__name__)

#: Distinct questions kept per day. Past this the tail is long and identical,
#: and the panel only ever shows the top handful.
MAX_QUESTIONS_PER_DAY = 400

#: Longest question text kept. Long enough for a real question, short enough
#: that a runaway transcript cannot fill the table.
MAX_QUESTION_CHARS = 160

_available = True


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def bump(metric: str, amount: int = 1) -> None:
    """Add to one of today's counters. Silent on failure."""
    if not _available or not metric:
        return
    try:
        with db._write_lock, db._connection() as conn:
            conn.execute(
                """
                INSERT INTO visit_stats (day, metric, count) VALUES (?, ?, ?)
                ON CONFLICT(day, metric) DO UPDATE SET count = count + excluded.count
                """,
                (_today(), metric[:40], int(amount)),
            )
    except sqlite3.Error:
        logger.exception("Failed to record %r", metric)


def note_question(text: str) -> None:
    """Record that this question was asked today, and how often.

    Stored without any speaker, deliberately -- see the module docstring. The
    text is what a visitor said out loud to a robot in a public room, and the
    useful fact is "eleven people asked about the masters", not who.
    """
    if not _available:
        return
    # Folded to letters, digits and single spaces before it becomes a key.
    # Case alone is not enough: "What is the AI XR Hub?" and "what is the ai
    # xr hub" are the same question asked twice, and counting them separately
    # is exactly the way a "most asked" list stops meaning anything.
    cleaned = " ".join(
        "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in (text or "")).split()
    ).lower()[:MAX_QUESTION_CHARS]
    if not cleaned:
        return
    try:
        with db._write_lock, db._connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM visit_questions WHERE day = ?", (_today(),)
            ).fetchone()
            # Counting an already-seen question is always allowed; only NEW
            # ones stop past the cap, so a busy day keeps sharpening its top
            # questions rather than freezing at whatever came first.
            known = conn.execute(
                "SELECT 1 FROM visit_questions WHERE day = ? AND question = ?",
                (_today(), cleaned),
            ).fetchone()
            if known is None and row and row[0] >= MAX_QUESTIONS_PER_DAY:
                return
            conn.execute(
                """
                INSERT INTO visit_questions (day, question, asked, first_seen)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(day, question) DO UPDATE SET asked = asked + 1
                """,
                (_today(), cleaned, db._now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to record a question")


def day(which: Optional[str] = None) -> dict:
    """One day's counters, its top questions, and its demo tally.

    Returns a well-formed empty day rather than raising or returning None, so
    the dashboard renders a panel saying "nothing yet" instead of an error.
    """
    blank = {"day": which or _today(), "counts": {}, "demos": {}, "questions": [], "available": _available}
    if not _available:
        return blank
    target = which or _today()
    try:
        with db._connection() as conn:
            counts = {}
            demos = {}
            for metric, count in conn.execute(
                "SELECT metric, count FROM visit_stats WHERE day = ?", (target,)
            ):
                if str(metric).startswith("demo:"):
                    demos[str(metric)[5:]] = int(count)
                else:
                    counts[str(metric)] = int(count)
            questions = [
                {"question": q, "asked": int(n)}
                for q, n in conn.execute(
                    "SELECT question, asked FROM visit_questions WHERE day = ? "
                    "ORDER BY asked DESC, question LIMIT 12",
                    (target,),
                )
            ]
    except sqlite3.Error:
        logger.exception("Failed to read the day's stats")
        return blank
    return {"day": target, "counts": counts, "demos": demos,
            "questions": questions, "available": True}


def recent_days(limit: int = 14) -> list[dict]:
    """Every day that has any activity, newest first, with its turn count."""
    if not _available:
        return []
    try:
        with db._connection() as conn:
            rows = conn.execute(
                "SELECT day, SUM(count) FROM visit_stats GROUP BY day "
                "ORDER BY day DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Failed to list days")
        return []
    return [{"day": r[0], "events": int(r[1] or 0)} for r in rows]


def _init_stats() -> None:
    """Create the tables if they do not exist. Safe to call repeatedly."""
    with db._write_lock, db._connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visit_stats (
                day    TEXT NOT NULL,
                metric TEXT NOT NULL,
                count  INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (day, metric)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS visit_questions (
                day        TEXT NOT NULL,
                question   TEXT NOT NULL,
                asked      INTEGER NOT NULL DEFAULT 0,
                first_seen TIMESTAMP,
                PRIMARY KEY (day, question)
            )
            """
        )


try:
    _init_stats()
except (sqlite3.Error, OSError):
    _available = False
    logger.exception("Visit stats unavailable; the dashboard panel will say so")
