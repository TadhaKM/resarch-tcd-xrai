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


#: How the robot is TOLD to say it does not know. These are not guesses at
#: model behaviour: brain/hub.py instructs "say you don't have that detail and
#: suggest asking Professor Laura Berry or whoever is hosting", and
#: brain/courses.py's REFUSALS says "say plainly that you do not have it and
#: send them to the Trinity Business School website or to whoever is hosting
#: them today". A deflection is therefore recognisable prose rather than a flag
#: the robot sets, because nothing in the pipeline ever tagged one.
#:
#: Matched on the reply, never on the question. Somebody ASKING "do you know
#: the fees?" is not a deflection; the robot answering "I don't have the fees"
#: is.
_DEFLECTION_MARKERS = (
    "i don't have", "i dont have", "i do not have",
    "i don't know", "i dont know", "i do not know",
    "i can't say", "i cant say", "i cannot say",
    "i'm not able to say", "im not able to say",
    "don't have that detail", "dont have that detail",
    "you'd want to check", "youd want to check", "you would want to check",
    "check the trinity", "trinity business school website",
    "whoever is hosting", "ask a member of staff", "speak to a member of staff",
    "best person to ask", "someone here can tell you", "ask whoever",
    "changes from year to year", "i'd rather not guess", "id rather not guess",
)


def looks_like_a_deflection(reply: str) -> bool:
    """Whether a reply was the robot saying it does not know.

    Deliberately conservative. A false positive puts a question a visitor
    HEARD answered onto a list of things to teach the robot, which wastes
    somebody's afternoon; a false negative just means one question is missed
    off a list that is advisory anyway.
    """
    lowered = (reply or "").lower()
    return any(marker in lowered for marker in _DEFLECTION_MARKERS)


def note_deflection(text: str) -> None:
    """Record that this question got "I don't know" for an answer.

    Keyed identically to note_question so the two land on the same row: the
    useful view is "asked eleven times, deflected nine of them", which needs
    both numbers against one question.
    """
    if not _available:
        return
    cleaned = _fold(text)
    if not cleaned:
        return
    try:
        with db._write_lock, db._connection() as conn:
            # Upsert rather than UPDATE: the reply is recorded from a different
            # place than the question, and if the two ever fold differently
            # this records the deflection rather than silently dropping it.
            conn.execute(
                """
                INSERT INTO visit_questions (day, question, asked, deflected, first_seen)
                VALUES (?, ?, 0, 1, ?)
                ON CONFLICT(day, question)
                DO UPDATE SET deflected = deflected + 1
                """,
                (_today(), cleaned, db._now()),
            )
    except sqlite3.Error:
        logger.debug("Could not record a deflection", exc_info=True)


def _fold(text: str) -> str:
    """The key a question is stored under. Case AND punctuation, so "What is
    the AI XR Hub?" and "what is the ai xr hub" are one question."""
    return " ".join(
        "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in (text or "")).split()
    ).lower()[:MAX_QUESTION_CHARS]


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
    blank = {"day": which or _today(), "counts": {}, "demos": {},
             "questions": [], "unanswered": [], "available": _available}
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
                {"question": q, "asked": int(n), "deflected": int(d or 0)}
                for q, n, d in conn.execute(
                    "SELECT question, asked, deflected FROM visit_questions "
                    "WHERE day = ? ORDER BY asked DESC, question LIMIT 12",
                    (target,),
                )
            ]
            # The list staff actually act on: what the robot was asked and
            # could not answer. Ordered by how often it happened, because a
            # question deflected nine times is worth teaching it and one
            # deflected once is somebody asking about the weather.
            unanswered = [
                {"question": q, "asked": int(n), "deflected": int(d or 0)}
                for q, n, d in conn.execute(
                    "SELECT question, asked, deflected FROM visit_questions "
                    "WHERE day = ? AND deflected > 0 "
                    "ORDER BY deflected DESC, question LIMIT 20",
                    (target,),
                )
            ]
    except sqlite3.Error:
        logger.exception("Failed to read the day's stats")
        return blank
    return {"day": target, "counts": counts, "demos": demos,
            "questions": questions, "unanswered": unanswered, "available": True}


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
                deflected  INTEGER NOT NULL DEFAULT 0,
                first_seen TIMESTAMP,
                PRIMARY KEY (day, question)
            )
            """
        )
        # Added after the first visits were recorded, so CREATE TABLE IF NOT
        # EXISTS does nothing on an existing database and every insert naming
        # the column would fail -- which, in a module contracted never to
        # raise, looks like statistics that simply stopped being collected.
        have = {row[1] for row in conn.execute("PRAGMA table_info(visit_questions)")}
        if "deflected" not in have:
            conn.execute("ALTER TABLE visit_questions ADD COLUMN "
                         "deflected INTEGER NOT NULL DEFAULT 0")
            logger.info("visit_questions: added deflected column")


try:
    _init_stats()
except (sqlite3.Error, OSError):
    _available = False
    logger.exception("Visit stats unavailable; the dashboard panel will say so")
