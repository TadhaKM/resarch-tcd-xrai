"""Answer cache: a question asked twice is answered from SQLite, not the model.

On the 1.5B local model a reply takes 4-9 seconds, and an open day is the same
handful of questions put to the robot all day by different groups -- so the
second group stands and waits while a sentence that already exists word for
word is generated again. Questions are matched after normalisation ("Who runs
the AI XR Hub?" and "so who runs the ai xr hub" are one entry), which is what
makes a hit likely at all: nobody asks twice in exactly the same words.

The store is brain/db.py's database, not a second one -- same file, same
short-lived-connection-per-call pattern, same process-wide write lock (which
is why the private names are reached for: a cache with its own lock would not
be serialized against the writes db.py is already doing). Same failure policy
too: every function here logs and degrades, because a cache that raises is
strictly worse than no cache.
"""

import hashlib
import logging
import re
import sqlite3
from typing import NamedTuple

from . import db

logger = logging.getLogger(__name__)

#: Entries kept, least-recently-used evicted. This is a cache, not a log: an
#: open day is a few dozen distinct questions, and the bound exists so that a
#: week of unattended running cannot grow the database without limit.
MAX_ENTRIES = 300

#: A question shorter than this is not safely comparable across conversations.
#: "yes", "what", "and the other one" all mean whatever was just said, so an
#: answer stored under one of them would be replayed into a conversation that
#: was about something else entirely.
MIN_QUESTION_WORDS = 3
MIN_QUESTION_CHARS = 10

#: Below this an answer is a fragment or a bare acknowledgement, neither of
#: which is worth pinning to a question for the rest of the day.
MIN_ANSWER_CHARS = 8

_APOSTROPHE_RE = re.compile(r"['’]")
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")

#: Removed from the FRONT of a question, repeatedly, longest phrase first.
#: These are how people actually open a question out loud, and they are noise
#: for matching purposes -- without stripping them, "so who runs the ai xr hub"
#: and "who runs the AI XR Hub?" are two entries and the cache never hits.
#: Written in post-normalisation form ("id like to know", not "I'd like to
#: know") because that is the text they are matched against.
_LEADING_FILLER = tuple(
    sorted(
        (
            tuple(phrase.split())
            for phrase in (
                "hey reachy",
                "hi reachy",
                "hello reachy",
                "ok reachy",
                "okay reachy",
                "excuse me reachy",
                "reachy",
                "so",
                "um",
                "uh",
                "erm",
                "well",
                "like",
                "just",
                "please",
                "and",
                "but",
                "actually",
                "can you tell me",
                "could you tell me",
                "can you tell us",
                "could you tell us",
                "tell me",
                "tell us",
                "i was wondering",
                "i wanted to know",
                "i want to know",
                "id like to know",
                "do you know",
                "can you",
                "could you",
                "would you",
                "quick question",
                "one question",
                "another question",
            )
        ),
        key=len,
        reverse=True,
    )
)

#: A question containing one of these is about the person asking, not about the
#: Hub or the robot, so its answer belongs to that person and nobody else.
#: Unrecognised visitors all share person_id 0, so without this the answer to
#: "what did I just say" would be served to the next stranger who asked it.
#: Object pronouns are deliberately absent apart from "me": "tell me about the
#: Hub" is a general question, and its "tell me" is stripped as filler above.
_PERSONAL_MARKERS = frozenset({"i", "im", "me", "my", "mine", "myself", "we", "our", "ours"})

#: Words that point at something only the conversation they were said in can
#: resolve. "Say that again", "what about the other one", "tell me more about
#: it" all normalise to short, stable keys and would otherwise be answered
#: forever with whatever the first group happened to be discussing -- a
#: plausible answer to a question nobody in the room asked. A minimum word
#: count does not catch these: "the other one" is three words and thirteen
#: characters, comfortably over both floors.
_DEICTIC_MARKERS = frozenset({
    "it", "its", "that", "this", "those", "these", "them", "they",
    "one", "ones", "again", "else", "other", "another", "there",
})

#: An answer that is a failure being read out rather than an answer. Caching
#: one turns a bad minute -- Ollama restarting, the network dropping -- into a
#: permanent wrong answer to a question that will be asked again all day.
_ERROR_ANSWER_RE = re.compile(
    r"\b(error|exception|traceback|timed out|timeout|connection refused|unavailable|nonetype)\b",
    re.IGNORECASE,
)


class CachedAnswer(NamedTuple):
    """An answer already given, and the emotion it was delivered with."""

    text: str
    emotion: str


def normalize_question(question: str) -> str:
    """Reduce a spoken question to the key that two askers of the same thing share."""
    text = _APOSTROPHE_RE.sub("", question.lower())
    text = _NON_WORD_RE.sub(" ", text)
    words = _WHITESPACE_RE.sub(" ", text).strip().split()

    stripped_one = True
    while stripped_one and words:
        stripped_one = False
        for filler in _LEADING_FILLER:
            # Never strip the question down to nothing: "hey reachy" on its own
            # is an unanswerable key, and an empty one would collide with every
            # other unanswerable key.
            if len(words) > len(filler) and tuple(words[: len(filler)]) == filler:
                words = words[len(filler) :]
                stripped_one = True
                break

    return " ".join(words)


def _context_digest(extra_system: str | None) -> str:
    """Fingerprint the per-turn system text, which changes what the answer is."""
    if not extra_system:
        return ""
    return hashlib.sha256(extra_system.strip().encode("utf-8")).hexdigest()[:16]


def _is_cacheable_question(normalized: str) -> bool:
    words = normalized.split()
    if len(words) < MIN_QUESTION_WORDS or len(normalized) < MIN_QUESTION_CHARS:
        return False
    if _PERSONAL_MARKERS.intersection(words):
        return False
    return not _DEICTIC_MARKERS.intersection(words)


def _is_cacheable_answer(answer: str) -> bool:
    text = answer.strip()
    if len(text) < MIN_ANSWER_CHARS:
        return False
    # No closing punctuation means the reply stopped early -- the token cap, or
    # a backend that died mid-sentence and left stream_reply holding half an
    # answer. Serving that half back instantly, forever, is a worse bug than
    # the slow answer this whole module exists to avoid.
    if text[-1] not in ".!?":
        return False
    return not _ERROR_ANSWER_RE.search(text)


#: False when the table could not be created, which turns lookup and remember
#: into no-ops. Every other function here already degrades rather than raising;
#: without this the module-scope call below would be the one place a database
#: problem could take the whole robot down, because brain.interface imports
#: this at import time and a failed CREATE TABLE would propagate out of it.
_available = True


def _init_cache() -> None:
    """Create the cache table/index if they don't exist yet. Safe to call repeatedly."""
    with db._write_lock, db._connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_cache (
                question_key TEXT NOT NULL,
                context_digest TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                emotion TEXT NOT NULL,
                hits INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP,
                last_used_at TIMESTAMP,
                PRIMARY KEY (question_key, context_digest)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_cache_last_used ON qa_cache(last_used_at)")


def lookup(question: str, extra_system: str | None = None) -> CachedAnswer | None:
    """Return the answer already given to this question, or None to ask the model."""
    if not _available:
        return None
    key = normalize_question(question)
    if not _is_cacheable_question(key):
        return None

    digest = _context_digest(extra_system)
    try:
        # Under the write lock despite being a read: the hit is recorded in the
        # same transaction, and that timestamp is what makes MAX_ENTRIES evict
        # the questions nobody asks rather than the ones asked first.
        with db._write_lock, db._connection() as conn:
            row = conn.execute(
                "SELECT answer, emotion FROM qa_cache WHERE question_key = ? AND context_digest = ?",
                (key, digest),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE qa_cache SET hits = hits + 1, last_used_at = ? "
                    "WHERE question_key = ? AND context_digest = ?",
                    (db._now(), key, digest),
                )
    except sqlite3.Error:
        logger.exception("Failed to read the answer cache for %r", question)
        return None

    if row is None:
        return None
    logger.info("Answering %r from cache", question)
    return CachedAnswer(row[0], row[1])


def remember(
    question: str,
    answer: str,
    emotion: str = "neutral",
    extra_system: str | None = None,
) -> None:
    """Store one answer, evicting the least recently used once past MAX_ENTRIES."""
    if not _available:
        return
    key = normalize_question(question)
    if not _is_cacheable_question(key) or not _is_cacheable_answer(answer):
        return

    digest = _context_digest(extra_system)
    now = db._now()
    try:
        with db._write_lock, db._connection() as conn:
            conn.execute(
                """
                INSERT INTO qa_cache (
                    question_key, context_digest, question, answer, emotion,
                    hits, created_at, last_used_at
                )
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(question_key, context_digest) DO UPDATE SET
                    question = excluded.question,
                    answer = excluded.answer,
                    emotion = excluded.emotion,
                    last_used_at = excluded.last_used_at
                """,
                (key, digest, question.strip(), answer.strip(), emotion, now, now),
            )
            conn.execute(
                """
                DELETE FROM qa_cache WHERE rowid NOT IN (
                    SELECT rowid FROM qa_cache ORDER BY last_used_at DESC, rowid DESC LIMIT ?
                )
                """,
                (MAX_ENTRIES,),
            )
    except sqlite3.Error:
        logger.exception("Failed to store a cached answer for %r", question)
        return
    logger.debug("Cached the answer to %r (context %s)", question, digest or "none")


def forget_all() -> int:
    """Empty the cache, returning how many answers were dropped. For the operator."""
    try:
        with db._write_lock, db._connection() as conn:
            return int(conn.execute("DELETE FROM qa_cache").rowcount)
    except sqlite3.Error:
        logger.exception("Failed to clear the answer cache")
        return 0


try:
    _init_cache()
except sqlite3.Error:
    _available = False
    logger.exception("Answer cache unavailable; every question will reach the model")
