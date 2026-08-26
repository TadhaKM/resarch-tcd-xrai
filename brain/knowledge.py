"""Answers the Hub has taught the robot, so it gets smarter visit by visit.

The learning loop this closes: every question the robot deflects is already
counted (brain/stats.py, the dashboard's "questions it could not answer"
list). Until now the only way that list improved the robot was a developer
editing Python. Now a staff member reads the list, types the right answer
once, and the robot knows it forever -- spoken instantly when the question
comes back, and handed to the language model as grounding when someone asks a
paraphrase of it.

Two ways a learned answer is used, mirroring how the Hub's own script works:

- SPOKEN VERBATIM (speak_if_taught): a short, direct match on the taught
  question or one of its phrasings. Instant -- no model round-trip -- because
  the answer is already written.
- RETRIEVED (brief): looser word-overlap match, bounded to two entries,
  appended to the model's per-turn system prompt exactly the way
  brain/courses.py briefs it on programmes. A paraphrase the verbatim tier
  will not touch still gets answered WITH the taught facts in front of the
  model.

Approval is the safety line. The suggest flow lets the cloud model DRAFT
answers for the unanswered list, but a draft is stored approved=0 and is
never spoken and never briefed until a person approves it -- this robot's
standing rule is that a wrong claim about the Hub is worse than a vague one,
and models draft plausible wrong claims fluently. approve() re-validates, so
a draft with "[bracketed placeholders]" must be edited by a human first.

Storage follows brain/features.py to the letter: db's process-wide write
lock and short-lived connections, an _available flag so a failed CREATE
TABLE degrades to "no learned answers" instead of taking the robot down,
validate-on-write / coerce-on-read, and db.backup_db() covers the table for
free because it lives in the same file.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

from . import db

logger = logging.getLogger(__name__)

_available = True

MAX_ANSWERS = 200
MAX_QUESTION_CHARS = 160
MAX_ANSWER_CHARS = 700
MAX_CUES = 8
MAX_CUE_CHARS = 80
#: A cue shorter than this fires on fragments of unrelated speech -- the
#: same lesson features.py records as "a bare 'welcome' fired on 'you're
#: welcome'".
MIN_CUE_WORDS = 2
MIN_CUE_CHARS = 8

#: How far past a matched phrasing an utterance may run and still be spoken
#: verbatim. The same allowance the Hub script uses, for the same reason.
_MAX_EXTRA_WORDS = 10

#: How many entries brief() will put in front of the model per turn, and the
#: minimum content-word overlap to qualify. Two, like courses.MAX_DETAIL:
#: more grounding than the reply is allowed to be is waste.
_BRIEF_LIMIT = 2
_BRIEF_MIN_OVERLAP = 2

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")

#: Words too common to count as topical overlap in brief().
_STOPWORDS = frozenset(
    "a an the is are was were be been do does did what whats who where when "
    "why how which i you we they it this that to of in on at for with and or "
    "not can could will would there here have has my your our me us "
    "about tell say said".split()
)


@dataclass(frozen=True)
class Learned:
    id: int
    question: str
    cues: tuple[str, ...]
    answer: str
    source: str
    approved: bool


def _row_to_learned(row) -> Learned:
    try:
        cues = tuple(str(c) for c in json.loads(row[2] or "[]") if str(c).strip())
    except (ValueError, TypeError):
        cues = ()
    return Learned(
        id=int(row[0]),
        question=_CONTROL_RE.sub(" ", str(row[1] or "")).strip()[:MAX_QUESTION_CHARS],
        cues=cues[:MAX_CUES],
        answer=_CONTROL_RE.sub(" ", str(row[3] or "")).strip()[:MAX_ANSWER_CHARS],
        source=str(row[4] or "operator"),
        approved=bool(row[5]),
    )


def list_answers(only_approved: bool = False) -> list[Learned]:
    """Every learned answer, newest first. [] when the store is down."""
    if not _available:
        return []
    try:
        with db._connection() as conn:
            rows = conn.execute(
                "SELECT id, question, cues, answer, source, approved "
                "FROM learned_answers "
                + ("WHERE approved = 1 " if only_approved else "")
                + "ORDER BY id DESC"
            ).fetchall()
        return [_row_to_learned(r) for r in rows]
    except sqlite3.Error:
        logger.exception("Could not list learned answers")
        return []


def get(answer_id: int) -> Optional[Learned]:
    if not _available:
        return None
    try:
        with db._connection() as conn:
            row = conn.execute(
                "SELECT id, question, cues, answer, source, approved "
                "FROM learned_answers WHERE id = ?",
                (answer_id,),
            ).fetchone()
        return _row_to_learned(row) if row else None
    except sqlite3.Error:
        logger.exception("Could not read learned answer %s", answer_id)
        return None


def _normalised_cues(question: str, cues: list[str]) -> list[str]:
    """The phrasings this entry answers, word-stream normalised, deduplicated.

    The taught question itself is always a cue -- an operator should not have
    to type it twice.
    """
    from demokit.runner import _word_stream

    out: list[str] = []
    for raw in [question, *cues]:
        stream = _word_stream(_CONTROL_RE.sub(" ", str(raw or ""))).strip()
        if stream and stream not in out:
            out.append(stream[:MAX_CUE_CHARS])
    return out[: MAX_CUES + 1]


def validate(question: str, answer: str, cues: list[str]) -> list[str]:
    """Problems with this entry, in words an operator can act on. [] is fine."""
    from demokit.registry import REGISTRY
    from demokit.runner import SLEEP_PHRASES, contains_phrase

    problems: list[str] = []
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question:
        problems.append("The question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        problems.append(f"The question is too long (over {MAX_QUESTION_CHARS} characters).")
    if not answer:
        problems.append("The answer is empty.")
    if len(answer) > MAX_ANSWER_CHARS:
        problems.append(
            f"The answer is too long to speak (over {MAX_ANSWER_CHARS} characters "
            "-- that is more than half a minute of talking)."
        )
    for text in (question, answer, *cues):
        if _CONTROL_RE.search(text or ""):
            problems.append("Something here contains characters the robot cannot speak.")
            break
    if _PLACEHOLDER_RE.search(answer):
        problems.append(
            "The answer still has [bracketed placeholders] -- fill them in "
            "with the real detail, or delete that part."
        )
    if len(cues) > MAX_CUES:
        problems.append(f"Too many extra phrasings (keep it to {MAX_CUES}).")

    for stream in _normalised_cues(question, cues):
        if len(stream.split()) < MIN_CUE_WORDS or len(stream) < MIN_CUE_CHARS:
            problems.append(
                f'"{stream}" is too short to match safely -- one word fires '
                "on fragments of unrelated conversation."
            )
            continue
        padded = f" {stream} "
        if any(contains_phrase(padded, p) for p in SLEEP_PHRASES):
            problems.append(f'"{stream}" contains a go-to-sleep phrase, which always wins.')
        if "reachy" in stream.split():
            problems.append(
                f'"{stream}" contains the robot\'s name, which is stripped '
                "before matching -- leave it out."
            )
        try:
            REGISTRY.ensure_discovered()
            for demo_id in REGISTRY.ids():
                demo = REGISTRY.get(demo_id)
                for trigger in getattr(demo, "triggers", ()):
                    if contains_phrase(padded, trigger):
                        problems.append(
                            f'"{stream}" contains "{trigger}", which switches '
                            f"to the {demo_id} demo before any answer is looked up."
                        )
        except Exception:  # pragma: no cover - a broken registry must not block teaching
            logger.debug("Could not audit cues against demo triggers", exc_info=True)
    return problems


def save(
    question: str,
    answer: str,
    cues: Optional[list[str]] = None,
    *,
    answer_id: Optional[int] = None,
    source: str = "operator",
    approved: bool = True,
) -> tuple[bool, list[str], Optional[int]]:
    """Create or update one learned answer. (ok, problems, id)."""
    if not _available:
        return False, ["The robot's database is unavailable, so nothing can be saved."], None
    cues = [c for c in (cues or []) if (c or "").strip()]
    problems = validate(question, answer, cues)
    if problems:
        return False, problems, None
    stored_cues = json.dumps(_normalised_cues(question, cues))
    try:
        with db._write_lock, db._connection() as conn:
            if answer_id is None:
                count = conn.execute("SELECT COUNT(*) FROM learned_answers").fetchone()[0]
                if count >= MAX_ANSWERS:
                    return False, [
                        f"The robot already knows {MAX_ANSWERS} taught answers -- "
                        "delete some before adding more."
                    ], None
                cursor = conn.execute(
                    "INSERT INTO learned_answers "
                    "(question, cues, answer, source, approved, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (question.strip(), stored_cues, answer.strip(), source,
                     int(approved), db._now(), db._now()),
                )
                return True, [], int(cursor.lastrowid)
            changed = conn.execute(
                "UPDATE learned_answers SET question = ?, cues = ?, answer = ?, "
                "updated_at = ? WHERE id = ?",
                (question.strip(), stored_cues, answer.strip(), db._now(), answer_id),
            ).rowcount
            if not changed:
                return False, ["That answer no longer exists."], None
            return True, [], answer_id
    except sqlite3.Error:
        logger.exception("Failed to save learned answer")
        return False, ["The robot could not save that. Try again."], None


def approve(answer_id: int, approved: bool = True) -> tuple[bool, list[str]]:
    """Turn an entry on or off. Approving RE-VALIDATES: a machine-drafted
    answer with placeholders must be edited by a person before it is ever
    spoken."""
    if not _available:
        return False, ["The robot's database is unavailable."]
    entry = get(answer_id)
    if entry is None:
        return False, ["That answer no longer exists."]
    if approved:
        problems = validate(entry.question, entry.answer, list(entry.cues))
        if problems:
            return False, problems
    try:
        with db._write_lock, db._connection() as conn:
            conn.execute(
                "UPDATE learned_answers SET approved = ?, updated_at = ? WHERE id = ?",
                (int(approved), db._now(), answer_id),
            )
        return True, []
    except sqlite3.Error:
        logger.exception("Failed to set approval on learned answer %s", answer_id)
        return False, ["The robot could not save that. Try again."]


def delete(answer_id: int) -> bool:
    if not _available:
        return False
    try:
        with db._write_lock, db._connection() as conn:
            return conn.execute(
                "DELETE FROM learned_answers WHERE id = ?", (answer_id,)
            ).rowcount > 0
    except sqlite3.Error:
        logger.exception("Failed to delete learned answer %s", answer_id)
        return False


# --- the two ways a learned answer reaches a visitor ---------------------


def match(text: str) -> Optional[Learned]:
    """The approved entry this utterance is asking for, or None.

    Same discipline as the Hub script's matcher: a taught phrasing must
    appear in the utterance, and the utterance must not run more than
    _MAX_EXTRA_WORDS past it -- a rich multi-part question goes to the
    model, WITH brief() putting the taught facts in front of it.
    """
    entries = list_answers(only_approved=True)
    if not entries:
        return None
    from demokit.runner import _word_stream, contains_phrase
    from demos._set_pieces import normalise

    words = normalise(_word_stream(text))
    total = len(words.split())
    for entry in entries:
        for cue in entry.cues:
            if (contains_phrase(words, cue)
                    and total <= len(cue.split()) + _MAX_EXTRA_WORDS):
                return entry
    return None


def speak_if_taught(ctx, text: str) -> bool:
    """Speak the taught answer if this utterance asks a taught question.

    True when one was delivered -- instantly, through the same pipelined
    script path the Hub's own blocks use, and recorded in conversation
    memory so a follow-up reaches the model with this answer in hand.
    """
    entry = match(text)
    if entry is None:
        return False
    from demokit.base import split_sentences

    ctx.status(f"Answered from what the Hub taught me: {entry.question[:60]}")
    ctx.say_script([(line, "neutral") for line in split_sentences(entry.answer)])
    try:
        from brain import memory

        memory.remember_turn(ctx.person_id(), text, entry.answer)
    except Exception:
        pass
    return True


def brief(message: str) -> str:
    """Taught answers relevant to this turn, for the model's system prompt.

    Word-overlap retrieval in the courses.brief mould: "" is the common case
    and costs nothing. This is what makes a PARAPHRASE of a taught question
    come back right -- the verbatim tier refuses it, the model answers it,
    and the model has the taught facts in front of it when it does.
    """
    entries = list_answers(only_approved=True)
    if not entries:
        return ""
    from demokit.runner import _word_stream

    asked = {
        w for w in _word_stream(message).split()
        if w not in _STOPWORDS and len(w) > 2
    }
    if not asked:
        return ""
    scored: list[tuple[int, Learned]] = []
    for entry in entries:
        topic = {
            w for cue in (entry.question, *entry.cues)
            for w in _word_stream(cue).split()
            if w not in _STOPWORDS and len(w) > 2
        }
        overlap = len(asked & topic)
        if overlap >= _BRIEF_MIN_OVERLAP:
            scored.append((overlap, entry))
    if not scored:
        return ""
    scored.sort(key=lambda pair: -pair[0])
    picked = [entry for _score, entry in scored[:_BRIEF_LIMIT]]
    lines = ["The Hub has taught you these exact answers -- prefer them over "
             "your own phrasing when they fit what was asked:"]
    for entry in picked:
        lines.append(f"Q: {entry.question}\nA: {entry.answer}")
    return "\n".join(lines)


def _init_knowledge() -> None:
    with db._write_lock, db._connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_answers (
                id         INTEGER PRIMARY KEY,
                question   TEXT NOT NULL,
                cues       TEXT NOT NULL DEFAULT '[]',
                answer     TEXT NOT NULL,
                source     TEXT NOT NULL DEFAULT 'operator',
                approved   INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
            """
        )


try:
    _init_knowledge()
except sqlite3.Error:
    _available = False
    logger.exception("Learned answers unavailable; the robot cannot be taught new ones")
