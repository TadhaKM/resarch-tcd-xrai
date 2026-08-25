"""Cross-session memory: summarize each finished conversation into one note,
and consolidate a person's notes into a single profile paragraph once there
are too many. Backed by brain/db.py (SQLite); LLM calls go through brain/llm.py.

brain/memory.py, by contrast, is the *in-session* turn buffer used to keep a
single conversation coherent turn-to-turn -- it's cleared once the
conversation ends and summarized here. This module is what carries facts
about a person *across* conversations.
"""

import logging

from . import db
from .llm import generate_response

logger = logging.getLogger(__name__)

CONSOLIDATION_THRESHOLD = 15

_SUMMARY_PROMPT = (
    "Summarize this conversation in 2-3 sentences, focused on facts about the "
    "person worth remembering next time (mood, topics they care about, "
    "anything they said they'd follow up on). Conversation:\n{transcript}"
)

_CONSOLIDATION_PROMPT = (
    "Merge these separate notes about the same person into a single condensed "
    "profile paragraph. Keep the facts worth remembering and drop anything "
    "redundant or stale.\n\nNotes:\n{notes}"
)


def get_context(person_id: int) -> str:
    """Return what's remembered about this person (recent notes, or the
    consolidated profile if they've been merged) as one string for the
    system prompt. Empty string if there's nothing yet."""
    if not person_id:
        # See end_conversation: person 0 is a shared id, and whatever notes it
        # accumulated before that gate existed are strangers' summaries that
        # must not colour -- or slow -- other strangers' answers.
        return ""
    notes = db.get_notes(person_id)
    return "\n".join(f"- {note}" for note in notes)


def _format_transcript(history: list[tuple[str, str]]) -> str:
    return "\n".join(f"User: {message}\nReachy: {reply}" for message, reply in history)


def end_conversation(person_id: int, history: list[tuple[str, str]]) -> None:
    """Summarize a finished conversation into one memory note, and consolidate
    if this person now has more than CONSOLIDATION_THRESHOLD notes.

    A summarization failure (LLM unreachable, etc.) is logged and skipped
    rather than raised -- losing one session's memory shouldn't take the
    app down.
    """
    if not history:
        return

    # Person 0 is EVERY unrecognised visitor sharing one id -- the open-day
    # default, not a person. Summarising their sessions wrote strangers'
    # conversations into one shared profile (12 notes deep when found), which
    # was then injected into every later stranger's prompt: a privacy smell,
    # ~930 tokens of irrelevant context per turn, and the single reason the
    # qa_cache never hit once -- its store gate requires "no context", and
    # person 0 always had context. Named, recognised people keep their memory
    # exactly as before.
    if not person_id:
        return

    transcript = _format_transcript(history)
    try:
        summary = generate_response(
            [{"role": "user", "content": _SUMMARY_PROMPT.format(transcript=transcript)}]
        ).strip()
    except Exception:
        logger.exception("Failed to summarize conversation for person_id=%s", person_id)
        return

    if not summary:
        return

    db.add_note(person_id, summary)

    if db.count_notes(person_id) > CONSOLIDATION_THRESHOLD:
        _consolidate(person_id)


def _consolidate(person_id: int) -> None:
    """Merge all of a person's notes into one profile paragraph, replacing them."""
    notes = db.get_notes(person_id)
    if len(notes) <= 1:
        return

    joined = "\n".join(f"- {note}" for note in notes)
    try:
        profile = generate_response(
            [{"role": "user", "content": _CONSOLIDATION_PROMPT.format(notes=joined)}]
        ).strip()
    except Exception:
        logger.exception("Failed to consolidate notes for person_id=%s", person_id)
        return

    if not profile:
        return

    db.replace_notes_with_profile(person_id, profile)
