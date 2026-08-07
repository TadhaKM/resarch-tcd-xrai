"""The public entry points body/ calls into brain/."""

import logging
import re
from typing import Iterator

from . import llm, long_term_memory, memory
from .emotion import extract_emotion_tag
from .llm import generate_response
from .prompts import build_messages

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

#: Token budget for a story. The word limit in the storyteller prompt is a
#: nudge, not a control -- asked for eighty words it writes closer to two
#: hundred whatever the phrasing -- so this is what actually bounds the
#: length, sized above what it really produces plus the emotion tag. Erring
#: high on purpose: a story that stops mid-sentence is worse than a long one.
_STORY_MAX_TOKENS = 280


def get_reply(person_id: int, message: str) -> tuple[str, str]:
    """Return (reply_text, emotion_tag) for a message from the given person."""
    history = memory.get_history(person_id)
    context = long_term_memory.get_context(person_id)
    messages = build_messages(context, history, message)
    raw_output = generate_response(messages)
    reply_text, emotion_tag = extract_emotion_tag(raw_output)
    memory.remember_turn(person_id, message, reply_text)
    return reply_text, emotion_tag


def stream_reply(
    person_id: int, message: str, style: str | None = None
) -> Iterator[tuple[str, str]]:
    """Yield (sentence_text, emotion_tag) pairs as the reply streams in, so the
    caller can start speaking sentence 1 while the model is still generating
    sentence 2+ -- the point is reducing time-to-first-words on hardware too
    slow for full-reply latency to feel conversational (see llm_max_tokens'
    docstring in config.py).

    Each sentence is yielded as soon as it is complete. An earlier version held
    the newest one back until the one after it arrived, so that the last
    sentence could be paired with the reply's trailing "[emotion: tag]". That
    cost a whole sentence of generation before the robot said anything --
    measured live at 3.8s to first word against 0.36s to first token -- and it
    read as the robot pausing before every answer. The tag is now carried by a
    final pair instead, whose text is usually empty (the tag is all that is
    left after the last sentence boundary); callers use it for the closing
    gesture and skip speaking when there is nothing to say.

    Sentences are tagged "thinking" because the real one isn't known until the
    reply ends. Every yielded sentence is passed through extract_emotion_tag,
    so a tag the model puts mid-reply is stripped rather than spoken.

    Failover between models lives here rather than in brain/llm.py because the
    decision needs a fact only this function has: whether anything has actually
    been SPOKEN. llm.py can only see tokens, and tokens are not speech -- a
    cloud model that dies after three tokens has produced nothing the caller
    ever passed to the speaker, so switching to the local model is free and
    invisible. Deciding at the token level instead meant that failure was
    treated as "already talking", and the visitor got the fragment the dead
    model managed ("The answer is probably") instead of a real answer. Once a
    complete sentence has gone out, though, the robot is genuinely mid-utterance
    and a second model must not start a different answer over the top of it.
    """
    history = memory.get_history(person_id)
    context = long_term_memory.get_context(person_id)
    messages = build_messages(context, history, message, style=style)

    max_tokens = _STORY_MAX_TOKENS if style == "story" else None
    backends = llm.streaming_backends()

    buffer = ""
    raw_parts: list[str] = []
    spoken_a_sentence = False

    for attempt, backend in enumerate(backends):
        # Reset per attempt: a failed backend's partial words must not be
        # glued onto the replacement's answer.
        buffer = ""
        raw_parts = []
        try:
            for piece in backend.stream(messages, max_tokens=max_tokens):
                raw_parts.append(piece)
                buffer += piece
                match = _SENTENCE_BOUNDARY_RE.search(buffer)
                while match:
                    candidate, _ = extract_emotion_tag(buffer[: match.end()])
                    buffer = buffer[match.end() :]
                    if candidate:
                        spoken_a_sentence = True
                        yield candidate, "thinking"
                    match = _SENTENCE_BOUNDARY_RE.search(buffer)
            break
        except Exception as exc:
            last_attempt = attempt == len(backends) - 1
            if spoken_a_sentence or last_attempt:
                logger.warning("%s stream failed: %s", backend.name, exc)
                break
            logger.warning(
                "%s stream failed before speaking (%s); retrying with %s",
                backend.name,
                exc,
                backends[attempt + 1].name,
            )

    raw_output = "".join(raw_parts)
    reply_text, emotion_tag = extract_emotion_tag(raw_output)
    tail, _ = extract_emotion_tag(buffer)
    yield tail, emotion_tag

    memory.remember_turn(person_id, message, reply_text)


def end_conversation(person_id: int) -> None:
    """Summarize this session's conversation into long-term memory, then clear it.

    Call this once a conversation with `person_id` is over (e.g. after a
    period of silence, or when the app is shutting down) -- not after every
    turn.
    """
    history = memory.get_history(person_id)
    long_term_memory.end_conversation(person_id, history)
    memory.clear_history(person_id)
