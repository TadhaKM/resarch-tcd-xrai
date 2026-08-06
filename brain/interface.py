"""The public entry points body/ calls into brain/."""

import re
from typing import Iterator

from . import long_term_memory, memory
from .emotion import extract_emotion_tag
from .llm import generate_response, stream_response
from .prompts import build_messages

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def get_reply(person_id: int, message: str) -> tuple[str, str]:
    """Return (reply_text, emotion_tag) for a message from the given person."""
    history = memory.get_history(person_id)
    context = long_term_memory.get_context(person_id)
    messages = build_messages(context, history, message)
    raw_output = generate_response(messages)
    reply_text, emotion_tag = extract_emotion_tag(raw_output)
    memory.remember_turn(person_id, message, reply_text)
    return reply_text, emotion_tag


def stream_reply(person_id: int, message: str) -> Iterator[tuple[str, str]]:
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
    """
    history = memory.get_history(person_id)
    context = long_term_memory.get_context(person_id)
    messages = build_messages(context, history, message)

    buffer = ""
    raw_parts: list[str] = []

    for piece in stream_response(messages):
        raw_parts.append(piece)
        buffer += piece
        match = _SENTENCE_BOUNDARY_RE.search(buffer)
        while match:
            candidate, _ = extract_emotion_tag(buffer[: match.end()])
            buffer = buffer[match.end() :]
            if candidate:
                yield candidate, "thinking"
            match = _SENTENCE_BOUNDARY_RE.search(buffer)

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
