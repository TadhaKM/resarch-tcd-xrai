"""The public entry points body/ calls into brain/."""

import re
from typing import Iterator, Optional

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

    Every sentence is tagged "thinking" except the last: the real tag is a
    trailing "[emotion: tag]" (see emotion.py), which only becomes known once
    the whole reply has arrived. The most recently completed sentence is
    always held back by one step, so it's never spoken until we've seen
    something follow it -- otherwise the final sentence and its trailing tag
    would be indistinguishable from "a sentence, mid-stream" the moment it
    completes.
    """
    history = memory.get_history(person_id)
    context = long_term_memory.get_context(person_id)
    messages = build_messages(context, history, message)

    buffer = ""
    pending: Optional[str] = None
    spoken_chars = 0
    raw_parts: list[str] = []

    for piece in stream_response(messages):
        raw_parts.append(piece)
        buffer += piece
        match = _SENTENCE_BOUNDARY_RE.search(buffer)
        while match:
            candidate = buffer[: match.end()].strip()
            buffer = buffer[match.end() :]
            if pending is not None:
                yield pending, "thinking"
                spoken_chars += len(pending) + 1
            pending = candidate
            match = _SENTENCE_BOUNDARY_RE.search(buffer)

    raw_output = "".join(raw_parts)
    reply_text, emotion_tag = extract_emotion_tag(raw_output)
    remaining = reply_text[spoken_chars:].strip()
    if remaining:
        yield remaining, emotion_tag

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
