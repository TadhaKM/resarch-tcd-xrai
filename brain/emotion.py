"""Emotion-tag parsing: expects a trailing '[emotion: tag]' the LLM is prompted to emit."""

import re

VALID_EMOTION_TAGS = frozenset({"happy", "sad", "curious", "thinking", "surprised", "neutral"})

# Tolerates one trailing punctuation mark after the bracket (e.g. "[emotion: happy].")
# -- small models often add closing punctuation as if the tag were a normal word.
_TAG_RE = re.compile(r"\[\s*emotion\s*:\s*(\w+)\s*\][.!?]?\s*$", re.IGNORECASE)


def extract_emotion_tag(raw_output: str) -> tuple[str, str]:
    """Split raw LLM output into (reply_text, emotion_tag).

    Expects the tag at the END of the reply, e.g. '...great to meet you! [emotion: happy]'.
    Small local models don't always comply -- a missing tag, or one outside
    VALID_EMOTION_TAGS, safely defaults to 'neutral' rather than failing the
    turn (body/motion.py only knows how to express these five anyway).
    """
    text = raw_output.strip()
    match = _TAG_RE.search(text)
    if not match:
        return text, "neutral"

    reply_text = text[: match.start()].strip()
    tag = match.group(1).lower()
    if tag not in VALID_EMOTION_TAGS:
        tag = "neutral"
    return reply_text, tag
