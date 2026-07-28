"""Emotion-tag parsing: expects a leading '[tag]' the LLM is prompted to emit."""

import re

VALID_EMOTION_TAGS = frozenset({"happy", "sad", "curious", "surprised", "neutral"})

_TAG_RE = re.compile(r"^\[(\w+)\]\s*(.*)$", re.DOTALL)


def extract_emotion_tag(raw_output: str) -> tuple[str, str]:
    """Split raw LLM output into (reply_text, emotion_tag).

    Small local models don't always stick to the requested tag vocabulary, so
    anything outside VALID_EMOTION_TAGS is clamped to 'neutral' rather than
    passed through -- body/motion.py only knows how to express these five.
    """
    match = _TAG_RE.match(raw_output.strip())
    if not match:
        return raw_output.strip(), "neutral"

    reply_text, tag = match.group(2).strip(), match.group(1).lower()
    if tag not in VALID_EMOTION_TAGS:
        tag = "neutral"
    return reply_text, tag
