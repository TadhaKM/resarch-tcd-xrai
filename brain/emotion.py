"""Emotion-tag parsing: expects a trailing '[emotion: tag]' the LLM is prompted to emit."""

import re

VALID_EMOTION_TAGS = frozenset({"happy", "sad", "curious", "thinking", "surprised", "neutral"})

# The tag body, matched in several forms because small models drift from the
# prompt: the requested "emotion: happy", a bare "happy", and -- seen live from
# qwen2.5 on nearly every turn -- some other word in the key position
# ("[thinking: happy]", "[neutral: curious]"). Any key is accepted and the tag
# read from the value after the colon, so a drifting key is ignored rather than
# leaving the whole bracket unmatched: unmatched meant it was never stripped
# and the robot read "[thinking: happy]" out loud. Restricted to the known tags
# so ordinary bracketed prose is never mistaken for one.
_TAG_BODY = r"(?:[a-z_]+\s*:\s*)?\s*(" + "|".join(sorted(VALID_EMOTION_TAGS)) + r")"

# Tolerates one trailing punctuation mark after the bracket (e.g. "[emotion: happy].")
# -- small models often add closing punctuation as if the tag were a normal word.
_TRAILING_TAG_RE = re.compile(r"\[\s*" + _TAG_BODY + r"\s*\][.!?]?\s*$", re.IGNORECASE)

# Same tag anywhere in the text. Models emit these mid-reply as well as at the
# end, and anything left in the string is spoken aloud -- the robot was heard
# saying "How can I help today? [thinking]" verbatim.
_ANY_TAG_RE = re.compile(r"\[\s*" + _TAG_BODY + r"\s*\]", re.IGNORECASE)


def extract_emotion_tag(raw_output: str) -> tuple[str, str]:
    """Split raw LLM output into (reply_text, emotion_tag).

    Prefers a tag at the END of the reply ('...great to meet you! [emotion:
    happy]'), which is what the prompt asks for. Small local models don't
    always comply: a missing tag, or one outside VALID_EMOTION_TAGS, safely
    defaults to 'neutral' rather than failing the turn (body/motion.py only
    knows how to express these five anyway).

    Any tag left elsewhere in the text is stripped rather than spoken. When
    there is no trailing tag but a stray one exists, the last stray is used as
    the emotion -- the model clearly intended it, just misplaced it.
    """
    text = raw_output.strip()

    tag = None
    match = _TRAILING_TAG_RE.search(text)
    if match:
        tag = match.group(1).lower()
        text = text[: match.start()]

    strays = _ANY_TAG_RE.findall(text)
    if strays and tag is None:
        tag = strays[-1].lower()
    text = _ANY_TAG_RE.sub(" ", text)

    # Collapse whitespace opened up by removing tags, and drop punctuation left
    # stranded before one (". ." from "sentence. [tag] .").
    reply_text = re.sub(r"\s+", " ", text).strip()
    reply_text = re.sub(r"\s+([.,!?])", r"\1", reply_text).strip()

    if tag not in VALID_EMOTION_TAGS:
        tag = "neutral"
    return reply_text, tag
