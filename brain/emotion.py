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

# Stage directions the model invents despite being told not to -- "[Pause]",
# "[Narrator]" -- which are not part of what should be said out loud; the cost
# of one reaching the speaker is the robot solemnly announcing "Pause"
# mid-story. Deliberately only a single bracketed word: real bracketed prose
# runs to several ("He said [see below] to me.") and must survive, which is
# what stops this from swallowing anything the model actually meant to say.
_STAGE_DIRECTION_RE = re.compile(r"\[\s*[a-z]+\s*\]", re.IGNORECASE)


#: Parenthetical asides and asterisk actions. Everything the model produces is
#: spoken immediately, and the synthesiser reads brackets as words: a reply
#: ending "(If asked directly about which headsets are used, Professor Laura
#: Berry might help)" was delivered to a visitor complete with "if asked
#: directly". The prompt forbids these, but a 1.5B model forgets, and the cost
#: of forgetting is paid out loud in front of people -- so they are stripped
#: here as well. Speech has no parentheses; nothing is lost by removing them.
_ASIDE_RE = re.compile(r"\([^)]*\)|\*[^*]+\*")

#: Meta-narration the model puts before an answer when it is restating its own
#: instructions -- "First: Start a brainstorming session about..." -- which is
#: the model talking to itself, out loud, at a visitor.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:first|next|then|step\s*\d+|let me start by|to begin|okay so)\s*[:,-]\s*",
    re.IGNORECASE,
)


def clean_for_speech(text: str) -> str:
    """Strip anything the model wrote that was never meant to be said aloud."""
    text = _ASIDE_RE.sub(" ", text)
    text = _PREAMBLE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Punctuation stranded by a removed aside ("Berry .", "help , if").
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text.strip()


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
    text = _STAGE_DIRECTION_RE.sub(" ", text)
    # Parenthetical asides, asterisk actions and "First:"-style preambles go
    # the same way as the tags: everything here is about to be spoken.
    text = clean_for_speech(text)

    # Collapse whitespace opened up by removing tags, and drop punctuation left
    # stranded before one (". ." from "sentence. [tag] .").
    reply_text = re.sub(r"\s+", " ", text).strip()
    reply_text = re.sub(r"\s+([.,!?])", r"\1", reply_text).strip()

    if tag not in VALID_EMOTION_TAGS:
        tag = "neutral"
    return reply_text, tag
