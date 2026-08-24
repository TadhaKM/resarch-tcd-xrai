"""Answering a question about what the camera can see.

Somebody holds something up and asks "what is this?". Everything needed was
already here: the camera returns a frame, cv2 can encode it, and the model the
robot already answers with reads images.

THIS IS THE ONE PLACE A PICTURE LEAVES THE LAPTOP
Everything else the camera does is local. Face detection, recognition and the
identity ledger all run on this machine, and brain/study.py is built so a
transcript cannot be walked back to a person. Sending a photograph of a public
foyer to an API is a different act, so it is bounded here rather than left to
whoever writes the next demo:

  - Only on request. There is no path that looks continuously, and nothing
    calls this from a loop -- a visitor has to ask.
  - Announced. The demo says it is taking a look before it does, so nobody is
    photographed by a robot that appeared to be idle.
  - Never stored. The frame is encoded, sent, and dropped; nothing is written
    to disk and nothing goes in the database.
  - Downscaled hard. A 640px-wide JPEG is plenty to name an object and is
    markedly less useful for identifying the people standing behind it.

NO LOCAL FALLBACK, AND IT SAYS SO
The local model has no vision at all, so this is the one feature that genuinely
cannot work when the wifi is down -- which on this network is a routine
condition rather than an edge case. It returns a plain "I cannot see just now"
rather than failing in a way that needs explaining to a group.
"""

import base64
import logging
from typing import Optional

logger = logging.getLogger(__name__)

#: The longest edge of the picture actually sent. Enough for "a Rubik's cube"
#: or "a Trinity prospectus", and deliberately too coarse to be a good
#: photograph of the room. Also keeps a turn to roughly a thousand tokens.
_MAX_EDGE = 640

#: JPEG quality. Below about 70 the model starts guessing at text on packaging,
#: which is exactly the sort of confident wrongness this must not add.
_JPEG_QUALITY = 80

#: Short, because it is spoken aloud to somebody holding something up. The
#: model's instinct is a paragraph, and a paragraph is the wrong answer to
#: "what is this?".
_PROMPT = (
    "You are a small robot in a university foyer and somebody is holding "
    "something up to your camera. Say what it is in ONE short sentence, as you "
    "would to a person standing in front of you. If you cannot tell, say so "
    "plainly rather than guessing -- a confident wrong answer to somebody "
    "holding their own possession is worse than an honest 'I am not sure'. "
    "Do not describe the room, the lighting or the people; answer about the "
    "thing being shown. No preamble."
)

#: The most that comes back. One sentence, plus room for the model to finish it.
_MAX_TOKENS = 120


def available() -> bool:
    """Whether looking is possible at all -- a cloud key and cv2."""
    try:
        import cv2  # noqa: F401

        from config import MODELS

        return bool(MODELS.api_key(MODELS.anthropic_key_env))
    except Exception:
        return False


def _encode(frame) -> Optional[str]:
    """A camera frame as base64 JPEG, downscaled. None if it cannot be done."""
    try:
        import cv2

        height, width = frame.shape[:2]
        if not width or not height:
            return None
        longest = max(width, height)
        if longest > _MAX_EDGE:
            scale = _MAX_EDGE / float(longest)
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        logger.exception("Could not encode the camera frame")
        return None


def describe(frame, question: str = "") -> str:
    """What the camera is looking at, in one sentence. "" if it cannot say.

    Returns "" rather than raising for every failure -- no key, no internet, a
    dead camera, a refused request. The caller is a demo speaking to a visitor,
    and every one of those means the same thing out loud.
    """
    if frame is None:
        return ""
    encoded = _encode(frame)
    if not encoded:
        return ""

    try:
        import httpx
        from anthropic import Anthropic

        from config import MODELS

        key = MODELS.api_key(MODELS.anthropic_key_env)
        if not key:
            return ""
        client = Anthropic(api_key=key, timeout=httpx.Timeout(20.0, connect=5.0),
                           max_retries=1)
        asked = (question or "").strip()
        # The visitor's own words are passed through, so "what colour is it"
        # and "is this a first edition" both work, but the instructions above
        # still bound the shape of the answer.
        text = f"{_PROMPT}\n\nWhat they asked: {asked}" if asked else _PROMPT
        response = client.messages.create(
            model=MODELS.anthropic_model,
            max_tokens=_MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg",
                                                 "data": encoded}},
                    {"type": "text", "text": text},
                ],
            }],
        )
        answer = "".join(b.text for b in response.content if b.type == "text").strip()
    except Exception as exc:
        # Logged at warning rather than exception: the overwhelmingly common
        # cause is the wifi being down, which is not a defect worth a traceback
        # in a log somebody is reading to find a real one.
        logger.warning("Could not look at what the camera sees: %s", exc)
        return ""
    return answer
