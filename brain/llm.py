"""LLM wrapper. Which model actually answers is decided in brain/llm_backends.py.

Model choice still lives entirely in config.py (MODELS.ollama_model /
MODELS.cloud_model / MODELS.llm_backend) -- switching between the local model
and a cloud one, or to the documented Ollama fallback, never touches this file
or any caller.
"""

import logging
from typing import Iterator, Optional

from .llm_backends import FALLBACK, PREFERRED, Backend

logger = logging.getLogger(__name__)


def active_backend_name() -> str:
    """Which backend is answering, for the dashboard and the About demo."""
    return PREFERRED.name


def describe() -> dict[str, str]:
    """What is answering, for the dashboard.

    Reports the configured preference and the fallback rather than claiming to
    know which one produced the last reply. stream_reply switches to the local
    model per request when the cloud one fails, and keeps that in a local
    variable -- so anything here asserting "this model is answering" would be
    telling a room full of people the cloud is in use on the exact day the
    network died, which is the one day it matters.
    """
    from config import MODELS

    models = {
        "anthropic": MODELS.anthropic_model,
        "openai": MODELS.openai_model,
        "ollama": MODELS.ollama_model,
    }
    return {
        "backend": PREFERRED.name,
        "model": models.get(PREFERRED.name, PREFERRED.name),
        "fallback": "" if PREFERRED is FALLBACK else models.get(FALLBACK.name, FALLBACK.name),
        "local": "yes" if PREFERRED is FALLBACK else "no",
    }


def streaming_backends() -> tuple[Backend, ...]:
    """The backends to try, in order, for one streamed reply.

    Handed out rather than wrapped because the caller that streams
    (interface.stream_reply) is the only one that knows whether a complete
    sentence has reached the speaker yet, and that -- not whether a token
    arrived -- is what decides if switching models is still safe.
    """
    return (PREFERRED,) if PREFERRED is FALLBACK else (PREFERRED, FALLBACK)


def generate_response(
    messages: list[dict[str, str]], max_tokens: Optional[int] = None
) -> str:
    """Return raw model output. Expected to embed a trailing '[emotion: tag]' (see emotion.py).

    max_tokens raises the default cap for one call, exactly as stream_response
    does and for the same reason. The default is sized for a spoken reply of
    one or two sentences, which is far too small for anything structured: a
    caller asking for a JSON draft got the opening ```json fence and then
    silence, because the budget ran out three tokens in.
    """
    if PREFERRED is FALLBACK:
        return PREFERRED.generate(messages, max_tokens)
    try:
        return PREFERRED.generate(messages, max_tokens)
    except Exception as exc:
        logger.warning("%s failed (%s); answering with the local model", PREFERRED.name, exc)
        return FALLBACK.generate(messages, max_tokens)


def stream_response(
    messages: list[dict[str, str]], max_tokens: Optional[int] = None
) -> Iterator[str]:
    """Yield raw model output incrementally, piece by piece, as it's generated.

    max_tokens raises the default cap for a single call. A story needs it: the
    cap is sized for one- or two-sentence replies, and asked for a story the
    model reliably overruns whatever word limit it is given, so the reply was
    being cut off mid-sentence ("...and that night," then silence).

    A cloud failure falls back to the local model, but only while nothing has
    been said yet. Once the first piece is out the robot is already speaking,
    and starting a second, different answer over the top of it would have it
    talk over itself and repeat the opening -- so a stream that dies mid-reply
    ends the turn early instead. Half an answer is recoverable; two answers at
    once in front of a room is not.
    """
    if PREFERRED is FALLBACK:
        yield from FALLBACK.stream(messages, max_tokens)
        return

    spoken_something = False
    try:
        for piece in PREFERRED.stream(messages, max_tokens):
            spoken_something = True
            yield piece
        return
    except Exception as exc:
        if spoken_something:
            logger.warning("%s stream failed mid-reply: %s", PREFERRED.name, exc)
            return
        logger.warning("%s failed (%s); answering with the local model", PREFERRED.name, exc)

    yield from FALLBACK.stream(messages, max_tokens)
