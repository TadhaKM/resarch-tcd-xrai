"""LLM wrapper: calls a locally-served Ollama model, configured via config.MODELS.

Model choice lives entirely in config.py (MODELS.ollama_model / ollama_host) --
switching to the documented fallback (OLLAMA_MODEL_FALLBACK) or a different
Ollama host never touches this file.
"""

from typing import Iterator

from ollama import Client

from config import MODELS

_client = Client(host=MODELS.ollama_host)
_OPTIONS = {"num_predict": MODELS.llm_max_tokens}
# Ollama unloads an idle model after 5 minutes by default; a cold reload on
# this hardware costs ~30+ seconds on top of generation time (measured: 53s
# cold vs 21s warm for a similar reply). -1 keeps it resident indefinitely --
# worth it since this process's whole job is serving this one model.
_KEEP_ALIVE = -1


def generate_response(messages: list[dict[str, str]]) -> str:
    """Return raw model output. Expected to embed a trailing '[emotion: tag]' (see emotion.py)."""
    response = _client.chat(
        model=MODELS.ollama_model, messages=messages, options=_OPTIONS, keep_alive=_KEEP_ALIVE
    )
    return response["message"]["content"]


def stream_response(
    messages: list[dict[str, str]], max_tokens: int | None = None
) -> Iterator[str]:
    """Yield raw model output incrementally, piece by piece, as it's generated.

    max_tokens raises the default cap for a single call. A story needs it: the
    cap is sized for one- or two-sentence replies, and asked for a story the
    model reliably overruns whatever word limit it is given, so the reply was
    being cut off mid-sentence ("...and that night," then silence).
    """
    options = _OPTIONS if max_tokens is None else {"num_predict": max_tokens}
    for chunk in _client.chat(
        model=MODELS.ollama_model, messages=messages, options=options, stream=True, keep_alive=_KEEP_ALIVE
    ):
        piece = chunk["message"]["content"]
        if piece:
            yield piece
