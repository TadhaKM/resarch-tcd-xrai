"""LLM wrapper: calls a locally-served Ollama model, configured via config.MODELS.

Model choice lives entirely in config.py (MODELS.ollama_model / ollama_host) --
switching to the documented fallback (OLLAMA_MODEL_FALLBACK) or a different
Ollama host never touches this file.
"""

from ollama import Client

from config import MODELS

_client = Client(host=MODELS.ollama_host)


def generate_response(messages: list[dict[str, str]]) -> str:
    """Return raw model output. Expected to embed a leading '[tag]' (see emotion.py)."""
    response = _client.chat(model=MODELS.ollama_model, messages=messages)
    return response["message"]["content"]
