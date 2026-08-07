"""Where the words come from: a local Ollama model, or a cloud one when configured.

Both paths have to hold at once. The robot's original value was that it works
with no internet at all -- STT, TTS and the model all local -- and that is what
makes it demonstrable in a room with bad wifi. But a cloud model answers a
visitor's question about AI far better than a 1.5B model on a laptop, and the
Hub has API credits. So: use the cloud when it is configured and reachable,
fall back to local when it is not, and never let a network failure end a turn
in front of an audience.

Adding a provider means adding a Backend subclass here and a branch in
`_build`. Nothing outside this file and config.py needs to change: brain/llm.py
exposes the same two functions it always did.

The message format used throughout is the OpenAI/Ollama shape that
prompts.build_messages already produces -- a list of {"role", "content"} with
the system prompt as the first entry. Providers that want it differently (as
Anthropic does, taking `system` as its own parameter) adapt internally, so
callers never have to know which backend is live.
"""

import logging
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from ollama import Client

from config import MODELS

logger = logging.getLogger(__name__)

Messages = list[dict[str, str]]

#: Ceiling on a single model call. Both vendor SDKs default to minutes with
#: retries (Anthropic: 10 minutes, 2 retries), which on venue wifi is a robot
#: standing silent in its thinking pose for longer than anyone will wait, with
#: the mode switch dead behind it. A spoken turn that takes 20 seconds has
#: already failed, so fail it fast enough to fall back to the local model and
#: still answer.
_REQUEST_TIMEOUT_S = 20.0
_CONNECT_TIMEOUT_S = 5.0
#: One retry, not the SDK default of two: retries are serial, so the effective
#: wait is a multiple of the timeout above.
_MAX_RETRIES = 1


class Backend(ABC):
    """One source of model output."""

    name: str

    @abstractmethod
    def generate(self, messages: Messages, max_tokens: Optional[int] = None) -> str:
        """Return the whole reply."""

    @abstractmethod
    def stream(self, messages: Messages, max_tokens: Optional[int] = None) -> Iterator[str]:
        """Yield the reply in pieces as it is produced."""


class OllamaBackend(Backend):
    """The local model. Always the fallback, so it must never need the network."""

    name = "ollama"

    def __init__(self) -> None:
        self._client = Client(host=MODELS.ollama_host, timeout=_REQUEST_TIMEOUT_S)
        self._default_options = {"num_predict": MODELS.llm_max_tokens}
        # Ollama unloads an idle model after 5 minutes by default; a cold
        # reload on this hardware costs ~30+ seconds on top of generation
        # (measured: 53s cold vs 21s warm for a similar reply). -1 keeps it
        # resident -- worth it since this process's whole job is one model.
        self._keep_alive = -1

    def _options(self, max_tokens: Optional[int]) -> dict[str, int]:
        return self._default_options if max_tokens is None else {"num_predict": max_tokens}

    def generate(self, messages: Messages, max_tokens: Optional[int] = None) -> str:
        response = self._client.chat(
            model=MODELS.ollama_model,
            messages=messages,
            options=self._options(max_tokens),
            keep_alive=self._keep_alive,
        )
        return response["message"]["content"]

    def stream(self, messages: Messages, max_tokens: Optional[int] = None) -> Iterator[str]:
        for chunk in self._client.chat(
            model=MODELS.ollama_model,
            messages=messages,
            options=self._options(max_tokens),
            stream=True,
            keep_alive=self._keep_alive,
        ):
            piece = chunk["message"]["content"]
            if piece:
                yield piece


def _split_system(messages: Messages) -> tuple[str, Messages]:
    """Split the system prompt out of the message list.

    Anthropic takes the system prompt as its own argument rather than as a
    message with role "system", so it has to come out of the list. Several are
    joined rather than the first one kept, because a demo may layer a persona
    or grounding block on top of the standing prompt.
    """
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [m for m in messages if m["role"] != "system"]
    return system, rest


def _cloud_max_tokens(max_tokens: Optional[int]) -> int:
    return max_tokens if max_tokens is not None else MODELS.cloud_max_tokens


class AnthropicBackend(Backend):
    """A Claude model over the API. Used when a key is configured."""

    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        # Imported here, not at module top: the package is optional, and the
        # robot must still start on a machine that never installed it.
        import httpx
        from anthropic import Anthropic

        self._client = Anthropic(
            api_key=api_key,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S),
            max_retries=_MAX_RETRIES,
        )

    def generate(self, messages: Messages, max_tokens: Optional[int] = None) -> str:
        system, rest = _split_system(messages)
        response = self._client.messages.create(
            model=MODELS.anthropic_model,
            max_tokens=_cloud_max_tokens(max_tokens),
            system=system,
            messages=rest,
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def stream(self, messages: Messages, max_tokens: Optional[int] = None) -> Iterator[str]:
        system, rest = _split_system(messages)
        with self._client.messages.stream(
            model=MODELS.anthropic_model,
            max_tokens=_cloud_max_tokens(max_tokens),
            system=system,
            messages=rest,
        ) as stream:
            yield from stream.text_stream


class OpenAIBackend(Backend):
    """An OpenAI model, or anything else speaking the same API.

    No message rewriting is needed here: the {"role", "content"} list that
    prompts.build_messages produces is already the Chat Completions shape,
    system entry included. Setting MODELS.openai_base_url points this at Azure
    OpenAI, OpenRouter, Groq or a self-hosted server without touching code.
    """

    name = "openai"

    def __init__(self, api_key: str) -> None:
        import httpx
        from openai import OpenAI

        base_url = MODELS.openai_base_url or None
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT_S, connect=_CONNECT_TIMEOUT_S),
            max_retries=_MAX_RETRIES,
        )

    def generate(self, messages: Messages, max_tokens: Optional[int] = None) -> str:
        response = self._client.chat.completions.create(
            model=MODELS.openai_model,
            messages=messages,
            max_tokens=_cloud_max_tokens(max_tokens),
        )
        return response.choices[0].message.content or ""

    def stream(self, messages: Messages, max_tokens: Optional[int] = None) -> Iterator[str]:
        for chunk in self._client.chat.completions.create(
            model=MODELS.openai_model,
            messages=messages,
            max_tokens=_cloud_max_tokens(max_tokens),
            stream=True,
        ):
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                yield piece


#: provider name -> (backend class, key env var name, package to install,
#: the configured model). Adding a provider is adding a row here plus its
#: Backend subclass above.
_CLOUD_PROVIDERS = {
    "anthropic": (AnthropicBackend, "anthropic_key_env", "anthropic", "anthropic_model"),
    "openai": (OpenAIBackend, "openai_key_env", "openai", "openai_model"),
}

#: Order tried by "auto". Whichever has a key set wins; both set means the
#: first one, which is only a tie-break -- pin llm_backend to be explicit.
_AUTO_ORDER = ("anthropic", "openai")


def _make_cloud(provider: str, local: Backend) -> Optional[Backend]:
    """Build one cloud backend, or return None with a logged reason."""
    backend_cls, key_env_attr, package, model_attr = _CLOUD_PROVIDERS[provider]
    key_env = getattr(MODELS, key_env_attr)
    api_key = MODELS.api_key(key_env)
    if not api_key:
        logger.warning("LLM: backend '%s' requested but %s is not set", provider, key_env)
        return None
    try:
        backend = backend_cls(api_key)
    except ImportError:
        logger.warning(
            "LLM: %s is set but the '%s' package is not installed (pip install %s); "
            "using local model %s",
            key_env,
            package,
            package,
            MODELS.ollama_model,
        )
        return None
    except Exception as exc:
        logger.warning("LLM: could not start %s backend (%s)", provider, exc)
        return None
    logger.info(
        "LLM: %s (%s), falling back to %s",
        getattr(MODELS, model_attr),
        provider,
        MODELS.ollama_model,
    )
    return backend


def _build() -> tuple[Backend, Backend]:
    """Return (preferred, fallback). Fallback is always the local model.

    Selection is deliberately never a hard failure: a missing key, a missing
    package or a broken client all mean "run locally", logged once at startup,
    rather than a robot that refuses to boot on the morning of an open day.
    """
    local = OllamaBackend()
    choice = MODELS.llm_backend

    if choice == "ollama":
        logger.info("LLM: local model %s (backend pinned to ollama)", MODELS.ollama_model)
        return local, local

    if choice in _CLOUD_PROVIDERS:
        cloud = _make_cloud(choice, local)
        return (cloud or local), local

    if choice != "auto":
        logger.warning("LLM: unknown backend %r; falling back to auto-selection", choice)

    for provider in _AUTO_ORDER:
        key_env = getattr(MODELS, _CLOUD_PROVIDERS[provider][1])
        if not MODELS.api_key(key_env):
            continue
        cloud = _make_cloud(provider, local)
        if cloud is not None:
            return cloud, local

    logger.info("LLM: local model %s (no cloud API key set)", MODELS.ollama_model)
    return local, local


PREFERRED, FALLBACK = _build()
