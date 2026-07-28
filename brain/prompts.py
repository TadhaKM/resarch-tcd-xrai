"""Prompt construction: builds the chat message list sent to the LLM."""

from .emotion import VALID_EMOTION_TAGS

_TAG_LIST = ", ".join(sorted(VALID_EMOTION_TAGS))

_BASE_SYSTEM_PROMPT = (
    "You are Reachy Mini, a small expressive robot having a spoken conversation. "
    "Keep replies to one or two short sentences. "
    f"End every reply with exactly one emotion tag from this list: {_TAG_LIST} -- "
    "formatted like '[emotion: happy]', as the very last thing you say, e.g. "
    "\"What's your name? [emotion: curious]\". Never use any other tag or format."
)


def build_messages(
    long_term_context: str, history: list[tuple[str, str]], message: str
) -> list[dict[str, str]]:
    """Build the chat message list sent to the LLM for this person's next turn.

    long_term_context is what's remembered about this person from *previous*
    conversations (brain/long_term_memory.py); history is this session's
    turns so far (brain/memory.py).
    """
    system_prompt = _BASE_SYSTEM_PROMPT
    if long_term_context:
        system_prompt += (
            "\n\nWhat you remember about this person from previous conversations:\n"
            f"{long_term_context}"
        )

    messages = [{"role": "system", "content": system_prompt}]
    for user_message, reply in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": message})
    return messages
