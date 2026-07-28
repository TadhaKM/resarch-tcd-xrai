"""Prompt construction: builds the chat message list sent to the LLM."""

from .emotion import VALID_EMOTION_TAGS

_TAG_LIST = ", ".join(f"[{tag}]" for tag in sorted(VALID_EMOTION_TAGS))

SYSTEM_PROMPT = (
    "You are Reachy Mini, a small expressive robot having a spoken conversation. "
    "Keep replies to one or two short sentences. Start every reply with exactly one "
    f"of these emotion tags: {_TAG_LIST} -- followed by the spoken reply, e.g. "
    "'[curious] What's your name?'. Never use any other tag."
)


def build_messages(history: list[tuple[str, str]], message: str) -> list[dict[str, str]]:
    """Build the chat message list sent to the LLM for this person's next turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_message, reply in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": message})
    return messages
