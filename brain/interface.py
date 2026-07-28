"""The single entry point body/ calls into brain/."""

from .emotion import extract_emotion_tag
from .llm import generate_response
from .memory import get_history, remember_turn
from .prompts import build_messages


def get_reply(person_id: int, message: str) -> tuple[str, str]:
    """Return (reply_text, emotion_tag) for a message from the given person."""
    history = get_history(person_id)
    messages = build_messages(history, message)
    raw_output = generate_response(messages)
    reply_text, emotion_tag = extract_emotion_tag(raw_output)
    remember_turn(person_id, message, reply_text)
    return reply_text, emotion_tag
