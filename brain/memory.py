"""Per-person, per-session conversation memory: in-process only, cleared once
a conversation ends. This keeps a single conversation coherent turn-to-turn;
brain/long_term_memory.py is what carries facts *across* conversations.
"""

from collections import defaultdict

_HISTORY: dict[int, list[tuple[str, str]]] = defaultdict(list)


def get_history(person_id: int) -> list[tuple[str, str]]:
    """Return the (message, reply) turns recorded for this person so far this session."""
    return list(_HISTORY[person_id])


def remember_turn(person_id: int, message: str, reply: str) -> None:
    """Record one conversation turn for this person."""
    _HISTORY[person_id].append((message, reply))


def clear_history(person_id: int) -> None:
    """Clear this person's in-session turn buffer. Call once a conversation ends."""
    _HISTORY.pop(person_id, None)
