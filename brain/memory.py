"""Per-person conversation memory. Stub: in-process dict, no real database yet."""

from collections import defaultdict

_HISTORY: dict[int, list[tuple[str, str]]] = defaultdict(list)


def get_history(person_id: int) -> list[tuple[str, str]]:
    """Return the (message, reply) turns recorded for this person so far."""
    return list(_HISTORY[person_id])


def remember_turn(person_id: int, message: str, reply: str) -> None:
    """Record one conversation turn for this person."""
    _HISTORY[person_id].append((message, reply))
