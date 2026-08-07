"""Per-person, per-session conversation memory: in-process only, cleared once
a conversation ends. This keeps a single conversation coherent turn-to-turn;
brain/long_term_memory.py is what carries facts *across* conversations.
"""

from collections import defaultdict

#: Turns kept per person. Every unrecognised visitor is person_id 0 (see
#: voice_loop.run_once), so on an open day this one bucket accumulates the
#: whole day's conversations from everybody. Unbounded, that grows the prompt
#: on every turn until it overruns the model's context, at which point the
#: oldest tokens -- the system prompt -- are dropped, and the robot goes back
#: to offering to play music (prompts.py) and reading "[emotion: happy]" out
#: loud (emotion.py). Trimming here rather than at the prompt also means the
#: turns being discarded are other visitors', which is what you want anyway.
MAX_TURNS = 12

_HISTORY: dict[int, list[tuple[str, str]]] = defaultdict(list)


def get_history(person_id: int) -> list[tuple[str, str]]:
    """Return the (message, reply) turns recorded for this person so far this session."""
    return list(_HISTORY[person_id])


def remember_turn(person_id: int, message: str, reply: str) -> None:
    """Record one conversation turn for this person, keeping the last MAX_TURNS."""
    turns = _HISTORY[person_id]
    turns.append((message, reply))
    if len(turns) > MAX_TURNS:
        del turns[: len(turns) - MAX_TURNS]


def clear_history(person_id: int) -> None:
    """Clear this person's in-session turn buffer. Call once a conversation ends."""
    _HISTORY.pop(person_id, None)


def known_people() -> list[int]:
    """Everyone with turns recorded this session, so they can be closed out."""
    return [person_id for person_id, turns in _HISTORY.items() if turns]
