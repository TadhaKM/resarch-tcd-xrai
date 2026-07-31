"""What the robot is currently doing, and the record of what it heard.

Shared between the voice loop and the web dashboard, which run on different
threads: the loop reads `mode` at each turn boundary and appends events, the
dashboard writes `mode` and reads events. One lock guards both -- these are
small, infrequent operations, so there's nothing to gain from anything finer.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

Mode = Literal["conversation", "greeter", "dance", "idle"]

#: Order here is the order shown in the dashboard.
MODES: tuple[Mode, ...] = ("conversation", "greeter", "dance", "idle")

MODE_LABELS: dict[Mode, str] = {
    "conversation": "Conversation",
    "greeter": "Greeter",
    "dance": "Dance",
    "idle": "Idle",
}

MODE_HELP: dict[Mode, str] = {
    "conversation": "Say \"Hey Reachy\", then talk. Follow-ups need no wake word.",
    "greeter": "Greets people it sees, then chats. Needs face detection.",
    "dance": "Dances on a loop. Say \"Hey Reachy\" to talk instead.",
    "idle": "Awake and listening for the wake word, but starts nothing itself.",
}

#: How many events the dashboard can show. Old ones are dropped rather than
#: kept forever -- this is a live view, not a transcript archive.
_MAX_EVENTS = 200


@dataclass
class Event:
    """One thing worth showing on the dashboard."""

    kind: str  # "heard" | "said" | "status" | "error"
    text: str
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "at": self.at}


class RobotState:
    """Current mode, recent events, and whether anything is actually working."""

    def __init__(self, mode: Mode = "conversation") -> None:
        self._lock = threading.Lock()
        self._mode = mode
        self._events: list[Event] = []
        self._seq = 0
        self._listening = False
        self._speaking = False
        self._face_visible = False
        self._started_at = time.time()
        self._last_heard_at: Optional[float] = None
        # Set once the loop is past model loading; the dashboard shows
        # "starting" until then, so a slow start is not mistaken for a fault.
        self._ready = False

    # --- mode ---

    @property
    def mode(self) -> Mode:
        with self._lock:
            return self._mode

    def set_mode(self, mode: Mode) -> bool:
        if mode not in MODES:
            return False
        with self._lock:
            if mode == self._mode:
                return True
            self._mode = mode
        self.add("status", f"Mode set to {MODE_LABELS[mode]}")
        return True

    # --- events ---

    def add(self, kind: str, text: str) -> None:
        """Record something the dashboard should show."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._seq += 1
            self._events.append(Event(kind=kind, text=text))
            if len(self._events) > _MAX_EVENTS:
                del self._events[: len(self._events) - _MAX_EVENTS]
            if kind == "heard":
                self._last_heard_at = time.time()

    def note(self, kind: str, text: str) -> None:
        """Record an event, unless it repeats the one before it.

        The wake-word wait is re-entered every second or two so mode changes
        are felt quickly, and logging "waiting for the wake word" each pass
        would bury everything else. This keeps the transition visible without
        the repetition.
        """
        with self._lock:
            if self._events and self._events[-1].kind == kind and self._events[-1].text == text:
                return
        self.add(kind, text)

    def events_since(self, seq: int) -> tuple[int, list[dict]]:
        """Return (latest_seq, events after `seq`).

        The dashboard polls with the sequence number it last saw, so it only
        ever transfers what is new and never misses an event between polls.
        """
        with self._lock:
            latest = self._seq
            first = len(self._events) - (latest - seq)
            if first < 0:
                first = 0
            return latest, [e.as_dict() for e in self._events[first:]]

    # --- live status ---

    def set_flags(
        self,
        *,
        listening: Optional[bool] = None,
        speaking: Optional[bool] = None,
        face_visible: Optional[bool] = None,
        ready: Optional[bool] = None,
    ) -> None:
        with self._lock:
            if listening is not None:
                self._listening = listening
            if speaking is not None:
                self._speaking = speaking
            if face_visible is not None:
                self._face_visible = face_visible
            if ready is not None:
                self._ready = ready

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "mode": self._mode,
                "ready": self._ready,
                "listening": self._listening,
                "speaking": self._speaking,
                "face_visible": self._face_visible,
                "uptime_s": time.time() - self._started_at,
                "last_heard_s_ago": (
                    None if self._last_heard_at is None else time.time() - self._last_heard_at
                ),
                "modes": [
                    {"id": m, "label": MODE_LABELS[m], "help": MODE_HELP[m]} for m in MODES
                ],
            }


#: The one instance the loop and the dashboard share.
STATE = RobotState()
