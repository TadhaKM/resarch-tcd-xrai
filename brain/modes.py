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

Mode = Literal["conversation", "greeter", "dance", "story", "idle"]

#: Order here is the order shown in the dashboard.
MODES: tuple[Mode, ...] = ("conversation", "greeter", "dance", "story", "idle")

MODE_LABELS: dict[Mode, str] = {
    "conversation": "Conversation",
    "greeter": "Greeter",
    "dance": "Dance",
    "story": "Storyteller",
    "idle": "Idle",
}

MODE_HELP: dict[Mode, str] = {
    "conversation": "Say a wake phrase, then talk. One wake phrase per question.",
    "greeter": "Greets people it sees, then chats. Needs face detection.",
    "dance": "Dances on a loop. Say a wake phrase to talk instead.",
    "story": "Tells a short story now and then. Ask for another one anytime.",
    "idle": "Awake and listening for the wake word, but starts nothing itself.",
}

#: How many events the dashboard can show. Old ones are dropped rather than
#: kept forever -- this is a live view, not a transcript archive.
_MAX_EVENTS = 200

#: The transcript archive, by contrast, keeps the whole session (capped so a
#: robot left running for a week cannot eat the laptop's memory).
_MAX_HISTORY = 10000

#: Queued dashboard requests waiting for the voice loop to pick them up.
#: Small on purpose: these are button presses, not a message bus.
_MAX_REQUESTS = 5


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
        # "Turn off" puts the robot to sleep rather than ending the process:
        # it stays listening for the wake word so it can be woken by voice,
        # but does nothing on its own until then.
        self._sleeping = False
        self._last_heard_at: Optional[float] = None
        # Set once the loop is past model loading; the dashboard shows
        # "starting" until then, so a slow start is not mistaken for a fault.
        self._ready = False
        self._history: list[Event] = []
        self._requests: list[tuple[str, str, str]] = []
        # Set on every mode change, consumed by whoever needs an
        # on-entry action (storyteller tells its first story from this).
        self._mode_dirty = False

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
            self._mode_dirty = True
        self.add("status", f"Mode set to {MODE_LABELS[mode]}")
        return True

    def pop_mode_changed(self) -> bool:
        """True exactly once after each mode change."""
        with self._lock:
            dirty = self._mode_dirty
            self._mode_dirty = False
            return dirty

    # --- dashboard requests ---

    def request(self, kind: str, text: str = "", tag: str = "neutral") -> bool:
        """Queue a dashboard action ("say" or "listen") for the voice loop.

        The loop is the only thing allowed to drive the speaker and mic --
        speaking from the web thread would interleave two audio streams into
        garbage -- so buttons queue here and the loop acts between turns.
        """
        with self._lock:
            if len(self._requests) >= _MAX_REQUESTS:
                return False
            self._requests.append((kind, text, tag))
            return True

    def pop_request(self) -> Optional[tuple[str, str, str]]:
        with self._lock:
            return self._requests.pop(0) if self._requests else None

    @property
    def sleeping(self) -> bool:
        with self._lock:
            return self._sleeping

    def set_sleeping(self, sleeping: bool) -> None:
        with self._lock:
            if sleeping == self._sleeping:
                return
            self._sleeping = sleeping
        self.add("status", "Asleep -- say \"Hey Reachy\" to wake me" if sleeping else "Awake")

    # --- events ---

    def add(self, kind: str, text: str) -> None:
        """Record something the dashboard should show."""
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._seq += 1
            event = Event(kind=kind, text=text)
            self._events.append(event)
            if len(self._events) > _MAX_EVENTS:
                del self._events[: len(self._events) - _MAX_EVENTS]
            self._history.append(event)
            if len(self._history) > _MAX_HISTORY:
                del self._history[: len(self._history) - _MAX_HISTORY]
            if kind == "heard":
                self._last_heard_at = time.time()

    def history(self) -> list[Event]:
        """The whole session's events, for transcript export."""
        with self._lock:
            return list(self._history)

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
                "sleeping": self._sleeping,
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
