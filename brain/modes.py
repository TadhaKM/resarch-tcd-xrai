"""What the robot is currently doing, and the record of what it heard.

Shared between the voice loop and the web dashboard, which run on different
threads: the loop reads `mode` at each turn boundary and appends events, the
dashboard writes `mode` and reads events. One lock guards both -- these are
small, infrequent operations, so there's nothing to gain from anything finer.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

#: What the robot is doing is now whichever demo is selected, and the demos are
#: whatever files exist in demos/ (see demokit/registry.py). So the list cannot
#: live here as a Literal and a set of dicts, the way it did when there were
#: five fixed modes -- adding a demonstration would mean editing this file, and
#: the whole point of the framework is that it means adding one file and
#: nothing else.
#:
#: The list is pushed in by the app at startup (set_demos) rather than pulled
#: from the registry here. That is deliberate: body/audio_io.py imports this
#: module, so anything this module imports from the demo side would close an
#: import cycle. Inverting it keeps RobotState a plain state container that
#: knows nothing about demos beyond the ids it has been handed.
DEFAULT_MODE = "conversation"

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

    def __init__(self, mode: str = DEFAULT_MODE) -> None:
        self._lock = threading.Lock()
        self._mode = mode
        #: Dashboard entries for the demos that exist, in display order. Empty
        #: until the app calls set_demos, which is why the dashboard tolerates
        #: an empty list on its first poll rather than latching it.
        self._demos: list[dict] = []
        #: Installed speaking voices and the one in use, published by the voice
        #: loop (which is the only thread allowed to read them off disk).
        self._voices: dict = {"available": [], "current": ""}
        #: Off by default. See set_web_search.
        self._web_search = False
        #: Off by default. See set_open_mic.
        self._open_mic = False
        #: Which answering style the personality demo should use next. Held in
        #: the core rather than in the demo's own store so the dashboard can
        #: set it directly -- picking one from a list is what an operator wants
        #: when showing the contrast, where saying "switch personality" twice to
        #: reach the third one is a demonstration of patience.
        self._persona = ""
        #: Names greeted this session. See mark_greeted.
        self._greeted_names: set[str] = set()
        #: Bumped whenever the dashboard picks one, so the demo can tell an
        #: operator's choice from its own last selection and re-answer.
        self._persona_seq = 0
        #: Whether the live backend can search at all, so the dashboard can
        #: grey the switch instead of offering something that does nothing.
        self._web_search_available = False
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

    def set_demos(self, entries: list[dict]) -> None:
        """Tell the dashboard which demos exist. Called once at startup.

        Entries are the dicts demokit.registry.dashboard_entries produces:
        id, label, help, available, note.
        """
        with self._lock:
            self._demos = list(entries)
            known = {e["id"] for e in self._demos}
            current_missing = self._mode not in known
            first = self._demos[0]["id"] if self._demos else self._mode
        if current_missing and self._demos:
            # The saved-or-default mode does not exist in this build. Better to
            # start in something real than to sit in a mode nothing implements.
            with self._lock:
                self._mode = first
            self.add("status", f"Starting in {first}")

    def demos(self) -> list[dict]:
        with self._lock:
            return list(self._demos)

    @property
    def web_search(self) -> bool:
        """Whether the robot may look things up online. Off unless switched on.

        Off by default deliberately. The robot's standing claim is that it has
        no live information, and a demonstration that quietly went and searched
        would make that claim a lie -- as well as adding seconds to a spoken
        turn and spending credits per question. An operator turns it on for the
        part of a visit where it is the point, and off again afterwards.
        """
        with self._lock:
            return self._web_search

    def set_web_search(self, enabled: bool) -> bool:
        with self._lock:
            if enabled == self._web_search:
                return enabled
            self._web_search = enabled
        self.add("status", "Web search on -- answers may take longer" if enabled else "Web search off")
        return enabled

    @property
    def open_mic(self) -> bool:
        """Whether a conversation may continue without the wake word each time.

        Off by default, and deliberately not a way to make the robot listen
        permanently. The wake word still opens a conversation; this only keeps
        it open afterwards, so follow-up questions can be asked the way they
        would be asked of a person.

        Always listening was the obvious reading and is the wrong one here. The
        AGC notes what happens when this room's background speech reaches the
        recognizer: it transcribes the room continuously, and that text becomes
        questions the robot answers out loud to nobody.
        """
        with self._lock:
            return self._open_mic

    def set_open_mic(self, enabled: bool) -> bool:
        with self._lock:
            if enabled == self._open_mic:
                return enabled
            self._open_mic = enabled
        self.add(
            "status",
            "Open mic on -- follow-ups need no wake word" if enabled else "Open mic off",
        )
        return enabled

    @property
    def persona(self) -> tuple[str, int]:
        """The requested answering style and the request's sequence number."""
        with self._lock:
            return self._persona, self._persona_seq

    def set_persona(self, persona_id: str) -> str:
        """Set the robot's standing manner. "" means its own, unstyled.

        Resolved through personas.active rather than personas.get, which falls
        back to Professional for anything it does not recognise -- including
        the empty string. Coerced that way, "no persona" became a robot-wide
        Professional register that nobody had chosen, and there was no value
        left that meant "off".
        """
        from brain import personas

        chosen = personas.active(persona_id)
        with self._lock:
            self._persona = chosen.id if chosen else ""
            self._persona_seq += 1
        self.add("status", f"Personality: {chosen.label}" if chosen else "Personality: default")
        return chosen.id if chosen else ""

    def mark_greeted(self, name: str) -> bool:
        """Claim the one per-session greeting for `name`. True for the claimant.

        In state rather than in the runner because two places spend it: the
        runner greets a recognised face, and the vision demo's enrolment ends
        on "thank you, Sarah" -- after which the tracker starts recognising
        her, and without a shared record the runner follows the thank-you with
        "oh, hello again Sarah" three seconds later.
        """
        with self._lock:
            if name in self._greeted_names:
                return False
            self._greeted_names.add(name)
            return True

    def forget_greeted(self, name: str) -> None:
        """Release a spent greeting -- used when a name turns out to be wrong."""
        with self._lock:
            self._greeted_names.discard(name)

    def set_web_search_available(self, available: bool) -> None:
        with self._lock:
            self._web_search_available = available

    def set_voices(self, available: list[str], current: str) -> None:
        """Publish the installed voices for the dashboard. Called by the loop."""
        with self._lock:
            self._voices = {"available": list(available), "current": current}

    def voices(self) -> dict:
        with self._lock:
            return dict(self._voices)

    def refresh_demo_availability(self, entries: list[dict]) -> None:
        """Update availability/notes without changing the selection.

        Called when a demo is set aside or re-enabled, so the dashboard greys
        the button and shows why while the session is still running.
        """
        with self._lock:
            self._demos = list(entries)

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> bool:
        with self._lock:
            known = {e["id"] for e in self._demos}
            label = next((e["label"] for e in self._demos if e["id"] == mode), mode)
            # Before any demo list arrives, accept anything: the voice loop
            # sets a mode during startup, and rejecting it would leave the
            # robot in whatever it booted with.
            if known and mode not in known:
                return False
            if mode == self._mode:
                return True
            self._mode = mode
            self._mode_dirty = True
        self.add("status", f"Mode set to {label}")
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
                "modes": list(self._demos),
                "web_search": self._web_search,
                "open_mic": self._open_mic,
                "persona": self._persona,
                "web_search_available": self._web_search_available,
            }


#: The one instance the loop and the dashboard share.
STATE = RobotState()
