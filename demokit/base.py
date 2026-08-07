"""What a demo is, and what it is given to work with.

This module is a stdlib-only leaf on purpose. It imports nothing from `brain`
or `body` at module scope -- the hardware types arrive only under
TYPE_CHECKING -- because `brain.modes` is imported by `body.audio_io`, which is
imported by the voice loop, which owns demos. Anything imported here at module
scope joins that cycle.

THE ONE RULE
Every method on DemoContext that touches the microphone or speaker must be
called from the voice-loop thread. One thread owns the audio hardware; two
threads driving it interleave into noise. The dashboard runs on its own thread
and must queue work through RobotState instead. This is checked at runtime
rather than documented and hoped for -- see _require_loop_thread.

THE OTHER RULE
Hooks return quickly. The core polls demos in short slices so an operator's
mode switch lands within a second or two, mid-visit, in front of people. A hook
that blocks for thirty seconds makes the robot unresponsive and deaf, because
nothing is consuming the microphone while it runs. If a demo wants the robot to
listen, it says so by returning IdleResult(listen_for=...) and lets the core do
the listening -- the core can bound that; a demo cannot.
"""

import logging
import math
import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from body.audio_io import AudioIO
    from body.face_tracker import FaceTracker
    from body.motion import MotionController

logger = logging.getLogger(__name__)

#: Longest single listening slice the core will honour, whatever a demo asks
#: for. A demo holding the microphone for longer than this cannot be
#: interrupted by the dashboard, which is the one thing an operator needs to
#: work while a group is watching.
MAX_LISTEN_WINDOW_S = 3.0

#: Slice length for ctx.sleep, so a long pause still notices a mode change.
_SLEEP_SLICE_S = 1.0


@dataclass(frozen=True)
class IdleResult:
    """What a demo wants to happen during one idle slice.

    listen_for is seconds to listen for the wake word. 0 means "just cycle" --
    the core still re-reads the mode and services the dashboard either way, so
    returning zero forever is safe, merely inert.
    """

    listen_for: float = 0.0

    @staticmethod
    def sanitised(value: Any, demo_id: str) -> "IdleResult":
        """Coerce whatever a hook returned into something the core can act on.

        Forgetting `return` is the most likely mistake in a first demo, and it
        yields None. Left alone that raises inside the runner rather than
        inside the guarded hook, which used to take the whole process down.
        NaN is worse than useless: it survives min()/max() and turns the idle
        loop into a busy spin that starves the motion thread of the GIL.
        """
        if isinstance(value, IdleResult):
            seconds = value.listen_for
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            seconds = float(value)
        elif value is None:
            return IdleResult(0.0)
        else:
            logger.warning("%s.on_idle returned %r; treating as idle", demo_id, type(value).__name__)
            return IdleResult(0.0)

        if not math.isfinite(seconds):
            logger.warning("%s.on_idle asked to listen for %r; treating as idle", demo_id, seconds)
            return IdleResult(0.0)
        return IdleResult(max(0.0, min(float(seconds), MAX_LISTEN_WINDOW_S)))


class DemoStopped(Exception):
    """Raised inside a demo when the operator switched away mid-hook.

    Caught by the runner, never an error. It exists so a demo part-way through
    speaking several lines unwinds promptly instead of finishing a script
    nobody selected any more.
    """


class DemoContext:
    """Everything a demo is allowed to do, and the only way it should do it.

    A demo could reach past this to `ctx.audio` or `ctx.motion` directly, and
    occasionally must -- a framework that forbids everything gets forked. But
    the methods here are the supported surface: they check the calling thread,
    they notice mode changes, and they keep the robot answerable to the
    dashboard while a demo is running.
    """

    def __init__(
        self,
        *,
        audio: "AudioIO",
        motion: "MotionController",
        tracker: Optional["FaceTracker"],
        state: Any,
        demo_id: str,
        store: dict,
    ) -> None:
        self.audio = audio
        self.motion = motion
        self.tracker = tracker
        self.state = state
        self.demo_id = demo_id
        #: Scratch space for this demo, keyed by demo id and kept for the life
        #: of the process -- it survives being switched away from and back to,
        #: which is what lets a demo remember that it has already introduced
        #: itself to the room. It is NOT cleared between visitors: a demo that
        #: wants a clean slate per session must clear it itself in on_enter.
        #: Use it rather than module globals, which the old voice loop used and
        #: which leaked state between demos as well as between visitors.
        self.store = store
        self._owner_thread = threading.get_ident()
        self._entered_mode = state.mode

    # --- guard rails -----------------------------------------------------

    def _require_loop_thread(self, what: str) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError(
                f"{self.demo_id}: {what}() was called from a different thread than the "
                "voice loop. One thread owns the microphone and speaker; calling audio "
                "from the dashboard thread interleaves two streams into noise. Queue the "
                "work with state.request(...) instead."
            )

    def mode_changed(self) -> bool:
        """True once the operator has selected a different demo."""
        return self.state.mode != self._entered_mode

    def _stop_if_switched(self) -> None:
        if self.mode_changed():
            raise DemoStopped(self._entered_mode)

    # --- speaking --------------------------------------------------------

    def say(self, text: str, emotion: str = "neutral", expressive: bool = False) -> None:
        """Speak one line, with a matching expression."""
        self._require_loop_thread("say")
        text = (text or "").strip()
        if not text:
            return
        self._stop_if_switched()
        self.state.add("said", text)
        self.state.set_flags(speaking=True)
        self.motion.express(emotion)
        self.audio.speak(text, emotion, motion=self.motion, expressive=expressive)
        self.state.set_flags(speaking=False)

    def say_lines(self, lines: Iterator[str] | list[str], emotion: str = "neutral") -> None:
        """Speak several lines, giving up promptly if the demo is switched away.

        Long scripts are spoken a line at a time rather than as one string
        because nothing consumes the microphone while the robot talks: a
        thirty-second utterance is thirty seconds during which the wake word
        cannot be heard and the backlog fills with the robot's own voice.
        """
        for line in lines:
            self._stop_if_switched()
            self.say(line, emotion)

    def reply(
        self,
        message: str,
        *,
        person_id: int = 0,
        style: Optional[str] = None,
        system: Optional[str] = None,
    ) -> str:
        """Ask the language model and speak the answer, sentence by sentence.

        `system` layers extra instructions on for this turn only -- Hub facts
        to answer from, a persona to answer in. Use it rather than pasting the
        material into `message`: anything in the message is what the visitor
        appears to have said, so it is remembered as a conversation turn and
        replayed on every later request for the rest of the session.

        Returns the full reply text. Imported here rather than at module scope
        to keep this module a leaf (see the module docstring).
        """
        self._require_loop_thread("reply")
        self._stop_if_switched()
        from brain.interface import stream_reply

        spoken: list[str] = []
        final_tag = "neutral"
        self.motion.express("thinking")
        for sentence, tag in stream_reply(person_id, message, style=style, extra_system=system):
            final_tag = tag
            if not sentence:
                continue
            # Checked between sentences, not within: cutting the robot off
            # mid-word reads as a fault, mid-sentence reads as responsive.
            self._stop_if_switched()
            spoken.append(sentence)
            self.say(sentence, tag, expressive=style == "story")
        self.motion.express_move(final_tag)
        return " ".join(spoken)

    # --- listening -------------------------------------------------------

    def listen(self) -> str:
        """Transcribe one utterance. Returns "" if nothing was understood."""
        self._require_loop_thread("listen")
        self._stop_if_switched()
        self.state.set_flags(listening=True)
        try:
            heard = self.audio.listen() or ""
        finally:
            self.state.set_flags(listening=False)
        heard = heard.strip()
        if heard:
            self.state.add("heard", heard)
        return heard

    def ask(self, question: str, emotion: str = "curious") -> str:
        """Say something and listen for the answer. "" if nothing came back."""
        self.say(question, emotion)
        return self.listen()

    def wait_for_wake_word(self, seconds: float) -> bool:
        """Listen for the wake word for at most `seconds`."""
        self._require_loop_thread("wait_for_wake_word")
        return bool(self.audio.wait_for_wake_word(timeout=max(0.0, seconds)))

    def sleep(self, seconds: float) -> None:
        """Pause, in slices, so a mode change is still noticed."""
        self._require_loop_thread("sleep")
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            self._stop_if_switched()
            time.sleep(min(_SLEEP_SLICE_S, deadline - time.monotonic()))

    # --- convenience -----------------------------------------------------

    def face_visible(self, max_age_s: float = 1.5) -> bool:
        """Whether someone is currently in view. False when face features are off."""
        if self.tracker is None or not self.tracker.enabled:
            return False
        _person_id, face = self.tracker.current(max_age_s=max_age_s)
        return face is not None

    def person_id(self) -> int:
        """The recognised person, or 0 for an unknown visitor."""
        if self.tracker is None or not self.tracker.enabled:
            return 0
        person_id, _face = self.tracker.current()
        return int(person_id or 0)

    def status(self, text: str) -> None:
        """Note something on the dashboard without speaking it."""
        self.state.add("status", text)


class Demo(ABC):
    """One demonstration. Subclass this in a file under `demos/`.

    Only `label` is worth setting by hand; `id` defaults to the module name,
    which is what the dashboard and the API use. Every hook is optional --
    a demo that only answers questions needs none of them.
    """

    #: Defaults to the module's filename. Set explicitly only to keep a URL
    #: stable while renaming the file.
    id: ClassVar[str] = ""
    #: Shown on the dashboard button.
    label: ClassVar[str] = ""
    #: One line under the button, telling an operator what this does.
    help: ClassVar[str] = ""
    #: Ordering on the dashboard; lower sorts first.
    order: ClassVar[int] = 100
    #: Capabilities this demo needs. "faces" marks it unavailable when face
    #: detection is off (as it always is on the robot's own CPU, where
    #: MediaPipe crashes), so the dashboard can grey it out with a reason
    #: instead of offering something that will silently do nothing.
    requires: ClassVar[tuple[str, ...]] = ()
    #: Spoken phrases that switch to this demo from anywhere, e.g. ("dance",).
    #: Matched as substrings of the transcript, longest first.
    triggers: ClassVar[tuple[str, ...]] = ()
    #: When True, this demo's on_utterance sees every transcript before any
    #: other demo's triggers are considered. For demos running a question-and-
    #: answer sequence, where a visitor's answer would otherwise be swallowed
    #: by another demo's trigger word.
    claims_utterances: ClassVar[bool] = False

    def on_enter(self, ctx: DemoContext) -> None:
        """Called once when this demo is selected. Keep it short.

        Speak at most a line here. Anything longer belongs in on_idle, one line
        per slice, so the operator can still switch away part-way through.
        """

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        """Called repeatedly while this demo is selected and nothing is happening.

        Do one small thing and return. Return IdleResult(listen_for=N) to have
        the core listen for the wake word for up to N seconds.
        """
        return IdleResult(listen_for=2.0)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        """Called with what a visitor said after the wake word.

        Return True if handled. Returning False falls through to the default
        behaviour, which is to answer conversationally.
        """
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        """Called when the operator switches away. Stop anything still running."""
