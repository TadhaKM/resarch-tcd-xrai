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
import queue
import re
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
#: A silence longer than this between spoken chunks is worth a log line. Below
#: it, the gap is the ordinary cost of rendering and reads as breathing.
_GAP_WARN_S = 1.5

#: How long the robot listens, wake-word-free, for the answer to a question
#: its own reply just asked. 9.0 said "long enough to think" and was measured
#: wrong the first live afternoon: asked what his ideas were FOR, a visitor
#: answered "For my startup." 27 seconds after the question finished -- well
#: past the window, so his three clean words were judged as ambient noise
#: rules instead of as the answer. People formulating a real answer take
#: 15-25s. The confidence gate still applies (relaxed, see runner._dispatch),
#: and any utterance closes the window, so the cost of the longer wait is
#: bounded to one stray remark after a question genuinely nobody answers.
ANSWER_WINDOW_S = 30.0

#: The wake-free window after a reply that did NOT ask anything. The ambient
#: word-count floor still applies here (nothing was asked, so a stray syllable
#: is the room), which is what makes the longer window cheap.
FOLLOW_UP_WINDOW_S = 20.0

MAX_LISTEN_WINDOW_S = 3.0

#: Slice length for ctx.sleep, so a long pause still notices a mode change.
_SLEEP_SLICE_S = 1.0

#: Where one spoken line ends. Mirrors the boundary rule brain/interface.py
#: uses to chunk a streamed reply, so a scripted line and a generated one are
#: broken the same way. It lives here, in the leaf both demos and the core can
#: import, because welcome.py's copy of it carries the warning that earned this
#: placement: "One definition of a spoken line for the whole robot; two would
#: drift the moment either was tuned."
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(script: str) -> tuple[str, ...]:
    """A script as the lines it should be spoken in, one per idle slice.

    Speaking a long string in one call is what makes a robot deaf: nothing
    consumes the microphone while it talks, so the whole script is a stretch
    during which no wake word is heard and the operator cannot switch away.
    """
    return tuple(
        part.strip() for part in _SENTENCE_BOUNDARY_RE.split((script or "").strip()) if part.strip()
    )


#: How long the robot rests after a spoken chunk, by what ended it. Piper
#: renders each chunk as its own isolated utterance, so nothing between them
#: is silent unless something puts silence there -- and once the barge-in scan
#: moved off the critical path there WAS nothing, so a scripted answer came out
#: as one long run with the full stops inaudible. These are breaths, not gaps:
#: the silences that made the robot sound finished mid-answer measured 1.7 to
#: 4.6 seconds, and the longest of these is a third of one.
#:
#: They are a real cost, paid deliberately: the next chunk starts this much
#: later, because its audio was ready and is being held back. That is what a
#: pause IS. An earlier draft of this comment claimed the breath hid inside
#: the barge-in scan; it does not -- the scan runs on its own thread and never
#: gated the next chunk.
_BREATH_FULL_STOP_S = 0.34
_BREATH_QUESTION_S = 0.40
_BREATH_CLAUSE_S = 0.16
_BREATH_DEFAULT_S = 0.22


def breath_after(text: str) -> float:
    """Seconds of rest owed after speaking `text`.

    A full stop is a beat, a question needs slightly longer for the answer to
    feel invited, and a comma is only a breath -- a chunk ending in one is a
    long sentence that was split for rendering, so stopping there as though it
    were a sentence is exactly the flat delivery this exists to fix.
    """
    tail = (text or "").rstrip()
    if not tail:
        return 0.0
    if tail.endswith("?"):
        return _BREATH_QUESTION_S
    if tail.endswith((".", "!", "…")):
        return _BREATH_FULL_STOP_S
    if tail.endswith((",", ";", ":", "-")):
        return _BREATH_CLAUSE_S
    return _BREATH_DEFAULT_S


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


class Interrupted(Exception):
    """Raised when a visitor said the wake word while the robot was talking.

    Like DemoStopped this is not a failure: it unwinds whatever was being said
    so the runner can listen to whatever the person actually wants. It is what
    makes a thirty-second answer escapable by voice instead of only from the
    dashboard, which nobody standing in front of the robot has open.
    """


def _is_stop_phrase(text: str) -> bool:
    """Whether this is one of the core "stop" phrases.

    Imported lazily: this module is a stdlib-only leaf (see the docstring) and
    demokit.runner imports it, so naming that at module scope closes a cycle.
    """
    from .runner import SLEEP_PHRASES, _word_stream, contains_phrase

    words = _word_stream(text)
    return any(contains_phrase(words, phrase) for phrase in SLEEP_PHRASES)


#: The wake phrase, as the robot might say it itself. A reply containing it
#: would be found in the robot's own recorded voice and interrupt itself, so
#: lines like "say hey Reachy if you want me to stop" skip the check.
_WAKE_WORDS = ("reachy",)


#: How long a recognised identity keeps its conversation after the face was
#: last actually seen. The tracker forgets a face 3 seconds after losing it,
#: and conversation history is keyed by person id -- so every time the robot's
#: head turned away in its thinking pose, or the visitor leaned out of frame,
#: the NEXT turn was filed under anonymous person 0 with an empty history.
#: Measured live as the robot asking "Ideas for what, exactly?" one turn after
#: a five-minute startup-ideas conversation, and answering a clean direct
#: answer to its own question with "could you say that again?".
#:
#: 75 seconds, not more, and the bound is privacy rather than tuning: review
#: found that a NEW visitor arriving outside the camera cone inside this
#: window would inherit the previous person's name and history, and -- at the
#: next "goodbye" -- have their conversation summarised into that person's
#: permanent profile. The dropout this exists to bridge is one reply plus one
#: thought (a long reply is ~30s, live thinking time measured at 27s);
#: anything longer is more likely a new person than a long blink. The other
#: half of the guard is the runner's: a wake word from nobody-in-view after a
#: real lull drops the identity outright (_WAKE_NEW_VISIT_S). The residual --
#: a second visitor diving into a live conversation, off camera, without the
#: wake word, inside this window -- is accepted and recorded here.
STICKY_IDENTITY_S = 75.0


class _StickyIdentity:
    """The person this conversation belongs to, surviving camera dropouts.

    Module-level rather than on DemoContext because contexts are per-demo: a
    conversation that wanders across demos ("tell me a story" mid-chat) must
    not lose its owner at the switch. Written only from the voice-loop thread.
    """

    person_id = 0
    seen_at = 0.0


_STICKY = _StickyIdentity()


def forget_identity() -> None:
    """Drop the sticky conversation identity. Called when a visit ends."""
    _STICKY.person_id = 0
    _STICKY.seen_at = 0.0


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
        owns_persona: bool = False,
    ) -> None:
        self.audio = audio
        self.motion = motion
        self.tracker = tracker
        self.state = state
        self.demo_id = demo_id
        #: Copied from the demo at construction; see Demo.owns_persona.
        self.owns_persona = owns_persona
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

    def persona(self):
        """The robot's standing manner, or None if the operator picked none.

        Read from state on every call rather than cached, because the operator
        changes it from the dashboard mid-visit and the next thing said should
        already be in the new manner.
        """
        from brain import personas

        chosen, _seq = self.state.persona
        return personas.active(chosen)

    def _stop_if_interrupted(self, just_said: str) -> None:
        """Give up the floor if someone said the wake word while we spoke.

        Checked between spoken lines rather than during one: the interruption
        lands at the end of the current sentence, not mid-word, which is both
        easier to implement honestly and closer to how a person yields a turn.

        Skipped when the robot has just said the wake word itself -- its own
        voice is in the same buffer being searched, so a line inviting someone
        to "say hey Reachy" would otherwise interrupt itself immediately.
        """
        lowered = just_said.lower()
        if any(word in lowered for word in _WAKE_WORDS):
            return
        if self.audio.wake_word_in_backlog():
            raise Interrupted()

    def _breathe(self, seconds: float) -> None:
        """Rest between two spoken chunks, still noticing a mode change.

        Deliberately not ctx.sleep: that slices at a full second, which is
        three breaths long, and raises DemoStopped from inside a reply the
        caller is midway through speaking.
        """
        if seconds <= 0:
            return
        time.sleep(seconds)

    def _begin_backlog_scan(self, just_said: str):
        """Start the concurrent barge-in scan for the chunk just spoken.

        Returns None for a chunk that itself contained the wake phrase -- its
        own recorded voice is in the buffer being searched, so scanning it
        would interrupt the robot with itself. On an audio object without the
        async scan (test doubles, local playback stubs) the old synchronous
        check runs instead, so nothing is lost beyond the gap it always cost.
        """
        lowered = just_said.lower()
        if any(word in lowered for word in _WAKE_WORDS):
            return None
        starter = getattr(self.audio, "scan_backlog_async", None)
        if starter is None:
            self._stop_if_interrupted(just_said)
            return None
        return starter()

    # --- speaking --------------------------------------------------------

    def say(
        self,
        text: str,
        emotion: str = "neutral",
        expressive: bool = False,
        pace: Optional[float] = None,
        variation: Optional[float] = None,
        *,
        interruptible: bool = True,
    ) -> None:
        """Speak one line, with a matching expression.

        `pace` and `variation` give this line a voice character: pace is a
        speaking-rate multiplier (higher is slower), variation is how much the
        prosody moves. Used by demos where the voice itself is the point --
        see brain/personas.py -- and ignored by everything else.

        Raises Interrupted if a visitor said the wake word over this line.
        Every scripted demo speaks through here one line per idle slice, so
        without this only generated replies and say_lines scripts could be
        talked over -- the welcome, the tour, the timer announcements all had
        to be waited out. Pass interruptible=False for a line that must finish:
        a question whose own answer is being waited for (see ask), or the
        robot's own "going to sleep".
        """
        self._require_loop_thread("say")
        text = (text or "").strip()
        if not text:
            return
        self._stop_if_switched()
        self.state.add("said", text)
        self.state.set_flags(speaking=True)
        self.motion.express(emotion)
        self.audio.speak(
            text, emotion, motion=self.motion, expressive=expressive,
            pace=pace, variation=variation,
        )
        self.state.set_flags(speaking=False)
        # ANY spoken question opens the answer window, not just generated
        # replies. The quiz, the greetings menu and every scripted demo ask
        # through say(), and a question that then demands a wake word before
        # its own answer is a trap -- reply() got this rule first, and the gap
        # was every question that never went through reply(). Demos that hold
        # the mic explicitly are unaffected: the hold outranks the window, and
        # any utterance clears both.
        if text.rstrip().endswith("?"):
            self.state.expect_answer(ANSWER_WINDOW_S)
        if interruptible:
            self._stop_if_interrupted(text)

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

    def say_script(self, lines, *, pace: Optional[float] = None,
                   interruptible: bool = True) -> None:
        """Speak scripted (text, emotion) lines as one flowing delivery.

        reply()'s pipeline applied to fixed text: line N+1 renders on a worker
        while line N is still coming out of the speaker, and the barge-in scan
        runs BESIDE the next line's playback instead of standing between
        lines. Before this, scripted blocks went through say() a line at a
        time, and every gap paid a render plus a full synchronous backlog scan
        -- measured live as 2-3 seconds of silence between one-word lines of a
        script whose whole point is that it is already written.

        interruptible=False skips the scans entirely (a stage performance may
        not be talked over); otherwise a wake word raises Interrupted exactly
        as say() would have. Ends by arming the same windows reply() does: a
        script closing on a question opens the answer window, anything else
        invites a follow-up.
        """
        self._require_loop_thread("say_script")
        script = [(t.strip(), e) for t, e in lines if t and t.strip()]
        if not script:
            return
        self._stop_if_switched()
        feed: queue.Queue = queue.Queue(maxsize=3)
        abandon = threading.Event()

        def _offer_item(item) -> bool:
            while not abandon.is_set():
                try:
                    feed.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def _produce() -> None:
            try:
                for text, emotion in script:
                    chunks = self.audio.render(text, pace=pace)
                    if not _offer_item(("say", text, emotion, chunks)):
                        return
                _offer_item(("end",))
            except Exception as exc:  # pragma: no cover - surfaced on the loop thread
                _offer_item(("error", exc))

        producer = threading.Thread(target=_produce, name="script-render", daemon=True)
        producer.start()
        scan = None
        last_line = ""
        try:
            while True:
                # Sliced like reply()'s wait, so a scan hit lands even while
                # the producer is still rendering.
                while True:
                    if scan is not None and scan.done and scan.finish():
                        raise Interrupted()
                    try:
                        item = feed.get(timeout=0.2)
                        break
                    except queue.Empty:
                        continue
                if item[0] == "end":
                    break
                if item[0] == "error":
                    raise item[1]
                _kind, text, emotion, chunks = item
                self._stop_if_switched()
                if scan is not None and scan.done and scan.finish():
                    raise Interrupted()
                self.state.add("said", text)
                self.state.set_flags(speaking=True)
                self.motion.express(emotion)
                self.audio.speak_rendered(chunks, motion=self.motion)
                self.state.set_flags(speaking=False)
                last_line = text
                if interruptible:
                    if scan is not None and scan.finish():
                        raise Interrupted()
                    scan = self._begin_backlog_scan(text)
                # The breath goes AFTER the scan is started, so the rest and
                # the scan happen at the same time rather than one after the
                # other. See breath_after.
                self._breathe(breath_after(text))
            if scan is not None and scan.finish():
                raise Interrupted()
        except BaseException:
            abandon.set()
            self.state.set_flags(speaking=False)
            if scan is not None:
                scan.finish()
            while True:
                try:
                    feed.get_nowait()
                except queue.Empty:
                    break
            raise
        if last_line.rstrip().endswith("?"):
            self.state.expect_answer(ANSWER_WINDOW_S)
        else:
            self.state.invite_followup(FOLLOW_UP_WINDOW_S)

    def reply(
        self,
        message: str,
        *,
        person_id: int = 0,
        style: Optional[str] = None,
        system: Optional[str] = None,
        cache: bool = True,
        web: Optional[bool] = None,
        pace: Optional[float] = None,
        variation: Optional[float] = None,
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

        # Tell the model who it is talking to, when it knows. Done here rather
        # than left to each demo so that being recognised feels the same
        # wherever you are in the robot -- and done through the prompt rather
        # than by pasting a name into the answer, because a model given a name
        # uses it where a person would ("Sure, Tadhg -- ...") and a template
        # would put it in the same place every time, which stops sounding
        # personal by about the third reply.
        name = self.person_name()
        if name:
            known = (
                f"You are speaking with {name}, whom you have met before and have just "
                f"recognised. Use their name occasionally and naturally, the way a person "
                f"would -- once in a while, not in every sentence and not at the start of "
                f"every reply."
            )
            system = f"{system}\n\n{known}" if system else known

        # What Trinity teaches, but only when the turn is actually about it.
        # Retrieved per turn rather than carried in the base prompt: thirteen
        # masters programmes in full is several times hub.GROUNDING, on a local
        # model with a 140-token reply budget, for something most turns never
        # ask about. brief() returns "" for the ordinary case and costs nothing.
        # Here rather than in one demo so that Conversation, About and a
        # staff-written feature's question all answer a prospective student the
        # same way -- which is what they will get if they ask twice.
        from brain import courses

        study = courses.brief(message)
        if study:
            system = f"{system}\n\n{study}" if system else study

        # And what the Hub has TAUGHT it, the same way: a paraphrase the
        # verbatim tier refused still reaches the model with the taught
        # answer in front of it, so the robot's answers improve everywhere a
        # staff member teaches it once. "" is the common case.
        from brain import knowledge

        taught = knowledge.brief(message)
        if taught:
            system = f"{system}\n\n{taught}" if system else taught

        # The robot's standing manner, layered on the same way. Deliberately
        # here rather than in prompts.py's base prompt: extra_system is what
        # brain/qa_cache.py digests into its key, so a persona that rides here
        # makes the cache persona-aware for free, where one buried in the base
        # prompt would replay a Professional answer to somebody who had just
        # switched to Friendly.
        #
        # Two exclusions, neither of which names a demo. A story swaps the
        # whole system prompt for the storyteller's and has its own voice; and
        # a demo that supplies its own style brief says so itself.
        persona = self.persona() if style is None and not self.owns_persona else None
        if persona is not None:
            from brain.personas import GLOBAL_STYLE_FRAME

            brief = f"{persona.global_prompt} {GLOBAL_STYLE_FRAME}"
            system = f"{system}\n\n{brief}" if system else brief

        spoken: list[str] = []
        final_tag = "neutral"
        self.motion.express("thinking")
        # Cleared automatically the moment the first sentence starts speaking
        # (set_flags(speaking=True) clears it), and belt-and-braces in the
        # finally below for replies that die before any sentence lands.
        self.state.set_flags(thinking=True)
        # None means "whatever the operator has the dashboard switch set to",
        # which is what every demo wants; a demo can still force it either way.
        use_web = self.state.web_search if web is None else web

        # THE PIPELINE. A producer thread pulls the model's stream and renders
        # each sentence to audio; this thread -- the one that owns the speaker
        # -- plays them back to back. Before this, each sentence was generated,
        # then synthesized, then played, then the next one begun, and the seams
        # were audible: measured live at 6-8 seconds between sentence STARTS,
        # of which a second or more was pure dead air between sentences. Now
        # generation and synthesis happen while the previous sentence is still
        # coming out of the speaker, so the only gap left is the one piper puts
        # there deliberately.
        #
        # The queue is small on purpose: an interruption abandons at most a
        # couple of rendered sentences, and the producer never runs the whole
        # reply ahead of what the room has actually heard.
        #
        # Interruption and switching still land between sentences, on THIS
        # thread, exactly as before -- the producer only generates and renders,
        # and rendering is the one AudioIO method documented as safe off the
        # loop thread (see AudioIO.render).
        feed: queue.Queue = queue.Queue(maxsize=3)
        abandon = threading.Event()

        def _offer(item) -> bool:
            """Put unless the reply has been abandoned. The short timeout is
            what lets a producer blocked on a full queue notice abandonment --
            without it, one drain on the consumer side was not enough, and the
            thread sat generating into a conversation that had moved on."""
            while not abandon.is_set():
                try:
                    feed.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def _produce() -> None:
            last_tag = "neutral"
            gen = stream_reply(
                person_id, message, style=style, extra_system=system, cache=cache, web=use_web
            )
            try:
                for sentence, tag in gen:
                    last_tag = tag
                    if abandon.is_set():
                        # Close inside the generator's own machinery: it raises
                        # GeneratorExit at the yield, so stream_reply unwinds
                        # without recording a reply the room never heard.
                        gen.close()
                        return
                    if not sentence:
                        continue
                    chunks = self.audio.render(
                        sentence, expressive=style == "story", pace=pace, variation=variation
                    )
                    if not _offer(("say", sentence, tag, chunks)):
                        gen.close()
                        return
                _offer(("end", last_tag))
            except Exception as exc:  # pragma: no cover - surfaced on the loop thread
                _offer(("error", exc))

        producer = threading.Thread(target=_produce, name="reply-render", daemon=True)
        producer.start()
        # Measures the symptom directly: how long the robot was SILENT between
        # one spoken chunk and the next. Live, a three-sentence answer had a
        # twenty-second gap in the middle, which reads to a visitor as the robot
        # having finished -- and there was no way to tell whether the time went
        # on the model generating, piper rendering, or the queue. Logged rather
        # than reasoned about, for the same reason the face-match score was.
        finished_at = None
        # The barge-in scan for the chunk that just finished, running WHILE the
        # next one plays. It used to run synchronously between chunks, and its
        # ~0.8-2.5s was the floor under every mid-reply gap the instrumentation
        # logged -- pure silence the room read as the robot having finished.
        # Now the scan of chunk N's backlog overlaps chunk N+1's playback; a
        # hit aborts that playback within about a second (see AudioIO
        # scan_backlog_async), and the Interrupted still rises from this loop.
        # None also encodes "the robot just said the wake word itself" -- that
        # chunk's backlog contains its own voice saying the phrase, so it is
        # not scanned at all, exactly as _stop_if_interrupted always skipped it.
        scan = None
        try:
            while True:
                # The wait for the producer is SLICED so a scan hit can
                # interrupt it: review confirmed that a wake word said over
                # chunk N, found by the scan while the producer was mid-stall,
                # otherwise sat ignored until the next rendered sentence
                # arrived -- the visitor barging in during the exact silence
                # that already reads as the robot having finished, and being
                # ignored the whole way through it.
                while True:
                    if scan is not None and scan.done and scan.finish():
                        raise Interrupted()
                    try:
                        item = feed.get(timeout=0.2)
                        break
                    except queue.Empty:
                        continue
                if finished_at is not None and item[0] not in ("end", "error"):
                    waited = time.monotonic() - finished_at
                    if waited >= _GAP_WARN_S:
                        logger.info("silent %.1fs waiting for the next words", waited)
                if item[0] == "end":
                    final_tag = item[1]
                    break
                if item[0] == "error":
                    raise item[1]
                _kind, sentence, tag, chunks = item
                # Checked between sentences, not within: cutting the robot off
                # mid-word reads as a fault, mid-sentence reads as responsive.
                self._stop_if_switched()
                # A scan that already concluded with a hit means the visitor
                # interrupted during the PREVIOUS chunk -- do not start another.
                if scan is not None and scan.done and scan.finish():
                    raise Interrupted()
                spoken.append(sentence)
                self.state.add("said", sentence)
                self.state.set_flags(speaking=True)
                self.motion.express(tag)
                self.audio.speak_rendered(chunks, motion=self.motion)
                finished_at = time.monotonic()
                self.state.set_flags(speaking=False)
                # The scan for the chunk that just played must be collected
                # before any new mic work begins -- two threads draining the
                # same backlog interleave it into nonsense.
                if scan is not None and scan.finish():
                    raise Interrupted()
                scan = self._begin_backlog_scan(sentence)
                self._breathe(breath_after(sentence))
            if scan is not None and scan.finish():
                # Found during the final chunk: the visitor talked over the
                # end of the reply, and their words are waiting in the backlog.
                raise Interrupted()
        except BaseException:
            # Switched away, interrupted, or a real error: the producer must
            # stop generating into a conversation that has moved on. It may be
            # blocked on the full queue, so drain space for it to notice.
            abandon.set()
            # The dashboard must not say Thinking about a reply that died --
            # interrupted, switched away, or a real failure all land here.
            self.state.set_flags(thinking=False)
            # The scan owns the microphone until joined; whoever catches this
            # exception may listen immediately.
            if scan is not None:
                scan.finish()
            while True:
                try:
                    feed.get_nowait()
                except queue.Empty:
                    break
            raise
        # A reply that ends by ASKING something opens a short wake-word-free
        # window for the answer. Observed live: the robot finished with "What
        # do you want to be quizzed on...?" and the visitor's answer needed a
        # wake word the question never told them to say. The window is set
        # here, in the one place every generated reply finishes, because the
        # demo that called reply() has no idea the model chose to end on a
        # question. Noise discrimination stays where it already lives: the
        # runner's confidence gate still applies to whatever arrives.
        if spoken and spoken[-1].rstrip().endswith("?"):
            self.state.expect_answer(ANSWER_WINDOW_S)
        elif spoken:
            # Every OTHER reply opens a follow-up window: wake-word-free, but
            # with the ambient floor still applied -- nothing was asked, so a
            # stray syllable is the room. Before this, follow-ups needed "hey
            # Reachy" at exactly the moment people think of them.
            self.state.invite_followup(FOLLOW_UP_WINDOW_S)

        self.state.set_flags(thinking=False)
        self.motion.express_move(final_tag)
        # Leave the body in the persona's resting pose. Without this a reply
        # ends on whatever the last sentence's tag was -- in practice
        # "thinking", which brain/interface.py hardcodes for every streamed
        # sentence -- so the robot sat pondering after every answer whatever
        # manner it was in. The personality demo restores its own.
        if persona is not None:
            self.motion.express(persona.pose)
        return " ".join(spoken)

    # --- listening -------------------------------------------------------

    def listen(self, wait_for_speech_s: Optional[float] = None) -> str:
        """Transcribe one utterance. Returns "" if nothing was understood.

        `wait_for_speech_s` bounds only the wait for speech to BEGIN -- once
        someone is talking, the recogniser's endpoint finishes the sentence.
        Pass it from any hook that listens directly: without a bound, a visitor
        who walks away mid-exchange leaves the robot holding the microphone for
        the full utterance ceiling, unswitchable the whole time.
        """
        self._require_loop_thread("listen")
        self._stop_if_switched()
        self.state.set_flags(listening=True)
        try:
            heard = self.audio.listen(wait_for_speech_s=wait_for_speech_s) or ""
        finally:
            self.state.set_flags(listening=False)
        heard = heard.strip()
        if heard:
            self.state.add("heard", heard)
        if heard and _is_stop_phrase(heard):
            # The core phrases have to work here too. An inline exchange --
            # ask() and everything built on it -- never reaches the runner's
            # dispatch, so for its whole duration "go to sleep" was just an
            # answer the demo could not make sense of: in enrolment it read as
            # a decline, or as a name that failed validation and got the
            # question asked again. Queued rather than raised so the demo
            # finishes its slice and the runner acts on it next time round.
            self.state.request("sleep")
            return ""
        return heard

    def ask(
        self,
        question: str,
        emotion: str = "curious",
        *,
        wait_for_speech_s: Optional[float] = None,
    ) -> str:
        """Say something and listen for the answer. "" if nothing came back.

        The question is deliberately not interruptible. This is say-then-listen
        in one call, so an Interrupted raised by the question would abandon the
        exchange at the exact moment it was about to listen -- and the visitor
        answering promptly is precisely what would trigger it.
        """
        self.say(question, emotion, interruptible=False)
        return self.listen(wait_for_speech_s=wait_for_speech_s)

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
        """The recognised person, or 0 for an unknown visitor.

        Sticky across camera dropouts: the tracker forgets a face 3 seconds
        after it leaves frame, but a conversation does not change owners
        because the robot turned its head. While NOBODY is in view, the last
        recognised person keeps the conversation for STICKY_IDENTITY_S. A face
        that IS in view always wins -- recognised, it takes over; unrecognised
        past the tracker's own hold, it is a new visitor and the sticky id is
        dropped rather than letting them inherit someone else's history.
        """
        if self.tracker is None or not self.tracker.enabled:
            return 0
        person_id, face = self.tracker.current()
        pid = int(person_id or 0)
        now = time.monotonic()
        if pid:
            _STICKY.person_id = pid
            _STICKY.seen_at = now
            return pid
        if face is not None:
            forget_identity()
            return 0
        if _STICKY.person_id and now - _STICKY.seen_at < STICKY_IDENTITY_S:
            return _STICKY.person_id
        return 0

    def person_name(self) -> Optional[str]:
        """The recognised visitor's name, or None for a stranger.

        None covers three different things that all mean "do not use a name":
        face features are off, nobody is in view, or the person in view has
        never told the robot who they are.

        This is the name to SAY -- a pronunciation the visitor gave the robot
        wins over the spelling, because that is what it is for. Anything that
        needs the real spelling (matching a correction, showing it on the
        dashboard) should read db.get_person_name directly.
        """
        person_id = self.person_id()
        if not person_id:
            return None
        from brain import db

        return db.get_spoken_name(person_id)

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
    #: When True, a transcript the recogniser had no confidence in is dropped
    #: SILENTLY instead of answered with "Sorry -- I did not catch that. Say
    #: it again?". For performance demos that hold the mic through a live
    #: event: the mic hears the whole room, most of what arrives is not a cue,
    #: and the robot asking the audience to repeat itself mid-performance is
    #: worse than missing one cue the presenter can simply say again.
    quiet_when_unsure: ClassVar[bool] = False

    #: The manner this demo asks to be run in, by persona id (see
    #: brain/personas.py), or "" for the robot's own. A default, not a setting:
    #: the operator's dropdown overrides it, and entering a demo snaps back to
    #: it. A welcome is warmer than a brainstorm and neither should need saying
    #: out loud, but whoever is standing there always gets the last word.
    persona: ClassVar[str] = ""

    #: Whether this demo supplies its own style brief and wants the robot's
    #: standing manner kept out of its replies. Almost always False: the
    #: personality demo sets it because it passes a fuller brief of its own,
    #: and two style instructions in one prompt pull against each other.
    #:
    #: A flag rather than the core checking for a demo by name, which is what
    #: it used to do. That broke three ways: renaming the file silently stopped
    #: the exclusion matching, deleting it left a dead condition, and no other
    #: demo could opt out without editing a core file -- the one thing adding a
    #: demo is supposed never to require.
    owns_persona: ClassVar[bool] = False

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
