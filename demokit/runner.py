"""Driving the selected demo, one short slice at a time.

This is the loop the robot actually lives in, and the contract it enforces is
that no demo can take the robot away from its operator. Every hook runs inside
a guard, every listening window is bounded by the core rather than by the demo,
and a demo that keeps failing is set aside with a note on the dashboard instead
of being retried forever in front of visitors.

Dispatch order for something a visitor said, most specific first:

1.  Core phrases -- "go to sleep" and friends. Never overridable: whatever a
    demo is doing, the robot must always be able to stop.
2.  The active demo, if it declares claims_utterances. A demo running a
    question-and-answer sequence has to see the answer before anyone else,
    otherwise "dance studios in Dublin" makes the robot dance in the middle of
    a brainstorm.
3.  Any demo's trigger phrase, longest first, which switches to that demo.
    Longest-first so "tell me a story" wins over a bare "story".
4.  The active demo's on_utterance.
5.  Conversation, which is what the robot does when nothing else claimed it.
"""

import logging
import re
import time
from typing import Any, Optional

from body.face import clean_spoken_name
from brain import db

from .base import Demo, DemoContext, DemoStopped, IdleResult, Interrupted
from .registry import FALLBACK_ID, REGISTRY


logger = logging.getLogger(__name__)

#: Spoken ways of saying "stop". Kept here in the core rather than in a demo:
#: shutting down has to work whatever is selected, and has to be instant rather
#: than a generated decision that might arrive three seconds later.
#:
#: Matched on whole words, and deliberately without "that's all" -- it used to
#: be here, and because these are checked before anything else and cannot be
#: overridden, "that's all really impressive, thank you" put the robot to sleep
#: in the middle of a visit. Every phrase left is one nobody says by accident
#: about anything other than the robot.
SLEEP_PHRASES = (
    "go to sleep",
    "goodbye",
    "good bye",
    "turn off",
    "shut down",
    "stop listening",
)

#: Spoken ways of saying "you have my name wrong". Also in the core, because
#: the robot now greets people by name from any demo, so the correction has to
#: work from any demo too -- being misnamed in three demos and able to fix it
#: in only one is worse than not being greeted at all.
#:
#: Written without apostrophes because _word_stream strips them, and matched on
#: whole words like everything else here. Only ever consulted for a visitor the
#: robot already has a name for, which is what keeps "call me" from being a
#: hazard: for a stranger these do nothing.
NAME_CORRECTION_CUES = (
    "not my name",
    "wrong name",
    "my name is not",
    "my name is",
    "my names",
    "call me",
)

#: The cues above that deny a name rather than supply one. What follows these
#: is the name being rejected -- "my name is not Telaget" -- so reading the
#: trailing words as the new name sets it to precisely the thing the visitor
#: just said was wrong. Found once by testing that exact sentence.
NEGATIVE_NAME_CUES = frozenset({"not my name", "wrong name", "my name is not"})

#: Words that introduce the replacement, after the denial: "that's not my name,
#: it's Tadhagath". Not correction cues in their own right -- "it is" opens far
#: too many ordinary sentences -- they only decide where the name starts once a
#: cue above has already established that a correction is happening.
NAME_HANDOFF_CUES = ("its", "it is", "im", "i am", "actually")


#: How long the mic stays open after an exchange, and how long each open-mic
#: listen waits for someone to start speaking before handing the cycle back.
#: The window is generous because thinking of the next question takes longer
#: than people expect, and it costs nothing to leave open -- the robot only
#: acts on speech it actually hears.
_OPEN_MIC_WINDOW_S = 30.0
_OPEN_MIC_SILENCE_S = 5.0

#: A leading wake phrase, for stripping off an open-mic follow-up. Written as a
#: pattern rather than a list of the spotter's keywords because the two are not
#: the same job: the spotter needs every spelling it might hear, while this only
#: needs to recognise that a sentence opened by addressing the robot. Matching
#: the shape covers spellings nobody has added to the keywords file yet, and a
#: miss here is cosmetic anyway -- the model sees a stray "hey reachy" -- so the
#: looser rule costs nothing and the stricter one would need maintaining twice.
_WAKE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:hey|hi|hello|ok|okay|attention)\s+)?(?:there\s+)?"
    r"(?:reachy|reachie|reechy|retchy|ricky|richie|ritchie|richy)\b[\s,.!?]*",
    re.IGNORECASE,
)


def _strip_wake_phrase(text: str) -> str:
    """Drop a leading wake phrase from something said with the mic already open.

    Only ever strips a *leading* one, so "what does hey reachy mean" survives
    intact -- somebody asking about the wake phrase is asking a real question.
    """
    stripped = _WAKE_PREFIX_RE.sub("", text, count=1).strip()
    # An utterance that was nothing but the robot's name is not a question, and
    # returning "" tells the caller so rather than sending an empty prompt.
    return stripped


def _word_stream(text: str) -> str:
    """Lowercased words, space-separated and space-padded, for phrase matching.

    Phrases have to be matched on word boundaries rather than as raw
    substrings, and doing it here means every demo gets it for free. Matched
    raw, "hub" fires inside "github", "robot" inside "robotics", "brainstorm"
    inside "brainstorming", and "dance" inside "dance studios in Dublin" --
    each of which hijacks whatever conversation was actually happening. Ten
    separate instances of that showed up across the demos, which is the signal
    that it belongs in the core and not in each demo's own guesswork.

    Apostrophes are dropped rather than kept, so "let's" and "lets" are the
    same word. The recognizer emits no punctuation at all -- a transcript is
    "LETS DANCE", never "LET'S DANCE" -- so a trigger written the way a person
    would type it would otherwise never match anything the robot actually
    hears. Demos currently work around that by listing both spellings; they
    should not have to, and the next one to be written would not know to.
    """
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))) + " "


def contains_phrase(word_stream: str, phrase: str) -> bool:
    """Whether a space-padded word stream contains `phrase` as whole words."""
    normalised = _word_stream(phrase).strip()
    return bool(normalised) and f" {normalised} " in word_stream

#: How long a visitor has to say something after the wake word before the robot
#: gives up and goes back to waiting.
_EMPTY_TRANSCRIPT_RETRIES = 1

#: How many times a visitor may talk over the robot before the turn ends and
#: they have to say the wake word again. Each interruption resumes inside the
#: turn it interrupted, so this bounds the stack rather than their patience;
#: nobody reaches it in conversation.
_MAX_INTERRUPT_DEPTH = 5


class DemoRunner:
    """Runs whichever demo is selected, and survives whatever it does."""

    def __init__(self, *, audio: Any, motion: Any, tracker: Any, state: Any, capabilities: frozenset[str]) -> None:
        self._audio = audio
        self._motion = motion
        self._tracker = tracker
        self._state = state
        self._capabilities = capabilities
        self._active_id: Optional[str] = None
        self._active_demo: Optional[Demo] = None
        self._ctx: Optional[DemoContext] = None
        self._stores: dict[str, dict] = {}
        #: While now is under this, follow-ups need no wake word. Zero means a
        #: conversation has not been opened yet, so the switch being on changes
        #: nothing until somebody says the wake word once.
        self._open_until = 0.0

    # --- the loop --------------------------------------------------------

    def cycle(self) -> None:
        """One slice: service the dashboard, run the demo, maybe take a turn."""
        if self._handle_dashboard_request():
            return
        if self._state.sleeping:
            self._wait_while_asleep()
            return

        demo, ctx = self._active()
        if demo is None or ctx is None:
            self._audio.wait_for_wake_word(timeout=1.0)
            return

        result = self._guarded(demo, ctx, "on_idle", lambda: demo.on_idle(ctx))
        result = IdleResult.sanitised(result, demo.id)

        if result.listen_for <= 0.0:
            return
        # Only when the demo is waiting rather than part-way through saying
        # something: listen_for above zero is the demo telling us it has
        # nothing in hand, which is the one safe moment to speak over nothing.
        self._greet_if_recognised(ctx)

        # One listen per cycle either way, so the dashboard stays responsive and
        # a visitor can still be switched away from or put to sleep mid-chat.
        if self._state.open_mic and time.monotonic() < self._open_until:
            if self._open_mic_turn(demo, ctx):
                self._open_until = time.monotonic() + _OPEN_MIC_WINDOW_S
            return

        if self._audio.wait_for_wake_word(timeout=result.listen_for):
            self._take_turn(demo, ctx)
            if self._state.open_mic:
                self._open_until = time.monotonic() + _OPEN_MIC_WINDOW_S

    def _greet_if_recognised(self, ctx: DemoContext) -> None:
        """Say hello, by name, the first time a known face appears.

        In the core rather than in a demo because being recognised should feel
        the same wherever the robot happens to be -- somebody who was greeted
        by name in the vision demonstration and then ignored by every other one
        has learned that the recognition was a trick of that screen.

        Once per person per session. The robot has no way to tell "came back
        after lunch" from "stepped out of frame for two seconds", and of the
        two mistakes available, greeting someone every time they turn their
        head is much the worse one.
        """
        # The greeting ledger lives in state (see RobotState.mark_greeted):
        # enrolment spends the greeting on its own thank-you, and without a
        # shared record this would greet the person who was just enrolled.
        name = ctx.person_name()
        if not name or not self._state.mark_greeted(name):
            return
        logger.info("Recognised %s.", name)
        self._motion.express("happy")
        ctx.say(f"Oh, hello again {name}.", "happy")

    def _open_mic_turn(self, demo: Demo, ctx: DemoContext) -> bool:
        """Take a follow-up question with no wake word. True if one was heard.

        The wake word still opens a conversation; this only keeps it open
        afterwards. Listening permanently was the other way to read the
        request and is the wrong one in a room like this -- the AGC's own notes
        record what happens when background speech reaches the recognizer, and
        a robot answering a conversation it merely overheard is worse than one
        that needs addressing by name.
        """
        self._state.note("status", "Listening -- no wake word needed")
        heard = self._listen(wait_for_speech_s=_OPEN_MIC_SILENCE_S)
        if not heard:
            return False

        # People keep saying it out of habit for the first minute, and it
        # arrives as ordinary words now that nothing is filtering it out.
        # Left in, "hey reachy what is XR" reaches the model as a question
        # about its own name.
        heard = _strip_wake_phrase(heard)
        if not heard:
            return False
        self._dispatch(demo, ctx, heard)
        return True

    def _correct_name_if_asked(self, ctx: DemoContext, words: str, heard: str) -> bool:
        """Let somebody the robot already knows fix the name it knows them by.

        Speech-to-text is at its worst on proper nouns, which is precisely what
        enrolment stores: a visitor here was recorded as "Telaget". Before this,
        the robot then greeted them by that name at every future visit and the
        only remedy was an UPDATE against the database by hand -- which the
        person being misnamed is in no position to run.

        Only for visitors who already have a name. A stranger saying "my name
        is..." is a first enrolment and belongs to the vision demo, which asks
        consent first; catching it here would bind a name to a face without
        ever asking. That restriction is also what makes the looser cues safe
        to accept, since the worst case is overwriting a name that can itself
        be corrected the same way a moment later.
        """
        person_id = ctx.person_id()
        if not person_id or not ctx.person_name():
            return False
        if not any(contains_phrase(words, cue) for cue in NAME_CORRECTION_CUES):
            return False

        # Cut at the LAST marker, and read `words` rather than the raw
        # transcript so apostrophes and punctuation are already gone. Both
        # matter for the sentence people actually say when correcting a robot,
        # which states the wrong name before the right one: "my name is not
        # Telaget, it's Tadhagath" cut at the first cue gives "Telaget Its
        # Tadhagath", and cut at the last gives the name.
        last_cue, at = "", -1
        for cue in NAME_CORRECTION_CUES + NAME_HANDOFF_CUES:
            found = words.rfind(f" {cue} ")
            if found > at:
                last_cue, at = cue, found

        # A denial with nothing after it names only what is wrong, so there is
        # no new name to take -- and taking those words anyway would set the
        # name to the one the visitor just rejected.
        tail = "" if last_cue in NEGATIVE_NAME_CUES else words[at + len(last_cue) + 2 :]
        name = clean_spoken_name(tail)
        if name is None:
            ctx.say("Sorry about that. Say hey Reachy, then my name is, and your name.", "sad")
            return True

        was = ctx.person_name()
        if name == was:
            return False
        db.rename_person(person_id, name)
        if was:
            self._state.forget_greeted(was)
        self._state.mark_greeted(name)
        logger.info("Renamed person %s from %r to %r.", person_id, was, name)
        ctx.say(f"Sorry about that. {name}. I have it right now.", "happy")
        return True

    # --- turn handling ---------------------------------------------------

    def _take_turn(self, demo: Demo, ctx: DemoContext, depth: int = 0) -> None:
        """Wake word heard: transcribe, then decide who handles it.

        `depth` counts interruptions chained without returning to the idle
        loop. Someone can talk over the robot repeatedly -- which is fine, and
        the whole point -- but each interruption resumes inside the previous
        turn, so without a bound a determined visitor grows the stack until it
        breaks. Past the limit the turn simply ends and the next wake word is
        picked up by the idle loop as usual, which costs one repetition and
        nothing else.
        """
        self._motion.acknowledge()
        heard = ""
        for _ in range(1 + _EMPTY_TRANSCRIPT_RETRIES):
            heard = self._listen()
            if heard:
                break
        if not heard:
            logger.info("Heard the wake word but nothing after it.")
            self._state.note("status", "Didn't catch that")
            return
        self._dispatch(demo, ctx, heard, depth)

    def _dispatch(self, demo: Demo, ctx: DemoContext, heard: str, depth: int = 0) -> None:
        """Decide who handles something a visitor said.

        Split out from _take_turn so open mic can reach it: a follow-up asked
        without the wake word has to be routed by exactly the same rules, or
        "go to sleep" would stop working the moment the mic stayed open.
        """
        words = _word_stream(heard)

        if any(contains_phrase(words, phrase) for phrase in SLEEP_PHRASES):
            self._go_to_sleep()
            return

        # Ahead of the demos, for the same reason sleep is: the robot spoke the
        # wrong name, so the correction is about the robot rather than about
        # whatever demo is running, and it has to land whichever one that is.
        if self._correct_name_if_asked(ctx, words, heard):
            return

        if demo.claims_utterances and self._offer(demo, ctx, heard):
            return

        offered = False
        switched = self._switch_on_trigger(words)
        if switched is not None:
            demo, ctx = switched
            # The demo a trigger phrase named gets the utterance too, so
            # "tell me a story" both selects the storyteller and asks for one.
            if self._offer(demo, ctx, heard):
                return
            offered = True

        # Not offered twice. Without the flag, a demo that was just switched to
        # and declined the utterance was handed the identical text again on the
        # next line -- harmless for a demo that only reads it, but any demo that
        # counts, appends or advances a script did it twice for one thing said.
        if not offered and not demo.claims_utterances and self._offer(demo, ctx, heard):
            return

        self._converse(ctx, heard, depth)

    def _offer(self, demo: Demo, ctx: DemoContext, heard: str) -> bool:
        handled = self._guarded(demo, ctx, "on_utterance", lambda: demo.on_utterance(ctx, heard))
        return bool(handled)

    def _converse(self, ctx: DemoContext, heard: str, depth: int = 0) -> None:
        """What the robot does when no demo claimed the utterance."""
        try:
            ctx.reply(heard, person_id=ctx.person_id())
        except Interrupted:
            # The visitor talked over the answer. Their new question is already
            # waiting, so take it now rather than making them repeat it.
            if depth >= _MAX_INTERRUPT_DEPTH:
                logger.info("Interruption limit reached; back to the idle loop.")
                return
            demo_now = self._active_demo
            if demo_now is not None:
                self._take_turn(demo_now, ctx, depth + 1)
        except DemoStopped:
            logger.info("Reply cut short: demo switched.")
        except Exception:
            logger.exception("Conversational reply failed")
            self._state.add("error", "I lost my train of thought there.")

    def _listen(self, wait_for_speech_s: Optional[float] = None) -> str:
        self._state.set_flags(listening=True)
        try:
            heard = (self._audio.listen(wait_for_speech_s=wait_for_speech_s) or "").strip()
        except Exception:
            logger.exception("Listening failed")
            return ""
        finally:
            self._state.set_flags(listening=False)
        if heard:
            self._state.add("heard", heard)
        return heard

    # --- demo selection --------------------------------------------------

    def _active(self) -> tuple[Optional[Demo], Optional[DemoContext]]:
        """The selected demo, entering it if the operator just switched."""
        wanted = self._state.mode
        if wanted == self._active_id and self._active_demo is not None:
            return self._active_demo, self._ctx

        available, reason = REGISTRY.is_available(wanted, self._capabilities)
        if not available:
            fallback = REGISTRY.default_id()
            if fallback != wanted and REGISTRY.get(fallback) is not None:
                logger.warning("Demo %r unavailable (%s); using %r", wanted, reason, fallback)
                self._state.add("error", f"{wanted} is unavailable ({reason})")
                self._state.set_mode(fallback)
                wanted = fallback
            else:
                return None, None

        self._leave_active()
        demo = REGISTRY.get(wanted)
        if demo is None:
            return None, None

        store = self._stores.setdefault(demo.id, {})
        ctx = DemoContext(
            audio=self._audio,
            motion=self._motion,
            tracker=self._tracker,
            state=self._state,
            demo_id=demo.id,
            store=store,
        )
        self._active_id, self._active_demo, self._ctx = wanted, demo, ctx
        logger.info("Demo: %s", demo.label)
        self._guarded(demo, ctx, "on_enter", lambda: demo.on_enter(ctx))
        return demo, ctx

    def _leave_active(self) -> None:
        if self._active_demo is None or self._ctx is None:
            return
        demo, ctx = self._active_demo, self._ctx
        # A context that has already noticed the switch would refuse to speak,
        # so on_exit gets one bound to the mode it is leaving.
        ctx._entered_mode = self._state.mode
        self._guarded(demo, ctx, "on_exit", lambda: demo.on_exit(ctx))
        self._active_id = self._active_demo = self._ctx = None

    def _switch_on_trigger(self, words: str) -> Optional[tuple[Demo, DemoContext]]:
        """Switch demo if the transcript contains a trigger phrase.

        `words` is the space-padded word stream from _word_stream, so a trigger
        only fires on whole words.
        """
        for phrase, demo_id in self._trigger_table():
            if not contains_phrase(words, phrase):
                continue
            if demo_id == self._active_id:
                return None
            available, reason = REGISTRY.is_available(demo_id, self._capabilities)
            if not available:
                logger.info("Trigger %r ignored: %s is %s", phrase, demo_id, reason)
                continue
            self._state.set_mode(demo_id)
            demo, ctx = self._active()
            if demo is None or ctx is None:
                return None
            return demo, ctx
        return None

    def _trigger_table(self) -> list[tuple[str, str]]:
        """(phrase, demo_id) pairs, longest phrase first."""
        pairs: list[tuple[str, str]] = []
        for demo_id in REGISTRY.ids():
            demo = REGISTRY.get(demo_id)
            if demo is None:
                continue
            for phrase in demo.triggers:
                pairs.append((phrase.lower(), demo_id))
        pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
        return pairs

    # --- core behaviours -------------------------------------------------

    def _handle_dashboard_request(self) -> bool:
        """Speak or listen on the operator's behalf. True if something was done."""
        request = self._state.pop_request()
        if request is None:
            return False
        kind, text, tag = request
        if kind == "say":
            logger.info("Dashboard say: %r", text)
            self._state.add("said", text)
            self._motion.express(tag)
            self._audio.speak(text, tag, motion=self._motion)
            return True
        if kind == "listen":
            logger.info("Dashboard listen-now pressed.")
            self._state.set_sleeping(False)
            demo, ctx = self._active()
            if demo is not None and ctx is not None:
                self._take_turn(demo, ctx)
            return True
        if kind == "voice":
            # Applied here rather than in the web handler because loading a
            # voice swaps the object every utterance goes through, and doing
            # that from another thread mid-sentence is the same class of fault
            # as speaking from it.
            if self._audio.set_voice(text):
                self._state.add("status", f"Voice set to {text}")
                self._audio.speak("This is my voice now.", "happy", motion=self._motion)
            else:
                self._state.add("error", f"Could not switch to voice {text}")
            return True
        return False

    def _wait_while_asleep(self) -> None:
        """Asleep: the wake word is the only thing that gets a response."""
        if not self._audio.wait_for_wake_word(timeout=2.0):
            return
        self._state.set_sleeping(False)
        self._motion.acknowledge()
        self._motion.express("happy")
        self._audio.speak("I'm awake.", "happy", motion=self._motion)

    def _go_to_sleep(self) -> None:
        logger.info("Going to sleep.")
        self._motion.express("neutral")
        self._audio.speak("Alright, going quiet. Say hey Reachy when you need me.", "neutral", motion=self._motion)
        self._state.set_sleeping(True)
        self.end_conversations()

    def end_conversations(self) -> None:
        """Close out every conversation this session has held.

        Being told to go quiet is the one unambiguous end-of-visit signal the
        robot gets, so it is where a conversation is summarised into long-term
        memory and the in-session history cleared. Nothing called this after
        the demo refactor and no demo picked it up, because no demo owns a
        person's session boundary -- the result was that every unrecognised
        visitor kept appending to person 0's history all day, so visitors read
        each other's conversations back, and nothing was ever written to the
        long-term store the SQLite schema exists for.
        """
        from brain import memory
        from brain.interface import end_conversation

        for person_id in list(memory.known_people()):
            try:
                end_conversation(person_id)
            except Exception:
                logger.exception("Could not close out the conversation for person %s", person_id)

    # --- the guard -------------------------------------------------------

    def _guarded(self, demo: Demo, ctx: DemoContext, hook: str, call) -> Any:
        """Run one demo hook, absorbing whatever it does wrong.

        A demo is written by a student, possibly the week before an industry
        visit. The framework's job is that a mistake in one costs that demo and
        nothing else: not the voice loop, not the other demos, and not the
        robot's ability to be switched to something that works.
        """
        started = time.monotonic()
        try:
            result = call()
        except DemoStopped:
            # Not an error: the operator switched away mid-hook and the demo
            # unwound promptly, which is exactly what it should do.
            return None
        except Interrupted:
            # Also not an error: a visitor said the wake word over the top of
            # whatever was being said. Take their turn straight away rather
            # than returning to the idle loop, which would make them say it
            # twice -- once to stop the robot and once to be heard.
            logger.info("Interrupted by the wake word; listening.")
            self._state.note("listen", "Interrupted -- listening")
            demo_now, ctx_now = self._active_demo, self._ctx
            if demo_now is not None and ctx_now is not None:
                self._take_turn(demo_now, ctx_now)
            return None
        except Exception:
            logger.exception("%s.%s failed", demo.id, hook)
            self._state.add("error", f"{demo.label} had a problem ({hook})")
            if REGISTRY.record_failure(demo.id):
                self._state.add("error", f"{demo.label} set aside after repeated failures")
                self._leave_active()
                fallback = REGISTRY.default_id()
                if fallback != demo.id:
                    self._state.set_mode(fallback)
            return None
        REGISTRY.record_success(demo.id)
        elapsed = time.monotonic() - started
        if elapsed > _SLOW_HOOK_S:
            logger.warning(
                "%s.%s took %.1fs; hooks should return quickly so the robot stays switchable",
                demo.id,
                hook,
                elapsed,
            )
        return result


#: A hook slower than this is not broken, but it is holding the robot. Warned
#: rather than enforced: the honest fix is in the demo, and killing a hook
#: mid-speech would be worse than the delay.
_SLOW_HOOK_S = 6.0
