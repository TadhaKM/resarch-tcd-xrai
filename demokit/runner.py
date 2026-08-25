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
#: How long somebody must stand there before the robot offers. Long enough
#: that walking past does not trigger it, short enough that somebody deciding
#: whether to approach is met rather than ignored.
_ATTRACT_AFTER_S = 4.0

#: And how long the room must have been quiet first. The offer is for people
#: who have not realised they can talk to it, never for people mid-conversation.
_ATTRACT_QUIET_S = 20.0

#: What an utterance must clear to count as addressed to the robot when the
#: operator's open-mic switch is on and no demo is waiting for an answer.
#: Measured against what actually went wrong: "EH" and "OH" were answered as
#: questions. Two words is the smallest thing anybody says TO a robot without
#: saying its name first; one syllable is the room.
_AMBIENT_MIN_WORDS = 2
_AMBIENT_MIN_CHARS = 7

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


#: The two-part test for "this trigger phrase, misheard". A single similarity
#: threshold was tried first and measured unable to separate the real cases:
#: "welcome to your irisimus group" (a mangled "welcome the erasmus group",
#: must fire) scored 0.750 -- EXACTLY what "a warm welcome to this group"
#: (must not fire) scored, with "we should welcome the new group" above both
#: at 0.800. The scaffolding words dominate a whole-window ratio. What
#: separates a mishearing from a coincidence is the DISTINCTIVE word: heard
#: "irisimus" resembles "erasmus", while "new" and "this" resemble nothing in
#: the phrase. So the window ratio sets a floor, and every substantial word of
#: the phrase must additionally have its own fuzzy counterpart in the window.
_FUZZY_WINDOW_RATIO = 0.70
_FUZZY_WORD_RATIO = 0.65
#: Words shorter than this are scaffolding ("the", "to", "our") and carry no
#: identity worth testing for.
_FUZZY_WORD_MIN_LEN = 4
#: Only phrases at least this long may match approximately. Short triggers
#: ("lets dance") are one mishearing away from everything, which is the same
#: reason features.py refuses short trigger phrases outright.
_FUZZY_MIN_WORDS = 3
_FUZZY_MIN_CHARS = 12


def _word_evidence(heard: list[str], word: str) -> bool:
    """Whether `word`, possibly misheard and possibly SPLIT, is in `heard`.

    Split matters: "erasmus" arrived as "de arass mis...", three fragments, so
    single heard words can never account for it -- joins of up to three
    consecutive words are counterparts too.
    """
    from difflib import SequenceMatcher

    for size in (1, 2, 3):
        for start in range(len(heard) - size + 1):
            candidate = "".join(heard[start:start + size])
            if SequenceMatcher(None, candidate, word).ratio() >= _FUZZY_WORD_RATIO:
                return True
    return False


def fuzzy_contains(word_stream: str, phrase: str) -> bool:
    """Whether the transcript contains something that IS `phrase`, misheard.

    Character-level similarity over a sliding window of words, because
    mishearings do not respect word boundaries; the window runs from one word
    shorter than the phrase to two longer, which covers a word split into
    pieces without letting the rest of the utterance dilute the comparison.
    See the constants above for why the window ratio alone is not the test.
    """
    from difflib import SequenceMatcher

    words = _word_stream(phrase).split()
    if len(words) < _FUZZY_MIN_WORDS or sum(len(w) for w in words) < _FUZZY_MIN_CHARS:
        return False
    heard_all = word_stream.split()
    substantial = [w for w in words if len(w) >= _FUZZY_WORD_MIN_LEN]
    target = "".join(words)
    low = max(2, len(words) - 1)
    high = min(len(heard_all), len(words) + 2)
    for size in range(low, high + 1):
        for start in range(len(heard_all) - size + 1):
            window = heard_all[start:start + size]
            if SequenceMatcher(None, "".join(window), target).ratio() < _FUZZY_WINDOW_RATIO:
                continue
            if all(_word_evidence(window, w) for w in substantial):
                return True
    return False

#: Pause on an idle slice that asked for no listening. See cycle().
_IDLE_SLICE_S = 0.05

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
        #: Whether the person currently standing there has already been invited
        #: to speak. Cleared when they leave -- see _attract_if_lingering.
        self._attracted = False
        #: When somebody last said something, so the robot never offers over a
        #: conversation that is already happening.
        self._last_heard_at = 0.0
        self._motion = motion
        self._tracker = tracker
        self._state = state
        # The hardware baseline, kept only as a fallback. What actually
        # gates a demo is read live from the state -- an operator can switch a
        # capability on mid-visit (research mode), and a set frozen here would
        # refuse the very demo the dashboard is showing as available.
        self._capabilities = capabilities
        self._active_id: Optional[str] = None
        self._active_demo: Optional[Demo] = None
        self._ctx: Optional[DemoContext] = None
        self._stores: dict[str, dict] = {}
        #: Interruptions currently nested inside one another. See _guarded.
        self._interrupt_depth = 0
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
            # A breath, not a wait. Zero means "call me again promptly" -- a
            # demo chaining two halves of one exchange wants the next slice now
            # -- but the loop above this has no sleep of its own, so returning
            # bare turned a demo that always says zero (the missing-return
            # mistake, and the storyteller between lines) into a spin that
            # starves the motion thread of the GIL. Short enough that a chained
            # dialogue still feels immediate.
            time.sleep(_IDLE_SLICE_S)
            return
        # Only when the demo is waiting rather than part-way through saying
        # something: listen_for above zero is the demo telling us it has
        # nothing in hand, which is the one safe moment to speak over nothing.
        self._greet_if_recognised(ctx)
        # After the greeting, so a recognised visitor is welcomed by name
        # rather than invited to introduce themselves.
        self._attract_if_lingering(ctx)

        # One listen per cycle either way, so the dashboard stays responsive and
        # a visitor can still be switched away from or put to sleep mid-chat.
        # Wake-word-free listening happens for either of two reasons: the
        # operator's open-mic switch, or the robot's own last reply having
        # ended in a question (state.answer_expected) -- asking "want to hear
        # more?" and then demanding a wake word before the yes is a trap.
        if (self._state.open_mic and time.monotonic() < self._open_until) \
                or self._state.answer_expected:
            if self._open_mic_turn(demo, ctx, listen_for=result.listen_for):
                if self._state.open_mic:
                    self._open_until = time.monotonic() + _OPEN_MIC_WINDOW_S
            return

        if self._audio.wait_for_wake_word(timeout=result.listen_for):
            # Interrupted can now reach here from any ctx.say inside the turn,
            # and cycle() is outside every guard -- unhandled it would end the
            # slice as a "Cycle failed" error in front of visitors. Handled the
            # same way _guarded does: the visitor's words are already waiting,
            # so take the turn again rather than making them repeat themselves.
            try:
                self._take_turn(demo, ctx)
            except Interrupted:
                self._retake_after_interruption(demo, ctx)
            if self._state.open_mic:
                self._open_until = time.monotonic() + _OPEN_MIC_WINDOW_S

    def _retake_after_interruption(self, demo: Demo, ctx: DemoContext) -> None:
        """Give the floor to whoever talked over the robot.

        Their question is already in the microphone backlog, so this takes the
        turn straight away rather than returning to the idle loop, which would
        make them say it twice -- once to stop the robot and once to be heard.

        The depth counter is the runner's rather than a parameter because an
        interruption surfaces from inside an arbitrary hook and there is no
        call chain to thread one through. Each interruption resumes inside the
        turn it interrupted, so without a bound a visitor talking over every
        answer grows the stack until it breaks. Latent until now: this path was
        unreachable from the default demo, which was swallowing Interrupted.
        """
        if self._interrupt_depth >= _MAX_INTERRUPT_DEPTH:
            logger.info("Interruption limit reached; back to the idle loop.")
            return
        logger.info("Interrupted by the wake word; listening.")
        self._state.note("listen", "Interrupted -- listening")
        self._interrupt_depth += 1
        try:
            self._take_turn(demo, ctx)
        except Interrupted:
            self._retake_after_interruption(demo, ctx)
        finally:
            self._interrupt_depth -= 1

    def _attract_if_lingering(self, ctx: DemoContext) -> bool:
        """Invite somebody who has stood there a while to say something.

        A robot sitting still gets walked past. The people this is for are the
        ones who stop, look at it, and do not know they are allowed to talk to
        it -- which at an open day is most of them.

        Only for STRANGERS. Somebody the robot knows already gets a greeting by
        name from _greet_if_recognised, and following that with "ask me
        something" is the robot talking at a person twice before they have said
        a word.

        Once per arrival, and an arrival ends when the face has been gone for
        _PRESENCE_GAP_S. The robot cannot tell "came back after lunch" from
        "turned their head", and of the two mistakes available, inviting the
        same person every four seconds is much the worse one -- that is a robot
        nagging a room, which is precisely what makes staff turn a feature off.
        """
        if self._tracker is None:
            return False
        # Never over a conversation. Somebody who has spoken recently knows
        # perfectly well that they can talk to it.
        if time.monotonic() - self._last_heard_at < _ATTRACT_QUIET_S:
            return False
        try:
            dwell = self._tracker.present_for()
        except Exception:
            return False
        if dwell < _ATTRACT_AFTER_S:
            # Nobody there, or not there long enough. Clear the latch so the
            # next person to arrive gets their own invitation.
            if dwell <= 0.0:
                self._attracted = False
            return False
        if self._attracted:
            return False
        if ctx.person_name():
            # Known: the greeting covers them. Latch anyway so this does not
            # re-check them every slice.
            self._attracted = True
            return False

        self._attracted = True
        logger.info("Someone has been standing there %.0fs -- offering.", dwell)
        # Perk up first: the movement is what makes a person look, and a line
        # spoken by a motionless robot reads as a recording.
        try:
            self._motion.express("curious")
        except Exception:
            logger.debug("Could not perk up", exc_info=True)
        # Not interruptible, for the same reason the greeting is not: this runs
        # outside _guarded, so an Interrupted would surface as "Cycle failed".
        ctx.say("Hello -- say Hey Reachy, and ask me something.",
                "happy", interruptible=False)
        return True

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
        # Not interruptible: cycle() calls this outside _guarded, so an
        # Interrupted raised here would leave the runner entirely and surface
        # to the operator as "Cycle failed". It is four words; nobody needs to
        # talk over it, and the wake-word window opens immediately after.
        ctx.say(f"Oh, hello again {name}.", "happy", interruptible=False)

    def _open_mic_turn(self, demo: Demo, ctx: DemoContext,
                       listen_for: float = _OPEN_MIC_SILENCE_S) -> bool:
        """Take a follow-up question with no wake word. True if one was heard.

        The wake word still opens a conversation; this only keeps it open
        afterwards. Listening permanently was the other way to read the
        request and is the wrong one in a room like this -- the AGC's own notes
        record what happens when background speech reaches the recognizer, and
        a robot answering a conversation it merely overheard is worse than one
        that needs addressing by name.
        """
        self._state.note("status", "Listening -- no wake word needed")
        # Capped at the demo's own listening window. A demo mid-script returns
        # a short listen_for to say "I have more lines to deliver", and the old
        # fixed five-second wait here stretched every scripted sequence by five
        # silent seconds per line -- measured live as fourteen-second gaps
        # between quiz questions with open mic on.
        heard = self._listen(wait_for_speech_s=min(_OPEN_MIC_SILENCE_S, listen_for))
        if not heard:
            return False

        # People keep saying it out of habit for the first minute, and it
        # arrives as ordinary words now that nothing is filtering it out.
        # Left in, "hey reachy what is XR" reaches the model as a question
        # about its own name.
        heard = _strip_wake_phrase(heard)
        if not heard:
            return False
        if not self._addressed_to_the_robot(heard):
            return False
        self._dispatch(demo, ctx, heard)
        return True

    def _addressed_to_the_robot(self, heard: str) -> bool:
        """Whether an open-mic fragment is somebody TALKING TO the robot.

        With the switch on in a live room, the recogniser returns whatever the
        room produces. Live, "EH" and "OH" -- two syllables of somebody
        reacting, or the tail of the robot's own speech coming back through the
        microphone -- were dispatched as questions and answered out loud: "Ha,
        sounds like a reaction!". A robot holding up its end of a conversation
        nobody is having is the exact failure open mic must not have.

        ONLY for the operator's switch. A demo HOLDING the mic has just asked
        a question, so "yes" or "camera" is precisely what it should hear, and
        applying this there would break every quiz answer. The two cases are
        indistinguishable through open_mic, which is why open_mic_held exists.

        Nothing here is a judgement about meaning -- it is a floor on how much
        was said. A single short syllable with no wake word in front of it is
        the room; a sentence is a person.
        """
        if self._state.open_mic_held:
            return True
        # An answer window is the robot having just ASKED something, so a
        # one-word reply is exactly what is expected -- "yes", "true",
        # "purple". The word-count floor is for ambient listening, where a
        # single syllable is the room; here the confidence gate alone decides,
        # and it still runs in _dispatch.
        if self._state.answer_expected:
            return True
        words = _word_stream(heard)
        # Never gate the way out. "Goodbye" is one word and has to work from
        # across a room, whatever else this refuses.
        if any(contains_phrase(words, phrase) for phrase in SLEEP_PHRASES):
            return True
        count = len(words.split())
        if count >= _AMBIENT_MIN_WORDS and len(heard.strip()) >= _AMBIENT_MIN_CHARS:
            return True
        logger.info("open mic: ignoring %r -- too little to be addressed to me", heard[:40])
        return False

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

    def _too_unsure_to_answer(self, ctx: DemoContext, heard: str) -> bool:
        """Whether to ask again instead of answering. Logs the score either way.

        The score is logged on EVERY turn, answered or not, because the
        threshold was chosen from synthesized speech and piper is cleaner than
        a real room -- the live distribution is the thing that will actually
        tune this, and it can only be read off a day's logs if the good turns
        are logged too.
        """
        score = getattr(self._audio, "last_confidence", None)
        if score is None:
            return False
        from body.audio_io import _MIN_MEAN_TOKEN_LOGPROB as floor

        if score >= floor:
            logger.info("transcript confidence %.2f (answering)", score)
            return False
        logger.info("transcript confidence %.2f below %.2f -- asking again: %r",
                    score, floor, heard[:60])
        self._state.add("status", "That did not come through clearly.")
        try:
            ctx.say("Sorry -- I did not catch that. Say it again?", "curious")
        except Exception:
            logger.debug("Could not ask for a repeat", exc_info=True)
        return True

    def _dispatch(self, demo: Demo, ctx: DemoContext, heard: str, depth: int = 0) -> None:
        """Decide who handles something a visitor said.

        Split out from _take_turn so open mic can reach it: a follow-up asked
        without the wake word has to be routed by exactly the same rules, or
        "go to sleep" would stop working the moment the mic stayed open.
        """
        words = _word_stream(heard)
        # Stamped at the one place every utterance passes through, so the
        # attract offer never talks over a conversation already happening.
        self._last_heard_at = time.monotonic()
        # Whatever was said, the question the robot asked has now been
        # responded to -- the answer window must not outlive its answer, or
        # the NEXT stray remark is also treated as one.
        self._state.expect_answer(0.0)

        # Counted here, at the one place every utterance passes through --
        # wake-word turns and open-mic follow-ups alike. Aggregate only: what
        # was asked and how often, never by whom. See brain/stats.py.
        try:
            from brain import stats

            stats.bump("turns")
            stats.note_question(heard)
            if self._active_id:
                stats.bump(f"demo:{self._active_id}")
        except Exception:  # pragma: no cover - a counter must never cost a turn
            logger.debug("Could not record visit stats", exc_info=True)

        if any(contains_phrase(words, phrase) for phrase in SLEEP_PHRASES):
            self._go_to_sleep()
            return

        # Ahead of the demos, for the same reason sleep is: the robot spoke the
        # wrong name, so the correction is about the robot rather than about
        # whatever demo is running, and it has to land whichever one that is.
        if self._correct_name_if_asked(ctx, words, heard):
            return

        # A transcript the recogniser had no confidence in gets asked again
        # rather than answered. Live, Whisper produced "Quizance" for "quiz us",
        # "Hey Ritchie" as the answer to a consent question, and "testing
        # testic" -- each sent to the model and answered confidently, which is
        # the robot at its silliest. The mishearing is a fact of loud rooms;
        # ANSWERING the mishearing is not.
        #
        # BELOW sleep and the name correction, and that ordering is the whole
        # point: both outrank a confidence score. A first attempt put this at
        # the top of dispatch and a test caught what the comment had already
        # claimed was safe -- "go to sleep" shouted across a noisy room scored
        # badly and was answered with "sorry, say that again?", so the one
        # thing that must always work stopped working exactly where it was
        # needed most.
        if self._too_unsure_to_answer(ctx, heard):
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
        # Availability is checked EVERY cycle, not just on a switch. Capabilities
        # change while a demo is running now -- research mode is a dashboard
        # toggle, not a boot-time fact -- and the old fast path returned the
        # running demo without looking, so turning research mode OFF left the
        # research demo running. "Off" has to mean off. The check is a dict
        # lookup and a short list comp, which is nothing against a cycle that
        # does audio.
        available, reason = REGISTRY.is_available(wanted, self._live_capabilities())
        if available and wanted == self._active_id and self._active_demo is not None:
            return self._active_demo, self._ctx

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
            owns_persona=getattr(demo, "owns_persona", False),
        )
        self._active_id, self._active_demo, self._ctx = wanted, demo, ctx
        # Applied before on_enter so the demo's first spoken line is already
        # in its own manner rather than the one before it. Snaps rather than
        # defers: the preset is the intent, and an operator who tried a
        # character once should not find every later demo stuck in it.
        self._state.apply_demo_persona(getattr(demo, "persona", ""))
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

    def _live_capabilities(self) -> frozenset:
        """What the robot can do right now: hardware, plus operator switches."""
        live = getattr(self._state, "capabilities", None)
        return live if live else self._capabilities

    def _switch_on_trigger(self, words: str) -> Optional[tuple[Demo, DemoContext]]:
        """Switch demo if the transcript contains a trigger phrase.

        `words` is the space-padded word stream from _word_stream, so a trigger
        only fires on whole words.
        """
        # Two passes, exact first: a transcript that literally contains one
        # trigger and merely resembles another must go to the literal one,
        # whatever order the table happens to be in. The fuzzy pass exists
        # because the final transcript comes from Whisper, which has no hotword
        # biasing -- a proper noun in a staff-written trigger arrives mangled
        # ("erasmus" came back as "irisimus" and "de-arass mis") and an exact
        # table can never fire on it.
        table = self._trigger_table()
        for approximate in (False, True):
            for phrase, demo_id in table:
                hit = fuzzy_contains(words, phrase) if approximate else contains_phrase(words, phrase)
                if not hit:
                    continue
                if demo_id == self._active_id:
                    return None
                available, reason = REGISTRY.is_available(demo_id, self._live_capabilities())
                if not available:
                    logger.info("Trigger %r ignored: %s is %s", phrase, demo_id, reason)
                    continue
                if approximate:
                    logger.info("Trigger %r matched approximately in %r", phrase, words.strip())
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
        if kind == "sound":
            # On the loop thread, which is the whole reason this is queued
            # rather than played in the web handler: two writers to the same
            # speaker interleave into noise.
            logger.info("Dashboard sound: %r", text)
            self._audio.play_sound(text, self._motion)
            return True
        if kind == "listen":
            logger.info("Dashboard listen-now pressed.")
            self._state.set_sleeping(False)
            demo, ctx = self._active()
            if demo is not None and ctx is not None:
                self._take_turn(demo, ctx)
            return True
        if kind == "sleep":
            # Queued by DemoContext.listen, which is where a sleep phrase said
            # during an inline exchange lands. That exchange runs entirely
            # inside a demo hook, so it never passes _dispatch and the core
            # phrases were unreachable by voice for its whole duration --
            # against the promise at the top of this file that the robot can
            # always be stopped. Routed through the existing queue rather than
            # a new exception so the demo unwinds on its own next slice.
            logger.info("Sleep phrase heard inside a demo exchange.")
            self._go_to_sleep()
            return True
        if kind == "voice":
            # Applied here rather than in the web handler because loading a
            # voice swaps the object every utterance goes through, and doing
            # that from another thread mid-sentence is the same class of fault
            # as speaking from it.
            if self._audio.set_voice(text):
                self._state.add("status", f"Voice set to {text}")
                # Persisted only once it actually loaded and spoke: a choice
                # that failed to apply must not come back at every restart as
                # a warning about a voice that was never installed.
                from brain import settings

                settings.put("voice", text)
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
            #
            # The depth counter is the runner's rather than a parameter,
            # because unlike _converse this path cannot thread one through: the
            # interruption surfaces from inside an arbitrary hook. Without it,
            # each interruption resumes inside the turn it interrupted and a
            # visitor talking over every answer grows the stack until it
            # breaks. That was latent until now -- this branch was unreachable
            # from the default demo, which swallowed Interrupted.
            demo_now, ctx_now = self._active_demo, self._ctx
            if demo_now is not None and ctx_now is not None:
                self._retake_after_interruption(demo_now, ctx_now)
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
