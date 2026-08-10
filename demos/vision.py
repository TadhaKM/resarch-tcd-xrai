"""Follow whoever is in view, and offer -- once -- to learn a new face's name.

The following is not this demo's doing: body/face_tracker.py aims the head at
whoever is in view under every demo, all the time. What this file adds is the
one thing the tracker cannot do by itself, which is put a name to a face, and
it deliberately adds nothing else. An earlier version narrated every arrival
and departure and explained the detection pipeline; watched live, that is a
robot talking over the thing it is meant to be showing. A head that follows you
is legible without a commentary, so most of this file is about not speaking.

The name is asked for and never taken. Voice enrolment was removed from this
project once before because a noisy transcript became a permanent name bound to
a face, so nothing is stored until two separate filters agree: body.face.
extract_spoken_name pulls the most name-like part out of whatever was said
("My name is Sarah, nice to meet you" enrols Sarah, not the sentence), and the
robot then says that name back and waits for a yes. Wrong, the visitor says so
and gives it again -- two more tries before it stops pestering. Someone who
declines, or who says nothing, is not asked again for _ASK_COOLDOWN_S; someone
the robot already recognises is greeted once and never asked at all.

Deliberately absent: any call to ctx.motion.express_move(). A recorded move
pauses the motion loop for its duration (see MotionController._run), and that
loop is what applies the tracking target -- so a flourish would freeze the head
in the one demo whose subject is the head not freezing.
"""

import re
import time
from typing import TYPE_CHECKING

from body.face import extract_spoken_name
from demokit import Demo, DemoContext, IdleResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from body.face_tracker import FaceTracker


#: How stale a detection may be and still count as somebody being here. The
#: tracker holds an aim for 1.5s; a little longer than that, so a couple of
#: missed frames read as a person standing still rather than as one leaving.
_PRESENT_AGE_S = 2.0

#: Unbroken presence before the robot says anything at all. Someone crossing
#: the room is not a visitor, and the offer below is worth making only to
#: somebody who has chosen to stand in front of the robot.
_SETTLE_S = 3.0

#: Silence after an offer, whatever the answer was. Ten minutes rather than
#: one because an unrecognised face is by definition unidentified: there is no
#: id to hang "this one already said no" on, so the cooldown has to cover
#: everybody. Enrolling inside that window is manage_people.py's job.
_ASK_COOLDOWN_S = 600.0

#: Silence between presence remarks of the same kind. A minute: often enough
#: that a visitor deliberately stepping out of frame and back sees the robot
#: notice, rare enough that a group standing around chatting is not narrated
#: at. The previous version of this demo had no limit and was unbearable; the
#: one after it removed the remarks entirely and demonstrated nothing.
_REMARK_COOLDOWN_S = 60.0

#: Wake-word window returned by every idle slice; the core clamps it to
#: MAX_LISTEN_WINDOW_S. Never zero: DemoRunner.cycle opens no microphone at
#: all for a zero, and nothing here is worth being deaf for.
_LISTEN_S = 3.0

#: How long each question in the enrolment exchange waits for the visitor to
#: START answering before handing the turn back. Five seconds covers "um" and a
#: think; once they are talking, the recogniser's endpoint finishes the
#: sentence however long it runs, so this bounds silence rather than answers.
_ANSWER_WAIT_S = 5.0

#: Fresh goes at hearing the name -- the first ask plus retries -- before the
#: robot stops pestering. The visitor asked to be remembered, so giving up on
#: the first mishearing wastes their yes; a fourth attempt is badgering.
_MAX_NAME_ROUNDS = 3

#: Enough of an answer to act on, matched as whole words: "no" sits inside
#: "know" and "now", and reading a no as a yes interrogates somebody who
#: declined. A no anywhere in the answer wins, because "ok, no thanks" is a no.
_YES_WORDS = frozenset({"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "alright"})
_NO_WORDS = frozenset({"no", "nope", "nah", "not", "never", "rather"})
_WORD_RE = re.compile(r"[a-z]+")

#: Enrolment stages, held in ctx.store["stage"]. One bounded exchange per idle
#: slice: each slice says one question and waits at most _ANSWER_WAIT_S for the
#: answer to begin, so the loop gets the robot back between questions and the
#: operator can always switch away.
_ASK_NAME = "ask_name"
_CONFIRM = "confirm"

#: Questions about the camera itself, which is what this demo's own trigger
#: phrase is, so it is the first thing many visitors say on arriving here.
_SEEING_PHRASES = ("see me", "seeing me", "see anyone", "can you see")


def _is_yes(answer: str) -> bool:
    """Whether an answer to a yes/no question was a yes.

    Decided by whichever of yes and no comes FIRST, not by a no anywhere
    overriding: "ok, no thanks" leads with a no and is one, but "sure, why not"
    leads with a yes and is plainly agreement -- read the other way it declines
    on the visitor's behalf and, because the offer is stamped before it is
    made, does not ask again for ten minutes.

    Anything with neither is treated as a no: silence, a shrug, or the
    recognizer picking up the conversation behind the visitor. Wrong that way
    costs an offer; wrong the other way presses a stranger for their name.
    """
    for word in _WORD_RE.findall(answer.lower()):
        if word in _YES_WORDS:
            return True
        if word in _NO_WORDS:
            return False
    return False


class Vision(Demo):
    label = "Vision & Face Tracking"
    help = "Follows whoever is in view, and offers once to learn a new face's name."
    order = 40
    requires = ("faces",)
    #: "face tracking" alone is the name of a topic taught two floors up, so it
    #: fires on "face tracking is part of the vision module" as readily as on a
    #: request for the demonstration. The request forms below cannot be said
    #: about the subject in the abstract.
    triggers = ("show me face tracking", "demo the face tracking", "can you see me")
    # Not claims_utterances: the enrolment questions are asked with ctx.ask,
    # which listens inside the hook (bounded by _ANSWER_WAIT_S), so their
    # answers never reach on_utterance and there is nothing to protect. Someone
    # who says "let's dance" while standing in front of it should get the dance.

    def on_enter(self, ctx: DemoContext) -> None:
        """Selected. One line naming what to watch for, then quiet.

        The demonstration is the head, and for a while this demo said nothing
        at all on the theory that a head following you is self-evident. Watched
        with visitors it is not: people do not know to look, and a robot that
        silently tracks somebody reads as a robot doing nothing. One sentence
        telling them what to try converts it into something they can test --
        and testing it is the demonstration, which is why the line asks them to
        move rather than explaining how detection works.
        """
        ctx.store["stage"] = None
        ctx.status("Vision: following faces.")
        ctx.say("Watch my head. Move around, and I'll follow you.", "curious")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        tracker = ctx.tracker
        if tracker is None or not tracker.enabled:
            # requires=("faces",) normally keeps this demo off the dashboard
            # when there is no tracker; a capability set that says otherwise
            # must leave it inert rather than raise once per slice.
            return IdleResult(listen_for=_LISTEN_S)

        store = ctx.store
        # A dialogue in progress owns the next slice outright. listen_for=0.0
        # on the way out is deliberate, against this file's own rule: the next
        # slice re-enters the dialogue immediately, so the "deafness" lasts one
        # trip round the loop -- and opening the wake-word or open-mic listener
        # between two questions of the same exchange would eat the answer.
        stage = store.get("stage")
        if stage == _ASK_NAME:
            return self._capture_name(ctx)
        if stage == _CONFIRM:
            return self._confirm_name(ctx, tracker)

        person_id, face = tracker.current(max_age_s=_PRESENT_AGE_S)
        if face is None:
            if store.get("present_since") is not None:
                self._on_leaving(ctx)
            store["present_since"] = None
            return IdleResult(listen_for=_LISTEN_S)

        present_since = store.get("present_since")
        if present_since is None:
            present_since = store["present_since"] = time.monotonic()
            self._on_arriving(ctx)
        if time.monotonic() - present_since < _SETTLE_S:
            return IdleResult(listen_for=_LISTEN_S)

        if person_id:
            # Greeting a recognised face is the runner's job now
            # (DemoRunner._greet_if_recognised), so that being known feels the
            # same under every demo rather than only under this one. Nothing to
            # do here but keep following them.
            return IdleResult(listen_for=_LISTEN_S)
        if store.get("stage") is None:
            return self._offer(ctx)
        return IdleResult(listen_for=_LISTEN_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        """Answer the camera question; the enrolment dialogue never comes here."""
        lowered = text.lower()
        if not any(phrase in lowered for phrase in _SEEING_PHRASES):
            return False
        if ctx.face_visible(max_age_s=_PRESENT_AGE_S):
            ctx.say("Yes -- you are in frame, and my head is following you.", "happy")
        else:
            ctx.say("Not at the moment. Step in front of me and I will pick you up.", "curious")
        return True

    def on_exit(self, ctx: DemoContext) -> None:
        """Nothing of this demo's is running; only the operator needs telling.

        The head keeps following faces after the switch, because the tracker
        belongs to the robot rather than to this demo, and from the outside
        that looks like this demo failing to stop.
        """
        # Presence is cleared too, not just the pending question: the store
        # survives being switched away from, so a stale present_since would let
        # somebody who walked up while another demo was showing skip the settle
        # window and be spoken to the instant Vision is selected.
        ctx.store["stage"] = None
        ctx.store["present_since"] = None
        ctx.status("Vision: off. Face tracking itself keeps running.")

    # --- meeting somebody ------------------------------------------------

    def _on_arriving(self, ctx: DemoContext) -> None:
        """Acknowledge somebody stepping into view, at most occasionally.

        This is the "reacts to presence" half of the demonstration, and it is
        rate-limited rather than absent because both extremes have been seen:
        narrating every arrival and departure made the robot exhausting to
        stand near, and saying nothing at all made a working tracker look
        broken. Once a minute is often enough that a visitor stepping in and
        out sees it happen, and rare enough that a group milling about in front
        of it is not talked at.
        """
        self._remark(ctx, "arrive", ("There you are.", "I see you.", "Got you."), "happy")

    def _on_leaving(self, ctx: DemoContext) -> None:
        """Notice somebody stepping out of frame. Same rate limit, same reason."""
        self._remark(ctx, "leave", ("And you're gone.", "Lost you.", "Where did you go?"), "curious")

    def _remark(self, ctx: DemoContext, key: str, lines: tuple[str, ...], emotion: str) -> None:
        """Say one of `lines`, no more than once per _REMARK_COOLDOWN_S per key.

        Rotated rather than random so a visitor never hears the same line twice
        running, and cooled down per kind so an arrival and a departure do not
        compete for the same budget.
        """
        now = time.monotonic()
        last = ctx.store.get(f"said:{key}")
        if last is not None and now - last < _REMARK_COOLDOWN_S:
            return
        ctx.store[f"said:{key}"] = now
        index = ctx.store.get(f"line:{key}", 0)
        ctx.store[f"line:{key}"] = index + 1
        ctx.say(lines[index % len(lines)], emotion)

    def _offer(self, ctx: DemoContext) -> IdleResult:
        """Ask an unrecognised visitor, at most once per cooldown, to be known.

        The whole exchange is direct question-and-answer -- no wake word, no
        instructions -- because a person who has just been asked a question is
        going to answer it, and the robot telling them how to answer was the
        last piece of operating manual it still read aloud. Each ctx.ask waits
        _ANSWER_WAIT_S for speech to start and then lets the recogniser finish
        the sentence, so no single hook holds the loop for long.
        """
        store = ctx.store
        offered_at = store.get("offered_at")
        if offered_at is not None and time.monotonic() - offered_at < _ASK_COOLDOWN_S:
            return IdleResult(listen_for=_LISTEN_S)
        # Stamped before the question, not after: a demo switched away
        # mid-sentence still counts as having asked, so the visitor is not
        # asked again on the way back.
        store["offered_at"] = time.monotonic()
        answer = ctx.ask(
            "I could remember you by name, if you like. Want me to?",
            "curious",
            wait_for_speech_s=_ANSWER_WAIT_S,
        )
        if not _is_yes(answer):
            ctx.status("Name declined; not asking again for a while.")
            return IdleResult(listen_for=_LISTEN_S)
        store["stage"] = _ASK_NAME
        store["rounds"] = 0
        # Straight into the next question rather than back to the idle window.
        # Returning _LISTEN_S here put three seconds of silence and a wake-word
        # window between "yes" and "what's your name?", which reads as the robot
        # having lost interest -- and is the point a visitor starts talking into
        # a microphone that is listening for something else.
        return IdleResult(listen_for=0.0)

    def _capture_name(self, ctx: DemoContext) -> IdleResult:
        """One attempt at hearing the name; the confirm slice decides its fate."""
        store = ctx.store
        store["rounds"] = store.get("rounds", 0) + 1
        prompt = "What's your name?" if store["rounds"] == 1 else "One more time. Just your name."
        heard = ctx.ask(prompt, "curious", wait_for_speech_s=_ANSWER_WAIT_S)

        name = extract_spoken_name(heard)
        if name is None:
            if heard:
                ctx.status(f"No name found in {heard!r}")
            if store["rounds"] >= _MAX_NAME_ROUNDS:
                store["stage"] = None
                ctx.say("I'm not catching it, sorry. We can try again later.", "sad")
                return IdleResult(listen_for=_LISTEN_S)
            return IdleResult(listen_for=0.0)  # straight back in to re-ask

        store["candidate"] = name
        store["confirm_silences"] = 0
        store["stage"] = _CONFIRM
        return IdleResult(listen_for=0.0)

    def _confirm_name(self, ctx: DemoContext, tracker: "FaceTracker") -> IdleResult:
        """Say the name back and only store it on a yes.

        This is the guard that lets extraction be permissive: a mis-decode here
        is not a wrong answer the next sentence corrects, it is a permanent
        name bound to a face embedding -- so nothing reaches the database that
        the visitor has not heard aloud and agreed to.
        """
        store = ctx.store
        name = store.get("candidate") or ""
        answer = ctx.ask(f"{name}. Did I get that right?", "curious", wait_for_speech_s=_ANSWER_WAIT_S)

        if _is_yes(answer):
            store["stage"] = None
            self._enroll(ctx, tracker, name)
            return IdleResult(listen_for=_LISTEN_S)

        # "No, it's Sarah" corrects and confirms in one breath: take the new
        # name and check that one instead of making them start over.
        corrected = extract_spoken_name(answer)
        if corrected and corrected.lower() != name.lower():
            store["candidate"] = corrected
            store["rounds"] = store.get("rounds", 0) + 1
            if store["rounds"] > _MAX_NAME_ROUNDS:
                store["stage"] = None
                ctx.say("I keep getting it wrong, sorry. Another time.", "sad")
                return IdleResult(listen_for=_LISTEN_S)
            return IdleResult(listen_for=0.0)

        if not answer.strip():
            # Silence is not a no: they may not have realised it was a
            # question. One repeat, then leave them alone.
            silences = store.get("confirm_silences", 0) + 1
            store["confirm_silences"] = silences
            if silences < 2:
                return IdleResult(listen_for=0.0)
            store["stage"] = None
            ctx.status("No confirmation; nothing stored.")
            return IdleResult(listen_for=_LISTEN_S)

        # A plain no. Ask again from the top if tries remain.
        if store.get("rounds", 0) >= _MAX_NAME_ROUNDS:
            store["stage"] = None
            ctx.say("Sorry about that. Another time, then.", "sad")
            return IdleResult(listen_for=_LISTEN_S)
        store["stage"] = _ASK_NAME
        return IdleResult(listen_for=0.0)

    def _enroll(self, ctx: DemoContext, tracker: "FaceTracker", name: str) -> None:
        """Bind the confirmed name to the face in view right now."""
        # Re-read rather than reusing a face from earlier in the exchange: that
        # one is many seconds old by now, and the face this name belongs to is
        # the one in front of the camera at the moment of storing.
        _person_id, face = tracker.current(max_age_s=_PRESENT_AGE_S)
        if face is None:
            ctx.say(f"I have lost sight of you, {name}. Another time.", "sad")
            return

        # The tracker's own identifier, reached past its underscore on purpose:
        # FaceTracker offers no accessor for it, and building a second
        # FaceIdentifier here would put a second MediaPipe detector on a second
        # thread, which face_tracker.py exists to avoid.
        person_id = tracker._face.enroll(name, face)
        if person_id is None:
            ctx.say("Sorry, I could not save that. Another time.", "sad")
            return
        # The session greeting is spent here, on the thank-you: the tracker
        # starts recognising this face within seconds, and without this the
        # runner follows "thank you, Sarah" with "oh, hello again Sarah".
        ctx.state.mark_greeted(name)
        ctx.status(f"Enrolled {name} as person {person_id}.")
        ctx.say(f"Thank you, {name}. I will know you next time.", "happy")
