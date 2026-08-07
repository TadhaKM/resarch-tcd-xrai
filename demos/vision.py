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
a face, so the answer goes through body.face.clean_spoken_name and anything it
rejects is dropped rather than stored. Someone who declines, or who says
nothing, is not asked again for _ASK_COOLDOWN_S; someone the robot already
recognises is greeted once and never asked at all.

Deliberately absent: any call to ctx.motion.express_move(). A recorded move
pauses the motion loop for its duration (see MotionController._run), and that
loop is what applies the tracking target -- so a flourish would freeze the head
in the one demo whose subject is the head not freezing.
"""

import re
import time
from typing import TYPE_CHECKING

from body.face import clean_spoken_name
from brain import db
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

#: Wake-word window returned by every idle slice; the core clamps it to
#: MAX_LISTEN_WINDOW_S. Never zero: DemoRunner.cycle opens no microphone at
#: all for a zero, and nothing here is worth being deaf for.
_LISTEN_S = 3.0

#: Said for a transcript that is not a plausible name, and again if enroll
#: refuses the same text one call later. One string, because it is one failure.
_NOT_CAUGHT = "Sorry, I did not catch that."

#: Enough of an answer to act on, matched as whole words: "no" sits inside
#: "know" and "now", and reading a no as a yes interrogates somebody who
#: declined. A no anywhere in the answer wins, because "ok, no thanks" is a no.
_YES_WORDS = frozenset({"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "alright"})
_NO_WORDS = frozenset({"no", "nope", "nah", "not", "never", "rather"})
_WORD_RE = re.compile(r"[a-z]+")

#: Questions about the camera itself, which is what this demo's own trigger
#: phrase is, so it is the first thing many visitors say on arriving here.
_SEEING_PHRASES = ("see me", "seeing me", "see anyone", "can you see")


def _is_yes(answer: str) -> bool:
    """Whether an answer to a yes/no question was a yes.

    Anything that is not clearly a yes is treated as a no: silence, a shrug,
    or the recognizer picking up the conversation happening behind the visitor.
    Being wrong that way costs nothing but the offer; being wrong the other way
    presses a stranger for their name.
    """
    words = set(_WORD_RE.findall(answer.lower()))
    return bool(words & _YES_WORDS) and not words & _NO_WORDS


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
    # Not claims_utterances: both questions here are asked with ctx.ask, which
    # listens inside the hook, so their answers never reach on_utterance and
    # there is nothing to protect. Someone who says "let's dance" while
    # standing in front of it should get the dance.

    def on_enter(self, ctx: DemoContext) -> None:
        """Selected. Says nothing: the demonstration is the head, not a sentence."""
        ctx.status("Vision: following faces, quiet unless it meets someone new.")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        tracker = ctx.tracker
        if tracker is None or not tracker.enabled:
            # requires=("faces",) normally keeps this demo off the dashboard
            # when there is no tracker; a capability set that says otherwise
            # must leave it inert rather than raise once per slice.
            return IdleResult(listen_for=_LISTEN_S)

        store = ctx.store
        person_id, face = tracker.current(max_age_s=_PRESENT_AGE_S)
        if face is None:
            store["present_since"] = None
            # A pending yes belongs to the person who said it, and they have
            # gone. Whoever appears next is not answering their question.
            store["wants_to_be_known"] = False
            return IdleResult(listen_for=_LISTEN_S)

        present_since = store.get("present_since")
        if present_since is None:
            present_since = store["present_since"] = time.monotonic()
        if time.monotonic() - present_since < _SETTLE_S:
            return IdleResult(listen_for=_LISTEN_S)

        if person_id:
            self._greet(ctx, person_id)
        elif store.pop("wants_to_be_known", False):
            # A slice later, not straight after the yes: two questions and two
            # answers in one hook is ten seconds in which nothing consumes the
            # microphone and the operator cannot switch away. Split, a mode
            # switch lands between them.
            self._take_name(ctx, tracker)
        else:
            self._offer(ctx)
        return IdleResult(listen_for=_LISTEN_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        """Answer "can you see me" from the tracker; leave everything else alone.

        Answered here rather than by the language model, which has no camera
        and will make something up, and in one sentence rather than a
        description of how detection works -- the answer is checkable by
        stepping sideways, which is the whole demonstration. Every other
        question falls through to conversation.
        """
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
        ctx.store["wants_to_be_known"] = False
        ctx.status("Vision: off. Face tracking itself keeps running.")

    # --- meeting somebody ------------------------------------------------

    def _greet(self, ctx: DemoContext, person_id: int) -> None:
        """Say hello by name, once per person for as long as the robot is up."""
        greeted = ctx.store.setdefault("greeted", set())
        if person_id in greeted:
            return
        # Recorded before the lookup and whatever it returns: a person row with
        # no name gives nothing to say, and re-checking it every couple of
        # seconds for the rest of the visit would say nothing in a loop.
        greeted.add(person_id)
        name = db.get_person_name(person_id)
        if name:
            ctx.say(f"Hello again, {name}.", "happy")

    def _offer(self, ctx: DemoContext) -> None:
        """Ask an unrecognised visitor, at most once per cooldown, to be known."""
        store = ctx.store
        offered_at = store.get("offered_at")
        if offered_at is not None and time.monotonic() - offered_at < _ASK_COOLDOWN_S:
            return
        # Stamped before the question, not after it, so a demo switched away
        # mid-question -- ctx.ask raises DemoStopped -- still counts as having
        # asked, and the visitor is not asked again on the way back.
        store["offered_at"] = time.monotonic()
        answer = ctx.ask("Would you like to tell me your name, so I know you next time?", "curious")
        store["wants_to_be_known"] = _is_yes(answer)
        if not store["wants_to_be_known"]:
            ctx.status("Name declined or unanswered; not asking again for a while.")

    def _take_name(self, ctx: DemoContext, tracker: "FaceTracker") -> None:
        """Ask for the name and enrol it against the face in view right now."""
        heard = ctx.ask("What is your name?", "curious")
        name = clean_spoken_name(heard)
        if name is None:
            # Nothing unvalidated is ever stored. A mis-decode here is not a
            # wrong answer that the next sentence corrects; it is a permanent
            # name bound to a face embedding, undone only by editing the
            # database.
            ctx.status(f"Name rejected as implausible: {heard!r}")
            ctx.say(_NOT_CAUGHT, "sad")
            return

        # Re-read rather than reusing the face from the top of the slice: that
        # one is several seconds old by the time the name arrives, and the face
        # this name belongs to is the one in front of the camera now.
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
            # enroll validates the name again and refuses exactly what
            # clean_spoken_name refuses, so this is that failure arriving late.
            ctx.say(_NOT_CAUGHT, "sad")
            return
        # The greeting is spent here, otherwise the next slice recognises the
        # face it has just been handed and says hello again to somebody who
        # never left.
        ctx.store.setdefault("greeted", set()).add(person_id)
        ctx.status(f"Enrolled {name} as person {person_id}.")
        ctx.say(f"Thank you, {name}. I will know you next time.", "happy")
