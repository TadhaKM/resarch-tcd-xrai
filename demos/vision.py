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

#: Said for a transcript that is not a plausible name, and again if enroll
#: refuses the same text one call later. One string, because it is one failure.
_NOT_CAUGHT = "Sorry, I did not catch that."

#: Enough of an answer to act on, matched as whole words: "no" sits inside
#: "know" and "now", and reading a no as a yes interrogates somebody who
#: declined. A no anywhere in the answer wins, because "ok, no thanks" is a no.
_YES_WORDS = frozenset({"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "alright"})
_NO_WORDS = frozenset({"no", "nope", "nah", "not", "never", "rather"})
_WORD_RE = re.compile(r"[a-z]+")

#: What the robot is waiting to hear. The exchange is spread across turns
#: rather than held open inside a hook, so each stage is one short question and
#: the answer arrives through on_utterance like anything else a visitor says.
_AWAITING_CONSENT = "consent"
_AWAITING_NAME = "name"

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
    # Not claims_utterances: both questions here are asked with ctx.ask, which
    # listens inside the hook, so their answers never reach on_utterance and
    # there is nothing to protect. Someone who says "let's dance" while
    # standing in front of it should get the dance.

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
        person_id, face = tracker.current(max_age_s=_PRESENT_AGE_S)
        if face is None:
            if store.get("present_since") is not None:
                self._on_leaving(ctx)
            store["present_since"] = None
            # A pending yes belongs to the person who said it, and they have
            # gone. Whoever appears next is not answering their question.
            store["wants_to_be_known"] = False
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
            pass
        elif store.get("stage") is None:
            self._offer(ctx)
        # A stage in progress means the robot has asked something and is
        # waiting for on_utterance to bring the answer. Nothing to do here but
        # keep listening.
        return IdleResult(listen_for=_LISTEN_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        """Take the answer to whatever this demo last asked, or the camera question.

        The two-step exchange lives here rather than in on_idle because
        listening inside a hook holds the voice loop for as long as the
        recogniser takes -- up to 25 seconds -- during which the robot cannot
        be switched away from or interrupted. Speaking the question and
        collecting the answer on a later turn keeps every hook short.
        """
        stage = ctx.store.get("stage")
        if stage == _AWAITING_CONSENT:
            ctx.store["stage"] = None
            if _is_yes(text):
                self._ask_name(ctx)
            else:
                ctx.status("Name declined; not asking again for a while.")
            return True
        if stage == _AWAITING_NAME:
            ctx.store["stage"] = None
            tracker = ctx.tracker
            if tracker is not None and tracker.enabled:
                self._take_name(ctx, tracker, text)
            return True

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

    def _offer(self, ctx: DemoContext) -> None:
        """Ask an unrecognised visitor, at most once per cooldown, to be known.

        Speaks and returns. The answer arrives through on_utterance on a later
        slice, because ctx.ask would listen inside this hook: AudioIO.listen
        returns only on a recogniser endpoint and is bounded at 25s, so one
        offer could hold the voice loop -- and the microphone, and the
        operator's ability to switch demo -- for the better part of half a
        minute. Speaking and handing the turn back costs a wake word from the
        visitor and keeps the robot answerable throughout.
        """
        store = ctx.store
        offered_at = store.get("offered_at")
        if offered_at is not None and time.monotonic() - offered_at < _ASK_COOLDOWN_S:
            return
        # Stamped before the question, not after: a demo switched away
        # mid-sentence still counts as having asked, so the visitor is not
        # asked again on the way back.
        store["offered_at"] = time.monotonic()
        store["stage"] = _AWAITING_CONSENT
        ctx.say(
            "Would you like to tell me your name, so I know you next time? "
            "Say hey Reachy, then yes or no.",
            "curious",
        )

    def _ask_name(self, ctx: DemoContext) -> None:
        """Invite the name. Answered through on_utterance, for the same reason."""
        ctx.store["stage"] = _AWAITING_NAME
        # The only place the robot still explains itself, and only when it must:
        # the answer arrives through the wake-word path, so with the mic shut a
        # visitor who just says their name is talking into a microphone nobody
        # opened, and the offer to remember them quietly fails. With open mic on
        # there is nothing to explain, so it does not.
        if ctx.state.open_mic:
            ctx.say("What's your name?", "curious")
        else:
            ctx.say("What's your name? Say hey Reachy first.", "curious")

    def _take_name(self, ctx: DemoContext, tracker: "FaceTracker", heard: str) -> None:
        """Enrol the name just heard against the face in view right now."""
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
