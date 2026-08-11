"""Follow whoever is in view, and offer -- once -- to learn a new face's name.

The following is not this demo's doing: body/face_tracker.py aims the head at
whoever is in view under every demo, all the time. What this file adds is the
one thing the tracker cannot do by itself, which is put a name to a face, and
it deliberately adds nothing else. An earlier version narrated every arrival
and departure and explained the detection pipeline; watched live, that is a
robot talking over the thing it is meant to be showing. A head that follows you
is legible without a commentary, so most of this file is about not speaking.

Every new face gets its own offer. That used to be one offer per ten minutes
for the whole room, which is not the same thing at all -- the second visitor of
the afternoon was simply never asked. Faces are now told apart by their own
embedding (_already_offered), so declining is remembered against the person who
declined and the visitor beside them still gets their turn. The offer waits
while somebody is mid-conversation with the robot, rather than interrupting a
question to talk about itself.

The name is asked for and never taken. Voice enrolment was removed from this
project once before because a noisy transcript became a permanent name bound to
a face, so nothing is stored until two separate filters agree: body.face.
extract_spoken_name pulls the most name-like part out of whatever was said
("My name is Sarah, nice to meet you" enrols Sarah, not the sentence), and the
robot then says that name back and waits for a yes. Wrong, the visitor says so
and gives it again -- two more tries before it stops pestering. Someone who
declines, or who says nothing, is not asked again; someone the robot already
recognises is greeted once and never asked at all.

Deliberately absent: any call to ctx.motion.express_move(). A recorded move
pauses the motion loop for its duration (see MotionController._run), and that
loop is what applies the tracking target -- so a flourish would freeze the head
in the one demo whose subject is the head not freezing.
"""

import re
import time
from typing import TYPE_CHECKING

import numpy as np

from body.face import extract_spoken_name
from config import MODELS
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

#: Quiet after an offer before ANY new face is offered to. Ten seconds, where
#: it used to be ten minutes -- and the ten minutes was not caution, it was the
#: only tool available: an unrecognised face has no id to hang "this one
#: already said no" on, so the silence had to cover everybody, and one visitor
#: declining meant the next four were never asked at all.
#:
#: With faces now told apart by their own embedding, all this has left to do is
#: stop a group being interrogated one after another as the tracker's attention
#: moves between them. Short, because it defers rather than drops: somebody
#: still standing there gets asked on the next slice.
_ASK_COOLDOWN_S = 10.0

#: Faces already offered to this session, matched by their own embedding, which
#: is what lets the robot tell one stranger from another without knowing who
#: either of them is. Somebody who declined is not asked again; the person
#: beside them is a different face and gets their own turn.
_OFFERED = "offered_faces"

#: How alike two embeddings must be to count as the same person for the purpose
#: above. The recognition threshold, deliberately: being asked twice is a small
#: annoyance and never being asked is the thing this exists to fix, so it errs
#: toward treating a doubtful match as somebody new.
_SAME_FACE = MODELS.face_match_threshold

#: How many faces to keep in that list. Every one is compared against on every
#: slice while somebody unrecognised is in view, and the store outlives the
#: demo, so this is both a memory and a per-slice cost.
_MAX_REMEMBERED_OFFERS = 64

#: A visitor who has spoken to the robot this recently is mid-conversation, and
#: an unprompted "shall I remember you?" is the robot talking over them. It
#: waits rather than dropping the offer -- they are still in front of it, so the
#: next quiet slice will do.
_BUSY_TALKING_S = 20.0

#: When this demo last put a question of its own to somebody. Needed because
#: the answers to those questions are themselves recorded as speech heard, so
#: without it a visitor saying "no thanks" to the offer counted as a
#: conversation in progress and silenced the offer to the person beside them
#: for the next twenty seconds -- the exact fault this was meant to fix.
_OUR_EXCHANGE_AT = "our_exchange_at"

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

#: Words that settle a yes/no question on their own, matched as whole words:
#: "no" sits inside "know" and "now".
_YES_WORDS = frozenset({
    "yes", "yeah", "yep", "yup", "yah", "aye", "sure", "ok", "okay", "okey",
    "please", "alright", "absolutely", "definitely", "certainly", "course",
    "go", "do", "id", "love", "great", "good", "fine", "perfect",
})
_NO_WORDS = frozenset({"no", "nope", "nah", "not", "never", "rather", "dont", "skip"})
_WORD_RE = re.compile(r"[a-z]+")

#: Phrases that mean the opposite of the words in them, checked before any
#: single word is. Every one of these was being read as a refusal: "why not"
#: and "no problem" and "I don't mind" are agreement, and they all contain a
#: word from _NO_WORDS. A visitor saying "go on then, why not" was told the
#: robot would leave them alone.
_YES_PHRASES = (
    "why not", "no problem", "dont mind", "do not mind", "dont see why not",
    "go on", "go ahead", "of course", "id like", "i would like", "sounds good",
    "sure thing", "if you like", "if you want", "that would be nice",
)

#: And the reverse: a refusal whose words could read as agreement. "no thanks"
#: is already caught by "no", but "maybe later" and "some other time" contain
#: nothing from either list and would otherwise be unclear rather than a no.
_NO_PHRASES = (
    "no thanks", "no thank you", "rather not", "not now", "not right now",
    "maybe later", "another time", "some other time", "im ok", "im good",
    "im fine", "leave it",
)

#: What the robot does with an answer it could not read. Deliberately not a
#: refusal: silence, a mumble, or a transcript the recogniser dropped all used
#: to count as "no", which is how a visitor saying yes was told the robot would
#: leave them alone. It asks once more instead, and only then lets it go.
_YES, _NO, _UNCLEAR = "yes", "no", "unclear"

#: Enrolment stages, held in ctx.store["stage"]. One bounded exchange per idle
#: slice: each slice says one question and waits at most _ANSWER_WAIT_S for the
#: answer to begin, so the loop gets the robot back between questions and the
#: operator can always switch away.
_ASK_NAME = "ask_name"
_CONFIRM = "confirm"

#: Questions about the camera itself, which is what this demo's own trigger
#: phrase is, so it is the first thing many visitors say on arriving here.
_SEEING_PHRASES = ("see me", "seeing me", "see anyone", "can you see")


def _read_answer(answer: str) -> str:
    """Read a yes/no answer as _YES, _NO, or _UNCLEAR.

    Three outcomes rather than two, and that is the whole point. This used to
    return a bool, so everything it could not read -- silence, a mumble, an
    answer the recogniser dropped, "go on then" -- came back as False and was
    acted on as a refusal. A visitor who said yes got told the robot would
    leave them alone, and the caller could not tell the difference between
    being turned down and not being heard.

    Whole phrases are checked before single words, because the phrases people
    actually use to agree are full of refusal words: "why not", "no problem",
    "I don't mind". Between single words the FIRST one wins -- "ok, no thanks"
    leads with agreement but is plainly a refusal, and "sure, why not" is the
    reverse -- so the phrase pass settles those before it matters.
    """
    words = _WORD_RE.findall(answer.lower())
    if not words:
        return _UNCLEAR
    flat = " ".join(words)
    for phrase in _NO_PHRASES:
        if phrase in flat:
            return _NO
    for phrase in _YES_PHRASES:
        if phrase in flat:
            return _YES
    for word in words:
        if word in _YES_WORDS:
            return _YES
        if word in _NO_WORDS:
            return _NO
    return _UNCLEAR


class Vision(Demo):
    label = "Vision & Face Tracking"
    help = "Follows whoever is in view, and offers to learn each new face's name."
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
        """Selected. Says nothing; the head is the demonstration.

        This used to open with "Watch my head. Move around, and I'll follow
        you." -- stage directions for a thing the visitor can already see
        happening. The head starts following the moment the demo is selected,
        and "There you are." when somebody settles in front of it (see
        _on_arriving) is a reaction rather than an instruction: it shows the
        robot noticing them, which is the same information without the robot
        telling anyone what to look at.

        Whoever is running the demonstration can say "watch its head" in half
        the time and to the right person. That is their line, not the robot's.
        """
        self._end_exchange(ctx)
        ctx.status("Vision: following faces.")

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
        # Releases the microphone hold too, which matters most here: switched
        # away mid-question, the hold would otherwise outlive the demo that
        # took it and leave the robot listening to the room all afternoon.
        self._end_exchange(ctx)
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

    def _begin_exchange(self, ctx: DemoContext, stage: str) -> None:
        """Enter the name exchange and hold the microphone open for it.

        Every reply from here to a name being locked in is expected, so a wake
        word before each one is absurd -- being asked your name and having to
        say "hey Reachy" to answer. Held rather than switched, so the
        operator's own open-mic setting is untouched and returns the moment
        this ends.
        """
        ctx.store["stage"] = stage
        ctx.state.hold_open_mic(True)

    def _end_exchange(self, ctx: DemoContext) -> None:
        """Leave it, and give the microphone back to whatever was set before.

        Every exit from the exchange goes through here -- enrolled, declined,
        given up on, or switched away from mid-question -- because a hold left
        on is a robot that listens to the room for the rest of the afternoon.
        """
        ctx.store["stage"] = None
        ctx.state.hold_open_mic(False)

    def _busy_talking(self, ctx: DemoContext) -> bool:
        """Whether somebody is mid-conversation with the robot right now.

        Counts only speech that arrived AFTER this demo's own last question.
        The answers to those questions are recorded as speech heard like
        anything else, so measured naively a visitor declining the offer looked
        exactly like a conversation in progress -- and silenced the offer to
        the next person for twenty seconds, which is the fault this whole rule
        exists to fix.
        """
        since = ctx.state.seconds_since_heard()
        if since is None or since >= _BUSY_TALKING_S:
            return False
        heard_at = time.time() - since
        ours = ctx.store.get(_OUR_EXCHANGE_AT)
        # A small margin: our own question and its answer land within a moment
        # of each other, and the two clocks are read at slightly different
        # points in the same turn.
        return ours is None or heard_at > ours + 1.0

    def _already_offered(self, ctx: DemoContext) -> bool:
        """Whether the face in view has had its turn being asked.

        Compared by embedding rather than by id, because the whole point is
        that these people have no id yet. Without this the robot had one blunt
        instrument -- a silence long enough to cover everybody -- so the second
        visitor of the afternoon was never asked at all.
        """
        tracker = ctx.tracker
        if tracker is None:
            return False
        current = tracker.current_embedding(max_age_s=_PRESENT_AGE_S)
        if current is None:
            # No face vector to compare. Treat as already handled rather than
            # asking blind: the alternative offers to remember a face the robot
            # cannot actually store.
            return True
        seen = ctx.store.setdefault(_OFFERED, [])
        probe = float(np.linalg.norm(current))
        for other in seen:
            stored = float(np.linalg.norm(other))
            if not stored or not probe:
                continue
            if float(np.dot(current, other) / (stored * probe)) >= _SAME_FACE:
                return True
        seen.append(current)
        # Bounded: the store lives for the whole process, so an open day would
        # otherwise grow this list all afternoon and compare against every face
        # it had ever seen on every slice. Dropping the oldest costs somebody
        # from hours ago being asked twice, which is not a problem.
        if len(seen) > _MAX_REMEMBERED_OFFERS:
            del seen[: len(seen) - _MAX_REMEMBERED_OFFERS]
        return False

    def _offer(self, ctx: DemoContext) -> IdleResult:
        """Ask an unrecognised visitor, once each, to be known.

        Once EACH, not once per session: every new face gets its own offer, so
        the fifth person to stand in front of the robot is asked the same as
        the first. Held off while somebody is mid-conversation with it, because
        an unprompted "shall I remember you?" over the top of a question is the
        robot interrupting a visitor to talk about itself.

        The exchange itself is direct question-and-answer -- no wake word, no
        instructions -- because a person who has just been asked a question is
        going to answer it. Each ctx.ask waits _ANSWER_WAIT_S for speech to
        start and then lets the recogniser finish the sentence, so no single
        hook holds the loop for long.
        """
        store = ctx.store
        if self._busy_talking(ctx):
            # Not dropped, just deferred -- they are still standing there.
            return IdleResult(listen_for=_LISTEN_S)
        offered_at = store.get("offered_at")
        if offered_at is not None and time.monotonic() - offered_at < _ASK_COOLDOWN_S:
            return IdleResult(listen_for=_LISTEN_S)
        if self._already_offered(ctx):
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
        store[_OUR_EXCHANGE_AT] = time.time()
        verdict = _read_answer(answer)
        if verdict is _UNCLEAR:
            # Not heard is not the same as turned down. Asked plainly the
            # second time, because "want me to?" answered with silence is
            # usually somebody who did not realise it was their turn.
            ctx.status(f"Unclear answer to the offer: {answer!r}")
            answer = ctx.ask("Sorry -- was that a yes?", "curious",
                             wait_for_speech_s=_ANSWER_WAIT_S)
            store[_OUR_EXCHANGE_AT] = time.time()
            verdict = _read_answer(answer)
        if verdict is not _YES:
            ctx.status(f"Name declined ({answer!r}); not asking this face again.")
            return IdleResult(listen_for=_LISTEN_S)
        self._begin_exchange(ctx, _ASK_NAME)
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
        store[_OUR_EXCHANGE_AT] = time.time()

        name = extract_spoken_name(heard)
        if name is None:
            if heard:
                ctx.status(f"No name found in {heard!r}")
            if store["rounds"] >= _MAX_NAME_ROUNDS:
                self._end_exchange(ctx)
                ctx.say("I'm not catching it, sorry. We can try again later.", "sad")
                return IdleResult(listen_for=_LISTEN_S)
            return IdleResult(listen_for=0.0)  # straight back in to re-ask

        store["candidate"] = name
        store["confirm_silences"] = 0
        self._begin_exchange(ctx, _CONFIRM)
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
        store[_OUR_EXCHANGE_AT] = time.time()
        verdict = _read_answer(answer)

        if verdict is _YES:
            self._end_exchange(ctx)
            self._enroll(ctx, tracker, name)
            return IdleResult(listen_for=_LISTEN_S)

        # "No, it's Sarah" corrects and confirms in one breath: take the new
        # name and check that one instead of making them start over.
        corrected = extract_spoken_name(answer)
        if corrected and corrected.lower() != name.lower():
            store["candidate"] = corrected
            store["rounds"] = store.get("rounds", 0) + 1
            if store["rounds"] > _MAX_NAME_ROUNDS:
                self._end_exchange(ctx)
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
            self._end_exchange(ctx)
            ctx.status("No confirmation; nothing stored.")
            return IdleResult(listen_for=_LISTEN_S)

        # A plain no. Ask again from the top if tries remain.
        if store.get("rounds", 0) >= _MAX_NAME_ROUNDS:
            self._end_exchange(ctx)
            ctx.say("Sorry about that. Another time, then.", "sad")
            return IdleResult(listen_for=_LISTEN_S)
        self._begin_exchange(ctx, _ASK_NAME)
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
