"""The opening demonstration: who we are, said out loud to a group that just arrived.

Two problems shape this file.

The first is timing. brain.hub.WELCOME_SCRIPT is about thirty-two seconds of
speech, and nothing consumes the microphone while the robot talks -- so
speaking it in one call would leave the robot deaf, and unswitchable from the
dashboard, for half a minute, in front of exactly the audience least willing to
wait for it. The script is therefore split into sentences at import and spoken
one per on_idle slice, with a short listening window between them: the same
audio, effectively back to back, but with the operator's mode switch and a
visitor's question both landing between any two lines.

The second is repetition. An idle loop left alone replays its script forever,
and a robot that re-introduces itself every ninety seconds to the same three
people reads as broken rather than welcoming. Replaying is gated on evidence
that the audience actually changed -- a room that emptied, a face never present
while the script was being spoken, or, with no camera at all, a long silence and
a long wait. That evidence is collected once per slice in _observe and latched,
because both kinds of evidence are momentary and the decision that uses them is
made later.

Every fact spoken here comes from brain/hub.py. This file contributes only the
interaction: when to start, when to stop, and when it is someone else's turn.
"""

import re
import time

from brain import hub
from demokit import Demo, DemoContext, IdleResult
from demokit.base import split_sentences


#: Words of a transcript, for tests that must not fire on a substring: "again"
#: sits inside "against", and a bare substring match turned "what have you got
#: against robots" into a request to repeat the last line.
_WORD_RE = re.compile(r"[a-z']+")


#: The splitter moved to demokit/base.py when a second demo needed it, which is
#: what the note here always said should happen rather than a copy being made.
_spoken_lines = split_sentences


#: Split once at import: the script is a constant, and its line count is the
#: only thing that knows when the welcome is over.
_WELCOME_LINES = _spoken_lines(hub.WELCOME_SCRIPT)

#: Spoken after the script, and not a claim about the Hub, which is why it is
#: here rather than in brain/hub.py. It used to name the wake phrase as well.
#: That is the operator's job -- it belongs in their guide, said once to a
#: group, rather than in the robot's closing line to every visitor, where it
#: turned the end of a thirty-second welcome into an instruction manual.
_INVITATION = "Ask me anything you like."

#: How long to wait for someone to appear before starting anyway. An operator
#: presses this button in front of a group; if the tracker cannot see them --
#: side-on, backlit, or a camera nobody plugged in -- the demonstration still
#: has to happen.
_FACE_PATIENCE_S = 20.0

#: No face for this long means the group moved on, so the next face is a new
#: audience. Generous, because the tracker drops a face every time someone
#: turns to talk to the person beside them.
_ROOM_EMPTY_S = 25.0

#: Shortest gap between two welcomes when the camera says the audience changed.
#: Short, because groups genuinely do arrive back to back on an open day; the
#: emptiness test above already absorbs a tracker blinking on someone who never
#: left, so this only stops the script rebounding straight off its own ending.
_SEEN_COOLDOWN_S = 60.0

#: Gap before replaying with no camera to consult. Long, because replaying
#: blind is a guess, and the cost of guessing wrong is the robot talking over a
#: conversation it is already part of.
_BLIND_COOLDOWN_S = 420.0

#: Anyone who has spoken to us this recently has plainly already been welcomed.
#: This is what keeps a blind replay off the back of a live conversation.
_QUIET_BEFORE_REPLAY_S = 120.0

#: Wake-word window once the robot has nothing left to say. The core clamps
#: this to MAX_LISTEN_WINDOW_S anyway; asking for the cap makes the gaps
#: between windows, during which a wake word is missed, as rare as allowed.
_LISTEN_S = 3.0

#: Wake-word window between two script lines. Not zero: DemoRunner.cycle returns
#: without opening the microphone at all when a demo asks for zero, so six
#: consecutive zeroes made the robot deaf for the whole thirty-two seconds --
#: no interruption, no question, no wake word, in front of a group hearing the
#: robot for the first time. One second is short enough that the lines still
#: land as continuous speech, and the wake-word stream survives across calls, so
#: a phrase spoken across the boundary of one window is still matched.
_BETWEEN_LINES_S = 1.0

#: Unmistakable requests for this demonstration. The same phrases switch to it
#: from another demo and restart it once it is running, because they mean the
#: same thing in both places.
#:
#: A bare "welcome" is deliberately absent. Triggers are matched as substrings
#: of any transcript, in any demo, so "you're welcome", "welcome back" and
#: "welcome to Dublin" each abandoned whatever the robot was doing and started
#: thirty-two seconds of script. Nothing here is a phrase anyone says by
#: accident in ordinary politeness.
_WELCOME_REQUESTS = (
    "do the welcome",
    "start the welcome",
    "do the welcome speech",
    "give the welcome speech",
    "welcome the group",
    "welcome us",
    "introduce the hub",
)

#: Restarts only, never triggers: they name no demo, so as triggers they would
#: hijack "tell me the story from the beginning" for this one. They are safe
#: here because on_utterance is only offered an utterance that no other demo's
#: trigger phrase claimed first.
_RESTART_EXTRA = ("from the beginning", "from the start", "the whole welcome")

#: A request to hear the whole script again. Deliberately without a word-count
#: guard: every phrase is an explicit ask, so a longer sentence wrapped around
#: one ("could you do the welcome speech for these folks") is still an ask.
_RESTART_PHRASES = _WELCOME_REQUESTS + _RESTART_EXTRA

#: A request to hear the last thing said again -- one sentence, not the script.
#: These are unambiguous enough to match anywhere in a transcript.
_REPEAT_PHRASES = (
    "say that again",
    "say it again",
    "repeat that",
    "repeat it",
    "one more time",
    "once more",
    "what was that",
    "come again",
    "didn't catch",
    "did not catch",
    "didn't hear",
    "did not hear",
)

#: The same request as one bare word. Matched only in a short utterance,
#: because "will you be here again tomorrow" is a question, not a request for a
#: repeat, and only as a whole word (see _WORD_RE).
_REPEAT_WORDS = frozenset({"again", "repeat", "pardon"})
_REPEAT_MAX_WORDS = 3

#: Enough of a question about the Hub itself to answer from hub.MISSION_SHORT
#: rather than from the model. Both halves must match, so "what does the Hub
#: do" is caught and "who runs the Hub" is not -- the second is a question about
#: people, which this demo has no business answering in one canned line. The
#: subject list is the Hub and nothing else for the same reason: "what is XR"
#: matched a bare "xr" and got the mission statement back, which is a confident
#: answer to a question nobody asked.
_MISSION_OPENERS = ("what is", "what's", "what does", "tell me about", "what do you do")
_MISSION_SUBJECTS = ("hub", "this place")


def _since(store: dict, key: str) -> float:
    """Seconds since `key` was stamped, or inf if it never was.

    inf rather than 0 because every caller here is asking "has enough time
    passed", and a thing that has never happened should read as long ago rather
    than as just now.
    """
    stamped = store.get(key)
    return float("inf") if stamped is None else time.monotonic() - stamped


def _faces_available(ctx: DemoContext) -> bool:
    """Whether there is a working face pipeline to consult at all.

    ctx.face_visible() answers False both for "nobody is there" and for "no
    camera", and this demo needs opposite behaviour in the two cases: wait for
    an audience in the first, start talking immediately in the second. That
    distinction is not on DemoContext, so the tracker is read directly -- the
    same two conditions face_visible() itself tests.
    """
    tracker = ctx.tracker
    return tracker is not None and tracker.enabled


def _emotion_for(index: int) -> str:
    """Delivery from position in the script, never a per-sentence table.

    brain/hub.py is edited by the people who run the Hub. A hand-written list
    of emotions indexed by sentence would silently mis-align the first time
    someone adds a line, and the result -- a cheerful delivery of the wrong
    clause -- is visible to the audience and to nobody else.
    """
    if index == 0 or index == len(_WELCOME_LINES) - 1:
        return "happy"
    return "neutral"


class Welcome(Demo):
    label = "Welcome"
    help = "Greets a new group and explains the Hub, one line at a time."
    order = 20
    #: Greeting a room of visitors. Warm is the register a welcome is in.
    persona = "friendly"
    triggers = _WELCOME_REQUESTS
    #: Deliberately not ("faces",): this has to work on the robot's own CPU,
    #: where face detection is off. The camera makes the welcome better timed,
    #: never possible -- so the demo adapts instead of greying out.
    requires = ()
    #: Deliberately not claims_utterances. This demo asks the visitor nothing,
    #: so it has no answer of its own to protect, and claiming utterances would
    #: stop "dance" or "tell me a story" working while the welcome is selected
    #: -- which, at order 20, is most of the time.
    claims_utterances = False

    def on_enter(self, ctx: DemoContext) -> None:
        """Arm the script. Speak nothing.

        Nothing is said here on purpose: the script's own first line is a
        greeting, and it follows within one idle slice, so a line here would
        simply be a second hello over the top of the first.
        """
        store = ctx.store
        store["entered_at"] = time.monotonic()
        # The store is kept per demo id for the life of the process and is
        # never cleared between visitors, so it is deliberately read rather
        # than reset here: the cooldown below is only meaningful because it
        # survives an operator toggling to another demo and back in front of
        # the same group.
        if _since(store, "finished_at") < _SEEN_COOLDOWN_S:
            store["line"] = len(_WELCOME_LINES)
            store["awaiting_face"] = False
            # Counted as invited too. The invitation follows the last script
            # line by one slice, so a group still inside the cooldown has
            # already heard it, and re-speaking it here would be the same
            # repetition the cooldown exists to prevent -- one sentence of it
            # rather than six, and contradicting the status note below.
            store["invited"] = True
            ctx.status("Welcome already given; listening instead of repeating it.")
            return
        self._begin(ctx, awaiting_face=_faces_available(ctx) and not ctx.face_visible())
        ctx.status("Waiting for someone to appear." if store["awaiting_face"] else "Welcoming.")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        # Once per slice, before anything decides anything: the camera evidence
        # this demo runs on is momentary and has to be caught as it happens.
        self._observe(ctx)
        line = store.get("line", 0)

        if line < len(_WELCOME_LINES):
            if store.get("awaiting_face") and not self._audience_arrived(ctx):
                # Listening while waiting: someone who speaks before the robot
                # has started should be answered, not talked over by a script.
                return IdleResult(listen_for=_LISTEN_S)
            store["awaiting_face"] = False
            self._say(ctx, _WELCOME_LINES[line], _emotion_for(line))
            store["line"] = line + 1
            if store["line"] >= len(_WELCOME_LINES):
                self._finish(store)
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        if not store.get("invited"):
            store["invited"] = True
            self._say(ctx, _INVITATION, "curious")
            # The handover from monologue to conversation is otherwise
            # invisible, and people wait for a cue that never comes. A gesture
            # gives them one without spending another sentence on it.
            ctx.motion.express_move("happy")
            return IdleResult(listen_for=_LISTEN_S)

        if self._audience_changed(ctx):
            self._begin(ctx, awaiting_face=False)
            ctx.status("New arrivals; welcoming again.")
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        return IdleResult(listen_for=_LISTEN_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        store = ctx.store
        # Stamped for everything heard, handled or not: with no camera this is
        # the only evidence that anybody is still standing here.
        store["spoke_at"] = time.monotonic()
        lowered = text.lower()
        words = set(_WORD_RE.findall(lowered))

        if any(phrase in lowered for phrase in _RESTART_PHRASES):
            # An explicit ask beats every cooldown in this file. A person in the
            # room knows better than any of these heuristics who just walked in.
            self._begin(ctx, awaiting_face=False)
            ctx.status("Welcome restarted on request.")
            return True

        last = store.get("last_said")
        if last and self._wants_a_repeat(lowered, words):
            # "say that again" asks for the sentence that was just spoken. It
            # used to match the restart list and cost the room the whole
            # thirty-two seconds a second time, which is not what anybody who
            # missed one line was asking for.
            said, emotion = last
            self._say(ctx, said, emotion)
            ctx.status("Repeated the last line.")
            return True

        if any(o in lowered for o in _MISSION_OPENERS) and any(s in lowered for s in _MISSION_SUBJECTS):
            # Answered from hub.MISSION_SHORT rather than from the model
            # because it is instant and word for word the claim the Hub signed
            # off on, and "what is this place" is the likeliest question in the
            # room right after the invitation above -- the one moment where a
            # pause for generation is most obvious. Anything else about the Hub
            # falls through to conversation, which grounds its own answers.
            self._say(ctx, hub.MISSION_SHORT, "happy")
            return True

        # Everything else falls through to conversation, which is a better
        # answer than anything a substring match in this file could produce.
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        """Nothing runs in the background here; only the record has to be honest.

        A welcome cut short was never delivered, so finished_at stays unset and
        the cooldown will not suppress the next attempt. The note is for the
        operator, who otherwise cannot tell whether the group heard the end of
        it or the last thing they heard was the robot being switched away.
        """
        line = ctx.store.get("line", 0)
        if line < len(_WELCOME_LINES):
            ctx.status(f"Welcome interrupted after {line} of {len(_WELCOME_LINES)} lines.")
        ctx.store["awaiting_face"] = False

    # --- speaking --------------------------------------------------------

    @staticmethod
    def _say(ctx: DemoContext, text: str, emotion: str) -> None:
        """Speak one line, and remember it as the line a repeat asks for.

        Recorded here rather than at each call site so that every sentence this
        demo says -- script, invitation, mission answer -- is repeatable, and no
        future line can be added that quietly is not.
        """
        ctx.say(text, emotion)
        ctx.store["last_said"] = (text, emotion)

    @staticmethod
    def _wants_a_repeat(lowered: str, words: set[str]) -> bool:
        """Whether that was "sorry, what?" rather than a question."""
        if any(phrase in lowered for phrase in _REPEAT_PHRASES):
            return True
        return len(lowered.split()) <= _REPEAT_MAX_WORDS and bool(words & _REPEAT_WORDS)

    # --- the state of the welcome ----------------------------------------

    @staticmethod
    def _begin(ctx: DemoContext, *, awaiting_face: bool) -> None:
        """Arm the script from line one, for an audience nobody has been told about.

        seen_ids is emptied rather than kept: the ids that matter are the ones
        in front of the robot for THIS delivery. Everyone still standing here is
        recorded again by the next _observe, before the script ends, so nobody
        present is later mistaken for an arrival.
        """
        store = ctx.store
        store["line"] = 0
        store["invited"] = False
        store["awaiting_face"] = awaiting_face
        store["seen_ids"] = set()
        store["new_face"] = False

    @staticmethod
    def _finish(store: dict) -> None:
        """Record the welcome as delivered, and start watching for a new audience.

        The two latches are cleared here rather than in _begin because what
        licenses a replay is the room emptying, or a face arriving, AFTER the
        group heard the welcome. Cleared at the start instead, the empty room
        the robot was waiting in when the operator pressed the button would
        still be latched at the end, and the script would rebound off its own
        ending onto the group that had just listened to it.
        """
        store["finished_at"] = time.monotonic()
        store["room_emptied"] = False
        store["new_face"] = False

    # --- deciding who is in front of the robot ---------------------------

    @staticmethod
    def _audience_arrived(ctx: DemoContext) -> bool:
        """Whether to start the script that is waiting for someone to appear."""
        if ctx.face_visible():
            return True
        return _since(ctx.store, "entered_at") >= _FACE_PATIENCE_S

    @staticmethod
    def _observe(ctx: DemoContext) -> None:
        """Read the camera once per slice and record what it means.

        Both conclusions are latched rather than recomputed when the replay
        decision is taken, for two different reasons.

        The room emptying is a moment: the instant a face reappears the gap
        resets, so a test made one slice later sees a full room and no evidence
        that it was ever empty -- and the slice where the gap crossed may well
        have been blocked by a cooldown that has since expired.

        A face is new only the first time it is seen, because recognition
        genuinely flickers between a real id and 0 from frame to frame. Asking
        later whether the person in view is the one who was welcomed makes every
        one of those flickers look like an arrival, which is what restarted a
        thirty-two second script at a group standing still listening to it. An
        id goes into seen_ids the first time it appears and stays there, so it
        can only ever be new once per welcome.
        """
        if not _faces_available(ctx):
            return
        store = ctx.store
        now = time.monotonic()
        last_seen = store.get("last_face_at")
        # A previous stamp is required: never having seen anyone at all is not
        # evidence that a room emptied, and inf would read as though it were.
        if last_seen is not None and now - last_seen >= _ROOM_EMPTY_S:
            store["room_emptied"] = True
        if not ctx.face_visible():
            return
        store["last_face_at"] = now
        person = ctx.person_id()
        if not person:
            return
        seen = store.setdefault("seen_ids", set())
        if person not in seen:
            # Only counted while there is nothing left to say: someone who
            # arrives part-way through the script is hearing it, not missing it.
            if store.get("line", 0) >= len(_WELCOME_LINES):
                store["new_face"] = True
            seen.add(person)

    @staticmethod
    def _audience_changed(ctx: DemoContext) -> bool:
        """Whether the group in front of the robot is a different one now."""
        store = ctx.store
        if _since(store, "spoke_at") < _QUIET_BEFORE_REPLAY_S:
            return False
        if not _faces_available(ctx):
            return _since(store, "finished_at") >= _BLIND_COOLDOWN_S
        if _since(store, "finished_at") < _SEEN_COOLDOWN_S:
            return False
        if not ctx.face_visible():
            # Somebody has to be there to welcome. The evidence stays latched
            # until they are.
            return False
        return bool(store.get("room_emptied") or store.get("new_face"))
