"""Hold something up and ask the robot what it is.

The most natural thing a person does with a robot that has a camera, and until
now the one thing it could not do: it could recognise your face and not the
book in your hand.

WHY IT ASKS YOU TO HOLD STILL
The head follows faces, so at the moment somebody holds something up the camera
is pointed at their face rather than at the object. Saying "hold it up in front
of me" is not politeness, it is the instruction that makes the picture usable
-- and it doubles as telling somebody they are about to be photographed, which
they are owed.

WHAT LEAVES THE ROOM
One downscaled JPEG per question, to the same model that answers everything
else, and nothing is stored. See brain/looking.py, which bounds all of that:
this demo is only allowed to ask.

IT NEEDS THE INTERNET, AND SAYS SO
The local model has no vision, so this is the one feature with no offline
fallback at all -- on this network that is a routine condition, not an edge
case, and a robot that says "I cannot see just now" is much better than one
that pauses and then answers about something else.
"""

import time

from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S

_ASKING = "asking"
_READY = "ready"
#: An answer has been given and nothing new has been asked. Split from
#: _READY because the two used to be one state, and in that state ANY
#: utterance took another photograph -- live, "Thank you." was answered with
#: "Let me take a look" and a picture of somebody putting a Pepsi can away.
#: Fresh after entering, anything is a question; after an answer, only
#: something that sounds like one is.
_ANSWERED = "answered"

#: The ways people close the exchange. Each gets its manners back, not a
#: camera shutter.
_CLOSINGS = ("thank you", "thats great", "thats brilliant", "thats right",
             "very good", "well done", "cool", "nice one", "perfect", "amazing")

#: And the ways they ask again without a full trigger phrase: anything about
#: identifying or seeing a thing. Deliberately generous -- in this mode, a
#: question-shaped sentence is almost always about the object.
_ASK_AGAIN_WORDS = frozenset(("what", "whats", "look", "see", "this",
                              "holding", "guess", "identify", "recognise",
                              "recognize", "one"))

_BETWEEN_S = 1.0

#: How long to let somebody get the thing in front of the camera. Long enough
#: to take a book out of a bag, short enough that nobody wonders if it heard.
_SETTLE_S = 2.0

#: Said when there is no usable picture. Kept separate from the "cannot reach
#: the model" line because they are different problems for whoever is running
#: the visit: one is the camera, the other is the wifi.
_NO_PICTURE = "I cannot see anything just now -- is my camera covered?"
_NO_MODEL = "I cannot get a look at that just now. My connection is down."


class Look(Demo):
    label = "Look at this"
    help = "Hold something up and ask what it is."
    order = 55
    #: The camera is the whole feature; without it the grid greys this out and
    #: says why, rather than offering something that cannot work.
    requires = ("camera",)
    triggers = (
        "what is this", "what am i holding", "look at this",
        "can you see this", "what do you see", "have a look at this",
        "tell me what this is", "what have i got here",
    )
    claims_utterances = True

    # --- hooks -----------------------------------------------------------

    def on_enter(self, ctx: DemoContext) -> None:
        ctx.store.clear()
        ctx.store["stage"] = _READY
        ctx.say("Hold it up in front of me and ask what it is.", "curious")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        if store.get("stage") == _ASKING:
            return self._answer(ctx)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        lowered = text.lower()
        if ctx.store.get("stage") == _ASKING:
            # Already looking. Swallowed so a visitor repeating themselves does
            # not queue a second picture.
            return True
        if ctx.store.get("stage") == _ANSWERED:
            from demokit.runner import _word_stream, contains_phrase

            words = _word_stream(text)
            if any(contains_phrase(words, c) for c in _CLOSINGS):
                ctx.say("You're welcome. Hold something else up any time.",
                        "happy")
                return True
            if not any(w in _ASK_AGAIN_WORDS for w in words.split()) and not any(
                    t in lowered for t in self.triggers):
                # Not about an object: hand it to the conversation model
                # rather than photographing whoever happens to be in frame.
                return False
        # _ANSWERED reaches here only after surviving the filters above, so
        # what is left IS a new question about an object.
        if (any(t in lowered for t in self.triggers)
                or ctx.store.get("stage") in (_READY, _ANSWERED)):
            # Split across slices: encoding a frame and waiting on the model is
            # well past the runner's six-second hook warning, and doing it here
            # would hold the robot unswitchable for the whole request.
            ctx.store["question"] = text
            ctx.store["stage"] = _ASKING
            ctx.store["asked_at"] = time.monotonic()
            # Announced before the picture is taken, never after. Somebody
            # should know they are being photographed while they still have the
            # chance to lower whatever they are holding.
            ctx.say("Let me take a look.", "curious")
            return True
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        ctx.store["stage"] = _READY

    # --- looking ---------------------------------------------------------

    def _answer(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        # A moment to get the thing in front of the camera, spent waiting
        # rather than grabbing the frame that was live when they finished
        # speaking -- which is a picture of them lowering their hands.
        if time.monotonic() - store.get("asked_at", 0.0) < _SETTLE_S:
            return IdleResult(listen_for=_BETWEEN_S)
        store["stage"] = _ANSWERED

        frame = None
        if ctx.tracker is not None:
            # The tracker's own read, so this never competes with it for the
            # camera.
            frame = ctx.tracker.latest_frame()
        if frame is None:
            ctx.say(_NO_PICTURE, "sad")
            ctx.status("Looked, but there was no camera frame.")
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

        from brain import looking

        answer = looking.describe(frame, store.get("question", ""))
        if not answer:
            ctx.say(_NO_MODEL, "sad")
            ctx.status("Could not reach the model to look at that.")
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

        ctx.status(f"Looked: {answer[:80]}")
        ctx.say(answer, "happy")
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)
