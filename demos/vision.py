"""Face tracking, narrated, so a visitor can watch it work instead of being told.

The tracking is not this demo's doing: body/face_tracker.py aims the head at
whoever is in view under every demo, all the time. What this adds is
commentary, because head tracking is nearly invisible without it. From two
metres away a head that follows you reads as idle motion, and the question it
actually raises in a room of business students -- is that thing recording me --
gets thought rather than asked.

So everything here is about rationing speech rather than producing it:

Presence is sampled once per idle slice, which is every couple of seconds. A
robot that remarks on every sample is unbearable inside a minute, so every line
passes a cooldown and comes out of a rotating pool. Hearing the same sentence
for the same event twice is what turns "it noticed me" back into "it is running
a loop", which is the illusion this demo exists to create. The rotation
counters outlive a visitor on purpose -- see on_enter -- because the pools are
there to stop the second group of the day hearing a recording of the first.

Nothing spoken here is a paragraph. Every line is a tuple of sentences handed
to ctx.say_lines, and the intro is one sentence per idle slice, because nothing
consumes the microphone while the robot talks: a forty-word block is twenty
seconds in which a visitor cannot interrupt and an operator's mode switch is
not looked at.

A detection is one bounding box in one frame at MODELS.face_detection_fps, and
it drops for a turned head, a raised hand, or someone crossing in front.
Arrival is therefore acted on the moment it is seen -- that responsiveness is
the thing being demonstrated -- and the line announcing it is retried every
slice until the quiet floor lets it through, because a face appearing is the
moment this demo exists for and a cooldown left over from the intro must not
eat it. The retry stops at the first presence milestone, twenty seconds in,
where "there you are" has stopped being true. A departure, by contrast, has to
persist for _LOST_GRACE_S before it counts, or the robot says goodbye to
someone who looked down at their phone.

Deliberately absent: any call to ctx.motion.express_move(). A recorded move
pauses the motion loop for its duration (see MotionController._run), and that
loop is what applies the tracking target -- so a flourish would freeze the head
mid-demo, in the one demo whose whole subject is the head not freezing.
"""

import time

from brain import hub
# Read, not quoted: the detection rate is a number this demo says out loud, and
# config.py is where it is actually set. :g so a fractional rate reads as "0.5"
# instead of being rounded to a confident zero.
from config import MODELS
from demokit import Demo, DemoContext, IdleResult

_RATE = f"{MODELS.face_detection_fps:g}"

#: How stale a detection may be and still count as "someone is here". The
#: tracker refreshes at MODELS.face_detection_fps and holds an aim for 1.5s;
#: this is a little longer, so a couple of missed frames read as a person
#: standing still rather than as a person leaving.
_PRESENT_AGE_S = 2.0

#: Detections must stop for this long before a departure is believed and
#: spoken. Tuned by what it costs to be wrong: announcing a departure that
#: didn't happen is far more noticeable than announcing a real one four
#: seconds late.
_LOST_GRACE_S = 4.0

#: Floor between any two narrated lines, whatever happened. Without it, a
#: visitor stepping in and out of frame gets a running commentary on their own
#: shoulder movements.
_QUIET_S = 8.0

#: Seconds of unbroken presence before each successive aside. Spaced so the
#: second one lands after a group has stopped watching the head and started
#: talking to each other, which is exactly when an invitation to move works.
_FOLLOW_AFTER_S = (20.0, 55.0, 110.0)

#: Silence about the search sweep until nobody has been in view this long. The
#: sweep only starts a few seconds into an empty frame, and explaining a
#: movement before it happens is worse than not explaining it.
_SEARCH_AFTER_S = 20.0

#: Listening windows. Shorter while deciding whether someone has left, because
#: that decision is what the next spoken line depends on; longest with an empty
#: frame, where there is nobody to narrate to and the microphone is the more
#: useful thing to be doing.
_WATCH_LISTEN_S = 2.0
_LOOK_AGAIN_S = 1.0
_EMPTY_LISTEN_S = 3.0
_INTRO_GAP_S = 1.0

#: Store keys holding rotation counters. Prefixed so on_enter can keep them
#: while wiping everything else; a constant rather than a literal in two places
#: because the two must agree or the counters are silently lost again.
_LINE_PREFIX = "line:"

#: One sentence per entry, one entry per idle slice. The intro is the stretch a
#: visitor is most likely to interrupt with a question, and each entry is a
#: window in which the microphone is open and a mode switch is checked.
_INTRO = (
    ("There is one camera, behind my eyes.", "neutral"),
    (f"A face detector runs on it about {_RATE} times a second, on the machine that runs me.", "neutral"),
    ("None of it goes to the internet.", "neutral"),
    ("Each detection is a box around a face.", "curious"),
    ("I take the middle of the largest box and turn it into an angle for my neck.", "curious"),
    ("That is the whole trick.", "curious"),
    ("So stand in front of me and move, slowly.", "happy"),
    (
        "Watch my head rather than my face: the camera is fixed behind my eyes, so the whole "
        "head has to turn.",
        "happy",
    ),
    ("On the question everyone asks: the frames are looked at and thrown away.", "neutral"),
    ("I keep no video and no photographs.", "neutral"),
    (
        "What gets written down is a list of numbers describing a face, and only for people "
        "who have told me their name.",
        "neutral",
    ),
)

#: Each pool entry is a tuple of sentences, not a paragraph, so _narrate can
#: hand it to ctx.say_lines and the demo stays switchable between sentences.
_ARRIVAL = (
    (
        "There you are.",
        "A box just appeared in the frame, and my neck is following the middle of it.",
    ),
    (
        "Got you.",
        "The nearest face is the biggest one in shot, and that is the one I follow.",
    ),
    (
        "Someone is in front of me again.",
        "Watch the head rather than the eyes -- the eyes are fixed, the neck is what moves.",
    ),
)

_DEPARTURE = (
    (
        "And gone.",
        "No box in the frame, so my head goes back to resting instead of staring at where you "
        "used to be.",
    ),
    (
        "I have lost you.",
        "Holding the last aim would look more attentive and be less honest, so I let it expire.",
    ),
    (
        "Out of shot.",
        "Nothing to follow, so the aim drifts back to centre on its own.",
    ),
)

#: One pool per milestone, so a visitor who stays two minutes gets three
#: different points rather than the same one three times. A second group later
#: that day gets different wording because the rotation counters survive
#: on_enter's wipe -- they count lines spent, not anything about a visitor.
_FOLLOWING = (
    (
        (
            "That is about {seconds} seconds of you steering my head.",
            "The aim is smoothed between frames, which is why it trails you slightly instead "
            "of snapping.",
        ),
        (
            "You have had my head for {seconds} seconds now.",
            "Every one of those was a fresh detection -- nothing here remembers where you "
            "were, it just keeps finding you.",
        ),
    ),
    (
        (
            "Try stepping slowly to one side, then the other.",
            "The limit you will find is my neck rather than the camera: it runs out of travel "
            "before you run out of frame.",
        ),
        (
            "Move sideways and I will follow until my neck stops.",
            "The camera can still see you past that point, which is a mechanical limit rather "
            "than a seeing one.",
        ),
    ),
    (
        (
            "If two of you stand here, I follow whichever face is largest in the frame, which "
            "is normally whoever is closest.",
            "Swap places and I will swap with you.",
        ),
        (
            "My expression and my aim are added together rather than fighting.",
            "I can look thoughtful without tilting the camera down at your shoes.",
        ),
    ),
)

_RECOGNISED = (
    (
        "And I know this face.",
        "That is a match against a stored set of numbers, not against a saved photograph.",
    ),
    (
        "I recognise you, which means somebody introduced you to me before.",
        "The comparison is number by number, and it happens on this machine.",
    ),
)

_SEARCHING = (
    (
        "Nobody in shot.",
        "The slow arc my head is doing now is a search -- someone sitting down is in a "
        "different place to someone standing up, so I sweep rather than wait.",
    ),
    (
        "Still an empty frame.",
        "I will keep sweeping.",
        "Staring straight ahead only ever finds people who stand exactly where I expect them.",
    ),
)


def _next_line(store: dict, bucket: str, pool: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """The next line from a pool, rotating rather than choosing at random.

    random.choice repeats verbatim roughly as often as a coin lands the same
    way twice, and a verbatim repeat is precisely what makes narration sound
    mechanical. Rotation spends the whole pool before anything is heard again.
    """
    key = f"{_LINE_PREFIX}{bucket}"
    index = store.get(key, 0)
    store[key] = index + 1
    return pool[index % len(pool)]


class Vision(Demo):
    label = "Vision & Face Tracking"
    help = "Follows a face and says what the camera, the detector and the neck are each doing."
    order = 40
    requires = ("faces",)
    #: "face tracking" alone is the name of a topic taught two floors up, so it
    #: fires on "face tracking is part of the vision module" as readily as on a
    #: request for the demonstration. The request forms below cannot be said
    #: about the subject in the abstract.
    triggers = ("show me face tracking", "demo the face tracking", "can you see me")
    # Not claims_utterances: this demo asks no questions, so it has no answer
    # of its own to protect. Someone who says "let's dance" while standing in
    # front of it should get the dance.

    def on_enter(self, ctx: DemoContext) -> None:
        # Cleared here rather than in on_exit because the runner keeps one
        # store per demo id for the life of the process and never empties it.
        # Without this, the second group of the day would be dropped into the
        # middle of the first group's demo, intro already spent.
        #
        # The rotation counters are carried across the wipe. They hold nothing
        # about a visitor -- just how many lines of each pool have been spent --
        # and clearing them along with everything else meant every group heard
        # the first line of every pool, which is exactly the recording the pools
        # exist to avoid.
        rotation = {key: value for key, value in ctx.store.items() if key.startswith(_LINE_PREFIX)}
        ctx.store.clear()
        ctx.store.update(rotation)
        ctx.status("Vision: narrating what the face tracker sees")
        ctx.say("Right, vision. Stand where I can see you.", "curious")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        now = time.monotonic()

        step = store.get("intro", 0)
        if step < len(_INTRO):
            text, emotion = _INTRO[step]
            store["intro"] = step + 1
            ctx.say(text, emotion)
            store["spoke_at"] = time.monotonic()
            # A gap rather than none: it reads as speech instead of a
            # recording, and it is a second in which someone can interrupt.
            return IdleResult(listen_for=_INTRO_GAP_S)

        if ctx.face_visible(max_age_s=_PRESENT_AGE_S):
            return self._watching(ctx, now)
        return self._empty_frame(ctx, now)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        lowered = text.lower()

        # "can you see me" is also this demo's trigger phrase, so it is the
        # first thing most visitors say on arriving here. Answering it from the
        # live tracker rather than from the language model is the entire point:
        # the answer has to be checkable by stepping sideways.
        if "see me" in lowered or "seeing me" in lowered or "see anyone" in lowered:
            if ctx.face_visible(max_age_s=_PRESENT_AGE_S):
                return self._answer(
                    ctx,
                    (
                        "Yes.",
                        "You are in frame right now, and my head is aimed at the middle of "
                        "your face.",
                        "Step to one side and watch it follow you.",
                    ),
                    "happy",
                )
            return self._answer(
                ctx,
                (
                    "Not at the moment.",
                    "There is no face in the frame, so I am sweeping.",
                    "Move into the space in front of me and I will pick you up within a second.",
                ),
                "curious",
            )

        if any(
            phrase in lowered
            for phrase in ("who am i", "know me", "recognise me", "recognize me", "know who i am")
        ):
            if ctx.person_id():
                return self._answer(
                    ctx,
                    (
                        "Your face matches somebody I have been introduced to before.",
                        "That is a comparison against stored numbers, done here, in about a "
                        "millisecond.",
                    ),
                    "happy",
                )
            return self._answer(
                ctx,
                (
                    "You are a stranger to me.",
                    "Nothing matches, which is what happens for everyone who has not told me "
                    "their name.",
                    "I can see your face; I just cannot put a name to it.",
                ),
                "curious",
            )

        if any(
            phrase in lowered
            for phrase in ("recording", "record me", "privacy", "storing", "stored", "photo", "saving")
        ):
            return self._answer(
                ctx,
                (
                    "No video is kept and no pictures are saved.",
                    "A frame is looked at, used to aim my head, and dropped.",
                    "The only thing that outlives it is a list of numbers describing a face, "
                    "and only for people who gave me their name.",
                ),
                "neutral",
            )

        # "how" plus something to do with seeing, rather than "how" alone: a
        # bare "how does that work" is fine here, but "how are you" is a
        # greeting and belongs to ordinary conversation, not to a lecture on
        # bounding boxes.
        if "how" in lowered and any(
            word in lowered for word in ("see", "seeing", "track", "follow", "camera", "detect", "eyes", "work")
        ):
            return self._answer(
                ctx,
                (
                    "A camera frame goes to a face detector.",
                    "The detector hands back a box.",
                    "The middle of the largest box becomes a left-right and up-down angle for "
                    "my neck, smoothed a little so it does not snap.",
                    "That is all of it.",
                ),
                "thinking",
            )

        # The one Hub claim this demo makes, taken from brain/hub.py rather
        # than written here, because a business-school audience asks what it is
        # for long before they ask how it works.
        if any(word in lowered for word in ("matter", "the point", "why bother", "useful", "so what")):
            return self._answer(
                ctx,
                (hub.MISSION_SHORT, "Noticing where a person is, is the cheap end of that."),
                "neutral",
            )

        return False

    def on_exit(self, ctx: DemoContext) -> None:
        # Nothing of this demo's is running -- it starts no threads and holds
        # no hardware. Worth a dashboard note anyway, because the head keeps
        # following faces after the operator switches away and that looks like
        # this demo failing to stop.
        ctx.status("Vision: narration off. Face tracking itself keeps running.")

    # --- narration -------------------------------------------------------

    def _watching(self, ctx: DemoContext, now: float) -> IdleResult:
        store = ctx.store
        # A gap that never reached _LOST_GRACE_S was a blink, not a departure,
        # so the presence session survives it and "following you for a minute"
        # stays true across the wobble.
        store["gone_since"] = None

        if not store.get("present"):
            store.update(
                present=True, present_since=now, stage=0,
                recognised=False, searched=False, announced_arrival=False,
            )
            store["absent_since"] = None
            ctx.status("Vision: face acquired")

        # Presence latches on the first detection because the milestones are
        # timed from it, but the arrival LINE is owned by its own flag and
        # retried every slice until it is actually spoken. Latching both at once
        # lost the headline moment in the ordinary case: an operator selecting
        # this demo with a group already in front of the robot gets presence at
        # the end of the intro, inside the quiet floor left by the last intro
        # sentence, so the one attempt was swallowed and never made again.
        if not store.get("announced_arrival"):
            if self._narrate(ctx, "arrival", _ARRIVAL, "happy"):
                store["announced_arrival"] = True
            elif now - store.get("present_since", now) >= _FOLLOW_AFTER_S[0]:
                # Given up on rather than said late: twenty seconds in, "there
                # you are" is no longer true, and the first milestone below is
                # the honest thing to say instead.
                store["announced_arrival"] = True
            # Nothing else is worth attempting this slice -- every other line
            # goes through the same quiet floor, so it would be refused too.
            return IdleResult(listen_for=_WATCH_LISTEN_S)

        # Recognition is checked every slice but spoken once per visit, and
        # only once it has actually been spoken -- a line the cooldown
        # swallowed should arrive late rather than be lost.
        if not store.get("recognised") and ctx.person_id():
            if self._narrate(ctx, "known", _RECOGNISED, "surprised"):
                store["recognised"] = True
            return IdleResult(listen_for=_WATCH_LISTEN_S)

        stage = store.get("stage", 0)
        held = now - store.get("present_since", now)
        if stage < len(_FOLLOWING) and held >= _FOLLOW_AFTER_S[stage]:
            # Rounded to five: "about twenty seconds" is a claim a listener can
            # check against their own sense of time, where "about twenty three
            # seconds" invites them to check the arithmetic instead.
            spoken_seconds = int(round(held / 5.0) * 5)
            if self._narrate(ctx, f"follow{stage}", _FOLLOWING[stage], "curious", seconds=spoken_seconds):
                store["stage"] = stage + 1

        return IdleResult(listen_for=_WATCH_LISTEN_S)

    def _empty_frame(self, ctx: DemoContext, now: float) -> IdleResult:
        store = ctx.store

        if store.get("present"):
            gone_since = store.get("gone_since")
            if gone_since is None:
                store["gone_since"] = now
                return IdleResult(listen_for=_LOOK_AGAIN_S)
            if now - gone_since < _LOST_GRACE_S:
                return IdleResult(listen_for=_LOOK_AGAIN_S)
            store["present"] = False
            # Dated from the last detection, not from now, so the sweep is
            # explained a fixed time after they actually left rather than a
            # grace period later.
            store["absent_since"] = gone_since
            store["gone_since"] = None
            ctx.status("Vision: face lost")
            # One attempt, and dropped if the quiet floor refuses it -- unlike
            # the arrival above. "And gone" is only worth saying while the room
            # still remembers the person leaving; retried a quarter of a minute
            # later it narrates an empty frame nobody is looking at. The sweep
            # line below covers the empty frame from then on.
            self._narrate(ctx, "departure", _DEPARTURE, "curious")
            return IdleResult(listen_for=_LOOK_AGAIN_S)

        absent_since = store.get("absent_since")
        if absent_since is None:
            absent_since = store["absent_since"] = now
        if not store.get("searched") and now - absent_since >= _SEARCH_AFTER_S:
            if self._narrate(ctx, "searching", _SEARCHING, "curious"):
                store["searched"] = True

        # Nobody to narrate to: hand the slice back to the wake word, which is
        # both the more useful thing to be doing and the reason an empty room
        # does not spin this demo at full speed.
        return IdleResult(listen_for=_EMPTY_LISTEN_S)

    def _narrate(
        self,
        ctx: DemoContext,
        bucket: str,
        pool: tuple[tuple[str, ...], ...],
        emotion: str,
        **fields: object,
    ) -> bool:
        """Speak one line from a pool if the robot has been quiet long enough.

        A line is a tuple of sentences and goes out through say_lines, so the
        microphone is between sentences rather than after a paragraph and an
        operator's switch is noticed a sentence later.

        Returns whether it was spoken, so callers can decide between retrying
        later and dropping the line -- a departure that missed its moment is
        not worth saying, an arrival or a milestone is.
        """
        now = time.monotonic()
        if now - ctx.store.get("spoke_at", 0.0) < _QUIET_S:
            return False
        sentences = _next_line(ctx.store, bucket, pool)
        ctx.say_lines([sentence.format(**fields) for sentence in sentences], emotion)
        ctx.store["spoke_at"] = time.monotonic()
        return True

    def _answer(self, ctx: DemoContext, lines: tuple[str, ...], emotion: str) -> bool:
        """Answer a question and restart the narration cooldown.

        `lines` is one sentence per entry rather than a paragraph per entry:
        say_lines checks for a mode switch between them, so an operator who
        switches away during a four-sentence answer waits for the end of a
        sentence rather than the end of the answer.

        Stamping spoke_at is the load-bearing part: without it the next idle
        slice can follow an answer with an unrelated aside a second later,
        which sounds like two robots sharing a speaker.
        """
        ctx.say_lines(lines, emotion)
        ctx.store["spoke_at"] = time.monotonic()
        return True
