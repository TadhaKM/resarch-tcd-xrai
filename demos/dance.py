"""Dancing: the demonstration that makes people walk over.

The whole of this demo is one call, `ctx.motion.dance()`, plus the discipline
of not trying to be clever about when to make it. The choreography runs about
4.4 seconds on its own thread and the controller now drops a second request
while one is in flight (see MotionController.dance -- overlapping dancers
corrupt each other's pause state and can starve the SDK's liveness check into
restarting the process). So this asks on every idle pass and lets the guard
say no. The alternative -- tracking when the last dance started and sleeping
the difference -- means a demo holding a private copy of a duration that lives
in body/motion.py, and it goes wrong the first time anyone retunes the
choreography.

The other half of the job is that a robot bobbing in total silence for two
minutes stops being a demonstration and becomes furniture. So it says
something every few passes, with the gap redrawn each time: a fixed cadence
reads as a loop, which is precisely the impression to avoid.
"""

import random
import re

from demokit import Demo, DemoContext, IdleResult


#: Listening window between dance requests. The core clamps this to its own
#: maximum, so the number only ever shortens the pass. 2.5s against a ~4.4s
#: choreography means requests land at roughly 0s (taken), 2.5s (dropped),
#: 5.0s (taken) -- about half a second of stillness between bars. This is not
#: synchronisation and must not become it: if the choreography's length
#: changes, the duty cycle shifts and nothing breaks.
_LISTEN_SLICE_S = 2.5

#: Idle passes between spoken asides, redrawn after each one. At ~2.5s a pass
#: that is roughly nine to eighteen seconds of dancing per remark, which in
#: front of a group is often enough to hold attention and rare enough that the
#: robot is mostly moving rather than mostly talking. Every remark is also a
#: window during which nothing is consuming the microphone, so more is worse.
#: Passes between asides. Deliberately long: dancing is the demonstration, and
#: a robot narrating its own dance is a robot talking over the thing people
#: came to watch. Every remark is also a few seconds of deafness. Heard live,
#: the earlier cadence read as chatter, so this is roughly one aside a minute
#: rather than one every twenty seconds.
_PASSES_BETWEEN_REMARKS = (12, 20)

#: Asides. Written for the ear and kept to a couple of seconds each, because
#: the robot is deaf for the length of every one. Each says something true
#: about what the audience is watching -- the point of an embodied demo is
#: that the explanation and the thing being explained are in the same room.
_REMARKS = (
    "All neck. No legs.",
    "Nobody choreographed this.",
    "No music. Keeping time myself.",
    "Join in. I'm not judging.",
    "Still going.",
    "This is my best move.",
)

#: Whole words, not substrings: see _mentions. "stop" alone covers "stop
#: dancing" and "enough" covers "that's enough", so both are left out.
_STOP_CUES = ("stop", "enough", "settle down", "sit still", "no more")

#: Spoken requests to dance. Shared between the class-level triggers and the
#: encore check, because they are the same sentences: while this demo is
#: already selected the runner does not re-trigger on its own phrases (same
#: demo id), so "let's dance" arrives at on_utterance instead -- which is how a
#: paused dancer is started again without an operator touching the dashboard.
_TRIGGERS = ("let's dance", "lets dance", "can you dance", "do a dance")

#: Ways of asking for another bar, on top of _TRIGGERS. Every one of these is
#: imperative in shape, which is the property that matters. Bare "dance" and
#: "dancing" used to be here, argued for on the grounds that a wrong match only
#: costs an extra bar. That was wrong: the extra bar is danced *instead of*
#: answering. "why do you dance like that" got "Once more, then." and nothing
#: else, as did "how does the dance work" and "who wrote the dance" -- a noun
#: that every genuine question about the dancing has to contain cannot tell a
#: question from a request. "dance again" and "go again" need no entry of their
#: own: _mentions matches whole words, so the bare "again" below covers both.
_ENCORE_CUES = ("again", "one more", "another", "encore", "keep going", "keep dancing")

#: "Don't stop!" is a normal thing to shout at a dancing robot, and it contains
#: a stop cue. Cheap to detect, and stopping is the exact opposite of what was
#: asked for.
_NEGATIONS = ("dont", "don't", "do not", "never")

#: A transcript opening with one of these is a question about the dancing, not
#: an instruction about it, and belongs to conversation. Only question words:
#: auxiliaries ("can you", "do it", "will you") open requests at least as often
#: as questions, and "can you dance" is one of this demo's own triggers, so
#: treating them as interrogative would cost more than it saves.
_QUESTION_OPENERS = ("what", "why", "how", "who", "whom", "whose", "when", "where", "which")


def _mentions(text: str, cues: tuple[str, ...]) -> bool:
    """Whether any cue appears in `text` as whole words.

    Substring matching is what the trigger table does and it is wrong at this
    scale: "again" is inside "against", "more" is inside "tomorrow", and a
    visitor asking a real question mid-demo would get a shrug and another bar
    instead of an answer. Apostrophes are kept in the token set so "that's"
    and "don't" survive as single words; the recognizer emits no other
    punctuation worth preserving.
    """
    padded = " " + " ".join(re.findall(r"[a-z']+", text.lower())) + " "
    return any(f" {cue} " in padded for cue in cues)


def _opens_as_question(text: str) -> bool:
    """Whether the transcript starts like a question rather than an instruction.

    Position is what separates the two: "again" anywhere in a sentence is a
    request, but a question word only means a question when it is the first
    thing said. Checked on the opening word alone, because the recognizer emits
    no punctuation and there is no question mark to look for.
    """
    words = re.findall(r"[a-z']+", text.lower())
    if not words or words[0] not in _QUESTION_OPENERS:
        return False
    # "how about one more" and "what about another" open with a question word
    # and are suggestions, not questions. Letting them through to the cues is
    # the difference between another bar and the language model cheerfully
    # agreeing to dance, which it has no way of making happen.
    return len(words) < 2 or words[1] != "about"


class Dance(Demo):
    label = "Dance"
    help = "Dances on a loop with the occasional aside. Say stop to settle it."
    order = 80
    #: No bare "dance": as a substring it fires inside ordinary sentences
    #: ("dance studios in Dublin") and would drag the robot out of whatever
    #: demo it was in mid-answer. Every phrase here is an actual request.
    triggers = _TRIGGERS

    def on_enter(self, ctx: DemoContext) -> None:
        """Start moving, then say one line over the top of it."""
        store = ctx.store
        # The store outlives a selection and is never cleared between visitors
        # (the runner keeps one per demo id for the life of the process), so a
        # demo re-entered after someone said "stop" would otherwise come back
        # already paused and sit there doing nothing. The remark deck below is
        # deliberately left alone: carrying it over is what stops the first
        # group's last aside being the second group's first.
        store["paused"] = False
        store["until_remark"] = random.randint(*_PASSES_BETWEEN_REMARKS)
        # The runner hands the phrase that switched us here straight to
        # on_utterance after this hook, and that phrase is by definition a
        # request to dance. This flag lets on_utterance recognise its own
        # trigger and stay quiet rather than answer it a second time.
        store["just_entered"] = True
        # Motion before speech: dance() hands off to another thread and returns
        # at once, so the head is already moving underneath the opening line
        # rather than starting after it. The line is the one this hook is
        # allowed -- everything else this demo says comes from on_idle.
        ctx.motion.dance()
        ctx.say("Right. Give me some room.", "happy")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        # Entered from the dashboard rather than by a trigger phrase: no
        # utterance is coming to consume the flag, and the next thing anyone
        # says is a fresh request, not the one that selected this demo.
        store["just_entered"] = False
        if store.get("paused"):
            # Told to stop, but still selected and still listening: getting it
            # going again should not mean an operator reaching for the
            # dashboard in front of people.
            return IdleResult(listen_for=_LISTEN_SLICE_S)

        # Asked unconditionally. Whether this starts a dance or is dropped is
        # the controller's business, and it does not tell us which -- see the
        # module docstring for why that is the right side of the line.
        ctx.motion.dance()

        remaining = int(store.get("until_remark", 0)) - 1
        if remaining <= 0:
            ctx.say(self._next_remark(store), "happy")
            remaining = random.randint(*_PASSES_BETWEEN_REMARKS)
        store["until_remark"] = remaining
        return IdleResult(listen_for=_LISTEN_SLICE_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        store = ctx.store
        # Consumed whether or not this turns out to be a request, because the
        # moment it describes -- the utterance that arrived with the switch --
        # has passed either way.
        just_entered = bool(store.pop("just_entered", False))

        # Before any cue matching: a visitor who asks a question while the
        # robot dances is owed an answer, and every cue below can appear inside
        # one ("do it again" vs "how often do you do that again"). Answering a
        # question with another bar is the one failure this demo can make that
        # a visitor cannot work around, since the same words keep failing.
        if _opens_as_question(text):
            return False

        asked_to_stop = _mentions(text, _STOP_CUES)
        negated = _mentions(text, _NEGATIONS)

        if asked_to_stop and not negated:
            store["paused"] = True
            ctx.status("Dancing paused by voice")
            # The bar already in flight cannot be called back -- the controller
            # exposes no way to interrupt one -- so the head keeps moving under
            # this line for up to a few more seconds. That reads as winding
            # down rather than as a fault, which is why this says "done" rather
            # than claiming to have stopped.
            ctx.say("Okay, that's me done.", "neutral")
            return True

        if _mentions(text, _ENCORE_CUES + _TRIGGERS) or (asked_to_stop and negated):
            store["paused"] = False
            store["until_remark"] = random.randint(*_PASSES_BETWEEN_REMARKS)
            if just_entered:
                # This is the phrase that selected the demo a second ago, and
                # on_enter has already started dancing and said its line.
                # Claimed rather than answered: a second line here is two more
                # seconds with nothing on the microphone, and "Once more, then"
                # is a strange thing to say about the first time.
                return True
            # Likely dropped: by the time a wake word, an acknowledgement and a
            # transcription have gone by, the previous bar is usually still
            # finishing. The next idle pass is a couple of seconds away and
            # will ask again, so the promise below is kept either way.
            ctx.motion.dance()
            ctx.say("Once more, then.", "happy")
            return True

        # Anything else is a real question asked of a robot that happens to be
        # dancing. Hand it to conversation rather than answering it with a jig.
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        """Switched away. Note what is still running, since it cannot be stopped."""
        ctx.store["paused"] = True
        # Worth saying on the dashboard: the next demo's first line can arrive
        # while this head is still swinging, and because the dance holds the
        # motion loop paused, that demo's expression pose will not appear until
        # the bar ends. An operator seeing it needs to know it is the tail of
        # the dance, not the new demo misbehaving.
        ctx.status("Dance stopping -- the last few beats play out first")

    def _next_remark(self, store: dict) -> str:
        """Next aside, dealt from a shuffled deck rather than picked at random.

        Random choice repeats within three or four draws often enough that an
        audience notices, and hearing the same line twice in a minute is what
        makes a demo look like a loop instead of a robot.
        """
        deck = store.get("deck") or []
        if not deck:
            deck = list(_REMARKS)
            random.shuffle(deck)
            # A reshuffle can put the line just spoken back on top, which is the
            # one repeat the deck exists to prevent.
            if len(deck) > 1 and deck[-1] == store.get("last_remark"):
                deck[0], deck[-1] = deck[-1], deck[0]
        line = deck.pop()
        store["deck"] = deck
        store["last_remark"] = line
        return line
