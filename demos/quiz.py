"""A short quiz about the Hub and about AI, run for a group.

Built for the audience that the rest of this robot serves least well. A school
group does not want to be explained to for twenty minutes; they want a turn.
Everything else here is the robot talking and the room listening -- this is the
room talking and the robot listening, which is the same hardware pointed the
other way round.

WHY THE ANSWERS ARE MATCHED LOOSELY
Thirty teenagers shouting at a microphone produces transcripts like "um is it
the head one" and "ROBOTS INNIT". A quiz that only accepts an exact answer is a
quiz nobody wins, so each question carries the words that would make an answer
right, and hearing any of them counts. Being generous here costs nothing --
nobody is being graded -- and being strict costs the whole point.

WHY THE QUESTIONS ARE WRITTEN DOWN
The obvious version generates them, and it is worse: the model invents facts
about the Hub, cannot be checked before a visit, and takes a second or two per
question in front of a group that is already losing interest. These are fixed,
true, and answerable by somebody who has been in the room ten minutes -- which
is what makes the quiz feel like a reward for paying attention rather than a
test of what they knew already.
"""

import random
import time

from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S

_ASK = "ask"
_SCORE = "score"
_DONE = "done"

#: (question, accepted words, the line said when they get it). Every answer is
#: a word somebody would actually shout, not a sentence. The "said when right"
#: line carries the fact, so a wrong answer still teaches the thing.
_QUESTIONS = (
    (
        "What does the X R in AI XR Hub stand for?",
        ("extended reality", "extended", "reality", "xr"),
        "Extended reality. Virtual and augmented reality, both.",
    ),
    (
        "Am I doing my thinking in this room, or somewhere else?",
        ("here", "this room", "laptop", "locally", "room", "inside"),
        "On a laptop right here. No internet needed for most of what I do.",
    ),
    (
        "How do I know where to look when you move?",
        ("camera", "eyes", "see", "vision", "face"),
        "A camera, and a model that finds faces about ten times a second.",
    ),
    (
        "What can I not do: dance, tell a story, or walk across the room?",
        ("walk", "walk across", "move", "wheels", "legs"),
        "Walk. I have no wheels and no legs -- I am bolted to this spot.",
    ),
    (
        "What university am I part of?",
        ("trinity", "trinity college", "tcd", "dublin"),
        "Trinity College Dublin, in the Business School.",
    ),
    (
        "True or false: I remember people I have met before.",
        ("true", "yes", "yeah", "remember", "you do"),
        "True. I learn faces and names, and I know you next time.",
    ),
)

#: Questions in one round. Six is about four minutes with a lively group, which
#: is where a standing audience is still with you.
_ROUND = 4

#: Tries at one question before the answer is given and the quiz moves on. A
#: group that does not know it will not know it on the third go either, and
#: the silence is where the energy dies.
_MAX_TRIES = 2

#: Idle slices a question waits before counting as unanswered. Longer than the
#: advisor's: a group has to confer and argue first, which is the fun part.
_MAX_WAITS = 5

_BETWEEN_LINES_S = 1.0
_SESSION_STALE_S = 600.0


#: Words that make up politeness rather than answers. An utterance built
#: ENTIRELY of these is somebody being nice to the robot mid-quiz, and it must
#: cost nothing -- see on_utterance.
_FILLER_WORDS = frozenset({
    "thank", "thanks", "you", "okay", "ok", "cheers", "nice", "cool", "ha",
    "haha", "hmm", "um", "uh", "oh", "eh", "ah", "wow", "reachy", "ricky",
    "richie", "rit", "please", "sorry",
})


def _all_filler(heard: str) -> bool:
    """Whether an utterance is courtesy noise rather than an answer attempt."""
    from demokit.runner import _word_stream

    words = _word_stream(heard).split()
    return bool(words) and all(w in _FILLER_WORDS for w in words)


def _is_right(heard: str, accepted: tuple[str, ...]) -> bool:
    """Whether a shouted answer counts. Generous on purpose -- see the docstring."""
    from demokit.runner import _word_stream

    words = _word_stream(heard)
    return any(f" {_word_stream(a).strip()} " in words for a in accepted)


class Quiz(Demo):
    label = "Quiz the group"
    help = "Runs a short quiz about the Hub and AI, and keeps score."
    order = 65
    persona = "friendly"
    #: Short phrases here are EXACT-only: the fuzzy matcher in demokit/runner.py
    #: refuses anything under three words or twelve characters, because a short
    #: phrase is one mishearing away from everything. That is why "quiz us"
    #: alone is fragile -- heard live as "Quizance", which matched nothing and
    #: fell through to conversation. The longer phrasings below all clear the
    #: bar, so they survive being misheard; the short ones stay for when the
    #: recogniser gets it right.
    triggers = (
        "quiz us", "quiz me",
        "start the quiz", "lets do a quiz", "can we do a quiz",
        "quiz the group", "ask us some questions", "give us some questions",
        "test us on the hub",
        # Alternate SPELLINGS, not alternate phrasings. "quiz us" said normally
        # comes back from Whisper as one invented word -- observed live three
        # times in a row as "Quizance", "Quizas" and "Quizzos" -- and a
        # two-word trigger is below the fuzzy matcher's floor, so none of them
        # could be rescued. Listing what the recogniser actually produces is
        # the same fix assets/wake_phrases.txt uses to wake on "Ricky" and
        # "Richie": the robot should answer to how people are HEARD, not to how
        # the phrase is spelled.
        "quizance", "quizas", "quizzos", "quizus", "quizzes", "quiz as",
    )
    #: A shouted answer must not be read as another demo's trigger word --
    #: "robots" and "dance" are both plausible answers here.
    claims_utterances = True

    # --- hooks -----------------------------------------------------------

    def on_enter(self, ctx: DemoContext) -> None:
        store = ctx.store
        fresh = time.monotonic() - store.get("touched", 0.0) < _SESSION_STALE_S
        if not (fresh and store.get("stage") not in (None, _DONE)):
            self._reset(ctx)
        store["touched"] = time.monotonic()
        ctx.say(f"Right -- {_ROUND} questions. Shout the answers.", "happy")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        stage = ctx.store.get("stage")
        if stage == _ASK:
            return self._ask(ctx)
        if stage == _SCORE:
            return self._final_score(ctx)
        if stage is None:
            self._reset(ctx)
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        from demokit.runner import _word_stream

        store = ctx.store
        lowered = text.lower()
        if any(p in lowered for p in ("start again", "another round", "again please")):
            self._reset(ctx)
            ctx.say("Another round then.", "happy")
            return True
        if not store.get("awaiting"):
            # The phrase that started the quiz, handed straight back by the
            # runner right after on_enter answered it. Swallowed, or the turn
            # falls through to the conversation demo and TWO demos answer the
            # same sentence: live, "quiz us" got "Right -- 4 questions. Shout
            # the answers." and then, over the top of it, "Fun idea! Want me to
            # quiz you on the AI XR Hub, or something related to picking
            # between those two Master's programmes?" -- fifteen seconds of the
            # robot talking to itself before question one.
            # Matched through the same word-stream the runner used to SWITCH
            # here, not by raw substring. Live, "Let's do a quiz" selected this
            # demo -- the runner strips apostrophes -- and then fell straight
            # through this check, because "lets do a quiz" is not a substring
            # of "let's do a quiz". The conversation model answered it over the
            # top ("Sure thing, Tadhagath, I'd love to!"), thirty seconds of
            # chatter before question one, in front of the exact bug this
            # swallow was written to prevent.
            from demokit.runner import contains_phrase

            trig_words = _word_stream(text)
            if store.get("step", 0) == 0 and any(
                    contains_phrase(trig_words, t) for t in self.triggers):
                return True
            # Otherwise: between questions, or finished. Let it fall through so
            # a real question about the robot still gets a real answer.
            return False

        index = store.get("order", [])[store.get("step", 0)]
        _question, accepted, fact = _QUESTIONS[index]
        store["waited"] = 0
        if not _is_right(text, accepted):
            if _all_filler(text):
                # "Okay, thank you" mid-question is somebody being polite to
                # the robot, not an attempt at the answer -- and it burned a
                # try live, which gave the answer away one guess early.
                # Swallowed at no cost: not right, not wrong, not an answer.
                return True
            if self._misheard(ctx, store):
                return True
        if _is_right(text, accepted):
            store["score"] = store.get("score", 0) + 1
            self._advance(ctx)
            # Before the words, not after: the sound is the answer landing, and
            # a group cheers over it rather than waiting politely for the line.
            ctx.audio.play_sound("correct", ctx.motion)
            ctx.motion.express_move("happy")
            ctx.say(f"Yes! {fact}", "happy")
            return True

        tries = store.get("tries", 0) + 1
        store["tries"] = tries
        if tries >= _MAX_TRIES:
            self._advance(ctx)
            ctx.audio.play_sound("wrong", ctx.motion)
            ctx.say(f"Close. {fact}", "neutral")
            return True
        ctx.say("Not quite -- anyone else?", "curious")
        self._hold(ctx, True)
        return True

    def on_exit(self, ctx: DemoContext) -> None:
        self._hold(ctx, False)
        ctx.status(f"Quiz paused: {ctx.store.get('score', 0)} right.")

    # --- the round -------------------------------------------------------

    def _ask(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        if store.get("awaiting"):
            store["waited"] = store.get("waited", 0) + 1
            if store["waited"] < _MAX_WAITS:
                return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)
            index = store["order"][store["step"]]
            self._advance(ctx)
            ctx.say(f"I'll tell you. {_QUESTIONS[index][2]}", "neutral")
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        step = store.get("step", 0)
        if step >= min(_ROUND, len(store.get("order", ()))):
            store["stage"] = _SCORE
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        index = store["order"][step]
        store["awaiting"] = True
        store["waited"] = 0
        store["tries"] = 0
        ctx.say(f"Number {step + 1}. {_QUESTIONS[index][0]}", "curious")
        # Held only while a question is out, so the room can answer without
        # anybody saying "hey Reachy" first -- the same rule the enrolment
        # exchange follows, and for the same reason.
        self._hold(ctx, True)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def _advance(self, ctx: DemoContext) -> None:
        store = ctx.store
        store["awaiting"] = False
        store["tries"] = 0
        store["reasked"] = False
        store["waited"] = 0
        store["step"] = store.get("step", 0) + 1
        store["touched"] = time.monotonic()
        self._hold(ctx, False)
        if store["step"] >= min(_ROUND, len(store.get("order", ()))):
            store["stage"] = _SCORE

    def _final_score(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        store["stage"] = _DONE
        score = store.get("score", 0)
        asked = min(_ROUND, len(store.get("order", ())))
        if score == asked:
            line, tag = f"{score} out of {asked}. All of them -- nobody does that.", "surprised"
        elif score >= asked - 1:
            line, tag = f"{score} out of {asked}. That is a good group.", "happy"
        elif score == 0:
            line, tag = "None right, but you know them all now. That is the same thing.", "happy"
        else:
            line, tag = f"{score} out of {asked}. Not bad at all.", "happy"
        # A clean sweep gets the fanfare; anything else gets applause, because
        # a group that got two out of four still played.
        ctx.audio.play_sound("fanfare" if score == asked else "applause", ctx.motion)
        ctx.motion.express_move("happy")
        ctx.say(line, tag)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    # --- session state ---------------------------------------------------

    def _misheard(self, ctx: DemoContext, store: dict) -> bool:
        """One free retry when the transcript itself was garbage.

        The runner's confidence gate deliberately exempts held-mic exchanges,
        so a noisy answer reaches this demo unfiltered -- and a mangled
        transcript is not a wrong answer, it is no answer. One retry only:
        persistent noise still has to resolve the question, or a loud room
        could hold one question open forever.
        """
        from body.audio_io import _MIN_MEAN_TOKEN_LOGPROB

        score = getattr(ctx.audio, "last_confidence", None)
        if score is None or score >= _MIN_MEAN_TOKEN_LOGPROB or store.get("reasked"):
            return False
        store["reasked"] = True
        ctx.say("Say that again?", "curious")
        self._hold(ctx, True)
        return True

    def _hold(self, ctx: DemoContext, held: bool) -> None:
        if bool(ctx.store.get("holding")) == bool(held):
            return
        ctx.store["holding"] = bool(held)
        ctx.state.hold_open_mic(bool(held))

    def _reset(self, ctx: DemoContext) -> None:
        holding = bool(ctx.store.get("holding"))
        # Shuffled so the second group of the day does not get the first
        # group's quiz in the first group's order -- staff run this repeatedly.
        order = list(range(len(_QUESTIONS)))
        random.shuffle(order)
        ctx.store.clear()
        ctx.store.update(
            stage=_ASK, step=0, score=0, tries=0, awaiting=False, waited=0,
            order=order, holding=holding, touched=time.monotonic(),
        )
