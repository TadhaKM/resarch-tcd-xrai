"""A facilitated idea session: three questions, then three directions back.

Structured as an interview rather than a chat because of who it is for. A
business-school visitor arrives with a half-formed idea and no shared frame for
interrogating it, and a model asked to "brainstorm this" without one produces
the same three startup platitudes for everybody. What, who, and what's
different -- or what's in the way -- is the smallest set of questions that
makes the generated directions specific to the person standing there.

Three decisions about how this talks to the framework are worth knowing:

- Questions are spoken with ctx.say and answered through the wake-word path,
  never with ctx.ask. ctx.ask is say-then-listen, and AudioIO.listen() has no
  timeout at all -- it returns only once the recogniser fires an endpoint
  (5s of trailing silence, then a Whisper re-decode), so a hook that calls it
  holds the robot for as long as the room stays quiet and the operator cannot
  switch away from it. The cost is that a visitor has to say "hey Reachy"
  before each answer unless open mic is on. That is explained by whoever is
  running the demonstration, not by the robot: saying it here made the first
  thing a visitor heard about their own idea a lesson in operating a robot.
  A slightly less fluid interview is worth one that is always switchable,
  which is the trade the framework exists to enforce.
- No hook returns listen_for=0.0 while there is script left to speak. Zero
  means DemoRunner.cycle returns without opening the microphone, so a run of
  such slices is a stretch of speech that cannot be interrupted and during
  which the wake word is simply unheard. In a facilitation demo, where the
  whole point is that the visitor takes part, that is the worst thing this
  file could do: the bridge, the three directions and the closing question
  each leave a one-second window instead. The wake-word stream survives across
  windows, so a phrase spoken over a boundary is still matched and the lines
  still sound back-to-back.
- claims_utterances, because a visitor answering "dance studios in Dublin"
  must not make the robot dance in the middle of their own brainstorm.

Everything advances one step per on_idle slice, including one generated
direction per slice: three model calls in a single hook would hold the robot
for the better part of a minute with the microphone shut and the dashboard
unable to switch away.
"""

import time

from brain import hub
from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S


#: Session stages, held in ctx.store["stage"].
_ASK = "ask"
_BRIDGE = "bridge"
_GENERATE = "generate"
_SUMMARY = "summary"
_DONE = "done"

#: (store key, question, emotion). Written for the ear -- one idea each, and
#: each one narrows the answer before it is given. "One kind of person, not
#: everybody" is the difference between an answer worth building on and
#: "consumers", and it costs four words to ask for.
_QUESTIONS = (
    ("problem", "First one. What's the idea, or the problem you want to work on?", "curious"),
    ("audience", "Good. Who is it for? One kind of person, not everybody.", "curious"),
    (
        "edge",
        "Last one. What makes it different from what's already out there, or what's the "
        "constraint you're stuck behind?",
        "curious",
    ),
)

#: One direction per ordinal, and the ordinal is spoken, so the count and the
#: words that announce it cannot drift apart. Three: two reads as the model
#: hedging between an option and its opposite, four outlasts a standing group's
#: attention.
_ORDINALS = ("first", "second", "third")

#: Multi-word on purpose. A bare "summary" or "summarise" turns up inside real
#: answers -- "a summarising tool for solicitors" is exactly the kind of thing
#: this demo gets asked about -- and would end the session the visitor is still
#: in the middle of.
_SUMMARISE_PHRASES = ("summarise that", "summarize that", "summarise it", "summarize it", "sum that up")
_RESTART_PHRASES = ("start over", "start again", "start from scratch")

#: The runner keeps one store per demo id for the life of the process and never
#: clears it (see DemoRunner._stores), so a session abandoned when the operator
#: switched away would otherwise resume mid-interview for the next group at the
#: next open-day slot. Anything older than this starts fresh instead.
_SESSION_STALE_S = 300.0

#: Longest answer read back verbatim in the recap. Past this the recap stops
#: being a recap and becomes the visitor's own paragraph played at them.
_RECAP_WORD_CAP = 25

#: The gap left after each spoken script line -- the bridge, each direction,
#: each recap line. Short enough that the lines still read as one continuous
#: passage, and long enough that the core actually opens the microphone, which
#: is the only thing standing between this demo and a minute of speech a
#: visitor cannot interrupt. Never 0.0 here: see the module docstring.
_BETWEEN_LINES_S = 1.0

def _for_speech(answer: str) -> str:
    """Tidy one transcript into something worth reading back aloud.

    The recap quotes the visitor's own words, and those words arrive from
    whichever recogniser won the turn: Whisper's transcript is cased and
    punctuated, the streaming model's fallback is bare capitals with no
    punctuation (its output is logged that way throughout body/audio_io.py).
    Read back unchanged, the second one lands as a shouted sentence fragment
    dropped into the middle of the robot's own.
    """
    text = answer.strip().rstrip(".!?").strip()
    if text.isupper():
        text = text.lower()
    words = text.split()
    if len(words) > _RECAP_WORD_CAP:
        text = " ".join(words[:_RECAP_WORD_CAP]) + ", and the rest of it"
    return text


def _direction_brief(answers: dict[str, str], ordinal: str) -> str:
    """The standing instructions for one direction, layered onto the system prompt.

    This goes to ctx.reply(system=...), not into the message. Everything here
    is instruction and context rather than something a person said, and
    anything in the message is recorded by brain/memory.py as the visitor's own
    turn: sent that way it would be replayed on every later request for the
    rest of the session, so a visitor asking an unrelated question after the
    session would get an answer generated on top of a transcript in which they
    apparently recited a prompt three times.

    The whole interview is restated in every brief rather than relied on from
    history: brain/memory.py trims to the last dozen turns and shares one bucket
    across every unrecognised visitor, so on a busy afternoon the answers can be
    gone by the third direction. What history is genuinely load-bearing for is
    the no-repeat instruction -- the directions already given are the model's
    own previous turns, which is what gives it something to refer to.

    The Hub's mission is a steer, not something to recite: without it the model
    proposes an app for everything, and told to describe the Hub it will
    paraphrase the mission into a claim nobody checked.
    """
    return (
        "You are facilitating a business brainstorm out loud, in a room with the person "
        "whose idea this is.\n"
        f"The idea or problem: {answers['problem']}\n"
        f"Who it is for: {answers['audience']}\n"
        f"What makes it different, or the constraint: {answers['edge']}\n"
        "The session runs at the AI XR Hub, so favour directions that fit its mission: "
        f"{hub.MISSION_SHORT} Do not describe the Hub in your answer.\n"
        f"Give the {ordinal} of three directions they could take this, and start your "
        f"answer with the word {ordinal}. Name the direction in a few words, then one "
        "sentence on why it could work. Two short sentences at most. Make it genuinely "
        "different from any direction you have already given in this conversation, and do "
        "not ask a question."
    )


def _recap_lines(answers: dict[str, str], directions: list[str]) -> list[tuple[str, str]]:
    """The session played back as (line, emotion) pairs, one per idle slice.

    Only ever reached now by someone explicitly asking for it. It used to run
    automatically after the third direction, and hearing it live is what settled
    that: a visitor who answers "everybody" and "it's cool" -- which is what
    people actually say to a robot in a corridor -- gets "It's for everybody"
    and "What makes it different, or what's in the way: It's cool" read back at
    them a minute later. Their own words in a template do not become a summary,
    and the session had already made its point with the three directions.
    """
    lines = []
    if "problem" in answers:
        lines.append((f"The problem: {_for_speech(answers['problem'])}.", "neutral"))
    if "audience" in answers:
        lines.append((f"It's for {_for_speech(answers['audience'])}.", "neutral"))
    if "edge" in answers:
        lines.append((f"What makes it different, or what's in the way: {_for_speech(answers['edge'])}.", "neutral"))
    return lines


class Brainstorm(Demo):
    label = "Business Brainstorming"
    help = "Interviews a visitor about an idea, then offers three directions back."
    order = 60
    #: Facilitating somebody's business idea. Measured and specific is what makes the directions usable.
    persona = "professional"
    #: Request-shaped, not topic-shaped. A bare "brainstorm" or "business idea"
    #: is ordinary vocabulary at a business school -- "we ran a brainstorm this
    #: morning", "students arrive with a business idea" -- and because this demo
    #: claims utterances, firing on those sentences captures the group into a
    #: three-question interview nobody asked for.
    triggers = (
        "lets brainstorm",
        "help me brainstorm",
        "brainstorm with me",
        "i have a business idea",
        "got a business idea",
    )
    #: A visitor's answer must reach this demo before any other demo's trigger
    #: word is considered. See the module docstring.
    claims_utterances = True

    # --- hooks -----------------------------------------------------------

    def on_enter(self, ctx: DemoContext) -> None:
        store = ctx.store
        fresh = time.monotonic() - store.get("touched", 0.0) < _SESSION_STALE_S
        resuming = fresh and store.get("stage") not in (None, _DONE)
        if not resuming:
            self._reset(ctx)
        else:
            # Whatever question was in the air when the operator switched away
            # gets asked again rather than waited on silently -- the visitor in
            # front of the robot now may not be the one who heard it.
            store["awaiting"] = None
            store["waited"] = 0
        store["touched"] = time.monotonic()
        # Says what is about to happen, not how to operate the robot. Explaining
        # the wake word is the operator's job and belongs in their guide -- said
        # here it was the first thing a visitor heard about their own idea.
        ctx.say(
            "Picking up where we left off."
            if resuming
            else "Let's brainstorm. Three questions from me, then some directions back.",
            "happy",
        )

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        stage = ctx.store.get("stage")
        if stage == _ASK:
            return self._ask_next(ctx)
        if stage == _BRIDGE:
            ctx.store["stage"] = _GENERATE
            ctx.say("Right. Three directions on that.", "thinking")
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        if stage == _GENERATE:
            return self._offer_direction(ctx)
        if stage == _SUMMARY:
            return self._recap_one_line(ctx)
        if stage is None:
            # on_enter is guarded, so it can have failed and left nothing set
            # up. Recovering here costs a line; assuming our own hooks ran costs
            # the demo every time someone changes the runner.
            self._reset(ctx)
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        # Finished. Anything said now is "start over", a question for the model,
        # or another demo's trigger, and all three want the microphone open.
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        store = ctx.store
        lowered = text.lower()

        if any(phrase in lowered for phrase in _RESTART_PHRASES):
            self._reset(ctx)
            ctx.say("Fresh start. Forget everything I just heard.", "happy")
            return True

        if any(phrase in lowered for phrase in _SUMMARISE_PHRASES):
            return self._recap_now(ctx)

        awaiting = store.get("awaiting")
        if awaiting:
            self._record(ctx, awaiting, text)
            return True

        if store.get("stage") in (_ASK, _BRIDGE) and not store.get("answers"):
            # This is the trigger phrase that selected the demo ("let's
            # brainstorm"), handed to us by the runner right after on_enter
            # already answered it. Swallowed rather than passed on, so the model
            # doesn't reply to it over the top of the first question.
            return any(trigger in lowered for trigger in self.triggers)

        # Mid-generation, or done: let it fall through to conversation, so
        # someone who asks about the Hub after the session still gets an answer
        # and other demos' triggers still work.
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        # Nothing to stop -- every hook is one step long. The note is for the
        # operator, who is the one who has to decide whether the next visitor
        # gets this session back (see _SESSION_STALE_S).
        answered = len(ctx.store.get("answers") or {})
        ctx.status(f"Brainstorm paused, {answered} of {len(_QUESTIONS)} answers in.")

    # --- the interview ---------------------------------------------------

    def _ask_next(self, ctx: DemoContext) -> IdleResult:
        """Speak the outstanding question, or wait -- listening -- for its answer.

        The question is spoken and the hook returns; the answer comes back
        through the wake-word path into on_utterance, which is where answers
        were always recorded anyway. The alternative, ctx.ask, would hold this
        hook for the whole of the visitor's thinking time with no upper bound
        (see the module docstring), and a hook that does not return is a robot
        the operator cannot switch away from.
        """
        store = ctx.store
        key, question, emotion = _QUESTIONS[store["step"]]

        if store.get("awaiting") == key:
            # The question is already out. Wait quietly rather than repeating
            # it: a robot that re-asks every three seconds in a room full of
            # people is worse than one that lets a pause be a pause.
            #
            # It used to nudge once with instructions about the wake word. That
            # is the operator's job to explain, not something to say to a
            # visitor mid-thought -- and a pause here usually is thinking, which
            # this then talked over.
            store["waited"] = store.get("waited", 0) + 1
            ctx.status(f"Waiting for an answer: {key}")
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

        # Set before speaking, not after: say() raises DemoStopped if the
        # operator switches away mid-question, and it unwinds through here, so
        # the store has to already say which question is outstanding.
        store["awaiting"] = key
        store["waited"] = 0
        ctx.say(question, emotion)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def _record(self, ctx: DemoContext, key: str, answer: str) -> None:
        """Take one answer and move to the next question, or to generating."""
        store = ctx.store
        store["answers"][key] = answer
        store["awaiting"] = None
        store["waited"] = 0
        store["touched"] = time.monotonic()
        store["step"] += 1
        # A nod instead of "got it": spoken acknowledgement between two
        # questions is a third of the session spent saying nothing.
        ctx.motion.acknowledge()
        ctx.status(f"Brainstorm {store['step']} of {len(_QUESTIONS)}: {key} = {answer}")
        if store["step"] >= len(_QUESTIONS):
            store["stage"] = _BRIDGE

    def _offer_direction(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        directions = store["directions"]
        ordinal = _ORDINALS[len(directions)]
        # The message is what the visitor is taken to have said, so it is kept
        # to the request itself and everything else goes in the system layer.
        spoken = ctx.reply(
            f"Give me the {ordinal} direction.",
            person_id=store["person_id"],
            system=_direction_brief(store["answers"], ordinal),
        )
        if not spoken:
            # Counted anyway. Retrying inside a hook only asks a backend that is
            # already failing to fail again, and the runner sets a demo aside
            # after three of those.
            ctx.status("A direction came back empty.")
        directions.append(spoken)
        store["touched"] = time.monotonic()
        if len(directions) >= len(_ORDINALS):
            store["stage"] = _SUMMARY
            # One closing line, not the session read back. It ends on the
            # question rather than a conclusion because the useful output here
            # is what the visitor tests on Monday, and a robot that picks the
            # winner has taken the interesting part away.
            store["queue"] = [
                (
                    "Which of those would you test first, and what's the cheapest way "
                    "to find out?",
                    "curious",
                )
            ]
        # A direction is a couple of sentences and there are three of them back
        # to back. Between each one the visitor gets a window to cut in, which
        # in a session about their own idea is the whole point.
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    # --- the recap -------------------------------------------------------

    def _recap_now(self, ctx: DemoContext) -> bool:
        """Handle "summarise that", wherever the session has got to."""
        store = ctx.store
        answers = store.get("answers") or {}
        if not answers:
            ctx.say("Nothing to sum up yet. Give me the idea first.", "curious")
            return True
        stage = store.get("stage")
        # The lead-in belongs here rather than in _recap_lines: asked for, a
        # summary needs announcing; volunteered, that same line was the robot
        # starting a speech nobody requested.
        store["queue"] = [("So far:", "neutral")] + _recap_lines(
            answers, store.get("directions") or []
        )
        # Mid-session this is a re-orienting move, not an exit: the interview
        # resumes at the stage it was in, and the pending question is asked
        # again because awaiting was cleared.
        store["resume_stage"] = _DONE if stage in (_SUMMARY, _DONE) else stage
        store["awaiting"] = None
        store["waited"] = 0
        store["stage"] = _SUMMARY
        return True

    def _recap_one_line(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        queue = store.get("queue") or []
        if not queue:
            resumed = store.pop("resume_stage", _DONE)
            store["stage"] = resumed
            if resumed == _DONE:
                # A visible full stop, so a group can tell the session ended
                # rather than stalled.
                ctx.motion.express_move("happy")
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        line, emotion = queue.pop(0)
        ctx.say(line, emotion)
        # Up to five lines of the visitor's own words read back at them is
        # exactly when someone says "no, that second one" -- so the microphone
        # opens between every one of them.
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    # --- session state ---------------------------------------------------

    def _reset(self, ctx: DemoContext) -> None:
        """Start a clean session, keeping nothing from the last visitor."""
        ctx.store.clear()
        ctx.store.update(
            stage=_ASK,
            step=0,
            answers={},
            directions=[],
            awaiting=None,
            waited=0,
            touched=time.monotonic(),
            # Fixed for the whole session. person_id() flickers between a
            # recognised id and 0 as a face comes and goes, and the no-repeat
            # rule for the directions only works if all three model calls land
            # in one person's history -- brain/memory.py keys on exactly this.
            person_id=ctx.person_id(),
        )
