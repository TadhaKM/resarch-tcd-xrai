"""The Dean's Office dialogue: a two-hander performed live with the Dean.

This is a STAGE SCRIPT, not a monologue. The Dean speaks their own lines to
the room; Reachy holds the microphone open, stays silent through everything
that is not a cue, and delivers the matching answer block when one of the
Dean's questions lands. The earlier version of this mode spoke a six-line
pitch on its own; it was rewritten for a live welcome event where the Dean
interviews the robot in front of new MSc students.

WHY IT IS BUILT THE WAY IT IS
- The mic is HELD (state.hold_open_mic) for the whole mode, and a long answer
  window is re-armed every slice, so the Dean never says a wake phrase. The
  runner's ambient floor is bypassed by the hold; that is safe here because
  everything unmatched is swallowed silently below.
- Every utterance that is not a cue returns True and does NOTHING. On stage,
  the robot chiming into the Dean's own lines -- or handing them to the
  conversation model for a freelance answer -- is the failure this mode must
  not have. While this mode is selected, Reachy performs the script and only
  the script.
- quiet_when_unsure: a garbled pickup (audience noise, applause) is dropped
  without "Sorry -- say it again?". The Dean repeats a missed cue naturally;
  the robot apologising to a room mid-event is not recoverable.
- Blocks are spoken with interruptible=False: nobody may talk the robot out
  of its own performance, and the per-line barge-in scan would only add dead
  air between dramatic beats.
- Cues match ANYWHERE in the running order, so a Dean who skips or reorders
  questions is followed, not corrected. Spoken fallbacks for the operator:
  "next" (first unspoken block), "say it again" (repeat the last block),
  "start again" (rewind everything, silently).
"""

from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S

#: How long the standing answer window is re-armed for. Effectively "for as
#: long as this mode is selected": it is refreshed on every idle slice and
#: after every utterance, and cleared on exit so it cannot outlive the mode.
_LISTEN_HOLD_S = 600.0

#: The performance, one entry per Dean question. `cues` are matched through
#: the runner's word stream (so punctuation and apostrophes do not matter);
#: `lines` are spoken in order, each with its own expression. The text is the
#: Dean's approved script verbatim -- change it only alongside the Dean's own
#: cue cards, or the robot answers a question that is no longer being asked.
_BLOCKS = (
    {
        "cues": ("introduce yourself", "would you like to introduce"),
        "lines": (
            ("Of course!", "happy"),
            ("Hello new friends, and welcome to Trinity Business School.", "happy"),
            ("I'm Reachy, the small robot who lives at the Trinity AI XR Hub.", "happy"),
        ),
    },
    {
        "cues": ("what exactly is", "what is the ai xr hub", "what is the xr hub",
                 "what is the hub"),
        "lines": (
            ("Well, you might expect a robot to talk about technology, but "
             "that's not really what we're about.", "curious"),
            ("Here, technology isn't the point. People are.", "neutral"),
            ("The Trinity AI XR Hub is a space here in the Business School "
             "where we explore how people and technology can work better "
             "together.", "neutral"),
            ("You'll get to experience artificial intelligence, virtual "
             "reality and, of course... embodied AI like me.", "happy"),
            ("But ultimately, the Hub is about you.", "happy"),
        ),
    },
    {
        "cues": ("students so important", "why are the students",
                 "students important"),
        "lines": (
            ("Because as AI gets better, human skills matter more, not "
             "less.", "neutral"),
            ("Skills like judgement, communication, presence and persuasion "
             "are increasingly important in an AI-enabled world.", "neutral"),
            ("The Hub gives you a space to develop those skills while also "
             "exploring how emerging technologies are changing the way we "
             "learn, work and collaborate.", "happy"),
        ),
    },
    {
        "cues": ("actually do", "do in the hub", "what will our msc"),
        "lines": (
            ("You'll get hands-on experience with immersive technology.", "happy"),
            ("You'll put on a VR headset and practise real-world situations, "
             "from presentations and interviews to challenging "
             "conversations.", "neutral"),
            ("And the best part?", "curious"),
            ("You can practise, get feedback, reflect, and try again.", "happy"),
            ("It's a safe space to experiment, make mistakes and "
             "improve.", "neutral"),
            ("But the Hub isn't only about developing your human "
             "skills.", "neutral"),
            ("It's also a place to experience emerging technologies and "
             "explore what happens when humans and technology work "
             "together.", "curious"),
            ("I suppose that's where I come in.", "happy"),
        ),
    },
    {
        "cues": ("final advice", "any advice", "advice for our new students"),
        "lines": (
            ("Yes.", "neutral"),
            ("Be curious.", "happy"),
            ("Try something unfamiliar.", "curious"),
            ("Don't be afraid to make mistakes.", "neutral"),
            ("And remember: the future isn't just about what AI can "
             "do.", "neutral"),
            ("It's about what you and AI can do together.", "happy"),
            ("So come and visit us, try the technology, and don't forget to "
             "say hello when you see me.", "happy"),
            ("Welcome to Trinity Business School, and welcome to the Trinity "
             "AI XR Hub.", "happy"),
            ("Where Immersive Intelligence collaborates with you and brings "
             "Positive Impact.", "happy"),
        ),
    },
)

#: Operator fallbacks, spoken from beside the stage. "next" is the one that
#: matters: if a cue is missed (noise, a rephrased question), it advances to
#: the first unspoken block without the Dean having to repeat anything.
_NEXT = ("next", "continue", "go on", "carry on")
_AGAIN = ("say it again", "one more time", "repeat that")
_RESTART = ("start again", "from the top", "restart the script")


class DeansOffice(Demo):
    label = "Dean's Office"
    help = ("Performs the Dean's welcome dialogue on cue: Reachy waits "
            "silently and answers each of the Dean's questions from the "
            "script. Say 'next' to advance if a cue is missed.")
    order = 21
    triggers = (
        "deans office",
        "the deans office",
        "do the deans office",
        "deans office introduction",
        "introduce yourself to the dean",
    )
    #: The performance sees everything first: another demo's trigger word
    #: inside one of the Dean's lines must not switch demos mid-event.
    claims_utterances = True
    #: And a garbled pickup is dropped without apologising to the room.
    quiet_when_unsure = True

    def on_enter(self, ctx: DemoContext) -> None:
        ctx.store.clear()
        ctx.store["spoken"] = set()
        ctx.store["last"] = None
        # The Dean speaks first. The robot's whole job until a cue lands is
        # to be a good listener on stage: mic held, mouth shut.
        ctx.state.hold_open_mic(True)
        ctx.state.expect_answer(_LISTEN_HOLD_S)
        ctx.status("Dean's Office: on cue. Waiting for the Dean.")

    def on_exit(self, ctx: DemoContext) -> None:
        # Both halves of the hold, or the next mode inherits a microphone
        # wedged open for ten minutes.
        ctx.state.hold_open_mic(False)
        ctx.state.expect_answer(0.0)

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        # Nothing is ever said from idle -- the Dean drives. Re-arm the
        # standing window so the wake-free listening branch stays open for
        # however long the Dean's own lines run between cues.
        ctx.state.expect_answer(_LISTEN_HOLD_S)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        from demokit.runner import _word_stream, contains_phrase

        words = _word_stream(text)
        spoken: set = ctx.store.setdefault("spoken", set())
        try:
            if any(contains_phrase(words, p) for p in _RESTART):
                spoken.clear()
                ctx.store["last"] = None
                ctx.status("Dean's Office: rewound to the top.")
                return True
            if any(contains_phrase(words, p) for p in _AGAIN):
                last = ctx.store.get("last")
                if last is not None:
                    self._perform(ctx, last)
                return True
            if any(contains_phrase(words, p) for p in _NEXT):
                for index in range(len(_BLOCKS)):
                    if index not in spoken:
                        self._perform(ctx, index)
                        break
                return True
            for index, block in enumerate(_BLOCKS):
                if any(contains_phrase(words, cue) for cue in block["cues"]):
                    # A re-asked question is re-answered -- the Dean repeating
                    # a cue means the room did not hear the answer.
                    self._perform(ctx, index)
                    return True
            # Everything else is the Dean's half of the dialogue, or the
            # room. Swallowed in silence: while this mode is on, Reachy
            # performs the script and nothing but the script.
            return True
        finally:
            ctx.state.expect_answer(_LISTEN_HOLD_S)

    def _perform(self, ctx: DemoContext, index: int) -> None:
        """Deliver one block, uninterruptible, and remember it was given."""
        for line, emotion in _BLOCKS[index]["lines"]:
            ctx.say(line, emotion, interruptible=False)
        ctx.store.setdefault("spoken", set()).add(index)
        ctx.store["last"] = index
        ctx.status(f"Dean's Office: answered cue {index + 1} of {len(_BLOCKS)}.")
