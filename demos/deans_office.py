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
from demos._set_pieces import BLOCKS as _BLOCKS
from demos._set_pieces import PACE as _PACE

#: How long the standing answer window is re-armed for. Effectively "for as
#: long as this mode is selected": it is refreshed on every idle slice and
#: after every utterance, and cleared on exit so it cannot outlive the mode.
_LISTEN_HOLD_S = 600.0

# The performance material itself lives in demos/_set_pieces.py -- ONE copy of
# the Dean's approved dialogue, shared with the conversation fast-path, so the
# robot cannot drift into answering the same question two different ways. Each
# block's stage_cues are the Dean's own lines; change them only alongside the
# Dean's cue cards, or the robot answers a question no longer being asked.

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
        from demos._set_pieces import normalise

        # The same mishearing map the conversation fast-path uses: on stage,
        # "the AI XR Hub" arriving as "the AIXR home" must still cue.
        words = normalise(_word_stream(text))
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
                if any(contains_phrase(words, cue) for cue in block.stage_cues):
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
        ctx.say_script(_BLOCKS[index].lines, pace=_PACE, interruptible=False)
        ctx.store.setdefault("spoken", set()).add(index)
        ctx.store["last"] = index
        ctx.status(f"Dean's Office: answered cue {index + 1} of {len(_BLOCKS)}.")
