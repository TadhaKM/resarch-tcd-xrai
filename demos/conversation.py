"""The default demonstration: ordinary questions, answered with the Hub's facts.

This is what the robot does whenever nothing else is selected, so it is the demo
most visitors actually meet, and the single thing it adds to an ordinary reply is
grounding. Asked "so who runs this place?" with the stock system prompt the model
answers fluently and wrongly, in a room that regularly contains the people it is
being wrong about. brain/hub.py exists to stop exactly that, and ctx.reply's
`system=` is how it reaches the model: the facts are layered onto the system
prompt for that one turn, alongside the standing persona, and are never spoken
aloud -- ctx.reply speaks only what comes back.

The briefing goes on every turn rather than being injected once. It can, because
`system=` does not accumulate: brain/memory.py records only the message, so the
history stays what the visitor actually said, and the next prompt is the briefing
plus real turns rather than the briefing plus a copy of the briefing that a
visitor appears to have recited. The fixed cost buys the one thing a keyword test
cannot -- "who's in charge of all this?" and "what are you lot working on?"
contain no Hub keyword, are exactly what a visitor opens with, and are precisely
where an invented answer gets heard by someone who knows better.

The consequence is that this demo answers everything itself rather than returning
False and letting the runner converse: the runner's conversational path has no
way to attach the briefing, so handing the turn back is handing back an
ungrounded answer. That makes on_utterance the only thing between a visitor and
silence, which is why it handles its own failures rather than letting the
runner's guard see them -- see the comment there.
"""

import logging

from brain import hub
from demokit import Demo, DemoContext, IdleResult
from demokit.base import DemoStopped, MAX_LISTEN_WINDOW_S


logger = logging.getLogger(__name__)

#: What ctx.reply(system=...) layers on for a turn. The facts come from hub.py
#: and are never restated here; the sentence after them is, and both halves of it
#: were earned. Handed a block of prose, a small model's first instinct is to
#: recite it, so the visitor gets the mission statement read at them instead of an
#: answer. And after several hundred words of facts it drifts off the length limit
#: prompts.py sets at the top of the system prompt -- the briefing is appended
#: after that limit, not before it -- so the limit is repeated where it is read
#: last.
_HUB_BRIEFING = (
    f"{hub.GROUNDING}\n\n"
    "You already know all of this: never read it out, never quote it, and never "
    "mention having been given it. Answer what was just said to you in your usual "
    "one or two short sentences."
)


class Conversation(Demo):
    # `id` is the filename, and brain/modes.py boots into the mode named
    # "conversation" -- renaming this file changes which demo the robot starts
    # in, and would need an explicit `id` here to stay correct.
    label = "AI Conversation"
    help = "General questions about AI, XR, the Hub and the robot itself."
    #: First by order, which also makes this registry.default_id(): where the
    #: runner falls back when a selected demo is unavailable or set aside.
    order = 10
    #: No triggers. Everything already lands here when nothing else claims it, so
    #: a trigger phrase could only pull a visitor OUT of the demonstration an
    #: operator deliberately selected.
    triggers = ()
    #: Deliberately not claiming utterances. This demo answers literally
    #: anything, and claiming would put it ahead of every other demo's trigger
    #: words -- after which "dance" would never reach the dance demo again.
    claims_utterances = False

    def on_enter(self, ctx: DemoContext) -> None:
        # One line, and a shorter one when an operator switches back mid-visit:
        # at startup these are the robot's first words and should say what the
        # robot is for, but by the third return in a demonstration the long
        # version has been heard twice already and starts to sound like a loop.
        # The store survives for the life of the process and is deliberately not
        # cleared here: what this tracks is what the room has already heard, not
        # what one visitor has, and on_enter fires on an operator's switch, which
        # is not a visitor boundary in any case.
        first_entry = not ctx.store.get("entered")
        ctx.store["entered"] = True
        if first_entry:
            ctx.say("I'm listening.", "happy")
        else:
            ctx.say("Back to questions.", "curious")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        # There is nothing to advance here, only someone to wait for, so this
        # asks for the longest window the core will honour: every expiry costs a
        # wake-word restart, and the core re-reads the mode between windows, so
        # an operator's switch still lands within one of them.
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        """Answer with the Hub's facts in front of the model, and never hand the turn back."""
        try:
            ctx.reply(text, person_id=ctx.person_id(), system=_HUB_BRIEFING)
        except DemoStopped:
            # The operator switched away mid-answer and ctx.reply unwound between
            # sentences. That is the demo behaving, not failing, so it goes to the
            # runner's guard, which is where the framework decides what it means.
            raise
        except Exception:
            # Claimed anyway, and this is the whole reason for catching. Left to
            # propagate, the runner's guard absorbs it, _offer reports the
            # utterance unhandled, and _take_turn falls through to its own
            # conversational reply -- putting the same question to the model a
            # second time, ungrounded, after this one may already have spoken half
            # an answer. A question answered once badly beats one answered twice.
            # Reported the way the runner reports its own conversational failures,
            # since this has taken that job over: logged, noted for the operator,
            # and not narrated at the visitor.
            logger.exception("Grounded reply failed")
            ctx.state.add("error", "I lost my train of thought there.")
        return True
