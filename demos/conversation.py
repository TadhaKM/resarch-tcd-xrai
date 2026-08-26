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
import time

from brain import hub
from demokit import Demo, DemoContext, IdleResult
from demokit.base import DemoStopped, Interrupted, MAX_LISTEN_WINDOW_S

#: Ways people ask the clock. Multi-word and question-shaped on purpose: a
#: bare "time" is inside "sometimes" and "storytime", and "the date" is inside
#: sentences about dates that are not questions about today.
_TIME_ASKS = (
    "what time is it", "whats the time", "what is the time", "have you got the time",
    "do you know the time", "tell me the time", "what time it is",
)
_DATE_ASKS = (
    "what day is it", "whats the date", "what is the date", "what date is it",
    "whats todays date", "what is todays date", "what day is it today",
)


def _clock_answer(text: str) -> str:
    """A spoken answer for a time/date question, or "" when it is not one.

    Written for the ear: "16:01" synthesises as "sixteen oh one", so the hour
    is spoken the way a person says it, with the part of day doing the am/pm
    work.
    """
    from demokit.runner import _word_stream

    words = _word_stream(text)
    asked_time = any(f" {p} " in words for p in (_word_stream(p).strip() for p in _TIME_ASKS))
    asked_date = any(f" {p} " in words for p in (_word_stream(p).strip() for p in _DATE_ASKS))
    if not asked_time and not asked_date:
        return ""
    now = time.localtime()
    if asked_time:
        hour = now.tm_hour % 12 or 12
        part = "in the morning" if now.tm_hour < 12 else (
            "in the afternoon" if now.tm_hour < 18 else "in the evening")
        minute = now.tm_min
        if minute == 0:
            clock = f"exactly {hour} o'clock {part}"
        elif minute < 10:
            clock = f"{hour} oh {minute} {part}"
        else:
            clock = f"{hour} {minute} {part}"
        return f"It's {clock}."
    return time.strftime("It's %A the %d of %B.").replace(" 0", " ")


logger = logging.getLogger(__name__)

#: What ctx.reply(system=...) layers on for a turn. The facts come from hub.py
#: and are never restated here; the sentence after them is, and both halves of it
#: were earned. Handed a block of prose, a small model's first instinct is to
#: recite it, so the visitor gets the mission statement read at them instead of an
#: answer. And after several hundred words of facts it drifts off the length limit
#: prompts.py sets at the top of the system prompt -- the briefing is appended
#: after that limit, not before it -- so the limit is repeated where it is read
#: last.
#: Deliberately does NOT repeat hub.GROUNDING. The same text is already in
#: the base system prompt (brain/prompts.py includes it, and Anthropic serves
#: it from the prompt cache), so carrying it here again re-sent ~1,200 tokens
#: of byte-for-byte duplicate as UNCACHED input on every single conversation
#: turn -- measured: this tail was 1,350 tokens, and drops to ~150 with the
#: duplicate removed, which is 100-300ms of prefill and most of the per-turn
#: token cost for no behavioural difference at all.
_HUB_BRIEFING = (
    "Answer from what you know about the Hub. Never read your briefing out, "
    "never quote it, and never mention having been given it. Answer what was "
    "just said to you in one or two short sentences, roughly twenty-five "
    "words -- unless they asked for ideas or options, in which case give two "
    "or three concrete ones with reasons."
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
        # The clock, answered from the clock. The model deliberately does not
        # know the time -- a clock in the prompt would churn every cache every
        # minute -- so asked the time it said it had no live information, and
        # the first visitor to hear that concluded the robot "does not work".
        # The laptop knows what time it is; no model, no cache, no delay.
        spoken_clock = _clock_answer(text)
        if spoken_clock:
            ctx.say(spoken_clock, "happy")
            return True
        # The Hub's own questions get the Hub's own answers -- the dialogue
        # approved for the Dean's welcome event, spoken identically whoever
        # asks, and instantly: it is already written, so no model round-trip.
        # Everything else, including richer questions that merely contain one
        # of the cues, still goes to the model below.
        from demos import _set_pieces

        if _set_pieces.perform(ctx, text):
            return True
        # Then whatever the Hub has TAUGHT it since -- the dashboard's
        # teach panel writes these, so the robot gets smarter without
        # anyone editing code. Same instant delivery, same reasons.
        from brain import knowledge

        if knowledge.speak_if_taught(ctx, text):
            return True
        try:
            ctx.reply(text, person_id=ctx.person_id(), system=_HUB_BRIEFING)
        except (DemoStopped, Interrupted):
            # The operator switched away mid-answer, or a visitor talked over it,
            # and ctx.reply unwound between sentences. Both are the demo
            # behaving, not failing, so they go to the runner's guard, which is
            # where the framework decides what they mean.
            #
            # Interrupted was missing from this line and that is the whole
            # reason barge-in never worked. It subclasses Exception, so the
            # catch below swallowed it, logged "Grounded reply failed" and told
            # the operator the robot had lost its train of thought -- in the
            # demo that is selected by default and therefore running almost all
            # the time. A visitor saying the wake word over an answer got an
            # error banner instead of the floor.
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
