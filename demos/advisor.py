"""Helps somebody work out which masters they should be looking at.

The question a prospective student actually arrives with. Thirteen taught
programmes is too many to read out and too many to choose between from a
website, and the person deciding is standing in a building on an open day with
a member of staff twenty feet away. This demo does what that member of staff
would do: asks three short questions, then names two programmes and says why
each, and hands them back to a human for anything that changes yearly.

ONE OR TWO, AND SHORT
Two invites a comparison, which is the conversation a real adviser has and is
usually the more honest answer -- these programmes often differ by emphasis
rather than by kind. But one is allowed when one plainly fits, because padding
to two in order to look even-handed sends somebody to a programme nobody
thought suited them.

Everything here is spoken standing up, in a queue. Three short sentences is
the whole budget.

WHAT IT WILL NOT DO
Fees, entry requirements, deadlines, whether THIS person would get in. All of
that is either time-sensitive or a judgement no robot should voice; every path
out of this demo ends by pointing at a human. brain/courses.REFUSALS carries
the wording, so it cannot drift from what the conversation demo would say to
the same question.

Structured like demos/brainstorm.py -- one step per idle slice, answers arriving
through on_utterance, the microphone held only while a question of ours is
outstanding -- because that shape is what keeps a talking demo switchable. See
that file's docstring for the reasoning; it is not repeated here.
"""

import time

from brain import courses
from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S

#: Stages, in ctx.store["stage"].
_ASK = "ask"
_SUGGEST = "suggest"
_CLOSING = "closing"
_DONE = "done"

#: The three questions, in order. Written to be answered in a sentence by
#: somebody who has not thought about it yet: "what did you study" is
#: answerable by anyone, where "what is your academic background" gets a pause
#: and then "um".
#:
#: Three, not five. This is a queue at an open day, and the fourth question is
#: where somebody starts looking at the person they came with.
_QUESTIONS = (
    ("background", "What did you study for your degree?"),
    ("draw", "And what part of it did you actually enjoy?"),
    ("technical", "Last one. Do you want to be hands-on with data and tools, or leading the people who are?"),
)

#: Long enough that a group finishes discussing, short enough that a silence
#: is not the demo hanging.
_MAX_WAITS = 4
_BETWEEN_LINES_S = 1.0
_SESSION_STALE_S = 300.0


def _brief(answers: dict) -> str:
    """The standing instructions for the recommendation.

    The whole catalogue goes in, not a retrieved subset: this is the one turn
    where the model genuinely needs to compare across all thirteen, and it is
    a single turn per visitor rather than something on every reply -- which is
    exactly the trade brain/courses.py's docstring describes, taken the other
    way for the one case that warrants it.
    """
    said = "\n".join(f"- {key}: {value}" for key, value in answers.items() if value)
    catalogue = "\n\n".join(p.block() for p in courses.PROGRAMMES)
    return (
        "You are helping somebody at an open day work out which taught masters "
        "at Trinity Business School to look at. This is the whole list:\n\n"
        f"{catalogue}\n\n"
        f"What they told you:\n{said}\n\n"
        "Name ONE or TWO of them -- one if a single programme plainly fits, two "
        "if the choice is genuinely open -- and say why each, in one short "
        "sentence each, using what they told you rather than generalities. "
        "Then stop.\n"
        "Rules that matter more than being helpful:\n"
        "- If their degree is not in business, say plainly which of them are "
        "open to them -- the MSc in Management is ONLY for non-business "
        "graduates, and Accounting and Analytics is built for people with no "
        "accounting behind them. Getting this wrong sends somebody to a "
        "programme they cannot take.\n"
        "- Never say whether they would be accepted, and never quote a fee, a "
        "deadline or an entry requirement.\n"
        "- THREE short sentences at most in total. This is spoken aloud to somebody "
        "standing up: three long sentences is over half a minute of talking at "
        "them, and they came to have a conversation."
    )


class Advisor(Demo):
    label = "Which masters fits me?"
    help = "Asks three questions, then suggests one or two programmes and why."
    order = 55
    persona = "professional"
    triggers = (
        "which masters", "which course should i", "what should i study",
        "help me choose a course", "which programme is right",
        "what masters should i do", "help me pick a masters",
    )
    #: An answer about somebody's own degree must not be mistaken for another
    #: demo's trigger word -- "I studied business analytics" would otherwise
    #: switch demo mid-interview.
    claims_utterances = True

    # --- hooks -----------------------------------------------------------

    def on_enter(self, ctx: DemoContext) -> None:
        store = ctx.store
        fresh = time.monotonic() - store.get("touched", 0.0) < _SESSION_STALE_S
        if not (fresh and store.get("stage") not in (None, _DONE)):
            self._reset(ctx)
        store["touched"] = time.monotonic()
        ctx.say("Three quick questions and I'll point you at the right masters.", "happy")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        stage = ctx.store.get("stage")
        if stage == _ASK:
            return self._ask(ctx)
        if stage == _SUGGEST:
            return self._suggest(ctx)
        if stage == _CLOSING:
            return self._close(ctx)
        if stage is None:
            self._reset(ctx)
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        store = ctx.store
        if not text.strip():
            return False
        if store.get("awaiting"):
            key = _QUESTIONS[store.get("step", 0)][0]
            store.setdefault("answers", {})[key] = text.strip()
            store["awaiting"] = False
            store["waited"] = 0
            store["step"] = store.get("step", 0) + 1
            store["touched"] = time.monotonic()
            self._hold(ctx, False)
            ctx.motion.acknowledge()
            if store["step"] >= len(_QUESTIONS):
                store["stage"] = _SUGGEST
            return True
        if store.get("stage") == _ASK and not store.get("answers"):
            # The trigger phrase that selected this demo, handed straight back
            # by the runner. Swallowed so the model does not answer it over the
            # top of the first question.
            return any(t in text.lower() for t in self.triggers)
        # Finished, or mid-suggestion: let it fall through, so a follow-up
        # question about one of the two programmes reaches the conversation
        # demo with courses.brief behind it.
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        self._hold(ctx, False)
        ctx.status(f"Advisor paused, {len(ctx.store.get('answers') or {})} of 3 answered.")

    # --- the interview ---------------------------------------------------

    def _ask(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        if store.get("awaiting"):
            store["waited"] = store.get("waited", 0) + 1
            if store["waited"] < _MAX_WAITS:
                return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)
            # Nobody answered. Work with whatever they did say rather than
            # asking again -- a queue at an open day does not wait.
            store["awaiting"] = False
            self._hold(ctx, False)
            store["stage"] = _SUGGEST if store.get("answers") else _DONE
            if not store.get("answers"):
                ctx.say("No bother. Ask me any time.", "neutral")
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        _key, question = _QUESTIONS[store.get("step", 0)]
        store["awaiting"] = True
        store["waited"] = 0
        ctx.say(question, "curious")
        self._hold(ctx, True)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def _suggest(self, ctx: DemoContext) -> IdleResult:
        """The recommendation itself. One hook, then the closing lines queue.

        Measured live at 62 seconds when this said everything in one hook --
        the model call, three sentences of answer, the fees handoff and the
        screen line, back to back. The runner warns past six seconds for a
        reason: a school group arriving mid-session cannot be switched to
        while a hook is running, and a minute is a long time to stand there.
        The reply itself cannot be split (it is one streamed answer, and
        ctx.reply checks for a switch between its sentences, so it does yield),
        but everything after it can, and now does.
        """
        store = ctx.store
        spoken = ctx.reply(
            "Which two should I look at?",
            person_id=store.get("person_id", 0),
            system=_brief(store.get("answers") or {}),
            cache=False,
        )
        if not spoken:
            store["stage"] = _DONE
            ctx.status("The recommendation came back empty.")
            ctx.say("I could not think of two just now. The team here can help.", "sad")
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

        # The handoff is said by the robot rather than left to the prompt, so
        # it comes out the same way every time and cannot be reworded into a
        # promise. Two facts a robot must never be the source of: what it costs
        # and whether they would get in.
        store["queue"] = [
            "Ask the team here about fees and entry -- those change every year, "
            "so I stay out of it."
        ]
        store["stage"] = _CLOSING
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    def _close(self, ctx: DemoContext) -> IdleResult:
        """One closing line per slice, so the robot stays switchable."""
        queue = ctx.store.get("queue") or []
        if not queue:
            ctx.store["stage"] = _DONE
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)
        ctx.say(queue.pop(0), "neutral")
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    # --- session state ---------------------------------------------------

    def _hold(self, ctx: DemoContext, held: bool) -> None:
        if bool(ctx.store.get("holding")) == bool(held):
            return
        ctx.store["holding"] = bool(held)
        ctx.state.hold_open_mic(bool(held))

    def _reset(self, ctx: DemoContext) -> None:
        holding = bool(ctx.store.get("holding"))
        ctx.store.clear()
        ctx.store.update(
            stage=_ASK, step=0, answers={}, awaiting=False, waited=0,
            holding=holding, touched=time.monotonic(), person_id=ctx.person_id(),
        )
