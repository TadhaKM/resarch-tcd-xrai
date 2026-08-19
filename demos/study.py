"""Runs an HRI study session: records the exchanges, and can be stopped aloud.

ARMING IS THE CONSENT. Earlier versions read a participant information notice
out loud and waited for a spoken yes. That is gone at the Hub's request: the
people who run this robot are Trinity researchers who take consent the way HRI
studies normally take it -- on paper, before the session -- and the spoken
version cost about fifty seconds of a visitor's attention before the first real
exchange, on top of misreading answers like "I consent to be recorded".

What that costs is that the DATABASE no longer evidences consent, so
brain/study.py records who armed the session against every row instead. That
is the trail an ethics reviewer asks for.

What it does NOT cost is the ability to stop. The refusal vocabulary from the
old consent reader was kept and folded into _WITHDRAW, because with the robot
no longer asking, an objection raised mid-session is the only way somebody in
the room can end this -- and it still deletes rather than mutes.

NOT A SUBSTITUTE FOR ETHICS APPROVAL. brain/study.py's docstring says this at
length and it is repeated here because this is the file somebody opens when
they want to run a study on Thursday. Trinity requires approval before data
collection begins. What this gives you is a technical setup that matches what
an approved protocol asks for; it does not give you the protocol.

Deliberately unavailable unless research mode has been switched on from the
dashboard, so it cannot be reached by somebody browsing the demo list during
an open day.
"""

import time

from brain import study as store
from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S

_RUNNING = "running"
_DONE = "done"

_BETWEEN_LINES_S = 1.0

#: Ways somebody stops the recording, checked on EVERY utterance -- a
#: withdrawal that only works when the robot happens to ask is not a
#: withdrawal. It deletes; it does not mute.
#:
#: The second group are refusals of consent rather than withdrawals of it.
#: They are here because the robot no longer asks for consent out loud, so
#: this is the only place an objection can land: somebody who says "I do not
#: consent to being recorded" mid-session has not been asked a question, they
#: have raised one, and the answer has to be the same either way. Keeping them
#: was the point of not deleting the consent vocabulary along with the script.
_WITHDRAW = (
    "delete my data", "delete that", "withdraw", "take me out of the study",
    "forget what i said", "stop recording", "i changed my mind",
    "do not consent", "dont consent", "don't consent", "i refuse", "opt out",
    "do not record", "dont record", "don't record", "no recording",
    "do not want to be recorded", "dont want to be recorded",
)


class Study(Demo):
    label = "Research session"
    help = "Asks for consent, then records the conversation for HRI research."
    order = 800
    #: Only offered when research mode is switched on. The runner greys out a
    #: demo whose requirement is missing and says why, which is exactly the
    #: behaviour wanted here -- it appears, explains itself, and cannot be run
    #: by accident.
    requires = ("study",)
    claims_utterances = True

    def on_enter(self, ctx: DemoContext) -> None:
        ctx.store.clear()
        if not store.running():
            ctx.store["stage"] = _DONE
            ctx.say("Research mode is not switched on, so there is nothing to record.",
                    "neutral")
            return
        # ARMING IS THE CONSENT. The robot no longer reads a consent notice and
        # waits for a spoken yes; brain/study.start() marks the session
        # consented the moment an operator switches research mode on.
        #
        # The Hub's decision, and a normal one: in most HRI studies consent is
        # taken on paper before the session and the robot never asks. The
        # spoken version cost about fifty seconds of a visitor's attention
        # before the first real exchange, and the operator switching it on is a
        # Trinity researcher who has already done the ethics work.
        #
        # What is lost is that the DATABASE no longer evidences consent -- so
        # brain/study.py records who armed the session instead. That is the
        # trail an ethics reviewer would ask for, and it is why the operator
        # field exists.
        ctx.store.update(stage=_RUNNING)
        ctx.status(f"Study session {store.status()['session']}: recording.")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        store_ = ctx.store
        stage = store_.get("stage")

        if stage == _RUNNING:
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in _WITHDRAW):
            gone = store.withdraw()
            store.consent(False)
            ctx.store["stage"] = _DONE
            ctx.status(f"Participant withdrew; {gone} turn(s) deleted.")
            ctx.say("Done -- I have deleted it. We can keep chatting off the record.",
                    "neutral")
            return True

        if ctx.store.get("stage") != _RUNNING:
            return False

        started = time.monotonic()
        spoken = ctx.reply(text, person_id=ctx.person_id())
        # Recorded only from here, where consent is known to have been given
        # and the exchange actually happened.
        #
        # The persona is the manipulated variable -- the same question answered
        # as Friendly or Professional -- and it is read HERE rather than at
        # arming time because an operator can change it mid-session, which is
        # the whole point of having it. It used to be recorded nowhere at all.
        #
        # Two timings, because they answer different questions: latency_s is
        # the whole turn including speaking it, and first_word_s is how long
        # the participant waited before hearing anything. Only the second is
        # responsiveness; the first grows with the length of the answer.
        from brain import interface

        stats = interface.last_turn_stats()
        persona = ""
        try:
            persona = ctx.state.persona()[0]
        except Exception:  # pragma: no cover - a missing persona must not lose the turn
            pass
        store.record(
            text, spoken, time.monotonic() - started,
            persona=persona,
            first_word_s=float(stats.get("first_word_s") or 0.0),
            backend=str(stats.get("backend") or ""),
        )
        return True

    def on_exit(self, ctx: DemoContext) -> None:
        ctx.state.hold_open_mic(False)
        if ctx.store.get("stage") == _RUNNING:
            ctx.status("Study session paused; still consented.")
