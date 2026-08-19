"""Runs an HRI study session: asks for consent, then records the exchanges.

The consent conversation is a demo rather than a dashboard checkbox because
consent is something a participant gives, not something an operator ticks on
their behalf. The robot says what is being recorded, waits for a yes, and
records nothing until it has one -- and a no ends the session rather than
merely muting it.

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

_CONSENT = "consent"
_RUNNING = "running"
_DONE = "done"

_ANSWER_WAIT_S = 6.0
_BETWEEN_LINES_S = 1.0

#: Said before anything is recorded. Every clause is here because a
#: participant information sheet would require it: what is collected, what is
#: not, what it is for, and that they can stop.
_SCRIPT = (
    "Before we start -- the Hub is researching how people talk with robots.",
    "If you agree, I'll keep a written record of what we say to each other, "
    "with no name and no photo attached to it.",
    "You can ask me to delete it at any point, and you can say no now and we "
    "will just have a normal chat.",
)

_ASK = "Would you like to take part?"

#: Asked when the first answer arrived but could not be read. Short, because
#: the participant has already heard the whole notice and is being asked to
#: repeat one word.
_REASK = "Sorry -- I did not catch that. Is that a yes or a no?"

#: Consent language, read BEFORE the general yes/no reader from the enrolment
#: exchange. That reader's vocabulary was built for "want me to remember you?"
#: and contains no word anybody uses on a consent form: live, a participant
#: said "I consent to be recorded" -- the most explicit consent there is -- and
#: was told "I'll take that as a no", because not one of "i/consent/to/be/
#: recorded" is in it. Somebody answering a consent question answers it in
#: consent language, so that language has to be read here.
_CONSENT_YES = (
    "i consent", "consent to", "i agree", "happy to take part", "happy to",
    "id like to take part", "i want to take part", "take part", "participate",
    "count me in", "you can record", "you may record", "feel free to record",
    "thats fine", "that is fine", "no objection", "i accept", "im in",
)

#: Refusals in the same register, and they are checked FIRST and always. "I do
#: not consent to be recorded" contains "consent to", so any other order reads
#: the clearest possible refusal as agreement -- which is the one mistake this
#: file must never make.
_CONSENT_NO = (
    "do not consent", "dont consent", "not consent", "i refuse", "opt out",
    "do not record", "dont record", "no recording", "rather not take part",
    "do not want to take part", "dont want to take part",
)


def _phrase_in(phrase: str, words: str) -> bool:
    from demokit.runner import _word_stream

    return f" {_word_stream(phrase).strip()} " in words


def _explicit_consent(text: str) -> str:
    """"yes"/"no" for consent-form language, "" for anything else.

    Separate from the general reader so a BARE "yes" cannot reach the paths
    that accept consent outside the question -- a stray "yeah" in a later
    conversation must never start a recording.
    """
    from demokit.runner import _word_stream
    from demos.vision import _NO, _YES

    words = _word_stream(text)
    if any(_phrase_in(p, words) for p in _CONSENT_NO):
        return _NO
    if any(_phrase_in(p, words) for p in _CONSENT_YES):
        return _YES
    return ""

#: Ways somebody withdraws mid-session. Checked on every utterance, because a
#: withdrawal that only works when the robot happens to ask is not a
#: withdrawal.
_WITHDRAW = (
    "delete my data", "delete that", "withdraw", "take me out of the study",
    "forget what i said", "stop recording", "i changed my mind",
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
        ctx.store.update(stage=_CONSENT, line=0)
        ctx.status(f"Study session {store.status()['session']}: awaiting consent.")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        store_ = ctx.store
        stage = store_.get("stage")

        if stage == _CONSENT:
            index = store_.get("line", 0)
            if index < len(_SCRIPT):
                # One sentence per slice, as everything scripted here is: the
                # robot must stay switchable while it reads a consent notice.
                ctx.say(_SCRIPT[index], "neutral")
                store_["line"] = index + 1
                return IdleResult(listen_for=_BETWEEN_LINES_S)
            return self._ask_consent(ctx)

        if stage == _RUNNING:
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def _ask_consent(self, ctx: DemoContext) -> IdleResult:
        reasked = bool(ctx.store.get("reasked"))
        ctx.state.hold_open_mic(True)
        answer = ctx.ask(_REASK if reasked else _ASK, "curious",
                         wait_for_speech_s=_ANSWER_WAIT_S)
        ctx.state.hold_open_mic(False)
        # Consent-form language first, then the three-way reader from the
        # enrolment exchange -- which already knows "why not" and "go on then"
        # are yes, and that silence is not consent.
        from demos.vision import _NO, _YES, _read_answer

        verdict = _explicit_consent(answer) or _read_answer(answer)
        if verdict is _YES:
            store.consent(True)
            ctx.store["stage"] = _RUNNING
            ctx.status("Consent given; recording this session.")
            ctx.say("Thank you. Ask me anything.", "happy")
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)
        if verdict is _NO:
            store.consent(False)
            ctx.store["stage"] = _DONE
            ctx.say("No bother at all. Nothing is being recorded -- ask me anything.",
                    "happy")
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)
        # Somebody SPOKE and was misheard -- ask once more before deciding for
        # them. Live, a participant's answer came back from the recogniser as
        # the single word "Everything" and was booked as a refusal on the spot;
        # they then said "I consent to be recorded" to a robot that had already
        # stopped asking. Silence still ends it, unchanged and deliberately: a
        # participant who says nothing to a consent question has answered it,
        # and re-asking silence is pestering somebody who is walking away.
        if answer and answer.strip() and not reasked:
            ctx.store["reasked"] = True
            ctx.status("Answer unclear; asking once more.")
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        store.consent(False)
        ctx.store["stage"] = _DONE
        ctx.status("No clear consent; nothing recorded.")
        ctx.say("I'll take that as a no. Nothing recorded -- ask away.", "neutral")
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

        stage = ctx.store.get("stage")
        # Consent offered AFTER the asking stopped. Live, the robot misheard an
        # answer, said "I'll take that as a no", and thirty seconds later the
        # participant said "I consent to be recorded" -- which fell through to
        # the conversation demo and got "Thanks for letting me know!". They had
        # heard the notice in full and could not have been clearer, so this
        # honours it. Only the explicit consent-form wording counts: a bare
        # "yeah" later in a chat is answering something else, and must never
        # start a recording.
        if stage == _DONE and store.running() and _explicit_consent(text) == "yes":
            store.consent(True)
            ctx.store["stage"] = _RUNNING
            ctx.status("Consent given after the question; recording this session.")
            ctx.say("Thank you -- I have you down as taking part. Ask me anything.",
                    "happy")
            return True

        if stage != _RUNNING:
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
