"""Spoken timers, which keep counting whatever else the robot is doing.

Migrated out of body/voice_loop.py, where timers were one of the five things
that file did at once. Two pieces survive deliberately unchanged: the parser,
because tools/selftest.py imports `_parse_timer` and pins the phrasings the
recogniser actually produces, and the rule that shaped the first version --
only the voice-loop thread may drive the speaker, so whatever notices that a
timer is due cannot simply say so from there.

That rule plus the way demokit/runner.py polls gives this file its shape:

- Pending timers live at module scope rather than in ctx.store. A visitor sets
  a timer, the operator switches to another demo, and the timer must still
  ring; ctx.store belongs to this demo and would strand it.
- on_idle rings the due ones, because a hook runs on the voice-loop thread and
  may speak. But the runner only calls hooks on the SELECTED demo, so a timer
  coming due while something else is on screen would never be looked at. One
  small watcher thread covers exactly that gap, and it never speaks: it puts
  the line on the same request queue the dashboard's Say button uses, which the
  runner pops at the top of every cycle whatever is selected, and even while
  the robot is asleep. Queueing is thread-safe; speaking is not, and
  DemoContext raises to prove it.
- Claiming a due timer removes it from the shared list, so whichever of those
  two rings it is briefly the only thing that knows it exists. Every path that
  claims one therefore hands it back if it did not manage to say it. The demo
  is allowed to be silent for a moment, muddled, or set aside by the runner; it
  is not allowed to promise a visitor a ring and then never ring.
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from demokit import Demo, DemoContext, IdleResult


logger = logging.getLogger(__name__)

#: Word numbers the recogniser actually emits for small counts; digits cover
#: the rest. "a"/"an" are here because "remind me in an hour" is how people
#: speak, and it has to mean 1.
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60,
}

_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}

#: The count is captured as one loose token so both "5" and "forty-five" land
#: in the same group. `timer[^a-z0-9]*(?:for|of)?` absorbs the comma the
#: recogniser sometimes inserts and the "for" people usually say but not
#: always ("timer, five minutes").
_TIMER_RE = re.compile(
    r"(?:timer[^a-z0-9]*(?:for|of)?|remind me in)\s+([a-z0-9-]+)\s+(second|minute|hour)s?"
)

#: Five seconds is the shortest timer worth arming. The robot spends a second
#: or two confirming the timer out loud and only checks for due ones once per
#: idle slice, so anything shorter rings while it is still saying "timer set",
#: which reads as a fault rather than a feature. The two-hour ceiling is not a
#: technical limit -- it is the point past which the promise outlives the visit.
#: Both bounds carried over from the original implementation, and the lower one
#: is asserted by tools/selftest.py ("timer for 3 seconds" must not parse).
_MIN_TIMER_S = 5
_MAX_TIMER_S = 2 * 3600


def _parse_timer(spoken: str) -> Optional[tuple[int, str]]:
    """Extract (seconds, human description) from a timer request, else None.

    Handles digits and the number words the recogniser produces: "set a timer
    for five minutes", "timer for 30 seconds", "remind me in an hour".

    The description is the parser's own normalised form -- "5 minute", singular
    -- not the words that were said. tools/selftest.py pins both halves of the
    return value, so change the shape here and the selftest is the contract you
    are breaking.
    """
    text = spoken.lower()
    if "timer" not in text and "remind me in" not in text:
        return None
    # Before the regex: "half an hour" would otherwise match "an hour" and set
    # a timer for twice what was asked.
    if "half an hour" in text:
        return 1800, "30 minute"
    if "half a minute" in text:
        return 30, "30 second"

    match = _TIMER_RE.search(text)
    if not match:
        return None
    raw, unit = match.group(1), match.group(2)
    amount = _NUMBER_WORDS.get(raw)
    if amount is None:
        if not raw.isdigit():
            return None
        amount = int(raw)
    seconds = amount * _UNIT_SECONDS[unit]
    if not _MIN_TIMER_S <= seconds <= _MAX_TIMER_S:
        return None
    return seconds, f"{amount} {unit}"


# --- pending timers, shared across demos --------------------------------


@dataclass(frozen=True)
class _Pending:
    """One armed timer. `due_at` is monotonic, so changing the wall clock or
    the machine suspending does not make a timer ring early."""

    due_at: float
    description: str


_PENDING: list[_Pending] = []
_LOCK = threading.Lock()
_WATCHER: Optional[threading.Thread] = None
#: Lets cancel-all wake the watcher so it exits at once instead of after the
#: poll. Also the watcher's sleep: threading.Event.wait is interruptible, which
#: time.sleep is not.
_WAKE = threading.Event()

_WATCH_POLL_S = 0.5

#: How long a due timer waits for the hook before the watcher hands it to the
#: request queue. Long enough that a ring already being spoken from on_idle is
#: never queued a second time, short enough that a visitor who switched to
#: another demo does not think the robot forgot.
_HANDOFF_GRACE_S = 5.0


def _arm(seconds: int, description: str, state: Any) -> None:
    """Start a timer and make sure something is watching it."""
    with _LOCK:
        _PENDING.append(_Pending(time.monotonic() + seconds, description))
        _PENDING.sort(key=lambda pending: pending.due_at)
    logger.info("Timer set: %s (%ds)", description, seconds)
    _ensure_watcher(state)


def _claim_due(cutoff: float) -> list[_Pending]:
    """Take the timers due at or before `cutoff`, removing them in one step.

    Claiming and removing under the same lock is what stops the hook and the
    watcher both ringing the same timer: whichever gets here first is the only
    one that sees it.
    """
    with _LOCK:
        due = [pending for pending in _PENDING if pending.due_at <= cutoff]
        if due:
            _PENDING[:] = [pending for pending in _PENDING if pending.due_at > cutoff]
    return due


def _restore(items: list[_Pending], state: Any) -> None:
    """Put claimed timers back because they were not actually rung.

    Restarting the watcher is part of putting them back rather than an extra
    courtesy: a claim that emptied _PENDING lets the watcher notice the empty
    list and exit within one poll, so by the time a failed ring hands its
    timers over there may be nothing left watching them. Restored but unwatched
    sounds exactly like dropped -- silence -- if the operator then switches to
    another demo, because only the watcher rings a timer the selected demo will
    never look at.
    """
    if not items:
        return
    with _LOCK:
        _PENDING.extend(items)
        _PENDING.sort(key=lambda pending: pending.due_at)
    # Outside the lock: _ensure_watcher takes _LOCK too and it is not reentrant.
    _ensure_watcher(state)


def _cancel_all() -> int:
    """Drop every timer, returning how many there were."""
    with _LOCK:
        count = len(_PENDING)
        _PENDING.clear()
    _WAKE.set()
    return count


def _remaining(now: float) -> list[tuple[float, str]]:
    """(seconds left, description) for each timer, soonest first."""
    with _LOCK:
        return [(max(0.0, pending.due_at - now), pending.description) for pending in _PENDING]


def _ensure_watcher(state: Any) -> None:
    """Start the watcher thread if one is not already running.

    `state` is the shared RobotState; it is only ever queued to, never read for
    a decision, so the watcher does not care which demo is selected.
    """
    global _WATCHER
    with _LOCK:
        if _WATCHER is not None and _WATCHER.is_alive():
            return
        _WATCHER = threading.Thread(target=_watch, args=(state,), name="reachy-timers", daemon=True)
        _WATCHER.start()


def _watch(state: Any) -> None:
    """Hand over the timers the selected demo is never going to ring itself.

    Exists only while something is pending, and never touches audio: it queues
    the line with state.request, which demokit/runner.py pops at the top of
    every cycle. Calling ctx.say from here would raise -- DemoContext checks the
    calling thread -- and if it did not, two threads would drive one speaker.

    Clearing _WATCHER inside the same critical section that finds the list
    empty is what closes the race with _arm: a timer added just before this
    check is seen here, and one added just after finds _WATCHER already None
    and starts a fresh thread.
    """
    global _WATCHER
    while True:
        _WAKE.wait(_WATCH_POLL_S)
        _WAKE.clear()
        for pending in _claim_due(time.monotonic() - _HANDOFF_GRACE_S):
            if not state.request("say", _ring_line(pending.description), "happy"):
                # The request queue holds five and refuses the sixth. Keep the
                # timer and try again next pass rather than dropping a ring.
                # _restore's _ensure_watcher call finds this very thread alive
                # and returns, so restoring from inside the watcher is safe.
                _restore([pending], state)
        with _LOCK:
            if not _PENDING:
                _WATCHER = None
                return


# --- what the robot says ------------------------------------------------


def _ring_line(description: str) -> str:
    # "your 5 minute timer" is correct as a compound modifier, which is why the
    # description is stored singular; the plural form is only needed below.
    return f"Ding ding! Your {description} timer is finished."


def _spoken_duration(description: str) -> str:
    """'5 minute' -> '5 minutes', '1 hour' -> 'an hour'.

    The parser's singular form reads wrong in "timer set for 5 minute", and the
    form itself is fixed by the selftest, so the grammar is repaired here at the
    one place it is spoken as a duration rather than as a modifier.
    """
    amount, unit = description.split(" ", 1)
    if amount == "1":
        # By vowel SOUND, not spelling: "an hour" is right and "a hour" is the
        # kind of wrongness a synthesised voice makes impossible to ignore.
        return "an hour" if unit == "hour" else f"a {unit}"
    return f"{description}s"


def _rough(seconds: float) -> str:
    """Time left in units a listener can hold. Nobody wants "247 seconds"."""
    seconds = max(0, int(round(seconds)))
    if seconds <= 1:
        return "a second"
    if seconds < 90:
        return f"{seconds} seconds"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"about {minutes} minutes"
    hours, minutes = divmod(minutes, 60)
    hour_part = "an hour" if hours == 1 else f"{hours} hours"
    return f"about {hour_part}" if not minutes else f"about {hour_part} and {minutes} minutes"


def _remaining_line(items: list[tuple[float, str]]) -> str:
    """One sentence covering what is still running, for saying or for the dashboard."""
    if not items:
        return "No timers running."
    left, description = items[0]
    line = f"Your {description} timer has {_rough(left)} to go."
    if len(items) > 1:
        line += f" And {len(items) - 1} more after that."
    return line


#: Spoken one line per idle slice, never as one utterance: nothing consumes the
#: microphone while the robot talks, so a paragraph is a paragraph of deafness.
_INTRO_LINES = (
    "I can keep time for you. Say, set a timer for five minutes, and I'll tell you when it's up.",
    "It keeps counting even if we move on to something else, so just ask me to cancel it if you change your mind.",
)

_CANCEL_WORDS = ("cancel", "forget", "stop", "clear")

#: Only consulted when the transcript also mentions a timer, so "how long have
#: you been here" still goes to the language model. Phrases rather than single
#: words for the same reason: bare "any" matches inside "company", and this is
#: a business school.
_QUERY_PHRASES = ("how long", "how much", "how many", "left", "still", "what timer", "any timer", "check my")

_LISTEN_S = 2.0
#: Floor on the listening window. Without it, a timer a fraction of a second
#: away would have the idle loop spinning instead of listening.
_MIN_LISTEN_S = 0.5
#: Gap between unspoken countdown notes on the dashboard.
_STATUS_EVERY_S = 20.0


class Timers(Demo):
    label = "Timers"
    help = "Spoken timers. They keep counting, and ring, even after you switch demos."
    order = 85
    #: "remind me in" is deliberately not a trigger, though _parse_timer still
    #: accepts it once this demo is selected. As a trigger it fires on "remind
    #: me in simple terms how the camera works" and "remind me in what year the
    #: hub was founded" -- where the parser then finds no duration, so the robot
    #: switches demo and answers nothing.
    triggers = ("set a timer", "set a reminder", "start a timer")

    def on_enter(self, ctx: DemoContext) -> None:
        """Selected. Stages what to say; on_idle speaks it a line at a time.

        Nothing is said here because a trigger phrase enters this demo and then
        immediately delivers the utterance that set it off: on_utterance clears
        the staged script, so "set a timer for five minutes" is answered with
        the confirmation rather than with an introduction nobody asked for.
        """
        now = time.monotonic()
        items = _remaining(now)
        if not ctx.store.get("greeted"):
            ctx.store["script"] = list(_INTRO_LINES)
        elif items:
            # Been here before: what a returning operator wants is the count,
            # not the pitch.
            ctx.store["script"] = [_remaining_line(items)]
        else:
            ctx.store["script"] = []
        ctx.store["greeted"] = True
        ctx.status(_remaining_line(items))

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        now = time.monotonic()

        due = _claim_due(now)
        if due:
            _ring(ctx, due)
            return IdleResult(listen_for=_LISTEN_S)

        script = ctx.store.get("script")
        if script:
            ctx.say(script.pop(0), "curious")
            return IdleResult(listen_for=_LISTEN_S)

        self._note_countdown(ctx, now)
        return IdleResult(listen_for=_listen_window(now))

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        spoken = text.lower()
        mentions_timer = "timer" in spoken

        if mentions_timer and any(word in spoken for word in _CANCEL_WORDS):
            # Checked before parsing: "cancel the five minute timer" would
            # otherwise set a second one.
            ctx.store["script"] = []
            count = _cancel_all()
            ctx.say(
                f"Okay, cancelled {count} timer{'s' if count != 1 else ''}."
                if count else "There were no timers running.",
                "neutral",
            )
            return True

        if mentions_timer and any(phrase in spoken for phrase in _QUERY_PHRASES):
            ctx.store["script"] = []
            ctx.say(_remaining_line(_remaining(time.monotonic())), "neutral")
            return True

        parsed = _parse_timer(spoken)
        if parsed is not None:
            seconds, description = parsed
            _arm(seconds, description, ctx.state)
            ctx.store["script"] = []
            ctx.status(f"Timer set for {_spoken_duration(description)} ({seconds}s)")
            ctx.say(f"Timer set for {_spoken_duration(description)}. I'll let you know.", "happy")
            return True

        if any(trigger in spoken for trigger in self.triggers):
            # A trigger phrase got us here, so the visitor meant a timer even
            # though no duration survived the recogniser. Asking is better than
            # falling through to the language model, which would answer as if
            # it had set one.
            ctx.store["script"] = []
            ctx.say("How long for?", "curious")
            return True

        # Anything else is ordinary conversation, and this demo has no business
        # holding onto it.
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        """Switched away. The timers deliberately keep running.

        Cancelling here would be the tidy-looking mistake: a visitor sets a
        timer, the operator moves to another demo to fill the wait, and the
        whole point is that the ring still arrives. The watcher thread carries
        them from here.
        """
        # A half-spoken introduction should not resume ten minutes later.
        ctx.store["script"] = []
        items = _remaining(time.monotonic())
        if items:
            ctx.status(f"{len(items)} timer(s) still counting in the background")

    def _note_countdown(self, ctx: DemoContext, now: float) -> None:
        """Keep the dashboard's countdown honest, unspoken and throttled.

        Unthrottled this writes an event every idle slice and buries everything
        else an operator is trying to read.
        """
        items = _remaining(now)
        if not items or now - ctx.store.get("noted_at", 0.0) < _STATUS_EVERY_S:
            return
        ctx.store["noted_at"] = now
        ctx.status(_remaining_line(items))


def _ring(ctx: DemoContext, due: list[_Pending]) -> None:
    """Announce timers that have come due, from the hook that may speak.

    _claim_due already removed these from _PENDING, so while this function runs
    they exist nowhere else in the process: any exit that leaves a line unspoken
    has deleted a timer outright, and the watcher cannot recover what it can no
    longer see. That is the one failure this demo must not have.

    DemoStopped is merely the tidiest way out. A dead audio device, a piper
    crash or a stalled link to the robot raise ordinary exceptions, and those
    are absorbed just as quietly by the runner's guard -- the demo gets an error
    on the dashboard and the timer is simply gone. So the hand-back is a finally
    rather than an except clause: every path out that did not speak a line gives
    it back, and the exception still travels on to the guard untouched.
    """
    unspoken = list(due)
    try:
        for pending in due:
            ctx.say(_ring_line(pending.description), "happy")
            # Dropped only once say has returned. A line that failed part way
            # through is counted unspoken and rung again, because half a ring
            # heard twice is a visitor hearing their timer and a lost ring is a
            # promise the robot made out loud and then broke.
            unspoken.pop(0)
    finally:
        _restore(unspoken, ctx.state)
    ctx.motion.express_move("happy")


def _listen_window(now: float) -> float:
    """Come back before the next timer is due, so the ring lands on time.

    The runner clamps this to MAX_LISTEN_WINDOW_S regardless; shortening it as
    a timer approaches is what keeps a 30 second timer from ringing at 32.
    """
    items = _remaining(now)
    if not items:
        return _LISTEN_S
    return max(_MIN_LISTEN_S, min(_LISTEN_S, items[0][0]))
