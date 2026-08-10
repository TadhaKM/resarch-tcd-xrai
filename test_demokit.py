"""Exercise the demo framework with fake hardware.

Everything the runner does -- entering demos, dispatching what a visitor said,
switching on a trigger phrase, absorbing a demo that throws -- is logic, and
logic is testable without a robot. The parts that genuinely need hardware
(does the wake word fire, does the speaker crackle) are covered by
tools/selftest.py and by standing in front of the thing.

    python test_demokit.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain.modes import RobotState  # noqa: E402
from demokit.base import Demo, DemoContext, IdleResult  # noqa: E402
from demokit.registry import Registry  # noqa: E402
from demokit.runner import DemoRunner  # noqa: E402

failures = 0


def check(label: str, got, want) -> None:
    global failures
    ok = got == want
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        print(f"       got  {got!r}\n       want {want!r}")


class FakeAudio:
    """Stands in for AudioIO. Wake words and transcripts are scripted."""

    def __init__(self, wake_at=(), transcripts=()):
        self.said = []
        self._wake_at = list(wake_at)
        self._transcripts = list(transcripts)
        self.calls = 0

    def wait_for_wake_word(self, timeout=None):
        self.calls += 1
        if self._wake_at and self._wake_at[0] == self.calls:
            self._wake_at.pop(0)
            return True
        return False

    def listen(self, wait_for_speech_s=None):
        # Signature kept in step with AudioIO.listen, like speak below.
        return self._transcripts.pop(0) if self._transcripts else ""

    def speak(self, text, emotion, motion=None, expressive=False, pace=None, variation=None):
        # Signature kept in step with AudioIO.speak on purpose. When it drifted,
        # ctx.say raised TypeError, the runner's guard absorbed it exactly as it
        # is meant to, and the only symptom was a demo that said nothing -- which
        # is what this check exists to notice.
        self.said.append(text)


class FakeMotion:
    def __init__(self):
        self.expressions = []

    def express(self, tag):
        self.expressions.append(tag)

    def express_move(self, tag):
        pass

    def acknowledge(self):
        pass

    def dance(self, **kw):
        pass


def build(demos, *, wake_at=(), transcripts=()):
    """A runner wired to fake hardware and a registry holding `demos`."""
    registry = Registry()
    registry._publish({d.id: d for d in demos})
    state = RobotState()
    state.set_demos(
        [{"id": d.id, "label": d.label, "help": d.help, "available": True, "note": ""} for d in demos]
    )
    audio, motion = FakeAudio(wake_at, transcripts), FakeMotion()

    import demokit.runner as runner_mod

    runner_mod.REGISTRY = registry
    import demokit.base  # noqa: F401

    runner = DemoRunner(
        audio=audio, motion=motion, tracker=None, state=state, capabilities=frozenset()
    )
    return runner, state, audio, registry


# --- demos used by the tests -------------------------------------------

class Chatty(Demo):
    id, label, help = "chatty", "Chatty", "h"

    def __init__(self):
        self.entered = self.exited = 0
        self.heard = []

    def on_enter(self, ctx):
        self.entered += 1

    def on_idle(self, ctx):
        return IdleResult(listen_for=1.0)

    def on_utterance(self, ctx, text):
        self.heard.append(text)
        ctx.say("noted")
        return True

    def on_exit(self, ctx):
        self.exited += 1


class Dancer(Demo):
    id, label, help = "dancer", "Dancer", "h"
    triggers = ("let's dance",)

    def on_idle(self, ctx):
        return IdleResult(listen_for=1.0)


class Broken(Demo):
    id, label, help = "broken", "Broken", "h"

    def on_idle(self, ctx):
        raise ValueError("student bug")


class Forgetful(Demo):
    """The classic first-demo mistake: no return statement."""

    id, label, help = "forgetful", "Forgetful", "h"

    def on_idle(self, ctx):
        pass


print("[1] entering and leaving a demo")
chatty, dancer = Chatty(), Dancer()
runner, state, audio, _ = build([chatty, dancer])
state.set_mode("chatty")
runner.cycle()
check("on_enter ran once", chatty.entered, 1)
runner.cycle()
check("on_enter not repeated while selected", chatty.entered, 1)
state.set_mode("dancer")
runner.cycle()
check("on_exit ran when switched away", chatty.exited, 1)

print()
print("[2] a wake word turns into an utterance the demo handles")
chatty = Chatty()
runner, state, audio, _ = build([chatty], wake_at=(1,), transcripts=["what is xr"])
state.set_mode("chatty")
runner.cycle()
check("demo received the utterance", chatty.heard, ["what is xr"])
check("and spoke", audio.said, ["noted"])

print()
print("[3] a trigger phrase switches demos mid-sentence")
chatty, dancer = Chatty(), Dancer()
runner, state, audio, _ = build([chatty, dancer], wake_at=(1,), transcripts=["let's dance please"])
state.set_mode("chatty")
runner.cycle()
check("switched to the triggered demo", state.mode, "dancer")

print()
print("[4] sleep phrases always win, whatever is selected")
chatty = Chatty()
runner, state, audio, _ = build([chatty], wake_at=(1,), transcripts=["ok go to sleep now"])
state.set_mode("chatty")
runner.cycle()
check("robot is asleep", state.sleeping, True)
check("demo never saw it", chatty.heard, [])

print()
print("[5] a demo that throws is contained, then set aside")
broken, chatty = Broken(), Chatty()
runner, state, audio, registry = build([broken, chatty])
state.set_mode("broken")
for _ in range(3):
    runner.cycle()
available, reason = registry.is_available("broken", frozenset())
check("set aside after 3 consecutive failures", available, False)
check("with a reason for the operator", "failed" in reason, True)
check("and the robot moved to a working demo", state.mode, "chatty")

print()
print("[6] re-enabling a set-aside demo")
registry.enable("broken")
check("available again", registry.is_available("broken", frozenset())[0], True)

print()
print("[7] the missing-return mistake does not spin the CPU")
forgetful = Forgetful()
runner, state, audio, _ = build([forgetful])
state.set_mode("forgetful")
started = time.monotonic()
for _ in range(5):
    runner.cycle()
check("cycles are cheap, not a busy loop", time.monotonic() - started < 1.0, True)
check("no listening was attempted", audio.calls, 0)

print()
print("[8] a recognised face is greeted by name, once")


class KnownTracker:
    """A tracker that always reports the same recognised person."""

    enabled = True

    def current(self, max_age_s=1.5):
        return (7, object())


class Quiet(Demo):
    id, label, help = "quiet", "Quiet", "h"

    def on_idle(self, ctx):
        return IdleResult(listen_for=1.0)


import brain.db as _db  # noqa: E402

_db.get_person_name = lambda person_id: "Ada" if person_id == 7 else None

quiet = Quiet()
runner, state, audio, _ = build([quiet])
runner._tracker = KnownTracker()
state.set_mode("quiet")
runner.cycle()
greeted_once = [line for line in audio.said if "Ada" in line]
check("greets a known face by name", len(greeted_once), 1)
for _ in range(4):
    runner.cycle()
check("and does not greet again", len([line for line in audio.said if "Ada" in line]), 1)

print()
print("[9] a misheard name can be corrected out loud")

_stored = {7: "Telaget"}
_db.get_person_name = lambda person_id: _stored.get(person_id)
_db.rename_person = lambda person_id, name: _stored.__setitem__(person_id, name)


def correct(said: str):
    """Run one utterance through a runner that can see a known, named face."""
    _stored[7] = "Telaget"
    runner, state, audio, _ = build([Quiet()], wake_at=(1,), transcripts=[said])
    runner._tracker = KnownTracker()
    state.set_mode("quiet")
    runner.cycle()
    return _stored[7]


check("takes the name offered", correct("my name is Tadhagath"), "Tadhagath")
check(
    "reads past the rejected name",
    correct("no my name is not Telaget its Tadhagath"),
    "Tadhagath",
)
# The sentence that made this necessary: cut at the first cue instead of the
# last and the robot renames the visitor to the very name they just rejected.
check("a denial alone changes nothing", correct("thats not my name"), "Telaget")
check("nor does one naming only what is wrong", correct("my name is not Telaget"), "Telaget")
check("ordinary talk is left alone", correct("what does my name mean"), "Telaget")
check("and an article is not a name", correct("can you call me a taxi"), "Telaget")

print()
print("[10] open mic carries a conversation without the wake word")


def converse(open_mic: bool, transcripts, cycles=2):
    """One wake-word turn, then `cycles-1` more, and what the demo heard."""
    chatty = Chatty()
    runner, state, audio, _ = build([chatty], wake_at=(1,), transcripts=list(transcripts))
    state.set_mode("chatty")
    state.set_open_mic(open_mic)
    for _ in range(cycles):
        runner.cycle()
    return chatty.heard, audio.calls, runner


heard, wake_calls, _ = converse(True, ["what is xr", "and what about vr"])
check("the follow-up needs no wake word", heard, ["what is xr", "and what about vr"])
check("and none was waited for", wake_calls, 1)

heard, _, _ = converse(False, ["what is xr", "and what about vr"])
check("off, the second question is not taken", heard, ["what is xr"])

# Said out of habit for the first minute of every open-mic conversation. Left
# in, the model is asked a question that begins with its own name.
heard, _, _ = converse(True, ["what is xr", "hey reachy what about vr"])
check("a habitual wake phrase is stripped", heard[-1], "what about vr")

# Quiet for long enough and it goes back to needing the wake word, rather than
# staying open all afternoon on the strength of one conversation at eleven.
_h, _c, runner = converse(True, ["what is xr"])
runner._open_until = 0.0
before = runner._audio.calls
runner.cycle()
check("a lapsed window waits for the wake word again", runner._audio.calls, before + 1)

print()
print("[11] audio from the wrong thread is refused, not silently interleaved")
import threading  # noqa: E402

ctx = DemoContext(
    audio=FakeAudio(), motion=FakeMotion(), tracker=None,
    state=RobotState(), demo_id="t", store={},
)
error = []


def from_other_thread():
    try:
        ctx.say("hello from the web thread")
    except RuntimeError as exc:
        error.append(str(exc))


t = threading.Thread(target=from_other_thread)
t.start()
t.join()
check("RuntimeError raised off-thread", bool(error), True)
check("and it explains why", "one thread owns the microphone" in (error[0] if error else "").lower(), True)

print()
print(f"{'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
sys.exit(1 if failures else 0)
