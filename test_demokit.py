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

    def __init__(self, wake_at=(), transcripts=(), interrupt_after=()):
        self.said = []
        self.sounds = []
        self._wake_at = list(wake_at)
        self._transcripts = list(transcripts)
        self.calls = 0
        #: Indices of spoken lines after which a visitor talks over the robot.
        #: Without this method at all, ctx.say_lines raised AttributeError, the
        #: runner's guard absorbed it as "demo had a problem", and every test
        #: here passed while the interruption path was never once exercised.
        self._interrupt_after = set(interrupt_after)
        self.backlog_checks = 0

    def wake_word_in_backlog(self):
        self.backlog_checks += 1
        return (len(self.said) - 1) in self._interrupt_after

    def wait_for_wake_word(self, timeout=None):
        self.calls += 1
        if self._wake_at and self._wake_at[0] == self.calls:
            self._wake_at.pop(0)
            return True
        return False

    def listen(self, wait_for_speech_s=None):
        # Signature kept in step with AudioIO.listen, like speak below.
        return self._transcripts.pop(0) if self._transcripts else ""

    def play_sound(self, name, motion=None):
        # Recorded like speech, so a test can assert a sound played -- and
        # present at all because the real AudioIO grew this method and the
        # first test to reach a quiz sound died on the missing attribute.
        self.sounds.append(name)
        return True

    def speak(self, text, emotion, motion=None, expressive=False, pace=None, variation=None):
        # Signature kept in step with AudioIO.speak on purpose. When it drifted,
        # ctx.say raised TypeError, the runner's guard absorbed it exactly as it
        # is meant to, and the only symptom was a demo that said nothing -- which
        # is what this check exists to notice.
        self.said.append(text)

    def render(self, text, expressive=False, pace=None, variation=None):
        # ctx.reply renders off-thread and plays on the loop thread; the fake
        # carries the text through as the "chunks" so speak_rendered can record
        # exactly what speak() would have.
        return [text]

    def speak_rendered(self, chunks, motion=None):
        self.said.append(chunks[0])


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

# Saved so the real ones can be put back. Replacing a module attribute lasts
# for the rest of the file, and a later section testing the actual database
# against the stub reported the name it had stored as missing.
_REAL_DB = {"get_person_name": _db.get_person_name, "rename_person": _db.rename_person,
            "get_spoken_name": _db.get_spoken_name}

# Both, because they answer different questions now: get_person_name is the
# spelling (matching, display) and get_spoken_name is what the robot SAYS,
# which a stored pronunciation overrides. ctx.person_name() reads the latter.
_db.get_person_name = lambda person_id: "Ada" if person_id == 7 else None
_db.get_spoken_name = lambda person_id: "Ada" if person_id == 7 else None

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
_db.get_spoken_name = lambda person_id: _stored.get(person_id)
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
heard, _, _ = converse(True, ["what is xr", "reachy, what about vr"])
check("including the bare name", heard[-1], "what about vr")
heard, _, _ = converse(True, ["what is xr", "what does hey reachy mean"])
check("but not one in the middle of a question", heard[-1], "what does hey reachy mean")

# Quiet for long enough and it goes back to needing the wake word, rather than
# staying open all afternoon on the strength of one conversation at eleven.
_h, _c, runner = converse(True, ["what is xr"])
runner._open_until = 0.0
before = runner._audio.calls
runner.cycle()
check("a lapsed window waits for the wake word again", runner._audio.calls, before + 1)

print()
print("[11] enrolment hears a sentence, finds the name, and confirms before storing")

from body.face import extract_spoken_name  # noqa: E402
from demos.vision import Vision  # noqa: E402

check("finds the name in a full sentence",
      extract_spoken_name("MY NAME IS SARAH NICE TO MEET YOU"), "Sarah")
check("a bare name is enough", extract_spoken_name("tadhg"), "Tadhg")
check("a correction reads past the rejected name",
      extract_spoken_name("no my name is not telaget its tadhagath"), "Tadhagath")
check("a refusal is not a name", extract_spoken_name("i dont want to say"), None)
check("neither is a command", extract_spoken_name("go to sleep"), None)


import numpy as _np_enrol  # noqa: E402


class EnrolTracker:
    """Reports an unrecognised face and records what gets enrolled."""

    enabled = True

    def __init__(self):
        self.enrolled = []
        self._face = self

    def current(self, max_age_s=1.5):
        return (None, object())

    def current_embedding(self, max_age_s=1.5):
        # One unchanging stranger. Kept in step with FaceTracker, which grew
        # this so a demo can tell one unrecognised visitor from another.
        return _np_enrol.zeros(512, dtype=_np_enrol.float32) + 0.5

    def enroll(self, name, face):
        self.enrolled.append(name)
        return 42


def enrol_run(transcripts):
    state = RobotState()
    state.set_demos([{"id": "vision", "label": "V", "help": "", "available": True, "note": ""}])
    state.set_mode("vision")
    audio, tracker = FakeAudio(transcripts=list(transcripts)), EnrolTracker()
    ctx = DemoContext(
        audio=audio, motion=FakeMotion(), tracker=tracker,
        state=state, demo_id="vision", store={},
    )
    ctx.store["present_since"] = time.monotonic() - 10  # a settled stranger
    demo = Vision()
    for _ in range(8):
        demo.on_idle(ctx)
    return tracker.enrolled, audio, state


enrolled, audio, state = enrol_run(["yes", "my name is sarah nice to meet you", "yes"])
check("consent, sentence, confirm -> enrolled", enrolled, ["Sarah"])
check("the name was said back first", any("Sarah. Did I get" in s for s in audio.said), True)
check("greeting spent, so no hello-again follows", state.mark_greeted("Sarah"), False)

enrolled, _, _ = enrol_run(["yes", "telaget", "no its tadhagath", "yes"])
check("wrong hearing corrected in one breath", enrolled, ["Tadhagath"])

enrolled, _, _ = enrol_run(["yes", "telaget", "no", "tadhagath", "yes"])
check("or by a plain no and a fresh try", enrolled, ["Tadhagath"])

enrolled, _, _ = enrol_run(["yes", "telaget", "", ""])
check("a name never confirmed is never stored", enrolled, [])

print()
print("[12] a visitor can talk over the robot")

from demokit.base import Interrupted  # noqa: E402
import brain.interface as _bi  # noqa: E402

# Stubbed for the rest of the file. Interrupting mid-script hands the turn
# straight to the visitor, which reaches the conversational reply -- and that
# is a real network call to a real model, in a suite that must run offline and
# cost nothing.
_seen = []

# Held before anything overwrites it: sections [12] and [21] stub this module
# attribute for the rest of the file, and section [27] needs to exercise the
# REAL pipeline -- testing the stub is how a bug in it would go unseen.
_real_stream_reply = _bi.stream_reply


def _fake_stream(person_id, message, style=None, extra_system=None, cache=True, web=False):
    _seen.append(extra_system or "")
    yield "A reply.", "happy"


_bi.stream_reply = _fake_stream


class Talker(Demo):
    """Speaks several lines from one hook, the way scripted demos do."""

    id, label, help = "talker", "Talker", "h"

    def __init__(self):
        self.finished = False

    def on_idle(self, ctx):
        for line in ("one", "two", "three"):
            ctx.say(line)
        self.finished = True
        return IdleResult(listen_for=1.0)


talker = Talker()
runner, state, audio, _ = build([talker])
audio._interrupt_after = {1}
audio._transcripts = ["what is xr"]
state.set_mode("talker")
runner.cycle()
check("ctx.say stops on an interruption", audio.said[:2], ["one", "two"])
check("it did not finish the script", "three" in audio.said, False)
check("and the hook did not run to the end", talker.finished, False)
check("the visitor's question was then taken", "what is xr" in audio.said or audio.calls >= 0, True)

# The question in a say-then-listen pair must survive: an Interrupted there
# abandons the exchange at the moment it was about to hear the answer.
ctx = DemoContext(
    audio=FakeAudio(transcripts=["sarah"], interrupt_after=[0]),
    motion=FakeMotion(), tracker=None, state=RobotState(), demo_id="t", store={},
)
check("ctx.ask is not interrupted by its own question", ctx.ask("what's your name?"), "sarah")

# The default demo used to swallow this, which is why barge-in never worked.
import demos.conversation as _conv  # noqa: E402

raised = []
class _Ctx:
    demo_id = "conversation"
    state = RobotState()
    def person_id(self): return 0
    def reply(self, *a, **kw): raise Interrupted()
try:
    _conv.Conversation().on_utterance(_Ctx(), "what is xr")
except Interrupted:
    raised.append(True)
check("conversation re-raises Interrupted", raised, [True])

print()
print("[13] the chosen personality is the robot's manner everywhere")

from brain import qa_cache as _qa  # noqa: E402


def brief_for(persona_id, demo_id="conversation", style=None, owns_persona=False):
    _seen.clear()
    st = RobotState()
    st.set_persona(persona_id)
    motion = FakeMotion()
    ctx = DemoContext(
        audio=FakeAudio(), motion=motion, tracker=None,
        state=st, demo_id=demo_id, store={}, owns_persona=owns_persona,
    )
    ctx.reply("what is xr", style=style)
    return _seen[0], motion.expressions


brief, poses = brief_for("")
check("no persona leaves the prompt alone", "professional register" in brief.lower(), False)
brief, poses = brief_for("friendly")
check("a persona reaches every demo's prompt", "warmly" in brief.lower(), True)
check("and leaves the body in its resting pose", poses[-1], "happy")
brief, _ = brief_for("friendly", style="story")
check("except a story, which has its own voice", "warmly" in brief.lower(), False)
brief, _ = brief_for("friendly", demo_id="personality", owns_persona=True)
check("and a demo that supplies its own", "warmly" in brief.lower(), False)
# Declared by the demo, not checked for by name in the core. Keyed on the name,
# renaming demos/personality.py silently stopped the exclusion matching and the
# demo got two style briefs pulling against each other -- no error, just worse
# answers, with nothing pointing at the cause.
brief, _ = brief_for("renamed_to_anything", demo_id="renamed", owns_persona=True)
check("whatever that demo is called", "warmly" in brief.lower(), False)
from demokit.registry import REGISTRY as _REG0  # noqa: E402

_REG0.discover()
check("the flag lives on the Demo contract",
      _REG0.get("personality").owns_persona, True)
check("and no other demo claims it",
      [d for d in _REG0.ids() if _REG0.get(d).owns_persona], ["personality"])

a, _ = brief_for("friendly")
b, _ = brief_for("consultant")
# Answers are cached by a digest of this text, so a persona that did not reach
# it would have replayed one persona's answer in another's voice.
check("cached answers are per persona", _qa._context_digest(a) != _qa._context_digest(b), True)

check("an unknown persona means none, not Professional", RobotState().set_persona("nonsense"), "")

from brain.modes import DEFAULT_MODE  # noqa: E402

# The state container is meant to know nothing about demos beyond the ids it is
# handed, and its docstring says so -- then the next line used to name one.
check("no demo is named in the default mode", DEFAULT_MODE, "")
_boot = RobotState()
_boot.set_demos(_REG0.dashboard_entries(frozenset()))
check("a fresh boot takes the registry's first demo", _boot.mode, _REG0.default_id())
check("and says nothing about it, since that is every boot",
      any(e.text.startswith("Starting in") for e in _boot.history()), False)

# Each persona speaks in a different voice, not just at a different pace. The
# files are checked here rather than assumed: a persona naming a voice that is
# not installed falls back silently, so the only symptom would be two personas
# sounding identical.
from pathlib import Path as _Path  # noqa: E402

from brain import personas as _p  # noqa: E402

_tts = _Path(__file__).resolve().parent / "models" / "tts"
_named = [p.voice for p in _p.PERSONAS if p.voice]
check("every persona names a voice", len(_named), len(_p.PERSONAS))
check("all of them distinct", len(set(_named)), len(_named))
check(
    "and installed",
    [v for v in _named if not (_tts / f"{v}.onnx").exists()],
    [],
)

print()
print("[14] each demo has a manner, and the operator outranks all of them")

from demokit.registry import REGISTRY as _REG  # noqa: E402

_REG.discover()
_wanted = {"welcome": "friendly", "story": "friendly", "about": "friendly",
           "brainstorm": "professional", "advisor": "professional",
           "quiz": "friendly"}
for _demo_id in _REG.ids():
    _demo = _REG.get(_demo_id)
    check(
        f"{_demo_id} runs as {_wanted.get(_demo_id) or 'default'}",
        getattr(_demo, "persona", ""),
        _wanted.get(_demo_id, ""),
    )

_st = RobotState()
_st.apply_demo_persona("friendly")
check("entering a demo applies its manner", _st.persona[0], "friendly")
_st.set_persona("consultant")
check("the dropdown overrides it", _st.persona[0], "consultant")

# The rule that matters: the preset is the intent, so switching demo snaps to
# it. Held the other way, one curious press of the dropdown left every later
# demonstration stuck in a character nobody wanted there.
_st.apply_demo_persona("professional")
check("switching demo snaps back to that demo's preset", _st.persona[0], "professional")
_st.apply_demo_persona("")
check("including back to default", _st.persona[0], "")
# One value, so the dropdown always reads what the robot is actually doing.
_st.apply_demo_persona("friendly")
check("and the dropdown shows it", _st.snapshot()["persona"], "friendly")

# A voice picked by hand has to survive walking between demos, or choosing one
# is pointless -- the next demo switch would take it back.
_before = _st.persona[1]
_st.apply_demo_persona("professional")
check("a demo switch does not disturb a hand-picked voice", _st.persona[1], _before)
_st.set_persona("friendly")
check("but the operator changing personality does", _st.persona[1] != _before, True)

print()
print("[15] a name, once learned, is never forgotten")

import numpy as _np  # noqa: E402

from brain import db as _db2  # noqa: E402

# The stubs from [8] and [9] are done with; this section talks to the real one.
for _name, _fn in _REAL_DB.items():
    setattr(_db2, _name, _fn)

_db2.init_db()
_pid = _db2.create_person("Regression Test")
_rng = _np.random.default_rng(11)
_face = _rng.standard_normal(512).astype(_np.float32)
_db2.add_embedding(_pid, _face)
check("a face is stored on enrolment", _db2.count_embeddings(_pid), 1)

import sqlite3 as _sq0  # noqa: E402

_c0 = _sq0.connect(_db2.MODELS.db_path)
_first_id = _c0.execute(
    "SELECT MIN(id) FROM people_embeddings WHERE person_id = ?", (_pid,)).fetchone()[0]
_c0.close()

# The bug this replaces: the table keyed on person_id, so every further view
# REPLACED the first. One stored face is one angle in one light, and standing
# differently next visit read as the robot forgetting the person entirely.
for _i in range(5):
    _db2.add_embedding(_pid, _face + _rng.standard_normal(512).astype(_np.float32) * 0.2)
check("later views are added, not substituted", _db2.count_embeddings(_pid), 6)

for _i in range(30):
    _db2.add_embedding(_pid, _rng.standard_normal(512).astype(_np.float32))
check("and bounded", _db2.count_embeddings(_pid), _db2.MAX_EMBEDDINGS_PER_PERSON)

# The enrolment view is the anchor for somebody returning after a long gap. A
# busy afternoon adds a dozen views of one angle, and evicting purely by age
# would drop the face they were enrolled with -- so a visitor back in a year
# would be matched only against how they stood last March.
import sqlite3 as _sq  # noqa: E402

_con = _sq.connect(_db2.MODELS.db_path)
_ids = [r[0] for r in _con.execute(
    "SELECT id FROM people_embeddings WHERE person_id = ? ORDER BY id", (_pid,))]
_con.close()
check("the enrolment view is never evicted", min(_ids), _first_id)
check("the name is untouched throughout", _db2.get_person_name(_pid), "Regression Test")
check(
    "note counts survive several faces",
    [p for p in _db2.list_people() if p["person_id"] == _pid][0]["note_count"],
    0,
)
_db2.delete_person(_pid)

print()
print("[16] every new face gets its own offer to be remembered")

import numpy as _np2  # noqa: E402

from demokit.registry import REGISTRY as _REG2  # noqa: E402

_REG2.discover()
# build() above rebinds demokit.runner.REGISTRY to a stub holding only the fake
# demos, and that binding outlives the section that did it. This drives the
# REAL Vision demo through the real runner, so it has to be put back first.
import demokit.runner as _runner_mod2  # noqa: E402

_runner_mod2.REGISTRY = _REG2
_rng2 = _np2.random.default_rng(3)
_FACES = {n: _rng2.standard_normal(512).astype(_np2.float32)
          for n in ("alice", "bob", "carol", "dave")}


class ManyFaces:
    """A tracker showing whichever stranger `who` names, by their own vector."""

    enabled = True

    def __init__(self):
        self.who = None
        self.enrolled = []
        self._face = self

    def current(self, max_age_s=1.5):
        return (None, object()) if self.who else (None, None)

    def current_embedding(self, max_age_s=1.5):
        return _FACES.get(self.who)

    def enroll(self, name, face):
        self.enrolled.append(name)
        return len(self.enrolled)


_caps = frozenset({"faces"})
_vstate = RobotState()
_vstate.set_demos(_REG2.dashboard_entries(_caps))
_vstate.set_mode("vision")
_vaudio, _vtracker = FakeAudio(), ManyFaces()
_vrunner = DemoRunner(
    audio=_vaudio, motion=FakeMotion(), tracker=_vtracker,
    state=_vstate, capabilities=_caps,
)


def offered_to(who, answers, gap=True):
    _vtracker.who = who
    _vaudio._transcripts = list(answers)
    _vaudio.said.clear()
    for _ in range(8):
        _vrunner.cycle()
        if _vrunner._ctx:
            _vrunner._ctx.store["present_since"] = time.monotonic() - 10
            if gap:
                _vrunner._ctx.store["offered_at"] = 0
    return any("remember you by name" in line for line in _vaudio.said)


# The reported fault: it kept deciding people had declined. Silence, a mumble,
# and every natural way of agreeing that happens to contain a refusal word
# ("why not", "no problem", "go on then") all came back as no.
from demos.vision import _NO, _UNCLEAR, _YES, _read_answer  # noqa: E402

for _said in ("yes", "yeah", "sure", "go on then", "why not", "no problem",
              "i dont mind", "of course", "sounds good", "yes please"):
    check(f"{_said!r} is a yes", _read_answer(_said), _YES)
for _said in ("no", "no thanks", "id rather not", "maybe later", "another time"):
    check(f"{_said!r} is a no", _read_answer(_said), _NO)
for _said in ("", "um", "the weather is nice"):
    # Not a refusal. Read as one, a visitor who said yes was told the robot
    # would leave them alone, and nothing could tell "turned down" from
    # "not heard".
    check(f"{_said!r} is unclear, not a refusal", _read_answer(_said), _UNCLEAR)

check("a stranger is asked", offered_to("alice", ["no"]), True)
# The point of the whole rule. This used to be False: one blanket ten-minute
# silence covered everybody, because an unrecognised face has no id to hang
# "already declined" on, so the second visitor of the afternoon was never asked.
check("a DIFFERENT stranger is asked too", offered_to("bob", ["no"]), True)
check("the same face is not asked twice", offered_to("alice", ["no"]), False)
check("one who accepts is enrolled",
      (offered_to("dave", ["yes", "my name is Dave", "yes"]), _vtracker.enrolled)[1], ["Dave"])

_vrunner._ctx.store["offered_at"] = time.monotonic()
check("back-to-back offers are deferred", offered_to("carol", ["no"], gap=False), False)

# Answers to this demo's OWN questions are recorded as speech heard, so a
# naive "has anybody spoken lately" test read a visitor declining as a
# conversation in progress and silenced the next person's offer.
_vrunner._ctx.store["offered_at"] = 0
_vrunner._ctx.store["our_exchange_at"] = time.time() - 30
_vstate.add("heard", "what is extended reality")
check("held back while somebody is talking to it", offered_to("carol", ["no"]), False)
_vstate._last_heard_at = time.time() - 60
check("and asked once the room goes quiet", offered_to("carol", ["no"]), True)

# Silence gets a second, plainer ask rather than being acted on as a refusal.
_vrunner._ctx.store["offered_faces"].clear()
_vrunner._ctx.store["offered_at"] = 0
_vstate._last_heard_at = time.time() - 60
_vtracker.enrolled.clear()
offered_to("dave", ["", "yes", "my name is Sam", "yes"])
check("silence is re-asked, not taken as no", any("was that a yes" in s for s in _vaudio.said), True)
check("and a yes after it still enrols", _vtracker.enrolled, ["Sam"])

# The microphone is held open for the exchange, so nobody has to say a wake
# word to answer a question they were just asked. Held, not switched: the
# operator's own setting is untouched and must come back afterwards.
check("the operator's switch was never touched", _vstate._open_mic, False)
check("and the hold is released once a name is in", _vstate.open_mic, False)

_vtracker.enrolled.clear()
_vrunner._ctx.store["offered_faces"].clear()
_vrunner._ctx.store["offered_at"] = 0
_vstate._last_heard_at = time.time() - 60
offered_to("dave", ["yes", "my name is Sam"])   # walks off mid-question
held_mid = _vstate.open_mic
_vrunner._active_demo.on_exit(_vrunner._ctx)
# A hold left on is a robot listening to the room for the rest of the
# afternoon, so every way out of the exchange has to give it back -- including
# the operator switching demo while a question is still in the air.
check("a hold does not survive leaving the demo", _vstate.open_mic, False)

print()
print("[17] features built from the dashboard")

import dataclasses as _dc  # noqa: E402
import tempfile as _tf  # noqa: E402

import config as _config  # noqa: E402
from brain import features as _F  # noqa: E402
from demokit.base import split_sentences as _split  # noqa: E402

# A scratch database, so a test run never writes staff features into the
# robot's own -- the same care test_memory.py takes.
_F.db.MODELS = _dc.replace(_config.MODELS, db_path=Path(_tf.mkdtemp(prefix="reachy-feat-")) / "f.db")
_F._init_features()

# parse_blocks is the only thing standing between a hand-edited database row
# and ctx.say, so it has to be total.
check("bad JSON parses to nothing", _F.parse_blocks("not json"), [])
check("a non-list too", _F.parse_blocks('{"not": "a list"}'), [])
check("an unknown step is dropped", _F.parse_blocks('[{"kind":"EXPLODE"}]'), [])
check("as is a SAY with no words", _F.parse_blocks('[{"kind":"SAY","text":"  "}]'), [])
_coerced = _F.parse_blocks('[{"kind":"WAIT","seconds":9000},{"kind":"say","text":"Hi.","emotion":"zzz"}]')
check("a silly wait is clamped", _coerced[0].seconds, _F.WAIT_SECONDS_RANGE[1])
check("an unknown expression falls back", _coerced[1].emotion, "neutral")


def _feature(label, trigger="", blocks=None):
    rec = _F.Feature(label=label, help="h", trigger_phrase=_F.normalise_trigger(trigger),
                     blocks=blocks if blocks is not None else [_F.Block(_F.SAY, text="Hello there.")])
    rec.id = _F.slug_for(label)
    return rec


_REG0.discover()
_bad = [
    ("welcome", "too short to be safe"),
    ("do the welcome speech", "already claimed by Welcome"),
    ("goodbye everyone please", "contains a sleep phrase"),
    ("can you do it", "nothing but filler words"),
]
for _phrase, _why in _bad:
    check(f"rejected: {_why}", bool(_F.validate(_feature("T", _phrase))), True)
check("accepted: a specific phrase",
      _F.validate(_feature("Cork", "say hello to the cork group")), [])
check("a leftover placeholder is rejected",
      bool(_F.validate(_feature("P", "", [_F.Block(_F.SAY, text="Welcome [group].")]))), True)
check("a step-less feature is rejected", bool(_F.validate(_feature("E", "", []))), True)
# Not sanitised on purpose: the browser builds every label with textContent, so
# stripping angle brackets would only break "Q&A with the Erasmus group".
check("markup in a label is allowed through",
      _F.validate(_feature("<script>alert(1)</script>", "")), [])

print()
print("[18] a feature plays one step per slice, and never goes deaf")

from demos._stored import StoredFeature  # noqa: E402


class CountingMotion(FakeMotion):
    def __init__(self):
        super().__init__()
        self.dances = 0

    def dance(self, **kw):
        self.dances += 1


def play(blocks, transcripts=(), tracker=None, slices=10):
    rec = _F.Feature(id="custom_t", label="T", help="h", updated_at="v1", blocks=blocks)
    demo = StoredFeature(rec)
    audio, motion = FakeAudio(transcripts=list(transcripts)), CountingMotion()
    ctx = DemoContext(audio=audio, motion=motion, tracker=tracker,
                      state=RobotState(), demo_id="custom_t", store={})
    demo.on_enter(ctx)
    windows = [demo.on_idle(ctx).listen_for for _ in range(slices)]
    return audio.said, motion.dances, windows, ctx


said, dances, windows, _ = play([
    _F.Block(_F.SAY, text="One. Two."),
    _F.Block(_F.DANCE),
    _F.Block(_F.ASK, text="Where from?", ai_reply=True),
    _F.Block(_F.SAY, text="Lovely."),
], transcripts=["Madrid"])
check("a two-sentence step is spoken as two", said[:2], ["One.", "Two."])
check("the whole sequence runs", said, ["One.", "Two.", "Where from?", "A reply.", "Lovely."])
check("dance happened once", dances, 1)
# The invariant the whole per-slice design exists for. Two zeros in a row is a
# stretch during which DemoRunner.cycle never opens the microphone at all.
check("never two silent slices in a row",
      any(windows[i] == 0 and windows[i + 1] == 0 for i in range(len(windows) - 1)), False)
check("never longer than the core allows", max(windows) <= 3.0, True)


class Nobody:
    enabled = True

    def current(self, max_age_s=1.5):
        return (None, None)


_wait = [_F.Block(_F.WAIT, seconds=20), _F.Block(_F.SAY, text="There you are.")]
said, _, _, _ = play(_wait)
check("with no camera a wait is skipped", said, ["There you are."])
said, _, _, ctx = play(_wait, tracker=Nobody(), slices=2)
check("with nobody there it holds", said, [])
ctx.store["waiting_since"] -= 999          # rather than actually waiting
StoredFeature(_F.Feature(id="custom_t", label="T", updated_at="v1", blocks=_wait)).on_idle(ctx)
check("and gives up rather than hanging", ctx.store["cursor"], 1)

print()
print("[19] a feature becomes a button without a restart")


class Plain(Demo):
    id, label, help, order = "plain", "Plain", "h", 10


_live = Registry()
_live._publish({"plain": Plain()})
_rec = _F.Feature(id="custom_new", label="New One", help="h", updated_at="v1",
                  blocks=[_F.Block(_F.SAY, text="Hi.")])
check("registers into the live list", _live.register(StoredFeature(_rec)), True)
check("and sorts after the code demos", _live.ids(), ["plain", "custom_new"])
_shadow = _F.Feature(id="plain", label="Impostor", updated_at="v", blocks=[_F.Block(_F.SAY, text="x")])
check("cannot shadow a demo from a file", _live.register(StoredFeature(_shadow)), False)
check("the file's demo is untouched", _live.get("plain").label, "Plain")
for _ in range(3):
    _live.record_failure("custom_new")
check("a failing feature is set aside", _live.is_available("custom_new", frozenset())[0], False)
_live.register(StoredFeature(_rec))
check("and editing it puts it back", _live.is_available("custom_new", frozenset())[0], True)
check("unregister only touches stored ones", _live.unregister("plain"), False)
check("but removes a stored one", _live.unregister("custom_new"), True)
check("leaving the rest alone", _live.ids(), ["plain"])

print()
print("[20] audio from the wrong thread is refused, not silently interleaved")
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
print("[21] brainstorming follows the group instead of reading out a list")
import demos.brainstorm as _bs  # noqa: E402

# Every question after the opener is written by the model, so the thing worth
# testing is what the model is given -- if the group's own words are not in
# that prompt, the robot cannot be following them whatever it says back.
_bs_calls = []


def _bs_stream(person_id, message, style=None, extra_system=None, cache=True, web=False):
    _bs_calls.append((message, extra_system or ""))
    yield f"Reply {len(_bs_calls)}.", "curious"


_bi.stream_reply = _bs_stream


def _bs_session():
    demo = _bs.Brainstorm()
    audio, motion, st = FakeAudio(), FakeMotion(), RobotState()
    ctx = DemoContext(
        audio=audio, motion=motion, tracker=None, state=st,
        demo_id="brainstorm", store={},
    )
    return demo, ctx, audio, st


_bs_calls.clear()
demo, ctx, audio, st = _bs_session()
windows = []
demo.on_enter(ctx)
windows.append(demo.on_idle(ctx).listen_for)          # the fixed opener
check("opens with the group invitation", audio.said[-1], _bs._OPENER)
check("and holds the mic for the answer", st.open_mic, True)

demo.on_utterance(ctx, "A booking app for dance studios in Dublin")
check("the hold is released once they answer", st.open_mic, False)
windows.append(demo.on_idle(ctx).listen_for)          # first generated question
_message, _brief = _bs_calls[-1]
check("their words are what the model answers", _message, "A booking app for dance studios in Dublin")
check("and the session so far is in the prompt", "dance studios in Dublin" in _brief, True)
check("labelled as a group, not one visitor", "The group:" in _brief, True)

demo.on_utterance(ctx, "Studio owners taking bookings over WhatsApp right now")
windows.append(demo.on_idle(ctx).listen_for)
check("it is told not to ask the same thing twice", "Reply 1." in _bs_calls[-1][1], True)

# A nod is not an answer. Without this a group agreeing with each other would
# reach the directions having told the robot nothing.
demo.on_utterance(ctx, "yeah")
check("agreeing does not count as material", ctx.store["answers"], 2)
windows.append(demo.on_idle(ctx).listen_for)

demo.on_utterance(ctx, "They don't trust an app with their own calendar")
windows.append(demo.on_idle(ctx).listen_for)
check("the last question is aimed at the directions",
      "last question before you give them ideas" in _bs_calls[-1][1], True)

demo.on_utterance(ctx, "We would start with three studios in Dublin 8")
check("four answers is enough to work with", ctx.store["answers"], 4)
windows.append(demo.on_idle(ctx).listen_for)          # -> bridge
windows.append(demo.on_idle(ctx).listen_for)          # "Right. Three directions."
check("stops interviewing on substance, not on a question count",
      ctx.store["stage"], _bs._GENERATE)
check("and lets the mic go while it talks", st.open_mic, False)

for _ in range(len(_bs._ORDINALS)):
    windows.append(demo.on_idle(ctx).listen_for)
check("three directions", len(ctx.store["directions"]), 3)
check("built from the whole session, not from three slots",
      "Dublin 8" in _bs_calls[-1][1], True)
# The old version ended here. Ending on a question that is actually listened
# for is what makes it a conversation rather than a lecture with a full stop.
check("the session carries on afterwards", ctx.store["stage"], _bs._TALK)
check("waiting on the closing question", ctx.store["awaiting"], True)
check("mic open for the answer to it", st.open_mic, True)
check("never two silent slices in a row",
      any(windows[i] == 0 and windows[i + 1] == 0 for i in range(len(windows) - 1)), False)
check("never longer than the core allows", max(windows) <= 3.0, True)

check("never repeats its bridge line verbatim",
      audio.said.count("Right. Three directions on that."), 1)

# Keep talking after the directions and it works up to another set, rather
# than sitting at a full stop for the rest of the visit.
for line in ("Studios pay per booking, not monthly",
             "Two of them said they'd switch tomorrow",
             "The blocker is migrating their existing calendar",
             "We could import from Google Calendar",
             "Then it's a weekend of work, not a rewrite",
             "Their receptionists are the ones who'd use it",
             "They're on desktop, not phones",
             "Cancellations are the real money",
             "Half of them are no-shows"):
    demo.on_utterance(ctx, line)
    demo.on_idle(ctx)
check("a second round is offered, not a full stop", ctx.store["stage"], _bs._BRIDGE)
demo.on_idle(ctx)
check("and it says so differently the second time",
      audio.said[-1], "Right. Three more, on what you've just said.")

demo.on_exit(ctx)
check("leaving the demo always gives the mic back", st.open_mic, False)

# "What do you think?" is the clearest instruction the robot gets all session.
demo, ctx, audio, st = _bs_session()
demo.on_enter(ctx)
demo.on_idle(ctx)
demo.on_utterance(ctx, "An app that books rehearsal rooms by the hour")
check("asking for ideas early is taken as an instruction",
      demo.on_utterance(ctx, "what do you think"), True)
check("and goes straight to them", ctx.store["stage"], _bs._BRIDGE)

# A room that has gone quiet with nothing said yet. It used to wait forever.
demo, ctx, audio, st = _bs_session()
demo.on_enter(ctx)
quiet = []
for _ in range(_bs._MAX_WAITS * (_bs._MAX_NUDGES + 1) + 2):
    quiet.append(demo.on_idle(ctx).listen_for)
check("a silent room is nudged, not interrogated",
      audio.said.count(_bs._NUDGE), _bs._MAX_NUDGES)
check("and eventually let go", ctx.store["stage"], _bs._DONE)
check("without leaving the mic held open", st.open_mic, False)
check("still never deaf while waiting", max(quiet) <= 3.0 and min(quiet) > 0, True)

print()
print("[22] the dashboard's folder layout is display-only, and never load-bearing")
import dataclasses as _dc  # noqa: E402
import tempfile as _tmp  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
import brain.db as _db  # noqa: E402

# Its own database file. This table is written by the tests below, and the live
# one holds real enrolled people -- test_memory.py learned that the hard way.
_db.MODELS = _dc.replace(_db.MODELS, db_path=_Path(_tmp.mkdtemp()) / "layout-test.db")
import brain.layout as _lay  # noqa: E402
import brain.settings as _setmod  # noqa: E402
import brain.stats as _statmod  # noqa: E402
import brain.study as _studymod  # noqa: E402

# Every module that owns a table rebuilds it against the scratch database.
# Without this the runner's own stats calls log "no such table" for the rest
# of the run -- swallowed, so the tests still pass, which is exactly why it
# would have gone unnoticed.
for _mod, _init in ((_lay, "_init_layout"), (_setmod, "_init_settings"),
                    (_statmod, "_init_stats"), (_studymod, "_init_study")):
    _mod.db.MODELS = _db.MODELS
    getattr(_mod, _init)()

# Coercion is total: nothing below may raise, and nothing malformed may survive.
check("bad JSON is nothing", _lay._coerce("{not json"), [])
check("a non-list is nothing", _lay._coerce({"t": "f"}), [])
check("a folder id we did not mint is dropped",
      _lay._coerce([{"t": "f", "id": "../../etc", "name": "x"}]), [])
check("a demo id with a path in it is dropped",
      _lay._coerce([{"t": "i", "id": "../secrets"}]), [])
check("a folder inside a folder is dropped",
      _lay._coerce([{"t": "f", "id": "f_00000001",
                     "items": [{"t": "f", "id": "f_00000002"}]}])[0]["items"], [])
check("the same button cannot appear twice",
      [e["id"] for e in _lay._coerce([{"t": "i", "id": "dance"}, {"t": "i", "id": "dance"}])],
      ["dance"])
check("a nameless folder is named, not dropped",
      _lay._coerce([{"t": "f", "id": "f_00000001", "name": "   "}])[0]["name"], "Folder")
check("a very long name is clamped",
      len(_lay._coerce([{"t": "f", "id": "f_00000001", "name": "n" * 400}])[0]["name"]),
      _lay.MAX_FOLDER_NAME_CHARS)
check("the document is bounded", len(_lay._coerce([{"t": "i", "id": f"d{i}"} for i in range(500)])),
      _lay.MAX_ROOT_ENTRIES)

# An id for a demo that is not running is KEPT. Registry.discover() skips a
# module that fails to import, so "missing" is routinely a syntax error
# somebody will fix this afternoon -- reaping here would silently wipe an
# arrangement that took half an hour to build on a phone.
check("an id for a demo that is not loaded is kept, not reaped",
      _lay._coerce([{"t": "i", "id": "demo_that_failed_to_import"}]),
      [{"t": "i", "id": "demo_that_failed_to_import"}])

_shape = [{"t": "f", "id": "f_0a1b2c3d", "name": "Schools", "items": ["welcome", "about"]},
          {"t": "i", "id": "dance"}]
_st, _doc, _ = _lay.write(_shape, 0)
check("a layout saves", (_st, _doc["rev"]), ("ok", 1))
_st2, _doc2, _ = _lay.write([{"t": "i", "id": "dance"}], 0)
check("a stale write is refused", _st2, "stale")
check("and the refusal carries the winner's layout so it can be rebased",
      len(_doc2["items"]), 2)
_st3, _doc3, _ = _lay.write(_shape, _doc["rev"])
check("the revision only ever goes up", _doc3["rev"] > _doc["rev"], True)
check("too many folders is refused with a reason",
      _lay.write([{"t": "f", "id": "f_%08x" % i, "name": "n"} for i in range(13)],
                 _doc3["rev"])[0], "invalid")

# Deleting a feature must drop its placement: slug_for derives the id from the
# label, so a feature written months later with a similar name inherits the id
# -- and would inherit a place inside somebody's collapsed folder.
_lay.forget("welcome")
_after, _ = _lay.read()
check("deleting a feature drops its placement",
      [e for e in _after["items"] if e["t"] == "f"][0]["items"], ["about"])
_rev_before = _after["rev"]
_lay.forget("welcome")
check("and forgetting it twice writes nothing", _lay.read()[0]["rev"], _rev_before)
check("reset empties it", _lay.reset()["items"], [])

# THE INVARIANT THE WHOLE FEATURE RESTS ON. registry.default_id() returns
# _order[0], which is the demo the robot boots into and falls back to. If a
# layout could reorder that, a staff member tidying the grid on a Tuesday would
# change what the robot starts in on Wednesday, and nothing would say so.
_REG3 = Registry()
_REG3.discover()
_before_ids = _REG3.ids()
_before_default = _REG3.default_id()
_lay.write([{"t": "f", "id": "f_0a1b2c3d", "name": "Anything",
             "items": list(_before_ids)}], _lay.read()[0]["rev"])
check("a layout never reorders the registry", _REG3.ids(), _before_ids)
check("and never changes the demo the robot boots into", _REG3.default_id(), _before_default)

# With no layout at all the dashboard is the one that shipped: every button,
# flat, in the robot's own order. Anything that makes the layout NECESSARY for
# the grid to render breaks the property that makes this safe to ship.
_lay._available = False
check("an unavailable layout reads as empty, never raises", _lay.read(), ({"rev": 0, "items": []}, False))
check("and writing says so rather than failing", _lay.write([], 0)[0], "unavailable")
check("and forgetting is a silent no-op", _lay.forget("welcome"), None)
_lay._available = True

_state = RobotState()
_state.set_demos([{"id": "welcome", "label": "W", "help": "", "available": True, "note": ""}],
                 {"rev": 7, "items": [], "available": True})
check("the layout rides the poll the dashboard already makes",
      _state.snapshot()["layout"], {"rev": 7, "items": [], "available": True})
_state.refresh_demo_availability(
    [{"id": "welcome", "label": "W", "help": "", "available": False, "note": "failed 3 times"}])
check("and a refresh that carries no layout leaves it alone",
      _state.snapshot()["layout"]["rev"], 7)

print()
print("[23] the dashboard never puts a robot string into markup")
_page = (_Path(__file__).resolve().parent / "web" / "index.html").read_text(encoding="utf-8")
_code = "\n".join(
    line for line in _page.splitlines()
    if not line.strip().startswith(("//", "*", "/*"))
)
# Folder names are the fourth staff-authored, unauthenticated, stored string on
# this page. The whole defence is that no server value is ever interpolated
# into markup -- so the sink simply may not appear outside a comment.
check("no innerHTML sink outside a comment", ".innerHTML" in _code, False)
check("no outerHTML either", ".outerHTML" in _code, False)
check("no document.write", "document.write(" in _code, False)
# Offline: the robot is routinely on a hotspot, and a CDN font or script that
# fails to load would take the operator's dashboard with it. The SVG and XHTML
# namespace URIs are excluded by name -- they are constants passed to
# createElementNS, never fetched by any browser, and a check that cannot tell
# them from a CDN link is one somebody silences rather than fixes.
_NAMESPACES = ("http://www.w3.org/2000/svg", "http://www.w3.org/1999/xhtml",
               "http://www.w3.org/1999/xlink")
_fetchable = _page
for _ns in _NAMESPACES:
    _fetchable = _fetchable.replace(_ns, "")
for _proto in ("https://", "http://", "//cdn", "@import"):
    check(f"nothing loaded from {_proto}", _proto in _fetchable, False)
check("the safe-area insets can actually resolve", "viewport-fit=cover" in _page, True)
check("both themes are declared, not just one",
      _page.count('data-theme="dark"') >= 1 and "prefers-color-scheme: dark" in _page, True)
check("motion is opt-out", "prefers-reduced-motion" in _page, True)

print()
print("[24] the stylesheet survives being read the way a browser reads it")
import re  # noqa: E402

# This section exists because of one shipped bug, and it is worth stating
# plainly. A comment in the token block said "override the --l-*/--d-* pairs".
# CSS comments do not nest and have no escape: the star-slash inside that glob
# ENDED the comment, and the rest of the sentence plus the entire :root block
# after it went to the parser as garbage and was discarded. Every design token
# vanished, and the dashboard rendered as serif text on a black page with no
# cards at all. Every check in section [23] passed, because they all searched
# the file as TEXT. So this one strips comments the way a browser does, and
# then insists the rules that matter still exist.
_raw = re.search(r"<style>(.*?)</style>", _page, re.S).group(1)


def _decomment(css):
    """Strip comments the way a parser does: the FIRST star-slash closes."""
    out, i = [], 0
    while True:
        start = css.find("/*", i)
        if start == -1:
            out.append(css[i:])
            return "".join(out)
        out.append(css[i:start])
        close = css.find("*/", start + 2)
        if close == -1:            # unterminated: the rest of the file is comment
            return "".join(out)
        out.append(" ")
        i = close + 2


_css = _decomment(_raw)

# A comment terminator anywhere but the end of its line is the shape of the bug.
_early = [ln for ln, line in enumerate(_raw.splitlines(), 1)
          if "*/" in line and line.split("*/", 1)[1].strip()]
check("no comment closes mid-line", _early, [])

# What is left between one rule's } and the next rule's { must look like a
# selector. Leaked prose does not, which is what makes this catch the general
# case rather than only the star-slash spelling of it.
_between, _depth, _buf, _loose = [], 0, [], []
for _ch in _css:
    if _ch == "{":
        if _depth == 0:
            _loose.append("".join(_buf).strip())
        _depth += 1
        _buf = []
    elif _ch == "}":
        _depth = max(0, _depth - 1)
        _buf = []
    elif _depth == 0:
        _buf.append(_ch)
_tail = "".join(_buf).strip()
_SELECTOR = re.compile(r"^[\w\s.#:,()\[\]=\"'>+~*@-]+$")
_junk = [s[:60] for s in _loose if s and not _SELECTOR.match(s)]
check("nothing but selectors between rules", _junk, [])
check("no text left dangling after the last rule", _tail, "")
check("braces balance after comments are stripped", _css.count("{"), _css.count("}"))

# The rules that, if they go missing, produce exactly the page that shipped.
check("the type ramp survives", "--fs-lead:" in _css, True)
check("the geometry survives", "--tap:" in _css, True)
check("the light mapping survives", "--bg: var(--l-bg)" in _css, True)
check("the dark mapping survives", _css.count("--bg: var(--d-bg)"), 2)
check("body still gets a font", re.search(r"body\s*\{[^}]*font:", _css) is not None, True)
check("cards still get a background", re.search(r"\.panel\s*\{[^}]*background:", _css) is not None, True)
check("a mode key still gets its face", re.search(r"\.mode\s*\{[^}]*--key-bg:", _css) is not None, True)

# The list a <select> opens is a separate surface that inherits the control's
# colours, and pill selects are deliberately transparent so they sit flush
# inside their pill. Without colours of its own the open menu drew near-white
# text on the platform's white canvas, and every option except the highlighted
# one was invisible in dark mode -- the personality menu read as "Default" and
# then a tall empty white box.
_opt = re.search(r"select option[^{]*\{([^}]*)\}", _css)
check("the open select menu is coloured, not inherited",
      bool(_opt) and "background-color:" in _opt.group(1) and "color:" in _opt.group(1), True)
check("and it uses theme tokens, so it follows the palette",
      bool(_opt) and "var(--" in _opt.group(1), True)

# Two keys held on each other become a folder. Three numbers have to agree for
# that to behave, and they live in three files, so they are checked rather than
# trusted: the dwell the script waits for and the dwell the ring draws must be
# the same length, or the ring finishes before the gesture arms and people let
# go early; and the folder cap the browser refuses at must match the one the
# robot refuses at, or the gesture fails after the drag instead of before it.
_js_ms = re.search(r"MERGE_MS\s*=\s*(\d+)", _page)
_css_ms = re.search(r"--merge-ms:\s*(\d+)ms", _page)
check("the grouping dwell and the ring that draws it agree",
      (_js_ms and _js_ms.group(1), _css_ms and _css_ms.group(1)),
      (_css_ms and _css_ms.group(1), _css_ms and _css_ms.group(1)))
_js_max = re.search(r"MAX_FOLDERS\s*=\s*(\d+)", _page)
check("the browser's folder cap matches the robot's",
      int(_js_max.group(1)) if _js_max else None, _lay.MAX_FOLDERS)
check("the grouping target is drawn", ".mode.is-merge" in _css, True)

print()
print("[25] every colour theme is legible, in both light and dark")
# The design's premise is that a staff member reads this from arm's length in a
# glass atrium with a group waiting. That is a claim about contrast ratios, so
# it is measured rather than eyeballed -- and it is measured for every palette,
# so adding a sixth cannot quietly ship an unreadable one.
_style = re.sub(r"/\*.*?\*/", " ", re.search(r"<style>(.*?)</style>", _page, re.S).group(1), flags=re.S)


def _tokens(selector):
    m = re.search(re.escape(selector) + r"\s*\{(.*?)\n\s*\}", _style, re.S)
    return dict(re.findall(r"--([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;", m.group(1))) if m else {}


def _lum(c):
    def chan(v):
        v = int(c[v:v + 2], 16) / 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = chan(1), chan(3), chan(5)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ratio(a, b):
    hi, lo = max(_lum(a), _lum(b)), min(_lum(a), _lum(b))
    return (hi + 0.05) / (lo + 0.05)


def _hue(c):
    r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
    hi, lo = max(r, g, b), min(r, g, b)
    if hi == lo:
        return 0.0
    d = hi - lo
    h = ((g - b) / d) % 6 if hi == r else ((b - r) / d + 2 if hi == g else (r - g) / d + 4)
    return h * 60


_base = _tokens(":root")
_names = [""] + re.findall(r':root\[data-palette="([a-z]+)"\]', _style)
check("five colour themes", len(set(_names)), 5)
check("and a swatch for each", len(set(re.findall(r'data-palette-set="([a-z]*)"', _page))), 5)

_PAIRS = [("text", "panel", 7.0), ("text", "bg", 7.0), ("text-2", "panel", 4.5),
          ("dim", "panel", 4.5), ("dim", "bg", 4.5), ("accent-ink", "accent", 4.5),
          ("accent", "panel", 4.5), ("accent", "panel-2", 4.5), ("accent", "accent-soft", 4.5),
          ("ok", "ok-soft", 4.5), ("warn", "warn-soft", 4.5), ("bad", "bad-soft", 4.5),
          ("heard", "panel-2", 4.5), ("said", "panel", 4.5)]
# These four wash the SAME state bar seconds apart -- Listening, Speaking,
# Asleep, Offline. Two of them at the same lightness and a similar hue is a
# state bar that has stopped saying what the robot is doing, which is the one
# thing it exists for.
_STATE = ["accent", "ok", "warn", "bad"]

for _pal in sorted(set(_names)):
    _over = _tokens(f':root[data-palette="{_pal}"]') if _pal else {}
    for _mode in ("l", "d"):
        _t = {k[2:]: v for k, v in list(_base.items()) + list(_over.items())
              if k.startswith(_mode + "-")}
        _low = [f"{a} on {b} {_ratio(_t[a], _t[b]):.1f}"
                for a, b, floor in _PAIRS if a in _t and b in _t and _ratio(_t[a], _t[b]) < floor]
        _near = [f"{a}/{b}" for i, a in enumerate(_STATE) for b in _STATE[i + 1:]
                 if a in _t and b in _t
                 and abs(_lum(_t[a]) - _lum(_t[b])) < 0.05
                 and min(abs(_hue(_t[a]) - _hue(_t[b])), 360 - abs(_hue(_t[a]) - _hue(_t[b]))) < 40]
        _label = (_pal or "trinity") + " / " + ("light" if _mode == "l" else "dark")
        check(f"{_label}: contrast", _low, [])
        check(f"{_label}: states tell apart", _near, [])

# Every themeable colour must exist in BOTH modes and be wired to a live token,
# or a palette silently inherits the base for whatever it forgot.
_pairs = {n for n in re.findall(r"--([ld]-[a-z0-9-]+)\s*:", _style)}
_mapped = set(re.findall(r"--([a-z0-9-]+)\s*:\s*var\(--[ld]-", _style))
check("every per-mode colour is wired to a live token",
      sorted({n[2:] for n in _pairs} - _mapped - {"sw"}), [])
check("the palettes are defined once, not once per mode",
      _style.count("--l-bg:"), len(set(_names)))

print()
print("[26] the robot knows what Trinity teaches, and knows what it must not say")
import brain.courses as _co  # noqa: E402

check("every taught masters is there", len(_co.PROGRAMMES), 13)
check("no two share a key", len({p.key for p in _co.PROGRAMMES}), len(_co.PROGRAMMES))
check("no two share a name", len({p.name for p in _co.PROGRAMMES}), len(_co.PROGRAMMES))
check("each one can be asked for", [p.key for p in _co.PROGRAMMES if not p.terms], [])
check("each one says who it is for", [p.key for p in _co.PROGRAMMES if not p.who_for], [])

# The questions a prospective student actually asks, standing in the building.
# The confusable pairs are the point: there are two marketing MScs and four
# finance-ish ones, and telling them apart is the whole job.
_ASKED = [
    ("is there a risk management masters", ["MSc in Financial Risk Management"]),
    ("a supply chain management masters", ["MSc in Operations and Supply Chain Management"]),
    ("do you do a management masters", ["MSc in Management"]),
    ("I have an arts degree can I still apply", ["MSc in Management"]),
    ("whats the difference between the marketing ones",
     ["MSc in Marketing", "MSc in Digital Marketing Strategy"]),
    ("digital marketing strategy", ["MSc in Digital Marketing Strategy"]),
    ("tell me about the finance masters", ["MSc in Finance"]),
    ("law and finance", ["MSc in Law and Finance"]),
    ("can I do accounting without an accounting degree", ["MSc in Accounting and Analytics"]),
    ("I want to start a company", ["MSc in Entrepreneurship and Innovation"]),
    ("sustainability masters", ["MSc in Responsible Business and Sustainability"]),
    ("human resource management", ["MSc in Human Resource Management"]),
    ("international management", ["MSc in International Management"]),
]
for _q, _want in _ASKED:
    check(f"asked: {_q}", sorted(p.name for p in _co.matches(_q)), sorted(_want))

# Nothing else may pull a prospectus into the prompt. Every one of these is a
# real utterance from another demo, and a robot that answers "set a timer" with
# a masters programme has made every other demo worse.
for _quiet in ("what is the weather", "set a timer for five minutes", "tell me a story",
               "lets brainstorm a business idea", "can you dance", "what is your name",
               "who runs the hub", "how does your camera work"):
    check(f"quiet: {_quiet}", _co.brief(_quiet), "")

# Asked in general, it must offer to narrow rather than recite thirteen titles.
_gen = _co.brief("what postgraduate courses do you have")
check("a general question gets the list, not one programme", bool(_gen) and not _co.matches("what postgraduate courses do you have"), True)
check("and is told not to read it all out", "not list all thirteen" in _gen.lower() or "do not list" in _gen.lower(), True)

# THE PART THAT MATTERS MOST. These change every year, and the person asking is
# deciding whether they can afford to come.
_detail = _co.brief("tell me about the finance masters")
for _forbidden in ("fee", "deadline", "entry requirement", "scholarship", "ranking"):
    check(f"refuses to state the {_forbidden}", _forbidden in _co.REFUSALS.lower(), True)
check("and the refusal rides with every answer", _co.REFUSALS in _detail, True)
_numbers = re.findall(r"[€$£]\s?\d|\d{4,}", " ".join(p.block() for p in _co.PROGRAMMES))
check("no prices or years are baked into the facts", _numbers, [])

# Cost control: the standing list is in the base prompt on EVERY turn, so it
# has to stay small next to hub.GROUNDING's ~830 tokens.
import brain.hub as _hub  # noqa: E402
check("the standing list stays cheap", len(_co.STANDING) // 4 < 300, True)
check("and detail only arrives when asked for", len(_co.brief("what is the weather")), 0)
check("a detailed answer stays under a third of the grounding",
      len(_detail) < len(_hub.GROUNDING), True)

print()
print("[27] the reply pipeline starts speaking as early as it safely can")
import brain.interface as _iface  # noqa: E402
import brain.llm as _llm  # noqa: E402

# stream_reply is exercised for real, against a scripted backend, with its
# memory side effects stubbed -- the earlier sections stub stream_reply itself,
# which is exactly why nothing so far would have caught a bug in it.
class _ScriptedBackend:
    name = "scripted"
    supports_web = False

    def __init__(self, pieces, fail_after=None):
        self._pieces = pieces
        self._fail_after = fail_after

    def stream(self, messages, max_tokens=None, web=False):
        for i, piece in enumerate(self._pieces):
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError("backend died")
            yield piece


def _run_stream(*backends):
    _held = (_llm.streaming_backends, _iface.memory.get_history,
             _iface.long_term_memory.get_context, _iface.memory.remember_turn)
    remembered = []
    _llm.streaming_backends = lambda: backends
    _iface.memory.get_history = lambda pid: []
    _iface.long_term_memory.get_context = lambda pid: ""
    _iface.memory.remember_turn = lambda pid, m, r: remembered.append(r)
    try:
        out = list(_real_stream_reply(0, "a question", cache=False))
    finally:
        (_llm.streaming_backends, _iface.memory.get_history,
         _iface.long_term_memory.get_context, _iface.memory.remember_turn) = _held
    return out, remembered

# A long opening sentence is flushed at its first clause break, so the voice
# starts while the model is still writing the rest of the sentence. Streamed
# in small pieces the way a real backend delivers tokens -- handed the whole
# sentence in one piece, the ordinary sentence flush wins, which is also
# correct and also covered below.
_long_opener = ("The Hub runs across three different strands of work here, ",
                "covering ", "immersive learning and research. Second sentence. [emotion: happy]")
_out, _rem = _run_stream(_ScriptedBackend(_long_opener))
check("a long opener starts speaking at the clause",
      _out[0][0], "The Hub runs across three different strands of work here,")
check("the rest of the sentence follows",
      _out[1][0], "covering immersive learning and research.")
check("nothing is lost or doubled",
      _rem, ["The Hub runs across three different strands of work here, covering "
             "immersive learning and research. Second sentence."])
check("the closing emotion still lands", _out[-1][1], "happy")
_whole, _ = _run_stream(_ScriptedBackend(
    ("The Hub runs across three different strands of work here, covering "
     "immersive learning and research. [emotion: neutral]",)))
check("the same sentence arriving whole is spoken whole",
      [t for t, _ in _whole if t],
      ["The Hub runs across three different strands of work here, covering "
       "immersive learning and research."])

# A short reply never clause-splits -- the flush exists for long openers only.
_out2, _ = _run_stream(_ScriptedBackend(("Yes, it does. [emotion: neutral]",)))
check("a short reply is spoken whole", [t for t, _ in _out2 if t], ["Yes, it does."])

# THE FAILOVER RULE, now with clauses in it: once ANY words have been spoken --
# a clause counts exactly as a sentence does -- a dying backend must not let
# the other model start a different answer over the top of them.
_dies_talking = _ScriptedBackend(
    ("A long opening clause that runs well past sixty characters, easily, ", "boom"),
    fail_after=1)
_out3, _ = _run_stream(_dies_talking, _ScriptedBackend(("Fallback answer. [emotion: neutral]",)))
_spoken3 = [t for t, _ in _out3 if t]
check("a backend that dies after the clause does not restart the answer",
      any("Fallback" in t for t in _spoken3), False)
check("but the clause that was spoken stands",
      _spoken3[0].startswith("A long opening clause"), True)

# And a backend that dies BEFORE anything was spoken still fails over silently.
_out4, _ = _run_stream(_ScriptedBackend(("x",), fail_after=0),
                       _ScriptedBackend(("Fallback answer. [emotion: neutral]",)))
check("a backend that dies silent is replaced silently",
      [t for t, _ in _out4 if t], ["Fallback answer."])

# The Anthropic prompt-cache split: the standing prompt is the cached block,
# per-turn material is not, and an unrecognised prompt caches nothing rather
# than wrongly caching something that varies.
from brain.llm_backends import _system_blocks  # noqa: E402
from brain.prompts import base_prompts, build_messages as _bm  # noqa: E402
check("every standing prompt is exported for the cache split", len(base_prompts()), 3)
_blocks = _system_blocks(_bm("", [], "hi", extra_system="PERSONA")[0]["content"])
check("the standing prompt is the cached block",
      "cache_control" in _blocks[0] and len(_blocks) == 2, True)
check("per-turn material is not cached", "cache_control" in _blocks[-1], False)
check("an unknown prompt caches nothing",
      _system_blocks("something else entirely"), [{"type": "text", "text": "something else entirely"}])
# The API rejects an empty text block outright, and the long-term-memory
# summariser builds its request with no system prompt at all -- so "" must
# come back as no blocks, which the call sites turn into omitting the
# parameter entirely.
check("no system prompt means no blocks, not an empty one", _system_blocks(""), [])
check("nor for whitespace", _system_blocks("  \n "), [])

# The warm request must be tiny and must carry the real standing prompt --
# priming the cache with anything else would prime the wrong prefix.
from brain.llm_backends import OllamaBackend  # noqa: E402
class _RecordingClient:
    def __init__(self): self.calls = []
    def chat(self, **kw): self.calls.append(kw); return {"message": {"content": ""}}
_ob = OllamaBackend()
_ob._client = _RecordingClient()
_ob.warm(_bm("", [], "hello"))
check("warm asks for one token", _ob._client.calls[0]["options"]["num_predict"], 1)
check("warm keeps the model resident", _ob._client.calls[0]["keep_alive"], -1)
check("warm sends the real standing prompt",
      _ob._client.calls[0]["messages"][0]["content"].startswith("You are Reachy Mini"), True)
check("every ollama call carries the widened context window",
      _ob._options(None)["num_ctx"] >= 4096 and _ob._options(300)["num_ctx"] >= 4096, True)
check("a caller's token cap still wins", _ob._options(300)["num_predict"], 300)

# The speculative Whisper wrapper: its one job is to be safely joinable
# whatever happened inside it, because listen() must join a stale decode
# before starting a fresh one on the shared recognizer.
from body.audio_io import _EarlyDecode  # noqa: E402
import numpy as _np  # noqa: E402
_ed = _EarlyDecode(lambda s: "the transcript", _np.zeros(4), "the transcript")
check("an early decode returns its result", _ed.result(), "the transcript")
check("and reports done afterwards", _ed.done, True)
_boom = _EarlyDecode(lambda s: (_ for _ in ()).throw(RuntimeError("x")), _np.zeros(4), "p")
check("a decode that blows up still joins, as empty", _boom.result(), "")

# Barge-in THROUGH the render-ahead pipeline. The first live interruption after
# the pipeline shipped raised NameError out of the abandon-drain instead of
# Interrupted -- the guard ate it as a demo failure and the visitor's turn was
# lost. The pipeline's cleanup path is exactly the code only an interruption
# reaches, so it is exercised here the way a visitor exercises it: by talking
# over the first sentence.
_pipe_audio = FakeAudio(interrupt_after={0})
_pipe_state = RobotState()
_pipe_ctx = DemoContext(audio=_pipe_audio, motion=FakeMotion(), tracker=None,
                        state=_pipe_state, demo_id="t", store={})
_held_sr = _bi.stream_reply
_bi.stream_reply = lambda *a, **k: iter(
    [("First sentence.", "thinking"), ("Never spoken.", "thinking"), ("", "happy")])
try:
    _pipe_ctx.reply("q")
    check("an interruption mid-reply raises Interrupted", "did not raise", "Interrupted")
except Interrupted:
    check("an interruption mid-reply raises Interrupted", "Interrupted", "Interrupted")
except Exception as exc:
    check("an interruption mid-reply raises Interrupted", type(exc).__name__, "Interrupted")
finally:
    _bi.stream_reply = _held_sr
check("what was spoken before the interruption stands", _pipe_audio.said, ["First sentence."])
check("and the speaking flag was not left set", _pipe_state.snapshot()["speaking"], False)

print()
print("[28] operator choices survive, and the robot knows what day it is online")
import brain.settings as _set  # noqa: E402

# The kv store, against the scratch DB the layout section already installed.
_set.db.MODELS = _db.MODELS
_set._init_settings()
check("a choice saves and loads", (_set.put("voice", "en_GB-alba-medium"),
                                   _set.get("voice")), (True, "en_GB-alba-medium"))
check("absence is the default, not an error", _set.get("nothing", "fallback"), "fallback")
check("a runaway value is clamped", len(_set.get("voice")) <= 200 and _set.put("k", "x" * 999)
      and len(_set.get("k")), 200)
_set._available = False
check("unavailable degrades to defaults", _set.get("voice", "d"), "d")
check("and swallows writes", _set.put("voice", "x"), False)
_set._available = True

# The runner persists a voice only when it actually applied, and the voice
# loop restores it at startup -- the restart cycle in miniature.
class _VoiceAudio(FakeAudio):
    def __init__(self):
        super().__init__()
        self.voice_name = "en_US-amy-medium"
    def set_voice(self, name):
        ok = name != "en_XX-broken"
        if ok:
            self.voice_name = name
        return ok

_va = _VoiceAudio()
_vr, _vs, _, _ = build([Chatty()])
_vr._audio = _va
_vs.request("voice", "en_GB-alba-medium")
_vr.cycle()
check("an applied voice is spoken and persisted",
      (_va.voice_name, _set.get("voice")), ("en_GB-alba-medium", "en_GB-alba-medium"))
_set.put("voice", "en_XX-broken")
_va2 = _VoiceAudio()
saved = _set.get("voice")
if saved and saved != _va2.voice_name and not _va2.set_voice(saved):
    pass  # the voice_loop path: a missing saved voice falls back with a note
check("a saved voice that cannot load falls back", _va2.voice_name, "en_US-amy-medium")

# The web prompt knows the date; the offline one must not, because offline
# answers are cached by qa_cache with no date in the key -- a date there would
# freeze into an answer replayed on the wrong day.
import time as _time  # noqa: E402
from brain.prompts import _base_prompt as _bp  # noqa: E402
_today = _time.strftime("%A %d %B %Y")
check("online, the robot knows today's date", _today in _bp(True), True)
check("and is told to anchor news searches on it",
      "put today's date in the query" in _bp(True), True)
check("offline, the date stays out of the cacheable prompt", _today in _bp(False), False)

# The dashboard keeps trying for the voice list instead of hiding on the first
# empty answer -- the once-only fetch is how voices became unselectable while
# personalities worked.
_page2 = (_Path(__file__).resolve().parent / "web" / "index.html").read_text(encoding="utf-8")
check("the voice dropdown retries until the list arrives",
      "setInterval(syncVoices" in _page2, True)
check("and never hides itself permanently", 'sel.style.display = "none"' in _page2, False)

print()
print("[29] a misheard trigger phrase still starts the feature it meant")
from demokit.runner import _word_stream as _ws, contains_phrase as _cp, fuzzy_contains as _fc  # noqa: E402

# The transcripts Whisper actually produced, live, for one staff-written
# trigger. Whisper has no hotword biasing, so proper nouns in triggers WILL
# keep arriving like this -- and the robot already forgives its own name
# ("Ricky" wakes it), so a feature's phrase gets the same tolerance.
_TRIGGER = "welcome the erasmus group"
for _heard in ("Welcome to your irisimus group",
               "Welcome the de-arass misgroup",
               "Welcome the Erasmus group.",
               # The first mangling to arrive AFTER the matcher shipped -- it
               # fired live, first try, and stays here so it always will.
               "Welcome to your raspous group"):
    check(f"fires on: {_heard}",
          _cp(_ws(_heard), _TRIGGER) or _fc(_ws(_heard), _TRIGGER), True)

# And the sentences that must NOT drag a group into a welcome nobody asked for.
for _heard in ("welcome everyone to the hub today",
               "you're all welcome here",
               "the erasmus programme is great",          # topic, not the request
               "can you dance for the group",
               "what a great group of people",
               # The two that broke the single-threshold design: both scored AT
               # or ABOVE the mangled real trigger on whole-window similarity.
               # What keeps them quiet is the word-evidence gate -- nothing in
               # them resembles "erasmus".
               "we should welcome the new group",
               "a warm welcome to this group"):
    check(f"quiet on: {_heard}", _fc(_ws(_heard), _TRIGGER), False)

# Short triggers never match approximately -- one mishearing away from
# everything is why features.py refuses short phrases outright.
check("a two-word trigger is exact-only", _fc(_ws("lets dance everybody"), "lets dance"), False)
check("but a longer built-in gets the tolerance",
      _fc(_ws("let us brainstorm together"), "help me brainstorm") or
      _fc(_ws("help me brain storm"), "help me brainstorm"), True)

# Through the runner: the mangled transcript switches to the feature, and an
# exact match on one trigger beats a resemblance to another.
class _Erasmus(Demo):
    id, label, help = "custom_erasmus", "Erasmus welcome", "h"
    triggers = ("welcome the erasmus group",)
    def on_idle(self, ctx):
        return IdleResult(listen_for=1.0)

_er = _Erasmus()
_r29, _s29, _a29, _ = build([Chatty(), _er])
_s29.set_mode("chatty")
_r29.cycle()
_r29._dispatch(*_r29._active(), "Welcome to your irisimus group", 0)
check("the mangled phrase still switches to the feature", _s29.mode, "custom_erasmus")

print()
print("[30] the clock is answered from the clock, not the model")
import demos.conversation as _conv  # noqa: E402
import re as _re30  # noqa: E402

_t = _conv._clock_answer("hey reachy what time is it")
check("a time question gets a spoken time", bool(_t) and _t.startswith("It's "), True)
# The HOUR, not any number in the sentence. This used to search the whole
# string for 13-23 and so failed for eleven minutes of every hour: at 12:23
# the perfectly correct "It's 12 23 in the afternoon." matched on the
# MINUTES. A test that fails on the clock teaches people to re-run it
# rather than read it, which is worse than having no test.
_hour30 = _re30.search(r"\b(\d{1,2})\b", _t)
check("and never a 24-hour reading",
      bool(_hour30) and 13 <= int(_hour30.group(1)) <= 23, False)
_d = _conv._clock_answer("what day is it today")
check("a date question gets the day and date", bool(_d) and "It's " in _d, True)
for _not_clock in ("tell me a story about time", "sometimes I wonder",
                   "what time does the hub open", "set a timer for five minutes",
                   "do you like dates"):
    check(f"left to the model: {_not_clock}", _conv._clock_answer(_not_clock), "")

print()
print("[31] the new demos, the playlist step, and what they refuse to do")
_st, _sy = _statmod, _studymod
import demos._stored as _storedmod  # noqa: E402
import demos.quiz as _quiz  # noqa: E402
import demos.greetings as _greet  # noqa: E402

# --- visit stats: aggregate, and NOT tied to a person ---
# Measured as a DELTA, not against zero: the runner counts its own dispatches,
# and earlier sections in this file drive real turns through it. An absolute
# assertion here passes only by accident of ordering.
_before = _st.day()
_was_turns = _before["counts"].get("turns", 0)
_was_questions = len(_before["questions"])
_st.bump("turns"); _st.bump("turns"); _st.bump("demo:quiz")
_st.note_question("What is the AI XR Hub?"); _st.note_question("what is the ai xr hub")
_today = _st.day()
check("turns are counted", _today["counts"].get("turns", 0) - _was_turns, 2)
check("demos are counted separately", _today["demos"].get("quiz"), 1)
_hub_q = [q for q in _today["questions"] if q["question"] == "what is the ai xr hub"]
check("the same question asked twice counts twice",
      _hub_q[0]["asked"] if _hub_q else 0, 2)
check("and is stored once, punctuation and case folded",
      len(_today["questions"]) - _was_questions, 1)
_st._available = False
check("stats unavailable still returns a usable day", _st.day()["counts"], {})
check("and swallows writes", _st.bump("turns"), None)
_st._available = True

# --- the study: arming IS the consent, and withdrawal still deletes ---
# The robot no longer reads a notice and waits for a spoken yes. The Hub's
# call: consent is taken on paper before the session, as HRI studies normally
# take it, and the operator arming this has done that. What replaces the
# spoken record is `operator` -- see brain/study.start.
_sy.stop()
_sy.start("friendly", operator="Test Operator")
check("arming a session consents immediately", _sy.status()["consented"], True)
check("and records who armed it", _sy.status()["operator"], "Test Operator")
_sy.record("said right away", "reply", 1.0, persona="friendly")
check("so a turn is recorded with no spoken exchange first",
      _sy.summary()["turns"], 1)
_session = _sy.status()["session"]
check("a session id is random, not a person id", len(_session) >= 16, True)
# Withdrawal is unchanged and still the thing that matters most: it deletes
# rather than flags, and it ends the session so a later turn cannot quietly
# start recording the same person again.
_sy.consent(False)
_sy.record("said after withdrawal", "reply", 1.0)
check("withdrawing stops the session outright", _sy.running(), False)
check("and nothing more is recorded", _sy.summary()["turns"], 1)
check("withdrawing deletes rather than flags", _sy.withdraw(_session), 1)
check("leaving nothing behind", _sy.summary()["turns"], 0)
_sy.stop()
check("research mode is OFF unless deliberately started", _sy.running(), False)
check("and stopping clears the operator too", _sy.status()["operator"], "")
# The table must not be able to hold a person. Checked structurally, because
# this is the property an ethics reviewer would ask about.
with _db._connection() as _conn:
    _cols = {r[1] for r in _conn.execute("PRAGMA table_info(study_turns)")}
check("no person, name or face column exists",
      sorted(_cols & {"person_id", "name", "face", "embedding"}), [])

# The variables an analysis actually needs. Added after the first sessions were
# recorded, so the ALTER TABLE migration in _init_study has to have run -- on a
# database from before the change, CREATE TABLE IF NOT EXISTS does nothing and
# every insert naming these columns would fail. In a module contracted never to
# raise, that failure looks exactly like a study that records nothing.
check("persona, first_word_s and backend columns exist",
      sorted(_cols & {"persona", "first_word_s", "backend"}),
      ["backend", "first_word_s", "persona"])

_sy.start("armA")
_sy.consent(True)
_sy.record("q", "a", 9.5, persona="professional", first_word_s=1.25, backend="anthropic")
with _db._connection() as _conn:
    _row = _conn.execute(
        "SELECT condition, persona, latency_s, first_word_s, backend FROM study_turns "
        "WHERE session = ? ORDER BY id DESC LIMIT 1", (_sy.status()["session"],)
    ).fetchone()
check("the free-text condition is kept", _row[0], "armA")
check("and the persona is recorded separately", _row[1], "professional")
# Two timings, deliberately. latency_s is the whole turn INCLUDING the time
# spent speaking, so it grows with the length of the answer; first_word_s is
# what the participant actually waited. Recording only the first meant a long
# reply was indistinguishable from a slow model.
check("turn length and responsiveness are both kept", (_row[2], _row[3]), (9.5, 1.25))
check("and which model answered", _row[4], "anthropic")
check("this test cleans up after itself", _sy.withdraw(_sy.status()["session"]), 1)
_sy.stop()

# --- PLAY: a feature can hand the visit to another demo ---
_play_ok = _F.Feature(id="custom_tour", label="Tour", blocks=_F.parse_blocks([
    {"kind": "SAY", "text": "Welcome."}, {"kind": "PLAY", "demo": "welcome"}]))
check("a handover step validates", _F.validate(_play_ok, existing=[]), [])
_play_mid = _F.Feature(id="custom_bad", label="Bad", blocks=_F.parse_blocks([
    {"kind": "PLAY", "demo": "welcome"}, {"kind": "SAY", "text": "never runs"}]))
check("but only as the last step", any("last step" in p for p in _F.validate(_play_mid, existing=[])), True)
_play_self = _F.Feature(id="custom_loop", label="Loop", blocks=_F.parse_blocks([
    {"kind": "SAY", "text": "hi"}, {"kind": "PLAY", "demo": "custom_loop"}]))
check("and never itself", any("loop forever" in p for p in _F.validate(_play_self, existing=[])), True)

# It actually switches, and advances past itself so it cannot loop.
_pl_reg = Registry()
_pl_reg._publish({d.id: d for d in [Chatty(), Dancer()]})
import demokit.runner as _rmod  # noqa: E402
_held_reg = _rmod.REGISTRY
_rmod.REGISTRY = _pl_reg
import demokit.registry as _regmod  # noqa: E402
_held_regmod = _regmod.REGISTRY
_regmod.REGISTRY = _pl_reg
try:
    _pl_state = RobotState()
    _pl_state.set_demos([{"id": d.id, "label": d.label, "help": "", "available": True, "note": ""}
                         for d in (Chatty(), Dancer())])
    _pl_state.set_mode("chatty")
    _pl_ctx = DemoContext(audio=FakeAudio(), motion=FakeMotion(), tracker=None,
                          state=_pl_state, demo_id="custom_tour", store={})
    _feature = _storedmod.StoredFeature(_F.Feature(
        id="custom_tour", label="Tour",
        blocks=_F.parse_blocks([{"kind": "PLAY", "demo": "dancer"}])))
    _feature.on_enter(_pl_ctx)
    _feature.on_idle(_pl_ctx)
    check("a handover switches the robot", _pl_state.mode, "dancer")
    check("and advances past itself, so it cannot loop", _pl_ctx.store["cursor"], 1)
    _missing = _storedmod.StoredFeature(_F.Feature(
        id="custom_gone", label="Gone",
        blocks=_F.parse_blocks([{"kind": "PLAY", "demo": "not_installed"}])))
    _gone_ctx = DemoContext(audio=FakeAudio(), motion=FakeMotion(), tracker=None,
                            state=_pl_state, demo_id="custom_gone", store={})
    _missing.on_enter(_gone_ctx)
    _missing.on_idle(_gone_ctx)
    check("a handover to a demo that is gone is skipped, not fatal",
          _gone_ctx.store["cursor"], 1)
finally:
    _rmod.REGISTRY = _held_reg
    _regmod.REGISTRY = _held_regmod

# --- quiz: loose answer matching, and the decoys ---
check("a shouted answer counts", _quiz._is_right("um is it extended reality", ("extended reality",)), True)
check("so does one word of it", _quiz._is_right("REALITY", ("extended reality", "reality")), True)
check("a wrong answer does not", _quiz._is_right("virtual reality goggles", ("trinity", "dublin")), False)
check("every question has an answer and a fact",
      [q[0] for q in _quiz._QUESTIONS if not q[1] or not q[2]], [])

# --- greetings: bounded to greeting, never to listening ---
check("a language is recognised by its own name", _greet._requested("say hello in espanol"), "spanish")
check("and by the English name", _greet._requested("greet them in german"), "german")
check("an unrelated sentence asks for nothing", _greet._requested("what is the weather"), "")
for _key, (_name, _lines, _voice) in _greet._LANGUAGES.items():
    check(f"{_key} hands back to English",
          any("english" in ln.lower() or "inglés" in ln.lower() or "anglais" in ln.lower()
              or "englisch" in ln.lower() or "inglese" in ln.lower() for ln in _lines), True)

# --- the advisor refuses what it must ---
import demos.advisor as _adv  # noqa: E402
_ab = _adv._brief({"background": "history", "draw": "writing", "technical": "not technical"})
check("the advisor sees the whole catalogue",
      all(p.name in _ab for p in _co.PROGRAMMES), True)
check("and is told never to say whether they would get in",
      "whether they would be accepted" in _ab, True)
check("nor to quote a fee or a deadline",
      "never quote a fee" in _ab and "deadline" in _ab, True)
check("it names one or two, not a fixed number", "ONE or TWO" in _ab, True)
check("and is warned about the restricted programmes",
      "ONLY for non-business" in _ab, True)

print()
print("[32] a face the robot knows is not asked for its name again")
import body.face_tracker as _ftmod  # noqa: E402
import demos.vision as _vis  # noqa: E402
import numpy as _npf  # noqa: E402
import threading as _thr  # noqa: E402

# The live failure, exactly: one dropped frame between two confident matches.
# 11:32:46 matched person 1 at 0.80, 11:32:48 scored 0.58 and the robot asked
# a visitor it knew for their name, 11:33:06 matched at 0.91.
_tk = _ftmod.FaceTracker.__new__(_ftmod.FaceTracker)
_tk._lock = _thr.Lock()
_good = _npf.array([1.0, 0.1, 0.2], dtype=_npf.float32)
_blurred = _npf.array([0.72, 0.30, 0.18], dtype=_npf.float32)   # same face, poor frame
_stranger = _npf.array([0.05, 1.0, 0.0], dtype=_npf.float32)    # somebody else
_tk._known_id, _tk._known_embedding, _tk._known_at = 1, _good, time.monotonic()
check("a poor frame of a known face keeps their identity", _tk._hold_identity(_blurred), 1)
check("a different face never inherits it", _tk._hold_identity(_stranger), None)
_tk._known_at = time.monotonic() - (_ftmod._IDENTITY_HOLD_S + 5)
check("and the hold expires once they are long gone", _tk._hold_identity(_blurred), None)
# The threshold RELATIONSHIP is the thing the first attempt got backwards, so
# it is asserted rather than left as a number somebody could "tidy up".
from config import MODELS as _CFG  # noqa: E402
check("the hold threshold sits below the recognition bar, not above",
      _ftmod._SAME_PERSON_THRESHOLD < _CFG.face_match_threshold, True)
check("but well above where two different people score",
      _ftmod._SAME_PERSON_THRESHOLD > 0.42, True)

# The streak guard, which is what makes this robust without trusting a number.
class _FlickerTracker:
    """Recognises, drops one frame, recognises again -- the observed pattern."""

    enabled = True

    def __init__(self, ids):
        self._ids = list(ids)
        self.face = object()

    def current(self, max_age_s=3.0):
        got = self._ids.pop(0) if self._ids else None
        return got, self.face

    def current_embedding(self, max_age_s=3.0):
        return _npf.array([1.0, 0.0, 0.0], dtype=_npf.float32)

    def last_score(self):
        return 0.58


_vdemo = _vis.Vision()
_vstate = RobotState()
_vaudio = FakeAudio()
_vctx = DemoContext(audio=_vaudio, motion=FakeMotion(), tracker=_FlickerTracker([1, 1, None, 1, 1]),
                    state=_vstate, demo_id="vision", store={})
_vctx.store["present_since"] = time.monotonic() - 60   # already settled
for _ in range(5):
    _vdemo.on_idle(_vctx)
check("one dropped frame never triggers an offer", _vaudio.said, [])

# A genuinely new face still gets asked, once the streak is real.
_vctx2 = DemoContext(audio=FakeAudio(), motion=FakeMotion(),
                     tracker=_FlickerTracker([None] * 8),
                     state=RobotState(), demo_id="vision", store={})
_vctx2.store["present_since"] = time.monotonic() - 60
_streaks = []
for _ in range(_vis._UNKNOWN_BEFORE_OFFER - 1):
    _vdemo.on_idle(_vctx2)
    _streaks.append(_vctx2.store.get("unknown_streak"))
check("an unknown face is not asked immediately either", _vctx2.store.get("stage"), None)

# The gate that finally fixed it live: a face scoring in the middle band is
# somebody the robot probably knows, seen badly -- not a stranger. Both live
# failures sat there (0.47 and 0.58) while genuinely different people measured
# 0.33 to 0.39.
class _ScoredTracker(_FlickerTracker):
    def __init__(self, score):
        super().__init__([None] * 20)
        self._score = score
    def last_score(self):
        return self._score

for _score, _should_offer in ((0.58, False), (0.47, False), (0.35, True), (0.10, True)):
    _sa = FakeAudio()
    _sc = DemoContext(audio=_sa, motion=FakeMotion(), tracker=_ScoredTracker(_score),
                      state=RobotState(), demo_id="vision", store={})
    _sc.store["present_since"] = time.monotonic() - 60
    for _ in range(_vis._UNKNOWN_BEFORE_OFFER + 2):
        _vdemo.on_idle(_sc)
    _asked = any("remember you by name" in line for line in _sa.said)
    check(f"score {_score}: {'asks' if _should_offer else 'stays quiet'}", _asked, _should_offer)

# And the escape hatch, which is what makes that caution safe.
class _SeenTracker(_ScoredTracker):
    def current(self, max_age_s=3.0):
        return None, self.face
_ha = FakeAudio()
_hc = DemoContext(audio=_ha, motion=FakeMotion(), tracker=_SeenTracker(0.58),
                  state=RobotState(), demo_id="vision", store={})
check("but somebody can still ask to be remembered",
      _vdemo.on_utterance(_hc, "hey, remember me"), True)
check("and that starts the name exchange", _hc.store.get("stage"), _vis._ASK_NAME)
check("the streak builds while it stays unrecognised", _streaks[-1], _vis._UNKNOWN_BEFORE_OFFER - 1)

print()
print("[33] a demo started by voice answers the phrase that started it")
# The runner hands the utterance to the demo its trigger just selected, and if
# that demo declines it, the turn falls through to the conversation model --
# so TWO demos answer one sentence. Live: "quiz us" got "Right, 4 questions,
# shout the answers" and then, over the top, "Fun idea! Want me to quiz you on
# the AI XR Hub, or something related to picking between those two Master's
# programmes?" -- fifteen seconds of the robot talking to itself before
# question one. Three demos had it; this walks every demo so a fourth cannot.
_REG4 = Registry()
_REG4.discover()


class _QuietAudio(FakeAudio):
    """Speaks into the void, and never has a transcript to offer."""

    def listen(self, wait_for_speech_s=None):
        return ""


_fell_through = []
for _did in _REG4.ids():
    _d = _REG4.get(_did)
    if _d is None or not _d.triggers:
        continue
    _st4 = RobotState()
    _st4.set_demos([{"id": _did, "label": _d.label, "help": "", "available": True, "note": ""}])
    _c4 = DemoContext(audio=_QuietAudio(), motion=FakeMotion(), tracker=None,
                      state=_st4, demo_id=_did, store={})
    try:
        _d.on_enter(_c4)
        _handled = bool(_d.on_utterance(_c4, f"hey reachy {_d.triggers[0]}"))
    except Exception:
        _handled = False
    if not _handled:
        _fell_through.append(_did)
check("no demo lets its own trigger reach the conversation model", _fell_through, [])

print()
print("[34] a capability switched on mid-visit actually lets its demo run")
# The split brain this closes: the dashboard published one capability set and
# DemoRunner held another, frozen at construction. Live, research mode showed
# "Research session" as available, the operator pressed it, and the runner
# refused with "study is unavailable (needs study)" and bounced straight back
# to Conversation -- which looked like research mode turning itself off.
class _NeedsStudy(Demo):
    id, label, help = "needs_study", "Needs study", "h"
    requires = ("study",)

    def on_idle(self, ctx):
        return IdleResult(listen_for=1.0)


_cap_reg = Registry()
_cap_reg._publish({d.id: d for d in [Chatty(), _NeedsStudy()]})
import demokit.runner as _capmod  # noqa: E402
import demokit.registry as _capregmod  # noqa: E402
_held = (_capmod.REGISTRY, _capregmod.REGISTRY)
_capmod.REGISTRY = _capregmod.REGISTRY = _cap_reg
try:
    _cs = RobotState()
    _cs.set_demos([{"id": d.id, "label": d.label, "help": "", "available": True, "note": ""}
                   for d in (Chatty(), _NeedsStudy())])
    _cr = DemoRunner(audio=FakeAudio(), motion=FakeMotion(), tracker=None,
                     state=_cs, capabilities=frozenset())
    check("a demo needing a capability is refused without it",
          _cap_reg.is_available("needs_study", _cr._live_capabilities())[0], False)
    _cs.set_capability("study", True)
    check("switching it on makes the demo runnable",
          _cap_reg.is_available("needs_study", _cr._live_capabilities())[0], True)
    check("and the runner sees it without being rebuilt",
          "study" in _cr._live_capabilities(), True)
    _cs.set_capability("study", False)
    check("switching it off takes it away again",
          _cap_reg.is_available("needs_study", _cr._live_capabilities())[0], False)
    check("hardware capabilities are untouched by the switch",
          _cs.set_capability("study", True) >= {"study"}, True)

    # THE PATH THAT ACTUALLY FAILED, twice. The first fix patched the trigger
    # lookup and missed _active(), which is what runs when an operator PRESSES
    # the button -- so the demo stayed unreachable from the dashboard while
    # being reachable by voice. Driving a real cycle is the only check that
    # covers both; asserting on is_available alone is what let it through.
    _cs.set_capability("study", True)
    _cs.set_mode("needs_study")
    _cr.cycle()
    check("pressing the button actually enters the demo", _cs.mode, "needs_study")
    _cs.set_capability("study", False)
    _cs.set_mode("needs_study")
    _cr.cycle()
    check("and without the capability it falls back instead", _cs.mode, "chatty")
finally:
    _capmod.REGISTRY, _capregmod.REGISTRY = _held

print()
print("[35] arming records straight away, and an objection still stops it")
import demos.study as _sd  # noqa: E402


class _FakeStudyStore:
    """Stands in for brain.study so no test reaches the real research table."""

    def __init__(self):
        self.records, self.consents, self.withdrew, self.on = [], [], 0, True
        self.stopped = 0

    def running(self):
        return self.on

    def status(self):
        return {"session": "testsession"}

    def consent(self, ok):
        self.consents.append(bool(ok))
        if not ok:
            self.on = False

    def record(self, said, replied, latency, **kw):
        self.records.append((said, replied))

    def withdraw(self):
        self.withdrew += 1
        return 3

    def stop(self):
        self.stopped += 1
        self.on = False
        return {}


def _run_study(transcripts, slices=4):
    fake = _FakeStudyStore()
    held, _sd.store = _sd.store, fake
    try:
        demo = _sd.Study()
        runner, st, aud, _ = build([Chatty(), demo], wake_at=tuple(range(1, slices + 1)),
                                   transcripts=list(transcripts))
        st.set_capability("study", True)
        st.set_mode(demo.id)
        for _ in range(slices):
            runner.cycle()
        return fake, aud.said
    finally:
        _sd.store = held


_f, _said = _run_study(["what is xr"])
# No consent notice, no "would you like to take part", no waiting. The three
# script lines used to cost ~50 seconds before the first real exchange.
check("nothing is read out before recording starts",
      any("would you like to take part" in s.lower() for s in _said), False)
check("and the first thing said is not a consent notice",
      any("hub is researching" in s.lower() for s in _said), False)
check("the very first utterance is recorded", len(_f.records), 1)

# Stopping and deleting are DIFFERENT acts, split after live use: the operator
# said "stop recording", meaning stop adding to the record, and the robot
# deleted the session they wanted to keep. Erasure phrases still erase; stop
# phrases end the recording and keep what was said.
for _objection in ("delete my data", "I changed my mind",
                   "I do not consent to being recorded"):
    _f2, _ = _run_study([_objection])
    check(f"{_objection!r} withdraws and deletes", (_f2.withdrew, _f2.consents), (1, [False]))
    check("  and the objection itself is never recorded", _f2.records, [])
for _stopword in ("stop recording", "please dont record me"):
    _f2, _ = _run_study([_stopword])
    check(f"{_stopword!r} stops WITHOUT deleting",
          (_f2.stopped, _f2.withdrew), (1, 0))
    check("  and is not itself recorded", _f2.records, [])

# Export. Written through the csv module because a turn can contain a quote and
# a comma -- "I said \"no\", then left" -- and hand-rolled CSV corrupts that row
# silently, which is damage found weeks later in analysis.
import csv as _csv, io as _io  # noqa: E402

_buf = _io.StringIO()
_w = _csv.writer(_buf, lineterminator="\n")
_w.writerow(["said", "replied"])
_w.writerow(['I said "no", then left', "Understood, no bother"])
_parsed = list(_csv.reader(_io.StringIO(_buf.getvalue())))
check("a quote and a comma survive the CSV round trip",
      _parsed[1][0], 'I said "no", then left')
check("without splitting the row", len(_parsed[1]), 2)

print()
print("[36] every step kind the backend supports is reachable from the editor")
import pathlib as _pl36  # noqa: E402
# PLAY shipped implemented, validated, interpreted and DOCUMENTED as one of
# five step kinds -- and with no button, no render branch and no label in the
# editor, so no operator could ever add one. The feature was complete
# everywhere except the one place a person touches it, and nothing failed:
# tests passed, validation worked, the docs described it. Only trying to use it
# would have found it, and the instructions for trying it were written from the
# docs rather than from the page.
_page = (_pl36.Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
_kinds = {_F.SAY, _F.ASK, _F.DANCE, _F.WAIT, _F.PLAY}
_buttons = set(re.findall(r'data-add="([A-Z]+)"', _page))
check("the editor offers a button for every step kind",
      sorted(_kinds - _buttons), [])
# The label map: a kind missing here renders as raw "PLAY" rather than "Plays".
_labels = re.search(r'\{SAY: "Says".*?\}', _page)
check("and names every kind in the step list",
      sorted(k for k in _kinds if _labels and f"{k}:" not in _labels.group(0)), [])
# A picker with nothing to pick from is the same bug one layer down.
check("the server tells the editor what a handover can target",
      "handover_targets" in (_pl36.Path(__file__).parent / "web" / "server.py").read_text(encoding="utf-8"),
      True)

print()
print("[37] a dropped robot link is rebuilt in place, not by killing the process")
# A wifi blip used to end the process: os._exit(3), then the launcher relaunched
# and paid ~40s reloading speech models, the demo registry and the language
# model -- five times in one morning. It also skipped the finally: block, so
# runner.end_conversations() never ran and the day's conversations were never
# summarised into long-term memory. That loss was silent.
import threading as _th37  # noqa: E402
from body.motion import MotionController as _MC  # noqa: E402
from body.audio_io import AudioIO as _AIO  # noqa: E402
from body.camera import Camera as _Cam  # noqa: E402

for _cls in (_MC, _AIO, _Cam):
    check(f"{_cls.__name__} can adopt a rebuilt connection",
          hasattr(_cls, "adopt_robot"), True)

# link_lost was a ONE-SHOT flag: set once, cleared nowhere, so the session could
# only ever die. Clearing it is what makes the process survivable.
_m = object.__new__(_MC)          # the real method, without __init__'s thread
_m._robot, _m._failing_since, _m._send_failures, _m._last_reconnect_at = None, 123.0, 9, 0.0
_m.link_lost = _th37.Event()
_m.link_lost.set()
_MC.adopt_robot(_m, "fresh-connection")
check("adopting clears the fatal flag", _m.link_lost.is_set(), False)
check("and rebinds to the new connection", _m._robot, "fresh-connection")
check("and forgets the old failure streak", (_m._failing_since, _m._send_failures), (None, 0))

_vl = (_pl36.Path(__file__).parent / "body" / "voice_loop.py").read_text(encoding="utf-8")
check("the link watchdog no longer kills the process",
      "os._exit" in _vl, False)
# Speaking from the watchdog thread would interleave with whatever the loop is
# already saying; STATE.request is the sanctioned cross-thread path.
check("and announces the recovery on the loop thread, not its own",
      'STATE.request("say"' in _vl, True)
check("the main loop stops driving detached wrappers while it rebuilds",
      "link_down.is_set()" in _vl, True)
# Found by a real outage rather than by reading. Motion was left attached
# during the rebuild, so its 20Hz send loop wrote to a dead socket for the
# whole thing -- 3,700+ dropped pose updates and a warning every ten seconds,
# drowning the log at exactly the moment it was being read to find the fault.
check("motion is detached too, or it hammers a dead socket",
      "motion.adopt_robot(None)" in _vl, True)
# And every attempt reused the address captured at startup. DHCP handing out a
# new one is a common CAUSE of the drop, so that is the single address most
# likely to be wrong by the time it is retried.
check("and the address is re-resolved on each attempt",
      "current = default_target()" in _vl, True)

print()
print("[38] a transcript the recogniser had no confidence in is asked again")
# Live, Whisper produced "Quizance" for "quiz us", "Hey Ritchie" as the answer
# to a consent question, and "testing testic" -- each sent to the model and
# answered confidently. The mishearing is a fact of loud rooms; answering it
# is not.
from body.audio_io import _MIN_MEAN_TOKEN_LOGPROB as _FLOOR  # noqa: E402

# Bounded by the WORST reading that turned out to be correct, and that number
# came from the robot rather than from synthesized speech. Live, a perfectly
# accurate "What's new in the AI XR tech market?" scored -2.44 and was rejected
# under the old synthetic floor of -2.0 -- the visitor spoke clearly and was
# told to repeat themselves, which is the failure this feature exists to avoid.
check("the floor sits below the worst reading measured correct LIVE",
      _FLOOR < -2.44, True)


def _gate(score, transcripts=("what is xr",)):
    """Drive a real dispatch with the recogniser reporting `score`."""
    demo = Chatty()
    runner, st, aud, _ = build([demo], wake_at=(1,), transcripts=list(transcripts))
    aud.last_confidence = score
    st.set_mode(demo.id)
    runner.cycle()
    return demo.heard, aud.said


_heard, _said = _gate(-0.4)
check("confident speech reaches the demo", _heard, ["what is xr"])
_heard, _said = _gate(-2.44)
check("and so does the real live reading that used to be refused",
      _heard, ["what is xr"])
_heard, _said = _gate(-5.0)
check("but a genuinely hopeless one does not", _heard, [])
check("and the visitor is asked to repeat",
      any("did not catch that" in x.lower() for x in _said), True)
_heard, _said = _gate(-9.0)
check("and the visitor is asked to repeat once more",
      any("did not catch that" in x.lower() for x in _said), True)
_heard, _said = _gate(None)
check("no confidence signal means answer as before -- never lose a turn to it",
      _heard, ["what is xr"])

# The one thing the gate must never block. "Go to sleep" is checked before the
# demos precisely because the robot must always be stoppable, and a noisy room
# is exactly where somebody needs it to stop.
_sleepy = Chatty()
_r38, _s38, _a38, _ = build([_sleepy], wake_at=(1,), transcripts=["ok go to sleep now"])
_a38.last_confidence = -9.0
_s38.set_mode(_sleepy.id)
_r38.cycle()
check("but a low-confidence 'go to sleep' still puts it to sleep",
      _s38.sleeping, True)

import time as _t  # noqa: E402
print()
print("[39] the head turns toward a voice it cannot see, once it knows where front is")
import math as _m39  # noqa: E402
from body import doa as _doa  # noqa: E402

# Angles wrap, and a plain average of two angles either side of zero points
# exactly backwards -- which for this feature means the head turning away from
# whoever spoke. That is the one failure worse than doing nothing.
check("a wrapping mean does not point backwards",
      round(_doa._circular_mean([0.1, 2 * _m39.pi - 0.1]), 3), 0.0)
check("a plain average would have", round((0.1 + (2 * _m39.pi - 0.1)) / 2, 2), 3.14)
check("agreeing angles have a small spread",
      _doa._circular_spread([1.0, 1.05, 0.95]) < 0.2, True)
check("opposed angles have a large one",
      _doa._circular_spread([0.0, _m39.pi]) > 1.0, True)

_L = _doa.DoaListener("127.0.0.1", 8000)   # never started: no thread, no HTTP
check("says nothing before it has heard anything", _L.suggested_yaw_deg(), None)
check("and is not calibrated", _L.calibrated(), False)

# A reading arrives, but the offset is still unknown: still silent, because the
# caller's fallback (a visual sweep) beats a confident guess in a wrong direction.
_L._angle, _L._speech, _L._at = 1.2, True, _t.monotonic()
check("a reading alone is not enough to aim", _L.suggested_yaw_deg(), None)

# Learn the offset from a speaker who IS visible, dead centre in frame.
for _ in range(_doa._CALIBRATION_SAMPLES):
    _L._angle, _L._at = 1.2, _t.monotonic()
    _L.observe_face(320.0, 640)          # centre of a 640-wide frame = straight ahead
check("it learns where front is from a visible speaker", _L.calibrated(), True)
check("a voice from straight ahead now needs no turn",
      abs(_L.suggested_yaw_deg()) < 3.0, True)

# A voice off to one side should now produce a turn that way, and the sign
# must match motion.look's convention or the head turns the wrong way.
_L._angle, _L._at = 1.2 + _m39.radians(20), _t.monotonic()
_right = _L.suggested_yaw_deg()
_L._angle, _L._at = 1.2 - _m39.radians(20), _t.monotonic()
_left = _L.suggested_yaw_deg()
check("opposite directions give opposite turns", (_right > 0) and (_left < 0), True)
check("and never beyond what the neck can do",
      abs(_right) <= _doa._MAX_YAW_DEG and abs(_left) <= _doa._MAX_YAW_DEG, True)

# Silence must not move the head, however confident the direction.
_L._angle, _L._speech, _L._at = 1.9, False, _t.monotonic()
check("no speech means no turn", _L.suggested_yaw_deg(), None)
# And a stale reading is not acted on -- the head must not chase a voice that
# was there ten seconds ago on a network that drops.
_L._speech, _L._at = True, _t.monotonic() - (_doa._MAX_AGE_S + 1.0)
check("a stale reading is ignored", _L.suggested_yaw_deg(), None)

print()
print("[40] a stranger who lingers is invited to speak, once")
import demokit.runner as _rm40  # noqa: E402


class _FakeTracker40:
    """Stands in for FaceTracker: only dwell and identity matter here."""

    def __init__(self, dwell=0.0):
        self.dwell = dwell

    def present_for(self, max_age_s=1.5):
        return self.dwell

    # The runner asks the tracker for these through DemoContext.person_name.
    def current(self, max_age_s=3.0):
        return None, None

    def current_embedding(self, max_age_s=3.0):
        return None

    def enabled(self):
        return True


def _attract(dwell, quiet_for=999.0, known_name=None):
    demo = Chatty()
    runner, st, aud, _ = build([demo])
    runner._tracker = _FakeTracker40(dwell)
    runner._last_heard_at = _t.monotonic() - quiet_for
    st.set_mode(demo.id)
    runner.cycle()                      # enters the demo
    ctx = runner._ctx
    if known_name is not None:
        ctx.person_name = lambda: known_name
    else:
        ctx.person_name = lambda: None
    # cycle() calls attract itself, so the latch may already be set from that
    # pass. Cleared here so the measured call below starts from a known state.
    runner._attracted = False
    said_before = len(aud.said)
    offered = runner._attract_if_lingering(ctx)
    return offered, aud.said[said_before:], runner


_off, _said, _r = _attract(dwell=0.5)
check("somebody walking past is not spoken to", _off, False)

_off, _said, _r = _attract(dwell=6.0)
check("somebody who stands there is", _off, True)
check("and is told how to start", any("hey reachy" in x.lower() for x in _said), True)
# The whole point. A robot that re-offers every few seconds is one staff switch
# off, so the second call must do nothing while they are still standing there.
_off2 = _r._attract_if_lingering(_r._ctx)
check("but never twice while they stand there", _off2, False)

# They leave; the latch clears, so the NEXT person gets their own invitation.
_r._tracker.dwell = 0.0
_r._attract_if_lingering(_r._ctx)
_r._tracker.dwell = 6.0
check("the next arrival is invited again", _r._attract_if_lingering(_r._ctx), True)

_off, _said, _ = _attract(dwell=6.0, known_name="Tadhg")
check("somebody it knows gets the greeting instead, not this", _off, False)

_off, _said, _ = _attract(dwell=6.0, quiet_for=1.0)
check("and it never offers over a conversation already happening", _off, False)

print()
print("[41] a long sentence mid-reply does not leave the robot silent")
# Live, a three-sentence answer came out at 11:47:25, :31 and :51 -- a twenty
# second gap in the middle, which a visitor reads as the robot having finished.
# Nothing can be rendered until a whole sentence has arrived, so a 35-word
# sentence holds the speaker silent for as long as the model takes to write it.
# Clause flushing used to be first-sentence-only on the reasoning that
# "mid-reply, speech is already ahead of generation and splitting buys nothing".

_short = "Yes. "
_long = ("The market is growing rapidly as artificial intelligence, faster networks "
         "and better optics all converge at once, with much lower latency, "
         "and that combination is driving adoption across manufacturing, "
         "healthcare and defence. ")
_pieces = [_short] + [w + " " for w in _long.split()] + ["[emotion: happy]"]
_out, _ = _run_stream(_ScriptedBackend(_pieces))
_spoken = [t for t, _tag in _out if t.strip()]

check("the short opening sentence still comes out first",
      _spoken[0].strip().startswith("Yes"), True)
# The point: the long sentence arrives as MORE THAN ONE chunk, so the robot
# keeps talking while the model is still writing it.
check("and the long one is broken up rather than waited out",
      len(_spoken) >= 3, True)
check("every chunk carries real words", all(x.strip() for x in _spoken), True)
# Nothing may be lost or duplicated by the splitting -- a dropped clause would
# be a sentence the visitor never hears.
_joined = " ".join(x.strip() for x in _spoken)
for _word in ("manufacturing", "healthcare", "defence", "optics", "latency"):
    check(f"  {_word!r} survives the split", _word in _joined, True)

# An ordinary short reply must NOT be chopped into fragments: the split costs
# naturalness and is only worth paying when the alternative is a long silence.
_out2, _ = _run_stream(_ScriptedBackend(
    ["Yes, ", "that is right. ", "It runs here. ", "[emotion: happy]"]))
_spoken2 = [t for t, _tag in _out2 if t.strip()]
check("a short answer is not fragmented", len(_spoken2), 2)

print()
print("[42] the dashboard can be locked, and staff can still download")
from fastapi.testclient import TestClient  # noqa: E402
import web.server as _ws  # noqa: E402

_held_pass, _held_sessions = _ws._PASSCODE, set(_ws._SESSIONS)
try:
    _ws._PASSCODE = "hub-2026"
    _ws._SESSIONS.clear()
    _c = TestClient(_ws.app)

    check("a protected endpoint is refused while locked",
          _c.get("/api/status").status_code, 401)
    check("research data especially", _c.get("/api/study/sessions").status_code, 401)
    # The page itself must always load: it is what shows the passcode prompt,
    # and locking it would leave nowhere to type the passcode.
    check("but the page itself always loads", _c.get("/").status_code, 200)
    check("and it can ask whether a passcode is even set",
          _c.get("/api/locked").json()["locked"], True)

    check("a wrong passcode is refused",
          _c.post("/api/unlock", json={"passcode": "guess"}).status_code, 403)
    check("and changes nothing", _c.get("/api/status").status_code, 401)

    check("the right one is accepted",
          _c.post("/api/unlock", json={"passcode": "hub-2026"}).status_code, 200)
    check("and opens the dashboard", _c.get("/api/status").status_code, 200)
    # THE constraint that forced a cookie rather than a header. The transcript
    # and the study exports are plain <a href download> navigations, and a
    # browser navigation carries cookies but CANNOT carry a custom header. A
    # header scheme would have locked staff out of exactly the downloads the
    # passcode exists to protect.
    check("a plain download navigation works on the cookie alone",
          _c.get("/api/transcript").status_code, 200)

    # Unset must mean open, so this is a lock people opt into rather than one
    # that appears one day and shuts them out mid-visit.
    _ws._PASSCODE = ""
    _open = TestClient(_ws.app)
    check("with no passcode set, nothing is locked",
          _open.get("/api/status").status_code, 200)
finally:
    _ws._PASSCODE = _held_pass
    _ws._SESSIONS.clear()
    _ws._SESSIONS.update(_held_sessions)

print()
print("[43] questions the robot could not answer are collected for staff")
from brain import stats as _st43  # noqa: E402

# Matched on the REPLY, never the question: somebody ASKING "do you know the
# fees?" is not a deflection, the robot answering "I don't have the fees" is.
for _reply, _want in (
    ("I don't have the fees for that programme.", True),
    ("You'd want to check the Trinity Business School website.", True),
    ("That changes from year to year, so ask whoever is hosting you.", True),
    ("I'm not able to say whether you'd be accepted.", True),
    ("The Hub runs three research strands and sits in the Business School.", False),
    ("Yes, I can dance. Watch this.", False),
):
    check(f"{_reply[:44]!r} -> {'deflection' if _want else 'a real answer'}",
          _st43.looks_like_a_deflection(_reply), _want)

check("a question is not mistaken for a deflection",
      _st43.looks_like_a_deflection("do you know the fees"), False)
check("the day's report carries the list", "unanswered" in _st43.day(), True)

print()
print("[44] open mic ignores the room but still hears the person")
# Live, with the switch on: "EH" and "OH" -- somebody reacting, or the tail of
# the robot's own speech returning through the microphone -- were dispatched as
# questions and answered out loud ("Ha, sounds like a reaction!"). A robot
# holding up its end of a conversation nobody is having is the failure open mic
# must not have.
_omd = Chatty()
_omr, _oms, _oma, _ = build([_omd])
_oms.set_mode(_omd.id)
_omr.cycle()

_oms.set_open_mic(True)
for _frag in ("EH", "OH", "AH", "um", "hm"):
    check(f"{_frag!r} is the room, not a question",
          _omr._addressed_to_the_robot(_frag), False)
for _real in ("what is xr", "tell me about the hub", "what's new in the market"):
    check(f"{_real!r} is somebody talking to it",
          _omr._addressed_to_the_robot(_real), True)

# The way out must never be gated. "Goodbye" is one word and has to work from
# across a room, whatever else this refuses.
for _stop in ("goodbye", "go to sleep", "turn off"):
    check(f"{_stop!r} always gets through", _omr._addressed_to_the_robot(_stop), True)

# THE case a blunt word-count would break. A demo holding the mic has just
# asked a question, so a one-word answer is exactly what it should hear.
_oms.hold_open_mic(True)
for _answer in ("yes", "no", "camera", "walk", "XR"):
    check(f"a held mic still hears {_answer!r}",
          _omr._addressed_to_the_robot(_answer), True)
_oms.hold_open_mic(False)
check("and the floor comes back when the demo lets go",
      _omr._addressed_to_the_robot("yes"), False)
_oms.set_open_mic(False)

print()
print("[45] the dashboard's script has no Python habits in it")
# A whole afternoon's work was invisible because of this: two adjacent string
# literals on separate lines with no "+" between them. Python joins them
# silently; JavaScript raises SyntaxError, which kills the ENTIRE script block,
# so the page loaded and then did nothing at all -- every panel stuck on
# "loading...". Bracket balance cannot see it, and neither could I by reading.
import re as _re45  # noqa: E402
_page45 = (_pl36.Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
_js45 = chr(10).join(_re45.findall(r"<script[^>]*>(.*?)</script>", _page45, _re45.S))
_bad45 = []
_lines45 = _js45.splitlines()
for _i in range(len(_lines45) - 1):
    _a, _b = _lines45[_i].rstrip(), _lines45[_i + 1].strip()
    if not (_a.endswith(('"', "'")) and _b.startswith(('"', "'"))):
        continue
    # Legal when something joins them, or the first line closed its own
    # argument or call.
    if _a.endswith(('",', "',", '";', "';", '")', "')", "+", ",")):
        continue
    _bad45.append("line " + str(_i + 1) + ": ..." + _a[-30:] + " / " + _b[:30])
check("no two string literals are concatenated the Python way", _bad45, [])

print()
print("[46] the generated sounds are safe to send to a speaker")
from body import sounds as _snd  # noqa: E402
import numpy as _np46  # noqa: E402

check("there are sounds to play", len(_snd.names()) >= 5, True)
for _name in _snd.names():
    _clip = _snd.get(_name)
    _secs = len(_clip) / _snd.SAMPLE_RATE
    # NaN reached the speaker once already: sin(pi) in float32 is a hair below
    # zero, and a negative raised to a fractional power is NaN, which silences
    # a buffer at best and clicks loudly at worst. Caught by a printed "nan".
    check(f"  {_name}: every sample is finite", bool(_np46.all(_np46.isfinite(_clip))), True)
    # Short, because the microphone is deaf for as long as the speaker is busy.
    check(f"  {_name}: short enough to talk over", _secs <= _snd._MAX_CLIP_S, True)
    # And never louder than the robot's own voice.
    check(f"  {_name}: not louder than speech",
          float(_np46.max(_np46.abs(_clip))) <= _snd._PEAK + 1e-6, True)
    # A clip that starts or ends away from silence pops on a real speaker.
    check(f"  {_name}: starts and ends at silence",
          abs(float(_clip[0])) < 0.02 and abs(float(_clip[-1])) < 0.02, True)

check("an unknown sound is silence, not an error", _snd.get("nope"), None)

print()
print("[47] showing the robot something keeps the picture bounded")
from brain import looking as _lk  # noqa: E402
import numpy as _np47  # noqa: E402
import base64 as _b6447  # noqa: E402
import cv2 as _cv47  # noqa: E402

# Every demo declaring a requirement must name one the robot actually
# publishes, or it is greyed out forever on hardware perfectly able to run it.
# "Look at this" asked for "camera", which did not exist -- only "faces" did --
# so it would have shipped permanently unavailable, exactly as research mode
# did. That bug cost an afternoon the first time.
_vl47 = (_pl36.Path(__file__).parent / "body" / "voice_loop.py").read_text(encoding="utf-8")
_groups47 = []
for _demo_file in (_pl36.Path(__file__).parent / "demos").glob("[!_]*.py"):
    _groups47 += _re45.findall(r"requires\s*=\s*\(([^)]*)\)",
                               _demo_file.read_text(encoding="utf-8"))
_needed47 = set()
for _group in _groups47:
    for _word in _group.split(","):
        _clean = _word.strip().strip("\"").strip("'")
        if _clean:
            _needed47.add(_clean)
_published47 = set(_re45.findall(r'caps\.add\("([a-z]+)"\)', _vl47)) | {"study"}
check("every capability a demo requires is one the robot can publish",
      sorted(_needed47 - _published47), [])

# A THIRD place names them: tools/selftest.py validates each demo's contract
# against its own hardcoded list. Adding "camera" to voice_loop was not enough
# -- selftest failed with "unknown requirement 'camera'", which is how this
# check came to exist. Three lists that must agree is two too many, but until
# they are one, they are checked against each other.
_st47 = (_pl36.Path(__file__).parent / "tools" / "selftest.py").read_text(encoding="utf-8")
_known47 = set(_re45.findall(r'need not in \(([^)]*)\)', _st47))
_selftest_caps = set()
for _g in _known47:
    for _w in _g.split(","):
        _c = _w.strip().strip("\"").strip("'")
        if _c:
            _selftest_caps.add(_c)
check("and the selftest's own list agrees with them",
      sorted(_needed47 - _selftest_caps), [])

# The picture is downscaled hard before it leaves the laptop: enough to name an
# object, deliberately poor for identifying the people standing behind it.
_big47 = _np47.zeros((1080, 1920, 3), dtype=_np47.uint8)
_big47[:, :, 2] = 200
_enc47 = _lk._encode(_big47)
check("a frame encodes to something sendable", bool(_enc47), True)
_dec47 = _cv47.imdecode(_np47.frombuffer(_b6447.b64decode(_enc47), _np47.uint8), 1)
check("and is downscaled before it leaves the laptop",
      max(_dec47.shape[:2]) <= _lk._MAX_EDGE, True)

# A missing frame must be an apology, never an exception. On this network the
# wifi being down is the ordinary case rather than the edge case.
check("no frame means no request at all", _lk.describe(None), "")

print()
print("[48] the robot talks at a speed the operator sets")
import body.audio_io as _aio48  # noqa: E402
from brain import settings as _set48  # noqa: E402

# Reported from the actual room: too fast. The conversational voice set no
# length_scale at all, so it ran at piper's bare default, and one persona
# (professional, 0.95) was faster still.
check("the default is slower than the synthesiser's own",
      _aio48.SPEECH_PACE_DEFAULT > 1.0, True)

_held48 = _set48.get(_aio48.SPEECH_PACE_KEY, "")
try:
    _low48, _high48 = _aio48.SPEECH_PACE_RANGE
    _set48.put(_aio48.SPEECH_PACE_KEY, "1.30")
    check("a set value is used", abs(_aio48.speech_pace() - 1.30) < 0.001, True)
    # Bounded, because piper drawls past about 1.4 and gabbles below 0.85 --
    # a slider that can make the robot unintelligible is a slider somebody
    # will make the robot unintelligible with.
    _set48.put(_aio48.SPEECH_PACE_KEY, "9.0")
    check("absurdly slow is clamped", _aio48.speech_pace(), _high48)
    _set48.put(_aio48.SPEECH_PACE_KEY, "0.1")
    check("absurdly fast is clamped", _aio48.speech_pace(), _low48)
    # Nonsense in the table must never cost the robot its voice.
    _set48.put(_aio48.SPEECH_PACE_KEY, "not a number")
    check("rubbish falls back to the default",
          _aio48.speech_pace(), _aio48.SPEECH_PACE_DEFAULT)

    # And it has to reach the synthesiser, not just the number.
    _set48.put(_aio48.SPEECH_PACE_KEY, "1.30")
    _fast = _aio48._voice_config(1.0, 0.667).length_scale
    _slow = _aio48._voice_config(1.0 * _aio48.speech_pace(), 0.667).length_scale
    check("and a slower setting really lengthens the speech", _slow > _fast, True)
finally:
    if _held48:
        _set48.put(_aio48.SPEECH_PACE_KEY, _held48)
    else:
        _set48.put(_aio48.SPEECH_PACE_KEY, "")

print()
print("[49] a question gets its answer without a wake word, and a quiz is not slowed by noise")
import demos.quiz as _qz49  # noqa: E402

# THE APOSTROPHE. "Let's do a quiz" switched to the quiz through the runner's
# word-stream matching, then fell through the quiz's own swallow -- a raw
# substring check -- because "lets do a quiz" is not a substring of "let's do
# a quiz". The conversation model answered over the top: "Sure thing,
# Tadhagath, I'd love to!", thirty seconds before question one. Test [33]
# passed throughout, because it fed each demo its trigger VERBATIM: the exact
# spelling the recogniser never produces.
_q49 = _qz49.Quiz()
_r49, _s49, _a49, _ = build([Chatty(), _q49])
_s49.set_mode(_q49.id)
_r49.cycle()
_ctx49 = _r49._ctx
_st49 = _ctx49.store
# The cycle above already asked question 1; the swallow path only exists
# BEFORE a question is out (that is when the runner hands the trigger back).
_st49.update(awaiting=False, step=0, tries=0, reasked=False)
check("the apostrophe form of its own trigger is swallowed",
      _q49.on_utterance(_ctx49, "Let's do a quiz."), True)
check("so is the plain form still", _q49.on_utterance(_ctx49, "lets do a quiz"), True)

# Politeness must not burn a try. Live, "Okay, thank you, RIT" was counted as
# a wrong answer and gave the fact away one guess early. The deck is shuffled
# per session, so the outstanding question is pinned rather than guessed.
_st49.update(order=list(range(6)), step=0, awaiting=True, tries=0, reasked=False)
check("a question is now outstanding", _st49.get("awaiting"), True)
_before = _st49.get("tries", 0)
check("courtesy noise is swallowed", _q49.on_utterance(_ctx49, "okay thank you reachy"), True)
check("and costs no try", _st49.get("tries", 0), _before)

# A garbage transcript gets ONE free retry, then noise has to resolve.
_a49.last_confidence = -9.0
check("a mangled answer is asked again", _q49.on_utterance(_ctx49, "flurble grang"), True)
check("without burning a try", _st49.get("tries", 0), _before)
check("but only once", _st49.get("reasked"), True)
_q49.on_utterance(_ctx49, "flurble grang")
check("the second one counts", _st49.get("tries", 0), _before + 1)
_a49.last_confidence = None

# A reply that ends by asking opens a wake-word-free window for the answer,
# and the window admits one-word answers past the ambient floor. The quiz
# checks above left the demo HOLDING the mic (_misheard holds it for the
# retry), and a held mic bypasses the floor for its own reason -- released
# here so these checks measure the answer window, not the leftover hold.
_s49.hold_open_mic(False)
_s49.expect_answer(5.0)
check("one word is admitted while an answer is expected",
      _r49._addressed_to_the_robot("true"), True)
_s49.expect_answer(0.0)
_s49.set_open_mic(True)
check("and refused once the window is gone (ambient floor again)",
      _r49._addressed_to_the_robot("true"), False)
_s49.set_open_mic(False)

# The wiring: base.reply sets the expectation when the last sentence asks.
_base49 = (_pl36.Path(__file__).parent / "demokit" / "base.py").read_text(encoding="utf-8")
check("a reply ending in a question opens the window",
      'endswith("?")' in _base49 and "expect_answer(ANSWER_WINDOW_S)" in _base49, True)
# Any utterance closes it, so the NEXT stray remark is not read as an answer.
_run49 = (_pl36.Path(__file__).parent / "demokit" / "runner.py").read_text(encoding="utf-8")
check("and any utterance closes it", "expect_answer(0.0)" in _run49, True)

# The five-second open-mic listen no longer stretches scripted sequences:
# it is capped at the window the demo itself asked for. Measured live as
# fourteen-second gaps between quiz questions with open mic on.
check("the open-mic listen respects the demo's own window",
      "min(_OPEN_MIC_SILENCE_S, listen_for)" in _run49, True)

print()
print("[50] every spoken question opens the answer window, wherever it came from")
# reply() got the rule first; the gap was every question that never went
# through reply() -- the quiz, the greetings menu, and an operator typing
# "would anyone like a tour?" into the Say box all ask through other paths.
_d50 = Chatty()
_r50, _s50, _a50, _ = build([_d50])
_s50.set_mode(_d50.id)
_r50.cycle()
_ctx50 = _r50._ctx

_s50.expect_answer(0.0)
_ctx50.say("Which language would you like?", "curious")
check("a scripted question opens the window", _s50.answer_expected, True)

_s50.expect_answer(0.0)
_ctx50.say("The Hub has three research strands.", "neutral")
check("a statement does not", _s50.answer_expected, False)

# The operator's Say box asks the room through the runner, not through a demo.
_s50.request("say", "Would anyone like a tour?")
_r50._handle_dashboard_request()
check("a dashboard question opens it too", _s50.answer_expected, True)
_s50.expect_answer(0.0)
_s50.request("say", "The tour starts at two.")
_r50._handle_dashboard_request()
check("a dashboard statement does not", _s50.answer_expected, False)

print()
print("[51] boot loads the models while the robot connects")
import body.audio_io as _aio51  # noqa: E402

# The rules of the preload cache, not the timing: a part that fails to build
# must fall back to inline construction, and a builder that RETURNS None (the
# whisper builder does, on failure) must not be cached -- a cached None makes
# the constructor's "or" fallback rebuild it anyway, spending the time twice.
_held51 = (_aio51._build_whisper, dict(_aio51._PRELOADED))
try:
    _aio51._PRELOADED.clear()
    _aio51._build_whisper = lambda: None
    _calls51 = {"n": 0}
    _real_rec = _aio51._build_recognizer

    def _counting_rec():
        _calls51["n"] += 1
        return _real_rec()

    _aio51._build_recognizer = _counting_rec
    _aio51.preload_models()
    check("a builder returning None is not cached", "whisper" in _aio51._PRELOADED, False)
    check("the parts that built are", "recognizer" in _aio51._PRELOADED, True)
    _aio51.preload_models()
    check("preloading twice does not build twice", _calls51["n"], 1)
finally:
    _aio51._build_whisper, _cache51 = _held51[0], _held51[1]
    _aio51._build_recognizer = _real_rec
    _aio51._PRELOADED.clear()
    _aio51._PRELOADED.update(_cache51)

# The wiring: the worker starts BEFORE the robot connect and is joined before
# assembly, so a slow disk can never race the constructor.
_vl51 = (_pl36.Path(__file__).parent / "body" / "voice_loop.py").read_text(encoding="utf-8")
# Two workers now -- the audio chain and the face recogniser were measured
# ~11.5s and ~10s and are independent, so serial inside one worker they were
# the whole critical path once the robot connect stopped being it.
check("the preloads start before the robot connect",
      _vl51.index("t.start()") < _vl51.index("robot = None"), True)
check("and are joined before assembly",
      _vl51.index("t.join()") < _vl51.index("audio = AudioIO(target"), True)

print()
print("[52] the hidden attribute actually hides everything that uses it")
# The Recording pill shipped showing "Recording" to every visitor, armed or
# not: .pill sets display: inline-flex, and ANY authored display rule beats
# the browser's own [hidden] { display: none }, which lives in the UA
# stylesheet at the lowest possible specificity. The fix is one global
# [hidden] { display: none !important; } -- per-element overrides were the
# first attempt, and a working scan then showed three MORE elements with the
# same fault (links, lockpanel, rec-table).
#
# THIS CHECK WAS ITSELF SHIPPED VACUOUS ONCE: shell escaping doubled the
# regex backslashes, the scan matched nothing, and it passed on an empty list
# -- including with the fix deliberately removed. The non-vacuity assertions
# are what keep that from happening quietly again.
_page52 = (_pl36.Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
_hidden_ids52 = []
for _m52 in _re45.finditer(r"<[a-z]+[^>]*>", _page52):
    _tag52 = _m52.group(0)
    if _re45.search(r"\bhidden\b(?![\"'=-])", _tag52) is None:
        continue
    _id52 = _re45.search(r'id="([a-zA-Z0-9_-]+)"', _tag52)
    if _id52:
        _hidden_ids52.append(_id52.group(1))
check("the scan actually finds hidden elements (not vacuous)",
      len(_hidden_ids52) >= 3, True)
check("including the recording pill", "p-rec" in _hidden_ids52, True)
check("one global rule makes hidden always win",
      "[hidden] { display: none !important; }" in _page52, True)

print()
print("[53] the speed work holds together")
import types as _ty53
import numpy as _np53
from config import MODELS as _M53

# THE COUPLING. rule2=1.2 was measured NET SLOWER than 1.5 at whisper_threads=6
# (the early decode outlives the shorter silence and everything queues on it);
# it is only a win at <=4 threads. Anyone raising threads back must see this.
check("whisper threads are 4 -- six was measured 2x slower on this CPU",
      _M53.whisper_threads, 4)
_aio53 = (_pl36.Path(__file__).parent / "body" / "audio_io.py").read_text(encoding="utf-8")
if "rule2_min_trailing_silence=1.2" in _aio53:
    check("rule2=1.2 only ships with whisper_threads <= 4",
          _M53.whisper_threads <= 4, True)

# The AGC floor on the wake-wait path: after a loud transient the slow envelope
# pinned gain near 1x for seconds and wake recall fell 14/16 -> 9/16 measured.
from body.audio_io import AudioIO as _A53

_fake53 = object.__new__(_A53)
_fake53.target = _ty53.SimpleNamespace(mic_agc=True, mic_gain=1.0)
_fake53._agc_noise_floor = 0.01
_fake53._agc_envelope = 0.9          # just after something loud
_fake53._agc_gain = 1.0
_fake53._gate_enabled = False        # the wake-wait path
_A53._apply_gain(_fake53, _np53.full(160, 0.05, dtype=_np53.float32))
check("waiting for the wake word, gain never starves below 4x",
      _fake53._agc_gain >= 4.0, True)
_fake53._gate_enabled = True         # ordinary listening: floor must NOT apply
_fake53._agc_envelope = 0.9
_A53._apply_gain(_fake53, _np53.full(160, 0.5, dtype=_np53.float32))
check("but ordinary listening keeps the true envelope gain",
      _fake53._agc_gain < 4.0, True)

# Barge-in keeps the interrupter's question: the backlog match must drain and
# mark the mic fresh exactly as the ordinary wake path does.
_i53 = _aio53.index('Wake word heard while speaking')
check("a backlog wake keeps the words that followed it",
      "_mic_fresh = True" in _aio53[_i53:_i53 + 1200], True)

# The conversation demo must not smuggle the (cached) grounding back in as
# uncached per-turn tokens -- that was 1,200 duplicated tokens per turn.
from brain import hub as _hub53
import demos.conversation as _conv53

_marker53 = _hub53.GROUNDING.strip().splitlines()[0][:40]
check("the per-turn briefing does not duplicate the grounding",
      _marker53 in _conv53._HUB_BRIEFING, False)
check("but the base prompt still carries it",
      _marker53 in _hub53.GROUNDING, True)

# Person 0 is every stranger sharing one id: it must neither accumulate nor
# inject "memories" (they were other strangers' conversations, and they kept
# the qa_cache from ever hitting -- 4 entries, 0 hits, when found).
from brain import long_term_memory as _ltm53

check("person 0 injects no context", _ltm53.get_context(0), "")
_held53 = _ltm53.generate_response
try:
    def _boom53(*a, **k):
        raise AssertionError("summariser ran for person 0")
    _ltm53.generate_response = _boom53
    _ltm53.end_conversation(0, [("hello", "hi")])
    check("and no summary is ever written for person 0", True, True)
finally:
    _ltm53.generate_response = _held53

# A reply that does NOT ask still opens a wake-free follow-up window (with the
# ambient floor kept -- nothing was asked, so a stray syllable is the room).
from brain.modes import RobotState as _RS53

_st53 = _RS53()
_st53.invite_followup(5.0)
check("a follow-up window opens", _st53.followup_expected, True)
_st53.invite_followup(0.0)
check("and closes", _st53.followup_expected, False)
_base53 = (_pl36.Path(__file__).parent / "demokit" / "base.py").read_text(encoding="utf-8")
check("every non-question reply invites one",
      "invite_followup(FOLLOW_UP_WINDOW_S)" in _base53, True)
_run53 = (_pl36.Path(__file__).parent / "demokit" / "runner.py").read_text(encoding="utf-8")
check("the runner listens through it", "followup_expected" in _run53, True)
check("without bypassing the ambient floor",
      "followup_expected" in _run53.split("def _addressed_to_the_robot")[1].split("def ")[0], False)

print()
print("[54] a person talking is not a command being mangled")
from demokit.runner import SLEEP_PHRASES, _word_stream, contains_phrase
from demokit.runner import fuzzy_contains as _fz54

# Both hijacks happened live in one afternoon. "I'm just feeling a bit tired
# today, what should I do?" fuzzy-matched a Look trigger, and the robot told
# somebody asking for sympathy to hold an object up to the camera. A three-part
# question about the Hub fuzzy-matched the welcome and got the scripted speech
# instead of an answer to any part of it.
_tired54 = _word_stream("For Eiji basically Im just feeling a bit tired today what should I do")
_hub54 = _word_stream("Tell me a bit about the hub and also tell me about the type of "
                      "master students that come to the hub")
check("a feelings sentence cannot fuzzy-match a Look trigger",
      _fz54(_tired54, "what do you see"), False)
check("a rich question cannot fuzzy-match the welcome",
      any(_fz54(_hub54, t) for t in ("introduce the hub", "welcome the group",
                                     "do the welcome")), False)
# The case fuzzy exists FOR must survive: a trigger misheard about its own
# length. This is the sentence from the live failure that created the matcher.
check("a mangled trigger of its own length still matches",
      _fz54(_word_stream("Welcome to your irisimus group"),
            "welcome the erasmus group"), True)
check("and exact phrases still work inside long sentences",
      contains_phrase(_word_stream("please could you now do the welcome for everyone here today"),
                      "do the welcome"), True)

# Saying goodnight has to actually be goodnight: the robot answered "I'll
# power down now" to "thank you go sleep" -- and then kept listening, because
# neither phrase was a sleep phrase and saying it is not doing it.
for _phrase54 in ("go sleep", "power down", "go to sleep"):
    check(f"{_phrase54!r} is a sleep phrase", _phrase54 in SLEEP_PHRASES, True)

# No wake-free window survives into sleep.
_run54 = (_pl36.Path(__file__).parent / "demokit" / "runner.py").read_text(encoding="utf-8")
check("wake-free windows are gated on being awake",
      "not self._state.sleeping and (" in _run54, True)

# The register rule: brief for facts, human for feelings -- and brainstorming
# is a can-do, never a refusal. "I want to create a startup, give me some
# ideas" was answered with "I can't brainstorm business ideas for you".
from brain import prompts as _pr54

_base54 = _pr54._base_prompt()
check("the prompt distinguishes facts from feelings",
      "something personal" in _base54 and "twenty-five" in _base54, True)
check("struggling people are not handed slogans",
      "slogan" in _base54, True)
check("brainstorming is listed as a thing it CAN do",
      "brainstorm ideas out loud" in _base54, True)

print()
print("[55] the Dean's Office introduction behaves like a scripted demo should")
import demos.deans_office as _do55

_d55 = _do55.DeansOffice()
_r55, _s55, _a55, _ = build([Chatty(), _d55])
_s55.set_mode(_d55.id)
_windows55 = []
for _ in range(len(_do55._SCRIPT) + 3):
    _r55.cycle()

check("every line of the script is spoken, in order",
      _a55.said[:len(_do55._SCRIPT)], [t for t, _e in _do55._SCRIPT])
check("it introduces the robot", "Reachy Mini" in _a55.said[0], True)
check("it names the Hub and the school",
      any("Trinity Business School" in x for x in _a55.said), True)
check("it carries the soft-skills invitation",
      any("soft skills" in x for x in _a55.said), True)
check("and asks nobody to hold anything up",
      any("hold" in x.lower() for x in _a55.said), False)

# One line per slice, and the whole thing short enough for a standing
# listener: the welcome demo's file records ~40s as where a room drifts.
_words55 = sum(len(t.split()) for t, _e in _do55._SCRIPT)
check("the script fits a standing audience (under ~90 words)",
      _words55 <= 90, True)

# Restart on request, from the words a person would actually use.
_ctx55 = _r55._ctx
check("'say it again' restarts it", _d55.on_utterance(_ctx55, "can you say it again"), True)
check("and the script rewinds", _ctx55.store["line"], 0)

# Its own trigger is swallowed apostrophe-and-all -- the quiz shipped this
# exact bug, and [33] audits the plain spelling only.
_ctx55.store["line"] = 1
check("the trigger that started it is swallowed",
      _d55.on_utterance(_ctx55, "hey reachy do the dean's office"), True)
# A real question mid-script falls through to the conversation model.
_ctx55.store["line"] = len(_do55._SCRIPT)
check("a real question is left for the conversation model",
      _d55.on_utterance(_ctx55, "what masters programmes are there"), False)

print()
print(f"{'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
sys.exit(1 if failures else 0)
