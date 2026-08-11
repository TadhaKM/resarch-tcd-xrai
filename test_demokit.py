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

# Saved so the real ones can be put back. Replacing a module attribute lasts
# for the rest of the file, and a later section testing the actual database
# against the stub reported the name it had stored as missing.
_REAL_DB = {"get_person_name": _db.get_person_name, "rename_person": _db.rename_person}

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


def brief_for(persona_id, demo_id="conversation", style=None):
    _seen.clear()
    st = RobotState()
    st.set_persona(persona_id)
    motion = FakeMotion()
    ctx = DemoContext(
        audio=FakeAudio(), motion=motion, tracker=None,
        state=st, demo_id=demo_id, store={},
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
brief, _ = brief_for("friendly", demo_id="personality")
check("and the personality demo, which has its own", "warmly" in brief.lower(), False)

a, _ = brief_for("friendly")
b, _ = brief_for("consultant")
# Answers are cached by a digest of this text, so a persona that did not reach
# it would have replayed one persona's answer in another's voice.
check("cached answers are per persona", _qa._context_digest(a) != _qa._context_digest(b), True)

check("an unknown persona means none, not Professional", RobotState().set_persona("nonsense"), "")

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
           "brainstorm": "professional"}
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
print("[17] audio from the wrong thread is refused, not silently interleaved")
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
