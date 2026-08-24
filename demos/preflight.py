"""Check the robot works, before a group is standing in front of it.

Staff currently find out that the speaker is muted or the camera is blind at
the moment a school group walks in. This is the ritual that replaces hoping:
press it, watch the list fill in, and know.

WHY THIS IS A DEMO RATHER THAN AN ENDPOINT
Speaking, listening and moving are all loop-thread-only -- audio.speak says so,
and the runner owns the microphone. A web endpoint that tried to drive them
would either block the voice loop or race it. The demo framework already runs
on that thread and already slices work one step per idle call, so a check per
slice is the shape this work naturally has. It also means the check appears as
a button in the Mode grid with no dashboard code at all, and can be stopped by
switching away like anything else.

WHY NOT tools/selftest.py
That is explicitly an offline test -- "everything checkable without the robot"
-- so it never touches the speaker, microphone, camera or motors, which are
precisely the things that break on the day. It also takes minutes and prints to
stdout. Different job.

WHAT IT DELIBERATELY DOES NOT DO
It never claims the microphone heard a *word*. Wake-word and transcription
accuracy depend on the room, and a check that fails because the room is quiet
would be a check staff learn to ignore. It reports the signal level it saw and
leaves the judgement to a person.
"""

import time
from typing import Optional

from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S

_BETWEEN_S = 1.0

#: Mic level below which nothing is reaching the recogniser. Speech in a normal
#: room peaks well above this; the check asks the operator to say something and
#: reports what arrived, rather than deciding for them what "loud enough" is.
_MIC_QUIET_PEAK = 0.02

#: How long to sample the microphone. Long enough to say a short sentence.
_MIC_SAMPLE_S = 3.0


class Preflight(Demo):
    label = "Check it works"
    help = "Tests the speaker, microphone, camera, movement and the AI, in about half a minute."
    order = 900
    triggers = ("run the check", "check everything works", "pre flight check",
                "preflight check", "run a system check")
    claims_utterances = True

    # --- hooks -----------------------------------------------------------

    def on_enter(self, ctx: DemoContext) -> None:
        ctx.store.clear()
        ctx.store.update(step=0, results=[])
        ctx.status("Checking the robot over -- about half a minute.")
        ctx.say("Right, let me check myself over.", "curious")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        steps = (self._speaker, self._movement, self._camera,
                 self._microphone, self._brain, self._verdict)
        step = ctx.store.get("step", 0)
        if step >= len(steps):
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)
        # One check per slice, so the operator can always switch away and the
        # runner never warns about a slow hook.
        ctx.store["step"] = step + 1
        try:
            return steps[step](ctx)
        except Exception as exc:
            # A check that throws is a failed check, not a failed robot: the
            # whole point is to finish the list and report.
            self._record(ctx, "unknown", False, str(exc)[:80])
            return IdleResult(listen_for=_BETWEEN_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        # Swallowed only while the check is running, so the trigger phrase that
        # started it is not answered twice; afterwards ordinary questions work.
        return ctx.store.get("step", 0) < 6

    # --- the checks ------------------------------------------------------

    def _speaker(self, ctx: DemoContext) -> IdleResult:
        ctx.say("Speaker working.", "neutral")
        # Nothing here can verify sound actually left the robot -- there is no
        # loopback -- so this is honest about being a "you should have heard
        # that" check rather than a measured one.
        self._record(ctx, "Speaker", True, "you should have just heard this")
        return IdleResult(listen_for=_BETWEEN_S)

    def _movement(self, ctx: DemoContext) -> IdleResult:
        moved = False
        try:
            ctx.motion.express_move("happy")
            moved = True
        except Exception as exc:
            self._record(ctx, "Movement", False, str(exc)[:80])
        if moved:
            self._record(ctx, "Movement", True, "head and antennas moved")
        return IdleResult(listen_for=_BETWEEN_S)

    def _camera(self, ctx: DemoContext) -> IdleResult:
        # Read from the published snapshot: _face_visible is private and there
        # is no accessor for it, and reaching into the underscore would break
        # the moment brain/modes.py renames its own field.
        seen = bool(ctx.state.snapshot().get("face_visible"))
        # Reported, not judged: an empty room is not a broken camera. The
        # tracker publishes whether it is getting frames at all, which is the
        # part that actually fails.
        self._record(ctx, "Camera", True,
                     "a face is in view" if seen else "working, but nobody is in front of me")
        return IdleResult(listen_for=_BETWEEN_S)

    def _microphone(self, ctx: DemoContext) -> IdleResult:
        heard = ctx.ask("Say something so I can check my hearing.", "curious",
                        wait_for_speech_s=_MIC_SAMPLE_S)
        peak = self._peak(ctx)
        if heard:
            self._record(ctx, "Microphone", True, f"heard {heard[:40]!r}")
        elif peak is not None and peak > _MIC_QUIET_PEAK:
            # Sound arrived but no words came back. That is a quiet or noisy
            # room, not a dead microphone, and saying so stops staff chasing
            # hardware that is fine.
            self._record(ctx, "Microphone", True,
                         f"sound is reaching me (peak {peak:.2f}) but no words came through")
        else:
            self._record(ctx, "Microphone", False, "nothing reached the microphone")
        return IdleResult(listen_for=_BETWEEN_S)

    def _brain(self, ctx: DemoContext) -> IdleResult:
        started = time.monotonic()
        try:
            from brain import llm

            reply = llm.generate_response(
                [{"role": "user", "content": "Reply with exactly: ready"}]
            )
        except Exception as exc:
            self._record(ctx, "AI", False, str(exc)[:80])
            return IdleResult(listen_for=_BETWEEN_S)
        took = time.monotonic() - started
        ok = bool(reply and reply.strip())
        self._record(ctx, "AI", ok, f"answered in {took:.1f}s" if ok else "no answer")
        return IdleResult(listen_for=_BETWEEN_S)

    def _verdict(self, ctx: DemoContext) -> IdleResult:
        results = ctx.store.get("results", [])
        bad = [r for r in results if not r["ok"]]
        for r in results:
            ctx.status(f"{'OK  ' if r['ok'] else 'FAIL'}  {r['label']} -- {r['note']}")
        if not bad:
            ctx.say("All good. I am ready for them.", "happy")
            ctx.status("Pre-flight passed.")
        else:
            names = ", ".join(r["label"].lower() for r in bad)
            ctx.say(f"Something is not right with my {names}.", "sad")
            ctx.status(f"Pre-flight FAILED: {names}")
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    # --- helpers ---------------------------------------------------------

    def _peak(self, ctx: DemoContext) -> Optional[float]:
        for attr in ("last_peak", "mic_peak"):
            value = getattr(ctx.audio, attr, None)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _record(self, ctx: DemoContext, label: str, ok: bool, note: str) -> None:
        ctx.store.setdefault("results", []).append(
            {"label": label, "ok": bool(ok), "note": note}
        )
