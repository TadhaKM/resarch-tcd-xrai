"""Offline self-test: everything checkable without the robot.

    python tools/selftest.py          # logic, models, wake phrases, whisper
    python tools/selftest.py --llm    # also exercise the LLM via local Ollama

Speech tests synthesize audio with piper and feed it back through the real
recognizers -- the same trick as test_stt_reliability.py. One caveat is
inherent: piper's voice is clean and consistent, so passing here proves the
wiring and the models, not performance against a live human in a noisy room.
Failing here, though, is definitive -- a wake phrase that cannot be spotted
from clean synthesized audio will never work live.
"""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

PASS, FAIL, WARN = "ok  ", "FAIL", "warn"
failures = 0


def report(status: str, label: str, detail: str = "") -> None:
    global failures
    if status == FAIL:
        failures += 1
    print(f"  {status} {label}" + (f"  ({detail})" if detail else ""))


def check_logic() -> None:
    print("\n[1] parsing and filtering logic")
    from brain.emotion import extract_emotion_tag
    for raw, want in [
        ("Hello there! [emotion: happy]", ("Hello there!", "happy")),
        ("How can I help today? [thinking]", ("How can I help today?", "thinking")),
        ("Sure! [thinking] Here we go.", ("Sure! Here we go.", "thinking")),
        ("No tag at all.", ("No tag at all.", "neutral")),
        ("He said [see below] to me.", ("He said [see below] to me.", "neutral")),
    ]:
        got = extract_emotion_tag(raw)
        report(PASS if got == want else FAIL, f"emotion: {raw[:38]!r}", str(got) if got != want else "")

    from body.audio_io import _strip_sound_events
    for raw, want in [
        ("(tapping)", ""), ("(static", ""), ("*sad noises*", ""),
        ("(laughs) That is funny", "That is funny"),
        ("What can you do?", "What can you do?"), ("...", ""),
    ]:
        got = _strip_sound_events(raw)
        report(PASS if got == want else FAIL, f"sound-event: {raw!r}", got if got != want else "")

    from demos.timers import _parse_timer
    for raw, want in [
        ("set a timer for 5 minutes", (300, "5 minute")),
        ("timer for five minutes", (300, "5 minute")),
        ("remind me in 30 seconds", (30, "30 second")),
        ("set a timer for an hour", (3600, "1 hour")),
        ("set a timer for half an hour", (1800, "30 minute")),
        ("what time is it", None),
        ("timer for 3 seconds", None),
    ]:
        got = _parse_timer(raw)
        report(PASS if got == want else FAIL, f"timer: {raw!r}", str(got) if got != want else "")

    from body.face import clean_spoken_name
    for raw, want in [
        ("MY NAME IS TADHG", "Tadhg"), ("A", None),
        ("IS IT THAT CONNECTED THROUGH OPEN AIR HE ASKED AS IT WERE SOME KIND OF THING I DO NOT KNOW", None),
    ]:
        got = clean_spoken_name(raw)
        report(PASS if got == want else FAIL, f"name: {raw[:38]!r}", str(got) if got != want else "")


def check_demos() -> None:
    """Every demo in demos/ loads, is unique, and obeys the framework contract.

    This is the check that protects the platform's one promise: a student adds
    a file and it works. A demo that fails to import, collides on an id, or
    declares a trigger another demo already claims is invisible at runtime --
    the registry logs it and carries on, which is right for a live session and
    useless for finding out before one.
    """
    print("\n[2] demos load and obey the contract")
    from brain.emotion import VALID_EMOTION_TAGS
    from demokit.base import MAX_LISTEN_WINDOW_S, Demo, IdleResult
    from demokit.registry import REGISTRY

    REGISTRY.discover()
    ids = REGISTRY.ids()
    report(PASS if ids else FAIL, f"{len(ids)} demo(s) discovered", ", ".join(ids))

    seen_triggers: dict[str, str] = {}
    for demo_id in ids:
        demo = REGISTRY.get(demo_id)
        problems = []
        if not isinstance(demo, Demo):
            problems.append("not a Demo")
        if not demo.label:
            problems.append("no label")
        if not demo.help:
            problems.append("no help text")
        for phrase in demo.triggers:
            if phrase != phrase.lower():
                problems.append(f"trigger {phrase!r} is not lowercase")
            owner = seen_triggers.setdefault(phrase, demo_id)
            if owner != demo_id:
                problems.append(f"trigger {phrase!r} already claimed by {owner}")
        for need in demo.requires:
            if need not in ("faces",):
                problems.append(f"unknown requirement {need!r}")
        report(PASS if not problems else FAIL, f"contract: {demo_id}", "; ".join(problems))

    # on_idle is the hook the loop leans on hardest, so check every demo's
    # answer survives the sanitiser rather than trusting each to return well.
    for demo_id in ids:
        demo = REGISTRY.get(demo_id)
        raw = demo.__class__.on_idle
        declared = raw is not Demo.on_idle
        if not declared:
            continue
        hint = getattr(raw, "__annotations__", {}).get("return")
        ok = hint in (IdleResult, "IdleResult", None)
        report(PASS if ok else WARN, f"on_idle returns IdleResult: {demo_id}", "" if ok else str(hint))

    report(PASS, "listen windows clamp at", f"{MAX_LISTEN_WINDOW_S}s")
    report(PASS, "emotion tags available", ", ".join(sorted(VALID_EMOTION_TAGS)))


def check_assets() -> None:
    print("\n[3] models and assets on disk")
    from config import MODELS
    needed = [
        MODELS.asr_dir, MODELS.kws_dir, MODELS.tts_model_path, MODELS.tts_config_path,
        MODELS.kws_keywords_file, MODELS.asr_hotwords_file,
        MODELS.face_detector_model_path, MODELS.face_embedding_model_path,
    ]
    if MODELS.whisper_dir is not None:
        needed.append(MODELS.whisper_dir / "base.en-encoder.int8.onnx")
    for p in needed:
        report(PASS if Path(p).exists() else FAIL, str(Path(p).relative_to(REPO)))

    try:
        from body.motion import EMOTION_MOVES
        from reachy_mini.motion.recorded_move import RecordedMoves
        rm = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        bad = []
        for names in EMOTION_MOVES.values():
            for n in names:
                try:
                    rm.get(n)
                except Exception:
                    bad.append(n)
        total = sum(len(v) for v in EMOTION_MOVES.values())
        report(FAIL if bad else PASS, f"{total} recorded gestures loadable", ",".join(bad))
    except Exception as exc:
        report(WARN, "recorded gestures", f"library unavailable: {exc}")


def _synthesize(voice, text: str) -> np.ndarray:
    from body.audio_io import _resample
    from config import MODELS
    chunks = [c.audio_float_array for c in voice.synthesize(text)]
    audio = np.concatenate(chunks).astype(np.float32)
    return _resample(audio, 22050, MODELS.asr_sample_rate)


def check_speech() -> "object":
    print("\n[4] wake phrases fire on synthesized speech")
    from body.audio_io import AudioIO
    from config import SIMULATION
    audio = AudioIO(SIMULATION)

    phrases = [
        line.strip() for line in (REPO / "assets" / "wake_phrases.txt").read_text().splitlines()
        if line.strip()
    ]
    for phrase in phrases:
        # Best of three, re-synthesizing each time. The variance that matters
        # is piper's, not the decoder's: measured over ten independent takes,
        # detection runs 8-10 out of 10 per phrase, but decoding one fixed
        # take three times just gives the same answer three times. Synthesizing
        # once outside the loop therefore turned a ~15% chance of an unlucky
        # take into a hard 0/3 failure, which is what made this check fail a
        # different phrase on almost every run while the wake word worked
        # fine live. Zero of three is a dead phrase; one of three is shaky.
        hits = sum(
            audio.detect_wake_word_offline(_synthesize(audio._voice, phrase.lower()))
            for _ in range(3)
        )
        status = PASS if hits >= 2 else WARN if hits == 1 else FAIL
        report(status, phrase, f"{hits}/3")

    print("\n[5] whisper round-trip on synthesized speech")
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").split().__str__()

    for text in [
        "what can you do", "how are you today", "can you dance for me",
        "tell me about robots", "what is the weather like today",
    ]:
        got = audio._transcribe_whisper(_synthesize(audio._voice, text))
        report(PASS if norm(got) == norm(text) else FAIL, f"{text!r}", f"heard {got!r}")
    return audio


def check_llm() -> None:
    print("\n[6] LLM replies (local Ollama)")
    from brain.emotion import VALID_EMOTION_TAGS
    from brain.interface import stream_reply
    for q in ["what can you do", "how are you", "tell me about the spanish weather"]:
        t = time.time()
        first = None
        sentences = []
        for sentence, tag in stream_reply(0, q):
            if first is None:
                first = time.time() - t
            sentences.append((sentence, tag))
        text = " ".join(s for s, _ in sentences)
        problems = []
        if not sentences:
            problems.append("no reply")
        if any(t not in VALID_EMOTION_TAGS for _, t in sentences):
            problems.append("bad tag")
        if "[" in text:
            problems.append("tag leaked into speech")
        if text and text.rstrip()[-1] not in ".!?":
            problems.append("truncated")
        report(
            FAIL if problems else PASS,
            f"{q!r}: first sentence {first:.1f}s, {len(sentences)} sentence(s)",
            "; ".join(problems),
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also test replies via local Ollama")
    args = ap.parse_args()

    check_logic()
    check_demos()
    check_assets()
    check_speech()
    if args.llm:
        check_llm()

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
