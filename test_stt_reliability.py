"""Correctness check for STT + wake word: synthesize test phrases with piper,
feed the resulting audio directly into the recognizer/spotter, and compare.

This is a synthetic round-trip (no live human mic input) -- it validates that
the ASR model, its config, and the wake-word tokens are wired correctly and
transcribe real (piper-generated) speech audio correctly. It does not
validate live-mic acoustics (room noise, mic placement, etc.).
"""

import numpy as np
from scipy.signal import resample

from body.audio_io import AudioIO
from config import MODELS, SIMULATION

TEST_PHRASES = [
    "Hello there.",
    "What is the weather like today?",
    "Turn left please.",
    "Can you tell me a joke?",
    (
        "I would like you to remember that my favorite color is blue and that "
        "I have a meeting scheduled for three o'clock this afternoon."
    ),
]

# The model spells some words in British English (LibriSpeech quirk, not an
# error) -- treat these as matches rather than misses.
_SPELLING_VARIANTS = {"favorite": "favourite", "color": "colour"}

WAKE_PHRASE = "Hey Reachy"


def synthesize(audio: AudioIO, text: str) -> np.ndarray:
    """Synthesize text with piper and return mono float32 samples at the ASR's sample rate."""
    chunks = list(audio._voice.synthesize(text))
    samples = np.concatenate([c.audio_float_array for c in chunks])
    native_rate = chunks[0].sample_rate
    if native_rate != MODELS.asr_sample_rate:
        samples = resample(samples, int(len(samples) * MODELS.asr_sample_rate / native_rate))
    return samples.astype(np.float32)


def normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace()).split()


def main() -> None:
    audio = AudioIO(SIMULATION)

    print("=== Wake word ===")
    # piper's synthesis is itself stochastic (confirmed: re-synthesizing the
    # same text yields measurably different audio each time), so a single
    # draw isn't a meaningful reliability signal -- sample several draws.
    trials = 10
    detections = sum(audio.detect_wake_word_offline(synthesize(audio, WAKE_PHRASE)) for _ in range(trials))
    print(f"'{WAKE_PHRASE}' -> detected {detections}/{trials} independent synthesized draws")

    print("\n=== False-positive check (test phrases should NOT trigger wake word) ===")
    false_positives = 0
    for phrase in TEST_PHRASES:
        samples = synthesize(audio, phrase)
        if audio.detect_wake_word_offline(samples):
            print(f"FALSE POSITIVE on: {phrase}")
            false_positives += 1
    print(f"False positives: {false_positives}/{len(TEST_PHRASES)}")

    print("\n=== STT transcription ===")
    passed = 0
    for phrase in TEST_PHRASES:
        samples = synthesize(audio, phrase)
        transcript = audio.transcribe_offline(samples)
        expected_words = normalize(phrase)
        got_words = set(normalize(transcript))
        matched = sum(
            1 for w in expected_words if w in got_words or _SPELLING_VARIANTS.get(w) in got_words
        )
        ok = matched >= len(expected_words) - 1  # allow one word miss
        passed += ok
        print(f"{'OK' if ok else 'MISS'} expected='{phrase}'")
        print(f"        got='{transcript}'")

    print(f"\n{passed}/{len(TEST_PHRASES)} phrases transcribed correctly.")


if __name__ == "__main__":
    main()
