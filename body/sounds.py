"""Short sound effects, synthesized rather than shipped.

A fanfare when a group sweeps the quiz, a chime when it finishes checking
itself over. Small things, and the difference between a robot that answers and
one that reacts.

WHY THEY ARE GENERATED AND NOT AUDIO FILES
Three reasons, in order of how much they mattered:

  - Licensing. Any recording good enough to want is somebody's work, and a
    robot in a university foyer playing a clip whose licence nobody checked is
    a small problem waiting for the wrong visitor.
  - No decoder. This environment has `wave` and numpy and nothing else -- no
    ffmpeg, no soundfile, no pydub -- so an MP3 could not be played even if the
    licence were clear, and adding a decoder for six short noises is a poor
    trade.
  - Nothing binary in git. These are a few lines of arithmetic that reproduce
    identically on any machine, rather than half a megabyte of WAVs nobody can
    diff or review.

They sound synthesized, which is right: this is a robot, and a chiptune
flourish reads as the robot being pleased rather than as a stock library.

EVERY CLIP IS SHORT ON PURPOSE
The microphone hears whatever the speaker plays. That is already why the robot
mishears its own speech, and a sound long enough to talk over is a sound long
enough to deafen it. Nothing here runs past about two seconds, and
_MAX_CLIP_S enforces it rather than trusting the author of the next one.
"""

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

#: Everything is generated at this rate and resampled on the way out, exactly
#: as piper's output is.
SAMPLE_RATE = 22050

#: Nothing may be longer than this. A clip the robot cannot talk over is a clip
#: that takes the room away from whoever is running the visit, and the mic is
#: deaf for its whole duration.
_MAX_CLIP_S = 2.5

#: Headroom. Piper's speech sits well below full scale, and a sound effect that
#: arrives twice as loud as the robot's voice is startling in a quiet foyer.
_PEAK = 0.35


def _envelope(n: int, attack: float = 0.01, release: float = 0.25) -> np.ndarray:
    """A gentle rise and fall, so a clip never starts or ends on a click.

    A raw sine cut at a zero-crossing still clicks on most speakers; cut
    anywhere else it always does. This is the difference between a chime and a
    pop followed by a chime.
    """
    env = np.ones(n, dtype=np.float32)
    a = max(1, int(SAMPLE_RATE * attack))
    r = max(1, int(SAMPLE_RATE * release))
    if a < n:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r < n:
        env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def _tone(freq: float, seconds: float, *, harmonics: int = 3,
          attack: float = 0.01, release: float = 0.2) -> np.ndarray:
    """One note. Harmonics because a bare sine sounds like a hearing test."""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    wave = np.zeros(n, dtype=np.float32)
    for h in range(1, harmonics + 1):
        # Each harmonic quieter than the last, which is roughly what a struck
        # or plucked thing does and is why this reads as an instrument.
        wave += (1.0 / h) * np.sin(2.0 * math.pi * freq * h * t)
    return (wave / max(harmonics, 1)) * _envelope(n, attack, release)


def _sequence(notes: list[tuple[float, float]], gap: float = 0.0) -> np.ndarray:
    """Notes one after another, as (frequency, seconds)."""
    parts = []
    for freq, seconds in notes:
        parts.append(_tone(freq, seconds))
        if gap:
            parts.append(np.zeros(int(SAMPLE_RATE * gap), dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _noise_swell(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Filtered noise that rises and falls -- applause, roughly.

    A moving average is the whole filter. It is not convincing applause and is
    not trying to be; it reads as "a pleased noise", which is the job.
    """
    n = int(SAMPLE_RATE * seconds)
    raw = rng.normal(0.0, 1.0, n).astype(np.float32)
    window = 12
    smoothed = np.convolve(raw, np.ones(window, dtype=np.float32) / window, mode="same")
    # Clipped at zero before the power. sin(pi) in float32 lands on about
    # -8.7e-8 rather than 0, and a negative raised to a fractional power is
    # NaN -- which propagates through the normalisation and sends a buffer of
    # NaN to the speaker. Caught only because the peak printed as "nan".
    shape = np.clip(np.sin(np.linspace(0.0, math.pi, n, dtype=np.float32)), 0.0, None)
    swell = shape ** 1.5
    return smoothed * swell


#: Middle C and friends, so the tunes below read as notes rather than numbers.
_C5, _E5, _G5, _C6, _G4, _E4 = 523.25, 659.25, 783.99, 1046.50, 392.00, 329.63


def _fanfare() -> np.ndarray:
    """Won the quiz. A rising major arpeggio, which is the shortest possible
    way to sound pleased."""
    return _sequence([(_C5, 0.12), (_E5, 0.12), (_G5, 0.12), (_C6, 0.5)])


def _correct() -> np.ndarray:
    """Right answer. Two notes up."""
    return _sequence([(_E5, 0.09), (_G5, 0.22)])


def _wrong() -> np.ndarray:
    """Wrong answer, and deliberately NOT a buzzer. This plays to school groups
    and a harsh noise aimed at a child who guessed is the wrong instrument
    entirely -- two soft notes down read as "not that one", not as a failure."""
    return _sequence([(_G4, 0.10), (_E4, 0.20)])


def _ready() -> np.ndarray:
    """Finished checking itself over, and everything works."""
    return _sequence([(_G5, 0.10), (_C6, 0.28)])


def _uhoh() -> np.ndarray:
    """Something is wrong -- the pre-flight found a fault."""
    return _sequence([(_E5, 0.14), (_G4, 0.30)])


def _applause() -> np.ndarray:
    return _noise_swell(1.4, np.random.default_rng(7))


_BUILDERS = {
    "fanfare": _fanfare,
    "correct": _correct,
    "wrong": _wrong,
    "ready": _ready,
    "uhoh": _uhoh,
    "applause": _applause,
}

#: Built on first use and kept, because the arithmetic is cheap but not free
#: and these are played from the voice loop, where a spare 30ms is a spare 30ms.
_CACHE: dict[str, np.ndarray] = {}


def names() -> list[str]:
    return sorted(_BUILDERS)


def get(name: str) -> Optional[np.ndarray]:
    """A clip as float32 at SAMPLE_RATE, or None if there is no such sound.

    None rather than an exception: a demo asking for a sound that does not
    exist should carry on silently, never take the turn down with it.
    """
    if name in _CACHE:
        return _CACHE[name]
    build = _BUILDERS.get(name)
    if build is None:
        logger.debug("No sound called %r", name)
        return None
    try:
        clip = build().astype(np.float32)
    except Exception:
        logger.exception("Could not build the %r sound", name)
        return None

    limit = int(SAMPLE_RATE * _MAX_CLIP_S)
    if len(clip) > limit:
        # Enforced rather than trusted. The microphone is deaf for as long as
        # the speaker is busy, so a long clip is a robot that cannot be
        # interrupted -- and the person adding the next sound will not be
        # thinking about that.
        logger.warning("Sound %r is longer than %.1fs; trimming.", name, _MAX_CLIP_S)
        clip = clip[:limit]

    # Nothing non-finite may reach a speaker. A single NaN or inf silences the
    # whole buffer at best and produces a loud click at worst, and the sound
    # that finds this out is the one playing to a room.
    if not np.all(np.isfinite(clip)):
        logger.error("Sound %r generated non-finite samples; refusing to play it.", name)
        return None

    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if peak > 0:
        clip = clip * (_PEAK / peak)
    _CACHE[name] = clip.astype(np.float32)
    return _CACHE[name]
