"""Demonstrate idle, face tracking, and speech motion in Reachy Mini Control."""

import argparse
import time

import numpy as np

from body.motion import MotionController
from config import SIMULATION


def fake_speech(motion: MotionController, duration: float = 4.0) -> None:
    sample_rate = 16000
    chunk_seconds = 0.12
    chunk_samples = int(sample_rate * chunk_seconds)
    started = time.monotonic()
    motion.begin_speech()
    try:
        while time.monotonic() - started < duration:
            t = np.arange(chunk_samples, dtype=np.float32) / sample_rate
            elapsed = time.monotonic() - started
            syllable = 0.25 + 0.65 * max(0.0, np.sin(elapsed * np.pi * 3.0))
            samples = (0.12 * syllable * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
            motion.feed_speech_audio(samples)
            time.sleep(chunk_seconds)
    finally:
        motion.end_speech()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument("--tracking-seconds", type=float, default=4.0)
    parser.add_argument("--speech-seconds", type=float, default=5.0)
    args = parser.parse_args()

    motion = MotionController(SIMULATION)
    try:
        print("idle")
        motion.express("neutral")
        time.sleep(args.idle_seconds)

        print("tracking")
        frame_shape = (480, 640, 3)
        started = time.monotonic()
        while time.monotonic() - started < args.tracking_seconds:
            elapsed = time.monotonic() - started
            x = int(80 + 420 * (0.5 + 0.5 * np.sin(elapsed * np.pi * 0.6)))
            motion.track_face((x, 150, 120, 120), frame_shape, ttl=0.6)
            time.sleep(0.15)

        print("speaking")
        motion.express("happy")
        fake_speech(motion, args.speech_seconds)

        print("done")
        motion.express("neutral")
        time.sleep(1.0)
    finally:
        motion.stop()


if __name__ == "__main__":
    main()
