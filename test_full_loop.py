"""End-to-end check: wake word -> transcribe -> generate -> speak.

Runs against a live `reachy-mini-daemon --sim` (confirms it's reachable and in
sim mode) while the audio itself goes through this machine's mic/speaker per
config, exactly as body/voice_loop.py does it -- except "listen" is replaced
by piper-synthesized audio standing in for a human voice, since this script
has no way to actually speak. wait_for_wake_word()/listen() (the live-mic
versions) are exercised by their own unit -- this proves the pieces chain
together correctly end to end.
"""

from reachy_mini import ReachyMini

from body.audio_io import AudioIO
from body.motion import MotionController
from brain.interface import get_reply
from config import SIMULATION
from test_stt_reliability import synthesize

TEST_MESSAGE = "What is the weather like today?"


def main() -> None:
    with ReachyMini(log_level="WARNING") as mini:
        status = mini.client.get_status()
        print(f"Daemon reachable, simulation_enabled={status.simulation_enabled}")

    audio = AudioIO(SIMULATION)
    motion = MotionController(SIMULATION)

    print("\n--- wake word ---")
    wake_samples = synthesize(audio, "Hey Reachy")
    print("detected:", audio.detect_wake_word_offline(wake_samples))

    print("\n--- transcribe (synthesized stand-in for a live utterance) ---")
    message_samples = synthesize(audio, TEST_MESSAGE)
    transcript = audio.transcribe_offline(message_samples)
    print(f"said: {TEST_MESSAGE!r}")
    print(f"heard: {transcript!r}")

    print("\n--- generate ---")
    reply_text, emotion_tag = get_reply(person_id=0, message=transcript)
    print(f"[{emotion_tag}] {reply_text}")

    print("\n--- speak (plays out loud on this machine's real speaker) ---")
    audio.speak(reply_text, emotion_tag)
    motion.express(emotion_tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
