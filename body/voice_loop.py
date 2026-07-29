"""The main loop: body captures a turn, brain decides the reply, body acts it out.

Models (STT/KWS/TTS) are loaded once in AudioIO.__init__, not per turn -- run_once
takes already-constructed components so run_forever can reuse them across turns.

The reply is streamed and spoken sentence-by-sentence (brain.interface.stream_reply)
rather than generated in full before speaking a word, because on hardware this
slow (see llm_max_tokens' docstring in config.py) waiting for the whole reply
means 10s of seconds of dead air. A short filler line plays immediately after
listening ends, before generation is even requested, so there's never silence
right after the user finishes speaking.
"""

import logging
import random
import time

from brain.interface import stream_reply
from config import HardwareTarget

from .audio_io import AudioIO
from .camera import Camera
from .face import FaceIdentifier
from .motion import MotionController

logger = logging.getLogger(__name__)

_THINKING_FILLERS = [
    "Hmm, let me think.",
    "Good question, one moment.",
    "Let's see.",
    "Okay, thinking.",
]


def run_once(audio: AudioIO, camera: Camera, face: FaceIdentifier, motion: MotionController) -> None:
    """Run a single wake -> listen -> think -> speak/express turn."""
    # A voice loop is invisible from the outside: when nothing happens there's
    # no way to tell "didn't hear the wake word" from "heard it but
    # transcribed nothing" from "still waiting on the LLM". These logs mark
    # each stage boundary so a silent robot is diagnosable from the log alone.
    logger.info("Waiting for wake word...")
    audio.wait_for_wake_word()
    logger.info("Wake word detected.")

    frame = camera.get_frame()
    person_id, active_face, _score = face.identify(frame, force=True)
    if active_face is not None and frame is not None:
        motion.track_face(active_face.bbox, frame.shape)
    message = audio.listen()
    logger.info("Heard: %r", message)

    # Silence is not a question. Without this the empty string went to the LLM,
    # which duly invented a greeting and spoke it -- burning a whole turn on
    # nothing. Retry once (people commonly pause after the wake word) before
    # giving up and going back to listening.
    if not message.strip():
        logger.info("Nothing transcribed -- listening once more.")
        message = audio.listen()
        logger.info("Heard: %r", message)
        if not message.strip():
            logger.info("Still nothing -- returning to wake word.")
            return

    if person_id is None and active_face is not None:
        motion.express("curious")
        audio.speak("I don't think we've met yet. What's your name?", "curious", motion=motion)
        name = audio.listen()
        person_id = face.enroll(name, active_face)

    if person_id is None:
        person_id = 0

    motion.express("thinking")
    audio.speak(random.choice(_THINKING_FILLERS), "thinking", motion=motion)

    logger.info("Generating reply (person_id=%s)...", person_id)
    started = time.monotonic()
    final_tag = "neutral"
    for index, (sentence, emotion_tag) in enumerate(stream_reply(person_id, message)):
        if index == 0:
            logger.info("First sentence after %.1fs", time.monotonic() - started)
        logger.info("Saying [%s]: %r", emotion_tag, sentence)
        motion.express(emotion_tag)
        audio.speak(sentence, emotion_tag, motion=motion)
        final_tag = emotion_tag
    logger.info("Turn complete in %.1fs", time.monotonic() - started)

    # Recorded-move flourish, once per turn with the real (final) tag -- not
    # per streamed sentence, since play_move() would otherwise add latency to
    # every sentence instead of just topping off the finished reply.
    motion.express_move(final_tag)


def run_forever(target: HardwareTarget) -> None:
    """Run the voice loop until interrupted."""
    # In "robot" mode audio, camera, and motion all share ONE ReachyMini.
    #
    # Not just an optimisation -- separate connections actively break each
    # other. MotionController's own connection asks for media_backend=
    # "no_media", which makes the SDK call release_media(): a daemon-wide
    # teardown that deletes the camera IPC socket and unregisters the WebRTC
    # producer (verified against the daemon: media_released flips to true and
    # /tmp/reachymini_camera_socket disappears). Whichever of the two connects
    # second then finds nothing to attach to -- media first meant motion
    # yanked the pipeline away mid-run; motion first meant the media
    # connection had no socket to open ("unixfdsrc: Failed to connect socket"
    # / "state change failed"). One connection, with a real media backend, is
    # the only ordering that has no such race.
    robot = None
    if target.mode == "robot":
        from reachy_mini import ReachyMini

        robot = ReachyMini(
            host=target.daemon_host,
            port=target.daemon_port,
            media_backend=target.media_backend,
            log_level="WARNING",
        )
        robot.__enter__()

    audio = AudioIO(target, robot=robot)
    camera = Camera(target, robot=robot)
    face = FaceIdentifier(target)
    motion = MotionController(target, robot=robot)

    try:
        while True:
            run_once(audio, camera, face, motion)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        audio.close()
        motion.stop()
        if robot is not None:
            robot.__exit__(None, None, None)
