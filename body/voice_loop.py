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
import queue
import random
import threading
import time
from typing import Iterator

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

#: How long to wait for the first sentence before covering the gap with a
#: filler. Below this, silence reads as normal conversational pause and
#: saying "let me think" only delays the answer it was meant to hide.
_FILLER_AFTER_S = 1.2


def _stream_reply_async(person_id: int, message: str) -> "queue.Queue":
    """Begin generating in a worker thread, delivering sentences via a queue.

    stream_reply is a generator, so nothing runs until it is iterated -- and
    iterating blocks. Handing it to a thread lets the caller start generation
    immediately and still make a timing decision (filler or not) while tokens
    are already being produced.

    Queue items are ``("sentence", (text, tag))``, ``("error", exception)``,
    or ``("done", None)``.
    """
    items: "queue.Queue" = queue.Queue()

    def worker() -> None:
        try:
            for sentence in stream_reply(person_id, message):
                items.put(("sentence", sentence))
        except Exception as exc:  # surfaced to the caller by _drain
            items.put(("error", exc))
        finally:
            items.put(("done", None))

    threading.Thread(target=worker, name="reply-stream", daemon=True).start()
    return items


def _drain(items: "queue.Queue", first: tuple) -> Iterator[tuple[str, str]]:
    """Yield sentences from the queue, starting with an already-taken item."""
    kind, payload = first
    while kind != "done":
        if kind == "error":
            raise payload
        yield payload
        kind, payload = items.get()


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

    # Deliberately no enrollment here. It used to interrupt mid-turn to ask
    # for a name, which meant the rest of the user's own question landed in
    # the answer -- one utterance became "HE MUST" as the question and "HAVE
    # COME" as the name, and that got stored as a person. Enrollment now only
    # happens when the user asks for it (see manage_people.py), where the
    # name can be typed and confirmed instead of guessed from noisy audio.
    # Unknown faces simply converse anonymously.
    if person_id is None:
        person_id = 0

    # Start generating *before* deciding whether to stall for time. The filler
    # used to be spoken first and generation only began once it finished, so
    # its ~2-3s was added to every reply. Now it overlaps generation and is
    # only spoken at all if the first sentence is slow to arrive -- on a fast
    # machine most replies skip it entirely.
    motion.express("thinking")
    logger.info("Generating reply (person_id=%s)...", person_id)
    started = time.monotonic()
    replies = _stream_reply_async(person_id, message)

    try:
        pending = replies.get(timeout=_FILLER_AFTER_S)
    except queue.Empty:
        logger.info("Reply slow (>%.1fs) -- speaking filler.", _FILLER_AFTER_S)
        audio.speak(random.choice(_THINKING_FILLERS), "thinking", motion=motion)
        pending = replies.get()
    final_tag = "neutral"
    for index, (sentence, emotion_tag) in enumerate(_drain(replies, pending)):
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
