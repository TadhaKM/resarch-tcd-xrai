"""The main loop: body captures a turn, brain decides the reply, body acts it out.

Models (STT/KWS/TTS) are loaded once in AudioIO.__init__, not per turn -- run_once
takes already-constructed components so run_forever can reuse them across turns.

The reply is streamed and spoken sentence-by-sentence (brain.interface.stream_reply)
rather than generated in full before speaking a word, so the robot starts
answering as soon as it has a complete sentence instead of after the whole
reply exists.

A wake word starts a conversation rather than a single question: run_once
keeps listening after each reply and only returns to the wake word once
nobody answers.
"""

import logging
import os
import random
import threading
import time

from typing import Optional

from brain import db
from brain.interface import stream_reply
from brain.modes import STATE
from config import HardwareTarget

from .audio_io import AudioIO
from .camera import Camera
from .face import FaceIdentifier
from .face_tracker import FaceTracker
from .motion import MotionController

logger = logging.getLogger(__name__)

#: Exit code meaning "the robot connection died and only a restart can fix it".
#: start_reachy.ps1 relaunches on this rather than leaving a deaf, frozen app.
_EXIT_LINK_LOST = 3


class ShutdownRequested(Exception):
    """Raised when the user asks the robot to stop listening."""


#: Spoken ways of saying "stop". Matched as substrings of the transcript
#: because the recognizer emits no punctuation and often trails extra words
#: ("OK TURN OFF NOW THANKS"). Kept explicit rather than asking the LLM,
#: since shutting down has to be reliable and instant, not a generated
#: decision that might arrive three seconds later.
_SHUTDOWN_PHRASES = (
    "turn off",
    "turn yourself off",
    "shut down",
    "shutdown",
    "shut off",
    "power off",
    "power down",
    "stop listening",
    "go to sleep",
    "goodbye",
    "good bye",
    "bye bye",
    "that is all",
    "that's all",
)

#: Spoken ways of asking for a dance.
_DANCE_PHRASES = (
    "dance",
    "dancing",
    "bust a move",
    "show me some moves",
)

#: Openers for greeter mode, when the person isn't recognised by name.
_GREETINGS = (
    "Oh, hello there!",
    "Hi! Nice to see you.",
    "Hello! I'm Reachy Mini.",
    "Hey there, good to see you.",
)

#: Minimum gap between greetings, so standing in view isn't greeted on every
#: pass -- the same runaway-turn problem the empty-transcript guard fixed.
_GREETING_COOLDOWN_S = 45.0
_last_greeting_at = 0.0

#: One repetition of the dance, so a mode change lands between repetitions.
_DANCE_BEAT_S = 5.0

#: Replies before a conversation returns to requiring the wake word. Bounds
#: how long a room full of talking can hold the robot's attention, since
#: follow-ups are accepted without one.
_MAX_EXCHANGES = 12


def _wait_for_wake_word_in_mode(
    audio: AudioIO, motion: MotionController, tracker: Optional[FaceTracker]
) -> bool:
    """Wait for the wake word, doing whatever the current mode does meanwhile.

    Returns True if the wake word was heard, False if the mode acted instead
    and the caller should start a fresh cycle (re-reading the mode, so a
    change made in the dashboard takes effect immediately).

    Modes are polled in short slices rather than run as long blocking
    behaviours, so switching mode in the dashboard is felt within a second or
    two instead of after whatever the robot happened to start.
    """
    mode = STATE.mode

    if mode == "dance":
        # Dance until told otherwise. dance() returns immediately and the
        # sleep covers the choreography, so a mode change lands between
        # repetitions rather than being ignored until this one finishes.
        motion.dance()
        if audio.wait_for_wake_word(timeout=_DANCE_BEAT_S):
            return True
        return False

    if mode == "greeter":
        # Greet someone who has just appeared, then fall through to the normal
        # conversation so they can reply without saying the wake word.
        if tracker is not None and tracker.enabled:
            person_id, active_face = tracker.current(max_age_s=1.5)
            seen = active_face is not None
            STATE.set_flags(face_visible=seen)
            if seen and _greeting_due():
                greeting = _greeting_for(person_id)
                logger.info("Greeting someone: %r", greeting)
                STATE.add("said", greeting)
                motion.express("happy")
                motion.express_move("happy")
                audio.speak(greeting, "happy", motion=motion)
                return True
        if audio.wait_for_wake_word(timeout=1.0):
            return True
        return False

    if mode == "idle":
        return audio.wait_for_wake_word(timeout=1.0)

    # conversation: the default, and the only mode that just waits.
    return audio.wait_for_wake_word(timeout=2.0)


def _greeting_due() -> bool:
    """True if enough time has passed to greet again.

    Without this the robot re-greets on every pass while someone stands in
    front of it, which is the same runaway-turn problem the empty-transcript
    guard fixed for silence.
    """
    global _last_greeting_at
    now = time.monotonic()
    if now - _last_greeting_at < _GREETING_COOLDOWN_S:
        return False
    _last_greeting_at = now
    return True


def _greeting_for(person_id: Optional[int]) -> str:
    """Greet by name when the person is recognised, generically otherwise."""
    if person_id:
        name = db.get_person_name(person_id)
        if name:
            return f"Hello again, {name}!"
    return random.choice(_GREETINGS)


def run_once(
    audio: AudioIO,
    camera: Camera,
    face: FaceIdentifier,
    motion: MotionController,
    tracker: Optional[FaceTracker] = None,
) -> None:
    """Wake once, then converse until the person stops replying.

    The wake word opens a conversation rather than buying a single question.
    Requiring "Hey Reachy" before every sentence made follow-ups feel like
    separate transactions instead of a conversation; now the robot keeps
    listening after each reply and only falls back to the wake word once
    nobody answers.
    """
    # A voice loop is invisible from the outside: when nothing happens there's
    # no way to tell "didn't hear the wake word" from "heard it but
    # transcribed nothing" from "still waiting on the LLM". These logs mark
    # each stage boundary so a silent robot is diagnosable from the log alone.
    logger.info("Waiting for wake word...")
    STATE.set_flags(listening=True, ready=True)
    STATE.note("listen", 'Waiting for "Hey Reachy"...')

    # Every mode still listens for the wake word -- talking to the robot must
    # work whatever it is doing, so a mode can never leave it unresponsive.
    # What differs is what it does *while* waiting, which is why the modes act
    # here rather than replacing this loop.
    if not _wait_for_wake_word_in_mode(audio, motion, tracker):
        return
    logger.info("Wake word detected.")
    STATE.add("listen", 'Heard "Hey Reachy" -- listening for your question')

    # Identity comes from the tracker, which is already watching continuously
    # and owns the detector -- calling it again here would mean two threads in
    # MediaPipe at once. Falls back to a one-off detection when tracking is
    # unavailable (e.g. the CM4, where the detector can't run at all).
    if tracker is not None and tracker.enabled:
        person_id, _active_face = tracker.current()
    else:
        frame = camera.get_frame()
        person_id, active_face, _score = face.identify(frame, force=True)
        if active_face is not None and frame is not None:
            motion.track_face(active_face.bbox, frame.shape)

    # Deliberately no enrollment here. It used to interrupt mid-turn to ask
    # for a name, which meant the rest of the user's own question landed in
    # the answer -- one utterance became "HE MUST" as the question and "HAVE
    # COME" as the name, and that got stored as a person. Enrollment now only
    # happens when the user asks for it (see manage_people.py), where the
    # name can be typed and confirmed instead of guessed from noisy audio.
    # Unknown faces simply converse anonymously.
    if person_id is None:
        person_id = 0

    message = audio.listen()
    logger.info("Heard: %r", message)
    STATE.add("heard", message)

    # Silence is not a question. Without this the empty string went to the LLM,
    # which duly invented a greeting and spoke it -- burning a whole turn on
    # nothing. Retry once (people commonly pause after the wake word) before
    # giving up and going back to listening.
    if not message.strip():
        logger.info("Nothing transcribed -- listening once more.")
        STATE.add("status", "Didn't catch that -- listening again")
        message = audio.listen()
        logger.info("Heard: %r", message)
        STATE.add("heard", message)
        if not message.strip():
            logger.info("Still nothing -- returning to wake word.")
            STATE.add("status", "Nothing heard -- back to the wake word")
            return

    exchanges = 0
    while message.strip():
        spoken = message.lower()

        # Checked before the LLM sees the message: "turn off" has to act, not
        # be answered, and waiting on generation to decide would make it feel
        # unresponsive at exactly the moment the user wants it to stop.
        if any(phrase in spoken for phrase in _SHUTDOWN_PHRASES):
            logger.info("Shutdown phrase heard in %r.", message)
            motion.express("sad")
            audio.speak("Okay, goodbye!", "sad", motion=motion)
            raise ShutdownRequested

        if any(phrase in spoken for phrase in _DANCE_PHRASES):
            logger.info("Dance requested.")
            motion.express("happy")
            # Started before speaking so the robot is already moving as it
            # answers, rather than talking about dancing and then dancing.
            motion.dance()
            audio.speak("Watch this!", "happy", motion=motion)
        else:
            _respond(audio, motion, person_id, message)
        exchanges += 1
        if exchanges >= _MAX_EXCHANGES:
            logger.info("Conversation length limit reached -- back to wake word.")
            return
        # No wake word needed for the follow-up. listen() returns "" once it
        # has waited out the silence, which doubles as "the conversation is
        # over" without needing a separate timeout mechanism.
        logger.info("Listening for a follow-up...")
        message = audio.listen()
        logger.info("Heard: %r", message)

    logger.info("No follow-up -- returning to wake word.")


def _respond(audio: AudioIO, motion: MotionController, person_id: int, message: str) -> None:
    """Generate and speak one reply, with matching expression and gesture."""
    # No spoken filler while thinking. It was there to cover slow generation on
    # the robot's own CPU, but the first sentence now arrives in a few seconds
    # and a canned "let me think" only delays the answer it was meant to hide.
    # The thinking *pose* stays -- that reads as considering the question
    # without putting words in the robot's mouth.
    motion.express("thinking")
    logger.info("Generating reply (person_id=%s)...", person_id)
    started = time.monotonic()

    final_tag = "neutral"
    for index, (sentence, emotion_tag) in enumerate(stream_reply(person_id, message)):
        if index == 0:
            logger.info("First sentence after %.1fs", time.monotonic() - started)
        logger.info("Saying [%s]: %r", emotion_tag, sentence)
        STATE.add("said", sentence)
        STATE.set_flags(speaking=True)
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

    # Follow whoever is in view for the whole session, not just at the moment
    # a turn starts -- the robot should hold your gaze while you talk to it and
    # between questions, which is what makes it feel present rather than
    # snapping to attention only when addressed.
    tracker = FaceTracker(camera, face, motion)
    tracker.start()

    # A visible greeting on startup. Idle motion is subtle by design, so a
    # working robot and a disconnected one look identical from across a room;
    # this makes "connected and under control" unmistakable without reading a
    # log, and doubles as a check that the motors are actually holding.
    motion.wake_up()

    # This session cannot repair its own connection: the SDK's websocket client
    # has no reconnect (disconnect() is terminal), and this connection carries
    # audio and camera too, so rebuilding it means rebuilding everything.
    # Rather than keep listening and replying with a frozen head -- which is
    # what happened, for thousands of consecutive dropped frames -- end the
    # process so the launcher can start a fresh session.
    def _watch_link() -> None:
        motion.link_lost.wait()
        logger.error("Motion link unrecoverable -- exiting for a clean restart.")
        os._exit(_EXIT_LINK_LOST)

    threading.Thread(target=_watch_link, name="link-watchdog", daemon=True).start()

    try:
        while True:
            run_once(audio, camera, face, motion, tracker)
    except (KeyboardInterrupt, ShutdownRequested):
        # Both are ordinary ways to end a session, not failures -- fall through
        # to the same cleanup as a normal exit.
        logger.info("Shutting down.")
    finally:
        tracker.stop()
        camera.close()
        audio.close()
        motion.stop()
        if robot is not None:
            robot.__exit__(None, None, None)
