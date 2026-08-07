"""The main loop: body captures a turn, brain decides the reply, body acts it out.

Models (STT/KWS/TTS) are loaded once in AudioIO.__init__, not per turn -- run_once
takes already-constructed components so run_forever can reuse them across turns.

The reply is streamed and spoken sentence-by-sentence (brain.interface.stream_reply)
rather than generated in full before speaking a word, so the robot starts
answering as soon as it has a complete sentence instead of after the whole
reply exists.

Each turn needs its own wake word, and "turn off" puts the robot to sleep
rather than ending the process -- it keeps listening for "Hey Reachy" so it
can be woken by voice.
"""

import logging
import os
import random
import re
import threading
import time

from typing import Optional

import requests

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

# How long to wait for the daemon to come back after _ensure_daemon_advertises
# restarts it. Measured at roughly a minute on this hardware.
_DAEMON_RESTART_TIMEOUT_S = 90.0


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

#: Spoken ways of asking for a story. Works in any mode; storyteller mode
#: also volunteers one on entry and then occasionally.
_STORY_PHRASES = (
    "tell me a story",
    "tell us a story",
    "another story",
    "one more story",
    "tell a story",
)

#: What the LLM is asked for when a story is due. How to tell it lives in the
#: storyteller system prompt (brain/prompts.py); this only sets the brief.
#: Length is capped in words here rather than trusting num_predict, which
#: truncates mid-sentence.
_STORY_REQUEST = (
    "Tell me a brand new story, sixty to eighty words, for all ages. "
    "Pick a different topic than last time."
)

#: Poses cycled through while telling a story, so the robot performs rather
#: than holding one "thinking" pose for the whole thing. The reply's real
#: emotion still drives the closing gesture (see _respond).
_STORY_POSES = ("curious", "happy", "surprised", "thinking")

#: How long storyteller mode waits before volunteering another story.
_STORY_INTERVAL_S = 240.0
_last_story_at = 0.0

#: Active spoken timers: (threading.Timer, description, fire_at_monotonic).
_timers: list[tuple[threading.Timer, str, float]] = []
_timers_lock = threading.Lock()

#: Word numbers Whisper actually emits for small counts; digits cover the rest.
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty-five": 45, "fifty": 50, "sixty": 60,
}

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

def _parse_timer(spoken: str) -> Optional[tuple[int, str]]:
    """Extract (seconds, human description) from a timer request, else None.

    Handles both digits and the number words Whisper produces ("set a timer
    for five minutes" / "timer for 30 seconds" / "remind me in an hour").
    """
    text = spoken.lower()
    if "timer" not in text and "remind me in" not in text:
        return None
    if "half an hour" in text:
        return 1800, "30 minute"
    if "half a minute" in text:
        return 30, "30 second"

    m = re.search(r"(?:timer[^a-z0-9]*(?:for|of)?|remind me in)\s+([a-z0-9-]+)\s+(second|minute|hour)s?", text)
    if not m:
        return None
    raw, unit = m.group(1), m.group(2)
    amount = _NUMBER_WORDS.get(raw)
    if amount is None:
        try:
            amount = int(raw)
        except ValueError:
            return None
    seconds = amount * {"second": 1, "minute": 60, "hour": 3600}[unit]
    if not 5 <= seconds <= 2 * 3600:
        return None
    return seconds, f"{amount} {unit}"


def _handle_timer_command(spoken: str, audio: AudioIO, motion: MotionController) -> bool:
    """Set or cancel spoken timers. Returns True if this turn was a timer."""
    if "cancel" in spoken and "timer" in spoken:
        with _timers_lock:
            cancelled = len(_timers)
            for t, _, _ in _timers:
                t.cancel()
            _timers.clear()
        motion.express("neutral")
        audio.speak(
            f"Okay, cancelled {cancelled} timer{'s' if cancelled != 1 else ''}."
            if cancelled else "There were no timers running.",
            "neutral", motion=motion,
        )
        return True

    parsed = _parse_timer(spoken)
    if parsed is None:
        return False
    seconds, desc = parsed

    def _fire() -> None:
        with _timers_lock:
            _timers[:] = [t for t in _timers if t[0] is not timer]
        # Queued rather than spoken from this thread: only the voice loop may
        # drive the speaker, or two audio streams interleave into garbage.
        STATE.request("say", f"Ding ding! Your {desc} timer is finished.", "happy")

    timer = threading.Timer(seconds, _fire)
    timer.daemon = True
    with _timers_lock:
        _timers.append((timer, desc, time.monotonic() + seconds))
    timer.start()
    logger.info("Timer set: %s (%ds)", desc, seconds)
    STATE.add("status", f"Timer set for {desc}")
    motion.express("happy")
    audio.speak(f"Timer set for {desc}. I'll let you know.", "happy", motion=motion)
    return True


def _tell_story(audio: AudioIO, motion: MotionController) -> None:
    global _last_story_at
    _last_story_at = time.monotonic()
    logger.info("Telling a story.")
    STATE.add("status", "Telling a story")
    _respond(audio, motion, 0, _STORY_REQUEST, style="story")


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
    # Dashboard requests come first, even asleep: "Say it" speaks regardless,
    # and "Listen now" behaves exactly like the wake word (waking if needed).
    request = STATE.pop_request()
    if request is not None:
        kind, text, tag = request
        if kind == "say":
            logger.info("Dashboard say: %r", text)
            STATE.add("said", text)
            motion.express(tag)
            audio.speak(text, tag, motion=motion)
            return False
        if kind == "listen":
            logger.info("Dashboard listen-now pressed.")
            STATE.add("status", "Listen button pressed")
            STATE.set_sleeping(False)
            return True

    # Asleep: the wake word is the only thing that gets a response. Modes stay
    # selected but dormant, so "turn off" quietens the robot without ending the
    # process and without losing what it was set to do.
    if STATE.sleeping:
        if audio.wait_for_wake_word(timeout=2.0):
            STATE.set_sleeping(False)
            motion.acknowledge()
            motion.express("happy")
            audio.speak("I'm awake.", "happy", motion=motion)
            return False
        return False

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

    if mode == "story":
        # A story on entering the mode, then occasionally; a wake phrase in
        # between works as normal (and "another story" asks for one directly).
        if STATE.pop_mode_changed() or time.monotonic() - _last_story_at > _STORY_INTERVAL_S:
            _tell_story(audio, motion)
            return False
        return audio.wait_for_wake_word(timeout=2.0)

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
    """Handle one wake -> listen -> answer turn, then return to the wake word.

    Follow-ups without a wake word were tried and removed: in a room with
    other people talking, the robot answered whatever was said next by anyone
    and kept the thread running from there. Requiring "Hey Reachy" each time
    is what makes it unambiguous who is being spoken to.
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
    STATE.add("listen", "Wake word heard -- listening for your question")
    # Visible acknowledgement before listening, so it is obvious the robot is
    # waiting for you rather than ignoring you.
    motion.acknowledge()

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

    # One exchange per wake word, then back to waiting.
    #
    # This did once continue listening for follow-ups, which reads better on
    # paper but not in a room with other people in it: the robot picked up
    # whatever was said next by anyone, answered it, and kept the thread going
    # from there -- it spent minutes replying to a conversation it was not part
    # of. Requiring "Hey Reachy" each time is what makes it clear who is being
    # addressed, and it costs one short phrase.
    spoken = message.lower()

    # Checked before the LLM sees the message: "turn off" has to act, not be
    # answered, and waiting on generation to decide would make it feel
    # unresponsive at exactly the moment the user wants it to stop.
    if any(phrase in spoken for phrase in _SHUTDOWN_PHRASES):
        # Sleep, don't exit. Ending the process meant the only way back was
        # relaunching it from a terminal, which is no use to someone standing
        # in front of the robot -- it should be wakeable the same way it is
        # woken any other time.
        logger.info("Sleep phrase heard in %r.", message)
        motion.express("sad")
        audio.speak("Okay, goodbye! Say Hey Reachy when you need me.", "sad", motion=motion)
        STATE.set_sleeping(True)
        return

    if _handle_timer_command(spoken, audio, motion):
        pass
    elif any(phrase in spoken for phrase in _DANCE_PHRASES):
        logger.info("Dance requested.")
        motion.express("happy")
        # Started before speaking so the robot is already moving as it answers,
        # rather than talking about dancing and then dancing.
        motion.dance()
        audio.speak("Watch this!", "happy", motion=motion)
    elif any(phrase in spoken for phrase in _STORY_PHRASES):
        _tell_story(audio, motion)
    else:
        _respond(audio, motion, person_id, message)

    logger.info("Turn finished -- waiting for the wake word again.")


def _respond(
    audio: AudioIO,
    motion: MotionController,
    person_id: int,
    message: str,
    style: Optional[str] = None,
) -> None:
    """Generate and speak one reply, with matching expression and gesture.

    style="story" swaps in the storyteller persona, a performed delivery, and
    a changing pose per sentence.
    """
    # No spoken filler while thinking. It was there to cover slow generation on
    # the robot's own CPU, but the first sentence now arrives in a few seconds
    # and a canned "let me think" only delays the answer it was meant to hide.
    # The thinking *pose* stays -- that reads as considering the question
    # without putting words in the robot's mouth.
    motion.express("thinking")
    logger.info("Generating reply (person_id=%s)...", person_id)
    started = time.monotonic()

    final_tag = "neutral"
    spoken = 0
    for sentence, emotion_tag in stream_reply(person_id, message, style=style):
        # The reply's real emotion arrives on a final pair that is usually just
        # the tag, with no text left to say (see stream_reply).
        final_tag = emotion_tag
        if not sentence:
            continue
        if spoken == 0:
            logger.info("First sentence after %.1fs", time.monotonic() - started)
        pose = _STORY_POSES[spoken % len(_STORY_POSES)] if style == "story" else emotion_tag
        spoken += 1
        logger.info("Saying [%s]: %r", pose, sentence)
        STATE.add("said", sentence)
        STATE.set_flags(speaking=True)
        motion.express(pose)
        audio.speak(sentence, pose, motion=motion, expressive=style == "story")
    logger.info("Turn complete in %.1fs", time.monotonic() - started)

    # Recorded-move flourish, once per turn with the real (final) tag -- not
    # per streamed sentence, since play_move() would otherwise add latency to
    # every sentence instead of just topping off the finished reply.
    motion.express_move(final_tag)


def _ensure_daemon_advertises(host: str, port: int) -> None:
    """Restart the daemon if it is still advertising a stale address.

    The SDK does not open its WebRTC media connection to the host we connected
    to. It opens it to whatever address the daemon reports as its own in
    /api/daemon/status, and the daemon reads that once at startup. Any network
    change the robot makes afterwards therefore leaves it advertising an
    address that no longer exists: joining a WiFi network, or falling back to
    its own hotspot, which the daemon's own wifi_config does automatically
    whenever a connection attempt fails.

    Nothing else notices, because the daemon still answers on the address we
    do have -- the launcher's reachability check passes, the REST API works.
    It surfaces only as ReachyMini() hanging and then raising a bare
    "TimeoutError: timed out" from inside the media stack, which kills the app
    before the first turn. Seen both ways round: advertising its hotspot
    address while on WiFi, and (starting the robot on its own WiFi, offline)
    advertising the old WiFi address while in hotspot mode.

    Restarting the daemon is the documented fix, so do it here rather than
    leaving a crash for someone to diagnose.
    """
    status_url = f"http://{host}:{port}/api/daemon/status"
    try:
        advertised = requests.get(status_url, timeout=5).json().get("wlan_ip")
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Could not read daemon status (%s); connecting anyway.", exc)
        return
    if advertised == host:
        return

    logger.warning(
        "Daemon reached at %s but advertising itself as %s -- media would be opened "
        "to an unreachable address. Restarting it (takes up to %.0fs).",
        host,
        advertised,
        _DAEMON_RESTART_TIMEOUT_S,
    )
    try:
        requests.post(f"http://{host}:{port}/api/daemon/restart", timeout=20)
    except requests.RequestException as exc:
        logger.warning("Daemon restart request failed (%s); connecting anyway.", exc)
        return

    deadline = time.monotonic() + _DAEMON_RESTART_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(5.0)
        try:
            status = requests.get(status_url, timeout=5).json()
        except (requests.RequestException, ValueError):
            continue  # still coming back up
        if status.get("state") == "running" and status.get("wlan_ip") == host:
            logger.info("Daemon restarted and now advertising %s.", host)
            return
    logger.warning(
        "Daemon did not come back advertising %s in time; connecting anyway.", host
    )


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

        # Only the remote path routes media over WebRTC to the advertised
        # address; on the robot itself the media backend is local, so a stale
        # value is harmless there and not worth a restart.
        if target.media_backend == "webrtc":
            _ensure_daemon_advertises(target.daemon_host, target.daemon_port)

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
