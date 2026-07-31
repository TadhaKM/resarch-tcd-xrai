"""Expressive Reachy Mini motion with idle, tracking, and speech priorities."""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import HardwareTarget

logger = logging.getLogger(__name__)

EmotionTag = str


@dataclass(frozen=True)
class HeadPose:
    """Tunable head pose and antenna target."""

    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    z: float = 0.0
    antennas: tuple[float, float] = (0.0, 0.0)


class EmotionMapper:
    """Map reply emotion tags to concrete pose targets.

    Angles are stored in degrees here because these are design-tuning values;
    they are converted to radians at the SDK boundary. This is the continuous
    resting/baseline expression, held throughout a reply; MotionController's
    express_move() layers a one-shot recorded animation (pollen-robotics/
    reachy-mini-emotions-library) on top for a stronger physical flourish.
    """

    _POSES: dict[EmotionTag, HeadPose] = {
        "neutral": HeadPose(pitch=0.0, yaw=0.0, roll=0.0, z=0.0, antennas=(-10.0, 10.0)),
        "happy": HeadPose(pitch=6.0, yaw=0.0, roll=5.0, z=8.0, antennas=(22.0, -22.0)),
        "curious": HeadPose(pitch=3.0, yaw=10.0, roll=-8.0, z=4.0, antennas=(8.0, 22.0)),
        "thinking": HeadPose(pitch=-4.0, yaw=-8.0, roll=7.0, z=0.0, antennas=(-4.0, 18.0)),
        "surprised": HeadPose(pitch=10.0, yaw=0.0, roll=0.0, z=12.0, antennas=(35.0, -35.0)),
        "sad": HeadPose(pitch=-12.0, yaw=0.0, roll=-5.0, z=-8.0, antennas=(-45.0, 45.0)),
    }

    def get(self, tag: str) -> HeadPose:
        return self._POSES.get(tag, self._POSES["neutral"])


# One representative recorded move per brain emotion tag, picked from the 85
# available in pollen-robotics/reachy-mini-emotions-library. No "neutral"
# entry -- neutral is the resting baseline pose above, not worth interrupting
# it for a flourish every reply.
EMOTION_MOVES: dict[EmotionTag, str] = {
    "happy": "cheerful1",
    "sad": "sad1",
    "curious": "curious1",
    "surprised": "surprised1",
    "thinking": "thoughtful1",
}


class HeadWobbler:
    """Convert TTS amplitude into speech-synced pitch offsets."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._level = 0.0
        self._last_audio_at = 0.0
        self._active = False

    def start(self) -> None:
        with self._lock:
            self._level = 0.0
            self._active = True
            self._last_audio_at = time.monotonic()

    def feed(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(mono))))
        # Piper chunks are normalized float audio. This maps ordinary speech to
        # a visible but restrained 0..1 control signal.
        level = max(0.0, min(1.0, rms * 9.0))
        with self._lock:
            self._level = 0.65 * self._level + 0.35 * level
            self._last_audio_at = time.monotonic()
            self._active = True

    def stop(self) -> None:
        with self._lock:
            self._active = False

    def offset(self, now: float) -> HeadPose:
        with self._lock:
            age = now - self._last_audio_at
            active = self._active or age < 0.25
            level = self._level * math.exp(-age * 7.0)
        if not active:
            return HeadPose()
        pulse = math.sin(now * math.tau * 5.2)
        pitch = 1.0 + level * (3.2 + 1.4 * pulse)
        roll = level * 1.2 * math.sin(now * math.tau * 2.1)
        return HeadPose(pitch=pitch, roll=roll)

    def is_active(self, now: float) -> bool:
        with self._lock:
            return self._active or now - self._last_audio_at < 0.25


class IdleAnimator:
    """Subtle breathing plus occasional slow look-around."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._look_until = 0.0
        self._next_look_at = time.monotonic() + self._rng.uniform(3.0, 7.0)
        self._look_target = HeadPose()

    def offset(self, now: float) -> HeadPose:
        # Amplitudes are deliberately well above the smallest visible movement.
        # These started around 1 degree of pitch, which is real motion the
        # motors do execute but nobody can see -- the robot read as switched
        # off while sitting there idling correctly.
        breathing = HeadPose(
            pitch=3.5 * math.sin(now * math.tau * 0.18),
            roll=1.8 * math.sin(now * math.tau * 0.11),
            z=7.0 * math.sin(now * math.tau * 0.18),
        )

        if now >= self._next_look_at:
            self._look_until = now + self._rng.uniform(1.4, 2.8)
            self._next_look_at = self._look_until + self._rng.uniform(3.0, 6.5)
            self._look_target = HeadPose(
                pitch=self._rng.uniform(-6.0, 8.0),
                yaw=self._rng.uniform(-22.0, 22.0),
                roll=self._rng.uniform(-6.0, 6.0),
            )

        if now < self._look_until:
            phase = (self._look_until - now) / max(0.001, self._look_until - (self._next_look_at - 6.0))
            blend = min(1.0, max(0.0, math.sin(math.pi * phase)))
            return _add_pose(breathing, _scale_pose(self._look_target, blend))

        return breathing


@dataclass
class _TrackingTarget:
    pose: HeadPose
    expires_at: float


#: Upward tilt held at all times so the camera frames faces rather than
#: torsos. The lens is in the head and sits below head height for anyone
#: standing or sitting near the robot: a captured frame showed a chest and
#: hands with the head cropped off the top, which is why face detection
#: reported "no faces" while working perfectly.
_CAMERA_PITCH_BIAS = 18.0

_SEND_WARNING_INTERVAL_S = 10.0

# Sustained failure before rebuilding the connection. Longer than the SDK's 1s
# liveness window and the multi-second stalls seen on this wireless link, so
# transient drops still resolve themselves without a reconnect.
_RECONNECT_AFTER_S = 8.0

# Minimum gap between reconnect attempts, so a robot that is off or
# unreachable is retried steadily rather than hammered.
_RECONNECT_COOLDOWN_S = 15.0

# Blend factor for daemon-side face tracking (see set_face_tracking). High
# enough to hold someone's gaze convincingly, low enough that the emotion
# poses and speech wobble still read on the head.
_FACE_TRACK_WEIGHT = 0.65


def _enable_motors(robot: object) -> None:
    """Arm the robot's motors, tolerating SDK versions that lack the call."""
    if not hasattr(robot, "enable_motors"):
        return
    try:
        robot.enable_motors()
    except Exception:
        logger.exception("Failed to enable Reachy Mini motors")


class MotionController:
    """Coordinates Reachy Mini head and antenna personality motion."""

    def __init__(
        self,
        target: HardwareTarget,
        *,
        connect: bool = True,
        robot: object | None = None,
    ) -> None:
        """Coordinate motion, optionally over a caller-supplied connection.

        Pass `robot` to reuse an existing ReachyMini instead of opening a
        second one. Anything that also needs the daemon's camera/mic must do
        this: MotionController's own connection (see _connect_robot) requests
        media_backend="no_media", which makes the SDK call release_media() --
        a *daemon-wide* teardown of the media server, not a per-connection
        opt-out. It deletes the camera IPC socket and unregisters the WebRTC
        producer, so any media connection made afterwards has nothing to
        attach to, and one made earlier gets torn out from under it.
        """
        self.target = target
        self.mapper = EmotionMapper()
        self.wobbler = HeadWobbler()
        self.idle = IdleAnimator(random.Random())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._robot: object | None = robot
        self._owns_robot = robot is None
        self._base_pose = self.mapper.get("neutral")
        self._current_pose = self._base_pose
        self._tracking: Optional[_TrackingTarget] = None
        # Every pose update is a websocket message. On-device that's cheap, but
        # to a remote client each one crosses the wireless link while WebRTC
        # media is already saturating it -- 50Hz there mostly produces dropped
        # frames and starves the status messages the SDK's 1s liveness check
        # depends on. 20Hz still reads as smooth for breathing/idle motion.
        self._control_hz = 20.0 if target.daemon_host not in ("localhost", "127.0.0.1") else 50.0
        self._send_failures = 0
        self._last_send_warning_at = 0.0
        self._failing_since: Optional[float] = None
        self._last_reconnect_at = 0.0
        #: Set when the link has been down long enough that it will not come
        #: back by itself and cannot be rebuilt here. run_forever watches this.
        self.link_lost = threading.Event()
        self._paused = False
        self._daemon_tracking = False
        self._moves = self._load_recorded_moves()

        if self._robot is None and connect:
            self._robot = self._connect_robot()
        elif self._robot is not None:
            # Caller-supplied connection: still ours to arm, since
            # _connect_robot's own enable_motors call is skipped.
            _enable_motors(self._robot)
        self.start()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="motion-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        robot = self._robot
        if robot is not None and self._owns_robot and hasattr(robot, "__exit__"):
            try:
                robot.__exit__(None, None, None)
            except Exception:
                logger.exception("Failed to close Reachy Mini SDK connection")
        self._robot = None

    def express(self, emotion_tag: str) -> None:
        """Set the primary expression pose for the next reply."""
        with self._lock:
            self._base_pose = self.mapper.get(emotion_tag)

    def set_face_tracking(self, enabled: bool) -> None:
        """Record whether something is continuously aiming the head at a face.

        Tracking itself is done by the caller (see body/face_tracker.py), not
        here and not by the daemon: the SDK exposes start_head_tracking(), but
        this robot's daemon (1.8.3) has no such command and rejects it over the
        websocket -- "Input tag 'set_head_tracking' does not match any of the
        expected tags" -- while the SDK call itself returns normally, so the
        feature looks enabled and does nothing. This flag only lets other
        motion (dance, per-turn aiming) know to stand aside.
        """
        self._daemon_tracking = enabled

    def acknowledge(self) -> None:
        """Perk up briefly: "I heard you, go ahead".

        Deliberately silent and short. Between the wake word and the first
        word of an answer there are several seconds of nothing, which is long
        enough to wonder whether it heard you at all and start repeating
        yourself -- and repeating yourself over the top of it is exactly what
        makes the transcript garbage. A spoken acknowledgement would collide
        with the question; a movement does not.

        Non-blocking, so listening starts immediately rather than after the
        gesture finishes.
        """
        if self._robot is None:
            return

        def _perform() -> None:
            with self._lock:
                self._paused = True
            try:
                # Antennas up and a small lift, then settle -- reads as
                # attention rather than as one of the emotion poses.
                for pose, hold in (
                    (HeadPose(pitch=10.0, z=8.0, antennas=(55.0, -55.0)), 0.22),
                    (HeadPose(pitch=4.0, z=3.0, antennas=(30.0, -30.0)), 0.18),
                ):
                    deadline = time.monotonic() + hold
                    while time.monotonic() < deadline:
                        self._send_pose(pose)
                        time.sleep(0.02)
            except Exception as exc:
                logger.debug("Acknowledge gesture skipped: %s", exc)
            finally:
                with self._lock:
                    self._paused = False

        threading.Thread(target=_perform, name="motion-ack", daemon=True).start()

    def wake_up(self) -> None:
        """Play a short, unmistakable greeting: look around, then nod."""
        if self._robot is None:
            logger.info("Wake-up skipped: no robot connection.")
            return

        def _perform() -> None:
            with self._lock:
                self._paused = True
            try:
                logger.info("Waking up.")
                for pose in (
                    HeadPose(yaw=22.0, antennas=(35.0, -35.0)),
                    HeadPose(yaw=-22.0, antennas=(-35.0, 35.0)),
                    HeadPose(pitch=14.0, z=10.0, antennas=(45.0, -45.0)),
                    HeadPose(pitch=-10.0, z=-6.0, antennas=(-20.0, 20.0)),
                    self.mapper.get("neutral"),
                ):
                    deadline = time.monotonic() + 0.55
                    while time.monotonic() < deadline:
                        self._send_pose(pose)
                        time.sleep(0.02)
                self._current_pose = self.mapper.get("neutral")
                logger.info("Awake.")
            except Exception as exc:
                logger.warning("Wake-up interrupted: %s", exc)
            finally:
                with self._lock:
                    self._paused = False

        threading.Thread(target=_perform, name="motion-wake", daemon=True).start()

    def dance(self, beats: int = 8, bpm: float = 108.0) -> None:
        """Perform a short head-and-antenna dance.

        Generated rather than played from the recorded-move library so it can
        run for any number of beats and stay in time with itself. Like
        express_move this pauses the procedural loop for the duration --
        otherwise breathing and idle motion fight the choreography for the
        same joints -- and runs on its own thread so a caller can keep talking
        over it.
        """
        if self._robot is None:
            logger.info("Dance skipped: no robot connection.")
            return

        def _perform() -> None:
            # Face tracking steers the head too, so it has to stand down for
            # the duration or the choreography and the tracker pull against
            # each other and the dance comes out muddy.
            was_tracking = self._daemon_tracking
            if was_tracking:
                self.set_face_tracking(False)
            with self._lock:
                self._paused = True
            try:
                self._run_dance(beats, bpm)
            except Exception as exc:
                logger.warning("Dance interrupted: %s", exc)
            finally:
                with self._lock:
                    self._paused = False
                if was_tracking:
                    self.set_face_tracking(True)

        threading.Thread(target=_perform, name="motion-dance", daemon=True).start()

    def _run_dance(self, beats: int, bpm: float) -> None:
        """Drive the joints through the choreography, in time."""
        beat_s = 60.0 / bpm
        step_hz = 50.0
        started = time.monotonic()
        total_s = beats * beat_s
        logger.info("Dancing: %d beats at %.0f bpm (%.1fs)", beats, bpm, total_s)

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= total_s:
                break
            # Position within the bar, so the moves repeat on a musical cycle
            # rather than drifting.
            beat = elapsed / beat_s
            phase = 2.0 * math.pi * beat

            # Head: bob on every beat, sway across two, tilt across four --
            # layering periods keeps it from looking like a metronome.
            pitch = 9.0 * math.sin(phase)
            yaw = 14.0 * math.sin(phase / 2.0)
            roll = 7.0 * math.sin(phase / 4.0)
            # Millimetres, like EmotionMapper's poses (converted at the SDK
            # boundary), so this is a visible bounce rather than a rounding error.
            z = 9.0 * math.sin(phase)

            # Antennas flick on the offbeat and alternate, which reads as
            # deliberate rather than as a symmetric twitch.
            offbeat = math.sin(phase + math.pi / 2.0)
            antennas = (35.0 * offbeat, -35.0 * offbeat)

            pose = HeadPose(pitch=pitch, yaw=yaw, roll=roll, z=z, antennas=antennas)
            self._current_pose = pose
            self._send_pose(pose)
            time.sleep(1.0 / step_hz)

        # Settle back to neutral rather than stopping wherever the last beat
        # happened to leave the head.
        self._send_pose(self.mapper.get("neutral"))
        logger.info("Dance finished.")

    def express_move(self, emotion_tag: str) -> None:
        """Play a recorded emotion animation once, as a physical flourish.

        Non-blocking (runs in a background thread) so it never adds latency
        to a reply -- call this once per turn with the *final* tag, not per
        streamed sentence. The continuous procedural loop is paused for the
        move's duration since both would otherwise fight over set_target().
        No-op for tags with no entry in EMOTION_MOVES (currently "neutral"),
        if the robot isn't connected, or if the library failed to load.
        """
        move_name = EMOTION_MOVES.get(emotion_tag)
        if move_name is None or self._robot is None or self._moves is None:
            return
        try:
            move = self._moves.get(move_name)
        except Exception:
            logger.exception("Failed to load recorded move %r", move_name)
            return

        def _play() -> None:
            with self._lock:
                self._paused = True
            try:
                self._robot.play_move(move, initial_goto_duration=0.3)
            except Exception as exc:
                # Same transient link stall as _send_pose, and just as
                # non-fatal -- the reply is already spoken; only the flourish
                # is lost. Log one line, not a traceback: it fails in exactly
                # the conditions that are already being reported there.
                logger.warning("Skipped recorded move %r: %s", move_name, exc)
            finally:
                with self._lock:
                    self._paused = False

        threading.Thread(target=_play, name="motion-recorded-move", daemon=True).start()

    def look(self, *, yaw: float = 0.0, pitch: float = 0.0, ttl: float = 1.2) -> None:
        """Aim the head at an absolute direction, held for `ttl` seconds.

        Shares the tracking slot with track_face, so a real detection replaces
        a search sweep immediately rather than the two fighting.
        """
        pose = HeadPose(
            yaw=max(-25.0, min(25.0, yaw)),
            pitch=max(-15.0, min(25.0, pitch)),
        )
        with self._lock:
            self._tracking = _TrackingTarget(pose=pose, expires_at=time.monotonic() + ttl)

    def track_face(
        self,
        bbox: tuple[int, int, int, int] | None,
        frame_shape: tuple[int, ...] | None,
        *,
        ttl: float = 1.2,
    ) -> None:
        """Aim the head at a detected face, held for `ttl` seconds.

        Called continuously by body/face_tracker.py to follow someone, and the
        ttl is what makes that work: each detection refreshes the target, and
        the head drifts back to its resting pose only once detections stop.
        """
        if bbox is None or frame_shape is None:
            return
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return
        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2
        x_norm = (cx - width / 2) / (width / 2)
        y_norm = (cy - height / 2) / (height / 2)
        pose = HeadPose(
            yaw=max(-16.0, min(16.0, -x_norm * 14.0)),
            pitch=max(-10.0, min(10.0, -y_norm * 9.0)),
            roll=max(-4.0, min(4.0, -x_norm * 3.0)),
        )
        with self._lock:
            self._tracking = _TrackingTarget(pose=pose, expires_at=time.monotonic() + ttl)

    def begin_speech(self) -> None:
        self.wobbler.start()

    def feed_speech_audio(self, samples: np.ndarray) -> None:
        self.wobbler.feed(samples)

    def end_speech(self) -> None:
        self.wobbler.stop()

    def _run(self) -> None:
        period = 1.0 / self._control_hz
        while not self._stop.is_set():
            started = time.monotonic()
            with self._lock:
                paused = self._paused
            if not paused:
                pose = self._compose_pose(started)
                self._current_pose = _lerp_pose(self._current_pose, pose, alpha=0.16)
                self._send_pose(self._current_pose)
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, period - elapsed))

    def _compose_pose(self, now: float) -> HeadPose:
        with self._lock:
            base = self._base_pose
            tracking = self._tracking
            if tracking is not None and tracking.expires_at <= now:
                self._tracking = None
                tracking = None

        speech_active = self.wobbler.is_active(now)
        # Where the camera points must not depend on the robot's mood. The
        # emotion pose *is* the base, so expressions with a downward tilt
        # (thinking -4, sad -12) were aiming the lens at people's chests, and
        # face detection went from working to finding nothing the moment the
        # robot expressed anything. Applying the aim here, on top of whatever
        # is being expressed, keeps expressions relative to each other while
        # the camera stays on faces.
        pose = _add_pose(base, HeadPose(pitch=_CAMERA_PITCH_BIAS))

        if tracking is not None:
            pose = _add_pose(pose, tracking.pose)
        elif not speech_active:
            pose = _add_pose(pose, self.idle.offset(now))

        if speech_active:
            pose = _add_pose(pose, self.wobbler.offset(now))

        return _clamp_pose(pose)

    def _load_recorded_moves(self) -> object | None:
        """Load pollen-robotics/reachy-mini-emotions-library. Independent of
        the robot connection -- just a HuggingFace dataset download/cache."""
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves

            return RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
        except Exception as exc:
            logger.warning("Recorded emotion moves disabled: %s", exc)
            return None

    def _connect_robot(self) -> object | None:
        try:
            from reachy_mini import ReachyMini

            connection_mode = "localhost_only" if self.target.mode == "simulation" else "network"
            robot = ReachyMini(
                host=self.target.daemon_host,
                port=self.target.daemon_port,
                connection_mode=connection_mode,
                timeout=1.5,
                media_backend="no_media",
                log_level="WARNING",
            )
            robot.__enter__()
            _enable_motors(robot)
            return robot
        except Exception as exc:
            logger.warning("Reachy Mini motion disabled: %s", exc)
            return None

    def _send_pose(self, pose: HeadPose) -> None:
        robot = self._robot
        if robot is None:
            return
        try:
            robot.set_target(
                head=_create_head_pose(pose),
                antennas=np.deg2rad(np.array(pose.antennas, dtype=np.float64)),
                body_yaw=0.0,
            )
        except Exception as exc:
            # Best-effort cosmetic channel: drop the frame, never raise, and
            # never log per-tick. The SDK flags the link dead when no status
            # message arrives within 1s (WSClient.is_connected) and fails every
            # send until one does -- routine over a congested wireless link,
            # where the WebRTC media stream competes with these updates. It
            # clears itself once messages resume, so retrying silently is the
            # correct response; at 50Hz, logging each failure produced
            # thousands of identical tracebacks and buried the real logs.
            if self._failing_since is None:
                self._failing_since = time.monotonic()
            self._log_send_failure(exc)
            self._maybe_reconnect()
        else:
            self._failing_since = None
            if self._send_failures:
                logger.info(
                    "Reachy Mini motion link recovered after %d dropped frame(s).",
                    self._send_failures,
                )
                self._send_failures = 0

    def _log_send_failure(self, exc: Exception) -> None:
        """Count a dropped pose frame, logging at most once per interval."""
        self._send_failures += 1
        now = time.monotonic()
        if now - self._last_send_warning_at < _SEND_WARNING_INTERVAL_S:
            return
        self._last_send_warning_at = now
        logger.warning(
            "Dropping Reachy Mini pose updates (%d so far): %s", self._send_failures, exc
        )

    def _maybe_reconnect(self) -> None:
        """Rebuild a connection that has stopped coming back on its own.

        Two different failures look alike here. A momentary stall -- no daemon
        status message within the SDK's 1s liveness window -- clears itself
        once messages resume, and reconnecting for that would be churn. But a
        websocket that actually closes ("no close frame received or sent")
        never returns: every later send raises, the head freezes, and the app
        carries on talking as though nothing is wrong. That was observed
        dropping 1400+ consecutive frames with no recovery.

        Distinguishing them by exception text is unreliable, so this waits out
        the transient case by duration and then rebuilds. Only connections
        this controller owns are replaced -- a shared one belongs to the
        caller, which would also have to rebuild its media session.
        """
        if self._robot is None:
            return
        now = time.monotonic()
        if not self._owns_robot:
            # A shared connection also carries audio and camera, and the SDK's
            # websocket client cannot reconnect (disconnect() is terminal), so
            # only the owner can rebuild it. Raise a flag instead of failing
            # silently forever: the alternative, observed live, is 2500+
            # consecutive dropped frames while the robot sits frozen and the
            # app cheerfully carries on listening and replying.
            if self._failing_since is not None and now - self._failing_since >= _RECONNECT_AFTER_S:
                if not self.link_lost.is_set():
                    logger.error(
                        "Motion link dead for %.0fs and cannot be rebuilt from here.",
                        now - self._failing_since,
                    )
                    self.link_lost.set()
            return
        if now - self._last_reconnect_at < _RECONNECT_COOLDOWN_S:
            return
        if self._failing_since is None or now - self._failing_since < _RECONNECT_AFTER_S:
            return

        self._last_reconnect_at = now
        logger.warning("Motion link down for %.0fs -- reconnecting.", now - self._failing_since)
        old = self._robot
        self._robot = None
        try:
            old.__exit__(None, None, None)
        except Exception:
            pass
        robot = self._connect_robot()
        if robot is None:
            logger.warning("Reconnect failed; will retry.")
            return
        self._robot = robot
        self._failing_since = None
        self._send_failures = 0
        logger.info("Motion link reconnected.")


def _create_head_pose(pose: HeadPose) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    roll = math.radians(pose.roll)
    pitch = math.radians(pose.pitch)
    yaw = math.radians(pose.yaw)

    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    matrix[:3, :3] = rz @ ry @ rx
    matrix[2, 3] = pose.z / 1000.0
    return matrix


def _add_pose(a: HeadPose, b: HeadPose) -> HeadPose:
    return HeadPose(
        pitch=a.pitch + b.pitch,
        yaw=a.yaw + b.yaw,
        roll=a.roll + b.roll,
        z=a.z + b.z,
        antennas=(a.antennas[0] + b.antennas[0], a.antennas[1] + b.antennas[1]),
    )


def _scale_pose(pose: HeadPose, scale: float) -> HeadPose:
    return HeadPose(
        pitch=pose.pitch * scale,
        yaw=pose.yaw * scale,
        roll=pose.roll * scale,
        z=pose.z * scale,
        antennas=(pose.antennas[0] * scale, pose.antennas[1] * scale),
    )


def _lerp_pose(current: HeadPose, target: HeadPose, alpha: float) -> HeadPose:
    return HeadPose(
        pitch=_lerp(current.pitch, target.pitch, alpha),
        yaw=_lerp(current.yaw, target.yaw, alpha),
        roll=_lerp(current.roll, target.roll, alpha),
        z=_lerp(current.z, target.z, alpha),
        antennas=(
            _lerp(current.antennas[0], target.antennas[0], alpha),
            _lerp(current.antennas[1], target.antennas[1], alpha),
        ),
    )


def _lerp(a: float, b: float, alpha: float) -> float:
    return a + (b - a) * alpha


def _clamp_pose(pose: HeadPose) -> HeadPose:
    return HeadPose(
        pitch=max(-18.0, min(18.0, pose.pitch)),
        yaw=max(-24.0, min(24.0, pose.yaw)),
        roll=max(-16.0, min(16.0, pose.roll)),
        z=max(-15.0, min(15.0, pose.z)),
        antennas=(
            max(-70.0, min(70.0, pose.antennas[0])),
            max(-70.0, min(70.0, pose.antennas[1])),
        ),
    )
