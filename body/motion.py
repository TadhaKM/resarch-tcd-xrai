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
    they are converted to radians at the SDK boundary.
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
        breathing = HeadPose(
            pitch=1.2 * math.sin(now * math.tau * 0.18),
            roll=0.7 * math.sin(now * math.tau * 0.11),
            z=2.5 * math.sin(now * math.tau * 0.18),
        )

        if now >= self._next_look_at:
            self._look_until = now + self._rng.uniform(1.4, 2.8)
            self._next_look_at = self._look_until + self._rng.uniform(4.0, 9.0)
            self._look_target = HeadPose(
                pitch=self._rng.uniform(-3.0, 4.0),
                yaw=self._rng.uniform(-13.0, 13.0),
                roll=self._rng.uniform(-3.0, 3.0),
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


class MotionController:
    """Coordinates Reachy Mini head and antenna personality motion."""

    def __init__(self, target: HardwareTarget, *, connect: bool = True) -> None:
        self.target = target
        self.mapper = EmotionMapper()
        self.wobbler = HeadWobbler()
        self.idle = IdleAnimator(random.Random())
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._robot: object | None = None
        self._base_pose = self.mapper.get("neutral")
        self._current_pose = self._base_pose
        self._tracking: Optional[_TrackingTarget] = None
        self._control_hz = 50.0

        if connect:
            self._robot = self._connect_robot()
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
        if robot is not None and hasattr(robot, "__exit__"):
            try:
                robot.__exit__(None, None, None)
            except Exception:
                logger.exception("Failed to close Reachy Mini SDK connection")

    def express(self, emotion_tag: str) -> None:
        """Set the primary expression pose for the next reply."""
        with self._lock:
            self._base_pose = self.mapper.get(emotion_tag)

    def track_face(
        self,
        bbox: tuple[int, int, int, int] | None,
        frame_shape: tuple[int, ...] | None,
        *,
        ttl: float = 1.2,
    ) -> None:
        """Bias the head toward the active face before words are spoken."""
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
        pose = base

        if tracking is not None:
            pose = _add_pose(pose, tracking.pose)
        elif not speech_active:
            pose = _add_pose(pose, self.idle.offset(now))

        if speech_active:
            pose = _add_pose(pose, self.wobbler.offset(now))

        return _clamp_pose(pose)

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
            if hasattr(robot, "enable_motors"):
                robot.enable_motors()
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
        except Exception:
            logger.exception("Failed to send Reachy Mini target")


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
