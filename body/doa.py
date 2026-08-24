"""Which direction a voice came from, so the head can turn toward it.

The ReSpeaker board on the robot estimates direction of arrival on-chip and the
daemon serves it at GET /api/state/doa as (angle in radians, speech detected).
This reads it over HTTP rather than through the SDK's AudioDoA, which opens the
USB device directly: in robot_remote mode AudioDoA is constructed on the laptop,
finds no ReSpeaker attached to a Windows machine, and returns None forever. The
daemon runs ON the robot, where the board actually is.

WHY THE ZERO DIRECTION IS LEARNED RATHER THAN CONFIGURED
The board reports an angle in its own frame, and nothing in the SDK says where
zero points relative to the robot's face. Guessing costs more than doing
nothing: a head that turns confidently AWAY from whoever spoke is worse than a
head that stays still, and it would be wrong on exactly the visits where it is
being shown off.

So the offset is measured from the robot's own experience. Whenever the camera
can see a face AND the board reports speech, the difference between where the
face is and where the sound came from is one sample of the offset. After
enough agreeing samples the robot knows its own geometry and can turn toward a
voice it CANNOT see -- which is the case worth having, someone speaking from
outside the camera's view.

Until then it does nothing at all. Every accessor returns None, the face
tracker keeps its existing blind search, and nothing regresses.
"""

import logging
import math
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

#: How often to ask the daemon. The board updates faster than this, but the
#: head cannot usefully chase a voice more often and every poll is an HTTP
#: round trip over the same wifi that carries audio and video.
_POLL_S = 0.25

#: A reading older than this is stale and is not acted on -- the network drops
#: here routinely, and a head that turns toward where a voice was ten seconds
#: ago looks broken rather than attentive.
_MAX_AGE_S = 1.5

#: Calibration samples needed before the learned offset is trusted, and how
#: much they must agree. Circular standard deviation above this means the
#: samples are contradicting each other -- usually several people talking, or
#: reflections in a hard-walled room -- and an average of contradictory
#: directions points nowhere in particular.
_CALIBRATION_SAMPLES = 12
_CALIBRATION_MAX_SPREAD_RAD = 0.6

#: The camera's approximate horizontal field of view, used to turn a face's
#: position in the frame into an angle. Approximate is enough: this is only
#: ever used to LEARN an offset that is then averaged over a dozen samples,
#: and a few degrees of error in each washes out.
_CAMERA_HFOV_RAD = math.radians(70.0)

#: The head cannot turn further than this anyway (motion.look clamps to 25),
#: so a voice from behind produces a glance in its general direction rather
#: than an impossible instruction.
_MAX_YAW_DEG = 25.0


def _circular_mean(angles: list[float]) -> float:
    """Mean of angles that wrap. A plain average of 0.1 and 6.2 is 3.15 -- the
    exact opposite of where both point."""
    x = sum(math.cos(a) for a in angles)
    y = sum(math.sin(a) for a in angles)
    return math.atan2(y, x)


def _circular_spread(angles: list[float]) -> float:
    """How much a set of angles disagree, 0 (identical) upward."""
    x = sum(math.cos(a) for a in angles) / len(angles)
    y = sum(math.sin(a) for a in angles) / len(angles)
    r = math.hypot(x, y)
    if r >= 1.0:
        return 0.0
    return math.sqrt(-2.0 * math.log(max(r, 1e-9)))


def _wrap(angle: float) -> float:
    """Fold an angle into -pi..pi."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class DoaListener:
    """Polls the daemon for direction of arrival, and learns where front is."""

    def __init__(self, daemon_host: str, daemon_port: int) -> None:
        self._url = f"http://{daemon_host}:{daemon_port}/api/state/doa"
        self._lock = threading.Lock()
        self._angle: Optional[float] = None
        self._speech = False
        self._at = 0.0
        self._samples: list[float] = []
        self._offset: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        #: Set once, so an unavailable board is reported to the log a single
        #: time rather than four times a second for the length of a visit.
        self._warned = False

    # --- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="doa", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(_POLL_S):
            try:
                r = requests.get(self._url, timeout=1.0)
                if r.status_code != 200:
                    self._unavailable(f"HTTP {r.status_code}")
                    continue
                body = r.json()
                angle = body.get("angle")
                if angle is None:
                    self._unavailable("no angle in response")
                    continue
                with self._lock:
                    self._angle = float(angle)
                    self._speech = bool(body.get("speech_detected"))
                    self._at = time.monotonic()
            except Exception as exc:
                # Never fatal, and never noisy: the robot works exactly as it
                # did before this existed when the board or the network is
                # unavailable.
                self._unavailable(str(exc))

    def _unavailable(self, why: str) -> None:
        if not self._warned:
            self._warned = True
            logger.info("Direction of arrival unavailable (%s); "
                        "the head will keep searching visually.", why)

    # --- reading ---------------------------------------------------------

    def _fresh(self) -> Optional[tuple[float, bool]]:
        with self._lock:
            if self._angle is None or time.monotonic() - self._at > _MAX_AGE_S:
                return None
            return self._angle, self._speech

    def speaking_now(self) -> bool:
        """Whether the board currently hears speech, from any direction."""
        reading = self._fresh()
        return bool(reading and reading[1])

    # --- calibration -----------------------------------------------------

    def observe_face(self, face_centre_x: float, frame_width: int) -> None:
        """One sample of "the person I can see is the person I can hear".

        Only counted while the board reports speech: a face sitting silently in
        view says nothing about where a sound came from, and averaging those in
        would drag the offset toward wherever people happen to stand.
        """
        reading = self._fresh()
        if not reading or not reading[1] or not frame_width:
            return
        angle, _ = reading
        # Where the face is, as an angle from straight ahead. Positive to the
        # frame's right, matching motion.look's yaw sign convention.
        offset_in_frame = (face_centre_x / float(frame_width)) - 0.5
        face_yaw = offset_in_frame * _CAMERA_HFOV_RAD
        with self._lock:
            if self._offset is not None:
                return
            self._samples.append(_wrap(angle - face_yaw))
            if len(self._samples) < _CALIBRATION_SAMPLES:
                return
            spread = _circular_spread(self._samples)
            if spread > _CALIBRATION_MAX_SPREAD_RAD:
                # Contradictory samples -- several people talking, or a room
                # full of reflections. Drop the oldest and keep listening
                # rather than committing to an average of disagreement.
                self._samples.pop(0)
                return
            self._offset = _circular_mean(self._samples)
            logger.info(
                "Learned where the microphone's zero points: %.0f degrees off the "
                "camera's centre, from %d samples (spread %.2f rad). The head can "
                "now turn toward a voice it cannot see.",
                math.degrees(self._offset), len(self._samples), spread,
            )

    def calibrated(self) -> bool:
        with self._lock:
            return self._offset is not None

    def suggested_yaw_deg(self) -> Optional[float]:
        """Where to look for the voice being heard now, or None.

        None whenever anything is unknown -- no reading, no speech, or no
        learned offset -- because the caller's fallback is its existing visual
        search, which is better than a confident guess in the wrong direction.
        """
        reading = self._fresh()
        if not reading:
            return None
        angle, speech = reading
        if not speech:
            return None
        with self._lock:
            offset = self._offset
        if offset is None:
            return None
        yaw = math.degrees(_wrap(angle - offset))
        return max(-_MAX_YAW_DEG, min(_MAX_YAW_DEG, yaw))
