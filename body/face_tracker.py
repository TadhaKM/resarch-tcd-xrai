"""Continuously aim the head at whoever is in view.

Runs here rather than on the robot: the SDK offers start_head_tracking(), but
this robot's daemon (1.8.3) has no such command and rejects it over the
websocket while the SDK call returns normally -- so it would look enabled and
do nothing. Detecting locally works on any daemon version.

This owns *all* face detection. MediaPipe's detector is not safe to call from
two threads, so the voice loop reads the latest result from here (see
FaceTracker.current) instead of running its own detection per turn.
"""

import logging
import threading
import time
from typing import Optional

from config import MODELS

from .camera import Camera
from .face import DetectedFace, FaceIdentifier
from .motion import MotionController

logger = logging.getLogger(__name__)

#: Seconds a detection keeps steering the head. Comfortably longer than the
#: interval between detections, so the aim persists through a missed frame
#: instead of the head twitching back to centre between updates.
_TRACK_TTL_S = 1.5

#: Backoff after a failed frame grab, so a disconnected camera doesn't spin
#: this thread at full speed.
_ERROR_BACKOFF_S = 1.0

#: How often to report what the tracker is actually seeing.
_STATS_INTERVAL_S = 5.0


class FaceTracker:
    """Background loop: detect the active face, aim the head, cache identity."""

    def __init__(self, camera: Camera, face: FaceIdentifier, motion: MotionController) -> None:
        self._camera = camera
        self._face = face
        self._motion = motion
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._person_id: Optional[int] = None
        self._active_face: Optional[DetectedFace] = None
        self._seen_at = 0.0

    @property
    def enabled(self) -> bool:
        return self._thread is not None

    def start(self) -> None:
        if self._thread is not None or not self._face.available:
            if not self._face.available:
                logger.info("Face tracking unavailable: no usable face detector.")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="face-tracker", daemon=True)
        self._thread.start()
        self._motion.set_face_tracking(True)
        logger.info("Face tracking started (%.0f Hz).", MODELS.face_detection_fps)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._motion.set_face_tracking(False)

    def current(self, max_age_s: float = 3.0) -> tuple[Optional[int], Optional[DetectedFace]]:
        """Return the most recent identity/face, or (None, None) if stale."""
        with self._lock:
            if time.monotonic() - self._seen_at > max_age_s:
                return None, None
            return self._person_id, self._active_face

    def _run(self) -> None:
        period = 1.0 / max(0.5, MODELS.face_detection_fps)
        frames = faces = no_frame = 0
        reported_at = time.monotonic()

        while not self._stop.wait(period):
            try:
                frame = self._camera.get_frame()

                # Report periodically: skipping a cycle because the camera
                # returned nothing looks exactly like skipping it because
                # nobody is in view, and from outside both look like "the head
                # isn't moving" with nothing in the log to tell them apart.
                now = time.monotonic()
                if now - reported_at >= _STATS_INTERVAL_S:
                    logger.info(
                        "Face tracking: %d frame(s), %d with a face, %d camera miss(es).",
                        frames,
                        faces,
                        no_frame,
                    )
                    frames = faces = no_frame = 0
                    reported_at = now

                if frame is None:
                    no_frame += 1
                    continue
                frames += 1

                # force=True because this loop already paces itself; the
                # detector's own rate limiter would otherwise drop most calls.
                detected = self._face.detect_active_face(frame, force=True)
                if detected is None:
                    continue
                faces += 1

                self._motion.track_face(detected.bbox, frame.shape, ttl=_TRACK_TTL_S)

                # Identity is resolved here too, since this thread owns the
                # detector -- the voice loop just reads the cached answer.
                person_id, _score = self._face.match_embedding(
                    self._face.embedding_for_face(detected)
                )
                with self._lock:
                    self._person_id = person_id
                    self._active_face = detected
                    self._seen_at = time.monotonic()
            except Exception as exc:
                logger.warning("Face tracking paused after error: %s", exc)
                self._stop.wait(_ERROR_BACKOFF_S)
