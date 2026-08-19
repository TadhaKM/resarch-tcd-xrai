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
import math
import threading
import time
from typing import Optional

import numpy as np

from config import MODELS

from brain import db

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

#: Recognising somebody well enough to add what it just saw to what it knows
#: about them. Above the match threshold by a margin on purpose: a match that
#: only just cleared the bar is the one most likely to be the wrong person, and
#: storing that face under their name is how an identity quietly rots into
#: somebody else's.
_REINFORCE_MARGIN = 0.08

#: A view already this close to one on file teaches nothing, and storing it
#: would fill the cap with near-duplicates of whatever angle the visitor is
#: standing at right now -- crowding out the varied ones that make recognition
#: work from anywhere. So faces are kept only while they still differ.
_REINFORCE_CEILING = 0.92

#: Minimum gap between stored views of the same person, so standing in front of
#: the robot for a minute is not eight database writes a second.
_REINFORCE_INTERVAL_S = 20.0

#: Search sweep, used when nobody is in view. Waits a beat first so a single
#: missed detection doesn't set the head wandering, then arcs slowly across
#: and around head height rather than sweeping fast enough to blur frames.
_SEARCH_AFTER_S = 4.0
_SEARCH_PERIOD_S = 12.0
_SEARCH_YAW_DEG = 20.0

#: Pitch the sweep centres on, as an offset from the resting pose -- NOT as an
#: absolute aim. That distinction was the bug: motion.py already holds an
#: upward camera bias, and this was written as though it did not, so the two
#: added up and the head sat at its upper limit staring at the ceiling for the
#: whole time nobody was detected.
#:
#: The upward lean that finding a face genuinely needs lives here rather than
#: in the resting bias, because it only applies while the head is visibly
#: sweeping. Somebody watching sees a robot looking around for them, which is
#: what it is doing; the same lean held still just looks like a robot facing
#: the wrong way.
_SEARCH_PITCH_DEG = 7.0

#: How far the sweep rocks above and below that centre. Wider than the 6 it
#: was, to cover both a seated visitor and a standing one now that the sweep is
#: no longer starting halfway up.
_SEARCH_PITCH_SWING_DEG = 9.0


#: How long a recognised identity survives frames that fail to match. Long
#: enough to ride out a turned head or a moment of blur; short enough that it
#: is never carried across somebody leaving and somebody else arriving. It is
#: bounded by the similarity check too -- see _hold_identity.
_IDENTITY_HOLD_S = 20.0

#: How alike this frame must be to the last identified one to count as the same
#: person still standing there. BELOW face_match_threshold (0.65), and the
#: first version of this got that backwards: it was set to 0.75 on the theory
#: that frame-to-frame is an easier question than a database lookup, and so
#: could afford a higher bar. It cannot. A degraded frame -- blur, a turned
#: head, half a face -- is degraded against EVERYTHING, including a good stored
#: view of the same person, so a frame that just failed the database at 0.65
#: can never clear 0.75 against one embedding.
#:
#: The measured gap is what sets this. On this robot's own data: a bad frame of
#: somebody known scored 0.58 against a 24-view database, between frames at
#: 0.80 and 0.91 either side. Two DIFFERENT people score 0.33 to 0.39 against
#: each other. 0.50 sits in that gap with room on both sides -- it rescues the
#: dropped frames without ever carrying one person's name to another's face.
_SAME_PERSON_THRESHOLD = 0.50


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
        #: Who was last positively identified, their face vector at the time,
        #: and when. This is what stops a recognised visitor flickering back to
        #: "stranger" on a single bad frame -- see _hold_identity.
        self._known_id: Optional[int] = None
        self._known_embedding = None
        self._known_at: float = 0.0
        #: Best database score for the last frame, whether or not it matched.
        #: Read by demos/vision.py so a decision to offer can say WHY -- a
        #: near-miss on a known face and a genuinely new face look identical
        #: from the outside and want opposite fixes.
        self._last_score: float = 0.0
        #: The embedding of the face in view. Kept because this thread has
        #: already computed it to do the matching, and a demo that wants to
        #: tell one unrecognised visitor from another would otherwise have to
        #: run the model a second time on another thread.
        self._embedding = None
        self._seen_at = 0.0
        self._searching_since: Optional[float] = None
        #: When each person last had a view stored. See _reinforce.
        self._reinforced_at: dict[int, float] = {}

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

    def last_score(self) -> float:
        """Best database similarity for the most recent frame."""
        with self._lock:
            return self._last_score

    def _hold_identity(self, embedding) -> Optional[int]:
        """Keep the last identity if this is plainly the same face, still here.

        Two guards, and both matter. The TIME limit means an identity is never
        carried across a gap in which the room could have changed -- somebody
        new stepping in front of the robot half a minute later gets a clean
        slate. The SIMILARITY check means it is never carried to a different
        face that happens to arrive in the meantime: consecutive frames of one
        person score far higher against each other than two different people
        do, which is why this threshold can sit well above the recognition one
        and still rescue the frames recognition drops.
        """
        if self._known_id is None or self._known_embedding is None:
            return None
        if time.monotonic() - self._known_at > _IDENTITY_HOLD_S:
            return None
        probe = float(np.linalg.norm(embedding))
        held = float(np.linalg.norm(self._known_embedding))
        if not probe or not held:
            return None
        similarity = float(np.dot(embedding, self._known_embedding) / (probe * held))
        return self._known_id if similarity >= _SAME_PERSON_THRESHOLD else None

    def current_embedding(self, max_age_s: float = 3.0):
        """The face vector for whoever is in view, or None if stale.

        Separate from current() rather than a third element of its tuple, so
        the two existing callers keep working unchanged.
        """
        with self._lock:
            if time.monotonic() - self._seen_at > max_age_s:
                return None
            return self._embedding

    def _search(self, now: float) -> None:
        """Sweep the head slowly while nobody is in view.

        Resting pitch aims where faces usually are, but "usually" depends on
        whether the person is standing or sitting and how close they are --
        get it wrong and the camera frames a chest and finds nothing, with no
        way to recover because the head never moves to look elsewhere. This
        scans a slow arc so the robot finds people instead of waiting to be
        stood in front of correctly.
        """
        if self._searching_since is None:
            self._searching_since = now
            return
        # Give tracking a moment to reacquire before starting to sweep, so a
        # single missed detection doesn't make the head wander.
        elapsed = now - self._searching_since
        if elapsed < _SEARCH_AFTER_S:
            return

        phase = (elapsed - _SEARCH_AFTER_S) / _SEARCH_PERIOD_S
        yaw = _SEARCH_YAW_DEG * math.sin(2.0 * math.pi * phase)
        pitch = _SEARCH_PITCH_DEG + _SEARCH_PITCH_SWING_DEG * math.sin(4.0 * math.pi * phase)
        self._motion.look(yaw=yaw, pitch=pitch, ttl=_TRACK_TTL_S)

    def _reinforce(self, person_id: int, score: float, face: DetectedFace, now: float) -> None:
        """Add this view of a known face to what is stored about them.

        This is what stops the robot forgetting people. A name is never lost --
        it sits in the database untouched -- but until this existed a person had
        exactly one stored face, captured at whatever angle and in whatever
        light they happened to be in when they gave their name. Come back
        standing a little differently and the match falls under the threshold,
        and from where the visitor stands that is the robot not knowing them.

        Three conditions, each guarding a different way this could go wrong:
        a comfortable margin over the threshold, so a marginal match cannot
        attach a stranger's face to somebody's name; a ceiling, so the stored
        set stays varied rather than filling with one angle; and an interval,
        so standing here does not write to the database eight times a second.
        """
        floor = MODELS.face_match_threshold + _REINFORCE_MARGIN
        if not (floor <= score < _REINFORCE_CEILING):
            return
        if now - self._reinforced_at.get(person_id, 0.0) < _REINFORCE_INTERVAL_S:
            return
        self._reinforced_at[person_id] = now
        try:
            self._face.remember(person_id, face)
            logger.info(
                "Remembered another view of person %d (match %.2f, %d on file).",
                person_id, score, db.count_embeddings(person_id),
            )
        except Exception as exc:
            logger.warning("Could not store an extra view of person %d: %s", person_id, exc)

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
                    self._search(now)
                    continue
                faces += 1
                self._searching_since = None

                self._motion.track_face(detected.bbox, frame.shape, ttl=_TRACK_TTL_S)

                # Identity is resolved here too, since this thread owns the
                # detector -- the voice loop just reads the cached answer.
                embedding = self._face.embedding_for_face(detected)
                person_id, score = self._face.match_embedding(embedding)
                self._last_score = score
                if person_id:
                    self._reinforce(person_id, score, detected, now)
                else:
                    # Recognition against the whole database is a hard question
                    # asked of every frame, and it fails on ordinary things --
                    # a turned head, motion blur, somebody looking down. Before
                    # this, one such frame dropped a known visitor to
                    # "stranger", and demos/vision.py offered to learn the name
                    # of somebody it had greeted by name seconds earlier.
                    #
                    # "Is this still the person from the last frame?" is a far
                    # easier question than "who is this out of everyone", so it
                    # is asked separately and at its own threshold.
                    person_id = self._hold_identity(embedding)
                with self._lock:
                    self._person_id = person_id
                    self._active_face = detected
                    self._embedding = embedding
                    self._seen_at = time.monotonic()
                    if person_id:
                        self._known_id = person_id
                        self._known_embedding = embedding
                        self._known_at = time.monotonic()
            except Exception as exc:
                logger.warning("Face tracking paused after error: %s", exc)
                self._stop.wait(_ERROR_BACKOFF_S)
