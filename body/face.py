"""Face detection, embedding, enrollment, and identity matching."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort

from brain import db
from config import MODELS, HardwareTarget


@dataclass(frozen=True)
class DetectedFace:
    """A detected face and the embedding-ready crop."""

    bbox: tuple[int, int, int, int]
    confidence: float
    crop: np.ndarray


class FaceIdentifier:
    """Resolves camera frames to enrolled person_ids."""

    def __init__(self, target: HardwareTarget) -> None:
        self.target = target
        self._min_interval = 1.0 / MODELS.face_detection_fps
        self._last_detection_at = 0.0
        detector_path = Path(MODELS.face_detector_model_path)
        if not detector_path.exists():
            raise FileNotFoundError(
                f"MediaPipe face detector model not found at {detector_path}."
            )
        base_options = mp.tasks.BaseOptions(model_asset_path=str(detector_path))
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(options)
        model_path = Path(MODELS.face_embedding_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face embedding model not found at {model_path}. Download the "
                "configured MobileFaceNet-class ONNX model before using face ID."
            )
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def detect_active_face(self, frame: Optional[np.ndarray], force: bool = False) -> Optional[DetectedFace]:
        """Detect faces and return the largest/closest one.

        Continuous callers should leave force=False. Calls arriving faster than
        MODELS.face_detection_fps return None, which deliberately skips frames
        instead of processing every one. Single-frame validation/conversation
        calls may pass force=True.
        """
        if frame is None:
            return None

        now = time.monotonic()
        if not force and now - self._last_detection_at < self._min_interval:
            return None
        self._last_detection_at = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(image)
        if not result.detections:
            return None

        height, width = frame.shape[:2]
        candidates: list[tuple[int, DetectedFace]] = []
        for detection in result.detections:
            box = detection.bounding_box
            x = max(0, int(box.origin_x))
            y = max(0, int(box.origin_y))
            w = min(width - x, int(box.width))
            h = min(height - y, int(box.height))
            if w <= 0 or h <= 0:
                continue
            crop = self._square_crop(frame, x, y, w, h)
            confidence = float(detection.categories[0].score) if detection.categories else 0.0
            candidates.append((w * h, DetectedFace((x, y, w, h), confidence, crop)))

        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def embedding_for_face(self, face: DetectedFace) -> np.ndarray:
        """Return a normalized embedding for a detected face."""
        image = cv2.resize(face.crop, (112, 112), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))[None, :, :, :]
        embedding = self._session.run(None, {self._input_name: image})[0][0].astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if norm else embedding

    def match_embedding(self, embedding: np.ndarray) -> tuple[Optional[int], float]:
        """Return the best matching person_id and cosine score."""
        best_person_id: Optional[int] = None
        best_score = -1.0
        for person_id, stored in db.get_embeddings():
            stored_norm = float(np.linalg.norm(stored))
            probe_norm = float(np.linalg.norm(embedding))
            if not stored_norm or not probe_norm:
                continue
            score = float(np.dot(embedding, stored) / (probe_norm * stored_norm))
            if score > best_score:
                best_person_id = person_id
                best_score = score

        if best_person_id is None or best_score < MODELS.face_match_threshold:
            return None, best_score
        return best_person_id, best_score

    def identify(self, frame: Optional[np.ndarray], force: bool = False) -> tuple[Optional[int], Optional[DetectedFace], float]:
        """Return (person_id, active_face, score). Unknown faces are not enrolled."""
        face = self.detect_active_face(frame, force=force)
        if face is None:
            return None, None, -1.0
        embedding = self.embedding_for_face(face)
        person_id, score = self.match_embedding(embedding)
        return person_id, face, score

    def enroll(self, name: str, face: DetectedFace) -> int:
        """Create a person record and save this face embedding."""
        person_id = db.create_person(name.strip() or None)
        db.save_embedding(person_id, self.embedding_for_face(face))
        return person_id

    @staticmethod
    def _square_crop(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        height, width = frame.shape[:2]
        size = int(max(w, h) * 1.35)
        cx = x + w // 2
        cy = y + h // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(width, x1 + size)
        y2 = min(height, y1 + size)
        x1 = max(0, x2 - size)
        y1 = max(0, y2 - size)
        return frame[y1:y2, x1:x2]
