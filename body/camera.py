"""Camera access, routed through HardwareTarget.camera_source."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import HardwareTarget


class Camera:
    """Wraps camera frame capture for a given hardware target."""

    def __init__(self, target: HardwareTarget) -> None:
        self.target = target
        self._capture: Optional[cv2.VideoCapture] = None

        if target.mode != "simulation":
            raise NotImplementedError(
                "Camera currently supports simulation mode only. Robot mode should "
                "route through the Reachy daemon camera/media pipeline."
            )

    def _source(self) -> int | str:
        return 0 if self.target.camera_source is None else self.target.camera_source

    def _get_capture(self) -> cv2.VideoCapture:
        if self._capture is None or not self._capture.isOpened():
            self._capture = cv2.VideoCapture(self._source())
        return self._capture

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest BGR camera frame, or None if capture fails."""
        capture = self._get_capture()
        ok, frame = capture.read()
        if not ok:
            return None
        return frame

    def save_frame(self, path: str | Path) -> bool:
        """Capture one frame and save it as an image."""
        frame = self.get_frame()
        if frame is None:
            return False
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(output), frame))

    def close(self) -> None:
        """Release the underlying camera handle."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
