"""Camera access: robot.media in "robot" mode, HardwareTarget.camera_source
(a local cv2.VideoCapture) in "simulation" mode -- same rationale as
audio_io.py's split, this app runs on the robot itself in robot mode, so
there's no separate machine's camera device to open.

In "robot" mode, this shares one ReachyMini media connection with AudioIO
rather than opening a second one -- the daemon's media backend (gstreamer or
webrtc) only supports one active session per client, so a second concurrent
connection to the same media pipeline fails with a GStreamer state-change
error (only surfaces once both Camera and AudioIO run at the same time,
which standalone component tests never happened to exercise). See
voice_loop.run_forever, which builds the one shared connection and passes it
to both.
"""

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from config import HardwareTarget


class Camera:
    """Wraps camera frame capture for a given hardware target."""

    def __init__(self, target: HardwareTarget, robot: Optional[Any] = None) -> None:
        self.target = target
        self._capture: Optional[cv2.VideoCapture] = None
        self._robot: Optional[Any] = robot
        self._owns_robot = robot is None

        if target.mode == "robot" and self._robot is None:
            from reachy_mini import ReachyMini

            self._robot = ReachyMini(
                host=target.daemon_host,
                port=target.daemon_port,
                media_backend=target.media_backend,
                log_level="WARNING",
            )
            self._robot.__enter__()

    def _source(self) -> int | str:
        return 0 if self.target.camera_source is None else self.target.camera_source

    def _get_capture(self) -> cv2.VideoCapture:
        if self._capture is None or not self._capture.isOpened():
            self._capture = cv2.VideoCapture(self._source())
        return self._capture

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the latest BGR camera frame, or None if capture fails."""
        if self.target.mode == "robot":
            return self._robot.media.get_frame()
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
        if self._robot is not None and self._owns_robot:
            self._robot.__exit__(None, None, None)
        self._robot = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
