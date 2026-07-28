"""Camera access. Stub: no real daemon frame capture yet."""

from typing import Any, Optional

from config import HardwareTarget


class Camera:
    """Wraps camera frame capture for a given hardware target."""

    def __init__(self, target: HardwareTarget) -> None:
        self.target = target

    def get_frame(self) -> Optional[Any]:
        """Return the latest camera frame. Stub: returns None."""
        return None
