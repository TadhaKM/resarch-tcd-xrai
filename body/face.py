"""Face detection/embedding for person identification. Stub: no real model yet."""

from typing import Any

from config import HardwareTarget


class FaceIdentifier:
    """Resolves a camera frame to a person_id."""

    def __init__(self, target: HardwareTarget) -> None:
        self.target = target

    def identify(self, frame: Any) -> int:
        """Return the person_id seen in this frame. Stub: always returns 0."""
        return 0
