"""Motion control, mapping emotion tags to robot movement. Stub: no real daemon calls yet."""

from config import HardwareTarget


class MotionController:
    """Wraps head/antenna/body motion for a given hardware target."""

    def __init__(self, target: HardwareTarget) -> None:
        self.target = target

    def express(self, emotion_tag: str) -> None:
        """Play the movement associated with an emotion tag. Stub: just prints."""
        print(f"[{self.target.mode}] expressing: {emotion_tag}")
