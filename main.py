"""Entry point: pick a hardware target and start the voice loop."""

from body.voice_loop import run_forever
from config import default_target


def main() -> None:
    run_forever(default_target())


if __name__ == "__main__":
    main()
