"""Entry point: pick a hardware target and start the voice loop."""

import logging
import sys

from body.voice_loop import run_forever
from config import default_target


def main() -> None:
    # force_utf8 on the stream: replies routinely contain curly apostrophes,
    # which crash or mangle to "?" under Windows' default cp1252 console
    # encoding -- and a transcript log is useless if it can't show the text
    # verbatim.
    handler = logging.StreamHandler(sys.stdout)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # Dashboard first, so it is already answering while the speech models load
    # (~20s) and can show "starting" rather than looking like nothing is there.
    # It runs in a daemon thread and never blocks the robot: if it fails, the
    # voice loop carries on regardless.
    try:
        from web.server import serve

        serve()
    except Exception:
        logging.getLogger(__name__).exception("Dashboard unavailable; continuing without it")

    run_forever(default_target())


if __name__ == "__main__":
    main()
