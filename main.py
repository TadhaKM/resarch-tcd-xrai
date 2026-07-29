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

    run_forever(default_target())


if __name__ == "__main__":
    main()
