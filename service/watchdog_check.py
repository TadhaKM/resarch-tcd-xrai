#!/usr/bin/env python3
"""Watchdog: force-restart the supervised service if its heartbeat has gone
stale. Runs periodically via reachy-watchdog.timer.

This exists because systemd's Restart= only fires when a process actually
exits -- a hung-but-still-running process (deadlock, stuck syscall, etc.)
looks perfectly healthy to systemd, which has no idea anything is wrong. This
is the only thing that can detect and recover from that case: read the
heartbeat file's age, and if it's too stale, SIGKILL the service. The main
unit's own Restart=on-failure then takes over automatically, since a SIGKILL
is an abnormal exit -- the watchdog's only job is forcing that exit to happen.
"""

import logging
import os
import subprocess
import sys
import time

HEARTBEAT_PATH = os.path.expanduser("~/reachy_supervisor/heartbeat")
STALE_THRESHOLD_S = 90  # 3x the 30s heartbeat interval, tolerating normal jitter
SERVICE_NAME = "reachy-supervised.service"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("watchdog")


def main() -> None:
    if not os.path.exists(HEARTBEAT_PATH):
        logger.warning("No heartbeat file yet at %s -- service may still be starting, skipping", HEARTBEAT_PATH)
        return

    age = time.time() - os.path.getmtime(HEARTBEAT_PATH)
    if age <= STALE_THRESHOLD_S:
        logger.info("Heartbeat OK (age=%.1fs)", age)
        return

    logger.error(
        "Heartbeat STALE (age=%.1fs > %ds) -- force-killing %s", age, STALE_THRESHOLD_S, SERVICE_NAME
    )
    subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", SERVICE_NAME], check=False)


if __name__ == "__main__":
    main()
