#!/usr/bin/env python3
"""The process systemd supervises for this reliability test.

Stands in for the reachy_companion voice loop's connection to the daemon --
this test is about process supervision (crash/hang recovery), not the ML
pipeline, so it talks to the daemon directly over its REST API (plain
`requests`) rather than pulling in the full reachy_mini SDK's native
dependency chain (mujoco, gstreamer/PyGObject, pycairo -- none of which have
clean prebuilt wheels for a minimal WSL install).

Normal mode: poll GET /api/daemon/status every POLL_INTERVAL seconds (proves
it's genuinely connected to the sim daemon, not just sleeping) and touch a
heartbeat file every HEARTBEAT_INTERVAL seconds.

Hang-test mode (REACHY_HANG_TEST=1): spin in a real blocking loop instead --
no daemon polling, no heartbeat -- to simulate a hung-but-still-alive process
that systemd's Restart= alone would never catch (it only fires on exit), which
is exactly what the separate heartbeat watchdog timer is for.
"""

import logging
import os
import signal
import sys
import time

import requests

DAEMON_STATUS_URL = "http://localhost:8000/api/daemon/status"
HEARTBEAT_PATH = os.environ.get(
    "REACHY_HEARTBEAT_PATH", os.path.expanduser("~/reachy_supervisor/heartbeat")
)
POLL_INTERVAL = 5
HEARTBEAT_INTERVAL = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("supervised_process")


def touch_heartbeat() -> None:
    with open(HEARTBEAT_PATH, "w") as f:
        f.write(str(time.time()))


def run_hang_test() -> None:
    # Ignore SIGTERM so this genuinely simulates a stuck process that a plain
    # `systemctl stop` can't clear -- like a real deadlock or blocked syscall
    # would. Only SIGKILL (which can't be caught or ignored) gets rid of it,
    # which is exactly what the watchdog is supposed to reach for.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    logger.warning("REACHY_HANG_TEST=1 -- ignoring SIGTERM, entering an unresponsive busy loop, no heartbeat")
    while True:
        pass  # deliberately never yields, never touches the heartbeat


def run_normal() -> None:
    logger.info("Connecting to sim daemon at %s", DAEMON_STATUS_URL)
    last_heartbeat = 0.0

    while True:
        try:
            resp = requests.get(DAEMON_STATUS_URL, timeout=5)
            resp.raise_for_status()
            status = resp.json()
            logger.info(
                "daemon status: state=%s simulation_enabled=%s",
                status.get("state"),
                status.get("simulation_enabled"),
            )
        except requests.RequestException:
            logger.exception("Failed to reach sim daemon")

        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            touch_heartbeat()
            last_heartbeat = now
            logger.info("heartbeat")

        time.sleep(POLL_INTERVAL)


def main() -> None:
    os.makedirs(os.path.dirname(HEARTBEAT_PATH), exist_ok=True)
    if os.environ.get("REACHY_HANG_TEST") == "1":
        run_hang_test()
    else:
        run_normal()


if __name__ == "__main__":
    main()
