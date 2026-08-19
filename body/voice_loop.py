"""Session lifecycle: bring the hardware up, hand it to the demo runner, tear it down.

What the robot *does* is no longer decided here. Every behaviour -- conversing,
greeting, dancing, telling a story, running a demonstration -- is a module in
demos/, selected through the dashboard and driven by demokit/runner.py. This
file owns only what a demo must never touch: the one shared robot connection,
the face tracker, the link watchdog, and the loop that keeps the session alive
when something inside it goes wrong.

That split is the point. Before it, every mode was a branch in a single
if/elif chain here, so adding a demonstration meant editing this file, the mode
list in brain/modes.py, and the phrase dispatch in the turn handler. Now it
means adding one file to demos/.
"""

import logging
import os
import threading
import time

from brain.modes import STATE
from config import HardwareTarget
from demokit.registry import REGISTRY
from demokit.runner import DemoRunner

from .audio_io import AudioIO
from .camera import Camera
from .face import FaceIdentifier
from .face_tracker import FaceTracker
from .motion import MotionController


logger = logging.getLogger(__name__)

#: The launcher relaunches on this, having been told the connection is
#: unrepairable in-process rather than that the operator asked to stop.
_EXIT_LINK_LOST = 3

#: How long to wait for the daemon to come back after _ensure_daemon_advertises
#: restarts it. Measured at roughly a minute on this hardware.
_DAEMON_RESTART_TIMEOUT_S = 90.0

#: Pause after a failed cycle. Long enough that a fault repeating every cycle
#: cannot spin the CPU and starve the motion thread, short enough that a
#: one-off blip is invisible to whoever is standing there.
_TURN_ERROR_BACKOFF_S = 0.5


class ShutdownRequested(Exception):
    """Raised when the session should end rather than continue."""


def _ensure_daemon_advertises(host: str, port: int) -> None:
    """Restart the daemon if it is still advertising a stale address.

    The SDK does not open its WebRTC media connection to the host we connected
    to. It opens it to whatever address the daemon reports as its own in
    /api/daemon/status, and the daemon reads that once at startup. Any network
    change the robot makes afterwards therefore leaves it advertising an
    address that no longer exists: joining a WiFi network, or falling back to
    its own hotspot, which the daemon's own wifi_config does automatically
    whenever a connection attempt fails.

    Nothing else notices, because the daemon still answers on the address we
    do have -- the launcher's reachability check passes, the REST API works.
    It surfaces only as ReachyMini() hanging and then raising a bare
    "TimeoutError: timed out" from inside the media stack, which kills the app
    before the first turn. Seen both ways round: advertising its hotspot
    address while on WiFi, and (starting the robot on its own WiFi, offline)
    advertising the old WiFi address while in hotspot mode.

    Restarting the daemon is the documented fix, so do it here rather than
    leaving a crash for someone to diagnose.
    """
    import requests

    status_url = f"http://{host}:{port}/api/daemon/status"
    try:
        advertised = requests.get(status_url, timeout=5).json().get("wlan_ip")
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Could not read daemon status (%s); connecting anyway.", exc)
        return
    if advertised == host:
        return

    logger.warning(
        "Daemon reached at %s but advertising itself as %s -- media would be opened "
        "to an unreachable address. Restarting it (takes up to %.0fs).",
        host,
        advertised,
        _DAEMON_RESTART_TIMEOUT_S,
    )
    try:
        requests.post(f"http://{host}:{port}/api/daemon/restart", timeout=20)
    except requests.RequestException as exc:
        logger.warning("Daemon restart request failed (%s); connecting anyway.", exc)
        return

    deadline = time.monotonic() + _DAEMON_RESTART_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(5.0)
        try:
            status = requests.get(status_url, timeout=5).json()
        except (requests.RequestException, ValueError):
            continue  # still coming back up
        if status.get("state") == "running" and status.get("wlan_ip") == host:
            logger.info("Daemon restarted and now advertising %s.", host)
            return
    logger.warning("Daemon did not come back advertising %s in time; connecting anyway.", host)


def _capabilities(tracker: FaceTracker) -> frozenset[str]:
    """What this machine can actually do, for demos that need to know.

    "faces" is absent on the robot's own CPU, where MediaPipe crashes the
    process outright (SIGILL: the binary wants an ARM crypto extension the
    BCM2711 lacks) and face.py therefore disables it. A demo that needs faces
    is greyed out with a reason rather than silently doing nothing.
    """
    caps = set()
    if tracker.enabled:
        caps.add("faces")
    return frozenset(caps)


def _layout_doc() -> dict:
    """How the dashboard's buttons are grouped, ready to publish.

    Imported inside the function rather than at module scope: this file pulls
    in nothing from brain.db today and there is no reason to start, least of
    all for a purely cosmetic table. layout.read() never raises, so there is
    nothing to guard -- an unavailable layout is an empty document, and an
    empty document is exactly the flat grid that shipped.
    """
    from brain import layout

    doc, available = layout.read()
    doc["available"] = available
    return doc


def run_forever(target: HardwareTarget) -> None:
    """Run the session until interrupted."""
    # In "robot" mode audio, camera, and motion all share ONE ReachyMini.
    #
    # Not just an optimisation -- separate connections actively break each
    # other. MotionController's own connection asks for media_backend=
    # "no_media", which makes the SDK call release_media(): a daemon-wide
    # teardown that deletes the camera IPC socket and unregisters the WebRTC
    # producer (verified against the daemon: media_released flips to true and
    # /tmp/reachymini_camera_socket disappears). Whichever of the two connects
    # second then finds nothing to attach to -- media first meant motion
    # yanked the pipeline away mid-run; motion first meant the media
    # connection had no socket to open ("unixfdsrc: Failed to connect socket"
    # / "state change failed"). One connection, with a real media backend, is
    # the only ordering that has no such race.
    robot = None
    if target.mode == "robot":
        from reachy_mini import ReachyMini

        # Only the remote path routes media over WebRTC to the advertised
        # address; on the robot itself the media backend is local, so a stale
        # value is harmless there and not worth a restart.
        if target.media_backend == "webrtc":
            _ensure_daemon_advertises(target.daemon_host, target.daemon_port)

        robot = ReachyMini(
            host=target.daemon_host,
            port=target.daemon_port,
            media_backend=target.media_backend,
            log_level="WARNING",
        )
        robot.__enter__()

    audio = AudioIO(target, robot=robot)
    camera = Camera(target, robot=robot)
    face = FaceIdentifier(target)
    motion = MotionController(target, robot=robot)

    # Follow whoever is in view for the whole session, not just at the moment
    # a turn starts -- the robot should hold your gaze while you talk to it and
    # between questions, which is what makes it feel present rather than
    # snapping to attention only when addressed.
    tracker = FaceTracker(camera, face, motion)
    tracker.start()

    # Demos are discovered after the hardware is up but before the first cycle,
    # so the dashboard's demo grid is populated by the time anyone can press
    # anything. A demo that fails to import is logged and skipped here.
    capabilities = _capabilities(tracker)
    REGISTRY.discover()
    # Features written from the dashboard, after the demos found on disk and
    # before the list is published -- so a staff-written button is up by the
    # time anyone can press anything, exactly like a committed demo. register()
    # refuses to shadow a demo from a file, so this cannot displace one.
    from demos._stored import load_into_registry

    loaded, problems = load_into_registry()
    for problem in problems:
        STATE.add("error", f"Feature not loaded -- {problem}")
    if loaded:
        STATE.add("status", f"{loaded} feature(s) from the dashboard loaded.")
    STATE.set_capabilities(capabilities)
    STATE.set_demos(REGISTRY.dashboard_entries(capabilities), _layout_doc())
    # The operator's chosen voice, restored before the first word is spoken.
    # Every network drop relaunches this process, and until this line each
    # relaunch silently put the robot back on the config default -- so a voice
    # picked at nine was gone by the first wifi blip of the morning.
    from brain import settings

    saved_voice = settings.get("voice")
    if saved_voice and saved_voice != audio.voice_name:
        if audio.set_voice(saved_voice):
            logger.info("Restored the chosen voice: %s", saved_voice)
        else:
            STATE.add("status", f"Saved voice {saved_voice} is not installed; using the default.")
    STATE.set_voices(audio.available_voices(), audio.voice_name)
    # Whether searching is even possible: only the Anthropic backend can, so
    # the dashboard greys the switch rather than offering a setting that
    # would silently do nothing on the local model.
    from brain.llm import streaming_backends, warm_in_background
    STATE.set_web_search_available(streaming_backends()[0].supports_web)
    # Load the local model NOW, in the background, rather than in front of the
    # first visitor -- the cold start is ~30s on this hardware and it used to
    # land on whoever asked the first question of the day.
    warm_in_background()

    runner = DemoRunner(
        audio=audio,
        motion=motion,
        tracker=tracker,
        state=STATE,
        capabilities=capabilities,
    )

    # A visible greeting on startup. Idle motion is subtle by design, so a
    # working robot and a disconnected one look identical from across a room;
    # this makes "connected and under control" unmistakable without reading a
    # log, and doubles as a check that the motors are actually holding.
    motion.wake_up()

    # This session cannot repair its own connection: the SDK's websocket client
    # has no reconnect (disconnect() is terminal), and this connection carries
    # audio and camera too, so rebuilding it means rebuilding everything.
    # Rather than keep listening and replying with a frozen head -- which is
    # what happened, for thousands of consecutive dropped frames -- end the
    # process so the launcher can start a fresh session.
    def _watch_link() -> None:
        motion.link_lost.wait()
        logger.error("Motion link unrecoverable -- exiting for a clean restart.")
        os._exit(_EXIT_LINK_LOST)

    threading.Thread(target=_watch_link, name="link-watchdog", daemon=True).start()

    STATE.set_flags(ready=True)
    logger.info("Ready. Say a wake phrase to talk.")

    try:
        while True:
            try:
                runner.cycle()
                # Availability changes when a demo is set aside or re-enabled,
                # and the operator needs to see that while the session runs.
                # The folder arrangement goes out with it: read here rather
                # than in /api/events so it never scales with the number of
                # open browsers, at roughly the rate this loop already calls
                # db.get_person_name() for a recognised visitor.
                STATE.refresh_demo_availability(
                    REGISTRY.dashboard_entries(STATE.capabilities),
                    _layout_doc(),
                )
                STATE.set_voices(audio.available_voices(), audio.voice_name)
            except (KeyboardInterrupt, ShutdownRequested):
                raise
            except Exception:
                # One bad cycle must not end the session. Before this, anything
                # unhandled -- a decode error, a transient SDK fault, a demo
                # module's own bug -- propagated out of run_forever and exited
                # with code 1, which the launcher treated as a deliberate stop.
                # The robot simply went dark, mid-visit, with the traceback
                # only in a log file nobody was watching.
                logger.exception("Cycle failed; continuing with the next one.")
                STATE.add("error", "Something went wrong there -- still here.")
                time.sleep(_TURN_ERROR_BACKOFF_S)
    except (KeyboardInterrupt, ShutdownRequested):
        # Both are ordinary ways to end a session, not failures -- fall through
        # to the same cleanup as a normal exit.
        logger.info("Shutting down.")
    finally:
        # Summarise the day's conversations into long-term memory before the
        # process goes. Cheap when there is nothing to write, and the only
        # chance to write anything at all when the robot is stopped rather
        # than told to go quiet.
        try:
            runner.end_conversations()
        except Exception:
            logger.exception("Could not close out conversations on shutdown")
        tracker.stop()
        camera.close()
        audio.close()
        motion.stop()
        if robot is not None:
            robot.__exit__(None, None, None)
