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
from typing import Optional

from config import HardwareTarget, default_target
from demokit.registry import REGISTRY
from demokit.runner import DemoRunner

from .audio_io import AudioIO
from .camera import Camera
from .face import FaceIdentifier
from .doa import DoaListener
from .face_tracker import FaceTracker
from .motion import MotionController


logger = logging.getLogger(__name__)

#: The launcher relaunches on this, having been told the connection is
#: unrepairable in-process rather than that the operator asked to stop.
_EXIT_LINK_LOST = 3

#: Backoff between reconnect attempts, growing per attempt to this cap. The
#: robot is usually back within a minute of a wifi blip; hammering the daemon
#: in the meantime achieves nothing.
_RECONNECT_BACKOFF_S = 3.0
_RECONNECT_BACKOFF_MAX_S = 30.0

#: How often the main loop checks whether the link came back.
_LINK_DOWN_POLL_S = 0.5

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


def _capabilities(tracker: FaceTracker, camera: Optional[Camera] = None) -> frozenset[str]:
    """What this machine can actually do, for demos that need to know.

    "faces" is absent on the robot's own CPU, where MediaPipe crashes the
    process outright (SIGILL: the binary wants an ARM crypto extension the
    BCM2711 lacks) and face.py therefore disables it. A demo that needs faces
    is greyed out with a reason rather than silently doing nothing.

    "camera" is separate from "faces" on purpose. A machine can have a working
    camera and no face detection -- that is exactly the robot's own CPU -- so a
    demo that just wants a picture must not be gated on recognition it does not
    use. Conflating the two would have left "Look at this" greyed out forever
    on hardware perfectly able to run it.
    """
    caps = set()
    if tracker.enabled:
        caps.add("faces")
    if camera is not None:
        caps.add("camera")
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
    # The expensive, robot-independent work -- four speech models, the face
    # recogniser, the language-model clients -- starts on a worker thread NOW,
    # so it runs while the robot connection is being established below. Boot
    # used to do these one after another and was measured at 43-77 seconds;
    # the two halves are independent, so the slower of them should set the
    # boot time, not their sum.
    preloaded: dict = {}

    def _preload() -> None:
        from body import audio_io

        audio_io.preload_models()
        try:
            preloaded["face"] = FaceIdentifier(target)
        except Exception:
            logger.exception("Could not preload face recognition; building inline.")
        try:
            from brain.llm import streaming_backends

            streaming_backends()
        except Exception:
            logger.exception("Could not preload the language model clients.")

    preload_thread = threading.Thread(target=_preload, name="model-preload", daemon=True)
    preload_thread.start()

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

    # The robot is connected; wait for the models. Usually they finished
    # first and this returns immediately -- the join is here so a slow disk
    # can never race the constructor into building a model twice.
    preload_thread.join()

    audio = AudioIO(target, robot=robot)
    camera = Camera(target, robot=robot)
    face = preloaded.get("face") or FaceIdentifier(target)
    motion = MotionController(target, robot=robot)

    # Follow whoever is in view for the whole session, not just at the moment
    # a turn starts -- the robot should hold your gaze while you talk to it and
    # between questions, which is what makes it feel present rather than
    # snapping to attention only when addressed.
    # The microphone array reports which direction a voice came from, which
    # lets the head turn toward somebody talking from outside the camera's
    # view -- the one case visual tracking cannot cover at all. Read from the
    # daemon over HTTP rather than through the SDK's AudioDoA: that opens the
    # USB device directly, and in remote mode it is constructed on the laptop,
    # where no ReSpeaker is attached. Entirely optional; without a board that
    # answers, every accessor returns None and the tracker behaves as before.
    doa = DoaListener(target.daemon_host, target.daemon_port)
    doa.start()

    tracker = FaceTracker(camera, face, motion, doa=doa)
    tracker.start()

    # Demos are discovered after the hardware is up but before the first cycle,
    # so the dashboard's demo grid is populated by the time anyone can press
    # anything. A demo that fails to import is logged and skipped here.
    capabilities = _capabilities(tracker, camera)
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

    # A dropped link used to end the process: the SDK's websocket client has no
    # reconnect (disconnect() is terminal), and this connection carries audio
    # and camera too, so rebuilding it means rebuilding the ReachyMini itself.
    # The launcher then relaunched, which cost ~40s of dead robot -- five times
    # in one morning -- and threw away everything expensive along with it: the
    # loaded speech models, the demo registry, the warm language model, and the
    # research session, which is module state and so silently disarmed.
    #
    # Only ONE object actually dies. The ReachyMini is referenced by exactly
    # three wrappers; demos hold the WRAPPERS (DemoContext.audio/.motion), not
    # the SDK object, so rebuilding it underneath them invalidates nothing --
    # not a demo's store, not the runner, not the tracker, not the dashboard.
    # So rebuild in place and keep the session.
    link_down = threading.Event()

    def _rebuild_link() -> bool:
        """One attempt at a fresh connection. True if the robot is back."""
        # The daemon advertises its own address for the media session, and a
        # stale one is exactly the failure _ensure_daemon_advertises exists to
        # prevent -- doubly likely here, since the address changing is often
        # WHY the link dropped.
        # Re-resolved per attempt rather than reusing the address captured at
        # startup. DHCP handing out a new address is one of the commonest
        # reasons this link drops in the first place, so retrying the old one
        # forever means retrying the single address now guaranteed to be wrong.
        current = default_target()
        if current.media_backend == "webrtc":
            _ensure_daemon_advertises(current.daemon_host, current.daemon_port)
        fresh = ReachyMini(
            host=current.daemon_host,
            port=current.daemon_port,
            media_backend=current.media_backend,
            log_level="WARNING",
        )
        fresh.__enter__()
        audio.adopt_robot(fresh)
        camera.adopt_robot(fresh)
        motion.adopt_robot(fresh)
        return True

    def _watch_link() -> None:
        nonlocal robot
        while True:
            motion.link_lost.wait()
            link_down.set()
            STATE.set_flags(ready=False)
            STATE.add("error", "Lost the connection to the robot -- reconnecting.")
            logger.error("Motion link lost; rebuilding the connection in place.")

            # Detach the wrappers BEFORE tearing the old one down, so nothing
            # dereferences a half-closed connection mid-turn.
            audio.adopt_robot(None)
            camera.adopt_robot(None)
            # Motion too. Without this its 20Hz send loop keeps writing to a
            # dead socket for the whole outage: measured at 3,700+ dropped
            # pose updates and a warning every ten seconds, which drowns the
            # log at precisely the moment somebody is reading it to find out
            # what is wrong. _maybe_reconnect returns immediately once _robot
            # is None, so detaching is what actually stops the noise.
            motion.adopt_robot(None)
            old, robot = robot, None
            if old is not None:
                try:
                    old.__exit__(None, None, None)
                except Exception:
                    logger.debug("Old connection did not close cleanly", exc_info=True)

            attempt = 0
            while True:
                attempt += 1
                try:
                    _rebuild_link()
                except Exception as exc:
                    delay = min(_RECONNECT_BACKOFF_S * attempt, _RECONNECT_BACKOFF_MAX_S)
                    logger.warning("Reconnect attempt %d failed (%s); retrying in %.0fs.",
                                   attempt, exc, delay)
                    if attempt == 1 or attempt % 10 == 0:
                        # Said in terms of what to DO. "Attempt 12 failed" tells
                        # an operator nothing; the robot being off, asleep or on
                        # a new address are the three real causes and all three
                        # are things a person standing next to it can check.
                        STATE.add("error",
                                  f"Cannot reach the robot ({attempt} tries). Check it is "
                                  f"powered on and on the same wifi.")
                        logger.warning(
                            "Still cannot reach the robot after %d attempts. It is powered "
                            "off, asleep, or has moved to a different address.", attempt)
                    time.sleep(delay)
                    continue
                break

            link_down.clear()
            STATE.set_flags(ready=True)
            STATE.add("status", "Robot reconnected.")
            logger.info("Motion link rebuilt after %d attempt(s); session kept.", attempt)
            # Said aloud because the visitor watched it freeze mid-sentence and
            # is owed an explanation rather than a robot that simply resumes.
            #
            # QUEUED, not spoken here. audio.speak is loop-thread-only -- its
            # sibling speak_rendered says so in one line -- and this runs on the
            # watchdog thread, so speaking directly would interleave with
            # whatever the loop is already saying and corrupt the media stream.
            # STATE.request is the established cross-thread path; the dashboard's
            # "Say it" box reaches the speaker exactly this way.
            STATE.request("say", "Sorry about that -- I lost my connection for a moment.")

    threading.Thread(target=_watch_link, name="link-watchdog", daemon=True).start()

    STATE.set_flags(ready=True)
    logger.info("Ready. Say a wake phrase to talk.")

    try:
        while True:
            try:
                if link_down.is_set():
                    # The link is being rebuilt. Cycling now would drive a
                    # detached wrapper; wait it out rather than logging a
                    # failure per cycle at loop speed.
                    time.sleep(_LINK_DOWN_POLL_S)
                    continue
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
        doa.stop()
        tracker.stop()
        camera.close()
        audio.close()
        motion.stop()
        if robot is not None:
            robot.__exit__(None, None, None)
