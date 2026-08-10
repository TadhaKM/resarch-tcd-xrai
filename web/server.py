"""Dashboard: see what the robot hears and choose what it does.

Runs in a thread alongside the voice loop and talks to it only through
brain.modes.STATE, so a browser can never block or crash the robot -- the
worst a broken request can do is return an error to itself.

Bound to all interfaces by default so it can be opened from a phone on the
same network. That also means anyone on that network can change the robot's
mode; there's no authentication, which is fine for a home network and worth
knowing before putting it on a shared one.
"""

import logging
import threading
from pathlib import Path

import time

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from brain.modes import STATE
from config import MODELS, default_target
from demokit.registry import REGISTRY
from demokit.runner import SLEEP_PHRASES

logger = logging.getLogger(__name__)

_PAGE = Path(__file__).parent / "index.html"


def read_wake_phrases() -> list[str]:
    """Wake phrases in readable form.

    The file the spotter loads holds BPE token sequences ("▁HE Y ▁RE A CH Y"),
    which is unreadable, so this reads the raw text the tokenized file was
    generated from. Returns [] rather than guessing if it is missing -- an
    empty list shows as "unavailable", which is honest, where a hardcoded
    fallback could confidently list phrases that do not work.
    """
    raw = MODELS.kws_keywords_file.with_name("custom_keywords_raw.txt")
    try:
        lines = raw.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Could not read %s", raw)
        return []
    return [line.strip().title() for line in lines if line.strip()]

app = FastAPI(title="Reachy Mini")


class ModeRequest(BaseModel):
    mode: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_PAGE)


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse(STATE.snapshot())


@app.get("/api/phrases")
def phrases() -> JSONResponse:
    """What the robot listens for.

    Read from the keyword file rather than hardcoded here, so the page can
    never disagree with what the spotter is actually matching.
    """
    return JSONResponse(
        {
            "wake": read_wake_phrases(),
            "sleep": list(SLEEP_PHRASES),
            "demos": _demo_triggers(),
        }
    )


def _demo_triggers() -> list[dict]:
    """Phrases that switch demos, collected from the demos themselves.

    Built from the registry rather than listed here, for the same reason the
    wake phrases are read from the keyword file: a panel that visitors read off
    the screen must not be able to disagree with what the robot actually
    matches. A demo added tomorrow shows its own phrases with no edit here.
    """
    out = []
    for demo_id in REGISTRY.ids():
        demo = REGISTRY.get(demo_id)
        if demo is None or not demo.triggers:
            continue
        out.append({"label": demo.label, "phrases": list(demo.triggers)})
    return out


@app.get("/api/events")
def events(since: int = 0) -> JSONResponse:
    """Return events after `since`, plus the current status.

    Polled rather than pushed: the payload is tiny, polling survives the
    dropouts this network produces without needing reconnect logic, and it
    keeps the page to plain HTML with no build step.
    """
    latest, items = STATE.events_since(since)
    return JSONResponse({"seq": latest, "events": items, "status": STATE.snapshot()})


class SayRequest(BaseModel):
    text: str


@app.post("/api/say")
def say(req: SayRequest) -> JSONResponse:
    """Queue a sentence for the robot to speak (dashboard 'Say it' box)."""
    text = req.text.strip()[:300]
    if not text:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    if not STATE.request("say", text):
        return JSONResponse({"ok": False, "error": "queue full"}, status_code=429)
    return JSONResponse({"ok": True})


@app.post("/api/listen")
def listen_now() -> JSONResponse:
    """Start listening as if the wake word had been heard.

    Exists because the wake-word model is the least reliable link in a loud
    room; a button on a phone works at any noise level.
    """
    STATE.request("listen")
    return JSONResponse({"ok": True})


@app.get("/api/transcript")
def transcript() -> PlainTextResponse:
    """The whole session as plain text, for download."""
    lines = []
    for e in STATE.history():
        stamp = time.strftime("%H:%M:%S", time.localtime(e.at))
        lines.append(f"{stamp}  {e.kind.upper():7}  {e.text}")
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        headers={"Content-Disposition": 'attachment; filename="reachy-transcript.txt"'},
    )


def _daemon_url() -> str:
    # Resolved per request, not at import: the robot's address changes (DHCP
    # renewals mid-session are routine here) and default_target() re-resolves
    # the mDNS name each call.
    target = default_target()
    return f"http://{target.daemon_host}:{target.daemon_port}"


class VolumeRequest(BaseModel):
    volume: int


@app.get("/api/volume")
def get_volume() -> JSONResponse:
    """Proxy the robot's current speaker volume.

    Proxied through this server rather than fetched by the browser so the
    page stays same-origin and never needs to know the robot's address.
    """
    try:
        r = requests.get(f"{_daemon_url()}/api/volume/current", timeout=3)
        return JSONResponse(r.json())
    except Exception as exc:
        return JSONResponse({"error": f"robot unreachable: {exc}"}, status_code=502)


@app.post("/api/volume")
def set_volume(req: VolumeRequest) -> JSONResponse:
    level = max(0, min(100, req.volume))
    try:
        r = requests.post(
            f"{_daemon_url()}/api/volume/set", json={"volume": level}, timeout=3
        )
        return JSONResponse(r.json())
    except Exception as exc:
        return JSONResponse({"error": f"robot unreachable: {exc}"}, status_code=502)


@app.post("/api/mode")
def set_mode(req: ModeRequest) -> JSONResponse:
    if not STATE.set_mode(req.mode):  # type: ignore[arg-type]
        return JSONResponse({"ok": False, "error": f"unknown mode {req.mode!r}"}, status_code=400)
    return JSONResponse({"ok": True, "mode": STATE.mode})


class VoiceRequest(BaseModel):
    voice: str


class WebSearchRequest(BaseModel):
    enabled: bool


@app.post("/api/websearch")
def set_web_search(req: WebSearchRequest) -> JSONResponse:
    """Turn looking things up online on or off.

    Applied immediately rather than queued: unlike the voice, this touches no
    hardware -- it is a flag the next turn reads.
    """
    enabled = STATE.set_web_search(bool(req.enabled))
    return JSONResponse({"ok": True, "enabled": enabled})


@app.post("/api/openmic")
def set_open_mic(req: WebSearchRequest) -> JSONResponse:
    """Let a conversation continue without the wake word before every question.

    Same shape as the switch above, and applied the same way: a flag the voice
    loop reads at the top of its next turn, touching no hardware.
    """
    enabled = STATE.set_open_mic(bool(req.enabled))
    return JSONResponse({"ok": True, "enabled": enabled})


@app.get("/api/model")
def model() -> JSONResponse:
    """Which language model the robot is set up to answer with.

    Worth showing to a visitor as much as to an operator: half the point of the
    demonstration is that the thing in front of them is a specific piece of
    technology rather than magic, and "this is Claude Sonnet, and it falls back
    to a model on the laptop if the wifi drops" is a more interesting sentence
    than either half on its own.
    """
    from brain import llm

    return JSONResponse(llm.describe())


@app.get("/api/voices")
def voices() -> JSONResponse:
    """Installed voices and the one in use.

    Read off disk by the voice loop rather than listed here, so dropping a
    piper voice into models/tts/ offers it with no code change.
    """
    return JSONResponse(STATE.voices())


@app.post("/api/voice")
def set_voice(req: VoiceRequest) -> JSONResponse:
    """Queue a voice change for the voice loop to apply between turns."""
    name = req.voice.strip()
    if not name:
        return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
    if not STATE.request("voice", name):
        return JSONResponse({"ok": False, "error": "queue full"}, status_code=429)
    return JSONResponse({"ok": True})


@app.post("/api/cache/clear")
def clear_cache() -> JSONResponse:
    """Drop every cached answer.

    An operator needs this reachable: a cached answer is served instantly and
    forever, so one that turns out to be wrong -- said before a fact was
    corrected in brain/hub.py, or produced on a day the model was misbehaving --
    outlives the mistake unless there is a button. Safe at any time; the next
    question simply reaches the model again.
    """
    from brain import qa_cache

    dropped = qa_cache.forget_all()
    STATE.add("status", f"Answer cache cleared ({dropped} entr{'y' if dropped == 1 else 'ies'})")
    return JSONResponse({"ok": True, "dropped": dropped})


@app.post("/api/demos/{demo_id}/enable")
def enable_demo(demo_id: str) -> JSONResponse:
    """Put a demo back in service after it was set aside for repeated failures.

    Without this the only way back is restarting the robot, which during an
    open day means the operator loses the session to a demo a student is still
    fixing. The dashboard shows the greyed button and its reason; this is the
    button that clears it.
    """
    if REGISTRY.get(demo_id) is None:
        return JSONResponse({"ok": False, "error": f"unknown demo {demo_id!r}"}, status_code=404)
    reinstated = REGISTRY.enable(demo_id)
    return JSONResponse({"ok": True, "reinstated": reinstated})


def serve(host: str = "0.0.0.0", port: int = 8080) -> threading.Thread:
    """Start the dashboard in a daemon thread and return it."""

    def _run() -> None:
        try:
            uvicorn.run(app, host=host, port=port, log_level="warning")
        except Exception:
            # The robot must keep working without the dashboard.
            logger.exception("Dashboard stopped")

    thread = threading.Thread(target=_run, name="dashboard", daemon=True)
    thread.start()
    logger.info("Dashboard on http://%s:%d (and this machine's LAN address)", host, port)
    return thread
