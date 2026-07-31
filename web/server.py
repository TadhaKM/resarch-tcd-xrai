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

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from body.voice_loop import _DANCE_PHRASES as DANCE_PHRASES
from body.voice_loop import _SHUTDOWN_PHRASES as SLEEP_PHRASES
from brain.modes import STATE
from config import MODELS

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
            "dance": list(DANCE_PHRASES),
        }
    )


@app.get("/api/events")
def events(since: int = 0) -> JSONResponse:
    """Return events after `since`, plus the current status.

    Polled rather than pushed: the payload is tiny, polling survives the
    dropouts this network produces without needing reconnect logic, and it
    keeps the page to plain HTML with no build step.
    """
    latest, items = STATE.events_since(since)
    return JSONResponse({"seq": latest, "events": items, "status": STATE.snapshot()})


@app.post("/api/mode")
def set_mode(req: ModeRequest) -> JSONResponse:
    if not STATE.set_mode(req.mode):  # type: ignore[arg-type]
        return JSONResponse({"ok": False, "error": f"unknown mode {req.mode!r}"}, status_code=400)
    return JSONResponse({"ok": True, "mode": STATE.mode})


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
