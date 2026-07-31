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

from brain.modes import STATE

logger = logging.getLogger(__name__)

_PAGE = Path(__file__).parent / "index.html"

app = FastAPI(title="Reachy Mini")


class ModeRequest(BaseModel):
    mode: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_PAGE)


@app.get("/api/status")
def status() -> JSONResponse:
    return JSONResponse(STATE.snapshot())


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
