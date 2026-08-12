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
    """Wake phrases worth showing a visitor, in readable form.

    The file the spotter loads holds BPE token sequences ("▁HE Y ▁RE A CH Y"),
    which is unreadable, so this reads the raw text the tokenized file was
    generated from. Returns [] rather than guessing if it is missing -- an
    empty list shows as "unavailable", which is honest, where a hardcoded
    fallback could confidently list phrases that do not work.

    Two things it has to drop, both learned by putting them on the screen.
    Comments and per-keyword boosts: this used to title-case every line, which
    was fine while the file held three bare phrases and turned six paragraphs
    of explanation into wake phrases the moment it did not. And the alternate
    spellings -- "Hey Retchy", "Hey Reechy" -- which exist so the spotter
    recognises an accent, not so anybody reads them off a screen and tries to
    pronounce them.

    Case comes from the file rather than .title(), which rendered "OK Reachy"
    as "Ok Reachy".
    """
    raw = MODELS.kws_keywords_file.with_name("custom_keywords_raw.txt")
    try:
        lines = raw.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Could not read %s", raw)
        return []

    phrases = []
    for line in lines:
        text, _, comment = line.partition("#")
        text = text.split(":", 1)[0].strip()  # drop any per-keyword boost
        if not text or "alt" in comment.lower():
            continue
        phrases.append(text)
    return phrases

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


class PersonaRequest(BaseModel):
    persona: str


@app.get("/api/personas")
def personas_list() -> JSONResponse:
    """The answering styles the dashboard can choose between.

    Read from brain/personas.py rather than listed here, so adding one stays
    the one-file change that module promises.
    """
    from brain import personas

    return JSONResponse(
        {
            "personas": [
                {"id": p.id, "label": p.label, "blurb": p.blurb} for p in personas.PERSONAS
            ]
        }
    )


@app.post("/api/persona")
def set_persona(req: PersonaRequest) -> JSONResponse:
    """Pick which style the personality demo answers in."""
    chosen = STATE.set_persona(req.persona)
    return JSONResponse({"ok": True, "persona": chosen})


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


#: Features Hub staff build from the dashboard. These handlers touch no
#: hardware, so none of them queue through STATE.request -- the precedent is
#: enable_demo above, which mutates the registry from this thread for the same
#: reason. What they must never do is raise: this app has no exception
#: middleware, so an uncaught sqlite3.Error becomes a bare 500 with a traceback
#: in the body, on a page with no authentication.


class FeatureRequest(BaseModel):
    label: str = ""
    help: str = ""
    trigger_phrase: str = ""
    persona: str = ""
    created_by: str = ""
    blocks: list = []


def _republish() -> None:
    """Put the demo list in front of the operator now, not next cycle.

    The loop republishes every cycle anyway, but a cycle can be three seconds
    and "Save" followed by pressing the new button is the first thing anybody
    will do.
    """
    try:
        STATE.refresh_demo_availability(REGISTRY.dashboard_entries(STATE.capabilities))
    except Exception:  # pragma: no cover - never worth failing a save over
        logger.exception("Could not refresh the demo list")


def _feature_from(req: "FeatureRequest", existing: "object | None" = None):
    from brain import features

    record = features.Feature(
        label=(req.label or "").strip(),
        help=(req.help or "").strip(),
        trigger_phrase=features.normalise_trigger(req.trigger_phrase),
        persona=(req.persona or "").strip(),
        blocks=features.parse_blocks(req.blocks),
        created_by=(req.created_by or "").strip(),
    )
    # The id follows the label, so renaming a feature makes a new one rather
    # than silently rewriting the old under a name nobody recognises. On an
    # edit the original id is kept, so its button and store survive.
    record.id = getattr(existing, "id", None) or features.slug_for(record.label)
    record.created_at = getattr(existing, "created_at", "") or ""
    return record


@app.get("/api/features")
def list_features() -> JSONResponse:
    """Everything staff have built, plus what the editor needs to offer."""
    from brain import features
    from brain.emotion import VALID_EMOTION_TAGS
    from brain import personas

    out = []
    for record in features.list_features():
        entry = record.as_dict()
        # Whether the button is actually up, as opposed to merely stored.
        entry["live"] = REGISTRY.get(record.id) is not None
        entry["warnings"] = features.warnings_for(record)
        out.append(entry)
    return JSONResponse({
        "features": out,
        "emotions": sorted(VALID_EMOTION_TAGS),
        "personas": [{"id": p.id, "label": p.label} for p in personas.PERSONAS],
        "available": features._available,
        "limits": {
            "max_features": features.MAX_FEATURES,
            "max_blocks": features.MAX_BLOCKS,
            "max_say": features.MAX_SAY_CHARS,
            "wait_range": list(features.WAIT_SECONDS_RANGE),
        },
    })


def _save(req: "FeatureRequest", feature_id: str = "") -> JSONResponse:
    from brain import features
    from demos import _stored

    existing = features.get_feature(feature_id) if feature_id else None
    if feature_id and existing is None:
        return JSONResponse({"ok": False, "errors": ["That feature no longer exists."]},
                            status_code=404)
    record = _feature_from(req, existing)
    if existing is not None:
        record.enabled = existing.enabled

    saved, problems = features.save(record)
    if not saved:
        return JSONResponse({"ok": False, "errors": problems}, status_code=400)

    _stored.sync_one(record.id)
    _republish()
    return JSONResponse({
        "ok": True, "id": record.id,
        "warnings": features.warnings_for(record),
        "live": REGISTRY.get(record.id) is not None,
    })


@app.post("/api/features")
def create_feature(req: FeatureRequest) -> JSONResponse:
    return _save(req)


@app.put("/api/features/{feature_id}")
def update_feature(feature_id: str, req: FeatureRequest) -> JSONResponse:
    return _save(req, feature_id)


@app.delete("/api/features/{feature_id}")
def delete_feature(feature_id: str) -> JSONResponse:
    from brain import features
    from demos import _stored

    if not features.delete(feature_id):
        return JSONResponse({"ok": False, "error": "no such feature"}, status_code=404)
    _stored.sync_one(feature_id)
    # Leaving the robot pointing at a demo that no longer exists would recover
    # on the next cycle anyway, but with an error banner in front of visitors.
    if STATE.mode == feature_id:
        STATE.set_mode(REGISTRY.default_id())
    _republish()
    return JSONResponse({"ok": True})


class EnabledRequest(BaseModel):
    enabled: bool


@app.post("/api/features/{feature_id}/enabled")
def enable_feature(feature_id: str, req: EnabledRequest) -> JSONResponse:
    """Take a feature off the dashboard without deleting it.

    The right move mid-visit: somebody presses the wrong button, and the fix
    should not require deciding whether to throw the script away.
    """
    from brain import features
    from demos import _stored

    if not features.set_enabled(feature_id, req.enabled):
        return JSONResponse({"ok": False, "error": "no such feature"}, status_code=404)
    _stored.sync_one(feature_id)
    if not req.enabled and STATE.mode == feature_id:
        STATE.set_mode(REGISTRY.default_id())
    _republish()
    return JSONResponse({"ok": True, "live": REGISTRY.get(feature_id) is not None})


class DraftRequest(BaseModel):
    messages: list = []
    steps: list = []


#: One draft at a time. Drafting shares the local model with the robot's own
#: voice, and a 1.5B reply is 4-9 seconds -- five staff drafting at once starves
#: the robot mid-visit. Non-blocking, so the answer is an immediate "busy"
#: rather than a request that sits there.
_drafting = threading.Lock()

#: Two turns, either a question or a finished draft, never both. Written as a
#: contract rather than a hope: the reply is parsed, and anything that is not
#: valid JSON is treated as a question, so a model that ignores this degrades
#: to a chat rather than to an error.
_DRAFT_SYSTEM = """You are helping a member of staff at the AI XR Hub build a short \
script for Reachy Mini, a small desk robot, to perform for visitors.

You never write code, and you never will. You produce exactly one of two things:

1. ONE short question, in plain text, when you still need to know something.
   Ask about one thing at a time: who the group is, what the robot should say
   to them, whether it should ask them anything.

2. A finished draft, as a single JSON object in a ```json fence and nothing
   else, once you know enough. Its shape is exactly:

```json
{"label": "Cork school visit",
 "blocks": [
   {"kind": "SAY", "text": "Welcome, all of you.", "emotion": "happy"},
   {"kind": "ASK", "text": "Where did you travel from?", "emotion": "curious", "ai_reply": true},
   {"kind": "DANCE"},
   {"kind": "WAIT", "seconds": 20}
 ]}
```

Rules for a draft:
- "kind" is one of SAY, ASK, DANCE, WAIT. Nothing else exists.
- "emotion" is one of: happy, curious, neutral, thinking, sad, surprised.
- Keep it under about 30 seconds of speech in total -- four or five sentences.
  Groups start looking at each other past that.
- Write for the ear. Short sentences, no lists, no headings, no stage
  directions, nothing in brackets.
- Every factual claim must come from the Hub facts below. If the person wants
  something you do not know -- a name, a number, a partner, a headset model --
  write it as a placeholder in square brackets, like [name of the visiting
  professor], and say in your next message that they need to fill it in.
  Never invent a fact about the Hub.
"""


@app.post("/api/features/draft")
def draft_feature(req: DraftRequest) -> JSONResponse:
    """Turn a description into a draft the person then edits and approves.

    The model writes words, never code, and never touches the database: this
    returns a draft to the browser and a human presses Save. That is the whole
    security model for a page with no authentication -- the four step kinds are
    fixed and tested, so the worst a bad draft produces is bad wording, which
    the person is looking at before it is stored.
    """
    import json as _json
    import re as _re

    from brain import hub, llm

    if not _drafting.acquire(blocking=False):
        return JSONResponse(
            {"ok": False, "error": "The assistant is busy with another draft. Try again in a moment."},
            status_code=429,
        )
    try:
        # Capped rather than trusted: this is an unauthenticated endpoint and
        # the whole conversation is posted back each turn.
        history = []
        for entry in (req.messages or [])[-20:]:
            role = "assistant" if str(entry.get("role")) == "assistant" else "user"
            content = str(entry.get("content", ""))[:2000]
            if content:
                history.append({"role": role, "content": content})
        if not history:
            return JSONResponse({"ok": True, "reply": "Tell me who the group is.", "draft": None})

        messages = [{"role": "system", "content": _DRAFT_SYSTEM + "\n\n" + hub.GROUNDING}] + history
        raw = llm.generate_response(messages)
    except Exception:
        logger.exception("Drafting failed")
        return JSONResponse(
            {"ok": False,
             "error": "The model could not be reached. You can still build the steps by hand."},
            status_code=503,
        )
    finally:
        _drafting.release()

    from brain import features
    from brain.emotion import extract_emotion_tag

    # The model is told to end nothing with an emotion tag, but it is trained
    # by every other prompt in this robot to add one.
    text, _tag = extract_emotion_tag(raw)
    fence = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.S) or _re.search(r"(\{.*\})", text, _re.S)
    draft = None
    if fence:
        try:
            parsed = _json.loads(fence.group(1))
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            blocks = features.parse_blocks(parsed.get("blocks"))
            if blocks:
                draft = {
                    "label": str(parsed.get("label", ""))[:features.MAX_LABEL_CHARS],
                    "blocks": [b.as_dict() for b in blocks],
                }
        # Whatever the JSON was, it is not something to show a staff member.
        text = text[: fence.start()].strip()
        if draft is None and not text:
            # A fence that produced no usable steps. The local model does this
            # often enough that echoing its JSON back would be the normal
            # experience, so say something a person can act on instead.
            text = ("I could not turn that into steps. Try describing the group in a "
                    "sentence, or add the steps yourself below.")

    if not text and draft:
        text = "Here is a draft — edit anything you like below, then Save."
    return JSONResponse({"ok": True, "reply": text or raw.strip()[:600], "draft": draft})


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
