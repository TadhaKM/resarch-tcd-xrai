"""Dashboard: see what the robot hears and choose what it does.

Runs in a thread alongside the voice loop and talks to it only through
brain.modes.STATE, so a browser can never block or crash the robot -- the
worst a broken request can do is return an error to itself.

Bound to all interfaces by default so it can be opened from a phone on the
same network. That also means anyone on that network can change the robot's
mode; there's no authentication, which is fine for a home network and worth
knowing before putting it on a shared one.
"""

import csv
import io
import logging
import threading
from pathlib import Path

import time

import requests
import uvicorn
from fastapi import FastAPI, Response
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
    """Put the demo list and the folder layout in front of the operator now.

    The loop republishes every cycle anyway, but a cycle can be three seconds
    and "Save" followed by pressing the new button is the first thing anybody
    will do.

    Both go out together. The voice loop publishes both every cycle too, and if
    only one publisher carried the layout, the other's next cycle would put a
    stale copy back -- so the grid would fall flat and then jump into folders
    again a couple of seconds later.
    """
    from brain import layout

    try:
        doc, available = layout.read()
        doc["available"] = available
        STATE.refresh_demo_availability(REGISTRY.dashboard_entries(STATE.capabilities), doc)
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
    # Not housekeeping. slug_for derives the id from the label, so a feature
    # written months later with a similar name gets the same id -- and would
    # inherit this one's place, which may be inside a collapsed folder. See
    # layout.forget.
    from brain import layout

    layout.forget(feature_id)
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


# --- the folder layout --------------------------------------------------
#
# Display only. Nothing here touches the demo list, the registry or the order
# the robot picks a demo by -- see brain/layout.py's docstring for why that
# separation is the whole design rather than an implementation detail.


class LayoutRequest(BaseModel):
    base: int = 0
    items: list = []


@app.get("/api/layout")
def get_layout() -> JSONResponse:
    """The stored arrangement.

    Always 200; whether it can be written is reported in the body, the same way
    /api/features reports its own availability. The dashboard normally gets
    this on the 700ms poll instead -- this route is for a cold read and for
    anybody driving the robot with curl.
    """
    from brain import layout

    doc, available = layout.read()
    return JSONResponse({
        "ok": True, "available": available, **doc,
        "limits": {
            "max_folders": layout.MAX_FOLDERS,
            "max_folder_name": layout.MAX_FOLDER_NAME_CHARS,
        },
    })


@app.put("/api/layout")
def put_layout(req: LayoutRequest) -> JSONResponse:
    """Replace the arrangement, if the browser was looking at the current one."""
    from brain import layout

    status, doc, problems = layout.write(req.items, int(req.base or 0))
    if status == layout.OK:
        _republish()
        return JSONResponse({"ok": True, **doc})
    if status == layout.STALE:
        # 409 carrying the winner's arrangement, so the loser can re-apply its
        # one move on top rather than discarding the gesture. Two staff tidying
        # up at once is an ordinary open day, not a theoretical race.
        return JSONResponse({"ok": False, "stale": True, **doc}, status_code=409)
    if status == layout.UNAVAILABLE:
        return JSONResponse(
            {"ok": False, "errors": [
                "The robot's database is unavailable, so the layout cannot be changed."]},
            status_code=503,
        )
    return JSONResponse({"ok": False, "errors": problems}, status_code=400)


@app.delete("/api/layout")
def reset_layout() -> JSONResponse:
    """Put every button back on the main grid.

    There is no undo stack, and the people using this are not technical. One
    button that certainly gets them back to a grid they recognise is worth more
    than any amount of care about not needing it.
    """
    from brain import layout

    doc = layout.reset()
    STATE.add("status", "Dashboard layout reset")
    _republish()
    return JSONResponse({"ok": True, **doc})


# --- QR codes, visit stats -----------------------------------------------


@app.get("/api/qr")
def qr(url: str = "") -> JSONResponse:
    """A QR matrix for one link, as rows of 0/1 for the browser to draw.

    A matrix rather than an image: the dashboard builds it as SVG rects with
    the same createElement discipline as everything else, so there is no image
    endpoint to cache, no data-URI to get wrong, and it recolours itself with
    the theme like the rest of the page.

    segno is pure Python and generates offline, which matters -- the robot is
    frequently on a hotspot or on nothing at all, and a QR that only works
    with internet would fail on exactly the days a visit is happening.
    """
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")) or len(url) > 300:
        return JSONResponse({"ok": False, "error": "not a link"}, status_code=400)
    try:
        import segno

        # Error correction M: readable from a phone at arm's length even with
        # a thumb over a corner, without the density that makes a code fail on
        # a low-resolution screen.
        code = segno.make(url, error="m")
        matrix = [[int(bit) for bit in row] for row in code.matrix]
    except Exception:
        logger.exception("Could not build a QR code")
        return JSONResponse({"ok": False, "error": "unavailable"}, status_code=503)
    return JSONResponse({"ok": True, "size": len(matrix), "matrix": matrix})


class StudyRequest(BaseModel):
    running: bool = False
    condition: str = ""
    #: Who is arming this. Since switching research mode on IS the consent
    #: now, this is the only record of somebody accountable for the data --
    #: an audit note, not authentication; this dashboard has no accounts.
    operator: str = ""


@app.get("/api/study")
def study_status() -> JSONResponse:
    """Whether a research session is running, and what has been collected."""
    from brain import study

    return JSONResponse({**study.status(), "summary": study.summary()})


@app.post("/api/study")
def set_study(req: StudyRequest) -> JSONResponse:
    """Switch research mode on or off.

    Switching it ON only ARMS it: brain/study.py records nothing until the
    participant has heard what is being recorded and agreed out loud, through
    the Research session demo. There is deliberately no way to consent on
    somebody's behalf from here.
    """
    from brain import study

    if req.running and not study._available:
        return JSONResponse(
            {"ok": False, "error": "The robot's database is unavailable, so nothing "
                                   "could be recorded."},
            status_code=503,
        )
    state = (study.start(req.condition, operator=req.operator)
             if req.running else study.stop())
    # ONE place capabilities change, and everything reads them back from the
    # state -- including DemoRunner, which used to hold its own frozen copy and
    # refuse the demo the dashboard was showing as available.
    STATE.set_capability("study", bool(req.running))
    STATE.add("status",
              (f"Recording on{' -- armed by ' + req.operator.strip() if req.operator.strip() else ''}"
               if req.running else "Recording off"))
    # Switching OFF while the research demo is showing: step the robot back to
    # conversation HERE, deliberately, rather than leaving the runner to notice
    # the capability has gone and evict it. Both end in the same place, but the
    # runner's route reports "study is unavailable (needs study)" as an ERROR --
    # which is what an operator saw on the dashboard after pressing the off
    # switch, reading as a fault rather than as the switch doing its job.
    if not req.running and STATE.mode == "study":
        fallback = REGISTRY.default_id()
        if fallback and fallback != "study":
            STATE.set_mode(fallback)
    # The demo is gated on the "study" capability, so the grid has to be
    # republished for it to appear or grey out.
    _republish()
    return JSONResponse({"ok": True, **state})


@app.delete("/api/study")
def withdraw_study(session: str = "") -> JSONResponse:
    """Delete a participant's data. The whole session, not a flag."""
    from brain import study

    gone = study.withdraw(session or None)
    STATE.add("status", f"Study data withdrawn ({gone} turn(s) deleted)")
    return JSONResponse({"ok": True, "deleted": gone})


@app.get("/api/study/sessions")
def study_sessions() -> JSONResponse:
    """Every recorded session: counts and timings, no transcript text."""
    from brain import study

    return JSONResponse({"sessions": study.sessions()})


@app.get("/api/study/transcript")
def study_transcript(session: str = "") -> JSONResponse:
    """One session's turns, for reading in the dashboard before downloading."""
    from brain import study

    if not session:
        return JSONResponse({"ok": False, "error": "No session given."}, status_code=400)
    return JSONResponse({"session": session, "turns": study.transcript(session)})


@app.get("/api/study/download")
def study_download(session: str = "", fmt: str = "csv") -> Response:
    """Download one session as CSV (for analysis) or text (for reading).

    Served as an attachment with the session id in the filename, because these
    files get dragged into a folder next to a dozen others and a download
    called "download" is one somebody has to open to identify.

    CSV is written through the csv module rather than by joining commas: a
    participant saying 'I said "no", then left' has a quote and a comma in one
    turn, and hand-rolled CSV corrupts the row silently -- the kind of damage
    found weeks later, in analysis, with the robot packed away.
    """
    from brain import study

    if not session:
        return JSONResponse({"ok": False, "error": "No session given."}, status_code=400)
    turns = study.transcript(session)
    if not turns:
        return JSONResponse({"ok": False, "error": "No such session."}, status_code=404)

    safe = "".join(c for c in session if c.isalnum() or c in "-_")[:32] or "session"
    if fmt == "txt":
        lines = [
            f"Reachy Mini research session {session}",
            f"Condition: {turns[0]['condition'] or '(none)'}",
            f"Armed by:  {turns[0]['operator'] or '(unnamed)'}",
            f"Turns:     {len(turns)}",
            "",
            "-" * 60,
            "",
        ]
        for t in turns:
            lines.append(f"[{t['at']}]  persona={t['persona'] or '-'}  "
                         f"first word {t['first_word_s']:.2f}s  ({t['backend'] or '-'})")
            lines.append(f"  Person: {t['said']}")
            lines.append(f"  Reachy: {t['replied']}")
            lines.append("")
        body = "\n".join(lines)
        media, ext = "text/plain; charset=utf-8", "txt"
    else:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(["at", "session", "condition", "persona", "operator",
                         "said", "replied", "first_word_s", "latency_s", "backend"])
        for t in turns:
            writer.writerow([t["at"], session, t["condition"], t["persona"],
                             t["operator"], t["said"], t["replied"],
                             f"{t['first_word_s']:.3f}", f"{t['latency_s']:.3f}",
                             t["backend"]])
        body = buf.getvalue()
        media, ext = "text/csv; charset=utf-8", "csv"

    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="reachy-session-{safe}.{ext}"'},
    )


@app.get("/api/stats")
def stats(day: str = "") -> JSONResponse:
    """What happened during a visit: counters, demos run, questions asked.

    Aggregate only -- no names, and no link between a question and a person.
    See brain/stats.py for why that boundary is where it is.
    """
    from brain import stats as visit_stats

    return JSONResponse({
        "today": visit_stats.day(day or None),
        "days": visit_stats.recent_days(),
    })


#: Room for a draft. The standing llm_max_tokens is 140 -- enough for what the
#: robot says out loud, nowhere near enough for five steps of JSON.
_DRAFT_MAX_TOKENS = 900


def _closed(text: str) -> str:
    """A truncated JSON object with its brackets shut, for one more parse attempt.

    Only ever a salvage attempt: the result is handed to json.loads like any
    other candidate, so a guess that does not parse is discarded rather than
    trusted.
    """
    trimmed = text.rstrip().rstrip(",")
    # Cut back to the last complete step rather than trying to repair a
    # half-written one. A reply cut off mid-sentence leaves both an unbalanced
    # quote and a dangling key, and guessing at either produces a step whose
    # text is a fragment -- which the robot would then say out loud.
    last = trimmed.rfind("}")
    if last != -1:
        trimmed = trimmed[: last + 1]
    closing = "]" * max(0, trimmed.count("[") - trimmed.count("]"))
    closing += "}" * max(0, trimmed.count("{") - trimmed.count("}"))
    return trimmed + closing


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
    # Three attempts, loosest last. A model that runs long leaves the closing
    # fence off, and a draft that is 95% there is worth salvaging rather than
    # showing somebody a bare ```json.
    fence = (
        _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, _re.S)
        or _re.search(r"```(?:json)?\s*(\{.*)", text, _re.S)
        or _re.search(r"(\{.*\})", text, _re.S)
    )
    draft = None
    if fence:
        candidate = fence.group(1).strip()
        parsed = None
        for attempt in (candidate, _closed(candidate)):
            try:
                parsed = _json.loads(attempt)
                break
            except (ValueError, TypeError):
                continue
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
