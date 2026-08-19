"""How the dashboard's buttons are arranged: folders, and the order in them.

Staff at the Hub end up with a grid of twenty-odd buttons -- ten demos in the
code and as many features as they have written for particular visits -- and no
way to say that four of them are the school-visit ones. This module stores that
grouping so it survives a restart and so two people looking at two phones see
one robot.

THE ONE RULE, and everything else follows from it: this is DISPLAY ORDER, never
the order the robot uses to choose a demo. brain/modes.py:132 takes the FIRST
entry of the demo list as the demo to boot into, and DEFAULT_MODE is "" so that
line runs on every start. If dragging a button to the top of the grid reordered
the demo list, a staff member tidying up on a Tuesday would change what the
robot does on Wednesday morning, and nothing on the dashboard would say so. So
the layout is a separate document that the BROWSER composes the grid from, and
demokit/registry.py is not modified at all -- its (order, label) sort, its
default_id() and its dashboard_entries() are exactly what they were.

That also makes this table non-load-bearing, which is the property worth
protecting on a robot used in front of school groups: with _available False, or
an empty document, or this whole file deleted, the dashboard is the one that
shipped -- every button, flat, in the order the code defines. Nothing here may
become something the grid needs in order to render.

One row, one JSON document, following brain/features.py's own stated test:
nothing ever reads part of a layout, the browser sends the whole materialised
order on every write, and a normalised table would add orphaned rows as a
failure mode to a module whose contract is to degrade rather than raise. The
private names borrowed from brain/db.py (_write_lock, _connection, _now) are
the ones qa_cache.py and features.py borrow, and for the same reason: a second
lock would not be serialized against the writes db.py is already doing.

`rev` is durable and lives in the same row, incremented inside the same
transaction as the write. A process-local counter would go backwards when the
robot restarts under a browser that never reloaded, which forces that browser
into comparing revisions for inequality rather than for age -- and inequality
is exactly what lets a poll issued before a write and resolved after it repaint
the pre-move order half a second after somebody's drop.
"""

import json
import logging
import re
import sqlite3
from typing import Any, Optional

from brain import db

logger = logging.getLogger(__name__)

#: Ceilings. Each is the point past which the thing stops working as a way to
#: find a button quickly, rather than a technical limit.
MAX_FOLDERS = 12               # a phone screen; also stops "New folder" being infinite
MAX_FOLDER_NAME_CHARS = 24     # shorter than a feature label's 40: this is a heading
MAX_ROOT_ENTRIES = 80
MAX_FOLDER_ITEMS = 80
#: Ids kept in total. Ten committed demos plus features.MAX_FEATURES (40) can
#: be live at once; the rest of this is history, and _coerce drops the tail.
MAX_IDS = 200

#: Folder ids are minted by the browser, so they are checked rather than
#: trusted. Anything else -- a path, a demo id, a name -- is dropped.
_FOLDER_ID_RE = re.compile(r"^f_[0-9a-f]{8}$")
#: What a demo id can look like: module stems and features.slug_for output.
_DEMO_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
#: Identical to features.py's. Printable text is stored exactly as typed --
#: see the note on validate() there about why nothing else is stripped.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: Outcomes of a write. Machine-readable so the web layer picks its status code
#: from a constant rather than string-matching the sentence it also shows.
OK, STALE, INVALID, UNAVAILABLE = "ok", "stale", "invalid", "unavailable"

#: The empty layout, which is also what every failure returns. A dashboard
#: given this renders every button flat, in the robot's own order.
EMPTY: dict = {"rev": 0, "items": []}

#: False when the table could not be created, which turns every function here
#: into a no-op returning EMPTY. Same reason as brain/features.py: this module
#: is imported by the web server and by the voice loop, and a failed CREATE
#: TABLE must not propagate out of an import and take the robot down over the
#: arrangement of some buttons.
_available = True


# --- reading, which must never raise ------------------------------------

def _clean_name(raw: Any) -> str:
    """A folder heading as it will be stored. Never empty, never control chars.

    Coerced on read as well as on write, so a hand-edited row cannot paint a
    heading the stylesheet was not built for. A name that is empty after
    cleaning becomes "Folder" rather than dropping the entry: a nameless folder
    still holds buttons, and dropping it would strand them.
    """
    text = raw if isinstance(raw, str) else ""
    text = _CONTROL_RE.sub(" ", text)
    text = " ".join(text.split())[:MAX_FOLDER_NAME_CHARS].strip()
    return text or "Folder"


def _coerce(raw: Any) -> list[dict]:
    """The items array from whatever was stored. Total, and never raises.

    Drops an entry that is not a dict, a folder id that is not one we minted, a
    demo id with anything odd in it, a folder nested inside a folder, and the
    SECOND appearance of any id -- which is what removes "the same button
    appears twice in the grid" from the set of things that can happen.

    Ids are never resolved against the registry here, deliberately.
    Registry.discover() skips any module that fails to import (registry.py:82),
    so "this demo is missing" is routinely temporary -- a student left a syntax
    error in demos/dance.py this morning -- and reaping on sight would mean one
    bad afternoon silently wipes an arrangement somebody spent half an hour
    building on a phone. The browser filters at render time, where it is free
    and where it undoes itself the moment the demo comes back.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []

    out: list[dict] = []
    seen: set[str] = set()
    folders = 0

    def take(demo_id: Any) -> Optional[str]:
        if not isinstance(demo_id, str) or not _DEMO_ID_RE.match(demo_id):
            return None
        if demo_id in seen or len(seen) >= MAX_IDS:
            return None
        seen.add(demo_id)
        return demo_id

    for entry in raw:
        if len(out) >= MAX_ROOT_ENTRIES:
            break
        if not isinstance(entry, dict):
            continue
        kind = entry.get("t")

        if kind == "f":
            folder_id = entry.get("id")
            if not isinstance(folder_id, str) or not _FOLDER_ID_RE.match(folder_id):
                continue
            if folders >= MAX_FOLDERS or folder_id in seen:
                continue
            seen.add(folder_id)
            folders += 1
            children: list[str] = []
            for child in (entry.get("items") or [])[:MAX_FOLDER_ITEMS]:
                kept = take(child)
                if kept is not None:
                    children.append(kept)
            out.append({
                "t": "f",
                "id": folder_id,
                "name": _clean_name(entry.get("name")),
                "collapsed": bool(entry.get("collapsed")),
                "items": children,
            })
            continue

        kept = take(entry.get("id"))
        if kept is not None:
            out.append({"t": "i", "id": kept})

    return out


def read() -> tuple[dict, bool]:
    """({"rev": n, "items": [...]}, available). Never raises, never None.

    The second value is whether the layout can be written at all, which the
    dashboard uses to grey its Arrange button -- offering a gesture that will
    not stick is worse than not offering it.
    """
    if not _available:
        return dict(EMPTY), False
    try:
        with db._connection() as conn:
            row = conn.execute(
                "SELECT doc, rev FROM dashboard_layout WHERE id = 1"
            ).fetchone()
    except sqlite3.Error:
        logger.exception("Failed to read the dashboard layout")
        return dict(EMPTY), False
    if row is None:
        return dict(EMPTY), True
    return {"rev": int(row[1] or 0), "items": _coerce(row[0])}, True


# --- writing ------------------------------------------------------------

def _problems(raw: Any) -> list[str]:
    """What is refused outright, in words an operator can act on.

    Checked against the RAW input, not against the coerced list -- _coerce has
    already clamped that to the limits, so counting folders there would be
    counting its own output and could never exceed anything. Only conditions
    _coerce cannot quietly fix are named; everything else (a dropped duplicate,
    a shortened name) is corrected silently and comes back corrected in the
    response, which is why the browser adopts the robot's copy over its own.
    """
    if not isinstance(raw, list):
        return ["The layout was not sent in a form the robot understands."]
    folders = sum(1 for e in raw if isinstance(e, dict) and e.get("t") == "f")
    if folders > MAX_FOLDERS:
        return [f"There can be at most {MAX_FOLDERS} folders, and that had {folders}."]
    return []


def write(items: Any, base_rev: int = 0) -> tuple[str, dict, list[str]]:
    """Replace the layout. (status, document, problems).

    STALE returns the CURRENT document rather than an error, so a browser that
    lost the race can re-apply its one move on top of the winner's layout and
    send again. Two staff tidying up at once is an ordinary afternoon at an
    open day, not a theoretical race, and silently discarding one of them is
    how somebody ends up dragging the same button four times.

    The revision is re-read INSIDE the write transaction, which is the same
    discipline features.save() uses for its uniqueness check and for the same
    reason: two browsers can otherwise both pass the check before either writes.
    """
    if not _available:
        return UNAVAILABLE, dict(EMPTY), []

    problems = _problems(items)
    if problems:
        return INVALID, dict(EMPTY), problems
    cleaned = _coerce(items)

    doc = json.dumps(cleaned)
    try:
        with db._write_lock, db._connection() as conn:
            row = conn.execute(
                "SELECT doc, rev FROM dashboard_layout WHERE id = 1"
            ).fetchone()
            current = int(row[1] or 0) if row else 0
            if int(base_rev or 0) != current:
                return STALE, {"rev": current, "items": _coerce(row[0]) if row else []}, []
            nxt = current + 1
            conn.execute(
                """
                INSERT INTO dashboard_layout (id, doc, rev, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    doc = excluded.doc,
                    rev = excluded.rev,
                    updated_at = excluded.updated_at
                """,
                (doc, nxt, db._now()),
            )
    except sqlite3.Error:
        # UNAVAILABLE rather than INVALID, and the distinction matters to the
        # person standing there: _available only covers a CREATE TABLE that
        # failed at import, while the failures that actually happen on an open
        # day are a full disk, a read-only filesystem and contention past
        # db._connect()'s five-second busy timeout -- all of which leave
        # _available True. Telling somebody their arrangement is wrong while
        # the disk is full is how you get them re-dragging a button forever.
        logger.exception("Failed to write the dashboard layout")
        return UNAVAILABLE, dict(EMPTY), []
    return OK, {"rev": nxt, "items": cleaned}, []


def forget(demo_id: str) -> None:
    """Drop one id from the stored layout. Called when a feature is deleted.

    Correctness rather than housekeeping. features.slug_for derives the id from
    the label, so "Cork welcome", "Cork Welcome!" and "cork  welcome" all slug
    to custom_cork_welcome. Without this, staff delete an old feature, write an
    unrelated one months later whose name happens to slug the same way, press
    Save, are told it saved and is live -- and no button appears anywhere,
    because the placement row that outlived the first one has put the brand-new
    feature inside a collapsed folder.

    It is also what keeps the document bounded in practice: committed demos are
    never deleted, so stored features are the only source of churn.
    """
    if not _available or not demo_id:
        return
    try:
        with db._write_lock, db._connection() as conn:
            row = conn.execute(
                "SELECT doc, rev FROM dashboard_layout WHERE id = 1"
            ).fetchone()
            if row is None:
                return
            items = _coerce(row[0])
            kept = []
            changed = False
            for entry in items:
                if entry["t"] == "f":
                    children = [c for c in entry["items"] if c != demo_id]
                    changed = changed or len(children) != len(entry["items"])
                    entry["items"] = children
                    kept.append(entry)
                elif entry["id"] == demo_id:
                    changed = True
                else:
                    kept.append(entry)
            if not changed:
                return
            conn.execute(
                "UPDATE dashboard_layout SET doc = ?, rev = ?, updated_at = ? WHERE id = 1",
                (json.dumps(kept), int(row[1] or 0) + 1, db._now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to drop %s from the dashboard layout", demo_id)


def reset() -> dict:
    """Empty the layout and return the new document.

    The escape hatch, and the same instinct as qa_cache.forget_all behind the
    Clear answers button: somebody non-technical, on a phone, will at some
    point produce an arrangement they cannot undo, and there is no undo stack.
    """
    if not _available:
        return dict(EMPTY)
    try:
        with db._write_lock, db._connection() as conn:
            row = conn.execute("SELECT rev FROM dashboard_layout WHERE id = 1").fetchone()
            nxt = (int(row[0] or 0) if row else 0) + 1
            conn.execute(
                """
                INSERT INTO dashboard_layout (id, doc, rev, updated_at)
                VALUES (1, '[]', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    doc = '[]', rev = excluded.rev, updated_at = excluded.updated_at
                """,
                (nxt, db._now()),
            )
    except sqlite3.Error:
        logger.exception("Failed to reset the dashboard layout")
        return dict(EMPTY)
    return {"rev": nxt, "items": []}


def _init_layout() -> None:
    """Create the table if it does not exist. Safe to call repeatedly."""
    with db._write_lock, db._connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dashboard_layout (
                id         INTEGER PRIMARY KEY CHECK (id = 1),
                doc        TEXT NOT NULL,
                rev        INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMP
            )
            """
        )


try:
    _init_layout()
except (sqlite3.Error, OSError):
    # OSError as well: db._connect() begins with MODELS.db_path.parent.mkdir(),
    # which raises OSError rather than sqlite3.Error on a read-only filesystem,
    # and that must not escape an import either.
    _available = False
    logger.exception(
        "Dashboard layout unavailable; the demo grid will show every button "
        "flat, in its built-in order"
    )
