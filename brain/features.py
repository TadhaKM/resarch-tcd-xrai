"""Features Hub staff build from the dashboard, stored as data rather than code.

A "feature" is a named button plus a short sequence of steps the robot performs
-- welcome this particular group, ask them something, dance. Staff write them
from the dashboard; nobody edits Python an hour before an industry visit.

Data, deliberately, not generated code. The obvious design is to have the model
write a file into demos/ and import it, and it is the wrong one: web/server.py
has no authentication (its own docstring says so), so an endpoint that writes
and imports Python is remote code execution for anyone on the Hub's network.
Four fixed step kinds, interpreted by one committed and tested file
(demos/_stored.py), bound what a feature can do to what we have already tested.

The store is brain/db.py's database rather than a JSON file beside it, for two
reasons: db.backup_db() copies that file daily and keeps a week of it, so these
are backed up for free where a sidecar would not be; and the private names
reached for below (_write_lock, _connection, _now) are the same ones qa_cache.py
reaches for, and for the same reason -- a second lock would not be serialized
against the writes db.py is already doing.

Blocks live in one JSON column rather than a second table. Nothing ever reads
part of a feature: they are loaded whole at registration and written whole on
save, so a normalised table would buy a capability nobody needs and add
orphaned rows as a failure mode to a module whose contract is to degrade rather
than raise.

The consequence of JSON is that SQLite cannot enforce block validity, so the
discipline is: validate on write, coerce on read. _parse_blocks never raises
and never returns anything malformed, which is what lets the interpreter treat
its input as well-formed by construction.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from brain import db
from brain.emotion import VALID_EMOTION_TAGS

logger = logging.getLogger(__name__)

#: Step kinds a feature may use. Anything else is dropped on read, so a build
#: that knows a fifth kind writes rows an older build merely skips a step in,
#: rather than failing to load the feature at all.
SAY, ASK, DANCE, WAIT = "SAY", "ASK", "DANCE", "WAIT"
BLOCK_KINDS = (SAY, ASK, DANCE, WAIT)

#: Ceilings. Every one is a guess informed by something already measured:
#: hub.py records that a 39-second welcome was "long enough for a group to start
#: looking at each other", and WELCOME_SCRIPT itself is ~32 seconds at roughly
#: 1000 characters.
MAX_FEATURES = 40
MAX_BLOCKS = 12
MAX_SAY_CHARS = 400
MAX_ASK_CHARS = 200
MAX_TOTAL_CHARS = 1200
WARN_TOTAL_CHARS = 600
MAX_LABEL_CHARS = 40
MAX_HELP_CHARS = 120
WAIT_SECONDS_RANGE = (5, 60)
DEFAULT_WAIT_SECONDS = 20

#: Where stored features sort on the dashboard: after every demo shipped in the
#: code, whose orders run 10-900 with the used range 10-90. One less decision
#: for a staff member, and _publish already breaks ties by label.
FEATURE_ORDER = 500

#: Shortest a spoken trigger may be. Directly the lesson of demos/welcome.py:
#: a bare "welcome" fired on "you're welcome", "welcome back" and "welcome to
#: Dublin", dragging a group out of whatever they were being shown. Three words
#: is roughly the point at which a phrase stops occurring by accident.
MIN_TRIGGER_WORDS = 3
MIN_TRIGGER_CHARS = 12

#: Words that carry no request on their own, so a trigger made only of these
#: ("can you do it") would fire on ordinary conversation.
_FUNCTION_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "can", "could", "do", "does",
    "for", "get", "go", "have", "i", "if", "in", "is", "it", "its", "just",
    "let", "lets", "me", "my", "now", "of", "ok", "okay", "on", "one", "or",
    "please", "put", "she", "so", "that", "the", "then", "there", "they",
    "this", "to", "up", "us", "want", "we", "will", "with", "would", "you",
    "your",
})

#: A placeholder the drafting model was told to leave when it does not know a
#: fact. Rejected at save time, which is what makes that instruction enforce
#: itself: ctx.say strips nothing, so "[name of the visiting professor]" would
#: be read out to the room, brackets and all.
_PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")

#: Anything the synthesiser would either voice as noise or choke on. Printable
#: text is left exactly as typed -- see validate() on why nothing is stripped.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: False when the table could not be created, which turns every function here
#: into a no-op. brain/features.py is imported by the web server and by the
#: demo loader; without this a failed CREATE TABLE would propagate out of an
#: import and take the robot down over a feature nobody is using yet.
_available = True


@dataclass
class Block:
    """One step. Only the fields its kind uses are meaningful."""

    kind: str
    text: str = ""
    emotion: str = "neutral"
    ai_reply: bool = False
    seconds: int = DEFAULT_WAIT_SECONDS

    def as_dict(self) -> dict:
        if self.kind == SAY:
            return {"kind": SAY, "text": self.text, "emotion": self.emotion}
        if self.kind == ASK:
            return {"kind": ASK, "text": self.text, "emotion": self.emotion,
                    "ai_reply": bool(self.ai_reply)}
        if self.kind == WAIT:
            return {"kind": WAIT, "seconds": int(self.seconds)}
        return {"kind": DANCE}


@dataclass
class Feature:
    """One stored feature, as the dashboard and the interpreter both see it."""

    id: str = ""
    label: str = ""
    help: str = ""
    trigger_phrase: str = ""
    persona: str = ""
    blocks: list[Block] = field(default_factory=list)
    enabled: bool = True
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "help": self.help,
            "trigger_phrase": self.trigger_phrase,
            "persona": self.persona,
            "blocks": [b.as_dict() for b in self.blocks],
            "enabled": self.enabled,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def spoken_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks if b.kind in (SAY, ASK))


# --- reading, which must never raise ------------------------------------

def _coerce_block(raw: Any) -> Optional[Block]:
    """One block from whatever was stored, or None to drop it.

    Total by design. Everything here is either coerced to something the
    interpreter can act on or discarded, so demos/_stored.py never has to check
    its own inputs and a hand-edited database row cannot reach ctx.say.
    """
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind", "")).upper()
    if kind not in BLOCK_KINDS:
        return None

    if kind == DANCE:
        return Block(kind=DANCE)

    if kind == WAIT:
        try:
            seconds = int(raw.get("seconds", DEFAULT_WAIT_SECONDS))
        except (TypeError, ValueError):
            seconds = DEFAULT_WAIT_SECONDS
        low, high = WAIT_SECONDS_RANGE
        return Block(kind=WAIT, seconds=max(low, min(high, seconds)))

    text = raw.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    cap = MAX_SAY_CHARS if kind == SAY else MAX_ASK_CHARS
    emotion = str(raw.get("emotion", "neutral"))
    if emotion not in VALID_EMOTION_TAGS:
        emotion = "neutral"
    return Block(
        kind=kind,
        text=_CONTROL_RE.sub(" ", text).strip()[:cap],
        emotion=emotion,
        ai_reply=bool(raw.get("ai_reply")) if kind == ASK else False,
    )


def parse_blocks(raw: Any) -> list[Block]:
    """Blocks from a JSON string or an already-decoded list. Never raises."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw[:MAX_BLOCKS]:
        block = _coerce_block(entry)
        if block is not None:
            out.append(block)
    return out


def _row_to_feature(row) -> Feature:
    return Feature(
        id=row[0], label=row[1], help=row[2], trigger_phrase=row[3],
        persona=row[4], blocks=parse_blocks(row[5]), enabled=bool(row[6]),
        created_by=row[7], created_at=row[8] or "", updated_at=row[9] or "",
    )


_COLUMNS = ("id, label, help, trigger_phrase, persona, blocks, enabled, "
            "created_by, created_at, updated_at")


def list_features(only_enabled: bool = False) -> list[Feature]:
    """Every stored feature, oldest first. [] if the table is unavailable."""
    if not _available:
        return []
    where = " WHERE enabled = 1" if only_enabled else ""
    try:
        with db._connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM custom_features{where} ORDER BY created_at"
            ).fetchall()
    except sqlite3.Error:
        logger.exception("Failed to read custom features")
        return []
    return [_row_to_feature(row) for row in rows]


def get_feature(feature_id: str) -> Optional[Feature]:
    if not _available:
        return None
    try:
        with db._connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM custom_features WHERE id = ?", (feature_id,)
            ).fetchone()
    except sqlite3.Error:
        logger.exception("Failed to read custom feature %s", feature_id)
        return None
    return _row_to_feature(row) if row else None


# --- naming and trigger normalisation -----------------------------------

def slug_for(label: str) -> str:
    """A stable id from a label. Prefixed so it can never shadow a code demo."""
    base = re.sub(r"[^a-z0-9]+", "_", (label or "").lower()).strip("_")
    return f"custom_{base[:40]}" if base else ""


def normalise_trigger(phrase: str) -> str:
    """The phrase exactly as the runner will match it, or "" for none.

    Stored in this form on purpose. The runner strips apostrophes and
    punctuation before matching (_word_stream), so a phrase saved as typed
    would be compared against something else -- and "let's dance" would never
    equal "lets dance".
    """
    from demokit.runner import _word_stream

    return _word_stream(phrase or "").strip()


# --- validation, the one implementation ---------------------------------

def validate(feature: Feature, *, existing: Optional[list[Feature]] = None) -> list[str]:
    """Problems with this feature, in words an operator can act on. [] is fine.

    Called before every write and again by tools/selftest.py, so a feature that
    was legal when it was saved and has since been shadowed by a newly
    committed demo is caught before an open day rather than during one.
    """
    from demokit.runner import (NAME_CORRECTION_CUES, SLEEP_PHRASES,
                                contains_phrase)
    from demokit.registry import REGISTRY
    from brain import personas

    problems: list[str] = []
    others = [f for f in (list_features() if existing is None else existing)
              if f.id != feature.id]

    # --- identity ---
    label = (feature.label or "").strip()
    if not label:
        problems.append("Give it a name -- that is what the button says.")
    elif len(label) > MAX_LABEL_CHARS:
        problems.append(f"The name is too long for a button (over {MAX_LABEL_CHARS} characters).")
    elif _CONTROL_RE.search(label):
        problems.append("The name contains characters the robot cannot show.")
    elif any(f.label.strip().lower() == label.lower() for f in others):
        problems.append(f"There is already a feature called {label!r}.")

    if len(feature.help or "") > MAX_HELP_CHARS:
        problems.append(f"The description is over {MAX_HELP_CHARS} characters.")

    if not feature.id:
        problems.append("That name has no letters or numbers in it.")
    elif REGISTRY.get(feature.id) is not None and feature.id not in {f.id for f in others}:
        # A built-in demo already owns this id, so register() would refuse and
        # the button would simply never appear.
        problems.append("That name clashes with one of the robot's own demos. Try another.")

    if len(others) >= MAX_FEATURES and feature.id not in {f.id for f in others}:
        problems.append(f"There are already {MAX_FEATURES} features. Delete one first.")

    # --- the trigger phrase, which is where the sharp edges are ---
    trigger = feature.trigger_phrase
    if trigger:
        words = trigger.split()
        if len(words) < MIN_TRIGGER_WORDS or len(trigger) < MIN_TRIGGER_CHARS:
            problems.append(
                f"The spoken phrase needs at least {MIN_TRIGGER_WORDS} words. Short ones "
                "fire by accident -- a bare \"welcome\" once matched \"you're welcome\"."
            )
        elif all(word in _FUNCTION_WORDS for word in words):
            problems.append("That phrase is too ordinary; it would fire in normal conversation.")

        padded = f" {trigger} "
        if any(contains_phrase(padded, p) for p in SLEEP_PHRASES):
            problems.append("That phrase contains a stop phrase, so it would send the robot to sleep instead.")
        if any(contains_phrase(padded, c) for c in NAME_CORRECTION_CUES):
            problems.append("That phrase is one the robot uses for correcting names, so it would never start this.")
        if "reachy" in words:
            problems.append("Leave the robot's name out of the phrase -- it is stripped before matching.")

        for demo_id in REGISTRY.ids():
            demo = REGISTRY.get(demo_id)
            if demo is None or demo_id == feature.id:
                continue
            for owned in getattr(demo, "triggers", ()):
                owned_norm = normalise_trigger(owned)
                if not owned_norm:
                    continue
                # Both directions are bugs. Containing an existing phrase steals
                # it (longest match wins); being contained by one means this
                # never fires at all.
                if contains_phrase(padded, owned_norm) or contains_phrase(f" {owned_norm} ", trigger):
                    problems.append(
                        f"That phrase clashes with {demo.label!r}, which already answers to "
                        f"{owned!r}."
                    )
                    break

    # --- the steps ---
    blocks = feature.blocks
    if not blocks:
        problems.append("Add at least one step.")
    if len(blocks) > MAX_BLOCKS:
        problems.append(f"That is more than {MAX_BLOCKS} steps. Keep it short enough to watch.")
    if blocks and not any(b.kind == SAY for b in blocks):
        problems.append("At least one step has to be something the robot says.")

    for index, block in enumerate(blocks, start=1):
        where = f"Step {index}"
        if block.kind in (SAY, ASK):
            if not block.text.strip():
                problems.append(f"{where} has nothing to say.")
            leftover = _PLACEHOLDER_RE.search(block.text)
            if leftover:
                problems.append(
                    f"{where} still has {leftover.group(0)!r} in it. The robot would read "
                    "that out -- fill it in or take it out."
                )
            if block.emotion not in VALID_EMOTION_TAGS:
                problems.append(f"{where} has an expression the robot does not have.")
        if block.kind == WAIT:
            low, high = WAIT_SECONDS_RANGE
            if not (low <= block.seconds <= high):
                problems.append(f"{where} must wait between {low} and {high} seconds.")
            if index == len(blocks):
                problems.append("The last step cannot be a wait -- nothing would follow it.")
        if (block.kind == DANCE and index > 1 and blocks[index - 2].kind == DANCE):
            # MotionController.dance drops a second dance while one is running,
            # so back-to-back would silently be a single dance.
            problems.append(f"{where} is a second dance in a row; the robot can only do one at a time.")

    total = feature.spoken_chars()
    if total > MAX_TOTAL_CHARS:
        problems.append(
            f"That is a lot to say at once ({total} characters). Keep it under "
            f"{MAX_TOTAL_CHARS} or split it into two features."
        )

    if feature.persona and feature.persona not in {p.id for p in personas.PERSONAS}:
        problems.append("That personality does not exist.")

    return problems


def warnings_for(feature: Feature) -> list[str]:
    """Things worth saying but not worth refusing over."""
    out = []
    total = feature.spoken_chars()
    if WARN_TOTAL_CHARS < total <= MAX_TOTAL_CHARS:
        out.append(
            f"About {total // 15} seconds of speech. Groups start looking at each "
            "other past about half a minute."
        )
    if not feature.trigger_phrase:
        out.append("No spoken phrase, so this can only be started from the dashboard.")
    return out


# --- writing ------------------------------------------------------------

def save(feature: Feature) -> tuple[bool, list[str]]:
    """Insert or update. (True, []) on success, (False, problems) otherwise.

    Uniqueness is re-checked inside the same transaction as the write. Two
    people with the dashboard open can otherwise both pass validate() before
    either of them writes.
    """
    if not _available:
        return False, ["The robot's database is unavailable, so nothing can be saved."]

    problems = validate(feature)
    if problems:
        return False, problems

    now = db._now()
    try:
        with db._write_lock, db._connection() as conn:
            rows = conn.execute(
                "SELECT id, label FROM custom_features WHERE id != ?", (feature.id,)
            ).fetchall()
            if len(rows) >= MAX_FEATURES:
                return False, [f"There are already {MAX_FEATURES} features. Delete one first."]
            if any(str(r[1]).strip().lower() == feature.label.strip().lower() for r in rows):
                return False, [f"There is already a feature called {feature.label!r}."]

            conn.execute(
                """
                INSERT INTO custom_features
                    (id, label, help, trigger_phrase, persona, blocks, enabled,
                     created_by, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = excluded.label,
                    help = excluded.help,
                    trigger_phrase = excluded.trigger_phrase,
                    persona = excluded.persona,
                    blocks = excluded.blocks,
                    enabled = excluded.enabled,
                    created_by = excluded.created_by,
                    updated_at = excluded.updated_at
                """,
                (
                    feature.id, feature.label.strip(), (feature.help or "").strip(),
                    feature.trigger_phrase, feature.persona,
                    json.dumps([b.as_dict() for b in feature.blocks]),
                    1 if feature.enabled else 0, (feature.created_by or "").strip(),
                    feature.created_at or now, now,
                ),
            )
    except sqlite3.Error:
        logger.exception("Failed to save custom feature %s", feature.id)
        return False, ["The robot could not save that. Try again."]
    logger.info("Saved custom feature %s (%s)", feature.id, feature.label)
    return True, []


def delete(feature_id: str) -> bool:
    if not _available:
        return False
    try:
        with db._write_lock, db._connection() as conn:
            changed = conn.execute(
                "DELETE FROM custom_features WHERE id = ?", (feature_id,)
            ).rowcount
    except sqlite3.Error:
        logger.exception("Failed to delete custom feature %s", feature_id)
        return False
    if changed:
        logger.info("Deleted custom feature %s", feature_id)
    return bool(changed)


def set_enabled(feature_id: str, enabled: bool) -> bool:
    if not _available:
        return False
    try:
        with db._write_lock, db._connection() as conn:
            changed = conn.execute(
                "UPDATE custom_features SET enabled = ?, updated_at = ? WHERE id = ?",
                (1 if enabled else 0, db._now(), feature_id),
            ).rowcount
    except sqlite3.Error:
        logger.exception("Failed to enable/disable custom feature %s", feature_id)
        return False
    return bool(changed)


def _init_features() -> None:
    """Create the table if it does not exist. Safe to call repeatedly."""
    with db._write_lock, db._connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS custom_features (
                id             TEXT PRIMARY KEY,
                label          TEXT NOT NULL,
                help           TEXT NOT NULL DEFAULT '',
                trigger_phrase TEXT NOT NULL DEFAULT '',
                persona        TEXT NOT NULL DEFAULT '',
                blocks         TEXT NOT NULL,
                enabled        INTEGER NOT NULL DEFAULT 1,
                created_by     TEXT NOT NULL DEFAULT '',
                created_at     TIMESTAMP,
                updated_at     TIMESTAMP
            )
            """
        )


try:
    _init_features()
except sqlite3.Error:
    _available = False
    logger.exception("Custom features unavailable; the robot runs with its built-in demos only")
