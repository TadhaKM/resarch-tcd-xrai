"""Runs the features Hub staff build from the dashboard.

Underscore-prefixed so Registry.discover() skips this file, which is
load-bearing rather than tidy: scanned, _demos_in would instantiate
StoredFeature with no record and publish a ghost demo called "stored" with no
script in it. Instances are built by load_into_registry() below and handed to
Registry.register() instead.

One class, one instance per stored feature. Not classes synthesised with
type(): registry.py already assigns id and label as instance attributes over
the ClassVars, and every consumer -- the sort key, dashboard_entries,
_trigger_table, is_available -- reads them off the instance. So N instances
need no framework strain, and a traceback names a file somebody can open.

What a feature may do is deliberately small: say something, ask something,
dance, wait for a face. Those four were chosen because each already exists and
is already tested somewhere in this codebase, so a staff member cannot assemble
a behaviour nobody has run. What they can write freely is the words, which is
the part they actually wanted.

The interpreter does one step per idle slice, exactly as demos/welcome.py
speaks one sentence per slice, and for the same reason: nothing consumes the
microphone while the robot talks, so a script spoken in one call is a script
during which the robot is deaf and cannot be switched away from.
"""

import logging
import time
from typing import Optional

from brain import features
from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S, split_sentences

logger = logging.getLogger(__name__)

#: Gap left after each spoken sentence. Never 0.0: DemoRunner.cycle returns
#: without opening the microphone at all on a zero, and welcome.py records what
#: that cost -- six consecutive zeroes made the robot deaf for the whole
#: thirty-two seconds of its script. One second is short enough that the lines
#: still land as continuous speech.
_BETWEEN_LINES_S = 1.0

#: How long each ASK waits for the visitor to START answering. Once they are
#: talking the recogniser's own endpoint finishes the sentence, so this bounds
#: silence rather than answers. Matches demos/vision.py's enrolment.
_ANSWER_WAIT_S = 5.0

#: Said when a feature has no steps left. Long, because there is nothing to
#: advance and the robot should simply be listening.
_IDLE_S = MAX_LISTEN_WINDOW_S


class StoredFeature(Demo):
    """One staff-written feature, played one step per idle slice."""

    def __init__(self, record: features.Feature) -> None:
        # Instance attributes over the ClassVars, which is how registry.py
        # assigns id and label to every demo it discovers.
        self.id = record.id
        self.label = record.label
        self.help = record.help or "Made from the dashboard."
        self.order = features.FEATURE_ORDER
        self.triggers = (record.trigger_phrase,) if record.trigger_phrase else ()
        self.persona = record.persona
        # Never claims utterances: ctx.ask listens inside the hook, so an answer
        # never reaches on_utterance and there is nothing to protect -- and a
        # visitor saying "let's dance" during a feature should still get the
        # dance. Never requires "faces" either: a WAIT step degrades to a skip
        # without a camera, which is better than greying the whole button out.
        self.claims_utterances = False
        self.requires = ()
        self._blocks = list(record.blocks)
        self._version = record.updated_at

    # --- hooks -----------------------------------------------------------

    def on_enter(self, ctx: DemoContext) -> None:
        """Arm from the first step, and say nothing yet.

        Speaking here would double the first line: on_idle runs immediately
        afterwards and would speak it again.
        """
        self._begin(ctx)
        ctx.status(f"{self.label}: {len(self._blocks)} step(s) ready.")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        if store.get("version") != self._version:
            # Edited from the dashboard while this store still held the old
            # cursor. Start the new version from the top rather than resuming
            # at a step number that no longer means the same thing.
            self._begin(ctx)

        # A sentence still owed from the step before.
        lines = store.get("lines") or ()
        index = store.get("line", 0)
        if index < len(lines):
            self._say(ctx, lines[index], store.get("emotion", "neutral"))
            store["line"] = index + 1
            if store["line"] >= len(lines):
                store["lines"] = ()
                store["cursor"] = store.get("cursor", 0) + 1
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        cursor = store.get("cursor", 0)
        if cursor >= len(self._blocks):
            return IdleResult(listen_for=_IDLE_S)

        return self._step(ctx, self._blocks[cursor])

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        """Only two things are ours; everything else is a question for the model."""
        lowered = text.lower().strip()
        if not lowered:
            return False

        # Its own trigger, said while it is already running. _switch_on_trigger
        # returns None for the demo already selected, so without this the phrase
        # would not start it again for the next group.
        if self.triggers and self.triggers[0] and self.triggers[0] in _words(lowered):
            self._begin(ctx)
            ctx.status(f"{self.label}: starting again.")
            return True

        last = ctx.store.get("last_said")
        if last and _wants_a_repeat(lowered):
            self._say(ctx, last[0], last[1])
            return True
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        done = ctx.store.get("cursor", 0)
        ctx.store["waiting_since"] = None
        if done < len(self._blocks):
            ctx.status(f"{self.label}: stopped after {done} of {len(self._blocks)} steps.")

    # --- the steps -------------------------------------------------------

    def _step(self, ctx: DemoContext, block: features.Block) -> IdleResult:
        """Do one step, then advance past it.

        The order matters. Advancing after the action means a step that throws
        is retried on the next slice and the runner sets the whole feature
        aside after three -- rather than a step being skipped silently, or
        retried forever.
        """
        store = ctx.store

        if block.kind == features.SAY:
            lines = split_sentences(block.text)
            if not lines:
                # Nothing to say. Validation and parse_blocks both reject an
                # empty SAY, so this is unreachable from the dashboard -- but
                # falling through without advancing would spin on this step
                # forever, and "unreachable" is a poor reason to risk that.
                store["cursor"] = store.get("cursor", 0) + 1
                return IdleResult(listen_for=_BETWEEN_LINES_S)
            store["lines"] = lines
            store["line"] = 0
            store["emotion"] = block.emotion
            # Straight into the sentence pump, so the first line lands in this
            # slice rather than a second later. One level of re-entry, and the
            # pump cannot reach back here.
            return self.on_idle(ctx)

        if block.kind == features.ASK:
            return self._ask(ctx, block)

        if block.kind == features.DANCE:
            ctx.motion.dance()
            store["cursor"] = store.get("cursor", 0) + 1
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        if block.kind == features.WAIT:
            return self._wait(ctx, block)

        if block.kind == features.PLAY:
            return self._play(ctx, block)

        # Unknown kind. parse_blocks drops these, so reaching here means a
        # build that knows fewer kinds than the one that wrote the row.
        store["cursor"] = store.get("cursor", 0) + 1
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    def _ask(self, ctx: DemoContext, block: features.Block) -> IdleResult:
        """Ask, listen, and answer on the following slice if asked to.

        Split across two slices deliberately. One slice that speaks a question,
        waits five seconds for speech, transcribes it, calls the model and
        speaks the answer is well past the runner's six-second hook warning and
        leaves fifteen-odd seconds in which the operator cannot switch away.
        """
        store = ctx.store
        pending = store.pop("pending_reply", None)
        if pending:
            # cache=False: this exchange belongs to the group standing here, and
            # a cached answer would be replayed verbatim to the next one.
            ctx.reply(pending, person_id=ctx.person_id(), cache=False)
            store["cursor"] = store.get("cursor", 0) + 1
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        # The question and the listen are separate slices. Fused, this hook
        # spoke the question AND sat through the whole answer wait in one
        # call -- logged live at 7.6s and 17.5s per on_idle, the two longest
        # hook warnings in the log, and for that whole time the operator
        # could not switch away.
        if not store.pop("asked", None):
            ctx.say(block.text, block.emotion)
            store["asked"] = True
            store["last_said"] = (block.text, block.emotion)
            ctx.state.hold_open_mic(True)
            return IdleResult(listen_for=0.0)
        ctx.state.hold_open_mic(False)
        heard = ctx.listen(wait_for_speech_s=_ANSWER_WAIT_S)
        if block.ai_reply and heard:
            store["pending_reply"] = heard
            # A short positive window, NOT zero: the question slice already
            # returned this hook's one permitted zero, and zero-zero in a row
            # is the never-deaf invariant test [18] enforces -- the split that
            # un-fused ask-and-listen tripped exactly that check the first
            # time it ran.
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        store["cursor"] = store.get("cursor", 0) + 1
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    def _play(self, ctx: DemoContext, block: features.Block) -> IdleResult:
        """Hand the visit to another demo, and stop.

        This is what turns a feature into a running order: welcome the group,
        then play the tour, then play the quiz. Staff currently improvise that
        sequence live and lose their place; scripting it is the same act as
        scripting one welcome.

        The cursor advances BEFORE the switch, so coming back to this feature
        later resumes after the step that already ran rather than looping
        through the same handover forever -- a two-line mistake that would look
        like the robot being stuck in the tour.

        Deliberately a HANDOVER, not a nested run: the other demo takes the
        floor and keeps it. A feature that resumed afterwards would have to sit
        watching for the other demo to finish, and demos here have no notion of
        finishing -- the storyteller and the conversation both run until
        somebody switches away.
        """
        store = ctx.store
        store["cursor"] = store.get("cursor", 0) + 1
        target = (block.demo or "").strip()
        # Checked now rather than at parse time: the registry is still being
        # built when features are first read, and a demo can be disabled or
        # deleted between saving this step and running it.
        from demokit.registry import REGISTRY

        demo = REGISTRY.get(target)
        if demo is None:
            ctx.status(f"{self.label}: '{target}' is not installed; skipping that step.")
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        available, reason = REGISTRY.is_available(target, ctx.state.capabilities)
        if not available:
            ctx.status(f"{self.label}: {target} is {reason}; skipping that step.")
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        ctx.status(f"{self.label}: handing over to {demo.label}.")
        # set_mode rather than anything more direct: it is the one path that
        # notifies the runner, clears the last visitor's QR links, and leaves
        # the dashboard showing what is actually running.
        ctx.state.set_mode(target)
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    def _wait(self, ctx: DemoContext, block: features.Block) -> IdleResult:
        """Hold until somebody is in front of the robot, or give up."""
        store = ctx.store
        tracker = ctx.tracker
        if tracker is None or not getattr(tracker, "enabled", False):
            # No camera. face_visible() would say False forever, so waiting
            # would mean the feature never runs at all.
            store["cursor"] = store.get("cursor", 0) + 1
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        if ctx.face_visible():
            store["waiting_since"] = None
            store["cursor"] = store.get("cursor", 0) + 1
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        if store.get("waiting_since") is None:
            store["waiting_since"] = time.monotonic()
        elif time.monotonic() - store["waiting_since"] >= block.seconds:
            ctx.status(f"{self.label}: nobody appeared; carrying on.")
            store["waiting_since"] = None
            store["cursor"] = store.get("cursor", 0) + 1
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        # Nothing to advance, so listen for as long as the core allows.
        return IdleResult(listen_for=_IDLE_S)

    # --- helpers ---------------------------------------------------------

    def _begin(self, ctx: DemoContext) -> None:
        ctx.store.update(
            version=self._version, cursor=0, lines=(), line=0,
            emotion="neutral", waiting_since=None, pending_reply=None,
        )

    @staticmethod
    def _say(ctx: DemoContext, text: str, emotion: str) -> None:
        """Speak one line, and remember it as the one a repeat asks for."""
        ctx.say(text, emotion)
        ctx.store["last_said"] = (text, emotion)


#: Ways a visitor asks to hear the last line again, borrowed from welcome.py
#: rather than reinvented -- it is the same question about the same robot.
_REPEAT_PHRASES = (
    "say that again", "say it again", "repeat that", "repeat it",
    "one more time", "once more", "what was that", "come again",
    "didn't catch", "did not catch", "didn't hear", "did not hear",
)
_REPEAT_WORDS = frozenset({"again", "repeat", "pardon"})
_REPEAT_MAX_WORDS = 3


def _words(text: str) -> str:
    from demokit.runner import _word_stream

    return _word_stream(text)


def _wants_a_repeat(lowered: str) -> bool:
    if any(phrase in lowered for phrase in _REPEAT_PHRASES):
        return True
    words = set(lowered.split())
    return len(words) <= _REPEAT_MAX_WORDS and bool(words & _REPEAT_WORDS)


def load_into_registry() -> tuple[int, list[str]]:
    """Put every enabled stored feature into the registry. (loaded, problems).

    Called once at startup after discover(), and again by the web layer each
    time a feature is saved or deleted. A feature whose blocks all failed to
    parse is skipped rather than registered empty -- an empty button that does
    nothing when pressed is worse than one that is not there.
    """
    from demokit.registry import REGISTRY

    loaded, problems = 0, []
    for record in features.list_features(only_enabled=True):
        if not record.blocks:
            problems.append(f"{record.label or record.id}: no usable steps; skipped")
            logger.warning("Stored feature %s has no usable steps; not registered", record.id)
            continue
        if REGISTRY.register(StoredFeature(record)):
            loaded += 1
        else:
            problems.append(f"{record.label or record.id}: name clashes with a built-in demo")
    if loaded:
        logger.info("Loaded %d feature(s) written from the dashboard.", loaded)
    return loaded, problems


def sync_one(feature_id: str) -> bool:
    """Register, replace or remove one feature to match the database."""
    from demokit.registry import REGISTRY

    record = features.get_feature(feature_id)
    if record is None or not record.enabled or not record.blocks:
        return REGISTRY.unregister(feature_id)
    return REGISTRY.register(StoredFeature(record))
