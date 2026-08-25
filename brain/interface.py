"""The public entry points body/ calls into brain/."""

import logging
import re
import time
from typing import Iterator

from . import llm, long_term_memory, memory, qa_cache
from .emotion import extract_emotion_tag
from .llm import generate_response
from .prompts import build_messages

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")

#: When the OPENING sentence runs past this many characters with no end in
#: sight, it is flushed to the speaker at a clause break instead of waited out.
#: Time-to-first-words is set by the first flush, and models writing a long
#: opener ("That's a great question about the Hub's three research strands,
#: which...") used to hold the robot silent for the whole sentence. A comma is
#: a place a person would breathe, so piper starting there sounds like pacing,
#: not a glitch. See _LONG_SENTENCE_CHARS for the mid-reply case, which this
#: comment used to deny existed.
#: 38, down from 60, on a measurement: piper renders a whole 12-14 word first
#: sentence in 1.1-1.7s on the live voice, but a 5-7 word clause in ~0.5s --
#: so flushing at the first comma of nearly every conversational opener puts
#: sound in the room 0.5-1.2s sooner. The prosody cost (a comma becomes a
#: harder stop) was accepted when this mechanism shipped at 60.
_FIRST_CLAUSE_CHARS = 38

#: And the same trick mid-reply, at a much higher bar. Only a sentence that has
#: genuinely run away -- roughly twenty-five words with no end in sight -- is
#: worth breaking, because every break trades a little naturalness for silence
#: avoided, and that trade is only good when the silence would be long.
_LONG_SENTENCE_CHARS = 150
_CLAUSE_BREAK_RE = re.compile(r"(?<=,)\s+|(?<=;)\s+|\s+(?=--\s)")

#: Token budget for a story. The word limit in the storyteller prompt is a
#: nudge, not a control -- asked for eighty words it writes closer to two
#: hundred whatever the phrasing -- so this is what actually bounds the
#: length, sized above what it really produces plus the emotion tag. Erring
#: high on purpose: a story that stops mid-sentence is worse than a long one.
_STORY_MAX_TOKENS = 280

#: Timing for the most recent reply, for callers that need to MEASURE the turn
#: rather than just hear it -- demos/study.py, which was recording the wall
#: time around ctx.reply() and therefore recording how long the answer took to
#: SPEAK. A ten-word reply and a hundred-word reply from the same model looked
#: like a fast robot and a slow one.
#:
#: A module global rather than a return value because stream_reply is a
#: generator consumed sentence by sentence by the speaking pipeline: there is
#: no single return to hang this on, and the caller has already moved on to
#: playing audio by the time the last sentence arrives. Replies happen one at a
#: time on the voice-loop thread, so there is nothing to race with.
_LAST_TURN = {"first_word_s": 0.0, "backend": ""}


def _reset_turn_stats() -> None:
    _LAST_TURN["first_word_s"] = 0.0
    _LAST_TURN["backend"] = ""


def _mark_first_words(elapsed: float, backend_name: str) -> None:
    _LAST_TURN["first_word_s"] = float(elapsed)
    _LAST_TURN["backend"] = backend_name or ""


def last_turn_stats() -> dict:
    """Time to first spoken words, and which backend produced them.

    A copy: the caller must not be able to edit the live record, and the next
    turn overwrites this one.
    """
    return dict(_LAST_TURN)


def get_reply(person_id: int, message: str) -> tuple[str, str]:
    """Return (reply_text, emotion_tag) for a message from the given person."""
    history = memory.get_history(person_id)
    context = long_term_memory.get_context(person_id)
    messages = build_messages(context, history, message)
    raw_output = generate_response(messages)
    reply_text, emotion_tag = extract_emotion_tag(raw_output)
    memory.remember_turn(person_id, message, reply_text)
    return reply_text, emotion_tag


def stream_reply(
    person_id: int,
    message: str,
    style: str | None = None,
    extra_system: str | None = None,
    cache: bool = True,
    web: bool = False,
) -> Iterator[tuple[str, str]]:
    """Yield (sentence_text, emotion_tag) pairs as the reply streams in, so the
    caller can start speaking sentence 1 while the model is still generating
    sentence 2+ -- the point is reducing time-to-first-words on hardware too
    slow for full-reply latency to feel conversational (see llm_max_tokens'
    docstring in config.py).

    Each sentence is yielded as soon as it is complete. An earlier version held
    the newest one back until the one after it arrived, so that the last
    sentence could be paired with the reply's trailing "[emotion: tag]". That
    cost a whole sentence of generation before the robot said anything --
    measured live at 3.8s to first word against 0.36s to first token -- and it
    read as the robot pausing before every answer. The tag is now carried by a
    final pair instead, whose text is usually empty (the tag is all that is
    left after the last sentence boundary); callers use it for the closing
    gesture and skip speaking when there is nothing to say.

    Sentences are tagged "thinking" because the real one isn't known until the
    reply ends. Every yielded sentence is passed through extract_emotion_tag,
    so a tag the model puts mid-reply is stripped rather than spoken.

    Failover between models lives here rather than in brain/llm.py because the
    decision needs a fact only this function has: whether anything has actually
    been SPOKEN. llm.py can only see tokens, and tokens are not speech -- a
    cloud model that dies after three tokens has produced nothing the caller
    ever passed to the speaker, so switching to the local model is free and
    invisible. Deciding at the token level instead meant that failure was
    treated as "already talking", and the visitor got the fragment the dead
    model managed ("The answer is probably") instead of a real answer. Once a
    complete sentence has gone out, though, the robot is genuinely mid-utterance
    and a second model must not start a different answer over the top of it.

    A question already answered is replayed from brain/qa_cache.py instead of
    being asked again, which is worth 4-9 seconds on the local model every time
    a new group asks the same thing. The replay is fed through the same
    sentence splitting as a live stream so the caller cannot tell the two
    apart, and it still records the turn in memory. Two things are never
    cached: anything with a `style`, because the storyteller prompt exists to
    produce a different story every time and serving yesterday's would be an
    obvious regression; and, inside qa_cache, anything about the person asking.
    `extra_system` is part of the cache key rather than a reason to skip the
    cache -- the Hub briefing is on every turn of the conversation demo, so
    skipping would leave the cache dead exactly where it earns its keep, while
    ignoring it would answer as the wrong persona.
    """
    if style is None:
        cached = qa_cache.lookup(message, extra_system) if cache else None
        if cached is not None:
            # A replayed answer has no model latency at all. Labelled rather
            # than left at the default so a study can tell "answered instantly
            # from cache" apart from "never got a first sentence away", which
            # are the same zero otherwise.
            _reset_turn_stats()
            _mark_first_words(0.0, "cache")
            # Reconstructed with the tag the model originally ended on, so the
            # loop below is the same text-shape the live path splits: N
            # sentences tagged "thinking", then the real tag on a final pair
            # whose text is whatever the last boundary left over.
            replay = f"{cached.text} [emotion: {cached.emotion}]"
            match = _SENTENCE_BOUNDARY_RE.search(replay)
            while match:
                candidate, _ = extract_emotion_tag(replay[: match.end()])
                replay = replay[match.end() :]
                if candidate:
                    yield candidate, "thinking"
                match = _SENTENCE_BOUNDARY_RE.search(replay)
            tail, _ = extract_emotion_tag(replay)
            yield tail, cached.emotion
            memory.remember_turn(person_id, message, cached.text)
            return

    history = memory.get_history(person_id)
    context = long_term_memory.get_context(person_id)
    messages = build_messages(
        context, history, message, style=style, extra_system=extra_system, web=web
    )

    max_tokens = _STORY_MAX_TOKENS if style == "story" else None
    backends = llm.streaming_backends()

    buffer = ""
    raw_parts: list[str] = []
    spoken_a_sentence = False
    asked_at = time.monotonic()
    # Cleared per turn so a turn whose backend never produced a first sentence
    # reports nothing, rather than silently inheriting the previous turn's
    # timing -- which in a study would read as a fast reply that never happened.
    _reset_turn_stats()

    for attempt, backend in enumerate(backends):
        # Reset per attempt: a failed backend's partial words must not be
        # glued onto the replacement's answer.
        buffer = ""
        raw_parts = []
        try:
            for piece in backend.stream(messages, max_tokens=max_tokens, web=web and backend.supports_web):
                raw_parts.append(piece)
                buffer += piece
                match = _SENTENCE_BOUNDARY_RE.search(buffer)
                while match:
                    candidate, _ = extract_emotion_tag(buffer[: match.end()])
                    buffer = buffer[match.end() :]
                    if candidate:
                        if not spoken_a_sentence:
                            _mark_first_words(time.monotonic() - asked_at, backend.name)
                            logger.info("first words in %.2fs (%s)", time.monotonic() - asked_at, backend.name)
                        spoken_a_sentence = True
                        yield candidate, "thinking"
                    match = _SENTENCE_BOUNDARY_RE.search(buffer)
                # A long opening sentence is flushed at its first clause break
                # rather than waited out -- see _FIRST_CLAUSE_CHARS. Only before
                # anything has been spoken: this exists purely to start the
                # voice sooner, and it also marks the turn as mid-utterance so
                # a backend failure after this point can no longer restart the
                # answer with the other model over the top of it, exactly as a
                # full sentence would have.
                # A sentence that runs long is flushed at a clause break rather
                # than waited out. Two thresholds, because the two cases are
                # different jobs:
                #
                #   Opening sentence  -- get the voice started at all.
                #   Mid-reply         -- keep it started.
                #
                # This used to be first-sentence-only, on the reasoning that
                # "mid-reply, speech is already ahead of generation and
                # splitting buys nothing". That holds for ordinary sentences and
                # fails badly for long ones: nothing can be rendered until a
                # whole sentence has arrived, so a 35-word sentence leaves the
                # speaker silent for as long as the model takes to finish it.
                # Measured live, a three-sentence answer had a twenty-second gap
                # in the middle of it -- long enough that a visitor walks away
                # believing the robot has finished.
                #
                # The mid-reply threshold is much higher than the opening one on
                # purpose. Splitting costs naturalness -- a comma becomes a hard
                # stop -- and is only worth paying when the alternative is
                # silence, which is what a very long run means.
                clause_floor = (_FIRST_CLAUSE_CHARS if not spoken_a_sentence
                                else _LONG_SENTENCE_CHARS)
                if len(buffer) >= clause_floor:
                    clause = _CLAUSE_BREAK_RE.search(buffer)
                    if clause:
                        candidate, _ = extract_emotion_tag(buffer[: clause.end()])
                        buffer = buffer[clause.end() :]
                        if candidate:
                            if not spoken_a_sentence:
                                _mark_first_words(time.monotonic() - asked_at, backend.name)
                                logger.info("first words in %.2fs (%s, clause)",
                                            time.monotonic() - asked_at, backend.name)
                            spoken_a_sentence = True
                            yield candidate, "thinking"
            break
        except Exception as exc:
            last_attempt = attempt == len(backends) - 1
            if spoken_a_sentence or last_attempt:
                logger.warning("%s stream failed: %s", backend.name, exc)
                break
            logger.warning(
                "%s stream failed before speaking (%s); retrying with %s",
                backend.name,
                exc,
                backends[attempt + 1].name,
            )

    raw_output = "".join(raw_parts)
    reply_text, emotion_tag = extract_emotion_tag(raw_output)

    # The one place that holds BOTH the visitor's question and the finished
    # reply. demokit/runner.py counts the question at dispatch, where the reply
    # does not exist yet, and the demos each hold both but recording it in each
    # would be five copies of one rule. Nothing in the pipeline ever tagged a
    # refusal, so "the robot did not know this" has to be read back out of its
    # own prose -- see stats.looks_like_a_deflection.
    try:
        from . import stats

        if stats.looks_like_a_deflection(reply_text):
            stats.note_deflection(message)
    except Exception:  # pragma: no cover - a counter must never cost a turn
        logger.debug("Could not record a deflection", exc_info=True)

    tail, _ = extract_emotion_tag(buffer)
    yield tail, emotion_tag

    memory.remember_turn(person_id, message, reply_text)
    if style is None:
        # Cached only when nothing in this session could have shaped the
        # answer. The live prompt is built from this person's history and their
        # long-term notes, so an answer given three turns into one group's
        # conversation is partly about that conversation -- stored globally, it
        # would be replayed verbatim to the next group who asked the same words.
        # Every unrecognised visitor shares person_id 0, so that is not a corner
        # case; it is the open-day default.
        # Never cached: a live answer is by definition one that changes, so
        # serving today's weather again tomorrow is worse than not caching.
        if cache and not web and not history and not context:
            qa_cache.remember(message, reply_text, emotion_tag, extra_system)


def end_conversation(person_id: int) -> None:
    """Summarize this session's conversation into long-term memory, then clear it.

    Call this once a conversation with `person_id` is over (e.g. after a
    period of silence, or when the app is shutting down) -- not after every
    turn.
    """
    history = memory.get_history(person_id)
    long_term_memory.end_conversation(person_id, history)
    memory.clear_history(person_id)
