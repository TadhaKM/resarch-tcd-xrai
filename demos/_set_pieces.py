"""The Hub's approved answers, spoken the same whoever asks, wherever asked.

One source of truth for the dialogue written with the Dean for the MSc welcome
event. The Dean's Office demo performs it on stage cues; conversation answers
the everyday phrasings of the same questions with the same blocks -- instantly,
because a question the script already answers should never wait seconds for a
language model to improvise a worse version of it.

Underscore-prefixed so the registry's discover() never publishes this as a
mode of its own: it is shared material, not a demo.

WHEN A QUESTION IS "THE SAME QUESTION"
Matching is deliberately conservative: a cue phrase must appear in the
utterance AND the utterance must not run more than _MAX_EXTRA_WORDS words past
the cue. That guard is load-bearing -- a three-part question that merely
CONTAINS "about the hub" must reach the model, which can answer all three
parts; handing it a fixed speech instead is the exact live failure recorded in
demokit/runner.py's fuzzy-trigger guard. Short direct question: the script.
Anything richer: the model, whose grounding (brain/hub.py) carries the same
message and can bend it to what was actually asked.
"""

from dataclasses import dataclass

#: A touch slower than the robot's conversational rate (the multiplier stacks
#: on the operator's speed slider). These lines are the Hub's own words to a
#: first-time listener, often a room of them; measured delivery reads as
#: considered, and none of the usual latency pressure applies -- the whole
#: answer is already written.
PACE = 1.1

#: How many words an utterance may run past the matched cue and still be "the
#: same question". Exists for the same reason as the runner's fuzzy-trigger
#: allowance. Was 5, raised to 8 after a live run: the natural way people
#: actually ask these ("Indeed it is -- any final advice for our new
#: students, Reachy?") runs 11-12 words and was being thrown to the model,
#: while the multi-part questions this guard protects run past 20.
_MAX_EXTRA_WORDS = 8

#: What the recogniser reliably mishears these questions as, mapped back --
#: every entry here was observed in a live transcript before being added.
#: Applied to the word stream (lowercased, no punctuation), whole words only.
#: "AI XR Hub" came back as "AIXR home"; "do in the Hub" as "do in a home".
_HEARD_AS = (
    ("aixr", "ai xr"),
    ("a i x r", "ai xr"),
    ("xr home", "xr hub"),
    ("xr hall", "xr hub"),
    ("xr hump", "xr hub"),
    ("xr hob", "xr hub"),
    ("do in a home", "do in the hub"),
    ("do in the home", "do in the hub"),
    ("do in a hall", "do in the hub"),
    ("do in the hall", "do in the hub"),
    ("do in a hub", "do in the hub"),
)


def normalise(words: str) -> str:
    """Map known mishearings back to what was asked. Word-stream in and out.

    The stream stays SPACE-PADDED on both ends -- contains_phrase matches
    " phrase " against it, so stripping the padding silently breaks every
    match that touches the first or last word. That was shipped for about an
    hour once; the whole-suite run caught it.
    """
    padded = f" {words.strip()} "
    for heard, meant in _HEARD_AS:
        padded = padded.replace(f" {heard} ", f" {meant} ")
    return padded


@dataclass(frozen=True)
class SetPiece:
    """One question the Hub has decided how to answer."""

    name: str
    #: The Dean's cue lines from the event script, matched by the Dean's
    #: Office performance mode.
    stage_cues: tuple[str, ...]
    #: Everyday phrasings of the same question, matched in conversation.
    ask_cues: tuple[str, ...]
    lines: tuple[tuple[str, str], ...]


BLOCKS: tuple[SetPiece, ...] = (
    SetPiece(
        name="introduction",
        stage_cues=("introduce yourself", "would you like to introduce"),
        # "tell me about yourself" belongs to the About demo (the technical
        # explainer) and is deliberately absent here.
        ask_cues=("who are you", "what are you", "introduce yourself"),
        lines=(
            ("Of course!", "happy"),
            ("Hello new friends, and welcome to Trinity Business School.", "happy"),
            ("I'm Reachy, the small robot who lives at the Trinity AI XR Hub.", "happy"),
        ),
    ),
    SetPiece(
        name="what the hub is",
        stage_cues=("what exactly is", "what is the ai xr hub", "what is the xr hub",
                    "what is the hub"),
        # "what is this place" is deliberately absent: it contains the Look
        # demo's "what is this" trigger verbatim, and the trigger table runs
        # before this module ever sees the words (found by the cue audit).
        # The "exactly is" and bare "the ai xr" forms are from live Whisper
        # transcripts that clipped the first word or the last.
        ask_cues=("what is the ai xr hub", "what is the xr hub", "what is the hub",
                  "whats the ai xr hub", "whats the hub", "what is the ai hub",
                  "tell me about the hub", "what is the ai xr", "whats the ai xr",
                  "exactly is the ai xr hub", "exactly is the hub",
                  "explain the ai xr hub", "explain the hub"),
        lines=(
            ("Well, you might expect a robot to talk about technology, but "
             "that's not really what we're about.", "curious"),
            ("Here, technology isn't the point. People are.", "neutral"),
            ("The Trinity AI XR Hub is a space here in the Business School "
             "where we explore how people and technology can work better "
             "together.", "neutral"),
            ("You'll get to experience artificial intelligence, virtual "
             "reality and, of course... embodied AI like me.", "happy"),
            ("But ultimately, the Hub is about you.", "happy"),
        ),
    ),
    SetPiece(
        name="why people matter",
        stage_cues=("students so important", "why are the students",
                    "students important"),
        ask_cues=("why are students important", "why are the students so important",
                  "why do human skills matter", "why are human skills important",
                  "why do people matter"),
        lines=(
            ("Because as AI gets better, human skills matter more, not "
             "less.", "neutral"),
            ("Skills like judgement, communication, presence and persuasion "
             "are increasingly important in an AI-enabled world.", "neutral"),
            ("The Hub gives you a space to develop those skills while also "
             "exploring how emerging technologies are changing the way we "
             "learn, work and collaborate.", "happy"),
        ),
    ),
    SetPiece(
        name="what you do here",
        stage_cues=("actually do", "do in the hub", "what will our msc"),
        # "what do you do in the hub" is deliberately absent: it fuzzy-matches
        # the Look demo's "what do you see" in the trigger table, which runs
        # first (found by the cue audit). "students exactly do" is Whisper's
        # rendering of "students actually do", live.
        ask_cues=("what do students do", "what will students do",
                  "what can i do in the hub", "what can we do in the hub",
                  "what happens in the hub", "what do people do in the hub",
                  "students do in the hub", "students actually do",
                  "students exactly do", "masters students do"),
        lines=(
            ("You'll get hands-on experience with immersive technology.", "happy"),
            ("You'll put on a VR headset and practise real-world situations, "
             "from presentations and interviews to challenging "
             "conversations.", "neutral"),
            ("And the best part?", "curious"),
            ("You can practise, get feedback, reflect, and try again.", "happy"),
            ("It's a safe space to experiment, make mistakes and "
             "improve.", "neutral"),
            ("But the Hub isn't only about developing your human "
             "skills.", "neutral"),
            ("It's also a place to experience emerging technologies and "
             "explore what happens when humans and technology work "
             "together.", "curious"),
            ("I suppose that's where I come in.", "happy"),
        ),
    ),
    SetPiece(
        name="advice",
        stage_cues=("final advice", "any advice", "advice for our new students"),
        ask_cues=("any advice", "final advice", "advice for new students",
                  "advice for students", "advice for our new students",
                  "what is your advice"),
        lines=(
            ("Yes.", "neutral"),
            ("Be curious.", "happy"),
            ("Try something unfamiliar.", "curious"),
            ("Don't be afraid to make mistakes.", "neutral"),
            ("And remember: the future isn't just about what AI can "
             "do.", "neutral"),
            ("It's about what you and AI can do together.", "happy"),
            ("So come and visit us, try the technology, and don't forget to "
             "say hello when you see me.", "happy"),
            ("Welcome to Trinity Business School, and welcome to the Trinity "
             "AI XR Hub.", "happy"),
            ("Where Immersive Intelligence collaborates with you and brings "
             "Positive Impact.", "happy"),
        ),
    ),
)


def match(text: str):
    """The SetPiece this utterance is asking for, or None for the model.

    None is the common and correct answer: only a short, direct phrasing of
    one of the scripted questions is taken over. See the module docstring for
    why the length guard is not negotiable.
    """
    from demokit.runner import _word_stream, contains_phrase

    words = normalise(_word_stream(text))
    total = len(words.split())
    for piece in BLOCKS:
        for cue in piece.ask_cues:
            if (contains_phrase(words, cue)
                    and total <= len(cue.split()) + _MAX_EXTRA_WORDS):
                return piece
    return None


def perform(ctx, text: str) -> bool:
    """Speak the scripted answer if this utterance asks a scripted question.

    True when a block was delivered. Interruptible on purpose -- this is
    conversation, not the stage; the Dean's Office mode does its own
    uninterruptible delivery. The turn is recorded in conversation memory so
    a follow-up ("what did you mean by that?") reaches the model with the
    script it is following up on.
    """
    piece = match(text)
    if piece is None:
        return False
    ctx.status(f"Answered from the Hub script: {piece.name}.")
    for line, emotion in piece.lines:
        ctx.say(line, emotion, pace=PACE)
    try:
        from brain import memory

        memory.remember_turn(ctx.person_id(), text,
                             " ".join(line for line, _e in piece.lines))
    except Exception:
        # Memory is a nicety here; the answer already happened.
        pass
    return True
