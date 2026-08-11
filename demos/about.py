"""What the robot is, for people who will never read a line of its code.

Two jobs that pull against each other. The first is the prepared explanation in
brain/hub.py, spoken one sentence per idle slice: nothing consumes the
microphone while the robot talks, so said as a single utterance it is half a
minute of deafness -- including to the "hang on, what's a wake word?" that the
explanation itself provokes, which is the best question anyone asks all day.

The second is answering that interruption truthfully. The four follow-ups
visitors actually ask -- how do you hear me, can you see me, what model is
that, do you need the internet -- all have answers that depend on how the robot
was started this morning, so every one of them is read from what is really
running (brain.llm.streaming_backends, config.MODELS, the live face tracker)
rather than from anything written down here. A cloud model gets configured for
an industry day, face detection is off whenever MediaPipe cannot run, and a
robot cheerfully claiming "no internet at all" while talking to a data centre,
in a room of people who work on exactly this, is the failure this file exists
to prevent.

The honesty has a limit worth stating out loud rather than papering over.
Nothing in brain/llm.py records which backend served the last reply, and
interface.stream_reply fails over from the cloud model to the local one per
request, silently, whenever the network misbehaves. So the robot reports what
it is *configured* to use and says plainly that it may already have swapped.
An unqualified "I'm running in the cloud" is a lie on any bad-wifi day, and
this room is full of the people most likely to notice.
"""

import re

from brain.hub import ABOUT_REACHY_SCRIPT
from brain.llm import streaming_backends
from brain.llm_backends import OllamaBackend
from config import MODELS
from demokit import Demo, DemoContext, IdleResult


#: hub.py writes the script as one string, for the ear. The robot needs it as
#: sentences. Split here rather than kept split there, so changing what is said
#: about the robot stays a one-file edit.
_SCRIPT_LINES: tuple[str, ...] = tuple(
    line.strip() for line in re.split(r"(?<=[.!?])\s+", ABOUT_REACHY_SCRIPT.strip()) if line.strip()
)

#: Substrings matched against a lowercased transcript. Several phrasings each,
#: because nobody asks the question the way it is written down, and the
#: recogniser's punctuation and hyphenation are not dependable.
_HEARING = ("how do you hear", "hear me", "microphone", "wake word", "how do you know when")
_SEEING = ("can you see", "do you see", "see me", "camera", "your eyes", "look at me")
_INTERNET = ("internet", "offline", "wifi", "wi-fi", "wi fi", "network", "cloud", "online")
_MODEL = ("model", "llm", "chatgpt", "your brain")


def _speakable(model_id: str) -> str:
    """'qwen2.5:1.5b-instruct-q4_K_M' -> 'qwen2.5'; a quantisation tag is unsayable."""
    return model_id.split(":")[0].replace("-", " ")


def _answering() -> tuple[bool, str, str | None]:
    """(all local, model configured to answer, model it falls back to or None).

    Read from llm.streaming_backends(), which is the exact ordered list
    interface.stream_reply tries for every reply -- so "is there anything to
    fall back to" is answered by the code that does the falling back, instead
    of being restated here and drifting away from it.

    What that list cannot say is which backend served the *last* reply: nothing
    in brain/llm.py records it, and the failover happens per request. Hence the
    third element rather than a bare "local or cloud" -- callers that name a
    cloud model are expected to name the fallback too and admit the swap can
    already have happened. Reporting the preference as fact is how the robot
    ends up announcing a data centre while every word comes off the laptop.
    """
    backends = streaming_backends()
    # Every backend in llm_backends.py is named after its config attribute
    # (ollama -> ollama_model, anthropic -> anthropic_model), so a provider
    # added there needs no edit here. Compared against OllamaBackend.name
    # rather than the literal "ollama": getting this backwards would have the
    # robot lie about the one thing it is being asked.
    names = [_speakable(getattr(MODELS, f"{b.name}_model", b.name)) for b in backends]
    local = backends[0].name == OllamaBackend.name
    return local, names[0], names[1] if len(names) > 1 else None


def _running_lines() -> tuple[str, ...]:
    """What is set up to answer today, in sentences short enough to speak."""
    local, model, fallback = _answering()
    if local:
        # Nothing else is configured, so this one is safe to state flatly.
        return (f"Right now that is exactly what is happening: {model} is answering you from the machine beside me.",)
    return (
        f"Today, though, I'm set up to do the answering with {model}, which sits out on the internet.",
        "Everything else is still local, and if the network fails I fall back to "
        f"{fallback} here mid-sentence and carry on.",
        "So I can tell you which model I was told to use. Which one just spoke, I honestly cannot promise.",
    )


def _explanation() -> tuple[str, ...]:
    """The whole thing said out loud: hub.py's script, then what is running today.

    Built as one sequence so there is one index advancing through it, and built
    per call so the live lines cannot go stale against the script they follow.
    """
    return _SCRIPT_LINES + _running_lines() + ("Ask me anything about how any of that works.",)


def _brief() -> tuple[str, ...]:
    """Three lines, for someone who asks again after the whole script has run."""
    local, model, fallback = _answering()
    if local:
        running = f"{model} is doing the thinking, on the machine beside me."
    else:
        running = f"I'm set up to think with {model} out on the internet, with {fallback} here as the fallback."
    # The headline is taken from hub.py rather than written again, so a reword
    # there carries; the slice rather than an index because an emptied script
    # should cost a line, not raise.
    return _SCRIPT_LINES[:1] + (
        running,
        "I went through the long version a moment ago. Ask me about any part of it and I'll go deeper.",
    )


class About(Demo):
    label = "About Reachy"
    help = "Explains the robot to non-engineers, and what is actually running inside it."
    order = 30
    #: Explaining itself to non-engineers, which is the friendly register's whole purpose.
    persona = "friendly"
    #: Triggers are bare substrings matched against every transcript, whatever
    #: demo is running, so each one here has to be unmistakably a request to be
    #: told about the robot. "what are you" was not: it is one of the commonest
    #: sentence openers in English, and "what are you working on here", "what
    #: are you good at" and "what are your research projects" all hijacked
    #: whatever was being discussed and started reading the spec sheet over it.
    #: Every phrase below is a complete question a visitor could only be asking
    #: about the machine in front of them, and none of them can prefix a longer
    #: word or turn into somebody else's subject. "tell me about you" is
    #: deliberately absent for the same reason as "what are you": it swallows
    #: "tell me about your projects" and "tell me about your research", which
    #: are Hub questions, not robot ones.
    #: "how do you work" is deliberately absent too, though it reads as a robot
    #: question: it is a fragment that takes a complement, so it fires on "how
    #: do you work with industry partners" and "how do you work with the
    #: students" -- both sentences someone says to a host at an open day while
    #: the robot happens to be listening. The complement-proof phrasings below
    #: cannot be continued into a question about anything else.
    triggers = (
        "tell me about yourself",
        "what kind of robot are you",
        "what sort of robot are you",
        "how are you built",
        "how does your brain work",
        "what is going on inside you",
    )

    def on_enter(self, ctx: DemoContext) -> None:
        # Cleared on entry, not on exit: the store is kept for the life of the
        # process and is never cleared between visitors, so a group arriving at
        # the dashboard button would otherwise get the tail end of the last
        # group's explanation. Cleared wholesale rather than key by key, so a
        # key added here later cannot outlive a visit by being forgotten.
        ctx.store.clear()
        ctx.say("Let me tell you what I actually am.", "curious")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        sequence = _explanation()
        index = ctx.store.get("line", 0)
        if index >= len(sequence):
            return IdleResult(listen_for=3.0)

        # Advanced before speaking, so every sentence is attempted exactly once.
        # The runner guards this hook and calls it again on failure; the other
        # order means a group hears the same sentence three times before the
        # demo is set aside, which reads far worse than a gap.
        ctx.store["line"] = index + 1
        last = index == len(sequence) - 1
        # Keyed to position, not to the words: hub.py owns the script and gains
        # sentences over time, so a per-sentence emotion table would quietly
        # attach itself to the wrong line the day someone edits it.
        if index == 0:
            emotion = "happy"
        elif last:
            emotion = "curious"
        else:
            emotion = "neutral"

        ctx.status(f"Explaining: {index + 1} of {len(sequence)}")
        ctx.say(sequence[index], emotion)
        if last:
            # Recorded only once the closing line has actually been spoken. Cut
            # off part way through, the group never heard the end, so a later
            # ask should still get the whole thing rather than the recap.
            ctx.store["delivered"] = True
            # One flourish, at the end. express_move pauses the procedural
            # motion loop for the length of the animation, so between every
            # sentence it reads as twitching rather than as expression.
            ctx.motion.express_move("happy")

        # A short window between sentences, so a visitor can interrupt with a
        # question instead of waiting out the whole explanation. Never zero:
        # the runner skips the microphone entirely for a slice that asks to
        # listen for nothing, and a run of those is a stretch where the wake
        # word goes unheard. The core's full three seconds once the last line
        # has just invited a question.
        return IdleResult(listen_for=3.0 if last else 1.0)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        # claims_utterances stays False: this demo never asks a question, so it
        # has no answer of its own to protect, and someone who says "dance"
        # halfway through the explanation should get the dance.
        lowered = text.lower()

        if any(phrase in lowered for phrase in self.triggers):
            if ctx.store.get("delivered"):
                # Asked again by a group that has just sat through the whole
                # thing -- the runner does not switch demos when the trigger
                # names the demo already running, so this is the second ask,
                # not the first. Restarting would read the spec sheet at people
                # who have heard it; three lines and an offer to go deeper is
                # the answer they were actually after.
                ctx.say_lines(_brief(), "happy")
                return True
            # First ask. The phrase that switched here arrives back as an
            # utterance, once -- the runner no longer offers it twice -- and
            # the prepared script answers it better than the model would
            # improvise, so start it from the top.
            ctx.store["line"] = 0
            return True

        if any(phrase in lowered for phrase in _HEARING):
            listening_chain = [
                line
                for line in _SCRIPT_LINES
                if "wake word" in line.lower() or "speech recognition" in line.lower()
            ]
            local, _model, _fallback = _answering()
            # The quoted sentences are looked up rather than restated, so a
            # reword in hub.py carries here; an empty list just means the answer
            # arrives as the one line that matters.
            where = ["Both of those run on this machine, so the sound of your voice never leaves the room."]
            if not local:
                where.append(
                    "Only the text of what you said goes any further, only once you have said my name, "
                    "and only when the cloud model is the one answering."
                )
            ctx.say_lines(listening_chain + where, "curious")
            return True

        if any(phrase in lowered for phrase in _SEEING):
            tracker = ctx.tracker
            if tracker is None or not tracker.enabled:
                # Past ctx.face_visible() deliberately: it returns False both for
                # "switched off" and for "nobody in view", and which of the two
                # is true is the entire answer. Detection is off every time this
                # runs on the robot's own CPU, where MediaPipe crashes.
                ctx.say(
                    "Not at the moment. Face detection is switched off on this machine, "
                    "so I am working by ear alone.",
                    "sad",
                )
            elif ctx.face_visible():
                ctx.say(
                    "I can. There is a face in front of me right now, and my head follows it while we talk.",
                    "happy",
                )
            else:
                ctx.say(
                    "I can, but I cannot pick anyone out this second. "
                    "Stand in front of me and my head will follow you.",
                    "curious",
                )
            return True

        # Before the model question, so "are you a cloud model" gets the answer
        # about the network -- which names the model anyway -- rather than the
        # other way round.
        if any(phrase in lowered for phrase in _INTERNET):
            local, model, fallback = _answering()
            if local:
                ctx.say_lines(
                    [
                        "Not right now. Recognising your speech, deciding what to say, and saying it "
                        "are all running on the machine here.",
                        f"That is {model} doing the deciding. You could unplug the network and I would carry on.",
                    ],
                    "happy",
                )
            else:
                ctx.say_lines(
                    [
                        f"Usually, for the answers. I'm set up to use {model}, which is a cloud model, "
                        "so the text of what you said goes out and comes back.",
                        "Hearing you and speaking are still local, and if the network fails I fall back to "
                        f"{fallback} on the machine here.",
                        "That swap happens question by question, so on a bad wifi day I may already be "
                        "answering you from this room without either of us noticing.",
                    ],
                    "curious",
                )
            return True

        if any(phrase in lowered for phrase in _MODEL):
            local, model, fallback = _answering()
            if local:
                ctx.say_lines(
                    [
                        f"The model answering you is {model}, running on the machine beside me "
                        "rather than in a data centre.",
                        "It is small enough to fit on a laptop, which is the part worth noticing.",
                    ],
                    "happy",
                )
            else:
                ctx.say_lines(
                    [
                        f"I'm set up to answer with {model}, over the internet, because it holds a "
                        "conversation better than anything I can fit on this machine.",
                        f"The local one, {fallback}, is still here underneath, and takes over by itself "
                        "the moment the cloud one does not come back.",
                        "Which is why I say set up to, rather than am. The reply you just heard could "
                        "have come from either of them.",
                    ],
                    "neutral",
                )
            return True

        # Anything else is a real question about AI, robots or the Hub, and the
        # model with hub.py's grounding answers those better than a keyword table.
        return False
