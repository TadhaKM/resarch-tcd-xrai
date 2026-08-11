"""Storyteller: one short performed story, on request and on a slow timer.

Migrated from the "story" mode that used to live in body/voice_loop.py. Almost
everything about HOW a story is told already lives elsewhere and stays there:
the persona is _STORYTELLER_SYSTEM_PROMPT in brain/prompts.py, the raised token
budget is _STORY_MAX_TOKENS in brain/interface.py, and the performed delivery is
what DemoContext.reply switches on when it sees style="story". Passing that one
keyword is the whole migration. This file decides only WHEN a story happens and
WHAT it is about.

What the visitor said stays in the message and everything else goes through
ctx.reply(system=...). The two are not interchangeable: the message is recorded
as a turn this person spoke, and person 0's history is read back by the ordinary
conversational path, so a brief written into the message is an instruction the
robot goes on obeying -- and appears to have been told -- for the rest of the
visit. The system text lasts exactly one turn, which is how long a brief for one
story should last.

Two decisions carried over from the old mode, both worth keeping:

Stories are requested through person 0 even when the visitor is recognised.
build_messages puts that person's history into the story prompt so it does not
retell the one it just told, and "don't repeat yourself" is a property of the
room, not of whoever the camera currently has a name for. Reading it off
ctx.person_id() would let a recognised visitor hear the story the stranger
before them just heard.

Stories are never grounded in brain/hub.py. The storyteller prompt replaces the
standing system prompt outright, so a story request reaches a model holding no
Hub facts whatsoever -- ask it for a story about the Hub and it invents staff,
projects and funding, and performs them, in a room that routinely contains the
actual staff. That request is intercepted rather than answered (see _tell).
"""

import re
import time

from brain import hub
from demokit import Demo, DemoContext, IdleResult


#: Phrases that switch to this demo from anywhere. The runner matches these as
#: bare substrings of the whole transcript, whatever demo is running, so each
#: one has to be unmistakably a request. Not a bare "story", which
#: business-school visitors say constantly and mean nothing by ("the story of
#: our company", "the story so far"). "another story" and "one more story" are
#: gone from here for the same reason -- "but that's another story" and "we
#: need one more story for the newsletter" are ordinary speech, and as triggers
#: they drag a room out of the middle of another demo to hear a fable. Every
#: phrase left is somebody addressing the robot. Encores do not need to be
#: here: they arrive while this demo is already selected, where _ALSO_ASKS
#: catches them.
_TRIGGERS = (
    "tell me a story",
    "tell us a story",
    "tell me another story",
    "tell us another story",
)

#: Also accepted, but only once this demo is already selected and the context
#: makes them unambiguous. These are not triggers: "one more" would switch the
#: robot into storytelling on somebody asking a colleague for one more minute.
_ALSO_ASKS = (
    "tell a story",
    "story about",
    "another story",
    "another one",
    "one more",
    "story please",
)

_ASK_PHRASES = _TRIGGERS + _ALSO_ASKS

#: The visitor's turn, as the transcript will record it. Deliberately the plain
#: request and nothing more: stream_reply remembers every message as something
#: this person said, and person 0's history is the same history the ordinary
#: conversational path reads back, so a brief pasted in here would be replayed
#: as a visitor turn on every later question for the rest of the session -- the
#: robot answering a question about the Hub in eighty words because of an
#: instruction nobody in the room ever spoke. The shaping goes in _STORY_BRIEF.
_ANY_STORY = "Tell me a story."

#: The same, with the visitor's subject where the visitor put it.
_ABOUT_STORY = "Tell me a story about {topic}."

#: Layered onto the storyteller prompt for this one turn via ctx.reply(system=),
#: so it shapes the story without becoming part of the conversation.
#:
#: The word budget is stated even though brain/interface.py has already
#: established that the model overruns it and that _STORY_MAX_TOKENS is what
#: actually bounds the length. It stays because it shapes the story rather than
#: the string: asked for eighty words the model writes one small incident,
#: which is a thing that can end, instead of a saga the token cap has to cut
#: off. The last clause is the reason build_messages puts history into the
#: story prompt at all -- without it the same crowd hears the same fox twice.
_STORY_BRIEF = (
    "Tell a brand new story, sixty to eighty words, suitable for all ages, "
    "on a different subject from any story earlier in this conversation."
)

#: Topics that must not reach the storyteller prompt, because a story about
#: them is a confident invention about real people and real money. The names
#: come from hub.PEOPLE rather than being typed here, so this guard cannot
#: drift away from the file that is true. Matched as whole words -- see
#: _mentions, and see what happens when they are not.
_UNGROUNDED = (
    "hub",
    "trinity",
    "business school",
    "this place",
    "this building",
) + tuple(person["name"].lower().removeprefix("professor ") for person in hub.PEOPLE)

#: Longest thing accepted as a topic. Whisper happily trails half of somebody
#: else's conversation onto the end of a transcript; past a phrase's worth of
#: words it is noise, not a subject.
_MAX_TOPIC_CHARS = 60

#: How long between volunteered stories, unchanged from the old mode. A
#: compromise: long enough that a group can hold a conversation in between
#: without being talked over, short enough that someone who walks in halfway
#: through a visit still hears one.
_STORY_INTERVAL_S = 240.0


class Storyteller(Demo):
    label = "Storyteller"
    help = "Tells a short story on entering, on request, and every few minutes."
    order = 70
    #: A story told warmly lands; a story told precisely is a report.
    persona = "friendly"
    triggers = _TRIGGERS

    def on_enter(self, ctx: DemoContext) -> None:
        """Announce, and leave the telling to the first idle slice.

        The story cannot happen here for two reasons. It is half a minute of
        speech during which nothing reads the microphone, which is what on_enter
        is explicitly not for. And the runner calls on_enter part-way through
        switching demos on a trigger phrase, then hands the same utterance to
        on_utterance -- so a story told here would be followed immediately by a
        second one. The flag defers it; _tell clears the flag, so whichever path
        gets there first is the only one that speaks.
        """
        ctx.store["story_due"] = True
        ctx.say("Story time. Let me think of one.", "thinking")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        # Deliberately not gated on ctx.face_visible(): face detection is off on
        # the robot's own CPU, so gating would mean the demo never volunteers
        # anything on the hardware it ships on. A story told to an empty room is
        # the cheaper mistake, and the operator has a dashboard.
        if ctx.store.get("story_due") or self._interval_elapsed(ctx):
            self._tell(ctx)
            # A short window, never zero: listen_for=0.0 means the core returns
            # without opening the microphone at all, and the seconds right after
            # a story ends are exactly when somebody says "hey Reachy, another
            # one". Shorter than the ordinary pass below because the operator's
            # dashboard is only read between slices, and a long telling has
            # already made them wait once.
            return IdleResult(listen_for=1.0)
        return IdleResult(listen_for=2.0)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        lowered = text.lower()
        if not _mentions(lowered, _ASK_PHRASES):
            # Everything else is an ordinary question, and the storyteller has no
            # business answering it: its prompt has no facts in front of it and a
            # performed delivery. False sends it to the conversational path,
            # which is the one with grounding and memory.
            return False
        self._tell(ctx, _topic_in(lowered))
        return True

    def on_exit(self, ctx: DemoContext) -> None:
        """Nothing to stop. The only long thing this demo does is ctx.reply,
        which checks for the switch between sentences and unwinds itself. Kept
        as an explicit no-op so the next reader does not go hunting for the
        thread or timer that was never started.
        """

    # --- internals -------------------------------------------------------

    def _interval_elapsed(self, ctx: DemoContext) -> bool:
        last = float(ctx.store.get("last_story_at", 0.0))
        return time.monotonic() - last >= _STORY_INTERVAL_S

    def _tell(self, ctx: DemoContext, topic: str = "") -> None:
        """Ask for one story and perform it. The only thing that resets the clock."""
        # Stamped before the attempt rather than after, so an attempt that fails
        # or is cut short still counts as one. Without that, a backend refusing
        # every request is retried on the very next idle slice and the runner's
        # three-strikes rule sets the whole demo aside within seconds of an
        # audience noticing anything is wrong. One dead moment every four
        # minutes is the cheaper failure, and it leaves the demo switchable.
        ctx.store["story_due"] = False
        ctx.store["last_story_at"] = time.monotonic()

        if topic and _mentions(topic, _UNGROUNDED):
            # A real question arrived wearing a story request. Answer it straight
            # and then tell an invented story about something else, rather than
            # inventing one about the Hub to the people who run it.
            ctx.say_lines(
                [
                    "I'd only be making that one up, and the people in it might be "
                    "standing right here. Here's what I can tell you straight.",
                    hub.MISSION_SHORT,
                ],
                "neutral",
            )
            topic = ""

        ctx.status("Telling a story")
        request = _ABOUT_STORY.format(topic=topic) if topic else _ANY_STORY
        told = ctx.reply(request, person_id=0, style="story", system=_STORY_BRIEF)
        if not told:
            # stream_reply yields no text at all when every backend died before
            # the first sentence. Silence after "let me think of one" reads as a
            # crash to a room that is watching; this reads as a person.
            ctx.say("My mind's gone blank. Ask me again in a minute.", "surprised")


def _mentions(text: str, phrases: tuple[str, ...]) -> bool:
    """Whether any of `phrases` appears in `text` as whole words.

    This used to be a bare `in`, and a substring test against a list of real
    names is a trap: "na fu" comes off hub.PEOPLE as Professor Na Fu's name and
    is four letters, so "tell me a story about banana fudge" was heard as a
    question about a member of staff and refused -- the visitor got "I'd only
    be making that one up, and the people in it might be standing right here"
    instead of a story about fudge. "hub" did the same to "hubcap". Joining the
    tokens back with single spaces and testing space-padded phrases keeps the
    multi-word entries ("business school", "laura berry", and "na fu" inside
    "professor na fu") matching, while ordinary words that merely contain one
    no longer do.

    Same shape as demos/dance.py::_mentions, and kept as a local copy rather
    than imported from it on purpose: one demo importing another means a
    student's broken file takes a working demo down with it at discovery time,
    which is the one thing the registry's per-module try is there to prevent.
    """
    padded = " " + " ".join(re.findall(r"[a-z']+", text.lower())) + " "
    return any(f" {phrase} " in padded for phrase in phrases)


def _topic_in(lowered: str) -> str:
    """What the visitor wanted the story to be about, or "" for the model's choice.

    Only "about X" is read. It is the one phrasing that survives transcription
    intact -- "a story about a lighthouse" always comes through, while "a story
    where the lighthouse keeper..." arrives in a dozen shapes -- and a topic
    guessed wrong is worse than no topic, because the visitor then hears the
    robot confidently telling somebody else's story.
    """
    _before, marker, rest = lowered.partition("about")
    if not marker:
        return ""
    topic = rest.strip(" ,.!?")
    return topic if 0 < len(topic) <= _MAX_TOPIC_CHARS else ""
