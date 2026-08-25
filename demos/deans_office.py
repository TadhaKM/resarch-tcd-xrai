"""The Dean's Office pitch: who Reachy is, what the Hub is, why to visit.

Written for a different room than the welcome. The welcome plays to a group
already standing in the Hub; this plays where students and visitors pass
through -- the Dean's office -- to people who have NOT been to the Hub and may
not know it exists. So it introduces the robot, says what the Hub is in one
breath, and spends its energy on the one thing a passer-by can act on: come
and visit, and practise the human side of business in VR while you are there.

THE SHAPE IS THE WELCOME'S, EARNED THE HARD WAY THERE
One short line per idle slice, never a zero listening window between lines, a
restart on request, and questions handed to the conversation model rather than
answered from a script. The welcome demo's file documents why each of those
rules exists; this file just obeys them.

WHAT THE SCRIPT MAY CLAIM
Only what hub.GROUNDING already states (Trinity Business School initiative,
responsible AI and XR, amplifying human skills) plus the soft-skills message
this mode was commissioned to carry. Nothing invented: no named equipment, no
schedules, no promises about availability -- "come visit" is the whole call to
action, and whoever is hosting fills in the rest.
"""

from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S

_BETWEEN_LINES_S = 1.0

#: One line per idle slice. Each line is short on purpose: at the robot's
#: speaking pace a word is roughly half a second, and this whole script should
#: land inside ~40 seconds -- the welcome demo's file records that beyond that
#: a standing audience starts looking at each other.
_SCRIPT = (
    ("Hello! I'm Reachy Mini, one of the robots from Trinity's AI XR Hub.", "happy"),
    ("The Hub is Trinity Business School's home for responsible AI and "
     "extended reality -- built to amplify what people are good at.", "neutral"),
    ("My favourite part: you can practise the human side of business there.", "curious"),
    ("Presenting, interviewing, negotiating -- in VR, with AI coaching you as "
     "you go.", "happy"),
    ("So come visit -- try a headset, and see what your soft skills can "
     "do.", "happy"),
    ("And if you see me there, say hello. I remember faces.", "happy"),
)

#: Spoken ways to hear it again. Checked through the runner's word stream so
#: "say it again" survives however the recogniser spells it.
_RESTART = ("say it again", "do it again", "one more time", "start again")


class DeansOffice(Demo):
    label = "Dean's Office"
    help = "Introduces Reachy and the Hub, and invites students to come train their soft skills in VR."
    order = 21
    triggers = (
        "deans office",
        "the deans office",
        "do the deans office",
        "deans office introduction",
        "introduce yourself to the dean",
    )

    def on_enter(self, ctx: DemoContext) -> None:
        ctx.store.clear()
        ctx.store["line"] = 0
        ctx.status("Dean's Office introduction.")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        line = ctx.store.get("line", 0)
        if line < len(_SCRIPT):
            text, emotion = _SCRIPT[line]
            ctx.store["line"] = line + 1
            ctx.say(text, emotion)
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        # Script done: sit and listen. Questions fall through to the
        # conversation model, which knows the Hub grounding and answers
        # adaptively -- a script that tried to answer questions is how a
        # three-part question ends up getting a speech.
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        from demokit.runner import _word_stream, contains_phrase

        words = _word_stream(text)
        if any(contains_phrase(words, phrase) for phrase in _RESTART):
            ctx.store["line"] = 0
            ctx.status("Dean's Office restarted on request.")
            return True
        # The sentence that selected this demo is handed straight back by the
        # runner; swallowed through the same word stream the runner matched it
        # with, or the conversation model answers it over the top of line one
        # -- the double-answer bug three demos have already had.
        if ctx.store.get("line", 0) <= 1 and any(
                contains_phrase(words, t) for t in self.triggers):
            return True
        # Anything else is a real question; the conversation model answers it.
        return False
