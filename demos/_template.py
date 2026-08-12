"""Copy me to demos/your_demo.py and edit. Underscore-prefixed files are skipped.

If all you want is for the robot to say something of your own -- a welcome for
a visiting group, say -- you do not need this file. Build it from the
dashboard's "Build a feature" panel instead: no code, and it appears without a
restart. This route is for when four fixed step kinds are not enough.

The shortest useful demo is about six lines: a class, a label, and one hook.
Everything below is optional -- delete what you do not need. Once the file
exists, the demo appears in the dashboard the next time the robot starts.

The one thing to get right: hooks return quickly. The robot polls the selected
demo in short slices so an operator can switch away mid-visit, and nothing is
listening to the microphone while your code runs. If you want the robot to
listen, say so with IdleResult(listen_for=...) and let the core do it -- it can
bound that; your demo cannot.
"""

from demokit import Demo, DemoContext, IdleResult


class Template(Demo):
    # `id` defaults to the filename, which is what the dashboard and API use.
    label = "Template"
    help = "What an operator sees under the button. One line."
    #: Lower sorts earlier on the dashboard.
    order = 900
    #: Say any of these to any demo and the robot switches to this one.
    triggers = ()
    #: ("faces",) marks the demo unavailable when face detection is off, so it
    #: greys out with a reason instead of silently doing nothing.
    requires = ()

    def on_enter(self, ctx: DemoContext) -> None:
        """Selected. Keep this short -- one spoken line at most.

        Longer scripts belong in on_idle, a line per slice, so the operator can
        still switch away part-way through.
        """
        ctx.say("Template demo.", "happy")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        """Called repeatedly while selected and nothing else is happening.

        Do one small thing, then say how long to listen for the wake word.
        """
        return IdleResult(listen_for=2.0)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        """Someone said something after the wake word.

        Return True if you handled it. Return False to let the robot answer
        conversationally, which is usually what you want for anything you did
        not specifically plan for.
        """
        if "template" in text.lower():
            ctx.say("That's me.", "happy")
            return True
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        """Switched away. Stop anything still running."""


# Things worth knowing, none of which need code here:
#
#   ctx.say(text, emotion)         speak one line
#   ctx.say_lines([...])           speak several, stopping if switched away
#   ctx.ask(question)              speak, then listen; "" if nothing came back
#   ctx.listen()                   listen; "" if nothing came back
#   ctx.reply(message)             ask the language model and speak the answer
#   ctx.sleep(seconds)             pause, still noticing a mode switch
#   ctx.face_visible()             is someone in view right now
#   ctx.person_id()                recognised person, or 0 for a stranger
#   ctx.status(text)               note something on the dashboard, unspoken
#   ctx.store                      dict that survives across hooks
#
# Emotions: happy, sad, curious, thinking, surprised, neutral.
