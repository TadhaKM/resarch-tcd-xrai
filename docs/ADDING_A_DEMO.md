# Adding a demo

Copy `demos/_template.py` to `demos/your_demo.py`, edit it, restart the robot.
That is the whole procedure. Your demo appears on the dashboard and can be
switched to live.

You should not need to edit any other file. If you find yourself doing that to
make something work, treat it as a bug in `demokit/` and say so — the framework
is supposed to absorb that, and the next person will hit the same wall.

```python
from demokit import Demo, DemoContext, IdleResult


class Weather(Demo):
    label = "Weather"
    help = "Explains why a robot with no internet cannot tell you the weather."

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        if "weather" in text.lower():
            ctx.say("I have no idea. I can't see outside and I have no internet.", "curious")
            return True
        return False
```

That is a complete, working demo.

## The two rules that matter

**Hooks return quickly.** The robot polls your demo in short slices so the
operator can switch away mid-visit while people are watching. Nothing is
listening to the microphone while your code runs, so a hook that takes thirty
seconds is thirty seconds during which the robot is deaf, unresponsive, and
filling its audio buffer with its own voice.

**Let the core do the listening.** If you want the robot to listen, return
`IdleResult(listen_for=2.0)` rather than calling into the audio stack yourself.
The core can bound that and interrupt it; your demo cannot. This is the single
most important thing the framework does for you — it is what makes it
impossible for a demo to leave the robot unswitchable in front of an audience.

## Hooks

| Hook | When | Notes |
|---|---|---|
| `on_enter(ctx)` | Your demo is selected | **One spoken line at most** |
| `on_idle(ctx)` | Repeatedly, while selected | Do one small thing, return `IdleResult` |
| `on_utterance(ctx, text)` | Someone spoke after the wake word | `True` if handled, `False` to fall through |
| `on_exit(ctx)` | Operator switched away | Stop anything still running |

All four are optional.

## Saying something long

Speak one line per `on_idle` call, keeping your place in `ctx.store`. Do not
speak a script in a loop, and do not speak it in `on_enter`.

```python
LINES = ["First line.", "Second line.", "Third line."]

def on_idle(self, ctx):
    i = ctx.store.get("line", 0)
    if i < len(LINES):
        ctx.say(LINES[i])
        ctx.store["line"] = i + 1
        return IdleResult(listen_for=0.5)
    return IdleResult(listen_for=2.0)
```

`demos/welcome.py` does exactly this with the Hub's welcome script, which runs
about thirty seconds — far too long to say in one go.

## What `ctx` gives you

```
ctx.say(text, emotion)              speak one line
ctx.say_lines(lines, emotion)       several, stopping if switched away
ctx.ask(question)   -> str          speak, then listen ("" if nothing)
ctx.listen()        -> str          listen ("" if nothing)
ctx.reply(message)  -> str          ask the language model, speak the answer
ctx.sleep(seconds)                  pause, still noticing a mode switch
ctx.face_visible()  -> bool         is anyone in view
ctx.person_id()     -> int          recognised person, 0 for a stranger
ctx.status(text)                    dashboard note, not spoken
ctx.mode_changed()  -> bool         has the operator switched away
ctx.store           -> dict         survives across your hooks
ctx.motion, ctx.audio, ctx.tracker  escape hatches; prefer the above
```

Emotions, and there are only these: `happy` `sad` `curious` `thinking`
`surprised` `neutral`.

**Never call `time.sleep`.** Use `ctx.sleep`, which notices a mode change and
unwinds instead of holding the robot.

**Never touch audio from another thread.** One thread owns the microphone and
speaker; two produce noise. `ctx` raises a `RuntimeError` if you try, rather
than letting you find out in front of visitors.

## Class attributes

```python
label = "AI Conversation"    # the dashboard button
help = "One line for the operator."
order = 40                   # lower sorts earlier
triggers = ("tell me a story",)   # say this to any demo and it switches here
requires = ("faces",)        # greyed out when face detection is off
claims_utterances = True     # see below
```

`id` defaults to the filename and is what the API uses. Set it explicitly only
to keep an id stable while renaming the file.

**Triggers are substrings**, matched longest-first across every demo. Keep them
distinctive: a bare `"dance"` matches inside "dance studios in Dublin" and
hijacks somebody else's conversation. That is a real bug that happened, which
is why the dance demo now uses `"let's dance"` and `"can you dance"`.

**`claims_utterances = True`** puts your demo ahead of every other demo's
triggers. Use it when you are running a question-and-answer sequence, so a
visitor's *answer* is not mistaken for someone else's trigger phrase.
`demos/brainstorm.py` needs this; most demos do not.

## Facts about the Hub

Import them from `brain/hub.py`. Never restate them in your file.

There is exactly one source of truth for things this robot says out loud about
the Hub, its staff, its funding and its partners, because those claims get made
in front of the people who run it. `brain/hub.py` also encodes what the robot
does *not* know — asked which headsets are in the room, it says so and points at
a human, rather than inventing a plausible model number in front of someone who
can see the shelf.

If you need the model to answer questions using those facts, pass
`hub.GROUNDING` into the prompt; see `demos/conversation.py`.

## When your demo breaks

It will not take the robot down. Every hook runs inside a guard: the exception
is logged with a traceback, a note appears on the dashboard, and after three
*consecutive* failures the demo is set aside and the robot falls back to
conversation. The dashboard greys the button and shows why.

Fix it, then press the button again to re-enable it — or restart the robot.

The count is consecutive, not cumulative, so a demo that fails once an hour is
not quietly retired after three hours of otherwise faultless operation.

## Checking your work without the robot

```
python tools/selftest.py
```

Section 2 loads every demo and checks the contract: that it imports, that its id
is unique, that no two demos claim the same trigger phrase, that it has a label
and help text. A demo that fails to import is invisible at runtime — the
registry logs it and carries on, which is right during a live session and no use
at all for finding out beforehand.

## Things the framework deliberately does not do

**It does not stop your demo being rude, wrong, or off-brand.** It contains
crashes, not taste. If demos are going in before an industry visit, someone
should listen to them first.

**It does not sandbox you.** `ctx.audio` and `ctx.motion` are right there and
occasionally you will need them — a framework that forbids everything gets
forked. But everything reachable through `ctx` is the supported surface: it
checks the calling thread, notices mode changes, and keeps the robot answerable
to its operator. Reach past it only when you must, and say why in a comment.
