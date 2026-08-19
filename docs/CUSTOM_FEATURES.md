# Building your own features

For Hub staff. No code, nothing to install, and the robot does not need
restarting — a feature you save is a button within a couple of seconds.

Open the dashboard and click **Build a feature**.

---

## What a feature is

A button, and a short list of steps the robot performs when you press it. The
usual one is a welcome for a particular group:

> **Says** "Welcome, all of you. It's good to have the Cork group here."
> **Waits** until someone is standing in front of it
> **Asks** "What are you all studying?" — and answers whatever they say
> **Dances**

There are five kinds of step and no others:

| Step | What it does |
|---|---|
| **Says** | Speaks a line, with an expression you choose |
| **Asks** | Asks a question and listens. Tick *answer back* and the AI replies to whatever they say |
| **Dances** | A short dance, about four seconds |
| **Waits for someone** | Holds until a face is in view, so a welcome doesn't play to an empty room |
| **Plays** | Hands the visit over to another demo &mdash; the tour, the quiz, the advisor |

You can reorder steps with the arrows and remove them with ✕.

**Plays** has to be the last step. Whatever it hands over to keeps the floor
&mdash; the tour does not finish and come back &mdash; so anything after it
would never run, and the editor refuses to save it. A feature cannot hand over
to itself either; that would restart it forever in front of a group.

## Letting the assistant write it

Type what you want in plain English — *"a welcome for twenty school students
visiting from Cork on Thursday"* — and it will draft the words and fill in the
steps.

**It drafts; you approve.** Everything it writes is editable before you save,
and nothing is saved until you press Save. If it doesn't know something about
the Hub it will leave a placeholder in square brackets like
`[name of the visiting professor]` — fill those in, because the robot would
otherwise read the brackets out loud. Saving with one still in it is refused.

The assistant is a shortcut, not the only way. If it is slow, confused, or the
internet is down, build the steps yourself — the editor works on its own.

## Starting a feature by voice

You can give a feature a phrase people can say out loud, as well as the button.
This is optional, and the rules on it are stricter than you might expect:

- **At least three words.** A one-word phrase fires by accident. The robot's
  built-in welcome once triggered on *"you're welcome"*, *"welcome back"* and
  *"welcome to Dublin"* — a visitor saying any of those dragged the whole group
  out of whatever they were being shown.
- **It cannot clash** with a phrase another demo already answers to, in either
  direction. You will be told which one if it does.
- **It cannot contain a stop phrase** like "goodbye", or the robot's own name.

Good: *"say hello to the Cork group"*. Bad: *"welcome"*.

## How long to make it

Around **half a minute of speech** — four or five sentences. That is not a
technical limit; it is where groups start looking at each other instead of at
the robot. You will get a warning past that, and it is refused past about
ninety seconds.

Everything the robot says is interruptible. Somebody can say "Hey Reachy" over
the middle of your script and it will stop and listen to them, which is the
behaviour you want in a room.

## Keeping the dashboard tidy

Once you have a few of these, the grid gets long. Press **Arrange** above the
buttons and you can group them into folders — drag them by the handle, or use
the *"Put … in"* dropdown that appears on each one, which is easier on a phone.
Folders hold the robot's built-in demos too, not just yours.

Grouping only changes how the buttons are **shown**. It never changes what a
spoken phrase does, and it never changes what the robot starts up in. Full
instructions are in `OPERATOR.md`.

## If something goes wrong

- **Wrong button pressed mid-visit** — click **Disable**. It comes off the
  dashboard and keeps the script, so you can put it back later.
- **You want it gone** — **Delete**.
- **A feature misbehaves** — the robot sets it aside on its own after it fails
  three times and carries on with everything else. Editing it puts it back.
- **The dashboard is unreachable** — a technical person can remove it with
  `sqlite3 data/memory.db "DELETE FROM custom_features WHERE id='...'"` and a
  restart.
- **Deleted by mistake** — the database is copied daily and seven days are
  kept in `data/backups/`.

## Worth knowing

Features are **not reviewed by anyone**. Whoever can open the dashboard can add
one, and the robot will say what it says to whoever is standing there. Put your
name in the *Your name* box so it is clear later who wrote what.

A feature can only do the four things above. It cannot be made to move around,
look things up, or run anything on the laptop — that is deliberate, and it is
why this is safe to hand to anyone.
