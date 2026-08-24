# Reachy — the guide for showing it to people

For whoever is standing next to the robot with visitors. No technical
knowledge needed, and nothing here requires touching code.

If something goes wrong, the fix is almost always: close the black window, and
double-click **Start Reachy** on the desktop.

---

## Starting it

1. Turn the laptop on and **sign in**.
2. Power on the robot. Give it about a minute.
3. **That's it.** Reachy starts by itself and finds the robot on the network.

**You know it's ready when the robot wiggles** — it looks left, right, and
nods. That takes a minute or two from sign-in, mostly loading the speech
software.

If it didn't start by itself, or you closed it, double-click **Start Reachy**
on the desktop. You never have to type an address; it goes and finds the robot,
and tells you where it found it.

## Talking to it

Say **"Hey Reachy"**, wait for its antennas to perk up, then ask your question.

**One "Hey Reachy" per question** — unless you turn on **🎤 Open mic**, below.
This is the thing visitors get wrong most often: they say it once and then keep
talking. Normally it needs the phrase each time, which is deliberate — in a room
full of people it's how the robot knows someone is talking to *it* and not to
each other.

**It answers to its name, not just to one magic phrase.** All of these work:

> "Hey Reachy" · "Hi Reachy" · "Hello Reachy" · "OK Reachy" · **just "Reachy"**
> · "Hello there Reachy" · "Attention Reachy"

And it answers to the name as people actually say it. Plenty of visitors, Irish
and Scottish accents especially, land somewhere nearer *Ricky* or *Richie* —
those wake it too, on purpose. **Nobody should have to say the name carefully
to be heard.**

**It hears you from across the room.** You don't need to lean in or raise your
voice.

The cost of all that: if someone in the room is actually called **Ricky**,
**Richie** or **Rachel**, the robot will occasionally think it was called. It
listens for a moment, hears nothing meant for it, and goes back to waiting — so
it's untidy rather than a problem. Tell visitors it's a robot that answers to
its name and they'll find it funny rather than broken.

**To interrupt it**, say "Hey Reachy" while it's still talking. It finishes the
sentence it's on, then stops and listens. Useful when it's mid-story and
someone has a question.

**To stop it**, say **"go to sleep"**, **"goodbye"**, or **"turn off"**. It
stays listening for the wake phrase, so any "Hey Reachy" brings it back.

---

## The screen

Open the browser page (there's a **Reachy Dashboard** icon on the desktop). It
shows everything the robot hears and says, and it's how you switch between
demonstrations. You can open it on your phone too — see the bottom of this
page.

Under the title is one wide bar saying what the robot is doing right now, in
words: **Starting up**, **Ready**, **Listening**, **Speaking**, **Asleep**, or
**Robot offline**. It changes colour with the state, so you can read it from
across a room without putting your glasses on. Asleep is normal, not a fault —
say "Hey Reachy" and it wakes.

Below that is the **Mode** grid. The demonstration that is running is the one
solid coloured block on the page; everything else stays plain. That is the
whole design: one glance, one colour, one answer.

The 🔊 volume slider stays on the surface, because "they can't hear it at the
back" is the one thing you actually change mid-visit. Everything else is behind
**⚙ Settings** (on a laptop it's just open all the time):

| What you see | What it means |
|---|---|
| 🧠 with a model name | Which AI is answering, and whether it's on the laptop or online |
| 🌐 **Web search** | Off by default. Tick it to let Reachy look things up — see below |
| 🎤 **Open mic** | Off by default. Tick it and follow-up questions don't need "Hey Reachy" — see below |
| 🎭 dropdown | The robot's manner **everywhere** — its speaking voice, its wording, how it holds its head. "Default" is its own |
| 🔊 dropdown | Which voice it speaks in — four to choose from |
| 🌗 **Appearance** | Light, dark, or Auto to follow the phone |
| 🎨 **Colour** | Five colour themes — tap a dot. See below |
| **Clear answers** | Makes it forget answers it has memorised. Only needed if it keeps repeating something wrong |

### The new demos

**Which masters fits me?** — say *"which masters"*. Three questions, then it
names **one or two** programmes and why each, in three short sentences. It
knows all thirteen taught masters in real detail (modules, accreditations, who
each one is actually open to) and it will **refuse** fees, deadlines, entry
requirements and whether somebody would get in — those change yearly and
belong to you, not to a robot. It finishes by sending them to you for exactly
those things.

**Quiz the group** — say *"quiz us"*. Four questions about the Hub and AI,
shouted answers accepted loosely, keeps score. Built for school groups, who
want a turn rather than a talk. Shuffled each time, so the second group of the
day gets a different order.

**Greet in another language** — say *"say hello in Spanish"* (or French,
German, Italian, Irish). Greeting only: it says hello, one true sentence, then
hands back to English, because it cannot *understand* those languages and must
never appear to. Drop a native piper voice into `models/tts/` and it uses it.

**Research session** — only appears when Research mode is switched on. See
below.

### Scripting a whole visit

A feature can now finish by handing over to another demo — a **Plays** step.
So one button can welcome the group, then hand to the tour, or to the quiz.
It has to be the last step (whatever it hands to keeps the floor), and it
cannot hand to itself.

### After a visit: how it went

Open **How the visit went**. Turns taken, which demos ran, and the questions
people actually asked, per day. It is counted as it happens and it is
**aggregate only** — no names, and no link between a question and a person.

### Locking the dashboard

By default the dashboard is **open to anyone on the same network**. It binds to
all interfaces so you can use it from a phone, which also means anyone on
TCD_IoT who finds the address can make the robot talk, delete research data, or
download a participant transcript.

To lock it, put a line in `.env`:

```
DASHBOARD_PASSCODE=whatever-you-choose
```

and restart. Everyone then types that once per browser and stays unlocked for
twelve hours. **No passcode set means no lock**, so nothing changes for you
unless you opt in.

Two honest limits. It is one shared passcode, not accounts — it tells you
nobody wandered in, not who did what. And the dashboard is plain HTTP, so
anyone able to watch the network traffic can read the passcode as it goes past.
It is a lock on a door, not a safe.

**The robot keeps working either way.** Locking the screen does not stop it
talking to visitors; it only stops people driving it from a browser. If you
forget the passcode mid-visit, the robot carries on and you lose the dashboard,
not the demo.

### Research mode

For Professor Berry's HRI work. Put your name in **Armed by**, tick **Recording
armed**, then run the **Research session** demo. No names, no faces, no person
ids are stored — just the condition, the persona, what was said, and the
timing.

**Switching it on is the consent.** The robot does not read a notice or ask
anybody out loud; it starts recording as soon as the session runs. So take
consent from your participants *before* you switch it on, the way an approved
protocol normally does it — on paper, before they sit down.

Because the robot no longer collects consent itself, the **Armed by** name is
the only record of somebody accountable for the data, so it is required and it
is written against every turn. It is an audit note, not a login: the dashboard
has no accounts and anybody on the network can type any name.

Anybody in the room can still stop it by saying so — *"stop recording"*,
*"delete my data"*, *"I changed my mind"*, or *"I do not consent"*. That
**deletes the session outright**, it does not merely mute it, and the sentence
that stopped it is not recorded either.

**It is not ethics approval.** Trinity requires approval before collecting data
from participants. This only makes the robot able to do it properly once you
have that. It is also off after every restart, on purpose — and since the robot
restarts whenever the wifi drops, check it is still armed if you have been
running a long session.

#### Getting the recordings out

**They are transcripts, not audio.** Nothing the robot hears is ever saved as
sound: speech is turned into text and the audio is discarded on the spot. There
is no play button and there is no file to recover, which is the point — a
transcript with no name and no voice attached cannot be traced back to the
person who said it, even by whoever holds the database.

The **Recordings** list at the bottom of the Research mode panel shows every
session: when it started, the condition, who armed it, how many turns, and the
average time to first word. The one being recorded into right now is marked
*live*.

- **View** shows the transcript in the page, so you can check a session is what
  you think it is before exporting it.
- **CSV** downloads it for analysis — one row per turn, with the persona, both
  timings and which model answered.
- **Text** downloads the same thing laid out to read.
- **Delete** removes that session outright. Use this if a participant asks
  afterwards; it is the same deletion the spoken withdrawal does.

Downloads are named `reachy-session-<id>.csv`, so a folder of them stays
identifiable.

**Two timings, and they mean different things.** `first_word_s` is how long the
participant waited before hearing anything — that is responsiveness.
`latency_s` is the whole turn including the robot speaking the answer, so it
grows with how long the reply was. For a study of responsiveness, use the
first; the second will make a talkative robot look like a slow one.

`backend` says whether Anthropic or the local model answered. They differ
enough in speed that a wifi drop mid-session changes your condition without
anybody choosing to, which is why it is recorded.

### What it could not answer

At the bottom of **How the visit went** is a list headed *Questions it could
not answer*. These are real questions from real visitors where the robot said
it did not know — the exact list worth teaching it, generated by the people who
came rather than guessed at in advance.

A question near the top of that list, asked repeatedly, is either something to
add to what the robot knows or something worth having a person ready to answer.

It is spotted from the robot's own words, since nothing marks a refusal
internally: replies that say "I don't have that", "you'd want to check the
Trinity website" or "ask whoever is hosting" are counted. Deliberately
cautious, so it under-reports rather than filling the list with questions that
were in fact answered.

Aggregate only, like the rest of that panel — the question, never who asked it.

### Colour themes

Under ⚙ Settings there is a row of coloured dots. Tap one and the whole page
changes. Each works in light and dark, so **Appearance** and **Colour** are
separate choices you can mix.

| | |
|---|---|
| **Trinity** | The Hub's blue. The default |
| **Violet** | Quieter, violet-tinted greys |
| **Teal** | Slate and water |
| **Rose** | Warm greys with a magenta accent |
| **High contrast** | Pure white or true black, heavy borders, the strongest colours. For a sunny atrium, or if the others are too soft to read |

Both choices are remembered **on that device only** — your phone and the
laptop can look different, and changing them affects nobody else. Everything
that is a fact about the *robot* — the folders, the switches, the personality —
is shared instead.

Whichever you pick, one rule holds: **a solid block of colour means the demo
that is running, and nothing else on the page is ever solid.** That is why
Reachy's own colours never move around — you can always find what's going by
looking for the one filled-in button.

### Tidying the buttons into folders

With twenty-odd buttons the grid gets long. Press **Arrange** above the grid
and you can group them — *School visits*, *Open day*, whatever suits you.

- **Drop one button on another to group them**, the way you make a folder on a
  phone. Drag by the handle, hold it over the middle of another button for a
  moment until a ring closes around it and it says *Release to group*, then let
  go. You get a new folder with both buttons in it, and it asks you to name it.
  Dragging *across* other buttons never groups anything — you have to stop on
  one and wait, which is what makes it safe to drag things about.
- **+ New folder** makes an empty one and asks for a name straight away.
- **Move a button**: drag it by the handle on its left. On a phone, a strip of
  folders appears along the bottom while you drag — just let go over one.
- **No dragging needed.** Every button grows a *"Put … in"* dropdown while you
  are arranging. One tap, pick the folder, done. Use this on a phone; it is
  faster and it cannot go wrong.
- **Keyboard**: Tab to a handle, press Enter to pick the button up, then the
  arrow keys move it (left and right move it in and out of folders), Enter
  drops it and Escape puts it back.
- Tap a folder's arrow to **collapse** it. The demonstration that is *running*
  always stays visible, even inside a collapsed folder.
- **Deleting a folder never deletes a button** — they move back to the grid.
- Press **Done** when you have finished. If you walk away it lets go by itself
  after a minute and a half, so nobody ever finds a dashboard whose buttons
  won't start anything.

While you are arranging, pressing a button does **not** start it. That is
deliberate: a thumb that slides slightly must never launch a demonstration in
front of fifteen students.

The arrangement lives on the robot, so it survives a restart and everyone
looking at the dashboard sees the same one. It is only how the buttons are
*shown* — it never changes what the robot starts up in, or what any spoken
phrase does.

> **Got in a mess?** There is no undo. Ask a technical person to run
> `curl -X DELETE http://localhost:8080/api/layout`, or just delete the folders
> one at a time — the buttons all come back.

---

## The demonstrations

Click one on the screen to switch to it, or say the phrase in the right-hand
column. **Switching is instant** — you don't have to wait for it to finish
talking.

Each one below has a suggested thing to say, because the difference between a
demo that lands and one that doesn't is usually knowing what to ask.

### The three to show first

**AI Conversation** — the normal one. General questions about AI, XR, the Hub,
and the robot.
> Try: *"Hey Reachy, who runs the AI XR Hub?"* — it knows the real answer.
> Then: *"Hey Reachy, what is extended reality?"*

**AI Personality** — the **🎭 dropdown changes the whole robot**, not just this
demo. Pick Professional, Friendly or Consultant and everything it says, in
every demonstration, changes with it: **a different speaking voice**, different
wording, and how it holds its head. "Default" is its own manner.

| Character | Sounds like |
|---|---|
| **Professional** | A level, formal male voice. Answers narrowly and says what it's unsure of |
| **Friendly** | The robot's usual warm voice, quicker and more animated. Everyday comparisons |
| **Consultant** | A lower male voice. Gives you the trade-off and asks what matters most |

The 🔊 voice dropdown still works, and it sticks: pick a voice by hand and the
robot keeps it as you move between demonstrations. Only changing the 🎭
character takes it back.

This demo is where the difference is easiest to *show*, because it re-answers
the **same question** as each character back to back — the only way a listener
can tell the personality apart from the question. Pick a character and it
re-answers straight away; saying *"switch personality"* still works too.
> Ask a question with no single right answer — *"Should a business student
> learn to code?"* — then change the dropdown twice. Factual questions make all
> three sound the same; arguable ones make the difference obvious.
>
> It won't explain itself first. That's deliberate: the point is that visitors
> *hear* the difference, and being told what to expect gives away the ending.

**Business Brainstorming** — say *"let's brainstorm"*. It talks an idea through
with whoever is standing there, writing each question from what they've just
said rather than working down a list, then gives three directions they could
take it and keeps talking afterwards. Genuinely good with business-school
groups because they participate rather than watch.
> Let them drive this one. Point it at a group, not one person — it opens by
> saying any of them can answer, and it holds the microphone open while it's
> waiting, so nobody has to say "hey Reachy" before answering a question it
> just asked them.
>
> It works out when to stop asking and start suggesting. If a group wants to
> skip ahead, any of them can say *"what do you think?"*
>
> For a summary of where the idea got to, say *"summarise that"* — it won't do
> it unprompted, because the group is usually still talking.

### The rest

| Demo | What it does | Say this |
|---|---|---|
| **Welcome** | Explains the Hub, about thirty seconds, a line at a time | "do the welcome" |
| **About Reachy** | Explains the robot to non-engineers, honestly, including what's running inside it | "tell me about yourself" |
| **Vision & Face Tracking** | Follows you with its head, and offers once to learn your name | "show me face tracking" |
| **Storyteller** | Makes up a short story | "tell me a story" |
| **Dance** | Dances on a loop | "let's dance" |
| **Timers** | Spoken timers that keep running even if you switch demos | "set a timer for five minutes" |
| **Idle** | Sits quietly, starts nothing. Good during a talk | (click it) |

### Getting it to remember someone

In **Vision & Face Tracking**, it offers to learn the name of **every** face it
doesn't know — the second and third visitor get asked the same as the first.
From there it's just a conversation: it **holds the microphone open** for the
whole exchange, so **no "Hey Reachy" is needed for any of it**, right up until
a name is locked in. Your 🎤 Open mic setting isn't changed and comes straight
back afterwards. Say yes, tell it your name however you like (*"my name's Sarah, nice to
meet you"* works fine — it picks the name out), and it says the name back to
check: *"Sarah. Did I get that right?"*

**Nothing is saved until you say yes to that.** If it heard wrong, say so —
*"no, it's Sara"* — and it checks the new one. It gets three goes, then leaves
you alone rather than nagging.

Answer however you like — *"go on then"*, *"why not"*, *"of course"* all count
as yes. If it doesn't catch you it asks once more rather than assuming you said
no, and it waits while you're mid-conversation rather than interrupting.

**It does not forget.** Names are kept on the laptop and survive restarts, and
it keeps learning your face — every time it recognises you clearly, standing at
a new angle or in different light, it stores that view too. So it recognises
you more reliably the more often you're around, rather than only from the spot
you happened to be standing the day you told it your name.

Once it knows you, it greets you by name in every demo. If a wrong name ever
slips through anyway, say *"Hey Reachy, that's not my name, it's Sarah"* any
time and it corrects itself.

It only offers once, and won't ask again for ten minutes, so it doesn't pester
a room full of people.

---

## Building your own features

Open **Build a feature** on the dashboard to make your own button — most often
a welcome for a particular group. Describe what you want in plain English and
the assistant drafts the words; you edit them and press Save. **The button
appears straight away. The robot does not need restarting.**

A feature can say things, ask a question and answer the reply, dance, and wait
until somebody is standing in front of it. That is all it can do, on purpose —
it is why this is safe to hand to anyone.

Pressed the wrong one mid-visit? **Disable** takes it off the dashboard and
keeps the script. **Delete** removes it for good.

Two honest notes. Features are **not reviewed** — anyone who can open the
dashboard can add one, and the robot will say it to whoever is standing there,
so put your name in the box. And if the dashboard is ever unreachable, a
technical person can remove a bad one with
`sqlite3 data/memory.db "DELETE FROM custom_features WHERE id='...'"` followed
by a restart.

Full instructions: `docs/CUSTOM_FEATURES.md`.

---

## Web search — the 🌐 tick box

**Off by default, and usually leave it off.**

Off, Reachy answers from what it already knows. Ask it today's weather and it
says it has no way to know — which is true.

On, it can look things up: the weather, recent news, anything current. Answers
take a few seconds longer and cost a small amount each time.

**It makes a good demonstration in itself.** Ask *"what's the weather in Dublin
right now?"* with it off, then tick it on and ask again. That contrast shows
visitors the difference between what an AI has learned and what it can go and
find — a distinction most people haven't thought about.

Turn it back off afterwards to keep conversation quick.

---

## Open mic — the 🎤 tick box

**Off by default.** Tick it and a conversation flows the way it would with a
person: you still say **"Hey Reachy"** once to start, and after that you can
just keep asking. No wake phrase before every question.

It stays open for about **half a minute** after the robot finishes answering.
Quiet for longer than that and it goes back to normal — say "Hey Reachy" again
and you're straight back into it. You'll see *"Listening — no wake word
needed"* on the screen while it's open.

Everything else still works while it's open. "Go to sleep" still stops it,
switching demos still works, and if you say "Hey Reachy" out of habit it just
ignores the phrase rather than treating it as part of your question.

**When to use which.** Open mic is better one-to-one — a colleague trying it
out, a visitor who wants a proper back-and-forth. Leave it **off** for a busy
room or a crowd demonstration: with several people talking at once, the wake
phrase is what stops the robot answering a conversation it merely overheard.

---

## On your phone

Connect the phone to the same WiFi as the laptop, then type the address the
black window prints when it starts — something like `http://10.19.4.73:8080`.

Handy when you're standing with visitors and the laptop is across the room.

The page is built for the phone rather than squeezed onto it: the buttons go to
one full-width column, the settings fold away behind ⚙, and everything you tap
is at least a fingertip across. Two people can have it open at once — the
folders, the switches and the personality are shared, because there is only one
robot. Only light-or-dark is per device.

---

## What to tell visitors it can't do

Worth knowing so you're not caught out:

- **It can't move around** — no wheels, no legs, no arms. Only its head and antennas.
- **It can't play music or sounds.**
- **It can't read or look at anything you show it.** It sees faces, not objects or text.
- **It doesn't know today's date or the news** unless web search is switched on.
- **It won't guess about the Hub.** Asked something it doesn't know — which
  headsets are in the room, for instance — it says so and suggests asking a
  human. That's deliberate: a robot inventing a plausible answer in front of
  the people who'd know is worse than one admitting it doesn't know.

---

## If it's not working

| What you see | What to do |
|---|---|
| It ignores "Hey Reachy" | Check the screen — does your speech appear as text? If yes, the microphone is fine. Try just saying **"Reachy"** on its own, which it also answers to, and pause before your question. |
| **"NOT FOUND — no Reachy daemon answered"** | The laptop and robot are on different WiFi networks. Check the robot is on, joined the same network, then **Start Reachy** again. |
| "UNREACHABLE" | The robot dropped off the network. Restart the robot, then **Start Reachy** again. |
| It talks over itself / two voices | Two copies are running. Close every black window, then **Start Reachy** once. |
| It answers something nobody asked it | Open mic is on and it picked up the room. Untick 🎤 **Open mic** — in a crowd the wake phrase is what keeps it out of other people's conversations. |
| It wakes when nobody said "Hey Reachy" | Someone nearby is called *Ricky*, *Richie* or *Rachel*. It answers to those on purpose, so visitors with accents are heard. Nothing to fix; it gives up after a moment. |
| Frozen but still talking | Wait a minute — it restarts itself. |
| Answers are slow or odd | Check the 🧠 label. If it says "on this laptop", the internet or the API key is down and it has fallen back to the smaller offline model. It still works, just less well. |
| A demo button is greyed out | It says why underneath. Usually a demo that failed a few times and was set aside; click it to re-enable, or restart. |
| Nothing works after 3 minutes | Power the robot off and on, close the window, **Start Reachy** again. |

Still stuck? Restart the laptop. Truly stuck? Call whoever set it up.

---

## Before a visit — a two-minute check

1. Sign in, robot on, wait for the wiggle.
2. Say *"Hey Reachy, who runs the AI XR Hub?"* — confirms speech, the AI, and
   the Hub facts all work.
3. Check the 🧠 label says a model **in the cloud**, not on this laptop.
4. Set the volume for the room. Too loud distorts; about 80% is usually right.
5. Leave it in **AI Conversation** for visitors arriving.
