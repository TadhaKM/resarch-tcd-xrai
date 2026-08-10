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

**One "Hey Reachy" per question.** This is the thing visitors get wrong most
often — they say it once and then keep talking. It needs the phrase each time,
which is deliberate: in a room full of people it's the only way the robot knows
someone is talking to *it* and not to each other.

These all work: **"Hey Reachy"**, **"Hello there Reachy"**, **"Attention
Reachy"**.

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

Along the top:

| What you see | What it means |
|---|---|
| A green dot | Connected and working |
| 🧠 with a model name | Which AI is answering, and whether it's on the laptop or online |
| 🌐 **Web search** | Off by default. Tick it to let Reachy look things up — see below |
| 🔊 slider | Speaker volume |
| A dropdown | Which voice it speaks in — four to choose from |
| **Clear answers** | Makes it forget answers it has memorised. Only needed if it keeps repeating something wrong |

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

**AI Personality** — say *"switch personality"*. It answers the **same
question** again as a different character: Professional, Friendly, then
Consultant. They differ in wording, in tone of voice, and in how the robot
holds its head.
> Ask a question with no single right answer — *"Should a business student
> learn to code?"* — then say *"Hey Reachy, switch personality"* two or three
> times. Factual questions make all three sound the same; arguable ones make
> the difference obvious.

**Business Brainstorming** — say *"let's brainstorm"*. It interviews the
visitor about an idea, then offers three directions and recaps it. Genuinely
good with business-school groups because they participate rather than watch.
> Let a visitor drive this one. It asks the questions.

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

In **Vision & Face Tracking**, it offers to learn the name of a face it doesn't
know. Say **"Hey Reachy, yes"**, then **"Hey Reachy,"** and your name. It
greets you by name next time it sees you.

It only asks once, and won't ask again for ten minutes, so it doesn't pester a
room full of people.

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

## On your phone

Connect the phone to the same WiFi as the laptop, then type the address the
black window prints when it starts — something like `http://10.19.4.73:8080`.

Handy when you're standing with visitors and the laptop is across the room.

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
| It ignores "Hey Reachy" | Check the screen — does your speech appear as text? If yes, the microphone is fine; say the phrase again more clearly and pause before your question. |
| **"NOT FOUND — no Reachy daemon answered"** | The laptop and robot are on different WiFi networks. Check the robot is on, joined the same network, then **Start Reachy** again. |
| "UNREACHABLE" | The robot dropped off the network. Restart the robot, then **Start Reachy** again. |
| It talks over itself / two voices | Two copies are running. Close every black window, then **Start Reachy** once. |
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
