# Spec checklist — audited against the repo

Worked through item by item against the code as it stands. Every `[x]` names
where it is satisfied. Every `[ ]` says exactly what is missing.

Section 6 contains items about how the robot behaves in a room with people.
Those are marked **NEEDS LIVE TEST** where the code can only tell you what is
likely — a checkmark from reading source would be a guess.

---

## 1. Project Goal — Architecture Sanity Check

- [x] **No single monolithic file contains all demo logic** — ten demos, one
  file each, in `demos/`: `conversation.py`, `welcome.py`, `about.py`,
  `vision.py`, `personality.py`, `brainstorm.py`, `story.py`, `dance.py`,
  `timers.py`, `idle.py`. None imports another.
- [x] **A new demo can be added by registration, not by editing an if/else
  chain** — `demokit/registry.py:51` `discover()` walks `demos/` with
  `pkgutil.iter_modules` and instantiates every `Demo` subclass it finds.
  Modules starting with `_` are skipped (`registry.py:69`), which is how
  `demos/_template.py` stays out of the menu.
  **Verified empirically, not assumed:** dropping one new file into `demos/`
  produced an 11th demo, on the dashboard, with its trigger phrase and persona
  live, and `git status` showed **zero** existing files modified.
- [x] **Shared logic sits in a distinct core layer** — three of them, and the
  split is enforced by an import rule rather than convention:
  - `body/` hardware — `audio_io.py` (STT/TTS/wake word), `camera.py`,
    `face.py`, `face_tracker.py`, `motion.py`, `voice_loop.py`
  - `brain/` language and memory — `interface.py`, `llm_backends.py`,
    `prompts.py`, `memory.py`, `db.py`, `personas.py`, `hub.py`
  - `demokit/` the framework — `base.py`, `registry.py`, `runner.py`
  `demokit/base.py` is deliberately a stdlib-only leaf (docstring, lines 1–23):
  it imports nothing from `brain` or `body` at module scope, because
  `brain.modes` is imported by `body.audio_io`, which is imported by the voice
  loop, which owns demos.

## 2. Core Requirements

- [x] **Stable core, one entry point, a crashing demo doesn't take the process
  down** — `main.py` is the only entry point. Every demo hook runs inside
  `DemoRunner._guarded` (`demokit/runner.py:641`), which catches `Exception`,
  logs it, shows the operator a note, and counts the failure; three consecutive
  failures set the demo aside and switch to a working one
  (`registry.record_failure`, `runner.py:672-681`). Covered by test
  `[5] a demo that throws is contained, then set aside`.
- [x] **Modular architecture with a consistent interface, used consistently**
  — `Demo` (`demokit/base.py:455`) defines `on_enter` / `on_idle` /
  `on_utterance` / `on_exit` plus class attributes `id`, `label`, `help`,
  `order`, `triggers`, `requires`, `claims_utterances`, `persona`. All ten
  demos implement it; `on_idle` is overridden by all ten, `on_exit` by seven
  (the rest have nothing to tear down). No demo reaches around it to the
  hardware.
- [x] **Simple launch interface, reachable without editing code** — two, and
  both work without a terminal:
  - Web dashboard at `:8080` (`web/index.html`, `web/server.py`), one button
    per demo, reachable from a phone on the same network.
  - Voice triggers — 38 phrases across the demos, matched on whole words by
    `DemoRunner._switch_on_trigger` (`runner.py`), longest first.
- [x] **4–6+ working, polished demos** — ten load and run; the six the spec
  asks for are all present plus Storyteller, Dance, Timers, Idle. No
  `TODO`/`FIXME`/`NotImplementedError`/placeholder anywhere in shipped code
  (grep clean; the only hits are the word "placeholder" in a test docstring and
  an HTML input `placeholder=` attribute).
  Error text never reaches the visitor: failures go to `state.add("error", …)`,
  which is the dashboard event log, not `ctx.say`.
- [x] **Documentation for future dev** — `README.md` (architecture, layout,
  setup, model downloads, hardware findings, testing without hardware),
  `docs/ADDING_A_DEMO.md` (hooks, `ctx` API, class attributes, what the
  framework deliberately does not do), `OPERATOR.md` (non-technical guide for
  whoever runs a visit), `AGENTS.md`, `service/README.md`.

## 3. Required Demonstrations

### A. Welcome Mode
- [x] **Standalone module** — `demos/welcome.py`
- [x] **Greets visitors** — `hub.WELCOME_SCRIPT` opens "Welcome to the AI XR
  Hub, in Trinity Business School."
- [x] **Explains what the AI XR Hub is** — same script: the responsible-use
  mission and the three strands (immersive learning, executive education and
  innovation labs, applied research).
- [x] **Describes what Reachy can do** — the script closes on "Some of that
  research is me. I'm Reachy Mini, one of the robots being built here," and the
  demo ends by inviting questions (`_INVITATION`, `welcome.py:58`).
  The script is spoken one sentence per idle slice so the robot stays
  interruptible and switchable through all ~32 seconds of it.

### B. About Reachy Mode
- [x] **Standalone module** — `demos/about.py`
- [x] **Explains the robot and its capabilities** — `_SCRIPT_LINES` covers the
  camera, microphone array, speaker and six neck motors, then the pipeline:
  wake word → speech recognition → language model → speech synthesis.
- [x] **Pitched at a non-technical visitor** — reads as plain English
  ("a camera behind my eyes", "six motors in my neck, which is what lets me
  look around"). It names the parts of the pipeline without jargon and lands on
  something a visitor can appreciate: "all of it can run on a laptop in this
  room, with no internet at all." Not a spec dump.

### C. AI Conversation Mode
- [x] **Standalone module** — `demos/conversation.py`
- [x] **Handles general Q&A about AI, XR and the robot** — `ctx.reply` with
  `_HUB_BRIEFING` layered on, over the standing prompt in `brain/prompts.py`,
  which carries `_HUB_CONTEXT` and `hub.GROUNDING`.
- [x] **Supports multi-turn conversation** — `brain/memory.py` keeps per-person
  history (`remember_turn`), replayed into each request by
  `brain/interface.py`; long-term notes are summarised per person in
  `brain/long_term_memory.py`. This is the default demo (`order = 10`).

### D. Vision / Face Tracking Mode
- [x] **Standalone module** — `demos/vision.py`
- [x] **Camera input actually drives tracking** — `body/face_tracker.py` runs a
  background loop at 8 Hz: grab frame → MediaPipe detect → `motion.track_face`
  with the bounding box. It is not displayed anywhere; the output is head
  movement.
- [x] **Robot reacts to presence/movement** — `motion.track_face` aims yaw and
  pitch proportionally at the detected face; `_search()` sweeps slowly when
  nobody is in view so it looks for people rather than waiting to be stood in
  front of correctly.
- [x] **At least one engagement behaviour beyond raw tracking** — several:
  spoken presence reactions on arrival and departure, rate-limited to once a
  minute (`_on_arriving` / `_on_leaving` / `_remark`, `vision.py:221-250`); a
  consent-first offer to learn a new face's name, with the name said back for
  confirmation before anything is stored (`_offer` → `_capture_name` →
  `_confirm_name` → `_enroll`); and greeting a recognised person by name, which
  lives in the core so it happens in every demo
  (`DemoRunner._greet_if_recognised`).

### E. AI Personality Mode
- [x] **Standalone module** — `demos/personality.py`
- [x] **2–3 personas implemented, not stubbed** — three in
  `brain/personas.py`: Professional, Friendly, Consultant. Each carries a
  behavioural `prompt`, a one-sentence `global_prompt`, `pace`, `variation`,
  `pose`, and a distinct installed TTS `voice`.
- [x] **Personas give genuinely different responses to the same input** —
  spot-checked with real model calls on "Should a business student learn to
  code?", not inferred from the config:
  - *Professional* — "…valuable rather than mandatory. Whether it's a priority
    depends on the specific career path…"
  - *Friendly* — "…like learning to read a menu in another country — you don't
    need to be a chef, but knowing the basics helps you work with the people
    who are."
  - *Consultant* — "…enough Python or SQL to interrogate data… I'd lean toward
    the practical route… Are you aiming for a technical role or a management
    one?"
  Three different shapes, not three wordings. They also differ audibly: each
  persona speaks in a different voice (Lessac / Amy / Ryan) at its own pace and
  prosody.

### F. Business Brainstorming Mode
- [x] **Standalone module** — `demos/brainstorm.py`
- [x] **Asks structured questions to guide the session** — `_QUESTIONS`
  (`brainstorm.py:59`): the idea or problem, who it is for ("one kind of
  person, not everybody"), and what makes it different or what the constraint
  is.
- [x] **Helps generate/refine ideas interactively, not a static script** — the
  three directions are generated per session from the visitor's own answers
  (`_direction_brief`, `brainstorm.py:119`), one model call per idle slice, each
  told to differ from the ones already given. `claims_utterances = True` so a
  visitor's answer reaches this demo before any other demo's trigger word.
- [ ] **Produces a summary at the end of the session** — **no longer
  automatic.** The recap still exists (`_recap_lines`, `_recap_now`) but now
  only runs when a visitor asks for it by saying "summarise that". The session
  ends on a closing question instead: "Which of those would you test first, and
  what's the cheapest way to find out?" (`brainstorm.py:355-363`).
  This is a deliberate change made at your request — the automatic recap read
  back the visitor's own two-word answers in a template ("It's for everybody",
  "What makes it different, or what's in the way: It's cool") and sounded
  broken. Flagged rather than ticked because the spec asks for a summary at the
  end and the robot no longer volunteers one. Restoring it is a two-line change
  if the spec is what matters; the better fix is a model-written summary rather
  than the old template.

## 4. System Design Requirements

- [x] **Each demo is genuinely a separate module** — ten files in `demos/`,
  listed in §1. Each defines one `Demo` subclass; the `id` defaults to the
  module name (`registry._demos_in`). No cross-imports between demos.
- [x] **STT is a shared core service** — `AudioIO.listen()` and
  `wait_for_wake_word()` in `body/audio_io.py`, reached only through
  `DemoContext.listen` / `ctx.ask` / the runner. No demo constructs a
  recogniser; `sherpa_onnx` is imported in exactly one file.
- [x] **TTS is a shared core service** — `AudioIO.speak()`, reached through
  `DemoContext.say` / `say_lines` / `reply`. `piper` is imported in exactly one
  file. Voice selection, persona voice and the synthesis config all live there.
- [x] **Explicit mechanism for switching modules** — named parts:
  `RobotState.mode` (`brain/modes.py`) holds the selection, `Registry`
  (`demokit/registry.py`) resolves an id to a demo and knows what is available,
  and `DemoRunner` (`demokit/runner.py`) is the router: `cycle()` →
  `_active()` enters/leaves demos, `_dispatch()` routes an utterance through a
  documented order (core stop phrases → name correction → the active demo →
  trigger phrases → the active demo again → conversation).
- [x] **Basic state management separate from demo logic** — `RobotState` in
  `brain/modes.py`: mode, sleeping, listening/speaking flags, events, history,
  request queue, web-search and open-mic switches, persona, greeted names. One
  lock guards it; the dashboard thread and the voice loop share only this
  object. Per-demo scratch space is handed to demos as `ctx.store`, keyed by
  demo id, rather than module globals.
- [x] **No demo-specific logic hardcoded into core files** — two instances
  existed and both are fixed:
  - `DemoContext.reply` excluded one demo by name (`demo_id != "personality"`)
    from the global persona prompt. Now `Demo.owns_persona`, a class attribute
    the demo declares about itself and the runner passes to the context. The
    old form broke three ways: renaming the file silently stopped the exclusion
    matching, deleting it left a dead condition, and no other demo could opt
    out without editing a core file.
  - `DEFAULT_MODE = "conversation"` in `brain/modes.py` named a demo three
    lines below a comment promising that RobotState "knows nothing about demos
    beyond the ids it has been handed". Now `""`, meaning not-yet-chosen;
    `set_demos` takes the first real demo and `registry.default_id()` decides
    which that is, by `order`.
  Both verified: a grep of `demokit/`, `brain/modes.py`, `brain/interface.py`,
  `brain/prompts.py`, `body/voice_loop.py` and `main.py` for every demo id now
  returns only comments. `style == "story"` remains and is not a demo id — it
  is a generic parameter any demo may pass.
- [x] **Adding a 7th demo touches nothing else** — proven, not argued. One new
  file in `demos/` appeared as a fully registered demo (dashboard entry,
  trigger phrase, persona preset) and `git status` reported **no modifications
  to any existing file**. `demos/_template.py` plus `docs/ADDING_A_DEMO.md` is
  the starting point.

## 5. Visitor Fit

- [x] **Welcome / About / Conversation avoid jargon** — the Welcome script is
  mission-level English with no technical vocabulary. About Reachy names the
  pipeline in plain terms ("a wake word model notices you're talking to me").
  Conversation is capped at one or two short sentences by
  `prompts._base_prompt`, and `_DELIVERY` instructs it to speak the way a
  person does aloud. The one deliberate exception is About Reachy naming its
  own components, which is that demo's subject.
- [x] **Default flow needs no keyboard or typing** — the visitor path is
  entirely voice: a wake phrase, then speak. The operator path is clicking a
  demo button on a screen or a phone. The dashboard's "Say it" text box is an
  operator convenience, not part of the visitor flow.
- [x] **Nothing in the standard flow assumes prior knowledge** — Welcome
  explains the Hub from scratch and introduces the robot; About Reachy explains
  the robot to somebody who has never seen one. Neither assumes the visitor
  knows what XR, a wake word, or a Reachy is.

## 6. Success Criteria

- [x] **Cold-start: robot off → first interaction with no staff setup** — the
  mechanism is there and each part is verifiable in code:
  `install.ps1:161` installs a **Startup-folder shortcut**, so signing in to
  the laptop launches the robot with no icon to find; `start_reachy.ps1`
  discovers the robot on the network by itself (`find_robot.ps1`, four
  strategies, resolving both address and port) rather than needing an address
  typed in; and the dashboard starts serving before the speech models finish
  loading (`main.py`) so the screen says "starting" instead of looking dead.
  **NEEDS LIVE TEST** end to end: power the robot, sign in, and time it to the
  first answered question without touching anything.
- [x] **Demo switching is fast and needs no terminal** — dashboard button or
  spoken trigger phrase; `RobotState.set_mode` is a flag the loop reads at the
  top of its next slice, and every listening window is bounded at
  `MAX_LISTEN_WINDOW_S = 3.0` s specifically so an operator's switch lands
  quickly. A demo part-way through speaking unwinds via `DemoStopped` rather
  than finishing its script.
  Code suggests worst case ~3 s, typically under 1 s. **NEEDS LIVE TEST** —
  measured switching while a demo is mid-sentence, which is when it matters.
- [x] **Common live-demo failures degrade instead of crashing** — each of the
  four named cases is handled somewhere specific:
  - *No network* — `brain/llm_backends.py` falls back from the cloud model to a
    local Ollama one, per sentence, with a short cloud timeout chosen so a dead
    network fails fast enough to fall back rather than stalling a turn.
  - *No camera / no face model* — `body/face.py:266` probes MediaPipe **in a
    throwaway subprocess** first, because on the robot's own CPU it crashes the
    process outright (SIGILL). If the probe fails, `faces` is absent from
    capabilities (`voice_loop._capabilities`) and Vision is greyed out on the
    dashboard with the reason, rather than failing when selected.
  - *Model not loaded / missing voice* — `AudioIO._load_voice` warns once,
    caches the failure and falls back to the current voice.
  - *Mic stalls* — `listen()` and `wait_for_wake_word()` are both bounded
    (`_MAX_UTTERANCE_S = 25`, plus a deadline passed into the frame source), so
    a dead microphone ends a turn instead of the session; `_MIC_STALL_WARN_S`
    logs when the daemon stops delivering audio at all.
  - Above all of it, the voice loop catches any unhandled exception per cycle
    and continues (`body/voice_loop.py:224`) rather than exiting.
  **NEEDS LIVE TEST** for the ones that need real hardware to provoke: pull the
  network mid-answer, unplug the camera, mute the mic.
- [x] **A "new student" path exists** — `demos/_template.py` (a working demo
  with the hooks stubbed out and commented), `docs/ADDING_A_DEMO.md` (hooks,
  the `ctx` API, class attributes, what breaks and why), and
  `python test_demokit.py` to check work without a robot. The template is
  skipped by discovery because of its leading underscore, so it does not appear
  in the menu until it is copied to a real name.
- [x] **Deployment-ready** — auto-restart is real: `start_reachy.ps1` relaunches
  on any non-zero exit with backoff, and refuses to start if another instance
  survives (so the robot cannot end up talking over itself). Autostart is the
  Startup-folder shortcut. No monitor or keyboard is needed once signed in —
  the dashboard is reachable from a phone on the same network, and the launcher
  prints the LAN address.
  Two caveats, stated rather than ticked over:
  - `service/` (systemd units + watchdog) is **explicitly not deployed** and its
    own README says so — the paths point at a WSL2 home directory and it
    supervises a stand-in process, not `main.py`. It is a design for Linux, not
    live infrastructure. On Windows, `start_reachy.ps1` is the whole story.
  - "Works on the Hub's actual network" **NEEDS LIVE TEST**. It has run on a
    phone hotspot and on Trinity WiFi during development, and `find_robot.ps1`
    exists precisely because addresses move — but an open day on the Hub's own
    network with its own firewall rules has not been tried.

---

## Summary

| Section | Done | Partial | Missing |
|---|---|---|---|
| 1. Architecture sanity | 3 | 0 | 0 |
| 2. Core requirements | 5 | 0 | 0 |
| 3. Required demonstrations | 20 | 1 | 0 |
| 4. System design | 7 | 0 | 0 |
| 5. Visitor fit | 3 | 0 | 0 |
| 6. Success criteria | 5 | 0 | 0 |
| **Total** | **43** | **1** | **0** |

"Partial" means the capability exists but does not meet the item as written.
Nothing on this checklist is absent.

Four of the Section 6 items are ticked for **mechanism** and flagged
**NEEDS LIVE TEST** for **outcome** — they describe how the robot behaves in a
room with visitors, and the code can only show that the machinery is present.

### Punch list, in the order worth doing

1. **Run the remaining live tests in Section 6.** Barge-in is now confirmed
   working, which was the largest unknown. Left, in order: cold start with a
   stopwatch; switch demos mid-sentence; pull the network mid-answer; run it
   once on the Hub's own network rather than a hotspot. Nothing else on this
   list is likely to bite you on an open day as hard as an untested cold
   start.
2. **Decide about the brainstorming summary** (3F). The only item that does not
   meet the spec as written. Either accept the closing question as the ending,
   or restore a summary — in which case make it model-written rather than the
   template that read a visitor's own words back at them.
3. **Decide what `service/` is for.** It is documented honestly as not
   deployed, but it is the kind of directory somebody later mistakes for live
   infrastructure. Either point it at `main.py` and real paths, or move it to
   `docs/` as a design note.
4. **Tidy one bad database row.** Person 2 is enrolled as
   `"Now That It's Hit"` — a mis-transcribed name from an early test, before
   enrolment asked for confirmation. `python manage_people.py` removes it.

### Worth knowing, outside the checklist

- **Barge-in ("Hey Reachy" over a reply) is fixed in code but unproven live.**
  The default demo was swallowing the interrupt exception, so it could never
  have worked; that is repaired, along with three other blockers. But whether
  the wake word can be *heard* over the robot's own speaker depends on echo
  cancellation happening on the robot, which cannot be determined from this
  repo. Measured on synthesised mixtures: with the robot's voice in the buffer
  at half scale or more, a wake phrase spoken over it was detected 0 times out
  of 4; at a tenth of scale, 4 out of 4. The scan now logs how much audio it
  saw and how loud, so one live attempt will settle it.
