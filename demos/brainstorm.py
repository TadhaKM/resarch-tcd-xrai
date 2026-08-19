"""A facilitated idea session that follows the conversation it is in.

Built for the case it actually meets: a small group -- four people around the
robot at an open day, not one visitor at a desk -- talking about a half-formed
idea, interrupting each other, and changing what the session is about halfway
through. Three fixed questions asked in order cannot survive that. Asking "who
is it for?" to a group that spent the last minute explaining exactly who it is
for is the moment they stop treating it as a conversation, and it is the moment
this demo used to fail.

So the robot writes each of its own turns from what has been said. It still
knows what a useful session needs -- what the idea is, who specifically it is
for, what makes it hard -- but that is a standing instruction on the prompt
rather than a script, so it can pick up somebody's own words, dig into the
assumption underneath them, answer a question the group puts back to it, and
come to the missing pieces in whatever order the room takes it. When it has
enough to work with it stops interviewing and gives directions back, and after
those the conversation simply continues.

Four decisions about how this talks to the framework are worth knowing:

- Questions are spoken with ctx.say or ctx.reply and answered through
  on_utterance, never with ctx.ask. ask is say-then-listen inside one hook, and
  a hook holding the microphone is a robot the operator cannot switch away
  from; here that hook would also be the one making a model call, so a slow
  backend and a quiet room would compound into a demo that is stuck rather than
  slow. One model call per group turn, each in its own idle slice, keeps every
  hook short.
- The microphone is held open only while a question of ours is outstanding
  (see _hold). That is what makes a group able to just answer -- four people
  taking turns to say "hey Reachy" first is not a conversation -- without the
  robot transcribing the room for the whole visit. It is released the moment
  the answer lands, whenever the session leaves the talking stage, and in
  on_exit, so the operator's own switch comes straight back into force.
- No hook returns listen_for=0.0 while there is script left to speak. Zero
  means DemoRunner.cycle returns without opening the microphone, so a run of
  such slices is a stretch of speech that cannot be interrupted and during
  which the wake word is simply unheard. In a facilitation demo, where the
  whole point is that the group takes part, that is the worst thing this file
  could do: the bridge, the three directions and the closing question each
  leave a one-second window instead. The wake-word stream survives across
  windows, so a phrase spoken over a boundary is still matched and the lines
  still sound back-to-back.
- claims_utterances, because somebody answering "dance studios in Dublin" must
  not make the robot dance in the middle of their own brainstorm.

Everything advances one step per on_idle slice, including one generated
direction per slice: three model calls in a single hook would hold the robot
for the better part of a minute with the microphone shut and the dashboard
unable to switch away.
"""

import time

from brain import hub
from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S


#: Session stages, held in ctx.store["stage"].
_TALK = "talk"
_BRIDGE = "bridge"
_GENERATE = "generate"
_SUMMARY = "summary"
_DONE = "done"

#: The first thing asked, before there is any conversation to follow. Fixed
#: rather than generated: with an empty transcript the model has nothing to be
#: specific about and produces the same opener as every other assistant, and
#: this one has a job that prompt wording cannot do -- telling a group of four
#: that any of them may answer, which is otherwise a thing they work out by
#: standing there in silence for a while first.
_OPENER = "So -- what are we working on? Any of you can jump in."

#: Said when a question has gone unanswered long enough that the silence has
#: stopped being thinking. Deliberately lowers the bar rather than repeating
#: the question, because a group that has not answered usually has not decided
#: who should.
_NUDGE = "Rough is fine. Even the half of it you're sure about."

#: What the robot falls back on when the model returns nothing at all -- a dead
#: backend, or a reply that came back empty. Generic on purpose: they are only
#: ever reached when the specific question could not be generated, and a
#: session that carries on with a blunt question is better than one that stalls
#: mid-sentence in front of a group.
_FALLBACK_QUESTIONS = (
    "Say more about that -- what's the part you're least sure about?",
    "Who feels this problem most? One kind of person, not everybody.",
    "What makes it hard? Either the competition, or what's in your way.",
    "What would have to be true for this to work?",
)

#: One direction per ordinal, and the ordinal is spoken, so the count and the
#: words that announce it cannot drift apart. Three: two reads as the model
#: hedging between an option and its opposite, four outlasts a standing group's
#: attention.
_ORDINALS = ("first", "second", "third")

#: Group turns carrying actual content before the robot stops asking and starts
#: offering directions. Counted rather than fixed at three questions: a group
#: that answers the first question with a paragraph has told the robot more
#: than one that needed four, and the old three-questions-then-directions rule
#: could not tell those apart.
_ENOUGH_ANSWERS = 4

#: What it takes to earn a SECOND set of directions, once a group has carried
#: on talking past the first. Much higher than the first threshold on purpose:
#: they already have directions, so the useful thing is to keep facilitating,
#: and a robot that offers three more ideas every couple of minutes has stopped
#: listening to what they are doing with the last three.
_MAX_ANSWERS = 9

#: Words below which a turn is a noise rather than material -- "yeah", "mm",
#: "sure", one person agreeing with another. Counted turns are what advance the
#: session, so without this a group nodding along would reach the directions
#: having said nothing.
_SUBSTANCE_WORDS = 4

#: How much of the session is restated to the model each turn. Bounded because
#: the whole thing rides on the system prompt of every call (see
#: _facilitator_brief), and an unbounded transcript on a local model is a
#: session that gets slower with every question it asks.
_TRANSCRIPT_TURNS = 16

#: Longest single turn quoted into the prompt. A group member who talks for a
#: minute would otherwise crowd out everything anyone else said.
_TURN_WORD_CAP = 60

#: Questions quoted back to the model as ones it has already asked. Enough to
#: stop it circling, short enough that the don't-repeat rule does not become
#: most of the prompt.
_ASKED_KEPT = 6

#: Idle slices a question may go unanswered before the robot does something
#: about it. Each slice is one listen window, so this is roughly fifteen
#: seconds -- long enough for four people to look at each other and decide who
#: is answering, which is what that pause usually is.
_MAX_WAITS = 4

#: Times the robot will lower the bar and re-open a question nobody answered
#: before it stops asking. Two is where a quiet group has made its point.
_MAX_NUDGES = 2

#: Multi-word on purpose. A bare "summary" or "summarise" turns up inside real
#: answers -- "a summarising tool for solicitors" is exactly the kind of thing
#: this demo gets asked about -- and would end the session the visitor is still
#: in the middle of.
_SUMMARISE_PHRASES = ("summarise that", "summarize that", "summarise it", "summarize it", "sum that up")
_RESTART_PHRASES = ("start over", "start again", "start from scratch")

#: Asking for the payoff early. A group that says this has finished talking,
#: and continuing to interview them past it is the robot ignoring the clearest
#: instruction it will get all session.
_IDEAS_PHRASES = (
    "give us ideas", "give me ideas", "give us some ideas", "give me some ideas",
    "what do you think", "any suggestions", "any ideas", "your ideas",
    "what would you do", "some directions", "give us directions",
)

#: The runner keeps one store per demo id for the life of the process and never
#: clears it (see DemoRunner._stores), so a session abandoned when the operator
#: switched away would otherwise resume mid-interview for the next group at the
#: next open-day slot. Anything older than this starts fresh instead.
_SESSION_STALE_S = 300.0

#: The gap left after each spoken script line -- the bridge, each direction.
#: Short enough that the lines still read as one continuous passage, and long
#: enough that the core actually opens the microphone, which is the only thing
#: standing between this demo and a minute of speech a group cannot interrupt.
#: Never 0.0 here: see the module docstring.
_BETWEEN_LINES_S = 1.0


def _tidy(text: str) -> str:
    """Trim one transcript into something worth putting in a prompt.

    Turns arrive from whichever recogniser won: Whisper's transcript is cased
    and punctuated, the streaming model's fallback is bare capitals with no
    punctuation (its output is logged that way throughout body/audio_io.py).
    The second one reads to a model as shouting, and a long one crowds out
    everybody else's turn, so both are flattened here rather than at each of
    the four call sites that build a prompt.
    """
    text = " ".join(text.split())
    if text.isupper():
        text = text.lower()
    words = text.split()
    if len(words) > _TURN_WORD_CAP:
        text = " ".join(words[:_TURN_WORD_CAP]) + "..."
    return text


def _has_substance(text: str) -> bool:
    """Whether a turn moved the session on, or was somebody agreeing."""
    return len(text.split()) >= _SUBSTANCE_WORDS


def _transcript(turns: list[tuple[str, str]]) -> str:
    """The session so far, as the model sees it.

    Labelled "The group" rather than "The visitor" throughout, and that is not
    cosmetic: told it is talking to one person, the model writes follow-ups
    that assume the last answer and the next one come from the same mouth --
    "you said earlier that..." to somebody who did not say it. Told it is a
    group, it asks the room.
    """
    lines = []
    for who, text in turns[-_TRANSCRIPT_TURNS:]:
        speaker = "You" if who == "me" else "The group"
        lines.append(f"{speaker}: {_tidy(text)}")
    return "\n".join(lines)


def _facilitator_brief(turns: list[tuple[str, str]], asked: list[str], closing: bool) -> str:
    """The standing instructions for one facilitation turn.

    This goes to ctx.reply(system=...), not into the message. Everything here
    is instruction and context rather than something a person said, and
    anything in the message is recorded by brain/memory.py as the group's own
    turn: sent that way it would be replayed on every later request for the
    rest of the session, so somebody asking an unrelated question afterwards
    would get an answer generated on top of a transcript in which they
    apparently recited a prompt every thirty seconds.

    The whole session is restated in every brief rather than relied on from
    history: brain/memory.py trims to the last dozen turns and shares one
    bucket across every unrecognised visitor, so on a busy afternoon the early
    answers can be gone by the time they matter most.

    The three things a session needs are stated as goals, not as an order to
    ask them in. That is the entire difference between this and the fixed
    question list it replaced -- the model is told what a useful session has
    covered and left to find its own route there, which is what lets it follow
    a group that answers two of them in one breath and none of them in the
    order anyone expected.
    """
    parts = [
        "You are facilitating an idea session out loud, standing in a room with a small "
        "group -- up to four people, any of whom may answer, and they can hear each other.",
        f"What has been said so far:\n{_transcript(turns)}",
        "Write only your next spoken turn. Rules:",
        "- Two short sentences at most, and finish on exactly one question.",
        "- Follow what they just said. Use their own words, and go at whatever is vaguest, "
        "most surprising, or the biggest thing they are assuming. Never ask something their "
        "last answer already covered.",
        "- If they asked YOU a question, answer it in one sentence, then ask one thing back.",
        "- Across the session you want three things: what the idea actually is, who "
        "specifically it is for, and what makes it hard or different from what exists. Go "
        "after whichever is thinnest -- but ask it about their idea, in their language, not "
        "as a form to fill in.",
        "- Do not summarise what they said, do not list, do not compliment them, and do not "
        "open with filler like 'great' or 'interesting'.",
        "- If the same person has answered several times, put the next question to the rest "
        "of them.",
    ]
    if asked:
        recent = "; ".join(f'"{q}"' for q in asked[-_ASKED_KEPT:])
        parts.append(f"- You have already asked these. Do not repeat or rephrase any: {recent}")
    if closing:
        parts.append(
            "- This is your last question before you give them ideas, so make it the one "
            "that would most change what you suggest."
        )
    parts.append(
        f"The session runs at the AI XR Hub: {hub.MISSION_SHORT} Do not describe the Hub."
    )
    return "\n".join(parts)


def _direction_brief(turns: list[tuple[str, str]], ordinal: str) -> str:
    """The standing instructions for one direction. See _facilitator_brief.

    The Hub's mission is a steer, not something to recite: without it the model
    proposes an app for everything, and told to describe the Hub it will
    paraphrase the mission into a claim nobody checked.

    What history is genuinely load-bearing for here is the no-repeat
    instruction -- the directions already given are the model's own previous
    turns, which is what gives it something to refer to.
    """
    return (
        "You are facilitating a business brainstorm out loud, in a room with the group "
        "whose idea this is.\n"
        f"Everything they have told you:\n{_transcript(turns)}\n"
        "The session runs at the AI XR Hub, so favour directions that fit its mission: "
        f"{hub.MISSION_SHORT} Do not describe the Hub in your answer.\n"
        f"Give the {ordinal} of three directions they could take this, and start your "
        f"answer with the word {ordinal}. Name the direction in a few words, then one "
        "sentence on why it could work -- built on the specifics they actually gave you, "
        "not on generic startup advice. Two short sentences at most. Make it genuinely "
        "different from any direction you have already given in this conversation, and do "
        "not ask a question."
    )


def _summary_brief(turns: list[tuple[str, str]], directions: list[str]) -> str:
    """The standing instructions for a recap, which only ever runs on request.

    It used to be a template of the group's own answers slotted into fixed
    sentences, and hearing that live is what settled this: a group who said
    "everybody" and "it's cool" -- which is what people actually say to a robot
    in a corridor -- got "It's for everybody" read back at them a minute later.
    Their own words in a template do not become a summary. A model given the
    session can at least say what it was about.
    """
    given = "\n".join(f"- {d}" for d in directions if d)
    tail = f"\nDirections you already gave them:\n{given}" if given else ""
    return (
        "You are in a room with a group whose idea you have been talking through, and one "
        "of them has just asked you to sum it up.\n"
        f"The session:\n{_transcript(turns)}{tail}\n"
        "Say where the idea has got to, in three short sentences at most: what it is, who "
        "it is for, and the thing still open. Use what they actually said. Do not quote "
        "them back word for word, do not list, and do not ask a question."
    )


class Brainstorm(Demo):
    label = "Business Brainstorming"
    help = "Talks an idea through with a group, following what they say, then offers directions back."
    order = 60
    #: Facilitating somebody's business idea. Measured and specific is what makes the directions usable.
    persona = "professional"
    #: Request-shaped, not topic-shaped. A bare "brainstorm" or "business idea"
    #: is ordinary vocabulary at a business school -- "we ran a brainstorm this
    #: morning", "students arrive with a business idea" -- and because this demo
    #: claims utterances, firing on those sentences captures the group into a
    #: session nobody asked for.
    triggers = (
        "lets brainstorm",
        "help me brainstorm",
        "brainstorm with me",
        "i have a business idea",
        "got a business idea",
    )
    #: An answer must reach this demo before any other demo's trigger word is
    #: considered. See the module docstring.
    claims_utterances = True

    # --- hooks -----------------------------------------------------------

    def on_enter(self, ctx: DemoContext) -> None:
        store = ctx.store
        fresh = time.monotonic() - store.get("touched", 0.0) < _SESSION_STALE_S
        resuming = fresh and store.get("stage") not in (None, _DONE)
        if not resuming:
            self._reset(ctx)
        else:
            # Whatever question was in the air when the operator switched away
            # is dropped rather than waited on -- the people in front of the
            # robot now may not be the ones who heard it, and the next turn is
            # written from the transcript anyway.
            store["awaiting"] = False
            store["waited"] = 0
        store["touched"] = time.monotonic()
        # Says what is about to happen, not how to operate the robot. Explaining
        # the wake word is the operator's job and belongs in their guide -- said
        # here it was the first thing a visitor heard about their own idea.
        ctx.say(
            "Picking up where we left off."
            if resuming
            else "Let's brainstorm. Talk it through with me and I'll dig in.",
            "happy",
        )

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        stage = ctx.store.get("stage")
        if stage == _TALK:
            return self._talk(ctx)
        if stage == _BRIDGE:
            self._hold(ctx, False)
            ctx.store["stage"] = _GENERATE
            # Worded from whether they have had a set already. The same line
            # twice in one session is the clearest tell there is that the robot
            # is running a script rather than listening.
            ctx.say(
                "Right. Three more, on what you've just said."
                if ctx.store.get("gave_directions")
                else "Right. Three directions on that.",
                "thinking",
            )
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        if stage == _GENERATE:
            return self._offer_direction(ctx)
        if stage == _SUMMARY:
            return self._summarise(ctx)
        if stage is None:
            # on_enter is guarded, so it can have failed and left nothing set
            # up. Recovering here costs a line; assuming our own hooks ran costs
            # the demo every time someone changes the runner.
            self._reset(ctx)
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        # Finished. Anything said now is "start over", a question for the model,
        # or another demo's trigger, and all three want the microphone open.
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        store = ctx.store
        lowered = text.lower()

        if any(phrase in lowered for phrase in _RESTART_PHRASES):
            self._hold(ctx, False)
            self._reset(ctx)
            ctx.say("Fresh start. Forget everything I just heard.", "happy")
            return True

        if any(phrase in lowered for phrase in _SUMMARISE_PHRASES):
            return self._summarise_now(ctx)

        if store.get("stage") in (_TALK, None) and self._is_opening_trigger(ctx, lowered):
            # This is the trigger phrase that selected the demo ("let's
            # brainstorm"), handed to us by the runner right after on_enter
            # already answered it. Swallowed rather than passed on, so the model
            # doesn't reply to it over the top of the first question.
            return True

        if any(phrase in lowered for phrase in _IDEAS_PHRASES):
            return self._wants_ideas(ctx, text)

        if store.get("awaiting"):
            self._record(ctx, text)
            return True

        # Mid-generation, mid-summary, or done. The turn is kept, because the
        # next direction and any later recap are written from the transcript and
        # something said over the top of a direction is usually the most
        # pointed thing anyone says all session. It is not claimed, though:
        # falling through is what lets somebody ask about the Hub afterwards and
        # get an answer, and what keeps other demos' triggers working.
        if text.strip():
            self._append(ctx, "them", text)
        return False

    def on_exit(self, ctx: DemoContext) -> None:
        # The one thing here that must happen. A demo that exits still holding
        # the microphone open leaves the next demo -- and the operator's own
        # switch -- with a hot mic nobody asked for.
        self._hold(ctx, False)
        answered = ctx.store.get("answers", 0)
        ctx.status(f"Brainstorm paused, {answered} answer(s) in.")

    # --- the conversation ------------------------------------------------

    def _talk(self, ctx: DemoContext) -> IdleResult:
        """Speak one facilitation turn, or wait -- listening -- for the answer.

        The turn is spoken and the hook returns; the answer comes back through
        on_utterance. The alternative, ctx.ask, would hold this hook for the
        whole of a group's thinking time as well as for the model call that
        wrote the question, and a hook that does not return is a robot the
        operator cannot switch away from.
        """
        store = ctx.store

        if store.get("awaiting"):
            return self._wait_for_answer(ctx)

        # Enough to work with the first time round; a good deal more before
        # volunteering a second set, because by then they have already had
        # directions and what they want is to keep talking about those.
        enough = _ENOUGH_ANSWERS if not store.get("gave_directions") else _MAX_ANSWERS
        if store.get("answers", 0) >= enough:
            store["directions"] = []
            store["stage"] = _BRIDGE
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        question = self._next_question(ctx)

        # Set before speaking, not after: say() raises DemoStopped if the
        # operator switches away mid-question, and it unwinds through here, so
        # the store has to already know a question is outstanding.
        store["awaiting"] = True
        store["waited"] = 0
        self._ask_aloud(ctx, question)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def _next_question(self, ctx: DemoContext) -> str:
        """What to say next -- generated from the session, or a fallback.

        Returned rather than spoken so that the one place that speaks a
        question also does the bookkeeping around it (see _ask_aloud). The
        model's own turns are spoken by ctx.reply as they stream, so this
        returns "" in that case and the caller has nothing left to say.
        """
        store = ctx.store
        turns = store["turns"]
        if not turns:
            return _OPENER

        last = turns[-1]
        if last[0] != "them":
            # We spoke last and got nothing back -- a nudge that lapsed, or a
            # resumed session. There is no new material to react to, so calling
            # the model would only ask it to rewrite its own last question.
            return self._fallback(ctx)

        closing = store.get("answers", 0) >= _ENOUGH_ANSWERS - 1
        # The message is what the group is taken to have said, so it is their
        # own words and nothing else; everything else goes in the system layer.
        # cache=False: this is one group's session, and a cached turn would be
        # replayed verbatim to the next group standing here.
        spoken = ctx.reply(
            _tidy(last[1]),
            person_id=store["person_id"],
            system=_facilitator_brief(turns, store["asked"], closing),
            cache=False,
        )
        if not spoken:
            # Counted anyway. Retrying inside a hook only asks a backend that is
            # already failing to fail again, and the runner sets a demo aside
            # after three of those.
            ctx.status("The next question came back empty.")
            return self._fallback(ctx)
        self._append(ctx, "me", spoken)
        self._remember_question(ctx, spoken)
        return ""

    def _fallback(self, ctx: DemoContext) -> str:
        """A blunt question, when a generated one could not be had."""
        store = ctx.store
        index = store.get("fallbacks", 0)
        store["fallbacks"] = index + 1
        return _FALLBACK_QUESTIONS[index % len(_FALLBACK_QUESTIONS)]

    def _ask_aloud(self, ctx: DemoContext, question: str) -> None:
        """Speak a question we wrote ourselves, and hold the mic for its answer.

        "" means ctx.reply already spoke it; the hold and the transcript entry
        still apply, which is why they live here rather than at each call site.
        """
        if question:
            self._append(ctx, "me", question)
            self._remember_question(ctx, question)
            ctx.say(question, "curious")
        # After speaking, never before: held while the robot is still talking,
        # the microphone would be open for its own voice.
        self._hold(ctx, True)

    def _wait_for_answer(self, ctx: DemoContext) -> IdleResult:
        """Let a pause be a pause, up to a point.

        A robot that re-asks every three seconds in a room full of people is
        worse than one that waits. But a group that has gone quiet for fifteen
        seconds has usually not understood that it was their turn, or has
        nothing on this particular question, and waiting forever is how the
        session ends without anybody deciding to end it.
        """
        store = ctx.store
        store["waited"] = store.get("waited", 0) + 1
        if store["waited"] < _MAX_WAITS:
            ctx.status("Waiting for an answer.")
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

        store["awaiting"] = False
        store["waited"] = 0
        if store.get("answers", 0) >= 1:
            # They have said enough to work with. Better to give them something
            # back than to keep asking a room that has stopped answering.
            self._hold(ctx, False)
            store["stage"] = _BRIDGE if not store.get("gave_directions") else _DONE
            return IdleResult(listen_for=_BETWEEN_LINES_S)

        nudges = store.get("nudges", 0) + 1
        store["nudges"] = nudges
        if nudges > _MAX_NUDGES:
            self._hold(ctx, False)
            store["stage"] = _DONE
            ctx.say("I'll leave that with you. Say let's brainstorm when you want another go.", "neutral")
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

        store["awaiting"] = True
        self._ask_aloud(ctx, _NUDGE)
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def _record(self, ctx: DemoContext, answer: str) -> None:
        """Take one turn from the group and let the next slice react to it."""
        store = ctx.store
        text = answer.strip()
        if not text:
            return
        self._append(ctx, "them", text)
        store["awaiting"] = False
        store["waited"] = 0
        store["nudges"] = 0
        # Let go the moment the answer lands. The next question re-takes it, so
        # the microphone is open across the whole exchange and shut across the
        # gaps -- which is the difference between a group being able to talk to
        # the robot and the robot transcribing the room all afternoon.
        self._hold(ctx, False)
        if _has_substance(text):
            store["answers"] = store.get("answers", 0) + 1
        # A nod instead of "got it": spoken acknowledgement between two
        # questions is a third of the session spent saying nothing.
        ctx.motion.acknowledge()
        ctx.status(f"Brainstorm: {store.get('answers', 0)} answer(s) in.")

    def _wants_ideas(self, ctx: DemoContext, text: str) -> bool:
        """"What do you think?" -- stop interviewing and deliver."""
        store = ctx.store
        stage = store.get("stage")
        if stage not in (_TALK, _DONE):
            # Mid-generation the directions are already on their way, and
            # mid-summary it is answered by the summary itself.
            return False
        self._append(ctx, "them", text)
        store["awaiting"] = False
        store["waited"] = 0
        if not store.get("answers"):
            # Set before speaking, and held after: say() can unwind through
            # here, and a hold taken first would be a microphone left open by a
            # demo that is no longer running.
            store["awaiting"] = True
            self._ask_aloud(ctx, "Give me the idea first and I'll have something to aim at.")
            return True
        self._hold(ctx, False)
        store["directions"] = []
        store["stage"] = _BRIDGE
        return True

    # --- the directions --------------------------------------------------

    def _offer_direction(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        directions = store["directions"]
        ordinal = _ORDINALS[len(directions)]
        spoken = ctx.reply(
            f"Give me the {ordinal} direction.",
            person_id=store["person_id"],
            system=_direction_brief(store["turns"], ordinal),
            cache=False,
        )
        if not spoken:
            ctx.status("A direction came back empty.")
        directions.append(spoken)
        if spoken:
            self._append(ctx, "me", spoken)
        store["touched"] = time.monotonic()
        if len(directions) >= len(_ORDINALS):
            store["gave_directions"] = True
            store["answers"] = 0
            store["stage"] = _TALK
            store["awaiting"] = True
            # It ends on a question rather than a conclusion because the useful
            # output here is what the group tests on Monday, and a robot that
            # picks the winner has taken the interesting part away. Asked as a
            # normal turn, so whatever they say back keeps the session going
            # instead of dropping them out of it.
            self._ask_aloud(
                ctx,
                "Which of those would you test first, and what's the cheapest way to find out?",
            )
            return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)
        # A direction is a couple of sentences and there are three of them back
        # to back. Between each one the group gets a window to cut in, which in
        # a session about their own idea is the whole point.
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    # --- the recap -------------------------------------------------------

    def _summarise_now(self, ctx: DemoContext) -> bool:
        """Handle "summarise that", wherever the session has got to."""
        store = ctx.store
        if not store.get("turns"):
            ctx.say("Nothing to sum up yet. Give me the idea first.", "curious")
            return True
        stage = store.get("stage")
        # Mid-session this is a re-orienting move, not an exit: the conversation
        # resumes at the stage it was in, and because awaiting was cleared the
        # next slice writes a fresh question rather than waiting on one nobody
        # can now remember.
        store["resume_stage"] = _DONE if stage in (_SUMMARY, _DONE) else stage
        store["awaiting"] = False
        store["waited"] = 0
        self._hold(ctx, False)
        store["stage"] = _SUMMARY
        return True

    def _summarise(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        spoken = ctx.reply(
            "Where has this got to?",
            person_id=store["person_id"],
            system=_summary_brief(store["turns"], store.get("directions") or []),
            cache=False,
        )
        if spoken:
            self._append(ctx, "me", spoken)
        else:
            ctx.status("The summary came back empty.")
        resumed = store.pop("resume_stage", _DONE)
        store["stage"] = resumed
        if resumed == _DONE:
            # A visible full stop, so a group can tell the session ended rather
            # than stalled.
            ctx.motion.express_move("happy")
        return IdleResult(listen_for=_BETWEEN_LINES_S)

    # --- session state ---------------------------------------------------

    def _append(self, ctx: DemoContext, who: str, text: str) -> None:
        """Add one turn to the session, keeping the tail that fits in a prompt."""
        turns = ctx.store.setdefault("turns", [])
        turns.append((who, text.strip()))
        if len(turns) > _TRANSCRIPT_TURNS * 2:
            del turns[:-_TRANSCRIPT_TURNS]
        ctx.store["touched"] = time.monotonic()

    def _remember_question(self, ctx: DemoContext, question: str) -> None:
        """Keep the questions already asked, so none of them is asked twice.

        Trimmed to what the prompt actually quotes back. A session that runs
        for half an hour otherwise accumulates a list nothing reads past the
        tail of.
        """
        asked = ctx.store.setdefault("asked", [])
        asked.append(question)
        if len(asked) > _ASKED_KEPT * 2:
            del asked[:-_ASKED_KEPT]

    def _hold(self, ctx: DemoContext, held: bool) -> None:
        """Hold the microphone open for one exchange, or let it go.

        Tracked in the store as well as set, so the release in on_exit is not a
        blind write over an operator setting -- hold_open_mic is a separate flag
        from the dashboard switch precisely so a demo never touches theirs, and
        this keeps that true in the one direction the API cannot enforce.
        """
        if bool(ctx.store.get("holding")) == bool(held):
            return
        ctx.store["holding"] = bool(held)
        ctx.state.hold_open_mic(bool(held))

    def _is_opening_trigger(self, ctx: DemoContext, lowered: str) -> bool:
        """Whether this is the phrase that just started the session.

        Only ever true before anyone has said anything, which is what stops a
        group saying "let's brainstorm the pricing" halfway through from having
        that turn swallowed instead of answered.
        """
        if ctx.store.get("turns"):
            return False
        return any(trigger in lowered for trigger in self.triggers)

    def _reset(self, ctx: DemoContext) -> None:
        """Start a clean session, keeping nothing from the last group."""
        holding = bool(ctx.store.get("holding"))
        ctx.store.clear()
        ctx.store.update(
            stage=_TALK,
            #: (who, what) in order, where who is "me" or "them". The session
            #: itself: every question, every direction and any recap is written
            #: from this rather than from named slots, which is what lets the
            #: robot follow a group that answers three questions in one breath.
            turns=[],
            #: The robot's own questions, quoted back to it so it does not ask
            #: the same thing twice in different words -- the single most
            #: noticeable failure of a model facilitating without a script.
            asked=[],
            directions=[],
            #: Group turns carrying content, not every turn. See _has_substance.
            answers=0,
            awaiting=False,
            waited=0,
            nudges=0,
            fallbacks=0,
            gave_directions=False,
            holding=holding,
            touched=time.monotonic(),
            # Fixed for the whole session. person_id() flickers between a
            # recognised id and 0 as a face comes and goes, and the no-repeat
            # rule for the directions only works if all three model calls land
            # in one person's history -- brain/memory.py keys on exactly this.
            person_id=ctx.person_id(),
        )
