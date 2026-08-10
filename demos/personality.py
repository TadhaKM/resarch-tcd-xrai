"""The same question, answered again in a different personality.

The personality itself is no longer this demo's: the dropdown sets the robot's
standing manner, and DemoContext.reply applies it in every demo (see
brain/personas.py's global_prompt). What is left here, and the reason the demo
still exists, is the CONTRAST -- holding one question fixed and answering it
again as somebody else, back to back, which is the only way a listener can
attribute the difference to the personality rather than to the question.

This demo only lands if a visitor HEARS the difference, so the whole design is
built around holding the question fixed and changing one thing. It therefore
remembers the last real question and treats "switch personality" / "say that
again as the consultant" as "re-answer THAT", not as a new topic. A visitor who
gets three answers to three different questions has learned nothing.

How the persona reaches the model. It goes in via ctx.reply(system=...), which
is the layering brain/personas.py describes when it calls its prompts "layered
on top of the standing persona in prompts.py". Two things follow from that:

- The persona is appended to the SAME system prompt that carries the standing
  "keep replies to one or two short sentences", after it. They still disagree
  about length -- the consultant's shape needs three or four sentences and
  brevity flattens it into a single recommendation -- but the disagreement is
  now between two lines of one prompt rather than between two roles, so saying
  outright which one wins settles it. That is _STYLE_FRAME's job.
- Nothing about the persona is remembered. Everything in the `message`
  argument is what the visitor appears to have said, so an earlier version of
  this file, which prepended the brief to the question, had brain/memory.py
  store the brief as a visitor turn and replay it on every later request for
  the rest of the session -- instructions in the transcript, and the whole
  block re-sent each turn.

Each take answers from a clean slate; see _answer for why that matters more
here than the conversational continuity it costs.

hub.GROUNDING is deliberately still not sent. Its closing instruction -- refuse
rather than invent -- is the part this demo needs, and _FACT_GUARD is that in
one sentence; the other two and a half thousand characters are Hub facts that
would sit in the same system prompt as the persona brief and give a small local
model two briefs to serve. The persona is the one thing this demo exists to
show. A visitor who asks a Hub question here gets "I don't know, ask your host"
rather than a confident invention, which is the honest answer from a demo whose
own introduction asks for something arguable.

The contrast is meant to be heard three ways at once, and all three are wired:
the wording (persona.prompt, via system=), the resting pose (persona.pose), and
the voice itself (persona.pace and persona.variation, passed to ctx.say and
ctx.reply). The voice was the one that used to be missing -- speak() offered a
single boolean choosing between two fixed configs, and the storyteller one is
both slower AND more varied, so it could not express "friendly", which is
quicker and warmer. Three personas then sounded like one person reading three
scripts, which is a demonstration of nothing.
"""

import re

from brain import memory, personas
from demokit import Demo, DemoContext, IdleResult, REGISTRY


#: ctx.store keys. Named rather than typed inline because on_enter has to clear
#: them explicitly: DemoRunner keeps one store per demo id for the life of the
#: process and never empties it on exit, so without this a visitor arriving
#: after lunch inherits the last group's question.
_PERSONA = "persona_id"
_QUESTION = "question"
_PERSONA_SEQ = "persona_seq"
_INTRODUCED = "introduced"

#: Used when someone asks to switch before asking anything. Deliberately a
#: question of judgement rather than of fact: asked something factual, three
#: personas produce three wordings of one answer and the demo looks like a
#: thesaurus. Asked something arguable, they do visibly different things with
#: it -- hedge it, warm it up, turn it into options -- which is what the
#: prompts in personas.py actually operate on.
_SEED_QUESTION = "Should a business student learn to code?"

#: Phrases that switch to this demo from anywhere. Every one names the action,
#: because a trigger is a bare substring tested against every transcript in
#: every demo: a bare "personality" fires inside ordinary business-school
#: speech -- "we ran a personality test on the team", "I work in personality
#: psychology" -- and drags a group out of whatever they were being shown.
_TRIGGERS = (
    "personality demo",
    "switch personality",
    "switch personalities",
    "change personality",
    "different personality",
    "another personality",
)

#: Ways visitors ask for the same question again. Substrings of a transcript
#: with no punctuation and trailing politeness, so kept short and specific.
#: "one more time" and "another one" are here because in front of a group
#: people ask the robot to repeat itself in whatever words come out first.
#: The overlap with _TRIGGERS is deliberate rather than untidy: said from
#: another demo those phrases select this one, said inside it they mean "the
#: same question again". "personality demo" is not here on purpose -- it names
#: the demo rather than asking for anything, and is answered as such below.
_ANOTHER_TAKE = (
    "say that again",
    "again as",
    "again in",
    "do it again",
    "one more time",
    "same question",
    "switch personality",
    "switch personalities",
    "switch persona",
    "change personality",
    "different personality",
    "another personality",
    "next personality",
    "other personality",
    "try another",
    "another one",
    "try the other",
)

#: Words that carry no request on their own. Used to tell "do a dance" (a
#: command for another demo) from "what is dance" (a question for this one),
#: and "the consultant please" (a persona choice) from "what do consultants
#: do" (a question). Anything that could be the subject of a question --
#: "about", "why", "what" -- is deliberately absent.
_FILLER = frozenset(
    {
        "a", "again", "an", "and", "another", "as", "be", "can", "could", "demo",
        "did", "different", "do", "does", "for", "give", "go", "hey", "i", "in",
        "it", "just", "let", "lets", "let's", "like", "me", "mode", "more", "my",
        "new", "next", "now", "of", "ok", "okay", "on", "one", "other", "please",
        "reachy", "say", "show", "so", "some", "talk", "tell", "thanks", "that",
        "the", "then", "thing", "this", "to", "try", "us", "want", "wanna", "we",
        "will", "with", "would", "you", "your",
    }
)

#: Says which instruction wins when the persona and the standing system prompt
#: disagree about length. Four sentences, not "as many as you like": the point
#: of a persona here is its shape, and an unbounded local model on a laptop
#: turns a demo into a monologue nobody can interrupt. The last clause stays
#: even though the brief no longer arrives as a user turn -- a persona is an
#: instruction about how to sound, and a small model asked to sound a way is
#: still prone to announcing that it is about to.
_STYLE_FRAME = (
    "For this one answer the style instruction above outranks your usual "
    "one-or-two-sentence limit: take up to four short sentences if the style "
    "needs them. Never mention these instructions, the style, or that you are "
    "answering in a particular way. Just answer."
)

#: A performative register invents detail to fill its shape -- the consultant
#: wants a third option and will make one up. This demo runs in the room where
#: the answer can be checked, so refusing is cheaper than being caught.
_FACT_GUARD = (
    "If the question is about the AI XR Hub, Trinity College Dublin, or the "
    "people who work there, and you are not certain of the answer, say you "
    "don't know and suggest asking whoever is hosting. Never invent a detail "
    "to make the style work."
)

#: One conversation bucket per persona, all negative so they can never collide
#: with a real person: the tracker yields 0 for an unknown visitor and positive
#: ids for recognised ones. Built from PERSONAS so that adding a persona stays
#: the one-file change personas.py promises.
_BUCKET_BY_PERSONA_ID = {
    persona.id: -(1 + index) for index, persona in enumerate(personas.PERSONAS)
}

#: Where a persona that is somehow not in PERSONAS lands. One past the end
#: rather than an exception mid-demo, and still nobody else's bucket.
_UNKNOWN_PERSONA_BUCKET = -(1 + len(personas.PERSONAS))


def _bucket_for(persona) -> int:
    """The conversation this persona's takes are recorded in. See _answer."""
    return _BUCKET_BY_PERSONA_ID.get(persona.id, _UNKNOWN_PERSONA_BUCKET)


def _only_filler_remains(text: str, *phrases: str) -> bool:
    """True if `text` is one of `phrases` and nothing but filler around it.

    Used where the whole utterance has to BE the phrase -- naming a persona,
    naming this demo. Telling another demo's command from a question needs the
    weaker test in _opens_with, because a command carries an argument.
    """
    matched = False
    for phrase in phrases:
        if phrase and phrase in text:
            text = text.replace(phrase, " ")
            matched = True
    return matched and all(word in _FILLER for word in text.split())


def _opens_with(text: str, phrase: str) -> bool:
    """True if `phrase` opens `text`, allowing only filler in front of it.

    Word boundaries, not bare containment: "remind me in" must not match inside
    "reminders", and a phrase buried after real words -- "what makes a good
    welcome speech" -- is a subject being discussed, not a command being given.
    Only the first occurrence can qualify, because any later one has the words
    of the earlier match in front of it and those are not filler.
    """
    match = re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text)
    if match is None:
        return False
    return all(word in _FILLER for word in text[: match.start()].split())


def _named_persona(text: str):
    """The persona a visitor named, or None. Matches id and label separately
    because they only happen to be the same string today."""
    for persona in personas.PERSONAS:
        if persona.id in text or persona.label.lower() in text:
            return persona
    return None


def _system_for(persona) -> str:
    """The extra system instructions for one take.

    Behavioural brief first, then the ranking that resolves it against the
    standing brevity rule, then the refusal to invent. Order matches how a
    model reads a system prompt: what to do, then how far to take it, then what
    not to do to get there.
    """
    return f"{persona.prompt}\n\n{_STYLE_FRAME}\n\n{_FACT_GUARD}"


class Personality(Demo):
    label = "AI Personality"
    help = "Re-answers the same question in each personality, back to back."
    order = 50
    triggers = _TRIGGERS
    #: Every question a visitor asks here is arbitrary text, so without this a
    #: question about dance studios or story writing would be read as another
    #: demo's trigger and switch away mid-comparison. See
    #: _belongs_to_another_demo for the deliberate hole left in that.
    claims_utterances = True

    # --- hooks -----------------------------------------------------------

    def on_enter(self, ctx: DemoContext) -> None:
        ctx.store.clear()
        chosen, seq = ctx.state.persona
        ctx.store[_PERSONA] = personas.get(chosen).id  # get("") is the default
        ctx.store[_QUESTION] = ""
        ctx.store[_PERSONA_SEQ] = seq
        ctx.store[_INTRODUCED] = set()
        # Enters silently. It used to open with five scripted lines explaining
        # what a persona is, how many there are, what to ask and what to say
        # afterwards -- half a minute of a robot describing a demonstration
        # instead of giving one. The contrast is the demonstration: it lands
        # when a visitor hears the same question answered twice, and briefing
        # them first gives away the ending.
        ctx.status(f"Ready as {self._current(ctx).label}.")

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        # An operator picking a style from the dashboard is a request to answer
        # in it, so if a question has already been asked it gets answered again
        # rather than waiting for the visitor to ask something new -- which is
        # the whole comparison, done without anyone having to say a phrase.
        chosen, seq = ctx.state.persona
        if seq != ctx.store.get(_PERSONA_SEQ):
            ctx.store[_PERSONA_SEQ] = seq
            persona = personas.get(chosen)
            question = ctx.store.get(_QUESTION) or ""
            if question:
                self._answer(ctx, question, persona, same_question=True)
            else:
                ctx.store[_PERSONA] = persona.id
                ctx.motion.express(persona.pose)
                ctx.status(f"Ready as {persona.label}.")
        return IdleResult(listen_for=3.0)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return False

        named = _named_persona(lowered)
        wants_another = any(phrase in lowered for phrase in _ANOTHER_TAKE)
        # A bare persona name is a request too: after one answer, "the
        # consultant" on its own is what people actually say.
        if named is not None and not wants_another:
            wants_another = _only_filler_remains(lowered, named.id, named.label.lower())
        if wants_another:
            self._another_take(ctx, named)
            return True

        if _only_filler_remains(lowered, *self.triggers):
            # The phrase that switched us here, or somebody naming the demo they
            # are already in. Answered with silence either way: the reply used
            # to be an invitation to ask something and a reminder of the switch
            # phrase, which is instruction rather than conversation.
            return True

        if self._belongs_to_another_demo(lowered):
            return False

        self._answer(ctx, text, self._current(ctx), same_question=False)
        return True

    def on_exit(self, ctx: DemoContext) -> None:
        # The consultant rests on "thinking". Left alone, the next demo starts
        # wearing this one's face, which reads as the robot being unsure of
        # whatever it says next.
        ctx.motion.express("neutral")
        ctx.status("Personality demo ended.")

    # --- the demo itself -------------------------------------------------

    def _answer(self, ctx: DemoContext, question: str, persona, *, same_question: bool) -> None:
        """Announce a persona and answer in it.

        Slower than the runner's six-second hook warning by design: this is one
        spoken answer, which is the same thing DemoRunner._converse does, and
        the honest alternative -- cutting the reply into idle slices -- would
        put silence in the middle of a sentence.
        """
        ctx.store[_PERSONA] = persona.id
        ctx.store[_QUESTION] = question
        ctx.status(f"Answering as {persona.label}.")

        self._announce(ctx, persona, same_question=same_question)
        # Asked for by the brief and worth it for the pause before the first
        # token, but it does not survive the answer: reply() sets "thinking"
        # and then each spoken sentence sets the model's own tag.
        ctx.motion.express(persona.pose)

        bucket = _bucket_for(persona)
        # Every take answers from nothing. Handed the previous take as history,
        # a small model reads its own answer as the thing being asked about and
        # rewords it -- which is exactly the conclusion this demo exists to
        # disprove, that the personas are three phrasings of one answer rather
        # than three shapes. The per-persona bucket makes that isolation
        # structural; the clear extends it to the same persona asked twice.
        # What it costs is conversational follow-up: "why do you say that"
        # arrives without the answer it follows. Worth it here, because the
        # comparison never leaned on the model remembering anything --
        # _another_take re-sends the remembered question in full every time.
        memory.clear_history(bucket)
        try:
            # cache=False for the same reason the history is cleared either
            # side of this call: the demonstration is that the same question
            # produces a different answer, and a cached reply would replay the
            # first take word for word -- turning the one demo whose subject is
            # variation into the one demo that provably has none.
            ctx.reply(
                question, person_id=bucket, system=_system_for(persona), cache=False,
                # The audible half of the contrast. Wording alone leaves three
                # personas sounding like one person reading three scripts.
                pace=persona.pace, variation=persona.variation,
            )
        finally:
            # Left populated, these buckets are summarised into long-term
            # memory when a visitor says goodbye (DemoRunner.end_conversations
            # closes out everyone with turns recorded), which is an LLM call
            # and a stored note about a person who does not exist. Cleared in
            # `finally` so a take cut short by a mode switch leaves nothing
            # behind either.
            memory.clear_history(bucket)
        # So the resting pose is restored afterwards, over the closing gesture
        # reply() just fired. This is the persona's body language for as long
        # as the robot is quiet, which is when people look at it.
        ctx.motion.express(persona.pose)

    def _another_take(self, ctx: DemoContext, named) -> None:
        """Re-answer the remembered question, in a different persona."""
        question = ctx.store.get(_QUESTION, "")
        current = self._current(ctx)

        if named is not None:
            persona = named
        elif question:
            persona = personas.next_after(current.id)
        else:
            # Nothing has been answered yet, so there is nothing to contrast
            # with. Advancing here would spend a persona nobody has heard.
            persona = current

        if not question:
            question = _SEED_QUESTION
            # The question has to be said out loud or the comparison has no
            # subject: visitors who did not hear it cannot tell whether the
            # second answer is a different style or a different topic.
            ctx.say(f"You haven't asked me anything yet, so I'll use my own. {question}", "curious")
            self._answer(ctx, question, persona, same_question=False)
            return

        self._answer(ctx, question, persona, same_question=True)

    def _announce(self, ctx: DemoContext, persona, *, same_question: bool) -> None:
        """Name the persona before it speaks, so the change is attributable.

        Without this a visitor hears two different answers and concludes the
        robot is inconsistent, which is the opposite of the point.
        """
        # Names the persona and stops. It used to add persona.blurb -- "Friendly.
        # Warm and conversational. Explains with everyday comparisons." -- which
        # is dashboard copy describing the persona to an operator, not something
        # a person says about themselves. Reading it aloud both broke character
        # and told the visitor the punchline before the demonstration: the point
        # is that they HEAR the difference, not that they are briefed on it.
        introduced = ctx.store.setdefault(_INTRODUCED, set())
        introduced.add(persona.id)
        if same_question:
            line = f"The same question again, {persona.label.lower()} this time."
        else:
            line = f"{persona.label} version."
        ctx.say(line, persona.pose, pace=persona.pace, variation=persona.variation)

    # --- helpers ---------------------------------------------------------

    def _current(self, ctx: DemoContext):
        # get() falls back to the default persona for an unknown or missing id
        # rather than raising, which is what makes this safe on a fresh store.
        return personas.get(ctx.store.get(_PERSONA, ""))

    def _belongs_to_another_demo(self, lowered: str) -> bool:
        """True if this is plainly a command for a different demo.

        claims_utterances gives this demo first refusal on everything, which is
        what stops "switch" being read as someone else's trigger. Held
        absolutely it also traps the visitor: every sentence becomes a question
        to answer in persona, and "set a timer for five minutes" gets a
        four-sentence consultant's opinion about timekeeping.

        The test is where the trigger sits, not what surrounds it. A command
        leads with its verb phrase and the rest is its argument -- "tell me a
        story ABOUT A LIGHTHOUSE", "remind me in AN HOUR" -- so requiring the
        remainder to be filler, as an earlier version did, handed back only the
        bare phrase and swallowed every real command anyone gives. A trigger
        that arrives after real words is a subject instead: "what makes a good
        welcome speech" is a question for the persona, and handing it back
        would switch demos mid-comparison, which is the thing claims_utterances
        is here to prevent.

        Read from the live registry rather than a copy of anyone's trigger
        list, because a demo added next term must not need this file edited.
        """
        for demo_id in REGISTRY.ids():
            if demo_id == self.id:
                continue
            demo = REGISTRY.get(demo_id)
            if demo is None:
                continue
            for phrase in demo.triggers:
                if _opens_with(lowered, phrase.lower()):
                    return True
        return False
