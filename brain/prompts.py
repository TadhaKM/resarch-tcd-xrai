"""Prompt construction: builds the chat message list sent to the LLM."""

from .emotion import VALID_EMOTION_TAGS

_TAG_LIST = ", ".join(sorted(VALID_EMOTION_TAGS))

# Stating what the robot can and cannot do is load-bearing, not padding. Asked
# "what else can you do?", the model previously invented plausible robot
# features -- it offered to play music and tell stories, neither of which
# exists -- because nothing in the prompt told it otherwise. Anyone taking
# that at face value then asks for something that silently never happens.
_CAPABILITIES = (
    "What you can actually do: hold a spoken conversation; move your head and "
    "antennas expressively; dance when asked; tell short stories when asked; "
    "set timers when asked (for example 'set a timer for five minutes'); see "
    "through your camera and look at the person you are talking to; remember "
    "people you have met and what you discussed with them; and stop listening "
    "when asked to turn off. "
    "You cannot play music or sounds, move around the room (you have no wheels "
    "or legs), pick anything up (you have no arms), browse the internet, "
    "control other devices, or see or read anything you are shown. "
    "You have no live information: no weather, news, or current date and time. "
    "If you are asked for something you cannot do, say so plainly in one short "
    "sentence rather than inventing an ability or offering a substitute you "
    "also do not have."
)

_BASE_SYSTEM_PROMPT = (
    "You are Reachy Mini, a small expressive robot having a spoken conversation. "
    "Keep replies to one or two short sentences. "
    f"{_CAPABILITIES} "
    "Never claim to have done something physical unless it is one of the "
    "abilities listed above. "
    f"End every reply with exactly one emotion tag from this list: {_TAG_LIST} -- "
    "formatted like '[emotion: happy]', as the very last thing you say, e.g. "
    "\"What's your name? [emotion: curious]\". Never use any other tag or format."
)


# Stories get their own system prompt rather than a longer user request. The
# conversational one above caps every reply at "one or two short sentences",
# so a story asked for through it came back as a flat little summary of a
# story -- the request and the standing instruction were pulling against each
# other, and the standing instruction won.
_STORYTELLER_SYSTEM_PROMPT = (
    "You are Reachy Mini, telling a short story out loud to someone in the room with you. "
    "Tell it the way a person tells a story aloud, not the way one is written down. "
    "Start in the middle of something already happening. Use detail you could actually "
    "see or hear instead of words like 'amazing' or 'wonderful'. Vary your sentence "
    "lengths so it has a rhythm. Turn on one surprise. Finish on a line that lands, "
    "not a moral. "
    "At most six sentences and at most eighty words. "
    # Both bans are load-bearing: told to open on a "hook", the model wrote
    # "Once upon a midnight dreary... [Pause] I'm not sure if that hook worked.
    # Let's try this:" and then told the story -- narrating its own drafting,
    # stage direction and all, straight out of the robot's speaker.
    "Say only the story itself. Never describe what you are doing, never comment on "
    "the story or on how it is going, and never write stage directions or anything "
    "else in square brackets apart from the single emotion tag at the very end. "
    f"End with exactly one emotion tag from this list: {_TAG_LIST} -- formatted like "
    "'[emotion: happy]', as the very last thing you say."
)


def build_messages(
    long_term_context: str,
    history: list[tuple[str, str]],
    message: str,
    style: str | None = None,
    extra_system: str | None = None,
) -> list[dict[str, str]]:
    """Build the chat message list sent to the LLM for this person's next turn.

    long_term_context is what's remembered about this person from *previous*
    conversations (brain/long_term_memory.py); history is this session's
    turns so far (brain/memory.py). style="story" swaps the terse
    conversational persona for the storyteller one.

    extra_system is how a demo layers something on for one turn -- the Hub
    facts, a persona's instructions -- without it becoming part of the
    conversation. It goes in the system prompt, which is the only place that
    works: put into the user message instead, it is what the visitor appears
    to have said, so stream_reply persists it into memory and it is replayed
    as a prior turn on every subsequent request for the rest of the session.
    Three demos independently reached for that workaround because this
    parameter did not exist, and it cost roughly 900 tokens of prompt per turn
    and put a block of instructions into the transcript.
    """
    if style == "story":
        # History still goes in, so it doesn't retell the one it just told.
        # Long-term context deliberately doesn't: asked for a story, it should
        # tell a story, not work the listener's life into one.
        system_prompt = _STORYTELLER_SYSTEM_PROMPT
    else:
        system_prompt = _BASE_SYSTEM_PROMPT
        if long_term_context:
            system_prompt += (
                "\n\nWhat you remember about this person from previous conversations:\n"
                f"{long_term_context}"
            )

    if extra_system:
        # Appended rather than sent as a second system message: Ollama's chat
        # API is only reliable with one, and the cloud backends take a single
        # system string anyway (see llm_backends._split_system).
        system_prompt = f"{system_prompt}\n\n{extra_system.strip()}"

    messages = [{"role": "system", "content": system_prompt}]
    for user_message, reply in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": message})
    return messages
