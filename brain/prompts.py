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


def build_messages(
    long_term_context: str, history: list[tuple[str, str]], message: str
) -> list[dict[str, str]]:
    """Build the chat message list sent to the LLM for this person's next turn.

    long_term_context is what's remembered about this person from *previous*
    conversations (brain/long_term_memory.py); history is this session's
    turns so far (brain/memory.py).
    """
    system_prompt = _BASE_SYSTEM_PROMPT
    if long_term_context:
        system_prompt += (
            "\n\nWhat you remember about this person from previous conversations:\n"
            f"{long_term_context}"
        )

    messages = [{"role": "system", "content": system_prompt}]
    for user_message, reply in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": message})
    return messages
