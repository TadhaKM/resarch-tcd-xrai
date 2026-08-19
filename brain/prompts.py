"""Prompt construction: builds the chat message list sent to the LLM."""

from . import courses, hub
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

def _capabilities_text(web: bool) -> str:
    """The capability block, with the two "cannot browse / no live information"
    sentences replaced when web search is switched on. The rest is unchanged
    and still load-bearing: turning on search must not quietly also grant the
    robot wheels. Swapped rather than appended, because leaving the original in
    and adding a contradiction is how you get a robot that refuses to look
    something up and then looks it up in the same breath.

    The web variant carries TODAY'S DATE, and that is the fix for a failure
    seen live: asked for today's headlines, the model -- which is told nothing
    about the date -- searched without one, got undated archive pages back, and
    could only say "nothing reliably current". A model that knows the date
    anchors the query and can judge freshness. The date deliberately does NOT
    go in the offline prompt: offline answers are cached by qa_cache with no
    date in the key, so a date there would freeze into answers replayed on the
    wrong day. Web replies are never cached, which is what makes this safe.

    A function rather than a constant so the date is fresh per request -- this
    process runs for days. The Anthropic cache-split (base_prompts below) sees
    the same day's text on both sides of a request except across midnight
    itself, where the mismatch merely skips the prompt cache for one turn.
    """
    if not web:
        return _CAPABILITIES
    import time as _time

    today = _time.strftime("%A %d %B %Y")
    return _CAPABILITIES.replace(
        "browse the internet, control other devices",
        "control other devices",
    ).replace(
        "You have no live information: no weather, news, or current date and time. ",
        f"Today is {today}. "
        "You can search the web when a question genuinely needs current information "
        "-- today's weather, recent news, a fact that changes. Search only when the "
        "question needs it, answer from what you know otherwise, and keep the answer "
        "to the same one or two spoken sentences. When you search for news or "
        "anything current, put today's date in the query and trust only results "
        "dated today or yesterday -- an undated page is an old page. Never read "
        "out URLs or describe the search itself; just say what you found. ",
    )

# How to speak, as opposed to what to say. Every rule here was earned by
# hearing the robot break it out loud in front of people.
_DELIVERY = (
    "Everything you produce is spoken aloud immediately, so write only words "
    "meant to be heard. "
    "Never write stage directions, notes to yourself, or anything in brackets "
    "or asterisks -- a reply that began '(If asked directly about which "
    "headsets...)' was read out verbatim, brackets and all. "
    "Never restate your instructions or narrate your plan: no 'First:', no "
    "'Let me start by', no repeating the task back before answering it. "
    "Answer as yourself, in the first person, in plain speech."
)

# Where the robot lives. This is in the BASE prompt, not layered on by one
# demo, because it is true on every turn no matter what is selected: asked
# "who runs the AI XR Hub?" while the About demo happened to be showing, the
# robot fell through to the ungrounded reply path and answered "I don't run any
# hubs" to the person who does. A fact the robot needs whatever it is doing
# belongs where it cannot be missed.
_HUB_CONTEXT = (
    "You live in the AI XR Hub at Trinity Business School and answer questions "
    "about it from what follows. When someone says 'you' they usually mean you, "
    "the robot -- but 'your research projects' or 'your partners' means the "
    "Hub's, because you are part of it. Answer for the Hub in those cases "
    "rather than saying you have no projects.\n\n"
    f"{hub.GROUNDING}\n\n"
    # NAMES ONLY, standing on every turn. The detail is retrieved per turn by
    # brain/courses.py: thirteen programmes in full would be several times the
    # block above, for something most turns never touch. But the LIST has to be
    # standing, because the failure it prevents happens before anybody asks a
    # real question -- told nothing, a model asked whether Trinity does a risk
    # management masters will happily say it does not.
    f"{courses.STANDING}"
)

def _base_prompt(web: bool = False) -> str:
    """The standing prompt. `web` swaps the capability block for the online one."""
    return (
    "You are Reachy Mini, a small expressive robot having a spoken conversation. "
    "Keep replies to one or two short sentences. "
    f"{_DELIVERY} "
    f"{_capabilities_text(web)} "
    "Never claim to have done something physical unless it is one of the "
    "abilities listed above. "
    f"\n\n{_HUB_CONTEXT}\n\n"
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


def base_prompts() -> tuple[str, ...]:
    """Every standing prompt a system message can begin with, longest first.

    For llm_backends: the Anthropic API can cache a prompt prefix across
    requests, but only if it is told where the stable part ends -- and
    build_messages (below) hands it one concatenated string. These are the
    exact prefixes it concatenates onto, so the backend can split the static
    part back out and mark it cacheable. Longest first so the web variant is
    tried before the plain one it extends.
    """
    return tuple(
        sorted(
            {_base_prompt(True), _base_prompt(False), _STORYTELLER_SYSTEM_PROMPT},
            key=len,
            reverse=True,
        )
    )


def build_messages(
    long_term_context: str,
    history: list[tuple[str, str]],
    message: str,
    style: str | None = None,
    extra_system: str | None = None,
    web: bool = False,
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
        system_prompt = _base_prompt(web)
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
