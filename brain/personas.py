"""Distinct personalities the robot can answer in, for the personality demo.

The demo only works if a visitor asking the SAME question twice gets visibly
different answers, so a persona is more than a change of adjective in the
prompt. Each one carries three things that a listener can actually perceive:

- `prompt`: a behavioural instruction, layered on top of the standing persona
  in prompts.py rather than replacing it. It says what to DO with an answer --
  hedge it, warm it up, turn it into a question -- not merely what tone to
  affect, because "be professional" alone produces the same reply with longer
  words.
- `pace` and `variation`: abstract voice character, mapped to a synthesis
  config by body/audio_io.py::_voice_config and reaching it through
  ctx.say/ctx.reply. Kept abstract here on purpose: brain/ stays
  hardware-independent, so nothing in this file imports piper.
- `pose`: the resting expression, so the body language shifts with the voice.

Adding a persona is adding an entry to PERSONAS. Nothing else needs editing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Persona:
    """One answering style. `id` is what the dashboard and demos refer to."""

    id: str
    label: str
    blurb: str
    prompt: str
    #: Speaking rate multiplier. 1.0 is the synthesiser's own pace; higher is
    #: slower. Kept inside 0.95-1.15 -- past that it stops reading as character
    #: and starts reading as a fault. Spread across most of that range on
    #: purpose: an earlier set sat between 0.98 and 1.06, which measured as a
    #: 3% difference in the length of the same sentence and is not something
    #: anyone can hear.
    pace: float
    #: How much prosody varies. The synthesiser's default is ~0.667; more
    #: sounds animated, less sounds level.
    variation: float
    #: Resting expression from emotion.VALID_EMOTION_TAGS.
    pose: str


PROFESSIONAL = Persona(
    id="professional",
    label="Professional",
    blurb="Precise and measured. Answers narrowly, and says what it is unsure of.",
    prompt=(
        "Answer in a precise, measured, professional register. Lead with the direct "
        "answer in your first sentence, then at most one sentence of qualification. "
        "Prefer the specific word to the enthusiastic one. State the limits of what "
        "you know rather than smoothing over them, and do not use exclamation marks "
        "or say how interesting the question is."
    ),
    pace=1.10,
    variation=0.55,
    pose="neutral",
)

FRIENDLY = Persona(
    id="friendly",
    label="Friendly",
    blurb="Warm and conversational. Explains with everyday comparisons.",
    prompt=(
        "Answer warmly, the way you would to someone you like who is new to the "
        "topic. Use everyday comparisons instead of technical vocabulary, address "
        "the person as 'you', and let some enthusiasm through. Keep it short enough "
        "that it still feels like conversation rather than a lecture."
    ),
    pace=0.95,
    variation=0.90,
    pose="happy",
)

CONSULTANT = Persona(
    id="consultant",
    label="Consultant",
    blurb="Turns the question back into options and trade-offs, and asks what matters most.",
    prompt=(
        "Answer like a consultant. Do not simply state a position: frame the answer "
        "as the two or three real options with the trade-off between them, and say "
        "which one you would take and why. Finish by asking the one question whose "
        "answer would most change your recommendation."
    ),
    pace=1.03,
    variation=0.68,
    pose="thinking",
)

#: Order here is the order offered to visitors.
PERSONAS: tuple[Persona, ...] = (PROFESSIONAL, FRIENDLY, CONSULTANT)

BY_ID: dict[str, Persona] = {p.id: p for p in PERSONAS}

DEFAULT_PERSONA = PROFESSIONAL


def get(persona_id: str) -> Persona:
    """Return a persona by id, falling back to the default rather than raising.

    A demo mid-session in front of visitors should not die because a stale id
    arrived from the dashboard.
    """
    return BY_ID.get(persona_id, DEFAULT_PERSONA)


def next_after(persona_id: str) -> Persona:
    """The next persona in order, wrapping. Used to cycle by voice command."""
    ids = [p.id for p in PERSONAS]
    try:
        index = ids.index(persona_id)
    except ValueError:
        return PERSONAS[0]
    return PERSONAS[(index + 1) % len(PERSONAS)]
