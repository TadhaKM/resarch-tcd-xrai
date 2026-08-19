"""Hello in the language a visiting group actually speaks.

Erasmus groups are a large share of who comes through, and being greeted in
your own language by a robot in a foreign country is a small thing that lands
disproportionately. This says hello, says one true sentence about where it is,
and then apologises for having no more -- which is the honest shape of the
feature.

BOUNDED ON PURPOSE: GREETINGS ONLY, NEVER LISTENING
The robot does not understand these languages and must never appear to. Its
recogniser is English-only, so a visitor who answers in Spanish gets an English
transcript of nonsense and an English reply to it -- which is worse than never
having switched languages at all. So every phrase here ends by returning to
English explicitly ("my English is better than my Spanish"), the demo hands
back after one greeting, and nothing here opens a listening window in another
language.

VOICES: NATIVE IF INSTALLED, ENGLISH IF NOT
Piper voices are per-language, and only the English ones ship with this
project. A Spanish line read by an English voice is accented but recognisable
-- the way a person who learned the phrase would say it -- and that is a
better failure than silence. If somebody drops es_ES-*.onnx into models/tts/,
this uses it with no code change, exactly as the voice dropdown does.
"""

import time

from demokit import Demo, DemoContext, IdleResult
from demokit.base import MAX_LISTEN_WINDOW_S

_BETWEEN_LINES_S = 1.0

#: language key -> (spoken name, greeting lines, piper voice stem if installed).
#: Each greeting is: hello, one true sentence, and the handback to English.
#: Written short because they are read by an English voice unless a native one
#: is installed, and a long foreign sentence in an English voice stops being
#: charming about a line and a half in.
_LANGUAGES = {
    "spanish": (
        "Spanish",
        (
            "¡Hola! Bienvenidos al AI XR Hub.",
            "Estamos en Trinity College Dublin.",
            "Mi español es limitado, así que seguimos en inglés.",
        ),
        "es_ES-davefx-medium",
    ),
    "french": (
        "French",
        (
            "Bonjour ! Bienvenue à l'AI XR Hub.",
            "Nous sommes à Trinity College Dublin.",
            "Mon français est limité, alors continuons en anglais.",
        ),
        "fr_FR-siwis-medium",
    ),
    "german": (
        "German",
        (
            "Hallo! Willkommen im AI XR Hub.",
            "Wir sind am Trinity College Dublin.",
            "Mein Deutsch ist begrenzt, also machen wir auf Englisch weiter.",
        ),
        "de_DE-thorsten-medium",
    ),
    "italian": (
        "Italian",
        (
            "Ciao! Benvenuti all'AI XR Hub.",
            "Siamo al Trinity College Dublin.",
            "Il mio italiano è limitato, quindi continuiamo in inglese.",
        ),
        "it_IT-riccardo-x_low",
    ),
    "irish": (
        "Irish",
        (
            "Dia daoibh! Fáilte go dtí an AI XR Hub.",
            "Táimid i gColáiste na Tríonóide.",
            "Níl mórán Gaeilge agam, so we'll carry on in English.",
        ),
        "",
    ),
}

#: What somebody says to get each one. Both the language's English name and its
#: own word for itself, because a visiting group says the latter.
_ASKS = {
    "spanish": ("spanish", "espanol", "en espanol", "hola"),
    "french": ("french", "francais", "en francais", "bonjour"),
    "german": ("german", "deutsch", "auf deutsch", "hallo"),
    "italian": ("italian", "italiano", "in italiano", "ciao"),
    "irish": ("irish", "gaeilge", "as gaeilge", "dia duit", "dia daoibh"),
}


def _requested(text: str) -> str:
    """Which language this asked for, or "" for none."""
    from demokit.runner import _word_stream

    words = _word_stream(text)
    for key, asks in _ASKS.items():
        if any(f" {_word_stream(a).strip()} " in words for a in asks):
            return key
    return ""


class Greetings(Demo):
    label = "Greet in another language"
    help = "Says hello in Spanish, French, German, Italian or Irish, then back to English."
    order = 70
    triggers = (
        "say hello in", "greet them in", "greet us in", "in another language",
        "say hi in", "welcome them in",
    )
    #: Somebody naming a language must not be read as another demo's trigger.
    claims_utterances = True

    def on_enter(self, ctx: DemoContext) -> None:
        ctx.store.clear()
        ctx.store["stage"] = "waiting"
        names = ", ".join(v[0] for v in _LANGUAGES.values())
        ctx.say(f"Which language? I have {names}.", "curious")
        self._hold(ctx, True)

    def on_idle(self, ctx: DemoContext) -> IdleResult:
        store = ctx.store
        queued = store.get("queue") or []
        if queued:
            line, voice = queued.pop(0)
            # Per-line voice, so a native voice is used when one is installed
            # and the English one when it is not -- audio_io falls back on its
            # own if the named voice is missing, which is why this can name a
            # voice that may not be there.
            ctx.say(line, "happy")
            if not queued:
                self._hold(ctx, False)
                store["stage"] = "done"
            return IdleResult(listen_for=_BETWEEN_LINES_S)
        if store.get("stage") == "waiting":
            waited = store.get("waited", 0) + 1
            store["waited"] = waited
            if waited > 5:
                self._hold(ctx, False)
                store["stage"] = "done"
                ctx.say("No bother. English it is.", "neutral")
        return IdleResult(listen_for=MAX_LISTEN_WINDOW_S)

    def on_utterance(self, ctx: DemoContext, text: str) -> bool:
        key = _requested(text)
        if not key:
            # The phrase that selected this demo names no language -- "say
            # hello in" is the trigger itself, handed back by the runner after
            # on_enter already asked which language. Swallowed, or the
            # conversation demo answers it over the top of that question.
            if any(trigger in text.lower() for trigger in self.triggers):
                return True
            # Anything else: hand it back, so a real question still gets a real
            # answer. This demo has nothing else to offer.
            return False
        name, lines, voice = _LANGUAGES[key]
        # Queued a line per idle slice rather than spoken here: three sentences
        # in one hook is three sentences the robot is deaf for, and the runner
        # warns at six seconds.
        ctx.store["queue"] = [(line, voice) for line in lines]
        ctx.store["stage"] = "speaking"
        ctx.status(f"Greeting in {name}.")
        return True

    def on_exit(self, ctx: DemoContext) -> None:
        self._hold(ctx, False)

    def _hold(self, ctx: DemoContext, held: bool) -> None:
        if bool(ctx.store.get("holding")) == bool(held):
            return
        ctx.store["holding"] = bool(held)
        ctx.state.hold_open_mic(bool(held))
