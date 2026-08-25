"""Microphone/speaker access: wake word, streaming STT, and TTS.

In "simulation" mode, audio flows through this machine's real mic/speaker via
sounddevice -- see config.py's HardwareTarget docstring for why (this mirrors
reachy_mini_conversation_app's sim-vs-robot audio routing, translated for a
project with no browser in the loop). In "robot" mode, audio instead flows
through the daemon's own media pipeline (robot.media.start_recording()/
get_audio_sample()/push_audio_sample()) -- the same thing LocalStream does in
reachy_mini_conversation_app -- because this app runs ON the robot itself, so
there's no separate machine's mic/speaker to reach for.

This shares one ReachyMini media connection with Camera rather than opening a
second one -- see camera.py's docstring for why (the daemon's media backend
only supports one session per client).

_mic_frames() is the one place that branches on target.mode for input; both
wait_for_wake_word() and listen() consume it without caring where frames come
from, so the STT/KWS decode loops aren't duplicated per mode.
"""

import logging
import re
import threading
import time
from typing import Any, Iterator, Optional

import numpy as np
import sherpa_onnx
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig
from scipy.signal import resample

from brain import personas
from brain.modes import STATE
from config import MODELS, HardwareTarget

logger = logging.getLogger(__name__)

FRAME_SAMPLES = int(MODELS.asr_sample_rate * 0.1)  # 100ms chunks

# How much synthesized audio to keep queued ahead of realtime on the robot.
# Sized against the ~1s control-link stalls seen over a phone hotspot (see
# body/motion.py's _send_pose), since the same congestion gaps the audio.
_PLAYBACK_LEAD_S = 1.0

# Piper normalises every utterance to full scale by default, which does two
# unwanted things: it hands the robot's amplifier a peak of exactly 1.0000
# (measured on every test sentence) which crackles at the speaker's stock 100%
# volume, and it flattens each sentence to the same loudness, so a hushed line
# and an exclamation come out level. Turning normalisation off fixes both --
# measured peak drops to ~0.62, leaving natural headroom and the dynamics the
# model actually produced.
_CHAT_SYNTHESIS = SynthesisConfig(normalize_audio=False)

# Storytelling delivery. Slower (measured 10.4s -> 11.0s on the same passage)
# so lines land instead of rattling past, with more prosody and rhythm
# variation than the conversational default (noise_scale 0.667, noise_w 0.8)
# so successive sentences aren't delivered on the same contour.
_STORY_SYNTHESIS = SynthesisConfig(
    length_scale=1.08,
    noise_scale=0.85,
    noise_w_scale=1.0,
    normalize_audio=False,
)

#: The synthesiser's own defaults for the two knobs a caller can vary, so a
#: voice character is expressed as a multiplier of "normal" rather than as raw
#: model parameters that mean nothing to a demo author.
_BASE_NOISE_SCALE = 0.667


#: The conversational voice's own rate, before the operator's setting. This
#: used to be piper's bare default -- _CHAT_SYNTHESIS set no length_scale at
#: all -- and was reported as too fast in a real room. It is written down here
#: so the default has a value somebody can see and change, rather than being
#: whatever the synthesiser happens to do.
_BASE_PACE = 1.0

#: What the operator's "speaking speed" control means at each end. Higher is
#: slower, matching `pace` everywhere else. The range is bounded by what piper
#: still sounds like a person at: past about 1.4 it drawls, below 0.85 it
#: gabbles, and _voice_config clamps anyway.
SPEECH_PACE_KEY = "speech_pace"
SPEECH_PACE_RANGE = (0.9, 1.4)

#: Slower than the synthesiser's default, on purpose. "Too fast" was the
#: report from the room this robot actually lives in, and a foyer with hard
#: surfaces needs more space between words than a desk does.
SPEECH_PACE_DEFAULT = 1.15


def speech_pace() -> float:
    """The operator's speaking-speed multiplier, or the default.

    Read per line rather than cached, so moving the slider changes the very
    next thing the robot says instead of the next time it is restarted.
    """
    try:
        from brain import settings

        raw = settings.get(SPEECH_PACE_KEY, "")
        if raw:
            value = float(raw)
            low, high = SPEECH_PACE_RANGE
            return max(low, min(high, value))
    except Exception:
        # A settings table that will not read must never cost the robot its
        # voice; the default is a perfectly good answer.
        pass
    return SPEECH_PACE_DEFAULT


def _story_synthesis(slower: float) -> SynthesisConfig:
    """The storyteller's voice, following the operator's speed setting."""
    return SynthesisConfig(
        length_scale=max(0.8, min(1.6, 1.08 * slower)),
        noise_scale=0.85,
        noise_w_scale=1.0,
        normalize_audio=False,
    )


def _voice_config(pace: float, variation: float) -> SynthesisConfig:
    """Build a synthesis config from an abstract voice character.

    `pace` is a speaking-rate multiplier (higher is slower) and `variation`
    scales how much the prosody moves. Both are named for what a demo author
    can reason about, because the alternative -- brain/personas.py holding
    piper's length_scale and noise_scale directly -- would put a synthesiser's
    parameter names in the hardware-independent half of the codebase.

    This exists because personas previously carried pace and variation that
    nothing read: speak() offered a single boolean picking between two fixed
    configs, and the storyteller one is both slower AND more varied, so it
    could not express "friendly" (quicker, warmer) without lying about the
    pace. The personality demo's whole premise is that the same question sounds
    different, and a third of that difference was unimplemented.
    """
    return SynthesisConfig(
        length_scale=max(0.8, min(1.4, pace)),
        noise_scale=max(0.3, min(1.2, _BASE_NOISE_SCALE * variation / 0.667)),
        noise_w_scale=max(0.4, min(1.4, variation)),
        normalize_audio=False,
    )

# Audio discarded right after a wake-word match, to clear the tail of the wake
# phrase out of the buffer before the request is transcribed.
_WAKE_DRAIN_S = 0.35

# How often to report mic level while listening.
_LEVEL_LOG_INTERVAL_S = 3.0

# How long the daemon can deliver no audio at all before saying so. Silence
# from a working mic still produces frames; nothing at all means the media
# session is gone, which otherwise presents only as a robot that stopped
# responding for no visible reason.
_MIC_STALL_WARN_S = 2.0

# Ceiling on one visitor utterance. Not a conversational limit -- it is well
# past any real answer -- but a guarantee that listen() returns even when the
# microphone has stopped producing audio entirely and no endpoint can fire.
_MAX_UTTERANCE_S = 25.0

# Automatic gain control (see AudioIO._apply_gain). Target sits below full
# scale so normal variation between words doesn't clip; the envelope decays
# slowly (~1.5s to halve at this chunk rate) so gain doesn't ramp up during
# pauses and drag room noise up with it.
_AGC_TARGET_PEAK = 0.65
_AGC_MAX_GAIN = 120.0
_AGC_DECAY = 0.985

# Noise gate. The floor tracks the room's quiet level: fast toward quieter
# input, slow away from it, so a pause pulls it down within a second or so
# while speech barely moves it. Audio only counts as speech once it exceeds
# that floor by _NOISE_GATE_RATIO -- 4x (~12dB) clears typical room tone
# without needing someone to raise their voice. _AGC_ABSOLUTE_FLOOR stops a
# dead-silent room from driving the gate to zero and re-opening on hiss.
_NOISE_FLOOR_ATTACK = 0.05
_NOISE_FLOOR_RELEASE = 0.0008
_NOISE_GATE_RATIO = 4.0
_AGC_ABSOLUTE_FLOOR = 0.002

# How often to start a fresh wake-word stream while idle (see
# wait_for_wake_word). Short enough that a degraded stream self-corrects
# within one attempt, long enough not to sit mid-phrase repeatedly.
_KWS_STREAM_RECYCLE_S = 20.0

# Upper bound on flush_mic, so a mic delivering samples faster than realtime
# can't hold the loop there indefinitely.
_MAX_FLUSH_CHUNKS = 2000

# Streaming zipformer models need silence before AND after the real audio in a
# one-shot (offline) decode: leading padding warms up the encoder's left
# context (a stream that starts cold garbles the first word or two), and
# trailing padding flushes the last chunk through -- without it the tail end
# of the utterance gets truncated or dropped. Live mic streaming doesn't need
# this: context builds up naturally as audio keeps arriving, and is_endpoint
# already waits for real trailing silence before firing.
_LEAD_PADDING = np.zeros(int(MODELS.asr_sample_rate * 0.5), dtype=np.float32)
_TAIL_PADDING = np.zeros(int(MODELS.asr_sample_rate * 0.66), dtype=np.float32)


#: Models built ahead of AudioIO() by preload_models, keyed by part. Filled on
#: a worker thread during boot and consumed exactly once by the constructor.
_PRELOADED: dict[str, Any] = {}


def _warm_voice():
    """Load the default piper voice AND speak one throwaway word into it.

    The first synthesize() after loading pays onnxruntime session warm-up --
    measured 0.73s cold against 0.26s warm for the same short text -- and that
    cost used to land on the first visitor's first reply. Here it lands on the
    preload worker, during the seconds the robot connection is being
    established, where nobody is waiting on it.
    """
    voice = PiperVoice.load(str(MODELS.tts_model_path), str(MODELS.tts_config_path))
    try:
        list(voice.synthesize("Hi."))
    except Exception:
        logger.debug("Voice warm-up failed; first reply pays it instead", exc_info=True)
    return voice


def preload_models() -> None:
    """Build the speech models before AudioIO() is called, off the boot path.

    Each part is independent and each failure is swallowed: a part missing
    from the cache is simply built inline by the constructor as it always was,
    so the worst this can do is save no time. Whisper's builder already
    returns None on failure -- that None must not be cached, or the "or"
    fallback in the constructor would rebuild it anyway and the preload time
    would be spent twice.
    """
    builders = (
        ("recognizer", _build_recognizer),
        ("spotter", _build_spotter),
        ("whisper", _build_whisper),
        ("voice", _warm_voice),
    )
    for key, build in builders:
        if key in _PRELOADED:
            continue
        try:
            built = build()
        except Exception:
            logger.exception("Could not preload the %s; it will build inline.", key)
            continue
        if built is not None:
            _PRELOADED[key] = built


def _build_recognizer() -> sherpa_onnx.OnlineRecognizer:
    d = MODELS.asr_dir
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(d / "tokens.txt"),
        encoder=str(d / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        decoder=str(d / "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        joiner=str(d / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        num_threads=2,
        sample_rate=MODELS.asr_sample_rate,
        feature_dim=80,
        enable_endpoint_detection=True,
        # Patience before any speech is decoded. The default (2.4s) ends the
        # turn while someone is still drawing breath after the wake word --
        # observed live, producing an empty transcript the LLM then "answered".
        # Only this first window is generous; rule2 (silence *after* speech
        # starts) is left at its default so replies still come promptly once
        # you've actually said something.
        rule1_min_trailing_silence=5.0,
        # Silence *after* speech has started, i.e. how long it waits to be sure
        # you have finished. This is pure dead air before anything begins, so
        # it was trimmed to 0.8s from the 1.2s default -- but that clipped
        # sentences exactly as the old comment here warned it might: an
        # ordinary pause mid-question ended the turn, and because the captured
        # audio stops at the endpoint, Whisper only ever saw the fragment too
        # ("hearing: AND", then a one-word answer to a full question). Back
        # to the default, since a turn that transcribes one word in five costs
        # far more than half a second of dead air.
        #
        # 1.2 -- sherpa's own default -- and only safe because whisper_threads
        # dropped to 4: at 6 threads the early decode outlived the shorter
        # silence and 1.2 was measured NET SLOWER than 1.5. At 4 threads it
        # moves transcript-in-hand from ~2.05s to ~1.8s after the last word,
        # and was measured to survive 1.1s mid-sentence pauses (it cuts at
        # 1.3s; the live failure that pushed this value up happened at 0.8).
        # If slow talkers start getting clipped -- the signature is a fragment
        # transcript like "hearing: AND" -- put it back to 1.5.
        rule2_min_trailing_silence=1.2,
        decoding_method="modified_beam_search",
        hotwords_file=str(MODELS.asr_hotwords_file),
        hotwords_score=MODELS.asr_hotwords_score,
    )


#: Whisper labels non-speech audio rather than returning nothing for it --
#: "(tapping)", "[door closes]", "*sounds of a bird*". Live, that turned every
#: knock and chair scrape into a turn: the robot solemnly replied "I can't
#: respond with a tap", which then became conversation history. These are
#: descriptions of the audio, never words anybody said, so they are dropped.
_SOUND_EVENT_RE = re.compile(r"[\(\[\*][^\)\]\*]*[\)\]\*]")

#: The same annotation with its closing bracket missing -- Whisper truncates
#: them ("(static"), and the paired pattern above leaves those untouched, so
#: "(static" reached the LLM and was answered.
_UNCLOSED_EVENT_RE = re.compile(r"[\(\[\*][^\)\]\*]*$")


def _strip_sound_events(text: str) -> str:
    """Remove Whisper's non-speech annotations; "" if nothing else remains."""
    cleaned = _SOUND_EVENT_RE.sub(" ", text)
    cleaned = _UNCLOSED_EVENT_RE.sub(" ", cleaned)
    cleaned = " ".join(cleaned.split())
    # Punctuation-only leftovers ("...", "-") are not speech either.
    if not any(ch.isalnum() for ch in cleaned):
        return ""
    return cleaned


def _build_whisper() -> Optional[sherpa_onnx.OfflineRecognizer]:
    """Load Whisper for utterance transcription, or None to stay with the
    streaming zipformer.

    int8 weights: on CPU they decode meaningfully faster than the float
    versions for no accuracy difference that matters at this size.
    """
    d = MODELS.whisper_dir
    if d is None:
        return None
    encoder = d / "base.en-encoder.int8.onnx"
    decoder = d / "base.en-decoder.int8.onnx"
    tokens = d / "base.en-tokens.txt"
    if not (encoder.exists() and decoder.exists() and tokens.exists()):
        logger.warning("Whisper files missing in %s -- using the streaming model.", d)
        return None
    try:
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            num_threads=MODELS.whisper_threads,
        )
    except Exception as exc:
        logger.warning("Whisper unavailable (%s) -- using the streaming model.", exc)
        return None


#: How long the streaming transcript must sit unchanged before Whisper starts
#: decoding it EARLY, while the endpoint rule is still counting silence. The
#: endpoint needs rule2_min_trailing_silence (1.5s) of quiet before it fires,
#: and that window used to be pure dead air with the whole utterance already
#: captured -- Whisper then ran AFTER it, serially, adding its own second or
#: two. Starting the decode a beat into the silence hides almost all of
#: Whisper's cost inside a wait that exists anyway. Short enough to finish
#: before the endpoint on ordinary answers; long enough that a mid-sentence
#: breath does not launch decodes that get thrown away.
#: Mean acoustic log-probability below which a transcript is treated as a
#: mishearing rather than an answer. The streaming zipformer publishes one
#: log-prob per token via recognizer.ys_probs(); Whisper, which produces the
#: text that actually gets answered, returns an EMPTY ys_log_probs on these
#: model files, so this is the only real confidence signal available.
#:
#: Measured with piper speech under added noise, five phrases per level:
#:
#:   noise   mean    worst   transcript
#:   0.00    -0.38   -0.68   correct
#:   0.10    -0.71   -1.07   correct
#:   0.15    -0.98   -1.30   correct
#:   0.25    -1.54   -2.36   WRONG ("LEFT THE KING" for "what can you do")
#:
#: Correct speech reached -1.30, so the obvious -1.0 would have rejected people
#: talking normally. -2.0 sits below every correct reading measured and above
#: the clearly-wrong ones. Deliberately generous: a visitor who spoke clearly
#: being told "sorry, say that again?" is a worse failure than the robot
#: occasionally answering nonsense, which is what it does today anyway.
#:
#: THEN LIVE SPEECH CONTRADICTED THE TABLE, and the table lost. Measured on the
#: robot, in the room, with the microphone it actually uses:
#:
#:   -1.03  correct    answered
#:   -1.35  correct    answered
#:   -2.44  CORRECT    rejected -- "What's new in the AI XR tech market?"
#:
#: That last one is the exact failure this was supposed to avoid: a visitor who
#: spoke perfectly clearly was told "sorry, I did not catch that". Piper is
#: cleaner than a real room -- tools/selftest.py says so in its own docstring --
#: and -2.0 came from piper, so the threshold was measuring the wrong world.
#:
#: Now set well below anything yet observed to be correct. It will fire rarely,
#: which is the right bias: the mishearing it was built for ("Quizance",
#: "testing testic") is an occasional embarrassment, while telling somebody who
#: spoke clearly to repeat themselves is the robot appearing not to work.
#:
#: This number should keep moving as the log fills. There is still NO live
#: measurement of what a genuinely wrong transcript scores -- the mishearings
#: that motivated this predate the logging -- so the gap between "correct at
#: -2.44" and "wrong at ???" is unmeasured, and this floor is bounded on one
#: side only. Every turn logs its score; a day of real visitors settles it.
_MIN_MEAN_TOKEN_LOGPROB = -3.5

_EARLY_DECODE_AFTER_S = 0.45


class _EarlyDecode:
    """One speculative Whisper decode, begun before the endpoint fires.

    Started on a snapshot of the captured audio once the streaming transcript
    has been stable for _EARLY_DECODE_AFTER_S. Valid only if the transcript is
    STILL the same when the endpoint fires -- the streaming model is already
    the authority on whether speech has ended (it is what fires the endpoint),
    so "no new words since the snapshot" is its own judgement that the snapshot
    covered everything said. If more words arrived, the result is discarded and
    the ordinary post-endpoint decode runs exactly as it always did.

    A stale decode is ABANDONED, never joined: it finishes on its own thread,
    writes a result nothing reads, and is garbage-collected. Waiting for one
    was tried first and measured worse than never speculating -- a decode of
    half a question, still running at the endpoint, held up the full decode
    behind it. Concurrent decodes on the shared recognizer are safe because
    onnxruntime sessions support concurrent Run calls; the cost is brief CPU
    contention, which is cheaper than any amount of serial waiting.
    """

    def __init__(self, decode, samples: np.ndarray, partial: str) -> None:
        self.partial = partial
        self.started = time.monotonic()
        self._result = ""
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(decode, samples), name="whisper-early", daemon=True
        )
        self._thread.start()

    def _run(self, decode, samples: np.ndarray) -> None:
        try:
            self._result = decode(samples)
        except Exception:
            # _transcribe_whisper already never raises; this is for anything
            # else a future caller passes. "" makes listen() fall back to the
            # ordinary decode, which is the same answer arriving later.
            logger.exception("Early Whisper decode failed")
        finally:
            self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def result(self) -> str:
        """The transcript, waiting for the decode if it is still running."""
        self._done.wait()
        return self._result


def _build_spotter() -> sherpa_onnx.KeywordSpotter:
    d = MODELS.kws_dir
    return sherpa_onnx.KeywordSpotter(
        tokens=str(d / "tokens.txt"),
        encoder=str(d / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
        decoder=str(d / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
        joiner=str(d / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
        keywords_file=str(MODELS.kws_keywords_file),
        keywords_threshold=MODELS.kws_threshold,
        keywords_score=MODELS.kws_score,
        num_threads=2,
        sample_rate=MODELS.asr_sample_rate,
    )


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return samples
    return resample(samples, int(len(samples) * to_rate / from_rate)).astype(np.float32)


class AudioIO:
    """Wraps mic input (wake word + streaming STT) and speaker output (TTS)."""

    def __init__(self, target: HardwareTarget, robot: Optional[Any] = None) -> None:
        self.target = target
        # Taken from the preload cache when boot filled it, built inline when
        # not. Boot used to pay for these AFTER connecting to the robot, one
        # after another -- ~10s of models on the critical path behind ~14s of
        # network. preload_models() builds them on a worker thread WHILE the
        # robot connects, and this constructor then assembles instantly. The
        # inline fallback keeps every other caller -- tests, tools, a preload
        # that failed -- exactly as it was.
        self._recognizer = _PRELOADED.pop("recognizer", None) or _build_recognizer()
        self._spotter = _PRELOADED.pop("spotter", None) or _build_spotter()
        self._whisper = _PRELOADED.pop("whisper", None) or _build_whisper()
        logger.info(
            "Transcription: %s", "Whisper" if self._whisper is not None else "streaming zipformer"
        )
        self._voice = (_PRELOADED.pop("voice", None)
                       or PiperVoice.load(str(MODELS.tts_model_path), str(MODELS.tts_config_path)))
        self._voice_name = MODELS.tts_model_path.stem
        #: Piper voices already loaded, by file stem. A persona owns a voice,
        #: so switching persona must not re-read a model off disk per sentence.
        self._voice_cache: dict[str, Any] = {self._voice_name: self._voice}
        #: A voice the operator chose by hand, which outranks the persona's
        #: until a different persona is picked. See _speaking_voice.
        self._voice_override: Optional[str] = None
        self._persona_seq_seen = -1

        self._robot: Optional[Any] = robot
        #: Confidence of the last utterance, or None when there was no opinion.
        #: Read by demokit/runner.py to decide whether to answer or ask again.
        self.last_confidence: Optional[float] = None
        self._owns_robot = robot is None
        self._level_peak = 0.0
        self._level_clipped = 0
        self._level_samples = 0
        self._level_logged_at = 0.0
        self._agc_envelope = 0.0
        self._agc_gain = float(target.mic_gain)
        self._agc_noise_floor = _AGC_ABSOLUTE_FLOOR
        #: Whether the noise gate applies (see _apply_gain). Off while waiting
        #: for the wake word, on while transcribing what was said.
        self._gate_enabled = True
        #: The spotter stream, kept across wait_for_wake_word calls (which the
        #: voice loop makes every 1-2s to stay responsive to mode changes).
        #: None means "build a fresh one, flushing first" -- set after the robot
        #: has used the mic itself. See wait_for_wake_word.
        self._kws_stream: Optional[Any] = None
        self._kws_recycle_at = 0.0
        #: True when the mic backlog was just drained on purpose by a wake-word
        #: match, so listen() knows the audio waiting for it is the question
        #: itself rather than a backlog to throw away. See listen().
        self._mic_fresh = False
        if target.mode == "robot":
            if self._robot is None:
                from reachy_mini import ReachyMini

                self._robot = ReachyMini(
                    host=target.daemon_host,
                    port=target.daemon_port,
                    media_backend=target.media_backend,
                    log_level="WARNING",
                )
                self._robot.__enter__()
            self._robot.media.start_recording()
            self._robot.media.start_playing()

    def close(self) -> None:
        """Release the robot media connection, if this target uses one."""
        if self._robot is not None:
            self._robot.media.stop_recording()
            self._robot.media.stop_playing()
            if self._owns_robot:
                self._robot.__exit__(None, None, None)
            self._robot = None

    # --- offline processing, independent of the live mic: used both by the
    # real loop's helpers below and by scripted correctness tests that feed
    # in synthesized audio instead of a live human voice ---

    def transcribe_offline(self, samples: np.ndarray) -> str:
        """Transcribe one complete float32 mono utterance array in a single shot."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(MODELS.asr_sample_rate, _LEAD_PADDING)
        stream.accept_waveform(MODELS.asr_sample_rate, samples)
        stream.accept_waveform(MODELS.asr_sample_rate, _TAIL_PADDING)
        stream.input_finished()
        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)
        return self._recognizer.get_result(stream).strip()

    def detect_wake_word_offline(self, samples: np.ndarray) -> bool:
        """Run the wake-word spotter over one complete float32 mono clip.

        Unlike transcribe_offline, this deliberately skips lead padding: for
        this KWS model, silence before the keyword suppresses detection
        almost entirely (measured empirically -- see kws_threshold's
        docstring in config.py), unlike the general ASR model where lead
        padding helps. Tail padding still matters here for the same reason
        it does for ASR (flushing the last real chunk through).
        """
        stream = self._spotter.create_stream()
        stream.accept_waveform(MODELS.asr_sample_rate, samples)
        stream.accept_waveform(MODELS.asr_sample_rate, _TAIL_PADDING)
        stream.input_finished()
        while self._spotter.is_ready(stream):
            self._spotter.decode_stream(stream)
        return bool(self._spotter.get_result(stream))

    def detect_wake_word_streaming(self, samples: np.ndarray) -> bool:
        """Run the spotter over a clip the way the live loop would hear it.

        The difference from detect_wake_word_offline is not cosmetic, and
        measuring the wrong one gives the wrong answer about the wake word. That
        one hands the decoder a single clip; this feeds 100ms frames through
        _apply_gain and checks after each, which is what wait_for_wake_word and
        wake_word_in_backlog both do, and nothing else in the robot does it any
        other way.

        Measured over ten takes per phrase, the two paths disagree sharply:
        "hey reachy" scored 4/10 offline and 9/10 streaming on identical audio,
        and widening the keyword list moved the two in opposite directions. The
        offline number is the one that looks like a broken wake word, and it is
        the one the robot never runs -- so tests use this instead.
        """
        stream = self._spotter.create_stream()
        was_gated, self._gate_enabled = self._gate_enabled, False
        try:
            padded = np.concatenate([samples, _TAIL_PADDING])
            for start in range(0, len(padded), FRAME_SAMPLES):
                frame = padded[start : start + FRAME_SAMPLES]
                if len(frame) < FRAME_SAMPLES:
                    frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
                stream.accept_waveform(MODELS.asr_sample_rate, self._apply_gain(frame))
                while self._spotter.is_ready(stream):
                    self._spotter.decode_stream(stream)
                if self._spotter.get_result(stream):
                    return True
            return False
        finally:
            self._gate_enabled = was_gated

    # --- live mic/speaker: robot.media in "robot" mode, this machine's real
    # devices in "simulation" mode ---

    def _log_level(self, samples: np.ndarray) -> None:
        """Report post-gain mic level periodically.

        Without this, "it didn't hear me" is unattributable: no audio arriving
        at all, audio arriving too quiet to trigger, and audio clipping into
        distortion all look identical from the outside. Peak is what the
        wake-word model effectively sees; `clip` flags gain set too high.
        """
        now = time.monotonic()
        self._level_peak = max(self._level_peak, float(np.abs(samples).max()))
        self._level_clipped += int(np.count_nonzero(np.abs(samples) >= 0.999))
        self._level_samples += samples.size
        if now - self._level_logged_at < _LEVEL_LOG_INTERVAL_S:
            return
        clip_pct = 100.0 * self._level_clipped / max(1, self._level_samples)
        logger.info(
            "mic level: peak=%.3f clip=%.2f%% (gain=%.1fx%s)",
            self._level_peak,
            clip_pct,
            self._agc_gain if self.target.mic_agc else self.target.mic_gain,
            " auto" if self.target.mic_agc else "",
        )
        self._level_logged_at = now
        self._level_peak = 0.0
        self._level_clipped = 0
        self._level_samples = 0

    def _transcribe_whisper(self, samples: np.ndarray) -> str:
        """Decode one complete utterance with Whisper. Returns "" on failure."""
        try:
            stream = self._whisper.create_stream()
            stream.accept_waveform(MODELS.asr_sample_rate, samples)
            self._whisper.decode_stream(stream)
            return _strip_sound_events(stream.result.text.strip())
        except Exception:
            # Never lose a turn to this -- listen() falls back to the streaming
            # model's transcript, which is already in hand.
            logger.exception("Whisper decode failed")
            return ""

    def _apply_gain(self, samples: np.ndarray) -> np.ndarray:
        """Bring mic audio to a usable level for the STT/wake-word models.

        A fixed multiplier can't work on this input: measured live, the same
        speaker's wake word peaked at full scale (22% of samples clipped)
        while their following question sat at 0.12 -- a ~20x spread, so any
        constant either clips the loud parts or leaves the quiet ones
        undecodable. Instead track a decaying peak envelope and derive the
        gain from it, so loud passages are attenuated and quiet ones lifted.

        The envelope decays slowly rather than following the signal down, so
        gain doesn't surge during the pauses between words and amplify room
        noise into the models.
        """
        if not self.target.mic_agc:
            if self.target.mic_gain == 1.0:
                return samples
            return np.clip(samples * self.target.mic_gain, -1.0, 1.0)

        peak = float(np.abs(samples).max())

        # Learn the room's own quiet level instead of assuming one. A fixed
        # floor was tried and failed live: set below this room's ambient, the
        # AGC normalised background conversation to full scale and the
        # recognizer transcribed the room continuously (which then reached
        # enrollment and stored overheard speech as a person's name). The
        # floor rises quickly toward quiet input and falls back slowly, so it
        # settles on the background level rather than on speech.
        if peak < self._agc_noise_floor:
            self._agc_noise_floor += (peak - self._agc_noise_floor) * _NOISE_FLOOR_ATTACK
        else:
            self._agc_noise_floor += (peak - self._agc_noise_floor) * _NOISE_FLOOR_RELEASE

        self._agc_envelope = max(peak, self._agc_envelope * _AGC_DECAY)

        # Gate: only treat this as speech when it stands clear of the room.
        # Below that, pass it through at unity so silence stays silent --
        # amplifying it would just manufacture input out of noise.
        #
        # Skipped entirely while listening for the wake word. The gate exists
        # to stop room noise being transcribed as a question, which only
        # matters once the robot is already listening to you. Applying it
        # beforehand meant quiet speech never reached the spotter and the wake
        # word had to be shouted. The two failures are not equally bad: a
        # false wake costs a moment of listening to nothing, a missed one
        # costs the user raising their voice at a robot.
        if self._gate_enabled and self._agc_envelope < max(
            _AGC_ABSOLUTE_FLOOR, self._agc_noise_floor * _NOISE_GATE_RATIO
        ):
            self._agc_gain = 1.0
            return samples

        gain = min(_AGC_MAX_GAIN, _AGC_TARGET_PEAK / self._agc_envelope)
        if not self._gate_enabled:
            # Waiting for the wake word: floor the gain. The envelope decays
            # slowly by design, so for ~2-4 seconds after any loud transient
            # (the robot's own speech, a door, laughter) the gain sits pinned
            # near 1x and a normal-volume "Hey Reachy" reaches the spotter too
            # quiet to match -- measured 9/16 wake phrases caught in that
            # state against 14/16 baseline, and today's live log shows gain
            # pinned at 0.8-1.4x repeatedly. A 4x floor restores the baseline;
            # genuinely loud speech just clips at the top, which the spotter
            # tolerates far better than starvation.
            gain = max(4.0, gain)
        self._agc_gain = gain
        return np.clip(samples * gain, -1.0, 1.0)

    def _mic_frames(self, deadline: Optional[float] = None) -> Iterator[np.ndarray]:
        """Yield mono float32 chunks at MODELS.asr_sample_rate.

        `deadline` (a time.monotonic() value) stops the generator even when no
        audio is arriving at all. Callers used to time out by checking the
        clock once per yielded frame, which silently assumes frames keep
        coming: in robot mode this loop spins on `sample is None` whenever the
        daemon's media pipeline stalls, so it yielded nothing, the caller's
        deadline check was never reached, and wait_for_wake_word(timeout=2.0)
        blocked forever -- taking the mode switch and the whole voice loop with
        it, while the dashboard kept answering from its own thread and made it
        look alive. A stalled pipeline is not hypothetical; it is what a
        dropped WebRTC media session looks like from this side.
        """
        if self.target.mode == "robot":
            if self._robot is None:
                # The link is being rebuilt underneath us. Ending the generator
                # is right: the caller opens a new one on the next turn, and
                # dereferencing media here would raise mid-sentence.
                return
            input_rate = self._robot.media.get_input_audio_samplerate()
            stalled_since: Optional[float] = None
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                if self._robot is None:
                    return
                sample = self._robot.media.get_audio_sample()
                if sample is None:
                    now = time.monotonic()
                    if stalled_since is None:
                        stalled_since = now
                    elif now - stalled_since > _MIC_STALL_WARN_S:
                        logger.warning(
                            "No mic audio from the daemon for %.0fs -- media session may be gone.",
                            now - stalled_since,
                        )
                        stalled_since = now
                    time.sleep(0.01)
                    continue
                stalled_since = None
                # robot.media returns interleaved-channel frames (stereo on this
                # hardware); sherpa-onnx expects a flat mono sequence.
                mono = sample[:, 0] if sample.ndim == 2 else sample
                mono = self._apply_gain(mono)
                self._log_level(mono)
                yield _resample(mono, input_rate, MODELS.asr_sample_rate)
        else:
            with sd.InputStream(
                samplerate=MODELS.asr_sample_rate,
                channels=1,
                dtype="float32",
                device=self.target.audio_input_device,
                blocksize=FRAME_SAMPLES,
            ) as mic:
                while True:
                    if deadline is not None and time.monotonic() >= deadline:
                        return
                    samples, _ = mic.read(FRAME_SAMPLES)
                    yield samples[:, 0]

    def wait_for_wake_word(self, timeout: Optional[float] = None) -> bool:
        """Wait for the wake word. Returns True if heard, False on timeout.

        `timeout` lets a caller interleave other behaviour (see the modes in
        body/voice_loop.py) without ever stopping listening: capture continues
        regardless, so returning early loses nothing.

        The stream and the mic backlog therefore have to survive across calls.
        Rebuilding either one per call destroys the phrase being spoken across
        the boundary: the loop polls every 1-2s, "hey reachy" takes about one,
        and a fresh stream starts with no memory of the audio that already
        arrived while flush_mic drops whatever landed between calls. That is
        why the wake word only answered about one time in five -- it was heard
        reliably, just not by a stream that lived long enough to finish
        matching it. The flush still happens, but only when the robot has just
        used the mic itself (see speak/listen, which clear the stream so the
        backlog of its own voice is dropped before listening resumes).

        The stream is still recycled periodically. This loop can run for hours
        between matches, and a single stream fed continuously that whole time
        stops detecting reliably -- observed live, where the wake word matched
        shortly after startup and then went unrecognised for minutes while
        audio was still clearly arriving at a healthy level.
        """
        self._gate_enabled = False  # maximum sensitivity; see _apply_gain
        now = time.monotonic()
        if self._kws_stream is None:
            self.flush_mic()
            self._kws_stream = self._spotter.create_stream()
            self._kws_recycle_at = now + _KWS_STREAM_RECYCLE_S
        deadline = None if timeout is None else now + timeout
        # The deadline goes to the frame source as well as being checked here:
        # a stalled mic yields nothing, and a check that only runs per frame
        # never fires. See _mic_frames.
        for frame in self._mic_frames(deadline=deadline):
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                # Caller wants to do something else and ask again; audio keeps
                # being captured either way, so nothing is missed by returning.
                return False
            if now >= self._kws_recycle_at:
                self._kws_stream = self._spotter.create_stream()
                self._kws_recycle_at = now + _KWS_STREAM_RECYCLE_S
            stream = self._kws_stream

            stream.accept_waveform(MODELS.asr_sample_rate, frame)
            while self._spotter.is_ready(stream):
                self._spotter.decode_stream(stream)
            result = self._spotter.get_result(stream)
            if result:
                logger.info("Wake word matched: %r", result)
                self._spotter.reset_stream(stream)
                # Drain whatever is still buffered from the wake phrase itself,
                # so listen() starts on the request rather than transcribing
                # the tail of "hey reachy" into it (that bleed produced
                # "ISN'T THAT MOTIVE HAIR REACHY" from a clean utterance).
                self._drain_mic(_WAKE_DRAIN_S)
                self._mic_fresh = True
                return True
        return False

    def _drain_mic(self, seconds: float) -> None:
        """Discard incoming mic audio for a short window."""
        deadline = time.monotonic() + seconds
        for _ in self._mic_frames(deadline=deadline):
            if time.monotonic() >= deadline:
                return

    def adopt_robot(self, robot: Any) -> None:
        """Rebind to a rebuilt connection and start its media flowing.

        The spotter stream is dropped rather than reused: it holds decoder
        state fed by the old connection's audio, and _kws_stream is already
        documented as "None means build a fresh one, flushing first".
        """
        self._robot = robot
        self._kws_stream = None
        if robot is not None:
            robot.media.start_recording()
            robot.media.start_playing()

    def available_voices(self) -> list[str]:
        """Voice names installed alongside the configured one.

        Read from disk rather than listed in config, so dropping a piper voice
        into models/tts/ is all it takes to offer it -- the same rule the demos
        follow. A voice is a matching .onnx and .onnx.json pair; anything else
        is a half-finished download and is skipped rather than offered and then
        failing when selected.
        """
        folder = MODELS.tts_model_path.parent
        names = []
        for model in sorted(folder.glob("*.onnx")):
            if model.with_suffix(".onnx.json").exists():
                names.append(model.stem)
        return names

    def _load_voice(self, name: str):
        """A loaded piper voice by file stem, cached. None if unusable.

        Cached because a persona owns a voice now, and loading a model takes
        long enough to hear: uncached, every sentence after a persona switch
        would pause while the same file was read off disk again. Three medium
        voices sit in memory at once, which is the cost of the personas being
        told apart by speaker rather than only by pace.
        """
        if name in self._voice_cache:
            return self._voice_cache[name]
        folder = MODELS.tts_model_path.parent
        model, config = folder / f"{name}.onnx", folder / f"{name}.onnx.json"
        loaded = None
        if not model.exists() or not config.exists():
            logger.warning("Voice %r is not installed in %s", name, folder)
        else:
            try:
                loaded = PiperVoice.load(str(model), str(config))
                logger.info("Loaded voice %s", name)
            except Exception:
                logger.exception("Could not load voice %r", name)
        # Cached even when it failed, so a missing voice is one warning rather
        # than one per sentence for the rest of the visit.
        self._voice_cache[name] = loaded
        return loaded

    def _speaking_voice(self):
        """(voice, name) to speak this line in.

        A persona names a voice, and picking a persona picks it. An operator
        who then chooses a voice by hand wins until the persona changes again,
        which is the behaviour that needs no explaining in either direction:
        choosing a character gives you its voice, and choosing a voice gives
        you that voice.
        """
        persona_id, seq = STATE.persona
        if seq != self._persona_seq_seen:
            self._persona_seq_seen = seq
            self._voice_override = None
        if self._voice_override is None:
            persona = personas.active(persona_id)
            if persona is not None and persona.voice:
                loaded = self._load_voice(persona.voice)
                if loaded is not None:
                    return loaded, persona.voice
        return self._voice, self._voice_name

    @property
    def voice_name(self) -> str:
        """The voice actually being spoken in, persona included.

        Reports the effective voice rather than the last one set by hand, so
        the dashboard cannot show a voice the robot is not using.
        """
        return self._speaking_voice()[1]

    def set_voice(self, name: str) -> bool:
        """Switch the speaking voice. Voice-loop thread only.

        Loading a piper voice takes a moment and replaces the object every
        utterance goes through, so it must not happen while another thread is
        mid-sentence. The dashboard therefore queues the change and the runner
        applies it between turns, the same way the Say button works.
        """
        folder = MODELS.tts_model_path.parent
        model = folder / f"{name}.onnx"
        config = folder / f"{name}.onnx.json"
        if not model.exists() or not config.exists():
            logger.warning("Voice %r is not installed in %s", name, folder)
            return False
        try:
            self._voice = PiperVoice.load(str(model), str(config))
        except Exception:
            logger.exception("Could not load voice %r; keeping %s", name, self._voice_name)
            return False
        self._voice_name = name
        # Remembered as a deliberate choice, so it outranks the persona's own
        # voice until a different persona is picked. Without this the operator
        # could not choose a voice at all while a persona was active -- the
        # next sentence would simply be back in the persona's.
        #
        # The sequence number is stamped here as well as in _speaking_voice,
        # and that is not redundant: a persona picked and a voice chosen with
        # nothing spoken in between left an unobserved change pending, so the
        # very next line cleared the override the operator had just set.
        self._voice_override = name
        self._persona_seq_seen = STATE.persona[1]
        logger.info("Voice set to %s", name)
        return True

    def wake_word_in_backlog(self) -> bool:
        """Was the wake word spoken while the robot was talking?

        Nothing consumes the microphone while the robot speaks, so a visitor
        who interrupts is not ignored so much as unheard -- but the daemon has
        been buffering the whole time, and that buffer is exactly the audio of
        them interrupting. flush_mic throws it away, which is right before a
        fresh question and wrong here.

        So this drains the same backlog and runs the spotter over it instead:
        a "hey Reachy" said three seconds into a long answer is found after the
        sentence ends, and the robot can stop and listen. No second thread and
        no listening during playback -- one thread owns the microphone, and
        that stays true.

        The backlog also contains the robot's own voice, so a reply that itself
        contained the wake phrase would interrupt itself. Callers pass what was
        just said and it is checked (see DemoContext).
        """
        if self.target.mode != "robot" or self._robot is None:
            return False

        input_rate = self._robot.media.get_input_audio_samplerate()
        # Decoded frame by frame through a streaming spotter, exactly as the
        # live path does, rather than handed to detect_wake_word_offline as one
        # clip. That was the first attempt and it did not work: the offline
        # detector is documented as needing the keyword near the start, because
        # audio before it suppresses detection almost entirely -- and this
        # buffer begins with several seconds of the robot's own voice, with the
        # visitor's "hey Reachy" somewhere in the middle. Feeding it in order,
        # checking after each frame, finds the phrase wherever it falls.
        #
        # The gain is applied per frame for the same reason: _apply_gain tracks
        # a decaying envelope, so running it over the whole buffer at once lets
        # the robot's much louder voice set the level and scales the visitor's
        # speech down to nothing.
        stream = self._spotter.create_stream()
        # The gate is not the only thing _apply_gain mutates: it also advances
        # _agc_envelope and _agc_noise_floor, and this buffer is mostly the
        # robot's own voice at full scale. Left un-restored, that inflated
        # envelope is still decaying when the visitor's next question arrives
        # and under-gains it -- the scan quietly degrading the listening that
        # follows it, every single time it runs.
        was_gated, self._gate_enabled = self._gate_enabled, False
        saved_envelope = self._agc_envelope
        saved_floor = self._agc_noise_floor
        chunks = 0
        loudest = 0.0
        try:
            # Gain per chunk, resample and decode per BLOCK. This scan runs
            # between every pair of spoken chunks, and per-chunk it was
            # measured at 3.84ms x 655 chunks = 2.5s of the robot standing
            # silent mid-reply -- the instrumentation logged 4.3s gaps live,
            # and every gap sat on exactly this loop. Gain stays per chunk
            # (the envelope tracking is why the robot's own loud voice does
            # not crush the visitor's words); the resample and the decoder
            # calls, which carry the fixed-overhead cost, run once per
            # ~half-second of audio instead of once per 11ms.
            block: list = []
            block_len = 0
            block_target = max(1, int(input_rate * 0.5))

            def _feed_block() -> None:
                nonlocal block, block_len
                if not block:
                    return
                frame = _resample(np.concatenate(block), input_rate,
                                  MODELS.asr_sample_rate)
                stream.accept_waveform(MODELS.asr_sample_rate, frame)
                while self._spotter.is_ready(stream):
                    self._spotter.decode_stream(stream)
                block, block_len = [], 0

            for _ in range(_MAX_FLUSH_CHUNKS):
                sample = self._robot.media.get_audio_sample()
                if sample is None:
                    break
                chunks += 1
                mono = sample[:, 0] if sample.ndim == 2 else sample
                loudest = max(loudest, float(np.abs(mono).max()) if mono.size else 0.0)
                block.append(self._apply_gain(mono))
                block_len += len(mono)
                if block_len < block_target:
                    continue
                _feed_block()
                if self._spotter.get_result(stream):
                    logger.info(
                        "Wake word heard while speaking -- interrupting (%d chunk(s), peak %.3f).",
                        chunks, loudest,
                    )
                    # Dropped so the turn that follows starts from a clean
                    # stream rather than resuming this one mid-phrase.
                    self._kws_stream = None
                    # KEEP the rest of the backlog. A one-breath interruption
                    # -- "hey Reachy, what about the masters?" -- has its
                    # question sitting right after the match, and the listen()
                    # that follows used to flush it (nothing set _mic_fresh on
                    # this path), so the visitor was stopped mid-reply and
                    # then asked to repeat themselves. Same drain-then-keep the
                    # ordinary wake path does at its match.
                    self._drain_mic(_WAKE_DRAIN_S)
                    self._mic_fresh = True
                    return True

            _feed_block()
            # Flush the tail. A phrase landing at the very end of the buffer --
            # somebody interrupting just as a sentence finishes, which is the
            # most natural moment to interrupt -- sits in the encoder
            # undecoded until more audio arrives, and there is none. Measured:
            # without this, an interruption buried mid-buffer is found and one
            # at the end is missed entirely.
            stream.accept_waveform(MODELS.asr_sample_rate, _TAIL_PADDING)
            stream.input_finished()
            while self._spotter.is_ready(stream):
                self._spotter.decode_stream(stream)
            if self._spotter.get_result(stream):
                logger.info(
                    "Wake word heard while speaking -- interrupting (%d chunk(s), peak %.3f).",
                    chunks, loudest,
                )
                self._kws_stream = None
                return True
        finally:
            self._gate_enabled = was_gated
            self._agc_envelope = saved_envelope
            self._agc_noise_floor = saved_floor
        # Logged even on a miss, because this is the only way to tell the two
        # failures apart: nobody spoke (low peak) against somebody spoke and
        # the spotter did not hear them (high peak). The scan used to be silent
        # either way, so a barge-in that did not work left nothing behind.
        #
        # The peak is also the measurement that matters most here. Synthesised
        # mixtures put the ceiling on what this can ever do: with the robot's
        # own voice in the buffer at half scale or more, the wake phrase spoken
        # over it was detected 0 times out of 4 at every visitor level tried;
        # at a tenth of scale it was 4/4. So this number says whether the
        # robot's speaker is being cancelled before the mic stream reaches us.
        # A high peak through a whole reply means it is not, and no amount of
        # tuning on this side will find a phrase buried under it.
        level = logging.INFO if loudest > 0.05 else logging.DEBUG
        logger.log(level, "No wake word in backlog (%d chunk(s), peak %.3f).", chunks, loudest)
        return False

    def flush_mic(self) -> None:
        """Drop mic audio captured while the robot was busy.

        Capture doesn't pause while the robot speaks, so by the end of a reply
        the daemon holds a backlog containing the robot's own voice. Consuming
        it takes real time and is decoded before any live audio, which is why
        a "Hey Reachy" spoken right after a reply appeared to be ignored -- the
        spotter was still working through the previous turn. Dropping whatever
        is already queued makes the next turn start from now.
        """
        if self.target.mode != "robot" or self._robot is None:
            return
        dropped = 0
        while dropped < _MAX_FLUSH_CHUNKS:
            if self._robot.media.get_audio_sample() is None:
                break
            dropped += 1
        if dropped:
            logger.info("Flushed %d stale mic chunk(s).", dropped)

    def listen(self, wait_for_speech_s: Optional[float] = None) -> str:
        """Record and transcribe one utterance.

        Logs the partial transcript as it grows, so it's visible *what* the
        recognizer is picking up and *when* -- a final-only log can't
        distinguish "never heard you" from "heard you and mistranscribed it".

        `wait_for_speech_s` gives up and returns "" if nobody has started
        speaking within that long, which is what open-mic follow-ups need: with
        no wake word to mark the start of a turn, this would otherwise sit on
        the voice loop for the full utterance ceiling every time a conversation
        simply ended, and the robot could not be switched away from or put to
        sleep for 25 seconds. It bounds only the wait for speech to *begin* --
        once someone is talking, the ordinary endpoint logic finishes the
        sentence however long it runs.
        """
        # Only drop the backlog when it is genuinely stale (the robot's own
        # voice from the previous turn). Straight after a wake-word match it
        # holds the start of the question -- wait_for_wake_word has already
        # drained the wake phrase itself -- and flushing it unconditionally
        # threw away the opening words of anyone who ran "hey reachy" and their
        # question together in one breath.
        if not self._mic_fresh:
            self.flush_mic()
        self._mic_fresh = False
        self._gate_enabled = True  # room noise must not become a question
        # These frames go to the recognizer, not the spotter, so the spotter's
        # stream would resume with a hole in it. Drop it and start clean.
        self._kws_stream = None
        stream = self._recognizer.create_stream()
        partial = ""
        captured: list[np.ndarray] = []
        started = time.monotonic()
        #: When the partial last gained words, and the speculative decode of
        #: the audio so far. See _EarlyDecode: the point is that Whisper runs
        #: DURING the endpoint's trailing-silence wait instead of after it.
        last_change = started
        early: Optional[_EarlyDecode] = None

        def best_transcript(streaming_text: str) -> str:
            """Whisper's transcript by the cheapest route, else the streaming one."""
            if self._whisper is None or not captured:
                return streaming_text
            better = ""
            route = "after endpoint"
            if early is not None and early.partial == partial:
                better = early.result()
                route = "during silence"
            # A STALE early decode -- launched on a mid-question pause, then
            # invalidated when they kept talking -- is simply abandoned, not
            # joined. The first cut waited for it before decoding afresh, which
            # made a pause mid-question SLOWER than before speculation existed
            # (observed live: a half-question decode still running at the
            # endpoint added its remnant to the full 3.3s decode). Running the
            # fresh decode alongside the dying one is safe: onnxruntime
            # sessions are thread-safe for concurrent Run calls -- it is how
            # sherpa's own concurrent servers use one recognizer -- and the
            # stale thread writes only its own _result, which nothing reads.
            if not better:
                decode_started = time.monotonic()
                better = self._transcribe_whisper(np.concatenate(captured))
                route = f"after endpoint, {time.monotonic() - decode_started:.2f}s"
            if better:
                logger.info("  whisper (%s): %s", route, better)
                return better
            return streaming_text
        # Bounded for the same reason wait_for_wake_word is: this returns only
        # when the recognizer declares an endpoint, and an endpoint needs
        # frames. If the daemon's media session drops, _mic_frames yields
        # nothing, no endpoint ever fires, and this blocks the voice loop
        # forever -- with the dashboard still answering from its own thread, so
        # the robot looks alive while being permanently deaf. The ceiling is
        # far longer than any real answer, so it never truncates a visitor; it
        # exists purely so a dead microphone ends the turn instead of the
        # session.
        deadline = started + _MAX_UTTERANCE_S
        for frame in self._mic_frames(deadline=deadline):
            stream.accept_waveform(MODELS.asr_sample_rate, frame)
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)

            current = self._recognizer.get_result(stream).strip()
            if current != partial:
                partial = current
                last_change = time.monotonic()
                logger.info("  [%5.1fs] hearing: %s", time.monotonic() - started, partial)
                # Streamed to the dashboard so words appear as they are spoken,
                # which is what makes a mishearing visible as it happens rather
                # than only in the final transcript.
                STATE.note("partial", partial)

            if self._whisper is not None:
                captured.append(frame)
                # Speech has paused. Start Whisper on what is in hand while the
                # endpoint rule keeps counting silence -- if they resume, the
                # changed partial invalidates this and a fresh speculation
                # replaces it the next time the transcript settles. The stale
                # thread is dropped, not joined: concurrent decodes are safe
                # (see best_transcript above), and waiting for a doomed decode
                # of half a question was measured costing more than never
                # speculating at all.
                if (
                    partial
                    and time.monotonic() - last_change >= _EARLY_DECODE_AFTER_S
                    and (early is None or early.partial != partial)
                    # Never stack speculations: an abandoned decode keeps its
                    # thread (and 4 whisper threads) running to completion, so
                    # launching a fresh one beside it puts 8+ threads on a
                    # 14-core laptop already running the zipformer and piper.
                    # Measured as the 5.1-6.8s transcript outliers on long
                    # questions -- every configuration showed them until
                    # stacking stopped. Skipping a speculation costs at most
                    # one ordinary post-endpoint decode (~0.6s at 4 threads).
                    and (early is None or not early._thread.is_alive())
                ):
                    early = _EarlyDecode(
                        self._transcribe_whisper, np.concatenate(captured), partial
                    )

            if wait_for_speech_s is not None and not partial:
                if time.monotonic() - started >= wait_for_speech_s:
                    # Nothing was said. Keep the backlog rather than letting the
                    # next call flush it: speech beginning in the final frames
                    # has not decoded into a partial yet, and dropping it would
                    # clip the first word off someone who spoke just as the
                    # window lapsed -- which, in open mic, is a whole turn lost
                    # with no wake word to make the visitor say it again.
                    self._mic_fresh = True
                    return ""

            if self._recognizer.is_endpoint(stream):
                text = self._recognizer.get_result(stream).strip()
                # Read BEFORE reset(), which discards the token probabilities
                # along with the rest of the decoder state.
                self.last_confidence = self._utterance_confidence(stream)
                self._recognizer.reset(stream)

                # The streaming model still runs the turn -- it is what detects
                # that speech has ended -- but Whisper's transcript answers,
                # because it is markedly better on live far-field speech. The
                # decode usually already ran during the trailing silence (see
                # _EarlyDecode); its result counts only if no words arrived
                # after its snapshot, which is the recognizer's own judgement
                # that the snapshot covered everything said. A transcript the
                # streaming model produced but Whisper returned "" for is far
                # more likely a decode failure than genuine silence, so the
                # streaming text still backstops it.
                return best_transcript(text)

        # Fell out of the loop: the deadline passed without an endpoint. Return
        # the best transcript so far rather than nothing -- a long answer that
        # ran past the ceiling is still worth answering, and an empty string
        # here would look to the caller like silence.
        logger.warning("Listening hit the %.0fs ceiling without an endpoint.", _MAX_UTTERANCE_S)
        return best_transcript(self._recognizer.get_result(stream).strip())

    def _utterance_confidence(self, stream: Any) -> Optional[float]:
        """Mean acoustic log-probability of the streaming hypothesis.

        None when the recogniser produced no tokens at all, which is silence
        rather than a bad transcript and must not be treated as one.
        """
        try:
            probs = list(self._recognizer.ys_probs(stream))
        except Exception:
            # Older sherpa builds may not expose this. A missing signal must
            # never cost a turn -- None means "no opinion", and the caller
            # answers the visitor exactly as it does today.
            return None
        if not probs:
            return None
        return float(sum(probs) / len(probs))

    def speak(
        self,
        text: str,
        emotion_tag: str,
        motion: Optional[Any] = None,
        expressive: bool = False,
        pace: Optional[float] = None,
        variation: Optional[float] = None,
    ) -> None:
        """Synthesize text with piper and play it.

        `expressive` performs it as a story. `pace` and `variation` give a
        voice a character of its own -- see _voice_config -- and take
        precedence, since a caller that named both meant them.
        """
        chunks = self._render_stream(text, expressive=expressive, pace=pace, variation=variation)
        if motion is not None:
            motion.begin_speech()
        try:
            if self.target.mode == "robot":
                self._speak_robot(chunks, motion)
            else:
                self._speak_local(chunks, motion)
        finally:
            if motion is not None:
                motion.end_speech()
            # The mic kept capturing while we spoke, so the backlog now holds
            # the robot's own voice. Dropping the stream makes the next
            # wait_for_wake_word flush that before it starts matching.
            self._kws_stream = None
            # And the backlog is stale for listen() too. listen() skips its
            # flush while _mic_fresh is set, and listen() itself sets that flag
            # when a wait_for_speech_s window lapses in silence -- so a question
            # asked, met with silence, and asked again left the flag standing
            # across this speak() and the second listen() transcribed the
            # robot's own question. In the enrolment exchange that is a
            # re-prompt being decoded as somebody's name.
            self._mic_fresh = False

    def _synthesis_config(
        self,
        expressive: bool = False,
        pace: Optional[float] = None,
        variation: Optional[float] = None,
    ) -> SynthesisConfig:
        """How this line should sound, resolved exactly as speak() always has.

        Everything ends up here -- demos, personas, the storyteller and the
        runner's own lines -- which is why the operator's speaking-speed
        setting is applied at this one point rather than at each caller.
        """
        slower = speech_pace()
        if pace is not None or variation is not None:
            return _voice_config((pace if pace is not None else 1.0) * slower,
                                 variation if variation is not None else 0.667)
        if expressive:
            # The storyteller's voice is the point of that demo, and it is both
            # slower AND more varied than any persona -- so a persona must not
            # quietly replace it. It still follows the operator's setting: a
            # room that needs everything slower needs the story slower too.
            return _story_synthesis(slower)
        # Resolved here rather than in DemoContext.say because this is the only
        # place that also covers the runner's own lines, which speak through
        # AudioIO directly. An explicit pace or variation still wins above.
        persona = personas.active(STATE.persona[0])
        if persona is not None:
            return _voice_config(persona.pace * slower, persona.variation)
        return _voice_config(_BASE_PACE * slower, 0.667)

    def _render_stream(self, text, expressive=False, pace=None, variation=None):
        syn_config = self._synthesis_config(expressive, pace, variation)
        voice, _name = self._speaking_voice()
        return voice.synthesize(text, syn_config=syn_config)

    def render(
        self,
        text: str,
        expressive: bool = False,
        pace: Optional[float] = None,
        variation: Optional[float] = None,
    ) -> list:
        """Synthesize a line WITHOUT playing it. Pure CPU, no audio hardware.

        The one AudioIO method deliberately callable off the voice-loop thread:
        DemoContext.reply renders sentence N+1 on a worker while sentence N is
        still coming out of the speaker, which is what removed the second or
        two of dead air between every pair of spoken sentences. Only one
        renderer runs at a time (reply's worker), so piper's session is never
        entered concurrently -- and playback of what this returns still happens
        on the loop thread, through speak_rendered, under the ordinary rules.
        """
        return list(self._render_stream(text, expressive=expressive, pace=pace, variation=variation))

    def speak_rendered(self, chunks: list, motion: Optional[Any] = None) -> None:
        """Play chunks render() produced. Loop thread only, like speak()."""
        if motion is not None:
            motion.begin_speech()
        try:
            if self.target.mode == "robot":
                self._speak_robot(chunks, motion)
            else:
                self._speak_local(chunks, motion)
        finally:
            if motion is not None:
                motion.end_speech()
            # Same bookkeeping as speak(): the mic heard the robot's own voice,
            # so the spotter stream and freshness flag are both stale now.
            self._kws_stream = None
            self._mic_fresh = False

    def play_sound(self, name: str, motion: Optional[Any] = None) -> bool:
        """Play a short generated effect. True if anything came out.

        Loop thread only, like speak() -- it writes to the same speaker, and
        two writers interleave into noise.

        Never raises and never blocks for long: an unknown name, a dead link or
        a missing device all return False and leave the turn exactly as it was.
        A robot that fails to play a fanfare should be a robot with no fanfare,
        not a robot that stopped.
        """
        from body import sounds

        clip = sounds.get(name)
        if clip is None or not len(clip):
            return False
        try:
            if self.target.mode == "robot":
                if self._robot is None:
                    return False
                rate = self._robot.media.get_output_audio_samplerate()
                audio = _resample(clip, sounds.SAMPLE_RATE, rate)
                if motion is not None:
                    # So the head moves with the sound rather than sitting
                    # still through it, exactly as it does for speech.
                    motion.feed_speech_audio(clip)
                self._robot.media.push_audio_sample(audio)
                # Returns when the sound has actually finished, because callers
                # sequence the next line against it. Short by construction --
                # sounds.py caps every clip.
                time.sleep(len(audio) / float(rate))
            else:
                import sounddevice as sd

                sd.play(clip, sounds.SAMPLE_RATE)
                sd.wait()
        except Exception:
            logger.debug("Could not play the %r sound", name, exc_info=True)
            return False
        return True

    def _speak_local(self, chunks, motion: Optional[Any]) -> None:
        """Play chunks as they are synthesized, synthesizing ahead of playback.

        sd.play returns immediately, so pulling the NEXT chunk from the
        generator before waiting on the current one means piper works while the
        speaker plays -- the old wait-then-synthesize order put an audible gap
        between every pair of piper's chunks, and the whole utterance took
        synthesis time PLUS playback time instead of whichever is longer.
        Motion is fed at each chunk's playback turn, not at synthesis, so the
        head still moves with what is coming out of the speaker.
        """
        playing = False
        for chunk in chunks:
            if playing:
                sd.wait()
            if motion is not None:
                motion.feed_speech_audio(chunk.audio_float_array)
            sd.play(
                chunk.audio_float_array,
                samplerate=chunk.sample_rate,
                device=self.target.audio_output_device,
            )
            playing = True
        if playing:
            sd.wait()

    def _speak_robot(self, chunks, motion: Optional[Any]) -> None:
        """Stream synthesized audio to the robot, keeping its buffer ahead.

        Pushing a chunk and then sleeping for exactly that chunk's duration
        (the obvious approach) leaves the robot with no buffered audio at any
        point, so every network hiccup lands as an audible gap mid-sentence.
        Instead, run ahead by up to _PLAYBACK_LEAD_S of audio so there's always
        a cushion queued to play through jitter, then sleep off whatever is
        still unplayed so this call returns when the speech actually ends
        (callers rely on that to sequence turns and motion).
        """
        output_rate = self._robot.media.get_output_audio_samplerate()
        started = time.monotonic()
        pushed_s = 0.0

        for chunk in chunks:
            if motion is not None:
                motion.feed_speech_audio(chunk.audio_float_array)
            audio = _resample(chunk.audio_float_array, chunk.sample_rate, output_rate)
            self._robot.media.push_audio_sample(audio)
            pushed_s += len(audio) / output_rate
            ahead = pushed_s - (time.monotonic() - started)
            if ahead > _PLAYBACK_LEAD_S:
                time.sleep(ahead - _PLAYBACK_LEAD_S)

        remaining = pushed_s - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
