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
import time
from typing import Any, Iterator, Optional

import numpy as np
import sherpa_onnx
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig
from scipy.signal import resample

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
        # above the default, since a turn that transcribes one word in five
        # costs far more than half a second of dead air.
        rule2_min_trailing_silence=1.5,
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


def _build_spotter() -> sherpa_onnx.KeywordSpotter:
    d = MODELS.kws_dir
    return sherpa_onnx.KeywordSpotter(
        tokens=str(d / "tokens.txt"),
        encoder=str(d / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
        decoder=str(d / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
        joiner=str(d / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
        keywords_file=str(MODELS.kws_keywords_file),
        keywords_threshold=MODELS.kws_threshold,
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
        self._recognizer = _build_recognizer()
        self._spotter = _build_spotter()
        self._whisper = _build_whisper()
        logger.info(
            "Transcription: %s", "Whisper" if self._whisper is not None else "streaming zipformer"
        )
        self._voice = PiperVoice.load(str(MODELS.tts_model_path), str(MODELS.tts_config_path))

        self._robot: Optional[Any] = robot
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
            input_rate = self._robot.media.get_input_audio_samplerate()
            stalled_since: Optional[float] = None
            while True:
                if deadline is not None and time.monotonic() >= deadline:
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

    def listen(self) -> str:
        """Record and transcribe one utterance.

        Logs the partial transcript as it grows, so it's visible *what* the
        recognizer is picking up and *when* -- a final-only log can't
        distinguish "never heard you" from "heard you and mistranscribed it".
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
                logger.info("  [%5.1fs] hearing: %s", time.monotonic() - started, partial)
                # Streamed to the dashboard so words appear as they are spoken,
                # which is what makes a mishearing visible as it happens rather
                # than only in the final transcript.
                STATE.note("partial", partial)

            if self._whisper is not None:
                captured.append(frame)

            if self._recognizer.is_endpoint(stream):
                text = self._recognizer.get_result(stream).strip()
                self._recognizer.reset(stream)

                # The streaming model still runs the turn -- it is what detects
                # that speech has ended -- but Whisper re-decodes the captured
                # audio for the answer, because it is markedly better on live
                # far-field speech. Its result is preferred only when it found
                # something; an empty Whisper result on audio the streaming
                # model did transcribe is far more likely a decode failure than
                # genuine silence.
                if self._whisper is not None and captured:
                    better = self._transcribe_whisper(np.concatenate(captured))
                    if better:
                        logger.info("  whisper: %s", better)
                        return better
                return text

        # Fell out of the loop: the deadline passed without an endpoint. Return
        # the best transcript so far rather than nothing -- a long answer that
        # ran past the ceiling is still worth answering, and an empty string
        # here would look to the caller like silence.
        logger.warning("Listening hit the %.0fs ceiling without an endpoint.", _MAX_UTTERANCE_S)
        if self._whisper is not None and captured:
            better = self._transcribe_whisper(np.concatenate(captured))
            if better:
                return better
        return self._recognizer.get_result(stream).strip()

    def speak(
        self,
        text: str,
        emotion_tag: str,
        motion: Optional[Any] = None,
        expressive: bool = False,
    ) -> None:
        """Synthesize text with piper and play it. `expressive` performs it as a story."""
        syn_config = _STORY_SYNTHESIS if expressive else _CHAT_SYNTHESIS
        if motion is not None:
            motion.begin_speech()
        try:
            if self.target.mode == "robot":
                self._speak_robot(text, motion, syn_config)
            else:
                self._speak_local(text, motion, syn_config)
        finally:
            if motion is not None:
                motion.end_speech()
            # The mic kept capturing while we spoke, so the backlog now holds
            # the robot's own voice. Dropping the stream makes the next
            # wait_for_wake_word flush that before it starts matching.
            self._kws_stream = None

    def _speak_local(
        self, text: str, motion: Optional[Any], syn_config: SynthesisConfig
    ) -> None:
        for chunk in self._voice.synthesize(text, syn_config=syn_config):
            if motion is not None:
                motion.feed_speech_audio(chunk.audio_float_array)
            sd.play(
                chunk.audio_float_array,
                samplerate=chunk.sample_rate,
                device=self.target.audio_output_device,
            )
            sd.wait()

    def _speak_robot(
        self, text: str, motion: Optional[Any], syn_config: SynthesisConfig
    ) -> None:
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

        for chunk in self._voice.synthesize(text, syn_config=syn_config):
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
