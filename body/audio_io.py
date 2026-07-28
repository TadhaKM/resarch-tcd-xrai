"""Microphone/speaker access: wake word, streaming STT, and TTS.

In "simulation" mode, audio flows through this machine's real mic/speaker via
sounddevice -- see config.py's HardwareTarget docstring for why (this mirrors
reachy_mini_conversation_app's sim-vs-robot audio routing, translated for a
project with no browser in the loop). "robot" mode should instead route
through the daemon's media pipeline (robot.media.*); that's not wired up yet.
"""

from typing import Any, Optional

import numpy as np
import sherpa_onnx
import sounddevice as sd
from piper import PiperVoice

from config import MODELS, HardwareTarget

FRAME_SAMPLES = int(MODELS.asr_sample_rate * 0.1)  # 100ms chunks

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
        decoding_method="modified_beam_search",
        hotwords_file=str(MODELS.asr_hotwords_file),
        hotwords_score=MODELS.asr_hotwords_score,
    )


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


class AudioIO:
    """Wraps mic input (wake word + streaming STT) and speaker output (TTS)."""

    def __init__(self, target: HardwareTarget) -> None:
        self.target = target
        if target.mode != "simulation":
            raise NotImplementedError(
                "AudioIO only talks to this machine's mic/speaker in simulation "
                "mode so far -- robot mode should route through robot.media "
                "instead, per config.py's HardwareTarget docstring."
            )
        self._recognizer = _build_recognizer()
        self._spotter = _build_spotter()
        self._voice = PiperVoice.load(str(MODELS.tts_model_path), str(MODELS.tts_config_path))

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

    # --- live mic/speaker on this machine ---

    def wait_for_wake_word(self) -> None:
        """Block on this machine's real mic until the wake word is heard."""
        stream = self._spotter.create_stream()
        with sd.InputStream(
            samplerate=MODELS.asr_sample_rate,
            channels=1,
            dtype="float32",
            device=self.target.audio_input_device,
            blocksize=FRAME_SAMPLES,
        ) as mic:
            while True:
                samples, _ = mic.read(FRAME_SAMPLES)
                stream.accept_waveform(MODELS.asr_sample_rate, samples[:, 0])
                while self._spotter.is_ready(stream):
                    self._spotter.decode_stream(stream)
                if self._spotter.get_result(stream):
                    self._spotter.reset_stream(stream)
                    return

    def listen(self) -> str:
        """Record and transcribe one utterance from this machine's real mic."""
        stream = self._recognizer.create_stream()
        with sd.InputStream(
            samplerate=MODELS.asr_sample_rate,
            channels=1,
            dtype="float32",
            device=self.target.audio_input_device,
            blocksize=FRAME_SAMPLES,
        ) as mic:
            while True:
                samples, _ = mic.read(FRAME_SAMPLES)
                stream.accept_waveform(MODELS.asr_sample_rate, samples[:, 0])
                while self._recognizer.is_ready(stream):
                    self._recognizer.decode_stream(stream)
                if self._recognizer.is_endpoint(stream):
                    text = self._recognizer.get_result(stream).strip()
                    self._recognizer.reset(stream)
                    return text

    def speak(self, text: str, emotion_tag: str, motion: Optional[Any] = None) -> None:
        """Synthesize text with piper and play it on this machine's real speaker."""
        if motion is not None:
            motion.begin_speech()
        try:
            for chunk in self._voice.synthesize(text):
                if motion is not None:
                    motion.feed_speech_audio(chunk.audio_float_array)
                sd.play(
                    chunk.audio_float_array,
                    samplerate=chunk.sample_rate,
                    device=self.target.audio_output_device,
                )
                sd.wait()
        finally:
            if motion is not None:
                motion.end_speech()
