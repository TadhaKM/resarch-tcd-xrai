"""Hardware target and model configuration.

Everything that differs between simulation and the real robot lives in
HardwareTarget. body/ modules take one of these instead of hardcoding a
daemon address, device name, or model path, so switching targets or models
never touches logic.

Audio routing follows the same pattern used by pollen-robotics'
reachy_mini_conversation_app (and dwain-barnes' local-inference fork of it):
sim-vs-robot is not a choice of *which local device* to open, it's a choice of
*transport*.

- Their apps: simulated robots have no physical mic/speaker, so audio is
  carried over a browser tab via FastRTC/WebRTC (robot.client.get_status()
  ["simulation_enabled"] gates this, and forces --gradio in that case).
  Against a real robot, audio instead flows through the daemon's own media
  pipeline (`robot.media.start_recording()/get_audio_sample()/
  push_audio_sample()`), via LocalStream.
- We have no browser here, so HardwareTarget.mode plays the same gating role
  their simulation_enabled check does, but routes to this machine's real
  audio hardware directly (sounddevice) instead of a browser tab: mode ==
  "simulation" -> sounddevice using audio_input_device/audio_output_device
  below; mode == "robot" -> the daemon's media pipeline via media_backend.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

Mode = Literal["simulation", "robot"]

MODELS_DIR = Path(__file__).parent / "models"


@dataclass(frozen=True)
class HardwareTarget:
    mode: Mode
    daemon_host: str
    daemon_port: int
    # Used only when mode == "simulation": sounddevice device name/index for
    # this machine's real mic/speaker. None = system default device.
    audio_input_device: Optional[str] = None
    audio_output_device: Optional[str] = None
    camera_source: Optional[str] = None
    # Used only when mode == "robot": ReachyMini media_backend, mirroring
    # reachy_mini_conversation_app's --wireless-version/--on-device flags.
    # "default" (Lite, USB), "gstreamer" (wireless, on-device),
    # "webrtc" (wireless, fully remote).
    media_backend: str = "default"


SIMULATION = HardwareTarget(
    mode="simulation",
    daemon_host="localhost",
    daemon_port=8000,
    audio_input_device=None,
    audio_output_device=None,
    camera_source=None,
)

ROBOT = HardwareTarget(
    mode="robot",
    daemon_host="reachy-mini.local",
    daemon_port=8000,
    camera_source="default",
    media_backend="webrtc",
)


def default_target() -> HardwareTarget:
    """Return the active HardwareTarget, selected via the REACHY_TARGET env var."""
    import os

    return ROBOT if os.getenv("REACHY_TARGET", "simulation") == "robot" else SIMULATION


@dataclass(frozen=True)
class ModelConfig:
    """Which STT/wake-word/TTS/LLM models to load. Swap without touching brain/body code."""

    # STT: sherpa-onnx streaming zipformer, English, int8 (~68MB total),
    # trained on LibriSpeech. https://github.com/k2-fsa/sherpa-onnx
    asr_dir: Path
    asr_sample_rate: int
    # "Reachy" is an invented word, out-of-vocabulary for a LibriSpeech-trained
    # model -- hotword biasing (built into sherpa-onnx) fixes this without
    # retraining. See models/asr/.../hotwords_raw.txt and hotwords.txt.
    asr_hotwords_file: Path
    asr_hotwords_score: float

    # Wake word: sherpa-onnx open-vocabulary keyword spotting (same toolkit as
    # STT, ~5MB int8). Custom phrases are added by tokenizing them with the
    # model's own bpe.model via `sherpa-onnx-cli text2token` -- see
    # models/kws/.../custom_keywords_raw.txt and custom_keywords.txt.
    kws_dir: Path
    kws_keywords_file: Path
    kws_threshold: float
    """Tuned empirically against synthesized "Hey Reachy" audio (see
    test_stt_reliability.py): 0.15 gave the best recall (13/15) with zero
    false positives on the test-phrase set. Values from 0.05-0.2 performed
    identically once tail padding was correct -- the remaining misses are a
    genuine model/voice-variance limit, not a threshold tuning issue."""

    # TTS: piper, English, medium quality (not high, per size/latency tradeoff).
    tts_model_path: Path
    tts_config_path: Path

    # LLM: served locally by Ollama (http://localhost:11434).
    # Primary: Qwen2.5-1.5B-Instruct Q4_K_M -- strong instruction-following
    # for its size (~1GB).
    # Documented fallback: Llama-3.2-1B-Instruct Q4_K_M -- smaller/faster,
    # slightly weaker instruction-following (~0.8GB). Switch by pointing
    # ollama_model at OLLAMA_MODEL_FALLBACK below; no code changes needed.
    ollama_host: str
    ollama_model: str

    # Long-term (cross-session) memory: SQLite in WAL mode. See brain/db.py.
    db_path: Path


OLLAMA_MODEL_PRIMARY = "qwen2.5:1.5b-instruct-q4_K_M"
OLLAMA_MODEL_FALLBACK = "llama3.2:1b-instruct-q4_K_M"

MODELS = ModelConfig(
    asr_dir=MODELS_DIR / "asr" / "sherpa-onnx-streaming-zipformer-en-2023-06-26",
    asr_sample_rate=16000,
    asr_hotwords_file=MODELS_DIR
    / "asr"
    / "sherpa-onnx-streaming-zipformer-en-2023-06-26"
    / "hotwords.txt",
    # NOTE: pushing this much higher does *not* reliably fix "Reachy" itself
    # (an invented, out-of-vocabulary proper noun) -- testing showed scores
    # above ~30 cause the decoder to hallucinate repeated hotword tokens
    # instead. Left conservative; the wake word doesn't depend on this at all
    # since it goes through the separate KWS model (see kws_* above), which
    # matches "Reachy" reliably because it's constrained keyword matching,
    # not free-form transcription.
    asr_hotwords_score=3.0,
    kws_dir=MODELS_DIR / "kws" / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
    kws_keywords_file=MODELS_DIR
    / "kws"
    / "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
    / "custom_keywords.txt",
    kws_threshold=0.15,
    tts_model_path=MODELS_DIR / "tts" / "en_US-amy-medium.onnx",
    tts_config_path=MODELS_DIR / "tts" / "en_US-amy-medium.onnx.json",
    ollama_host="http://localhost:11434",
    ollama_model=OLLAMA_MODEL_PRIMARY,
    db_path=Path(__file__).parent / "data" / "memory.db",
)
