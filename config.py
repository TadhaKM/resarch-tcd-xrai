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
from typing import Literal, Optional, Union

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
    # Used by body/camera.py. None means the system default camera; an int
    # selects a local camera index; a string may be a device path or stream URL.
    camera_source: Optional[Union[int, str]] = None
    # Used only when mode == "robot": ReachyMini media_backend, mirroring
    # reachy_mini_conversation_app's --wireless-version/--on-device flags.
    # "default" (Lite, USB), "gstreamer" (wireless, on-device),
    # "webrtc" (wireless, fully remote).
    media_backend: str = "default"
    # Linear gain applied to captured mic audio before STT/wake-word decoding.
    # Measured on the Reachy Mini's own mic through the daemon's pipeline:
    # normal speech at conversational distance peaked at 0.04 of full scale
    # (median RMS 0.0009), quiet enough that the wake-word spotter only fired
    # if you shouted. The models want a signal near full scale, so this is a
    # calibration constant for that hardware path, not a preference -- leave
    # it 1.0 for a normal (already correctly-levelled) laptop mic.
    mic_gain: float = 1.0
    # When True, mic_gain is only the fallback used near silence, and the live
    # gain is derived from the signal instead (AudioIO._apply_gain). Needed on
    # the robot, where one speaker's wake word clipped at full scale while
    # their next sentence sat at 0.12 -- no single multiplier covers both.
    mic_agc: bool = False


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
    # This app runs *on* the robot's own CM4, alongside the daemon -- "localhost",
    # not "reachy-mini.local" (that hostname is for a remote client controlling
    # the robot over the network -- see ROBOT_REMOTE below).
    daemon_host="localhost",
    daemon_port=8000,
    camera_source="default",
    # "default", not "webrtc" or the deprecated "gstreamer" name: with
    # daemon_host="localhost", connection_mode resolves to "localhost_only",
    # so the SDK auto-selects LOCAL (reads camera/audio via the daemon's Unix
    # socket / dmix-shared ALSA devices -- no WebRTC/ICE involved). WebRTC
    # over loopback was tried and hit "Network error: Connection timed out"
    # -- ICE doesn't like connecting to yourself. LOCAL only works once the
    # daemon has actually built its media pipeline at least once since boot
    # (its camera IPC socket is created at daemon startup, not before) --
    # if this ever regresses to a connection error again, check whether the
    # daemon needs restarting, not whether this value needs changing back.
    media_backend="default",
    # The daemon's own mic volume is already at 100%, so gain here is the only
    # remaining lever. mic_agc drives it live; this fixed value applies only
    # near silence, where dividing by a tiny envelope would otherwise amplify
    # room noise to speech level.
    mic_gain=20.0,
    mic_agc=True,
)

# A laptop/desktop runs the STT/LLM/TTS pipeline (far more CPU than the CM4 has)
# while the robot itself only serves as the media endpoint -- its own mic,
# speaker, camera, and motors -- reached over the network through the daemon's
# WebRTC transport instead of a local device. Same code path as ROBOT (mode
# stays "robot": audio/camera/motion all still go through robot.media/
# play_move), only daemon_host and media_backend differ, per media_backend's
# docstring above ("webrtc" (wireless, fully remote)).
#
# daemon_host is the robot's current LAN address, not "reachy-mini.local":
# mDNS resolution isn't reliable from this machine (confirmed failing while
# the IP itself still pings fine), and DHCP has already renewed this IP once
# this session -- re-check with `ping reachy-mini.local` or the robot's Wi-Fi
# menu if this stops connecting.
ROBOT_REMOTE = HardwareTarget(
    mode="robot",
    daemon_host="10.142.104.231",
    daemon_port=8000,
    camera_source=None,
    media_backend="webrtc",
    # Same physical mic and daemon pipeline as ROBOT -- see mic_gain above.
    mic_gain=20.0,
    mic_agc=True,
)


def default_target() -> HardwareTarget:
    """Return the active HardwareTarget, selected via the REACHY_TARGET env var."""
    import os

    target = os.getenv("REACHY_TARGET", "simulation")
    if target == "robot":
        return ROBOT
    if target == "robot_remote":
        return ROBOT_REMOTE
    return SIMULATION


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
    # Hard cap on generated tokens (Ollama's num_predict). Measured on the
    # robot's CM4: ~1-1.5 tokens/sec regardless of which small model was
    # tried, so reply length -- not model choice -- is what makes worst-case
    # latency unpredictable (a 73-token reply took 66s). This bounds it.
    llm_max_tokens: int

    # Long-term (cross-session) memory: SQLite in WAL mode. See brain/db.py.
    db_path: Path

    # Face identity: MediaPipe handles detection; this ONNX MobileFaceNet-class
    # recognizer maps the active face crop to a 512-d embedding on CPU.
    face_detector_model_path: Path
    face_embedding_model_path: Path
    face_match_threshold: float
    face_detection_fps: float


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
    # 0.05, not the 0.15 tuned in simulation: lower = easier to trigger. Over
    # the robot's own (very quiet) mic, 0.15 needed shouting. Earlier testing
    # found 0.05-0.2 indistinguishable on clean synthesized audio, so this
    # trades nothing there and buys real sensitivity on weak live input. Raise
    # it back toward 0.15 if "Reachy" starts firing on unrelated speech.
    kws_threshold=0.05,
    tts_model_path=MODELS_DIR / "tts" / "en_US-amy-medium.onnx",
    tts_config_path=MODELS_DIR / "tts" / "en_US-amy-medium.onnx.json",
    ollama_host="http://localhost:11434",
    ollama_model=OLLAMA_MODEL_PRIMARY,
    llm_max_tokens=60,
    db_path=Path(__file__).parent / "data" / "memory.db",
    face_detector_model_path=MODELS_DIR / "face" / "blaze_face_short_range.tflite",
    face_embedding_model_path=MODELS_DIR / "face" / "w600k_mbf.onnx",
    face_match_threshold=0.65,
    face_detection_fps=4.0,
)
