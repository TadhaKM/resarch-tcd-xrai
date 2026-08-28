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

# Load .env before anything reads the environment. Done here rather than in
# main.py because this module is imported by everything and read at import time
# by brain/llm_backends.py, which decides there and then whether a cloud model
# is available -- a key loaded any later would arrive after that decision and
# silently do nothing. Missing python-dotenv is not an error: the robot runs on
# the local model, which is the whole point of it.
def _load_env_file() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional dependency
        return
    load_dotenv(env_path, override=False)


_load_env_file()


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


# The daemon's FastAPI port, which has to match what the daemon on the robot
# was actually started with (`reachy-mini-daemon --fastapi-port ...`); the
# stock daemon uses 8000. Overridable per run with REACHY_DAEMON_PORT so a
# robot running the default can be talked to without editing this.
DAEMON_PORT = 8888


SIMULATION = HardwareTarget(
    mode="simulation",
    daemon_host="localhost",
    daemon_port=DAEMON_PORT,
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
    daemon_port=DAEMON_PORT,
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
    daemon_host="172.20.10.3",
    daemon_port=DAEMON_PORT,
    camera_source=None,
    media_backend="webrtc",
    # Same physical mic and daemon pipeline as ROBOT -- see mic_gain above.
    mic_gain=20.0,
    mic_agc=True,
)


ROBOT_HOSTNAME = "reachy-mini.local"


def resolve_robot_host(fallback: str, hostname: str = ROBOT_HOSTNAME) -> str:
    """Return the robot's current address, preferring mDNS over a fixed IP.

    The robot's IP is assigned by whatever network it joins and has changed
    repeatedly mid-session on a phone hotspot -- each time presenting as the
    app hanging or exiting with a connection error, and each time needing this
    file edited. Its mDNS name follows it, so resolve that when it answers and
    keep the last known address only as a fallback for networks where mDNS is
    blocked (which has also happened here).
    """
    import socket

    try:
        resolved = socket.gethostbyname(hostname)
    except OSError:
        return fallback
    return resolved


def default_target() -> HardwareTarget:
    """Return the active HardwareTarget, selected via the REACHY_TARGET env var."""
    import dataclasses
    import os

    port = int(os.getenv("REACHY_DAEMON_PORT") or DAEMON_PORT)
    target = os.getenv("REACHY_TARGET", "simulation")
    if target == "robot":
        return dataclasses.replace(ROBOT, daemon_port=port)
    if target == "robot_remote":
        # Override at use time rather than editing the constant, so the
        # committed fallback stays a record of a known-good address.
        host = os.getenv("REACHY_HOST") or resolve_robot_host(ROBOT_REMOTE.daemon_host)
        return dataclasses.replace(ROBOT_REMOTE, daemon_host=host, daemon_port=port)
    return dataclasses.replace(SIMULATION, daemon_port=port)


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
    #: Tokenized wake phrases. Edit custom_keywords_raw.txt (one phrase per
    #: line, uppercase) and regenerate this with the model's own bpe.model --
    #: the spotter matches token sequences, not text, so a phrase added here
    #: by hand would never fire. Several phrasings are listed because people
    #: do not reliably say the one they were told to: "hi reachy", "okay
    #: reachy" and "excuse me reachy" all now work.
    kws_keywords_file: Path
    kws_threshold: float
    kws_score: float
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

    # Whisper (offline, via the same sherpa-onnx runtime). When set, it decodes
    # what was said; the streaming zipformer above still runs the turn, since
    # it is what detects that speech has ended, and remains the fallback.
    #
    # The tradeoff is deliberate. Whisper decodes a finished utterance rather
    # than streaming, so the reply starts slightly later. Measured on the same
    # audio it was 0.08s slower per utterance and noticeably more accurate
    # ("HOW ARE YOU TO DAY" -> "How are you today?"), and it is trained on
    # varied real-world speech rather than read audiobooks -- which is what
    # the robot's noisy far-field mic actually delivers.
    whisper_dir: Optional[Path] = None
    # 4, not more, and MEASURED rather than assumed: on this 14-logical-CPU
    # laptop, 6 threads decoded the same 2.8s utterance in 0.96s median where
    # 4 threads took 0.56s -- and under a full session's load (zipformer,
    # piper, WebRTC all competing) 6-thread decodes ballooned to 2.4-4.4s,
    # which is what turned the 1.5s endpoint silence into a 3.6s+ wait for the
    # transcript live. More threads is not more speed once they contend.
    whisper_threads: int = 4

    # Cloud LLM, used when a key is present and reachable (see
    # brain/llm_backends.py). The local model above stays the fallback, so the
    # robot still works on a dead network -- which is not hypothetical: the
    # venue wifi is exactly what fails on the day.
    #
    # "auto" picks whichever provider has a key set, preferring Anthropic when
    # both do, and falls back to local when neither does. Pin it to "ollama" to
    # force offline behaviour (useful for demonstrating that it needs no
    # internet), or to a provider name to make a missing key loud rather than
    # silently local.
    llm_backend: str = "auto"

    #: Haiku over Sonnet, one more step down the same slope Sonnet-over-Opus
    #: was: a spoken turn is judged on latency as much as on wording, and the
    #: whole laptop-as-brain design exists to get replies from ~12s down to
    #: ~2s. Sonnet's first token was measuring 2-4.5s live; Haiku roughly
    #: halves that, at some cost in nuance -- a trade being judged by ear at
    #: the Hub. TO GO BACK: "claude-sonnet-5" here and restart. Note the Hub
    #: script and taught answers never touch the model either way; this only
    #: changes the improvised replies.
    anthropic_model: str = "claude-haiku-4-5"
    anthropic_key_env: str = "ANTHROPIC_API_KEY"
    openai_model: str = "gpt-4o-mini"
    openai_key_env: str = "OPENAI_API_KEY"
    #: Optional base URL for anything speaking the OpenAI API (Azure OpenAI,
    #: OpenRouter, Groq, a self-hosted vLLM). Empty means api.openai.com, so
    #: pointing the robot at a different provider is a config change, not code.
    openai_base_url: str = ""

    #: Higher than llm_max_tokens: the cap that protects a slow local model from
    #: a runaway reply is not needed when generation is fast, and a cloud model
    #: is mostly used for the demos that want a fuller answer.
    cloud_max_tokens: int = 400

    def api_key(self, env_name: str) -> str:
        """Read an API key from the environment, or "" when unset."""
        import os

        return os.getenv(env_name, "").strip()


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
    whisper_dir=MODELS_DIR / "asr" / "sherpa-onnx-whisper-base.en",
    # Lower = easier to trigger. 0.15 was tuned against clean synthesized
    # audio and needed shouting over the robot's quiet far-field mic; 0.05 was
    # still not enough. Testing found 0.05-0.2 indistinguishable on clean
    # audio, so nothing is being protected at the top of that range. A false
    # wake costs a moment of listening to nothing, so this errs that way
    # deliberately. Raise it if the robot starts waking on unrelated speech.
    kws_threshold=0.02,
    # How hard the decoder leans toward the wake phrase while searching. The
    # threshold decides whether a match counts; this decides whether the match
    # is found at all, and at the quiet, smeared input a voice across a room
    # produces, that is the one that was losing. Measured against synthesized
    # takes attenuated to stand in for distance, 18 per cell, two seeds: at
    # roughly two metres the default 1.0 scored 15-16/18 and this scores 18/18;
    # at roughly four metres, 4/18 against 11-18/18.
    #
    # 5.0 rather than more because 6.0 and 7.0 started missing the wake word
    # spoken directly into the robot -- the boost widens what counts as the
    # phrase until close, clean speech is no longer the best match for it.
    #
    # The cost is names that genuinely rhyme with it: "Hey Rachel" and "Hey
    # Richie" both wake it now, and no setting tested separated them from the
    # real thing. Left that way on purpose -- a false wake costs a moment of
    # listening to nothing, a missed one costs a visitor repeating themselves
    # at a robot that appears to be ignoring them.
    kws_score=5.0,
    tts_model_path=MODELS_DIR / "tts" / "en_US-amy-medium.onnx",
    tts_config_path=MODELS_DIR / "tts" / "en_US-amy-medium.onnx.json",
    # 127.0.0.1, never "localhost". On Windows that name resolves to IPv6 ::1
    # first, and Ollama listens on IPv4 only -- so every request spent ~2s
    # failing over before it was even seen. Measured: 2.87s vs 0.82s for an
    # identical call, against ~0.4s of actual generation. It looked like a
    # slow model and was entirely name resolution.
    ollama_host="http://127.0.0.1:11434",
    ollama_model=OLLAMA_MODEL_PRIMARY,
    # 140, not the 60 tuned for the CM4's ~1 token/sec. The cap exists so a
    # runaway reply cannot stall a turn; at 60 it was also truncating ordinary
    # answers mid-sentence ("highs around 25-30(" -- observed live) because on
    # the CM4 every token cost a full second. The laptop generates ~25/sec, so
    # 140 buys complete sentences for ~1s of worst-case latency, and the
    # prompt still asks for one-to-two short sentences -- the cap is headroom
    # for the emotion tag, not the brevity mechanism.
    llm_max_tokens=140,
    db_path=Path(__file__).parent / "data" / "memory.db",
    face_detector_model_path=MODELS_DIR / "face" / "blaze_face_short_range.tflite",
    face_embedding_model_path=MODELS_DIR / "face" / "w600k_mbf.onnx",
    face_match_threshold=0.65,
    # 8Hz, not the 4Hz this ran at for most of its life. At 4Hz a quarter of a
    # second passes between one look at the world and the next, and the head
    # moves in visible steps behind somebody walking across the room. The cost
    # is a MediaPipe detection and a face embedding per cycle, which was worth
    # weighing when the same laptop was also generating every reply locally --
    # with the language model on the API that CPU is free, so this is the
    # cheapest available improvement to how alive the robot looks.
    face_detection_fps=8.0,
)
