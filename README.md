# reachy_companion

A voice assistant for the Reachy Mini Wireless: wake it by voice, talk to it,
ask it to dance, and watch what it hears from a phone. Fully offline -- STT,
TTS, and the LLM all run locally; nothing leaves the machines involved.

The deployment that works best in practice is **laptop-as-brain**: speech
recognition (Whisper), the LLM (Ollama), and TTS run on a laptop, while the
robot supplies its mic, speaker, camera, and motors over the daemon's WebRTC
transport. The robot's CM4 generates ~1 token/sec; a laptop is ~25x faster,
which is the difference between 12s and ~2s to the first spoken word.

## Running it

```powershell
cd reachy_companion
.\start_reachy.ps1            # laptop-as-brain (default)
.\start_reachy.ps1 -OnRobot   # everything on the robot's CM4 instead
```

The script checks the robot is reachable, reclaims its camera/mic from the
daemon, stops any stale instance, and supervises the app -- if the robot
connection dies mid-session (see "Hardware findings"), the app exits with
code 3 and the script relaunches it automatically.

- **Dashboard**: `http://<this-machine>:8080` from any device on the same
  network -- live transcript (with a Logs toggle showing wake-word activity
  and partial recognition as you speak), speaker volume, and mode switching:
  Conversation / Greeter / Dance / Idle. Every mode keeps listening for the
  wake word; a mode changes what the robot does while waiting, never whether
  it answers.
- **Console transcript**: `.\watch_reachy.ps1` tails the same events in a
  terminal.

Say any listed wake phrase ("Hey Reachy", "Hi Reachy", "Okay Reachy", ... --
the dashboard shows the full list) and ask your question. One wake word per
question. "Turn off" / "go to sleep" / "goodbye" puts it to sleep; any wake
phrase wakes it. Asking it to dance dances immediately, no LLM involved.

## Layout

- `brain/` -- hardware-independent: Ollama wrapper, prompts (including an
  explicit list of what the robot can and cannot do, so it answers those
  questions truthfully), emotion-tag parsing, per-person SQLite memory,
  shared robot state for the dashboard (`brain/modes.py`).
- `body/` -- hardware-facing: audio in/out (`audio_io.py`), camera, face
  detection/tracking (`face.py`, `face_tracker.py`), motion (`motion.py`),
  and the main loop (`voice_loop.py`).
- `web/` -- the dashboard (FastAPI + one static page, no build step).
- `config.py` -- `HardwareTarget` (SIMULATION / ROBOT / ROBOT_REMOTE) and
  `ModelConfig`. `REACHY_TARGET` selects the target; `REACHY_HOST` overrides
  the robot address.

## Setup

```powershell
pip install sherpa-onnx piper-tts sounddevice sentencepiece ollama
pip install mediapipe opencv-python onnxruntime fastapi uvicorn requests
```

Install Ollama (https://ollama.com/download), then:

```powershell
ollama pull qwen2.5:1.5b-instruct-q4_K_M
```

### Model files (`models/`, gitignored)

```bash
mkdir -p models/asr models/kws models/tts models/face

# Streaming STT (turn-taking + live partials): sherpa-onnx zipformer (~68MB)
curl -L -o models/asr/zipformer-en.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xjf models/asr/zipformer-en.tar.bz2 -C models/asr/

# Transcription: Whisper base.en int8 (~67MB). The streaming model detects
# when you stop speaking; Whisper re-decodes the utterance for accuracy.
curl -L -o models/asr/whisper-base.en.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2
tar xjf models/asr/whisper-base.en.tar.bz2 -C models/asr/

# Wake word: sherpa-onnx KWS, open vocabulary (~5MB)
curl -L -o models/kws/kws-gigaspeech.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
tar xjf models/kws/kws-gigaspeech.tar.bz2 -C models/kws/

# TTS: piper en_US-amy-medium (~63MB)
curl -L -o models/tts/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -L -o models/tts/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# Face detection + embedding
curl -L -o models/face/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
curl -L -o models/face/w600k_mbf.onnx \
  https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx
```

Recorded emotion gestures come from the HuggingFace dataset
`pollen-robotics/reachy-mini-emotions-library` (fetched and cached on first
run; several moves are mapped per emotion and picked at random per turn).

### Wake phrases

`models/kws/.../custom_keywords_raw.txt` holds the phrases (uppercase, one
per line). The spotter matches BPE token sequences, not text, so after
editing it the tokenized file must be regenerated with the model's own
`bpe.model` -- encode each line with sentencepiece and verify every piece
exists in `tokens.txt` (a phrase with unknown tokens can never fire). The
dashboard reads the raw file, so it always shows what is actually active.

## Hardware findings (why parts of this code look the way they do)

Learned on the real robot; each is documented at the relevant code site.

- **One `ReachyMini` connection, shared.** A `media_backend="no_media"`
  connection makes the SDK call `release_media()` -- a *daemon-wide*
  teardown that deletes the camera IPC socket and unregisters the WebRTC
  producer, breaking any other connection's media. Audio, camera, and motion
  therefore share a single connection (`voice_loop.run_forever`).
- **The SDK cannot reconnect.** Its websocket client declares the link dead
  after 1s without a status message and `disconnect()` is terminal. On a
  congested network the link dies for good; the app detects sustained
  failure, exits with code 3, and the launcher rebuilds the session (~70s).
  A phone hotspot (seen swinging 47ms-1400ms RTT) triggers this constantly;
  a normal router mostly doesn't.
- **Pitch is negated at the SDK boundary** (`motion._create_head_pose`). The
  robot's convention is positive-pitch-down; every pose in `motion.py` is
  written positive-up. A permanent upward camera bias is added *on top of*
  the current emotion pose -- the lens sits below face height, and tying aim
  to mood meant "sad" pointed the camera at the floor.
- **Mic gain is adaptive (AGC), and the noise gate only applies to
  questions.** The robot's mic delivers ~4% of full scale; one speaker's
  wake word clipped while their next sentence sat at 0.12, so no fixed gain
  works. The gate that stops room noise becoming a "question" is disabled
  while waiting for the wake word -- gating there meant shouting to wake it.
- **Whisper labels non-speech** ("(tapping)", "*sad noises*"); those
  annotations are stripped before the LLM sees them, else the robot
  earnestly replies to chair scrapes.
- **MediaPipe hard-crashes the CM4** (SIGILL: the binary wants an ARM crypto
  extension the BCM2711 lacks). `face.py` probes it in a throwaway
  subprocess and disables face features on hardware where it dies, so
  on-robot deployments run face-blind instead of crashing.
- **The daemon's advertised `wlan_ip` can go stale** after network changes
  (it once advertised its old access-point address; WebRTC signalling then
  times out while ping/SSH work). Fix: restart the daemon. The app resolves
  `reachy-mini.local` at startup, with the last-known IP as fallback and
  `REACHY_HOST` as an override.

## People

Face *detection/tracking* aims the head at whoever is in view (with a slow
search sweep when nobody is). Face *identification* exists but enrollment by
voice was removed -- noisy-room transcripts became permanent names bound to
face embeddings ("Have Come", person 9). Enroll deliberately instead:

```powershell
python manage_people.py list
python manage_people.py enroll "Ada"    # captures the face currently in view
python manage_people.py delete 3
```

## Testing without hardware

`test_stt_reliability.py` and `test_full_loop.py` synthesize speech with
piper and exercise the recognizers offline. The emotion-tag parser and AGC
have inline verification snippets in their commit history; the voice loop
itself needs the robot.
