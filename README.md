# reachy_companion

A brain/body-separated Reachy Mini voice assistant prototype.

- `brain/` -- all logic, zero hardware dependency: LLM wrapper, per-person
  memory, prompt templates, emotion-tag parsing. Public interface:
  `brain.interface.get_reply(person_id, message) -> (reply_text, emotion_tag)`.
- `body/` -- everything that talks to hardware: mic/speaker (wake word,
  streaming STT, TTS), camera, motion. Reads all device/model choices from
  `config.py`; nothing hardcoded.
- `config.py` -- one `HardwareTarget` (simulation vs. robot; sim uses this
  machine's real mic/speaker via `sounddevice`, robot mode is a documented
  stub for routing through the daemon's `robot.media` instead) and one
  `ModelConfig` (STT/wake-word/TTS/LLM model paths and the Ollama model name).

## Setup

```powershell
pip install sherpa-onnx piper-tts sounddevice sentencepiece pypinyin ollama
pip install mediapipe opencv-python onnxruntime
```

Install Ollama (https://ollama.com/download) and pull the models:

```powershell
ollama pull qwen2.5:1.5b-instruct-q4_K_M
ollama pull llama3.2:1b-instruct-q4_K_M   # documented fallback, see config.py
```

### Model files (`models/`, gitignored -- ~700MB, not tracked)

```bash
mkdir -p models/asr models/kws models/tts

# STT: sherpa-onnx streaming zipformer, English, int8 (~68MB)
curl -L -o models/asr/zipformer-en.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xjf models/asr/zipformer-en.tar.bz2 -C models/asr/

# Wake word: sherpa-onnx KWS, open-vocabulary (~5MB)
curl -L -o models/kws/kws-gigaspeech.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01.tar.bz2
tar xjf models/kws/kws-gigaspeech.tar.bz2 -C models/kws/

# TTS: piper, English, medium quality (~63MB)
curl -L -o models/tts/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -L -o models/tts/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# Face detection: MediaPipe BlazeFace short-range (~225KB)
mkdir -p models/face
curl -L -o models/face/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite

# Face embedding: InsightFace buffalo_s/buffalo_sc MobileFaceNet recognizer (~13.6MB)
curl -L -o models/face/w600k_mbf.onnx \
  https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx
```

Then generate the custom wake word and ASR hotword token files (bias the
models toward the invented word "Reachy" and the "Hey Reachy" wake phrase --
see the `kws_threshold` / `asr_hotwords_score` docstrings in `config.py` for
why these values are what they are):

```bash
cd models/kws/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
echo "HEY REACHY" > custom_keywords_raw.txt
sherpa-onnx-cli text2token --tokens tokens.txt --tokens-type bpe --bpe-model bpe.model \
  custom_keywords_raw.txt custom_keywords.txt

cd ../../asr/sherpa-onnx-streaming-zipformer-en-2023-06-26
echo "REACHY" > hotwords_raw.txt
sherpa-onnx-cli text2token --tokens tokens.txt --tokens-type bpe --bpe-model bpe.model \
  hotwords_raw.txt hotwords.txt
```

## Testing

```powershell
python test_stt_reliability.py   # STT + wake-word correctness, synthesized audio
python test_full_loop.py         # wake -> transcribe -> generate -> speak, against a live daemon
```

Both scripts synthesize test speech with piper rather than using a live mic --
a reproducible way to validate the models/config are wired correctly. They
don't replace testing with a real human voice.

## People management

Face detection uses `config.HardwareTarget.camera_source`; `None` selects the
system default camera, an integer selects a local camera index, and a string can
point at a device path or stream URL. Detection is throttled by
`MODELS.face_detection_fps` and skips frames above that rate. The largest
detected face is treated as the active conversation partner.

Unknown faces are not enrolled just because they appear in a frame. Enrollment
happens only after the person starts a conversation; Reachy asks for their name,
then stores one embedding in SQLite. Matching uses cosine similarity with
`MODELS.face_match_threshold` initially set to `0.65`.

```powershell
python manage_people.py list
python manage_people.py delete 3
```

## Motion and personality

`body/motion.py` runs a single motion-control loop that composes a fixed
emotion pose, idle animation, face-tracking offset, and speech wobble into one
`ReachyMini.set_target(...)` command stream. Speech overrides idle motion, and
face tracking suppresses idle look-around while a face target is fresh.

The supported emotion poses match the brain tags: `neutral`, `happy`,
`curious`, `thinking`, `surprised`, and `sad`. Poses are tunable in
`EmotionMapper`.

To demonstrate the personality layer in Reachy Mini Control's 3D viewer:

```powershell
python demo_motion.py
```

## Known limitations

- "Reachy" is out-of-vocabulary for the (LibriSpeech-trained) STT model.
  Hotword biasing helps some but can't be pushed hard without causing
  repetition artifacts -- capped conservatively in `config.py`. The wake word
  itself doesn't depend on this: KWS matches "Hey Reachy" reliably because
  it's constrained keyword matching, not free-form transcription.
- `body/`'s hardware-facing classes only support `HardwareTarget.mode ==
  "simulation"` so far (this machine's real mic/speaker via `sounddevice`).
  `mode == "robot"` raises `NotImplementedError` -- routing audio through the
  daemon's `robot.media` pipeline instead is not yet implemented.
- `body/motion.py` is still a stub.
- Face recognition is wired for simulation camera frames, but threshold/accuracy
  still need validation with the actual robot camera and real enrolled people.
