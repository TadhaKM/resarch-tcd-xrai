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
- `body/camera.py`, `body/face.py`, `body/motion.py` are still stubs.
