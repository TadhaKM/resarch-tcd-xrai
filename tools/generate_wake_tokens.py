"""Regenerate the wake-word and hotword token files from assets/wake_phrases.txt.

The phrase list lives in assets/ (tracked in git) because everything under
models/ is gitignored -- without this split, a fresh clone silently loses the
wake phrases and the robot answers to nothing but whatever shipped with the
KWS model.

The spotter matches BPE token sequences, not text, so each phrase must be
encoded with the model's own bpe.model, and every piece checked against its
tokens.txt: a phrase containing an unknown token never fires, silently.
Run this after editing assets/wake_phrases.txt, then restart the app.
"""

import sys
from pathlib import Path

import sentencepiece as spm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from config import MODELS  # noqa: E402


def tokenize_into(model_dir: Path, phrases: list[str], out_name: str, raw_name: str) -> int:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_dir / "bpe.model"))
    known = {
        line.split()[0]
        for line in (model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines()
        if line.split()
    }

    kept: list[tuple[str, str]] = []
    for phrase in phrases:
        pieces = sp.encode(phrase, out_type=str)
        missing = [p for p in pieces if p not in known]
        if missing:
            print(f"  DROPPED {phrase!r}: unknown tokens {missing}")
            continue
        kept.append((phrase, " ".join(pieces)))

    (model_dir / raw_name).write_text(
        "".join(p + "\n" for p, _ in kept), encoding="utf-8"
    )
    (model_dir / out_name).write_text(
        "".join(t + "\n" for _, t in kept), encoding="utf-8"
    )
    for phrase, _ in kept:
        print(f"  ok {phrase}")
    return len(kept)


def main() -> int:
    phrases = [
        line.strip().upper()
        for line in (REPO / "assets" / "wake_phrases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not phrases:
        print("assets/wake_phrases.txt is empty -- refusing to wipe the keyword file.")
        return 1

    print(f"Wake phrases -> {MODELS.kws_dir.name}:")
    n = tokenize_into(MODELS.kws_dir, phrases, "custom_keywords.txt", "custom_keywords_raw.txt")

    # ASR hotword biasing: "Reachy" is an invented word the STT model has
    # never seen; biasing nudges the decoder toward it in transcripts.
    print(f"Hotwords -> {MODELS.asr_dir.name}:")
    tokenize_into(MODELS.asr_dir, ["REACHY"], "hotwords.txt", "hotwords_raw.txt")

    print(f"\n{n} wake phrase(s) active. Restart the app to pick them up.")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
