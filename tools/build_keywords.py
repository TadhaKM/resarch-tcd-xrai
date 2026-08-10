"""Turn the readable wake-phrase list into the token file the spotter loads.

The spotter does not read words. It reads BPE pieces:

    HEY REACHY   ->   ▁HE Y ▁RE A CH Y

Hand-maintaining that is how a wake phrase ends up silently never matching --
a plausible-looking but wrong split is not an error, it is a keyword nobody can
say. So the phrases live in custom_keywords_raw.txt in English, and this
regenerates custom_keywords.txt from them.

    python tools/build_keywords.py           # rewrite the token file
    python tools/build_keywords.py --check   # verify it is up to date

--check is the useful one in a hurry: it fails if somebody edited the phrase
list and forgot to rebuild, which otherwise shows up as the robot ignoring a
wake phrase that is plainly listed in the file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODELS  # noqa: E402


def parse(raw_text: str) -> list[tuple[str, str | None]]:
    """(phrase, boost) per line. Blank lines and # comments are skipped."""
    entries = []
    for line in raw_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        phrase, _, boost = line.partition(":")
        entries.append((phrase.strip().upper(), boost.strip() or None))
    return entries


def render(entries: list[tuple[str, str | None]], bpe_model: Path) -> str:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(bpe_model))
    lines = []
    for phrase, boost in entries:
        pieces = " ".join(sp.encode(phrase, out_type=str))
        lines.append(f"{pieces} :{boost}" if boost else pieces)
    return "\n".join(lines) + "\n"


def main() -> int:
    raw_path = MODELS.kws_dir / "custom_keywords_raw.txt"
    out_path = MODELS.kws_keywords_file
    entries = parse(raw_path.read_text(encoding="utf-8"))
    rendered = render(entries, MODELS.kws_dir / "bpe.model")

    if "--check" in sys.argv:
        current = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        if current != rendered:
            print(f"STALE: {out_path.name} does not match {raw_path.name}.")
            print("Run: python tools/build_keywords.py")
            return 1
        print(f"OK: {len(entries)} wake phrases, token file up to date.")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {len(entries)} wake phrases to {out_path}.")
    for phrase, boost in entries:
        print(f"  {phrase}" + (f"  (boost {boost})" if boost else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
