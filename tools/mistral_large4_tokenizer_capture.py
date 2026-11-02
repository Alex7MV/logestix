#!/usr/bin/env python3
"""Capture the Mistral-Large-4 tokenizer golden (Model Ingestion Protocol
stage 4, tokenizer bridge).

The golden is produced by the checkpoint's OWN tokenizer.json (the `tokenizers`
library), never by the engine's bridge, so the parity test is an external oracle
for the new pre-tokenizer kind (o200k case runs without contractions, digits
grouped 1..3 aligned to the right end of a run).

Usage:
    python3 tools/mistral_large4_tokenizer_capture.py \
        /mnt/fast_storage/huggingface/Mistral-Large-4-1T-A52B \
        tests/fixtures/mistral_large4/tokenizer_parity.jsonl
"""
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

CORPUS = [
    # Number runs of every length 1..9 (right-grouped).
    "1", "12", "123", "1234", "12345", "123456", "1234567", "12345678",
    "123456789", "a1234", "v2beta", "3.14159", "1,000,000", "2026-11-02",
    "x0y00z000",
    # Case runs / acronyms / mixed scripts.
    "getUserName", "HTTPRequest", "iPhone15Pro", "XMLHttpRequest",
    "Maße", "ÄÖÜäöü", "приветМир", "Σίσυφος",
    # Contractions are NOT peeled (no contraction alternative).
    "it's", "don't", "we'll", "I'M",
    # CJK is not isolated by this pattern.
    "中文ab", "abcあいうえお",
    # Punctuation runs, slash and CRLF tails, whitespace lookahead.
    "a...b", "a//\n", "a/b", "hello  world", "a\n\nb", "a \n b", "\t\tx",
    "()[]{}", "!!!", "a/b/c", "\n\n\n",
    # Chat markers (added/special tokens are matched literally).
    "[INST]What is 2+2?[/INST]", "<s>[SYSTEM_PROMPT]be brief[/SYSTEM_PROMPT]",
    "[TOOL_CALLS]get_weather[ARGS]{\"city\":\"Paris\"}",
    "[AVAILABLE_TOOLS][]" "[/AVAILABLE_TOOLS]",
    "[MODEL_SETTINGS]{\"reasoning_effort\": \"high\"}[/MODEL_SETTINGS]",
    "[THINK]plan[/THINK]",
    # Mixed realistic prompt text.
    "The capital of France is Paris.",
    "def f(x):\n    return x * 2\n",
    "SELECT * FROM t WHERE id = 42;",
    "Ünicode text with 123 numbers and 中文字符.",
]


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    model_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        f.write("# captured from %s/tokenizer.json (tokenizers %s)\n"
                % (model_dir.name, __import__("tokenizers").__version__))
        for text in CORPUS:
            ids = tok.encode(text, add_special_tokens=False).ids
            f.write(json.dumps({"text": text, "ids": ids},
                               ensure_ascii=False) + "\n")
            n += 1
    print("wrote %d cases to %s" % (n, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
