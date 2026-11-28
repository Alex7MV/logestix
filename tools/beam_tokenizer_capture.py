#!/usr/bin/env python3
"""Capture Beam's tokenizer parity golden from the checkpoint tokenizer.json.

The reference is the checkpoint's own tokenizer (HF `tokenizers`), never the
logestix bridge. Only `tokenizer.json` is read.

    python3 tools/beam_tokenizer_capture.py \
        <source root>/reflection/Beam-1.0-501B-A23B \
        tests/fixtures/beam/tokenizer_parity.jsonl
"""
import lsx_paths
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

CORPUS = [
    "The capital of France is",
    "Hello, world!",
    "getUserName",
    "1234567",
    "v2beta",
    "中文ab",
    "hello  world",
    "a\n\nb",
    "a...b",
    "it's",
    "</tool_call>",
    "<|start|>assistant<|channel|>analysis<|message|>",
    "<|end|><|start|>assistant<|channel|>final<|message|>Paris",
    "<|constrain|>xml",
    "<rfl:parameter name=\"city\">\nParis\n</rfl:parameter>",
    "to=get_weather<|channel|>analysis <|constrain|>xml<|message|>",
    "  leading and trailing  ",
    "\ttabbed\ntext\r\n",
    "def f(x: int) -> int:\n    return x + 1\n",
    "The quick brown fox jumps over the lazy dog.",
    "naïve café — résumé",
    "emoji 😀 test",
    "1, 2, 3, 42",
    "a_b-c/d",
    "Multiple   spaces   here",
    "<|return|>",
    "<|call|>",
    "<|im_start|>user<|im_sep|>hi<|im_end|>",
    "System: you are Beam.",
    "{\"name\": \"search\", \"arguments\": {\"q\": \"巴黎\"}}",
]


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else str(lsx_paths.source_dir("beam")))
    dst = Path(sys.argv[2] if len(sys.argv) > 2
               else "tests/fixtures/beam/tokenizer_parity.jsonl")
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("# Capture: tools/beam_tokenizer_capture.py from "
                "Beam-1.0-501B-A23B/tokenizer.json\n")
        for text in CORPUS:
            ids = tok.encode(text, add_special_tokens=False).ids
            f.write(json.dumps({"text": text, "ids": ids},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(CORPUS)} cases to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
