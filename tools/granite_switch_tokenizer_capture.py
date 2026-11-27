#!/usr/bin/env python3
"""Capture Granite-Switch tokenizer parity from the checkpoint tokenizer.json."""
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
    "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>",
    "<|start_of_role|>assistant<|end_of_role|>",
    "<|guardian-core|>guardian>As a judge agent",
    "<|decision|> DECISION",
    "<|end_of_text|>\n<|pad|>",
    "  leading and trailing  ",
    "\ttabbed\ntext\r\n",
    "def f(x: int) -> int:\n    return x + 1\n",
    "The quick brown fox jumps over the lazy dog.",
    "naïve café — résumé",
    "emoji 😀 test",
    "1, 2, 3, 42",
    "a_b-c/d",
    "Multiple   spaces   here",
    "drawdown > threshold",
    "Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C",
    "PV=CF*((1-exp(-r*n))/(1-exp(-r)))",
    "{\"score\": \"yes\"}",
]


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1
               else "/mnt/fast_storage/lsx_model/ibm-granite/granite-switch-4.1-30b")
    dst = Path(sys.argv[2] if len(sys.argv) > 2
               else "tests/fixtures/granite_switch/tokenizer_parity.jsonl")
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        f.write("# Capture: tools/granite_switch_tokenizer_capture.py\n")
        for text in CORPUS:
            ids = tok.encode(text, add_special_tokens=False).ids
            f.write(json.dumps({"text": text, "ids": ids},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(CORPUS)} cases to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
