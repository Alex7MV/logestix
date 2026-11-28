#!/usr/bin/env python3
"""Capture Ling-3.1-flash tokenizer ids from the checkpoint's OWN tokenizer.

The engine bridge (tokenizer_kind 3, o200k case-run pre-tokenizer) must
reproduce these ids exactly. The corpus is adversarial for the split: case
runs, digit runs of length 1..7, CJK runs, contractions, multi-space
lookahead, punctuation runs, mixed code and multiline text.

    python3 tools/ling_tokenizer_capture.py \
        --tokenizer <source root>/inclusionAI/Ling-3.1-flash/tokenizer.json \
        --out tests/fixtures/ling/tokenizer_parity.jsonl
"""
import argparse
import json

from tokenizers import Tokenizer


CORPUS = [
    "The capital of France is",
    "getUserName",
    "HTTPRequest",
    "FooBarBaz",
    "v2beta FooBarBaz _call __init__",
    "1234567 12345 1 12 123",
    "2026-10-30T18:42:00Z",
    "中文ab 日本語テキスト",
    "abcあいうえお",
    "hello  world   two",
    "  leading",
    "trailing  ",
    "\n\n\n",
    "a\n\nb",
    "don't won't it's I'LL we'VE",
    "def f(x):\n    return x**2\n",
    '{"a": 1, "b": [1,2,3], "c": null}',
    "Emoji 😀🚀 done",
    "— \u201cquotes\u201d — naïve café",
    "a+=b>>2 50% #tag @user",
    "/((^&%$#@!",
    "/\n/",
    "a...b",
    "x" * 300,
    "  mixed   spaces\tand\ttabs  ",
    "numbers 42 4 2 42 424242",
    "Eine Straße in Zürich, 2026.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    with open(args.out, "w", encoding="utf-8") as f:
        for text in CORPUS:
            ids = tok.encode(text, add_special_tokens=False).ids
            f.write(json.dumps({"text": text, "ids": ids}, ensure_ascii=False) + "\n")
    print(f"wrote {len(CORPUS)} cases to {args.out}")


if __name__ == "__main__":
    main()
