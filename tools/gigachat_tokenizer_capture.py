#!/usr/bin/env python3
"""Capture GigaChat3.5 tokenizer ids from the checkpoint's OWN tokenizer.

The engine bridge (tokenizer_kind 3, Llama-3/o200k case-run pre-tokenizer) must
reproduce these ids exactly. The corpus is adversarial for the split: case
runs, digit runs of length 1..7, CJK runs, contractions, multi-space
lookahead, punctuation runs, mixed code, multiline text and the chatml-v5
control markers. Ids exclude special tokens (add_special_tokens=False):
like every o200k bridge in this repo, `arrow_tokenizer_encode` is pure BPE and
the chat framing prepends `<s>` explicitly in BuildPromptInputIds.

    python3 tools/gigachat_tokenizer_capture.py \
        --tokenizer /mnt/fast_storage/huggingface/GigaChat3.5-432B-A28B-Reasoning/tokenizer.json \
        --out tests/fixtures/gigachat/tokenizer_parity.jsonl
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
    "user<|role_sep|>\n\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440!<|message_sep|>\n\n",
    "<think></think>",
    "<|message_sep|>\n\n",
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
