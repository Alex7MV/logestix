#!/usr/bin/env python3
"""Capture Flint's tokenizer ids from the checkpoint's OWN tokenizer.

The engine's byte-level BPE bridge (tokenizer_kind 4, FlintWordStream) must
reproduce these ids exactly; the golden is captured here, never from the bridge.
The corpus is deliberately adversarial for the three splits: case runs
(camelCase / acronyms), digit runs of length 1..7, CJK runs, contractions,
multi-space lookahead, punctuation runs, the chat/tool markers (added tokens,
matched literally by the reference encoder) and mixed code.

    python3 tools/flint_tokenizer_capture.py \
        --tokenizer <source root>/RecursiveSI/flint/tokenizer.json \
        --out tests/fixtures/flint/tokenizer_parity.jsonl
"""
import argparse
import json
import os

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
    "</tool_call><arg_key>city</arg_key><arg_value>Paris</arg_value>",
    "<|user|>hi<|assistant|><think>ok</think>",
    '<tool_call>get_weather<arg_key>city</arg_key><arg_value>"Paris"</arg_value></tool_call>',
    "def f(x):\n    return x**2\n",
    '{"a": 1, "b": [1,2,3], "c": null}',
    "Emoji 😀🚀 done",
    "— \u201cquotes\u201d — naïve café",
    "a+=b>>2 50% #tag @user",
    "/((^&%$#@!",
    "/\n/",
    "a...b",
    "x" * 300,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizers", action="store_true",
                    help="use the tokenizers library (default: transformers)")
    args = ap.parse_args()

    if args.tokenizers:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(args.tokenizer)
        encode = lambda t: tok.encode(t, add_special_tokens=False).ids
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            os.path.dirname(os.path.abspath(args.tokenizer)),
            trust_remote_code=False,
        )
        encode = lambda t: tok.encode(t, add_special_tokens=False)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for text in CORPUS:
            ids = list(encode(text))
            f.write(json.dumps({"text": text, "ids": ids},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(CORPUS)} cases to {args.out}")


if __name__ == "__main__":
    main()
