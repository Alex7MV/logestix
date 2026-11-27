#!/usr/bin/env python3
"""Capture Hy4-flash tokenizer ids from the checkpoint's OWN tokenizer.

The engine's byte-level BPE bridge (tokenizer_kind 13, HunyuanWordStream) must
reproduce these ids exactly; the golden is captured here, never from the bridge.

    python3 tools/hy4_tokenizer_capture.py \
        --tokenizer /mnt/fast_storage/lsx_model/tencent/Hy4-flash/tokenizer.json \
        --out tests/fixtures/hy4/tokenizer_parity.jsonl

The corpus is deliberately adversarial for this checkpoint's three-split
Sequence: 1-3 digit isolation (so 4+ digit runs split), CJK isolation (Han,
Hiragana, Katakana), the punctuation+ASCII-letter alternative that stands in for
a contraction rule ("don't" -> ["don", "'t"]), the single-codepoint prefix of the
letter alternative ("hello" vs "  hello"), the optional space before a
punctuation run, CRLF runs, the three whitespace forms, the Hunyuan chat/tool
markers (added tokens, matched literally by the reference encoder) and code.
"""
import argparse
import json
import os

CORPUS = [
    # Control prompts the gates use.
    "The capital of France is",
    "What is the present value of a 10-year annuity",
    # Punctuation + ASCII letters (the contraction stand-in).
    "don't won't it's I'LL we'VE",
    "a'b abc'def",
    # Digit isolation at every length.
    "1234567 12345 1 12 123",
    "v2beta FooBarBaz _call __init__ 2.5e-3",
    "2026-10-30T18:42:00Z",
    # CJK isolation and mixed script.
    "中文ab 日本語テキスト",
    "abcあいうえお",
    "汉字。 mixed, 漢字!",
    # Whitespace forms.
    "hello  world   two",
    "  leading",
    "trailing  ",
    "\n\n\n",
    "a\n\nb",
    "line\r\nnext\r\n",
    "a \n b",
    "x " * 4,
    # Punctuation / symbol runs.
    "— \u201cquotes\u201d — naïve café",
    "a+=b>>2 50% #tag @user",
    "/((^&%$#@!",
    "/\n/",
    "a...b",
    '(x, y) => { return x["y"] }',
    # Emoji and symbols.
    "Emoji 😀🚀 done",
    "→ ≈ ≠ ±",
    # Chat / tool markers (added tokens, special:true — matched literally).
    "<｜hy_start:opensource｜>user<｜hy_middle:opensource｜>hi<｜hy_end:opensource｜>",
    "<think:opensource>ok</think:opensource>",
    "<tool_calls:opensource><tool_call:opensource>get_weather"
    "<arg_key:opensource>city</arg_key:opensource>"
    "<arg_value:opensource>Paris</arg_value:opensource>"
    "</tool_call:opensource></tool_calls:opensource>",
    "<tool_response:opensource>{\"ok\": true}</tool_response:opensource>",
    # Code and structured text.
    "def f(x):\n    return x**2\n",
    '{"a": 1, "b": [1,2,3], "c": null}',
    "SELECT * FROM t WHERE id = 42;",
    "x" * 300,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for text in CORPUS:
            ids = list(tok.encode(text, add_special_tokens=False).ids)
            f.write(json.dumps({"text": text, "ids": ids},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(CORPUS)} cases to {args.out}")


if __name__ == "__main__":
    main()
