#!/usr/bin/env python3
"""Capture Hy-Memory-30B-A3B tokenizer-parity cases from the checkpoint's own
tokenizer (transformers), so the engine's BPE bridge can be compared against
the reference instead of against itself.

Format: JSON Lines, one object per case:

    {"text": "...", "ids": [1, 2, 3]}

Usage:
    tools/hy_tokenizer_capture.py --model <dir> --out <file>
"""
import argparse
import json
import os
import sys

# Adversarial for the PRE-tokenizer (digit runs of <=3, CJK isolation, the
# 6-alternative main pattern) and for the two vocabulary quirks this checkpoint
# has: no CR byte token (the reference DROPS "\r") and no single-byte token for
# the 13 UTF-8-illegal lead bytes.
CORPUS = [
    "The capital of France is",
    "Compute the present value PV of a bond with cash flow CF, rate r, "
    "and n periods. Return the value.",
    "getUserName and setHTTPHeader and parseXMLResponse",
    "def compute_pv(cash_flow, rate):\n    return cash_flow * rate",
    "    indented code line with   multiple   spaces",
    "Numbers 1234567 and 3.14159 and -42 and 0x1F and 1e-9",
    'JSON {"key": [1, 2, 3], "nested": {"a": true, "b": null}}',
    "Привет, мир! 你好世界 こんにちは",
    "<｜hy_User｜>Hello<｜hy_Assistant｜><think></think>",
    "<tool_calls>\n<tool_call>get_weather<tool_sep>\n<arg_key>city</arg_key>"
    "\n<arg_value>Paris</arg_value>\n</tool_call>\n</tool_calls>",
    "don't won't they're I'll we've you'd it's",
    "line one  \nline two   \n\nline three",
    "carriage\rreturn and\r\rraw CR",
    "a\rb",
    "ALLCAPS mixedCASE UPPER lower TitleCase snake_case kebab-case",
    "Drawdown > Threshold && Rate_A * 100000000 - Slip_B",
    "The quick brown fox jumps over the lazy dog 0123456789.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# captured from %s (transformers %s)\n"
                 % (args.model, transformers.__version__))
        for text in CORPUS:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            fh.write(json.dumps({"text": text, "ids": ids},
                                ensure_ascii=False) + "\n")
    print("wrote", args.out, "with", len(CORPUS), "cases")


if __name__ == "__main__":
    sys.exit(main())
