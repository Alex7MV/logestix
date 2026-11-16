#!/usr/bin/env python3
"""Capture MiniMax-M3 tokenizer-parity cases from the checkpoint's own
tokenizer (transformers), so the engine's BPE bridge can be compared against
the reference instead of against itself.

Format: JSON Lines, one object per case (newlines inside `text` are escaped,
so a multi-line prompt cannot corrupt the fixture):

    {"text": "...", "ids": [1, 2, 3]}

Usage:
    tools/minimax_tokenizer_capture.py --model <dir> --out <file>
"""
import argparse
import json
import os
import sys

# Deliberately adversarial for the PRE-tokenizer: the MiniMax checkpoint uses
# the Llama-3/o200k-style 6-alternative pattern (case runs, <=3-digit groups,
# multi-space lookahead), which is NOT the GPT-2/cl100k split the engine's
# legacy WordStream implements.
CORPUS = [
    "The capital of France is",
    "Compute the present value PV of a bond with cash flow CF, rate r, "
    "and n periods. Return the value.",
    "getUserName and setHTTPHeader and parseXMLResponse",
    "def compute_pv(cash_flow, rate):\n    return cash_flow * rate",
    "    indented code line with   multiple   spaces",
    "Numbers 1234567 and 3.14159 and -42 and 0x1F and 1e-9",
    'JSON {"key": [1, 2, 3], "nested": {"a": true, "b": null}}',
    "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! 你好世界 こんにちは",
    "]~b]user\nHello, world[e~[\n]<]minimax[>[<tool_call>",
    "don't won't they're I'll we've you'd it's",
    "line one  \nline two   \n\nline three",
    "ALLCAPS mixedCASE UPPER lower TitleCase snake_case kebab-case",
    "Drawdown > Threshold && Rate_A * 100000000 - Slip_B",
    "<|im_start|> is not a marker here; <mm:think> is.",
]


def _transformers_version():
    import transformers
    return transformers.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("# captured from %s (transformers %s)\n"
                 % (args.model, _transformers_version()))
        for text in CORPUS:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            fh.write(json.dumps({"text": text, "ids": ids},
                                ensure_ascii=False) + "\n")
    print("wrote", args.out, "with", len(CORPUS), "cases")


if __name__ == "__main__":
    sys.exit(main())
