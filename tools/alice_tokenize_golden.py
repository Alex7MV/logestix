#!/usr/bin/env python3
"""Capture SentencePiece-BPE tokenizer goldens for AliceAI.

Reference implementation: the real `sentencepiece` SentencePieceProcessor
loaded from the checkpoint's tokenizer.model. This is an independent encoder
(not a reconstruction), so the committed goldens are a genuine parity oracle
for the C++ SP-BPE backend. `add_bos_token=true` in tokenizer_config.json is
mirrored by prepending bos id 1, matching `arrow_tokenizer_encode`.

Usage:
    PYTHONPATH=<spm> python3 tools/alice_tokenize_golden.py <model_dir> \
        --out tests/fixtures/tokenizer/alice.json
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        print("sentencepiece is required to capture goldens", file=sys.stderr)
        return 2

    model = f"{a.model_dir}/tokenizer.model"
    sp = spm.SentencePieceProcessor(model_file=model)
    bos = sp.bos_id()

    prompts = [
        "The capital of France is",
        "Привет, мир! Как дела?",
        "def f(x):\n    return x ** 2\n",
        "  leading and trailing  ",
        "emoji \U0001F600 and tabs\there",
        "Статья 105 Уголовного кодекса Российской Федерации",
        "1 2 3 4 5 6 7 8 9 10",
        "",
    ]

    cases = []
    for text in prompts:
        ids = ([bos] if bos >= 0 else []) + sp.encode(text)
        decoded = sp.decode(sp.encode(text))
        cases.append({"text": text, "ids": ids, "decoded": decoded})

    out = {
        "bos_id": bos,
        "eos_id": sp.eos_id(),
        "vocab_size": sp.get_piece_size(),
        "cases": cases,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {a.out}: {len(cases)} cases, vocab {out['vocab_size']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
