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

    # Non-English prompts are the point of the parity oracle (the model must
    # tokenize arbitrary scripts). They are written as escapes so this source
    # file stays ASCII, per the repository language policy.
    prompts = [
        "The capital of France is",
        "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! \u041a\u0430\u043a \u0434\u0435\u043b\u0430?",
        "def f(x):\n    return x ** 2\n",
        "  leading and trailing  ",
        "emoji \U0001F600 and tabs\there",
        "\u0421\u0442\u0430\u0442\u044c\u044f 105 \u0423\u0433\u043e\u043b\u043e\u0432\u043d\u043e\u0433\u043e \u043a\u043e\u0434\u0435\u043a\u0441\u0430 \u0420\u043e\u0441\u0441\u0438\u0439\u0441\u043a\u043e\u0439 \u0424\u0435\u0434\u0435\u0440\u0430\u0446\u0438\u0438",
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
