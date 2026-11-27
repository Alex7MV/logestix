#!/usr/bin/env python3
"""Capture Kimi-K2.7-Code token ids from the checkpoint's own tokenizer.

The reference encode path is `tok(text)` (identical to `tok.encode(text)` and
`tok(text, add_special_tokens=False)` for this checkpoint): no BOS/EOS is
added. The fixture is binary-safe (length-prefixed), so newlines and control
markers survive:

  CASE <byte_len> <id_count> <id...>
  <byte_len raw UTF-8 bytes>

The corpus is one case per line; `\\n`, `\\t` and `\\\\` are unescaped here so
multi-line text fits on one corpus line.

Usage:
  python3 tools/kimi_tokenizer_capture.py \
      --model /mnt/fast_storage/lsx_model/moonshotai/Kimi-K2.7-Code \
      --corpus tests/fixtures/kimi/tokenizer_corpus.txt \
      --out tests/fixtures/kimi/tokenizer_parity.txt
"""
import argparse
import os

from transformers import AutoTokenizer


def unescape(line):
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            nxt = line[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "t":
                out.append("\t")
                i += 2
                continue
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    cases = []
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh.read().split("\n"):
            if not line or line.startswith("#"):
                continue
            cases.append(unescape(line))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as out:
        for text in cases:
            ids = tok(text).input_ids
            raw = text.encode("utf-8")
            out.write(("CASE %d %d %s\n" % (
                len(raw), len(ids), " ".join(str(i) for i in ids))).encode())
            out.write(raw + b"\n")
    print("wrote", args.out, "cases:", len(cases))


if __name__ == "__main__":
    main()
