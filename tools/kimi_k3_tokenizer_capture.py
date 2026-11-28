#!/usr/bin/env python3
"""Capture Kimi-K3 token ids from the checkpoint's own tokenizer.

The reference is the checkpoint's `TikTokenTokenizer` (tokenization_kimi.py):
tiktoken BPE over `tiktoken.model` with the file's `pat_str`, and the
`special_tokens` dict built from `tokenizer_config.json.added_tokens_decoder`
for every id in [num_base_tokens, num_base_tokens + 256). `encode(text)` adds
no BOS/EOS.

Corpus lines are unescaped (\\n, \\t, \\\\) and encoded as one segment with
`allowed_special="all"` — the same call the reference makes per segment.

The fixture is binary-safe (length-prefixed):

  CASE <byte_len> <id_count> <id...>
  <byte_len raw UTF-8 bytes>

Usage:
  python3 tools/kimi_k3_tokenizer_capture.py \
      --model <source root>/moonshotai/Kimi-K3 \
      --corpus tests/fixtures/kimi_k3/tokenizer_corpus.txt \
      --out tests/fixtures/kimi_k3/tokenizer_parity.txt
"""
import argparse
import json
import os

# Verbatim from the checkpoint's tokenization_kimi.py.
PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


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

    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    vocab_file = os.path.join(args.model, "tiktoken.model")
    ranks = load_tiktoken_bpe(vocab_file)
    num_base = len(ranks)
    with open(os.path.join(args.model, "tokenizer_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    decoder = cfg["added_tokens_decoder"]
    special_tokens = {}
    for i in range(num_base, num_base + 256):
        entry = decoder.get(str(i))
        name = entry["content"] if entry else f"<|reserved_token_{i}|>"
        special_tokens[name] = i
    enc = tiktoken.Encoding(
        name=os.path.basename(vocab_file),
        pat_str=PAT_STR,
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )
    print(f"base ranks {num_base}, specials {len(special_tokens)}")

    cases = []
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            text = unescape(line)
            ids = enc.encode(text, allowed_special="all")
            cases.append((text, ids))

    with open(args.out, "wb") as fh:
        for text, ids in cases:
            raw = text.encode("utf-8")
            fh.write(f"CASE {len(raw)} {len(ids)} ".encode("ascii"))
            fh.write(" ".join(str(i) for i in ids).encode("ascii"))
            fh.write(b"\n")
            fh.write(raw)
            fh.write(b"\n")
    print(f"wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
