#!/usr/bin/env python3
"""Capture Kimi-K2.7-Code RoPE + attention-score goldens from the vendored
reference implementation (`modeling_deepseek.py`, shipped with the checkpoint).

Why this oracle: the vendor's `apply_rotary_pos_emb` de-interleaves with
`view(d//2, 2).transpose(4, 3).reshape(...)` before `rotate_half`, i.e. the
effective rotation is on ADJACENT pairs `(2i, 2i+1)` with `inv_freq[i]`.
`transformers`' DeepseekV3 defaults `rope_interleave = True` for the same
reason. A half-split implementation (pairs `(i, i+32)`) is a different
operator and shifts every attention score.

The function returns the permuted layout, so the output is un-permuted to the
natural element order before it is written. The fixture is plain text with
`%.9g` floats (exact round-trip for float32), which the C++ test parses with
`std::strtof`.

Usage:
  python3 tools/kimi_rope_capture.py \
      --model /mnt/fast_storage/huggingface/Kimi-K2.7-Code \
      --out tests/fixtures/kimi/rope_attn.txt
"""
import argparse
import importlib
import os
import shutil
import sys
import tempfile

import torch


def load_vendored(model_dir):
    """Import the checkpoint's own modeling_deepseek as a package."""
    root = tempfile.mkdtemp(prefix="kimi_vendor_")
    pkg = os.path.join(root, "kimi_vendor")
    os.makedirs(pkg)
    for name in ("modeling_deepseek.py", "configuration_deepseek.py"):
        shutil.copy(os.path.join(model_dir, name), pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    sys.path.insert(0, root)
    return importlib.import_module("kimi_vendor.modeling_deepseek")


def unpermute(y):
    # y = [x0, x2, ..., x62, x1, x3, ..., x63] -> natural [x0, x1, ..., x63]
    n = y.shape[-1]
    out = torch.empty_like(y)
    out[..., 0::2] = y[..., : n // 2]
    out[..., 1::2] = y[..., n // 2 :]
    return out


def fmt(vals):
    return " ".join("%.9g" % float(v) for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = load_vendored(args.model)
    torch.manual_seed(20260824)

    # Config values from config.json text_config (YaRN factor 64, window 4096).
    rope = m.DeepseekV3YarnRotaryEmbedding(
        64,
        max_position_embeddings=262144,
        base=50000.0,
        scaling_factor=64.0,
        original_max_position_embeddings=4096,
        beta_fast=32.0,
        beta_slow=1.0,
        mscale=1.0,
        mscale_all_dim=1.0,
    )
    seq = 9
    cos, sin = rope(torch.zeros(1, 1, seq, 64), seq_len=seq)
    q = torch.randn(1, 1, seq, 64)
    k = torch.randn(1, 1, 1, 64)
    pos = torch.arange(seq).view(1, seq)
    zero = torch.zeros(1, 1, dtype=torch.long)
    qr, _ = m.apply_rotary_pos_emb(q, q.clone(), cos, sin, position_ids=pos)
    _, kr = m.apply_rotary_pos_emb(k, k.clone(), cos, sin, position_ids=zero)
    qn, kn = unpermute(qr), unpermute(kr)

    # Score dots use q at position p and k at position 0: a dot of two vectors
    # rotated at the SAME angle is invariant under the pairing choice, so only
    # a cross-position dot can distinguish the conventions.
    lines = [
        "rope_dim 64",
        "positions %d" % seq,
        "inv_freq " + fmt(rope.inv_freq.tolist()),
        "k_raw " + fmt(k[0, 0, 0].tolist()),
        "k_roped " + fmt(kn[0, 0, 0].tolist()),
    ]
    for p in range(seq):
        lines.append("q_raw %d %s" % (p, fmt(q[0, 0, p].tolist())))
        lines.append("q_roped %d %s" % (p, fmt(qn[0, 0, p].tolist())))
        dot = float((qn[0, 0, p] * kn[0, 0, 0]).sum())
        lines.append("dot %d %.9g" % (p, dot))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
