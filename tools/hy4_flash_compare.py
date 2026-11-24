#!/usr/bin/env python3
"""Compare a Hy4-flash engine layer dump against the reference's, boundary by boundary.

Both sides write raw fp32 for the SAME tensor at the SAME point in the stack:

    engine   LSX_HY4_DUMP_DIR=<dir> ... ./tests/test_hy_moe_flash_model "[.hy4diag]"
    reference /mnt/fast_storage/tmp/hy4venv2/bin/python tools/hy4_flash_reference.py \
                  --ids <same ids> --dump <engine dir>/ref.npz

Then:

    /mnt/fast_storage/tmp/hy4venv2/bin/python tools/hy4_flash_compare.py \
        --engine /mnt/fast_storage/tmp/hy4-eng-dump \
        --reference /mnt/fast_storage/tmp/hy4-eng-dump/ref.npz

The FIRST boundary whose cosine similarity collapses names the component: every
boundary before it is a correct computation in both implementations, so the
divergence belongs to the block that produced the bad one. That is the whole
point of the instrument — it replaces guessing with a measurement.

Expected agreement is not bit-exactness: the engine carries activations in
FP16 while the reference runs BF16 with FP32 router/softmax interiors, so a
healthy boundary sits at cos ~0.999+ and a broken one at cos << 0.99.
"""
import argparse
import sys

import numpy as np


# Boundaries in stack order. The engine writes one file per name; the reference
# uses the same names (npz keys) from its hooks.
ORDER = ["embed"] + ["layer%d" % i for i in range(42)] + ["hc_head", "norm",
                                                         "logits"]


def load_engine(path, name):
    f = "%s/%s.f32" % (path, name)
    try:
        return np.fromfile(f, dtype=np.float32)
    except OSError:
        return None


def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size != b.size:
        return None
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(np.dot(a, b) / (na * nb)) if na and nb else float("nan")
    rel = float(np.linalg.norm(a - b) / nb) if nb else float("nan")
    return cos, rel, float(np.abs(a - b).max()), float(np.abs(a).max()), \
        float(np.abs(b).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--floor", type=float, default=0.999,
                    help="a boundary below this cosine is reported as divergent")
    args = ap.parse_args()

    ref = np.load(args.reference)
    print("%-9s %10s %10s %10s %10s %10s" %
          ("boundary", "cos", "rel_l2", "max|d|", "max|eng|", "max|ref|"))
    first_bad = None
    for name in ORDER:
        if name not in ref.files:
            continue
        b = ref[name]
        a = load_engine(args.engine, name)
        if a is None:
            print("%-9s  (no engine dump)" % name)
            continue
        s = stats(a, b)
        if s is None and name == "embed" and a.size == 4 * b.size:
            # The engine expands the embedding into the 4 iHC streams before the
            # first block; the reference expands inside the model, so its hook is
            # the single 3072-wide vector.
            a = a[:b.size]
            s = stats(a, b)
        if s is None:
            print("%-9s  SHAPE MISMATCH engine=%s ref=%s" %
                  (name, a.shape, b.shape))
            first_bad = first_bad or name
            continue
        cos, rel, mx, ma, mb = s
        flag = ""
        if cos < args.floor and first_bad is None:
            first_bad = name
            flag = "   <-- first divergence"
        print("%-9s %10.6f %10.4f %10.4f %10.4f %10.4f%s" %
              (name, cos, rel, mx, ma, mb, flag))

    if first_bad:
        print("\nfirst divergence: %s — the block that produced it is the "
              "component to investigate" % first_bad)
        return 1
    print("\nevery boundary agrees: the divergence is in the sampler or the "
          "tokenizer, not the stack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
