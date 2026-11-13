#!/usr/bin/env python3
"""Per-tensor NVFP4 quantization error of a converted logestix tree vs its source.

Stage-2 evidence for an ingestion: decode the records the converter WROTE back
to float and compare them against the same tensor read from the source
safetensors. This is not an offline simulation of the codec — an offline
round-trip measures the codec's own error, while decoding the written record
also catches what the converter actually stored (a dropped `global_scale`, a
wrong block size, a truncated last block).

Codec (mirrors libs/lsxcommon/src/safetensors_gpu.cu, `quantize_tensor_gpu`):

  blocks      linear over the FLATENED tensor, 16 elements per block
  global      global_scale = max|v| / 6.0            (E2M1 top magnitude is 6)
  block scale e4m3(block_max / global_max), stored as an E4M3 byte
  quantize    nibble = E2M1(v / (e4m3_to_float(byte) * global_scale))
  pack        element 2k = LOW nibble, element 2k+1 = HIGH nibble
  dequant     v = E2M1(nibble) * e4m3_to_float(byte) * global_scale

Tree layout it reads (see libs/lsxcommon/src/converter.cpp):
  pinned/global.arrow, pinned/layer_<i>[_nvfp4].arrow   name/shape/dtype/...
      raw records carry a `data` column (BF16/F32 passthrough, never quantized)
      nvfp4 records carry packed/scales/global_scale
  experts/layer_<i>/exp_<e>.arrow   w1/w2/w3_{weight,scale,global_scale}
      w1 is the absent gate projection when the checkpoint is `experts_no_gate`

Usage:
  python3 tools/nemotron_bf16_quant_probe.py --selftest
  python3 tools/nemotron_bf16_quant_probe.py --model <vault> --tree <vault>/logestix \
      [--out report.txt]
"""
import argparse
import json
import os
import sys

import numpy as np

# E2M1 (float4 e2m1): 1 sign bit, 2 exponent, 1 mantissa. The eight positive
# magnitudes are the complete level set; there is no infinity and no NaN.
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
BLOCK = 16


def e2m1_table():
    """256-entry nibble -> float table (bit 3 is the sign)."""
    t = np.zeros(16, dtype=np.float32)
    for code in range(8):
        t[code] = E2M1_MAG[code]
        t[code | 8] = -E2M1_MAG[code]
    return t


def e4m3_table():
    """256-entry byte -> float table for E4M3FN (finite-only, 0x7F/0xFF NaN).

    Built from the bit layout rather than a literal list so the table can be
    checked against the code it claims to mirror: sign(1) exponent(4, bias 7)
    mantissa(3), with subnormals at exponent 0.
    """
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b >> 7) & 1 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF:
            t[b] = np.nan          # no infinity in E4M3FN
        elif e == 0:
            t[b] = s * (m / 8.0) * (2.0 ** (1 - 7))   # subnormal
        else:
            t[b] = s * (1.0 + m / 8.0) * (2.0 ** (e - 7))
    return t


E2M1 = e2m1_table()
E4M3 = e4m3_table()


def float_to_e4m3(x):
    """Round a positive float to the nearest E4M3FN byte (ties away from 0)."""
    finite = E4M3.copy()
    finite[np.isnan(finite)] = np.inf
    mag = np.abs(finite)
    idx = int(np.argmin(np.abs(mag - x)))
    return idx & 0xFF


def quantize(w, block=BLOCK):
    """NVFP4-quantize a flat float32 vector. Returns (packed, scales, global)."""
    v = np.asarray(w, dtype=np.float32).ravel()
    n = v.size
    pad = (-n) % block
    if pad:
        v = np.pad(v, (0, pad))
    nblk = v.size // block
    blocks = v.reshape(nblk, block)
    global_max = float(np.max(np.abs(v))) if v.size else 0.0
    gs = global_max / 6.0 if global_max > 0.0 else 1.0
    block_max = np.max(np.abs(blocks), axis=1)
    scales = np.zeros(nblk, dtype=np.uint8)
    eff = np.zeros(nblk, dtype=np.float32)
    for i in range(nblk):
        s = (1.0 / 128.0) if block_max[i] == 0.0 else float(block_max[i] / global_max)
        b = float_to_e4m3(s)
        scales[i] = b
        eff[i] = E4M3[b] * gs
    q = np.divide(blocks, eff[:, None],
                  out=np.zeros_like(blocks), where=eff[:, None] != 0.0)
    codes = np.argmin(np.abs(np.abs(q)[..., None] - E2M1_MAG), axis=-1)
    codes = np.where(q < 0, codes | 8, codes).astype(np.uint8).ravel()
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    packed[:] = codes[0::2] | (codes[1::2] << 4)
    return packed, scales, float(gs)


def dequantize(packed, scales, gs, numel, block=BLOCK):
    """Decode an NVFP4 record to float32 (the inverse of `quantize`)."""
    packed = np.asarray(packed, dtype=np.uint8)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2] = lo
    codes[1::2] = hi
    codes = codes[:numel]
    vals = E2M1[codes]
    scales = np.asarray(scales, dtype=np.uint8)
    nblk = (numel + block - 1) // block
    pad = nblk * block - numel
    eff = np.repeat(E4M3[scales[:nblk]] * np.float32(gs), block)
    if pad:
        vals = np.pad(vals, (0, pad))
    return (vals * eff).astype(np.float32)[:numel]


def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    den = float(np.linalg.norm(a))
    return float(np.linalg.norm(a - b) / den) if den > 0.0 else 0.0


# ── tree / source readers ───────────────────────────────────────────────────

def source_index(model_dir):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def read_source(model_dir, weight_map, name):
    """Read one tensor from the BF16 source shards (single shard, single tensor)."""
    from safetensors import safe_open

    shard = weight_map[name]
    with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
        return f.get_tensor(name).float().numpy()


def _ipc(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(path) as src:
        return ipc.open_file(src).read_all()


def _bytes(col):
    """Arrow column value -> bytes. `packed`/`data` are large_binary, `scales`
    is list<uint8> (see DenseNvfp4Schema), so both shapes must be handled."""
    v = col.as_py()
    if v is None:
        return b""
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v)
    return np.asarray(v, dtype=np.uint8).tobytes()


def read_pinned(tree_dir):
    """name -> record dict for every pinned tensor (raw or nvfp4)."""
    out = {}
    d = os.path.join(tree_dir, "pinned")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".arrow"):
            continue
        batch = _ipc(os.path.join(d, fn))
        cols = set(batch.schema.names)
        # tokenizer_vocab/tokenizer_merges live in the same directory with a
        # different schema; only weight records carry name/shape/dtype.
        if not {"name", "shape", "dtype"} <= cols:
            continue
        for i in range(batch.num_rows):
            shape = batch.column("shape")[i].as_py()
            rec = {"file": fn,
                   "shape": list(shape) if not isinstance(shape, str)
                            else json.loads(shape),
                   "dtype": batch.column("dtype")[i].as_py()}
            if "data" in cols:
                rec["data"] = np.frombuffer(_bytes(batch.column("data")[i]),
                                            dtype=np.uint8)
            if "packed" in cols:
                rec["packed"] = np.frombuffer(_bytes(batch.column("packed")[i]),
                                              dtype=np.uint8)
                rec["scales"] = np.frombuffer(_bytes(batch.column("scales")[i]),
                                              dtype=np.uint8)
                rec["gs"] = float(batch.column("global_scale")[i].as_py())
            out[batch.column("name")[i].as_py()] = rec
    return out


def read_expert(tree_dir, layer, exp):
    path = os.path.join(tree_dir, "experts", "layer_%d" % layer,
                        "exp_%d.arrow" % exp)
    batch = _ipc(path)
    row = 0
    rec = {}
    for slot in ("w1", "w2", "w3"):
        packed = _bytes(batch.column(slot + "_weight")[row])
        # An absent projection (the checkpoint is `experts_no_gate`, so w1 is
        # the gate) is written as an EMPTY binary, not null.
        rec[slot] = None if not packed else {
            "packed": np.frombuffer(packed, dtype=np.uint8),
            "scales": np.frombuffer(_bytes(batch.column(slot + "_scale")[row]),
                                    dtype=np.uint8),
            "gs": float(batch.column(slot + "_global_scale")[row].as_py()),
        }
    return rec


def raw_to_float(rec):
    """Decode a passthrough (BF16/F32) pinned record."""
    if rec["dtype"] == "BF16":
        u = rec["data"].view(np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    if rec["dtype"] == "F32":
        return rec["data"].view(np.float32)
    raise SystemExit("unhandled passthrough dtype: %s" % rec["dtype"])


# ── probe ───────────────────────────────────────────────────────────────────

PROBES = [
    ("attention projection", "backbone.layers.5.mixer.q_proj.weight"),
    ("mamba projection", "backbone.layers.0.mixer.in_proj.weight"),
    ("shared expert up", "backbone.layers.1.mixer.shared_experts.up_proj.weight"),
    ("lm_head", "lm_head.weight"),
    ("embeddings", "backbone.embeddings.weight"),
]


def probe(model_dir, tree_dir):
    weight_map = source_index(model_dir)
    pinned = read_pinned(tree_dir)
    rows = []

    for label, name in PROBES:
        if name not in weight_map or name not in pinned:
            rows.append({"label": label, "name": name, "relL2": None,
                         "note": "absent from source or tree"})
            continue
        src = read_source(model_dir, weight_map, name).ravel()
        rec = pinned[name]
        if "packed" in rec:
            got = dequantize(rec["packed"], rec["scales"], rec["gs"], src.size)
            note = "nvfp4 gs=%.6g" % rec["gs"]
        else:
            got = raw_to_float(rec)
            note = "passthrough %s" % rec["dtype"]
        rows.append({"label": label, "name": name, "relL2": rel_l2(src, got),
                     "note": note, "numel": int(src.size)})

    # Routed expert: the record stores no shape, and w2/w3 are both 1856x2688.
    # Score all four pairings and take the cheapest assignment, so a wrong
    # w2/w3 convention shows up as a reported number instead of a silent swap.
    exp = read_expert(tree_dir, 1, 0)
    up = read_source(model_dir, weight_map,
                     "backbone.layers.1.mixer.experts.0.up_proj.weight").ravel()
    down = read_source(model_dir, weight_map,
                       "backbone.layers.1.mixer.experts.0.down_proj.weight").ravel()
    pairs = []
    for slot in ("w2", "w3"):
        if exp[slot] is None:
            continue
        got = dequantize(exp[slot]["packed"], exp[slot]["scales"],
                         exp[slot]["gs"], up.size)
        pairs.append((rel_l2(up, got), slot, "up", exp[slot]["gs"]))
        pairs.append((rel_l2(down, got), slot, "down", exp[slot]["gs"]))
    if pairs:
        pairs.sort()
        best_up = min((p for p in pairs if p[2] == "up"), default=None)
        best_down = min((p for p in pairs if p[2] == "down"), default=None)
        worst_up = max((p for p in pairs if p[2] == "up"), default=None)
        worst_down = max((p for p in pairs if p[2] == "down"), default=None)
        for best, worst, which in ((best_up, worst_up, "up_proj"),
                                   (best_down, worst_down, "down_proj")):
            if best is None:
                continue
            rows.append({"label": "routed expert " + which,
                         "name": "backbone.layers.1.mixer.experts.0.%s" % which,
                         "relL2": best[0],
                         "note": "nvfp4 %s gs=%.6g (crossed slot %.3f)"
                                 % (best[1], best[3], worst[0]),
                         "numel": int(up.size)})
        rows.append({"label": "routed expert w1 (gate)",
                     "name": "backbone.layers.1.mixer.experts.0.gate_proj",
                     "relL2": None,
                     "note": "absent (experts_no_gate)" if exp["w1"] is None
                             else "present",
                     "numel": 0})
    return rows


def selftest():
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)

    # 1. The two decode tables must match the codec they mirror.
    check(abs(E2M1[0]) == 0.0 and E2M1[2] == 1.0 and E2M1[7] == 6.0,
          "E2M1 levels: got %s" % E2M1[:8])
    check(E2M1[9] == -0.5 and E2M1[15] == -6.0, "E2M1 sign: got %s" % E2M1[8:])
    check(abs(E4M3[0x38] - 1.0) < 1e-6, "E4M3 0x38 should be 1.0, got %s" % E4M3[0x38])
    check(abs(E4M3[0x3C] - 1.5) < 1e-6, "E4M3 0x3C should be 1.5, got %s" % E4M3[0x3C])
    check(np.isnan(E4M3[0x7F]), "E4M3 0x7F is NaN in e4m3fn")
    check(abs(E4M3[0x01] - 2.0 ** -9) < 1e-9, "E4M3 subnormal")

    # 2. A round-trip on a smooth tensor must land inside the 4-bit noise band.
    rng = np.random.default_rng(7)
    w = (rng.standard_normal(4 * 64 * 16) * 0.02).astype(np.float32)
    packed, scales, gs = quantize(w)
    ok = rel_l2(w, dequantize(packed, scales, gs, w.size))
    check(ok < 0.12, "clean round-trip relL2 %.4f should be < 0.12" % ok)

    # 3. Packing order: element 2k is the LOW nibble. A single 1.0 in a block of
    #    zeros must appear at an even index, so byte 0 reads 0x01 not 0x10.
    one = np.zeros(16, dtype=np.float32)
    one[0] = 6.0
    p, s, g = quantize(one)
    check(p[0] & 0x0F == 7, "low nibble of element 0 should be code 7 (=6.0), got %d"
          % (p[0] & 0x0F))
    check(p[0] >> 4 == 0, "high nibble of element 1 should be 0, got %d"
          % (p[0] >> 4))

    # 4. global_scale is applied: with gs != 1 a dropped factor is catastrophic.
    big = (rng.standard_normal(1024) * 4.0).astype(np.float32)   # max ~ >6
    p, s, g = quantize(big)
    check(abs(g) > 1.5, "a 4-sigma tensor should need gs > 1.5, got %.4f" % g)
    applied = rel_l2(big, dequantize(p, s, g, big.size))
    dropped = rel_l2(big, dequantize(p, s, 1.0, big.size))
    check(applied < 0.12, "gs-applied relL2 %.4f" % applied)
    check(dropped > 0.5,
          "dropping global_scale must be detected, got relL2 %.4f" % dropped)

    # 5. A below-grid block flushes to zero instead of poisoning the tensor.
    dev = np.zeros(32, dtype=np.float32)
    dev[0] = 100.0
    dev[16] = 1e-4
    p, s, g = quantize(dev)
    got = dequantize(p, s, g, dev.size)
    check(abs(got[0] - 100.0) / 100.0 < 0.15, "large element survives")
    check(abs(got[16]) <= 1e-3, "tiny block flushes, got %s" % got[16])

    if fails:
        for f in fails:
            print("FAIL:", f)
        return 1
    print("selftest OK (%d checks)" % 10)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--tree")
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not (args.model and args.tree):
        ap.error("--model and --tree are required (or pass --selftest)")

    rows = probe(args.model, args.tree)
    lines = ["# per-tensor quantization error vs the BF16 source",
             "# name  relL2  note  numel"]
    for r in rows:
        val = "n/a" if r["relL2"] is None else "%.6f" % r["relL2"]
        line = "%-22s %-52s %10s  %s" % (r["label"], r["name"], val, r["note"])
        lines.append(line)
    text = "\n".join(lines) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
