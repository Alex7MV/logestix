#!/usr/bin/env python3
"""Semantic cross-check of the Inkling-2-Small engine against the AUTHORITATIVE
`transformers` implementation, on identical weights.

Why this exists: `tools/inkling_layer_ref.py` reads the SAME converted artifact
the engine reads, so it can prove the engine implements *my reading* of the
reference faithfully — it can never prove my reading is right. This script closes
that gap. It

  1. writes a tiny synthetic checkpoint in the released naming convention (every
     expert plane NVFP4, so the converter's pass-through is byte-exact),
  2. runs the repo's own converter over it (`lsxforge --inkling`),
  3. DEQUANTIZES the artifact back to float32 and loads those exact tensors into
     `transformers.models.inkling2.Inkling2ForCausalLM`,
  4. prints the logits row for a fixed token sequence.

Because both sides consume identical numbers, any disagreement is a SEMANTIC
defect in the engine (or in the converter's re-indexing), never quantization.
Compare against `tools/inkling_layer_ref.py` and the engine's `TraceForward`
output for the same ids.

Usage:
  python3 tools/inkling_transformers_crosscheck.py <workdir> [--convert]
"""
import argparse
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import os

# ── geometry ───────────────────────────────────────────────────────────────
# Defaults mirror tests/test_inkling_fixture.h. Every value can be overridden
# from the environment so the SAME cross-check can be run at the real model's
# RATIOS — GQA group 6, top-8 of many experts, several AttnRes blocks — which is
# where a semantic defect that the tiny fixture cannot see would hide.
def _env(name, default):
    return int(os.environ.get("XC_" + name, default))


H = _env("H", 64)
L = _env("L", 6)
NH = _env("NH", 4)
NKV = _env("NKV", 2)
HD = _env("HD", 16)
DREL = _env("DREL", 4)
REL_EXTENT = _env("REL_EXTENT", 8)
WINDOW = _env("WINDOW", 8)
CONVK = 4
BLOCK = _env("BLOCK", 2)
E = _env("E", 4)
TOPK = _env("TOPK", 2)
INTER = _env("INTER", 32)
VOCAB = _env("VOCAB", 256)
UNPADDED = _env("UNPADDED", 200)
NBLOCKS = _env("NBLOCKS", 1)
FULL_PER = L // NBLOCKS if NBLOCKS > 1 else 0
# Full-attention layers: the real model puts one at every block boundary that is
# also `(layer+1) % 6 == 0`; parameterise instead of hardcoding 6.
FULL_STRIDE = _env("FULL_STRIDE", 6)
FULL_LAYERS = [l for l in range(L) if (l + 1) % FULL_STRIDE == 0]
LOCAL = [l for l in range(L) if l not in FULL_LAYERS]

E2M1_MAX = 6.0
E4M3_MAX = 448.0
DENOM = E2M1_MAX * E4M3_MAX          # 2688, the checkpoint's own denominator
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e4m3_round(x: np.ndarray) -> np.ndarray:
    """Round a positive magnitude to the nearest E4M3FN value."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    for i, v in enumerate(x.ravel()):
        if v <= 0:
            out.ravel()[i] = 0.0
            continue
        best, bd = 0.0, 1e30
        for e in range(16):
            for m in range(8):
                cand = (m * 2.0 ** -9) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - 7))
                d = abs(cand - v)
                if d < bd:
                    bd, best = d, cand
        out.ravel()[i] = best
    return out


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """Nearest E4M3FN byte for a positive magnitude."""
    x = np.asarray(x, dtype=np.float64)
    vals, codes = [], []
    for e in range(16):
        for m in range(8):
            code = (e << 3) | m
            if code == 0x7F:
                continue
            val = (m * 2.0 ** -9) if e == 0 else (1.0 + m / 8.0) * (2.0 ** (e - 7))
            vals.append(val)
            codes.append(code)
    vals = np.array(vals)
    codes = np.array(codes)
    flat = x.ravel()
    idx = np.abs(flat[:, None] - vals[None, :]).argmin(axis=1)
    return codes[idx].astype(np.uint8).reshape(x.shape)


def quantize_per_expert(w3: np.ndarray, block: int = 16):
    """Quantizes a [E, R, C] expert stack with ONE global scale per expert.

    This is the checkpoint's own rule: `amax` is `[E]` (one value per expert) and
    the per-16 block scales are relative to it, so `g_e = amax_e/2688` applies to
    the whole expert matrix. Quantizing the stack as a single flat matrix — the
    obvious shortcut — produces a per-ROW amax of shape `E*R`, which the
    converter then rejects.
    """
    E_, R, C = w3.shape
    packed = np.zeros((E_, R, C // 2), dtype=np.uint8)
    scales = np.zeros((E_, R, C // block), dtype=np.uint8)
    amax = np.zeros(E_, dtype=np.float32)
    for e in range(E_):
        p, s, a = quantize_nvfp4(w3[e], block)
        packed[e] = p
        scales[e] = s
        amax[e] = float(np.abs(w3[e]).max())
    return packed, scales, amax


def quantize_nvfp4(w: np.ndarray, block: int = 16):
    """NVFP4 (E2M1 nibbles + per-`block` E4M3 scales + a per-row global scale).

    The global scale is amax/DENOM, which is exactly what the checkpoint's
    `w1_global_scale = amax/outer_scale_denominator` reconstructs, so the
    converter's pass-through reproduces these values bit for bit.
    """
    rows, cols = w.shape
    amax = np.array([np.abs(w[r]).max() for r in range(rows)])
    gs = amax / DENOM
    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    scales = np.zeros((rows, cols // block), dtype=np.uint8)
    for r in range(rows):
        g = gs[r] if gs[r] > 0 else 1.0
        for b0 in range(0, cols, block):
            blk = w[r, b0:b0 + block]
            # scale so the block maximum lands on E2M1's 6.0
            mx = np.abs(blk).max()
            s = (mx / E2M1_MAX) / g if mx > 0 else 1.0
            s = min(max(s, 2.0 ** -9), 448.0)
            s = float(e4m3_round(np.array([s]))[0])
            scales[r, b0 // block] = e4m3_encode(np.array([[s]]))[0, 0]
            for i in range(block):
                v = blk[i] / (s * g) if (s * g) > 0 else 0.0
                if not math.isfinite(v):
                    v = 0.0
                neg = v < 0
                mag = min(abs(v), E2M1_MAX)
                qi = int(np.abs(E2M1 - mag).argmin())
                nib = qi | (8 if neg else 0)
                byte_i = b0 + i
                if i % 2 == 0:
                    packed[r, byte_i // 2] = (packed[r, byte_i // 2] & 0xF0) | nib
                else:
                    packed[r, byte_i // 2] = (packed[r, byte_i // 2] & 0x0F) | (nib << 4)
    return packed, scales, amax


def bf16_bytes(a: np.ndarray) -> np.ndarray:
    """float32 -> BF16 (round-to-nearest-even), as raw uint16."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    finite = (u & 0x7F800000) != 0x7F800000
    r = u + (0x7FFF + ((u >> 16) & 1))
    r = np.where(finite, r, u)
    return (r >> 16).astype(np.uint16)


def to_bytes(dtype: str, arr: np.ndarray) -> np.ndarray:
    """Serializes `arr` in the DECLARED dtype. Writing float32 bytes under a
    BF16 header is what made the converter reject the fixture as malformed."""
    if dtype == "BF16":
        return bf16_bytes(arr)
    if dtype == "F32":
        return np.ascontiguousarray(arr, dtype=np.float32)
    return np.ascontiguousarray(arr)


def write_safetensors(path: Path, tensors: dict):
    header = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    offs = {}
    for name, (dtype, arr) in tensors.items():
        raw = to_bytes(dtype, arr).tobytes()
        offs[name] = (dtype, list(arr.shape), [len(data), len(data) + len(raw)])
        data += raw
    for name, (dt, shp, o) in offs.items():
        header[name] = {"dtype": dt, "shape": shp, "data_offsets": o}
    hjson = json.dumps(header).encode()
    hjson += b" " * ((8 - (len(hjson) + 8) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(bytes(data))


def gen():
    rng = np.random.default_rng(20261114)

    def field(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    t = {}
    t["model.llm.embed.weight"] = ("BF16", field(VOCAB, H))
    t["model.llm.unembed.weight"] = ("BF16", field(VOCAB, H))
    t["model.llm.embed_norm.weight"] = ("BF16", 1.0 + field(H, scale=0.02))
    t["model.llm.norm.weight"] = ("BF16", 1.0 + field(H, scale=0.02))
    t["model.llm.attn_res.output_query"] = ("F32", field(H))
    for l in range(L):
        t[f"model.llm.attn_res.mlp_queries.{l}"] = ("F32", field(H))
        if l == 0:
            continue
        shape = (3, H) if l in FULL_LAYERS else (H,)
        # SOURCE convention: the attention bank is indexed by `owning layer - 1`
        # (layer 0 has no attention mixture), which is what the converter
        # re-indexes to the owning layer in the artifact.
        t[f"model.llm.attn_res.attn_queries.{l - 1}"] = ("F32", field(*shape))

    for l in range(L):
        p = f"model.llm.layers.{l}."
        full = l in FULL_LAYERS
        t[p + "attn.wq_du.weight"] = ("BF16", field(NH * HD, H))
        t[p + "attn.wk_dv.weight"] = ("BF16", field(NKV * HD, H))
        t[p + "attn.wv_dv.weight"] = ("BF16", field(NKV * HD, H))
        t[p + "attn.wo_ud.weight"] = ("BF16", field(H, NH * HD) * 0.5)
        t[p + "attn.wr_du.weight"] = ("BF16", field(NH * DREL, H))
        t[p + "attn.gate_attn.weight"] = ("BF16", field(NH * HD, H))
        t[p + "attn.attn_sink_proj.weight"] = ("BF16", field(NH, H) * 0.1)
        t[p + "attn.rel_logits_proj.proj"] = ("BF16", field(DREL, REL_EXTENT))
        t[p + "attn.q_l2_clip.scale"] = ("BF16", 1.0 + field(NH, HD, scale=0.02))
        t[p + "attn.rel_l2_clip.scale"] = ("BF16", 1.0 + field(NH, DREL, scale=0.02))
        t[p + "attn_sconv.weight"] = ("BF16", field(H, 1, CONVK) * 0.5)
        t[p + "mlp_sconv.weight"] = ("BF16", field(H, 1, CONVK) * 0.5)
        t[p + "mlp_norm.weight"] = ("BF16", 1.0 + field(H, scale=0.02))
        t[p + "mlp.gate.weight"] = ("BF16", field(E, H))
        t[p + "mlp.gate.bias"] = ("F32", field(E, scale=0.05))
        if full:
            t[p + "q_input_norm.weight"] = ("BF16", 1.0 + field(H, scale=0.02))
            t[p + "k_input_norm.weight"] = ("BF16", 1.0 + field(H, scale=0.02))
            t[p + "v_input_norm.weight"] = ("BF16", 1.0 + field(H, scale=0.02))
            t[p + "attn.k_sconv.static_w"] = ("BF16", field(CONVK, NKV * HD) * 0.5)
            t[p + "attn.v_sconv.static_w"] = ("BF16", field(CONVK, NKV * HD) * 0.5)
            t[p + "attn.k_sconv.weight_proj"] = ("BF16", field(NKV * CONVK, H))
            t[p + "attn.v_sconv.weight_proj"] = ("BF16", field(NKV * CONVK, H))
        else:
            t[p + "attn_norm.weight"] = ("BF16", 1.0 + field(H, scale=0.02))
            t[p + "attn.k_sconv.weight"] = ("BF16", field(NKV * HD, 1, CONVK) * 0.5)
            t[p + "attn.v_sconv.weight"] = ("BF16", field(NKV * HD, 1, CONVK) * 0.5)
        # experts: NVFP4 source for EVERY layer (byte-exact pass-through)
        g13 = field(E, 2 * INTER, H) * 0.4
        d2 = field(E, H, INTER) * 0.4
        p13, s13, a13 = quantize_per_expert(g13)
        p2, s2, a2 = quantize_per_expert(d2)
        t[p + "mlp.experts.w13_weight"] = ("U8", p13)
        t[p + "mlp.experts.w13_weight_scale"] = ("F8_E4M3", s13)
        t[p + "mlp.experts.w13_weight_amax"] = ("F32", a13.astype(np.float32))
        t[p + "mlp.experts.w13_weight_outer_scale_denominator"] = (
            "F32", np.array([DENOM], dtype=np.float32))
        t[p + "mlp.experts.w2_weight"] = ("U8", p2)
        t[p + "mlp.experts.w2_weight_scale"] = ("F8_E4M3", s2)
        t[p + "mlp.experts.w2_weight_amax"] = ("F32", a2.astype(np.float32))
        t[p + "mlp.experts.w2_weight_outer_scale_denominator"] = (
            "F32", np.array([DENOM], dtype=np.float32))
    t["model.visual.backbone.ln_pre.weight"] = ("BF16", field(H))
    return t


def write_checkpoint(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    cfg = {
        "architectures": ["Inkling2ForConditionalGeneration"],
        "model_type": "inkling2",
        "eos_token_id": 200006,
        "text_config": {
            "hidden_size": H, "num_hidden_layers": L, "vocab_size": VOCAB,
            "unpadded_vocab_size": UNPADDED,
            "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD,
            "d_rel": DREL, "rel_extent": REL_EXTENT,
            "sliding_window_size": WINDOW, "sconv_kernel_size": CONVK,
            "n_layers_per_attn_res_block": BLOCK,
            "n_routed_experts": E, "num_experts_per_tok": TOPK,
            "intermediate_size": INTER, "n_shared_experts": 0,
            "route_scale": 8.0, "rms_norm_eps": 1e-06,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
            "final_logit_softcapping": 30.0,
            "local_layer_ids": LOCAL,
        },
    }
    (dirpath / "config.json").write_text(json.dumps(cfg, indent=2))
    groups = {"nvfp4": [], "bfloat16": []}
    t = gen()
    for name in t:
        (groups["nvfp4"] if ".mlp.experts.w" in name and
         ("_weight" in name) and "scale" not in name and "amax" not in name
         and "denominator" not in name else groups["bfloat16"]).append(name)
    q = {"model": {"parameter_groups": [
        {"compute_dtype": "nvfp4", "tensors": groups["nvfp4"]},
        {"compute_dtype": "bfloat16", "tensors": groups["bfloat16"]}]}}
    (dirpath / "hf_quant_config.json").write_text(json.dumps(q, indent=1))
    write_safetensors(dirpath / "model-00001-of-00001.safetensors", t)
    return dirpath


def dequant_artifact(base: Path):
    """Reads the converted artifact into float32 numpy arrays."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    def read(p):
        with pa.memory_map(str(p), "r") as s:
            return ipc.open_file(s).read_all()

    def col_bytes(col, i):
        s = col[i]
        if hasattr(s, "as_buffer"):
            return np.frombuffer(s.as_buffer(), dtype=np.uint8)
        return s.values.to_numpy(zero_copy_only=False).astype(np.uint8)

    def e4m3(v):
        v = v.astype(np.int32)
        sgn = np.where((v >> 7) & 1, -1.0, 1.0)
        e = (v >> 3) & 0xF
        m = v & 0x7
        return sgn * np.where(e == 0, m * 2.0 ** -9,
                              (1.0 + m / 8.0) * 2.0 ** (e - 7))

    out = {}
    b = read(base / "pinned" / "global.arrow")
    names = b.column(0).to_pylist()
    shapes = b.column(1).to_pylist()
    dtypes = b.column(2).to_pylist()
    for i, n in enumerate(names):
        a = col_bytes(b.column(3), i)
        if dtypes[i] == "BF16":
            arr = ((a.view(np.uint16).astype(np.uint32) << 16).view(np.float32))
        elif dtypes[i] == "F32":
            arr = a.view(np.float32)
        else:
            continue
        out[n] = arr.reshape(shapes[i]).astype(np.float32)
    # A global NVFP4 batch only exists for trees converted by an older build
    # (embed/unembed now stay BF16, like the release).
    if (base / "pinned" / "global_nvfp4.arrow").exists():
        b = read(base / "pinned" / "global_nvfp4.arrow")
        names = b.column(0).to_pylist()
        shapes = b.column(1).to_pylist()
        for i, n in enumerate(names):
            rows, cols = int(shapes[i][0]), int(shapes[i][1])
            pk = col_bytes(b.column(3), i).reshape(rows, cols // 2)
            sc = col_bytes(b.column(4), i).reshape(rows, cols // 16)
            gs = float(b.column(5)[i].as_py())
            lo = (pk & 0x0F).astype(np.int32)
            hi = (pk >> 4).astype(np.int32)
            nib = np.empty((rows, cols), dtype=np.int32)
            nib[:, 0::2] = lo
            nib[:, 1::2] = hi
            q = E2M1[nib & 7] * np.where(nib & 8, -1.0, 1.0)
            out[n] = (q * np.repeat(e4m3(sc), 16, axis=1) * gs).astype(np.float32)
    # routed experts live per (layer, expert) in the neutral store
    for l in range(L):
        gate_up = np.zeros((E, 2 * INTER, H), dtype=np.float32)
        down = np.zeros((E, H, INTER), dtype=np.float32)
        for e in range(E):
            eb = read(base / "experts" / f"layer_{l}" / f"exp_{e}.arrow")
            cols = [col_bytes(eb.column(c), 0) for c in range(6)]
            gs = [float(eb.column(c)[0].as_py()) for c in (6, 7, 8)]

            def dec(pk, sc, g, rows, c):
                pk = pk.reshape(rows, c // 2)
                sc = sc.reshape(rows, c // 16)
                lo = (pk & 0x0F).astype(np.int32)
                hi = (pk >> 4).astype(np.int32)
                nib = np.empty((rows, c), dtype=np.int32)
                nib[:, 0::2] = lo
                nib[:, 1::2] = hi
                q = E2M1[nib & 7] * np.where(nib & 8, -1.0, 1.0)
                return (q * np.repeat(e4m3(sc), 16, axis=1) * g).astype(np.float32)

            gate_up[e, :INTER] = dec(cols[0], cols[3], gs[0], INTER, H)
            gate_up[e, INTER:] = dec(cols[2], cols[5], gs[2], INTER, H)
            down[e] = dec(cols[1], cols[4], gs[1], H, INTER)
        out[f"model.llm.layers.{l}.mlp.experts.gate_up_proj"] = gate_up
        out[f"model.llm.layers.{l}.mlp.experts.down_proj"] = down

    for l in range(L):
        b = read(base / "pinned" / f"layer_{l}.arrow")
        names = b.column(0).to_pylist()
        shapes = b.column(1).to_pylist()
        dts = b.column(2).to_pylist()
        for i, n in enumerate(names):
            a = col_bytes(b.column(3), i)
            if dts[i] == "BF16":
                arr = ((a.view(np.uint16).astype(np.uint32) << 16).view(np.float32))
            elif dts[i] == "F32":
                arr = a.view(np.float32)
            else:
                continue
            out[n] = arr.reshape(shapes[i]).astype(np.float32)
        lvp = base / "pinned" / f"layer_{l}_nvfp4.arrow"
        if not lvp.exists():
            continue
        b = read(lvp)
        names = b.column(0).to_pylist()
        shapes = b.column(1).to_pylist()
        for i, n in enumerate(names):
            rows, cols = int(shapes[i][0]), int(shapes[i][1])
            pk = col_bytes(b.column(3), i).reshape(rows, cols // 2)
            sc = col_bytes(b.column(4), i).reshape(rows, cols // 16)
            gs = float(b.column(5)[i].as_py())
            lo = (pk & 0x0F).astype(np.int32)
            hi = (pk >> 4).astype(np.int32)
            nib = np.empty((rows, cols), dtype=np.int32)
            nib[:, 0::2] = lo
            nib[:, 1::2] = hi
            q = E2M1[nib & 7] * np.where(nib & 8, -1.0, 1.0)
            out[n] = (q * np.repeat(e4m3(sc), 16, axis=1) * gs).astype(np.float32)
    return out


# ── artifact -> transformers module names ─────────────────────────────────
SUFFIX_MAP = {
    "attn.wq_du.weight": "self_attn.q_proj.weight",
    "attn.wk_dv.weight": "self_attn.k_proj.weight",
    "attn.wv_dv.weight": "self_attn.v_proj.weight",
    "attn.wo_ud.weight": "self_attn.o_proj.weight",
    "attn.wr_du.weight": "self_attn.r_proj.weight",
    "attn.gate_attn.weight": "self_attn.gate_proj.weight",
    "attn.attn_sink_proj.weight": "self_attn.sink_proj.weight",
    "attn.rel_logits_proj.proj": "self_attn.rel_logits_proj.proj",
    "attn.q_l2_clip.scale": "self_attn.q_l2_clip.scale",
    "attn.rel_l2_clip.scale": "self_attn.rel_l2_clip.scale",
    "attn.k_sconv.weight": "self_attn.k_sconv.conv1d.weight",
    "attn.v_sconv.weight": "self_attn.v_sconv.conv1d.weight",
    "attn.k_sconv.static_w": "self_attn.k_sconv.static_weight",
    "attn.v_sconv.static_w": "self_attn.v_sconv.static_weight",
    "attn.k_sconv.weight_proj": "self_attn.k_sconv.weight_proj.weight",
    "attn.v_sconv.weight_proj": "self_attn.v_sconv.weight_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "q_input_norm.weight": "q_input_layernorm.weight",
    "k_input_norm.weight": "k_input_layernorm.weight",
    "v_input_norm.weight": "v_input_layernorm.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.bias": "mlp.gate.e_score_correction_bias",
    "attn_sconv.weight": "post_attention_convolution.conv1d.weight",
    "mlp_sconv.weight": "post_mlp_convolution.conv1d.weight",
}


def load_into_transformers(base: Path, layer_types=None):
    import torch
    from transformers.models.inkling2 import configuration_inkling2 as C
    from transformers.models.inkling2 import modeling_inkling2 as M

    tensors = dequant_artifact(base)
    cfg = C.Inkling2TextConfig(
        vocab_size=VOCAB, unpadded_vocab_size=UNPADDED,
        hidden_size=H, num_hidden_layers=L,
        num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD,
        swa_num_attention_heads=NH, swa_num_key_value_heads=NKV, swa_head_dim=HD,
        sliding_window_size=WINDOW, d_rel=DREL, rel_extent=REL_EXTENT,
        n_layers_per_attn_res_block=BLOCK,
        log_scaling_n_floor=128000, log_scaling_alpha=0.1,
        layer_types=layer_types or ["hybrid" if l in FULL_LAYERS else "hybrid_sliding"
                                    for l in range(L)],
        rms_norm_eps=1e-6, conv_kernel_size=CONVK,
        moe_intermediate_size=INTER, n_routed_experts=E,
        num_experts_per_tok=TOPK, route_scale=8.0, scoring_func="sigmoid",
        qk_l2_clip_threshold=22.627416997969522, rel_l2_clip_threshold=8.0,
        mlp_gate_softcap=4.0, mlp_up_softcap=25.0,
        final_logit_softcapping=30.0,
    )
    model = M.Inkling2ForCausalLM(cfg)
    model.eval()

    sd = {}
    def put(name, arr):
        sd[name] = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))

    for name, arr in tensors.items():
        if not name.startswith("model.llm."):
            continue
        rest = name[len("model.llm."):]
        if rest == "embed.weight":
            put("model.embed_tokens.weight", arr); continue
        if rest == "unembed.weight":
            put("lm_head.weight", arr); continue
        if rest == "embed_norm.weight":
            put("model.embed_tokens.embed_norm.weight", arr); continue
        if rest == "norm.weight":
            put("model.norm.weight", arr); continue
        if rest == "attn_res.output_query":
            put("model.final_input.query", arr); continue
        if rest.startswith("attn_res.mlp_queries."):
            l = int(rest.rsplit(".", 1)[1])
            put(f"model.layers.{l}.mlp_input.query", arr); continue
        if rest.startswith("attn_res.attn_queries."):
            l = int(rest.rsplit(".", 1)[1])
            put(f"model.layers.{l}.attention_input.query", arr); continue
        if rest.startswith("layers."):
            l = int(rest.split(".")[1])
            tail = rest.split(".", 2)[2]
            if tail in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                # dequant_artifact already names the expert stacks by their
                # transformers module path.
                put(f"model.layers.{l}.{tail}", arr); continue
            if tail == "mlp.experts.w13_weight":
                put(f"model.layers.{l}.mlp.experts.gate_up_proj", arr); continue
            if tail == "mlp.experts.w2_weight":
                put(f"model.layers.{l}.mlp.experts.down_proj", arr); continue
            if tail in SUFFIX_MAP:
                put(f"model.layers.{l}.{SUFFIX_MAP[tail]}", arr); continue
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model, tensors, sorted(set(missing)), sorted(set(unexpected))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--convert", action="store_true",
                    help="write the checkpoint and run lsxforge --inkling first")
    ap.add_argument("--tokens", default="10,20,30")
    args = ap.parse_args()

    wd = Path(args.workdir)
    ckpt = wd / "ckpt"
    if args.convert:
        write_checkpoint(ckpt)
        print(f"wrote synthetic checkpoint -> {ckpt}")
        env = dict(os.environ, LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY="1")
        forge = wd / "lsxforge"
        cmd = [str(forge), "--inkling", "--input", str(ckpt), "--output", str(wd)]
        print("converting:", " ".join(cmd))
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:])
            return r.returncode
        print("converted ->", wd / "logestix")

    import torch
    model, tensors, missing, unexpected = load_into_transformers(
        wd / "logestix")
    print("missing keys :", [m for m in missing if "rotary" not in m][:8])
    print("unexpected   :", unexpected[:8])

    ids = [int(x) for x in args.tokens.split(",")]
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True,
                    use_cache=False)
    logits = out.logits[0, -1].float().numpy()[:UNPADDED]
    top = np.argsort(-logits)[:5]
    print(f"\ntransformers top5 for ids {ids}: "
          f"{[(int(i), round(float(logits[i]), 5)) for i in top]}")
    np.save(wd / "tf_logits.npy", logits)
    hs = out.hidden_states
    print("hidden_states layers:", len(hs) if hs is not None else None)
    if hs is not None:
        arr = np.stack([h[0, -1].float().numpy() for h in hs])
        np.save(wd / "tf_states.npy", arr)
        print("saved tf_states.npy", arr.shape, " |last| =",
              float(np.linalg.norm(arr[-1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
