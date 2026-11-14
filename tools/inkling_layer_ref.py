#!/usr/bin/env python3
"""Independent float64 host reference for ONE Inkling-2-Small layer.

Reads the CONVERTED artifact (logestix/pinned/*.arrow + experts/*.arrow), not
the engine's code, and recomputes the residual stream layer by layer from the
same seed state the engine reported. Divergence then names the layer.

Written for the parity investigation in
docs/open-tasks/2026-11-14-inkling-2-small-ingest.md (the real-weight SelfCheck
is red). Deliberately a separate implementation: it shares no code with the
kernels, so agreement is evidence and disagreement is localized.

Usage:
  python3 tools/inkling_layer_ref.py <vault> <trace.bin> [--layers N]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

# ── codec ──────────────────────────────────────────────────────────────────
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def e4m3_to_float(b: np.ndarray) -> np.ndarray:
    b = b.astype(np.int32)
    sign = np.where((b >> 7) & 1, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    man = b & 0x7
    # e == 0 -> subnormal m * 2^-9; 0x7F stays NaN (unused for block scales)
    val = np.where(exp == 0, man * (2.0 ** -9), (1.0 + man / 8.0) * (2.0 ** (exp - 7)))
    return sign * val


def dequant_nvfp4(packed: np.ndarray, scales: np.ndarray, gs: float,
                  rows: int, cols: int) -> np.ndarray:
    """[rows, cols/2] nibbles + [rows, cols/16] scales -> [rows, cols] float64."""
    b = packed.view(np.uint8).reshape(rows, cols // 2)
    lo = (b & 0x0F).astype(np.int32)
    hi = (b >> 4).astype(np.int32)
    nib = np.empty((rows, cols), dtype=np.int32)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    q = E2M1[nib & 7] * np.where(nib & 8, -1.0, 1.0)
    s = e4m3_to_float(scales.view(np.uint8).reshape(rows, cols // 16))
    return q * np.repeat(s, 16, axis=1) * float(gs)


def col_bytes(col, i: int) -> np.ndarray:
    """Reads row `i` of a large_binary OR list<uint8> column as a byte array."""
    s = col[i]
    if hasattr(s, "as_buffer"):
        return np.frombuffer(s.as_buffer(), dtype=np.uint8)
    return s.values.to_numpy(zero_copy_only=False).astype(np.uint8)


def read_arrow(path: Path):
    with pa.memory_map(str(path), "r") as src:
        return ipc.open_file(src).read_all()


def raw_tensors(path: Path) -> dict:
    b = read_arrow(path)
    out = {}
    names = b.column(0).to_pylist()
    shapes = b.column(1).to_pylist()
    dtypes = b.column(2).to_pylist()
    data = b.column(3)
    for i, n in enumerate(names):
        buf = data[i].as_buffer()
        a = np.frombuffer(buf, dtype=np.uint8)
        shp = shapes[i]
        if dtypes[i] == "BF16":
            a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtypes[i] == "F32":
            a = a.view(np.float32)
        out[n] = a.astype(np.float64).reshape(shp)
    return out


def nvfp4_tensors(path: Path) -> dict:
    b = read_arrow(path)
    out = {}
    names = b.column(0).to_pylist()
    shapes = b.column(1).to_pylist()
    packed_col, scale_col, gs_col = b.column(3), b.column(4), b.column(5)
    for i, n in enumerate(names):
        packed = col_bytes(packed_col, i)
        scales = col_bytes(scale_col, i)
        gs = float(gs_col[i].as_py())
        rows, cols = int(shapes[i][0]), int(shapes[i][1])
        out[n] = dequant_nvfp4(packed, scales, gs, rows, cols)
    return out


def expert_planes(path: Path):
    """Returns (gate[I,H], up[I,H], down[H,I]) plus their global scales."""
    b = read_arrow(path)
    cols = [col_bytes(b.column(c), 0) for c in range(6)]
    gs = [float(b.column(c)[0].as_py()) for c in (6, 7, 8)]
    return cols, gs


# ── layer maths (float64, transcribed from the reference forward) ──────────
def rmsnorm(x, w, eps=1e-6):
    return (x / math.sqrt((x * x).mean() + eps)) * w


def unweighted_rmsnorm(x, eps=1e-6):
    return x / math.sqrt((x * x).mean() + eps)


def l2_clip(x, scale, threshold):
    v = x * scale if scale is not None else x
    n = math.sqrt(float((v * v).sum()))
    return v * min(threshold / (n + 1e-6), 1.0)


def depth_mix(sources, queries, eps=1e-6):
    n = np.stack([unweighted_rmsnorm(s, eps) for s in sources])       # [n_src, H]
    out = []
    for q in queries:
        logits = n @ q / math.sqrt(n.shape[1])
        w = np.exp(logits - logits.max())
        w /= w.sum()
        out.append(w @ n)
    return out


def short_conv_static(x_hist, w, k, inv):
    """x_hist: [k, C] chronological, newest last. w: [C, k] conv1d taps."""
    conv = np.einsum("ckt?->", None, None) if False else (w * x_hist.T).sum(axis=1)
    return (conv + x_hist[-1]) * inv


def short_conv_dynamic(x_hist, taps, k, inv):
    """taps: [C, k] lag taps, lag 0 = current token."""
    flipped = x_hist[::-1]                       # lag order
    conv = (taps * flipped.T).sum(axis=1)
    return (conv + x_hist[-1]) * inv


def moe_only(base, idx, l, hh):
    """Recomputes one layer's MoE from a given router input."""
    H = idx["hidden_size"]; E = idx["num_experts"]; topk = idx["num_experts_per_tok"]
    I = idx["moe_intermediate_size"]
    rw = raw_tensors(base / "pinned" / f"layer_{l}.arrow")
    nm = lambda s: f"model.llm.layers.{l}.{s}"
    logits = rw[nm("mlp.gate.weight")] @ hh
    scores = 1.0 / (1.0 + np.exp(-logits))
    order = np.argsort(-(scores + rw[nm("mlp.gate.bias")]), kind="stable")[:topk]
    sel = np.sort(order)
    wsel = scores[sel]
    coeff = wsel / (wsel.sum() + 1e-20) * idx["routed_scaling_factor"]
    print("  [host] ids:", " ".join(str(x) for x in sel))
    print("  [host] coeff:", " ".join(f"{c:.8g}" for c in coeff))
    acc = np.zeros(H)
    for c, e in enumerate(sel):
        cols, gs = expert_planes(base / "experts" / f"layer_{l}" / f"exp_{e}.arrow")
        gate_e = dequant_nvfp4(cols[0], cols[3], gs[0], I, H)
        down_e = dequant_nvfp4(cols[1], cols[4], gs[1], H, I)
        up_e = dequant_nvfp4(cols[2], cols[5], gs[2], I, H)
        g_e = gate_e @ hh
        u_e = up_e @ hh
        act = (np.tanh(g_e / idx["mlp_gate_softcap"]) * idx["mlp_gate_softcap"]) \
            * (1.0 / (1.0 + np.exp(-g_e))) \
            * (np.tanh(u_e / idx["mlp_up_softcap"]) * idx["mlp_up_softcap"])
        acc += coeff[c] * (down_e @ act)
    print(f"  [host] |moe_acc| {np.linalg.norm(acc):.8g}")
    return acc



def expert_block(base, idx, l, mlp_x):
    """One layer's routed MoE: sigmoid scores, selection bias for the top-k
    CHOICE only, weights renormalised then scaled, softcapped SwiGLU, FP32 sum."""
    H = idx["hidden_size"]; topk = idx["num_experts_per_tok"]
    I = idx["moe_intermediate_size"]
    rw = raw_tensors(base / "pinned" / f"layer_{l}.arrow")
    nm = lambda s: f"model.llm.layers.{l}.{s}"
    logits = rw[nm("mlp.gate.weight")] @ mlp_x
    scores = 1.0 / (1.0 + np.exp(-logits))
    sel = np.sort(np.argsort(-(scores + rw[nm("mlp.gate.bias")]),
                             kind="stable")[:topk])
    wsel = scores[sel]
    coeff = wsel / (wsel.sum() + 1e-20) * idx["routed_scaling_factor"]
    acc = np.zeros(H)
    for c, e in enumerate(sel):
        cols, gs = expert_planes(base / "experts" / f"layer_{l}" / f"exp_{e}.arrow")
        gate_e = dequant_nvfp4(cols[0], cols[3], gs[0], I, H)
        down_e = dequant_nvfp4(cols[1], cols[4], gs[1], H, I)
        up_e = dequant_nvfp4(cols[2], cols[5], gs[2], I, H)
        g_e = gate_e @ mlp_x
        u_e = up_e @ mlp_x
        act = (np.tanh(g_e / idx["mlp_gate_softcap"]) * idx["mlp_gate_softcap"]) \
            * (1.0 / (1.0 + np.exp(-g_e))) \
            * (np.tanh(u_e / idx["mlp_up_softcap"]) * idx["mlp_up_softcap"])
        acc += coeff[c] * (down_e @ act)
    return acc


def layer0_sequence(base, idx, states, tok_ids):
    """Runs LAYER 0 over `tok_ids` sequentially and returns the per-position
    residual stream plus the stage magnitudes at the last position.

    Layer 0 is the only layer with no AttnRes mixture, so its output depends
    solely on: the residual stream (supplied by the trace), the FOUR conv rings
    and the KV cache carried from earlier positions. Comparing its output at
    position 1 against the engine's trace therefore tests exactly the
    multi-position machinery — conv history, KV append and attention over more
    than one cached key — with no block stack and no accumulated FP16 rounding
    from earlier layers.
    """
    H = idx["hidden_size"]; nh = idx["num_attention_heads"]
    nkv = idx["num_key_value_heads"]; hd = idx["head_dim"]
    drel = idx["d_rel"]; k = idx["conv_kernel"]; eps = idx["rms_norm_eps"]
    scale = idx["attn_scale_denom"] ** -0.5
    l = 0
    nv = nvfp4_tensors(base / "pinned" / f"layer_{l}_nvfp4.arrow")
    rw = raw_tensors(base / "pinned" / f"layer_{l}.arrow")
    g_raw = raw_tensors(base / "pinned" / "global.arrow")
    nm = lambda x: f"model.llm.layers.{l}.{x}"

    ring_k = np.zeros((k, nkv * hd))
    ring_v = np.zeros((k, nkv * hd))
    ring_o = np.zeros((k, H))
    ring_m = np.zeros((k, H))
    keys = []          # one [nkv, hd] per cached position
    vals = []
    outs = []
    mags = {}
    w_ao = rw[nm("attn_sconv.weight")].reshape(-1, k)
    w_mo = rw[nm("mlp_sconv.weight")].reshape(-1, k)

    for pos, x in enumerate(states):
        qs = ks = vs = rmsnorm(x, rw[nm("attn_norm.weight")], eps)
        q = nv[nm("attn.wq_du.weight")] @ qs
        kv_k = nv[nm("attn.wk_dv.weight")] @ ks
        kv_v = nv[nm("attn.wv_dv.weight")] @ vs
        rel = nv[nm("attn.wr_du.weight")] @ qs
        gate = nv[nm("attn.gate_attn.weight")] @ qs
        sink = rw[nm("attn.attn_sink_proj.weight")] @ qs

        ring_k = np.vstack([ring_k[1:], kv_k[None, :]])
        ring_v = np.vstack([ring_v[1:], kv_v[None, :]])
        kconv = short_conv_static(ring_k, rw[nm("attn.k_sconv.weight")].reshape(-1, k),
                                  k, (k + 1) ** -0.5)
        vconv = short_conv_static(ring_v, rw[nm("attn.v_sconv.weight")].reshape(-1, k),
                                  k, (k + 1) ** -0.5)

        qc = np.stack([l2_clip(q[h * hd:(h + 1) * hd],
                               rw[nm("attn.q_l2_clip.scale")][h].astype(np.float64),
                               idx["qk_l2_clip"]) for h in range(nh)])
        kc = np.stack([l2_clip(kconv[h * hd:(h + 1) * hd], None,
                               idx["qk_l2_clip"]) for h in range(nkv)])
        vn = np.stack([unweighted_rmsnorm(vconv[h * hd:(h + 1) * hd], eps)
                       for h in range(nkv)])
        rc = np.stack([l2_clip(rel[h * drel:(h + 1) * drel],
                               rw[nm("attn.rel_l2_clip.scale")][h].astype(np.float64),
                               idx["rel_l2_clip"]) for h in range(nh)])
        keys.append(kc)
        vals.append(vn)
        rel_proj = rw[nm("attn.rel_logits_proj.proj")]
        group = nh // nkv
        window = idx["sliding_window_size"]
        kv_len = len(keys)
        out = np.zeros((nh, hd))
        for h in range(nh):
            kvh = h // group
            prof = rc[h] @ rel_proj
            logits = np.full(kv_len, -np.inf)
            for j in range(kv_len):
                if j > pos - window:
                    d = pos - j
                    b = prof[d] if 0 <= d < window else 0.0
                    logits[j] = scale * float(qc[h] @ keys[j][kvh]) + b
            mx = max(logits.max(), sink[h])
            e = np.exp(logits - mx)
            denom = e.sum() + math.exp(sink[h] - mx)
            out[h] = (e / denom) @ np.stack([vals[j][kvh] for j in range(kv_len)])
        attn_o = out.reshape(-1) * (1.0 / (1.0 + np.exp(-gate)))
        o_out = nv[nm("attn.wo_ud.weight")] @ attn_o
        ring_o = np.vstack([ring_o[1:], o_out[None, :]])
        post_attn = short_conv_static(ring_o, w_ao, k, (k + 1) ** -0.5)
        residual = post_attn                      # layer 0 is a block boundary
        mq = g_raw["model.llm.attn_res.mlp_queries.0"]
        mlp_in = depth_mix([x, residual], [mq], eps)[0]
        mlp_x = rmsnorm(mlp_in, rw[nm("mlp_norm.weight")], eps)
        acc = expert_block(base, idx, l, mlp_x)
        ring_m = np.vstack([ring_m[1:], acc[None, :]])
        post_mlp = short_conv_static(ring_m, w_mo, k, (k + 1) ** -0.5)
        x_out = residual + post_mlp
        outs.append(x_out)
        if pos == len(states) - 1:
            mags = {"kconv": kconv, "vconv": vconv, "attn_o": attn_o,
                    "o_out": o_out, "residual": residual, "moe_acc": acc}
    return outs, mags


def two_position_check(base, idx, args):
    L = idx["num_layers"]
    H = idx["hidden_size"]
    t0 = np.fromfile(args.trace, dtype=np.float32).astype(np.float64).reshape(L + 1, H)
    t1 = np.fromfile(args.trace2, dtype=np.float32).astype(np.float64).reshape(L + 1, H)
    print(f"pos0 trace |row0|={np.linalg.norm(t0[0]):.6g} "
          f"|row1|={np.linalg.norm(t0[1]):.6g}")
    print(f"pos1 trace |row0|={np.linalg.norm(t1[0]):.6g} "
          f"|row1|={np.linalg.norm(t1[1]):.6g}")
    # Both positions in one sequential pass; every position's layer-0 input is
    # the ENGINE's own state, so the only thing under test is layer 0 itself.
    states = [t0[0].copy(), t1[0].copy()]
    outs, mags = layer0_sequence(base, idx, states, [args.tok, args.tok2])
    for pos, (ref, eng) in enumerate(zip(outs, (t0[1], t1[1]))):
        rel = np.linalg.norm(ref - eng) / (np.linalg.norm(ref) + 1e-30)
        print(f"layer 0 at pos {pos}: relL2={rel:.6g} "
              f"|ref|={np.linalg.norm(ref):.6g} |eng|={np.linalg.norm(eng):.6g}")
    for name, v in mags.items():
        print(f"  [host pos1] {name:10s} {np.linalg.norm(v):.8g}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vault")
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--tok", type=int, default=1000)
    ap.add_argument("--trace2", default="",
                    help="second trace file (pos 1); required with --tok2")
    ap.add_argument("--tok2", type=int, default=-1,
                    help="second token: run pos 0 with --tok, then compare the "
                         "trace at pos 1 (multi-position path)")
    ap.add_argument("--moe-from", default="",
                    help="file of the engine's own mlp_norm output (fp16); "
                         "computes ONLY the MoE from it, separating expert "
                         "SELECTION from expert MATH")
    ap.add_argument("--variant", default="ref",
                    choices=["ref", "single_stream", "no_dyn_taps", "no_conv"],
                    help="deliberately-wrong full-layer variant, to see which "
                         "one the engine actually implements")
    ap.add_argument("--from-trace", type=int, default=-1,
                    help="seed from trace[N] and recompute only layer N")
    args = ap.parse_args()

    base = Path(args.vault) / "logestix"
    _pre = None
    idx = json.loads((base / "model.index.json").read_text())
    H = idx["hidden_size"]
    L = idx["num_layers"]
    nh = idx["num_attention_heads"]
    nkv = idx["num_key_value_heads"]
    hd = idx["head_dim"]
    drel = idx["d_rel"]
    rel_extent = idx["rel_extent"]
    k = idx["conv_kernel"]
    E = idx["num_experts"]
    topk = idx["num_experts_per_tok"]
    I = idx["moe_intermediate_size"]
    eps = idx["rms_norm_eps"]
    scale = idx["attn_scale_denom"] ** -0.5
    full_layers = set(idx["full_layers"])

    if args.tok2 >= 0:
        idx0 = json.loads((base / "model.index.json").read_text())
        return two_position_check(base, idx0, args)

    if args.moe_from:
        idx0 = json.loads((base / "model.index.json").read_text())
        hh = np.fromfile(args.moe_from, dtype=np.float16).astype(np.float64)
        moe_only(base, idx0, args.from_trace, hh)
        return 0

    g_raw = raw_tensors(base / "pinned" / "global.arrow")
    g_nv = nvfp4_tensors(base / "pinned" / "global_nvfp4.arrow")

    trace = np.fromfile(args.trace, dtype=np.float32).astype(np.float64)
    trace = trace.reshape(L + 1, H)
    if not np.all(np.isfinite(trace[0])):
        print("seed state (post-embed_norm) is not finite -- engine is broken "
              "before layer 0")
        return 1

    embed = g_nv["model.llm.embed.weight"]
    embed_norm = g_raw["model.llm.embed_norm.weight"]
    x = rmsnorm(embed[args.tok].astype(np.float64), embed_norm, eps)
    print(f"token {args.tok}: engine seed vs host seed "
          f"relL2={np.linalg.norm(x - trace[0]) / (np.linalg.norm(x) + 1e-30):.6g}")

    # Replaying one layer from a known-good seed isolates that layer's maths
    # from every earlier layer's fp16 noise. The block stack cannot be rebuilt
    # from the trace alone, so only the FIRST-attention-mixture-free case (a
    # layer whose mixture ignores the stack) is exact; full layers still mix, so
    # they are compared with the stack replaced by the trace row l-1 -- good
    # enough to tell "the full-layer path is broken" from "it is fine".
    if args.from_trace >= 0:
        l0 = args.from_trace
        # The block stack holds the state ENTERING each boundary layer, which is
        # exactly trace[boundary] (row 0 is the post-embed_norm state).
        blocks = [trace[b].copy() for b in range(0, l0, idx["attn_res_block"])]
        x = trace[l0].copy()
        layers = [l0]
    else:
        blocks = []
        layers = list(range(min(args.layers, L)))

    for l in layers:
        nv = nvfp4_tensors(base / "pinned" / f"layer_{l}_nvfp4.arrow")
        rw = raw_tensors(base / "pinned" / f"layer_{l}.arrow")
        full = l in full_layers
        is_boundary = (l % idx["attn_res_block"]) == 0

        def nm(suffix):
            return f"model.llm.layers.{l}.{suffix}"

        # (a) attention input mixture
        block_residual = x.copy()
        sources = list(blocks) + [x]
        if l > 0:
            qname = f"model.llm.attn_res.attn_queries.{l}"
            bank = g_raw[qname].reshape(-1, H)
            mix = depth_mix(sources, list(bank), eps)
            if full and args.variant == "single_stream":
                qin = kin = vin = mix[0]
            elif full:
                qin, kin, vin = mix[0], mix[1], mix[2]
            else:
                qin = kin = vin = mix[0]
        else:
            qin = kin = vin = x

        # (b) norms + projections
        if full:
            qs = rmsnorm(qin, rw[nm("q_input_norm.weight")], eps)
            ks = rmsnorm(kin, rw[nm("k_input_norm.weight")], eps)
            vs = rmsnorm(vin, rw[nm("v_input_norm.weight")], eps)
        else:
            qs = ks = vs = rmsnorm(qin, rw[nm("attn_norm.weight")], eps)

        q = nv[nm("attn.wq_du.weight")] @ qs
        kv_k = nv[nm("attn.wk_dv.weight")] @ ks
        kv_v = nv[nm("attn.wv_dv.weight")] @ vs
        rel = nv[nm("attn.wr_du.weight")] @ qs
        gate = nv[nm("attn.gate_attn.weight")] @ qs
        sink = rw[nm("attn.attn_sink_proj.weight")] @ qs

        # (c) k/v short convolutions on the RAW projections
        # Position 0: the k-1 history taps are ZERO (the engine's ring starts
        # zeroed), so only the newest tap contributes. Filling the history with
        # copies of the current token -- the obvious first guess -- is a
        # different convolution.
        hist_k = np.zeros((k, kv_k.size))
        hist_k[-1] = kv_k
        hist_v = np.zeros((k, kv_v.size))
        hist_v[-1] = kv_v
        if full:
            taps_k = rw[nm("attn.k_sconv.static_w")]      # [k, kv_dim]
            stat = taps_k          # lag-major already: [k][ch]
            dyn_k = nv[nm("attn.k_sconv.weight_proj")] @ ks
            dyn_k = dyn_k.reshape(nkv, k)
            taps_kc = np.stack([stat[lag] for lag in range(k)], axis=1) + \
                np.repeat(dyn_k, hd, axis=0)              # [kv_dim, k]
            if args.variant == "no_dyn_taps":
                taps_kc = np.repeat(np.stack([stat[lag] for lag in range(k)], 1)[:, :1], k, axis=1)
            kconv = short_conv_dynamic(hist_k, taps_kc, k, (2 * k + 1) ** -0.5)
            taps_v = rw[nm("attn.v_sconv.static_w")]
            dyn_v = (nv[nm("attn.v_sconv.weight_proj")] @ vs).reshape(nkv, k)
            taps_vc = np.stack([taps_v[lag] for lag in range(k)], axis=1) + \
                np.repeat(dyn_v, hd, axis=0)
            if args.variant == "no_dyn_taps":
                taps_vc = np.repeat(np.stack([taps_v[lag] for lag in range(k)], 1)[:, :1], k, axis=1)
            vconv = short_conv_dynamic(hist_v, taps_vc, k, (2 * k + 1) ** -0.5)
        else:
            w_k = rw[nm("attn.k_sconv.weight")].reshape(-1, k)   # [kv_dim, k]
            kconv = short_conv_static(hist_k, w_k, k, (k + 1) ** -0.5)
            w_v = rw[nm("attn.v_sconv.weight")].reshape(-1, k)
            vconv = short_conv_static(hist_v, w_v, k, (k + 1) ** -0.5)

        if args.variant == "no_conv":
            kconv = kv_k
            vconv = kv_v
        def magp(name, v):
            print(f"  [host] {name:10s} {np.linalg.norm(v):.8g}")
        magp("qs", qs); magp("ks", ks); magp("vs", vs)
        magp("q", q); magp("rel_raw", rel)
        magp("kconv", kconv); magp("vconv", vconv)
        # (d) clips / value norm / log scaling
        qc = np.stack([l2_clip(q[h * hd:(h + 1) * hd],
                               rw[nm("attn.q_l2_clip.scale")][h].astype(np.float64),
                               idx["qk_l2_clip"]) for h in range(nh)])
        kc = np.stack([l2_clip(kconv[h * hd:(h + 1) * hd], None,
                               idx["qk_l2_clip"]) for h in range(nkv)])
        vn = np.stack([unweighted_rmsnorm(vconv[h * hd:(h + 1) * hd], eps)
                       for h in range(nkv)])
        rc = np.stack([l2_clip(rel[h * drel:(h + 1) * drel],
                               rw[nm("attn.rel_l2_clip.scale")][h].astype(np.float64),
                               idx["rel_l2_clip"]) for h in range(nh)])
        if full:
            pos = 0
            tau = 1.0 + idx["log_scaling_alpha"] * math.log(
                max((pos + 1) / idx["log_scaling_n_floor"], 1.0))
            qc = qc * tau
            rc = rc * tau

        # (e) attention over the cache (position 0: the token itself)
        group = nh // nkv
        rel_proj = rw[nm("attn.rel_logits_proj.proj")]      # [d_rel, extent]
        extent = rel_extent if full else idx["sliding_window_size"]
        bias = np.zeros((nh, 1))
        for h in range(nh):
            prof = rc[h] @ rel_proj                        # [extent]
            bias[h, 0] = prof[0] if 0 < extent else 0.0
        out = np.zeros((nh, hd))
        for h in range(nh):
            kvh = h // group
            logit = np.array([scale * float(qc[h] @ kc[kvh]) + bias[h, 0]])
            mx = max(logit.max(), sink[h])
            e = np.exp(logit - mx)
            denom = e.sum() + math.exp(sink[h] - mx)
            out[h] = (e / denom)[0] * vn[kvh]
        attn_o = out.reshape(-1) * (1.0 / (1.0 + np.exp(-gate)))
        magp("attn_o", attn_o); magp("sink", sink); magp("bias", bias[:, 0])
        o_out = nv[nm("attn.wo_ud.weight")] @ attn_o
        magp("o_out", o_out)
        w_ao = rw[nm("attn_sconv.weight")].reshape(-1, k)
        hist_o = np.zeros((k, H)); hist_o[-1] = o_out
        post_attn = short_conv_static(hist_o, w_ao, k, (k + 1) ** -0.5)

        # (g) residual / block stack
        if is_boundary:
            blocks.append(block_residual)
            residual = post_attn
        else:
            residual = block_residual + post_attn
        mlp_sources = list(blocks) + [residual]
        mq = g_raw[f"model.llm.attn_res.mlp_queries.{l}"]
        mlp_in = depth_mix(mlp_sources, [mq], eps)[0]
        magp("residual", residual); magp("mlp_mix", mlp_in)
        mlp_x = rmsnorm(mlp_in, rw[nm("mlp_norm.weight")], eps)

        # (h) router + experts
        router_w = rw[nm("mlp.gate.weight")]
        router_b = rw[nm("mlp.gate.bias")]
        logits = router_w @ mlp_x
        scores = 1.0 / (1.0 + np.exp(-logits))
        idx_sel = np.argsort(-(scores + router_b), kind="stable")[:topk]
        sel = np.sort(idx_sel)
        wsel = scores[sel]
        coeff = wsel / (wsel.sum() + 1e-20) * idx["routed_scaling_factor"]
        acc = np.zeros(H)
        for c, e in enumerate(sel):
            cols, gs = expert_planes(base / "experts" / f"layer_{l}" / f"exp_{e}.arrow")
            gate_e = dequant_nvfp4(cols[0], cols[3], gs[0], I, H)
            down_e = dequant_nvfp4(cols[1], cols[4], gs[1], H, I)
            up_e = dequant_nvfp4(cols[2], cols[5], gs[2], I, H)
            g_e = gate_e @ mlp_x
            u_e = up_e @ mlp_x
            act = (np.tanh(g_e / idx["mlp_gate_softcap"]) * idx["mlp_gate_softcap"]) \
                * (1.0 / (1.0 + np.exp(-g_e))) \
                * (np.tanh(u_e / idx["mlp_up_softcap"]) * idx["mlp_up_softcap"])
            acc += coeff[c] * (down_e @ act)
        magp("moe_acc", acc)
        # (i) post-MLP conv + residual
        w_mo = rw[nm("mlp_sconv.weight")].reshape(-1, k)
        hist_m = np.zeros((k, H)); hist_m[-1] = acc
        post_mlp = short_conv_static(hist_m, w_mo, k, (k + 1) ** -0.5)
        x = residual + post_mlp

        ref = x
        eng = trace[l + 1]
        rel = np.linalg.norm(ref - eng) / (np.linalg.norm(ref) + 1e-30)
        print(f"layer {l:2d} ({'full' if full else 'swa '}): "
              f"relL2={rel:.6g}  |ref|={np.linalg.norm(ref):.4g} "
              f"|eng|={np.linalg.norm(eng):.4g}")
        if not np.isfinite(rel) or rel > 0.2:
            print(f"  ^ FIRST DIVERGENCE at layer {l}")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
