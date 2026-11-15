#!/usr/bin/env python3
"""Full-stack float64 forward of the CONVERTED Inkling-2-Small artifact.

Purpose: the engine and `transformers` now agree on the same weights
(tools/inkling_transformers_crosscheck.py), and the engine agrees with
tools/inkling_layer_ref.py on every layer. What none of those can test is the
QUANTIZATION the converter applies to the real checkpoint — every reference so
far reads the same 4-bit artifact.

This script runs the whole 48-layer stack from the artifact and can swap ONE
tensor family back to the source precision, so the question "did 4-bit on this
tensor cost us the model?" becomes a measurement instead of a guess.

Usage:
  python3 tools/inkling_full_forward.py <vault> --tokens 1,2,3 [--embed bf16]
                                      [--unembed bf16] [--attn bf16]
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "layerref", Path(__file__).parent / "inkling_layer_ref.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def src_tensor(src: Path, wm: dict, cache: dict, name: str) -> np.ndarray:
    """Reads one tensor straight from the SOURCE safetensors shards."""
    f = src / wm[name]
    if f not in cache:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            cache[f] = (json.loads(fh.read(n)), 8 + n)
    hdr, off = cache[f]
    v = hdr[name]
    with open(f, "rb") as fh:
        fh.seek(off + v["data_offsets"][0])
        buf = fh.read(v["data_offsets"][1] - v["data_offsets"][0])
    if v["dtype"] == "BF16":
        return ((np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16)
                .view(np.float32)).reshape(v["shape"]).astype(np.float64)
    return np.frombuffer(buf, dtype=np.float32).reshape(v["shape"]).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("vault")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--embed", default="artifact", choices=["artifact", "bf16"])
    ap.add_argument("--unembed", default="artifact", choices=["artifact", "bf16"])
    ap.add_argument("--attn", default="artifact", choices=["artifact", "bf16"])
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    base = Path(args.vault) / "logestix"
    src = Path(args.vault)
    idx = json.loads((base / "model.index.json").read_text())
    H = idx["hidden_size"]; L = idx["num_layers"]
    nh = idx["num_attention_heads"]; nkv = idx["num_key_value_heads"]
    hd = idx["head_dim"]; drel = idx["d_rel"]; k = idx["conv_kernel"]
    eps = idx["rms_norm_eps"]; I = idx["moe_intermediate_size"]
    topk = idx["num_experts_per_tok"]; B = idx["attn_res_block"]
    scale = idx["attn_scale_denom"] ** -0.5
    ext_full = idx["rel_extent"]; window = idx["sliding_window_size"]
    full_layers = set(idx["full_layers"])
    cap = idx["final_logit_softcap"]; unpadded = idx["unpadded_vocab_size"]

    wm = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    scache = {}
    g_raw = _ref.raw_tensors(base / "pinned" / "global.arrow")
    # A tree converted by the current converter keeps NO global NVFP4 plane:
    # embed/unembed are BF16, like the release.
    _gp = base / "pinned" / "global_nvfp4.arrow"
    g_nv = {**g_raw, **(_ref.nvfp4_tensors(_gp) if _gp.exists() else {})}

    embed = g_nv["model.llm.embed.weight"].astype(np.float64)
    unembed = g_nv["model.llm.unembed.weight"].astype(np.float64)
    if args.embed == "bf16":
        embed = src_tensor(src, wm, scache, "model.llm.embed.weight")
    if args.unembed == "bf16":
        unembed = src_tensor(src, wm, scache, "model.llm.unembed.weight")

    ids = [int(x) for x in args.tokens.split(",")]
    blocks = []
    # The short convolutions are over POSITIONS *within one layer*: every layer
    # owns its own taps, so the history ring must be per layer. Sharing one ring
    # across layers (the first version of this tool) leaks layer l-1's
    # projection into layer l's convolution. At position 0 that is the only
    # surviving tap, which made the whole-token reference disagree with both the
    # engine and the per-layer reference from layer 1 onward while layer 0 —
    # where the ring is still zero — agreed to 5e-5.
    ring_k = {}; ring_v = {}; ring_o = {}; ring_m = {}

    def push(store, l, v):
        r = store.get(l)
        if r is None:
            r = np.zeros((k, v.size))
        r = np.vstack([r[1:], v[None, :]])
        store[l] = r
        return r
    keys = []; vals = []
    x = None
    for pos, tok in enumerate(ids):
        x = _ref.rmsnorm(embed[tok].astype(np.float64),
                         g_raw["model.llm.embed_norm.weight"], eps)
        for l in range(L):
            full = l in full_layers
            boundary = (l % B) == 0
            _lp = base / "pinned" / f"layer_{l}_nvfp4.arrow"
            rw = _ref.raw_tensors(base / "pinned" / f"layer_{l}.arrow")
            # dense weights are BF16 in the artifact (an older tree kept them in
            # an NVFP4 batch), so the projection map is rw + any legacy plane.
            nv = {**rw, **(_ref.nvfp4_tensors(_lp) if _lp.exists() else {})}
            if args.attn == "bf16":
                for nm_src, key in (("attn.wq_du.weight", "attn.wq_du.weight"),
                                    ("attn.wk_dv.weight", "attn.wk_dv.weight"),
                                    ("attn.wv_dv.weight", "attn.wv_dv.weight"),
                                    ("attn.wo_ud.weight", "attn.wo_ud.weight"),
                                    ("attn.wr_du.weight", "attn.wr_du.weight"),
                                    ("attn.gate_attn.weight", "attn.gate_attn.weight")):
                    full_name = f"model.llm.layers.{l}.{nm_src}"
                    nv[full_name] = src_tensor(src, wm, scache, full_name)
            nm = lambda s: f"model.llm.layers.{l}.{s}"

            if l > 0:
                bank = g_raw[f"model.llm.attn_res.attn_queries.{l}"].reshape(-1, H)
                mix = _ref.depth_mix(blocks + [x], list(bank), eps)
                qin, kin, vin = (mix[0], mix[1], mix[2]) if full else (mix[0],) * 3
            else:
                qin = kin = vin = x
            if full:
                qs = _ref.rmsnorm(qin, rw[nm("q_input_norm.weight")], eps)
                ks = _ref.rmsnorm(kin, rw[nm("k_input_norm.weight")], eps)
                vs = _ref.rmsnorm(vin, rw[nm("v_input_norm.weight")], eps)
            else:
                qs = ks = vs = _ref.rmsnorm(qin, rw[nm("attn_norm.weight")], eps)
            q = nv[nm("attn.wq_du.weight")] @ qs
            kv_k = nv[nm("attn.wk_dv.weight")] @ ks
            kv_v = nv[nm("attn.wv_dv.weight")] @ vs
            rel = nv[nm("attn.wr_du.weight")] @ qs
            gate = nv[nm("attn.gate_attn.weight")] @ qs
            sink = rw[nm("attn.attn_sink_proj.weight")] @ qs

            rk = push(ring_k, l, kv_k)
            rv = push(ring_v, l, kv_v)
            if full:
                sk = rw[nm("attn.k_sconv.static_w")]
                dk = (nv[nm("attn.k_sconv.weight_proj")] @ ks).reshape(nkv, k)
                tk = np.stack([sk[lag] for lag in range(k)], 1) + np.repeat(dk, hd, 0)
                sv = rw[nm("attn.v_sconv.static_w")]
                dv = (nv[nm("attn.v_sconv.weight_proj")] @ vs).reshape(nkv, k)
                tv = np.stack([sv[lag] for lag in range(k)], 1) + np.repeat(dv, hd, 0)
                kconv = _ref.short_conv_dynamic(rk, tk, k, (2 * k + 1) ** -0.5)
                vconv = _ref.short_conv_dynamic(rv, tv, k, (2 * k + 1) ** -0.5)
            else:
                kconv = _ref.short_conv_static(
                    rk, rw[nm("attn.k_sconv.weight")].reshape(-1, k), k,
                    (k + 1) ** -0.5)
                vconv = _ref.short_conv_static(
                    rv, rw[nm("attn.v_sconv.weight")].reshape(-1, k), k,
                    (k + 1) ** -0.5)

            qc = np.stack([_ref.l2_clip(q[h * hd:(h + 1) * hd],
                                        rw[nm("attn.q_l2_clip.scale")][h].astype(np.float64),
                                        idx["qk_l2_clip"]) for h in range(nh)])
            kc = np.stack([_ref.l2_clip(kconv[h * hd:(h + 1) * hd], None,
                                        idx["qk_l2_clip"]) for h in range(nkv)])
            vn = np.stack([_ref.unweighted_rmsnorm(vconv[h * hd:(h + 1) * hd], eps)
                           for h in range(nkv)])
            rc = np.stack([_ref.l2_clip(rel[h * drel:(h + 1) * drel],
                                        rw[nm("attn.rel_l2_clip.scale")][h].astype(np.float64),
                                        idx["rel_l2_clip"]) for h in range(nh)])
            if full and idx["log_scaling_n_floor"]:
                tau = 1.0 + idx["log_scaling_alpha"] * math.log(
                    max((pos + 1) / idx["log_scaling_n_floor"], 1.0))
                qc = qc * tau; rc = rc * tau
            keys.append(kc); vals.append(vn)
            extent = ext_full if full else window
            win = 0 if full else window
            grp = nh // nkv
            out = np.zeros((nh, hd))
            kv_len = len(keys)
            for h in range(nh):
                kvh = h // grp
                prof = rc[h] @ rw[nm("attn.rel_logits_proj.proj")]
                logits = np.full(kv_len, -np.inf)
                for j in range(kv_len):
                    if win == 0 or j > pos - win:
                        d = pos - j
                        b = prof[d] if 0 <= d < extent else 0.0
                        logits[j] = scale * float(qc[h] @ keys[j][kvh]) + b
                mx = max(logits.max(), sink[h])
                e = np.exp(logits - mx)
                den = e.sum() + math.exp(sink[h] - mx)
                out[h] = (e / den) @ np.stack([vals[j][kvh] for j in range(kv_len)])
            attn_o = out.reshape(-1) * (1.0 / (1.0 + np.exp(-gate)))
            o_out = nv[nm("attn.wo_ud.weight")] @ attn_o
            ro = push(ring_o, l, o_out)
            post = _ref.short_conv_static(
                ro, rw[nm("attn_sconv.weight")].reshape(-1, k), k,
                (k + 1) ** -0.5)
            if boundary:
                blocks.append(x.copy())
                residual = post
            else:
                residual = x + post
            ms = list(blocks) + [residual]
            mlp_in = _ref.depth_mix(ms, [g_raw[f"model.llm.attn_res.mlp_queries.{l}"]],
                                    eps)[0]
            mlp_x = _ref.rmsnorm(mlp_in, rw[nm("mlp_norm.weight")], eps)
            acc = _ref.expert_block(base, idx, l, mlp_x)
            rm = push(ring_m, l, acc)
            post_mlp = _ref.short_conv_static(
                rm, rw[nm("mlp_sconv.weight")].reshape(-1, k), k,
                (k + 1) ** -0.5)
            x = residual + post_mlp
        print(f"pos {pos}: |x|={np.linalg.norm(x):.6g}", flush=True)

    final = _ref.depth_mix(list(blocks) + [x],
                           [g_raw["model.llm.attn_res.output_query"]], eps)[0]
    hh = _ref.rmsnorm(final, g_raw["model.llm.norm.weight"], eps)
    lg = unembed @ hh
    lg = np.tanh(lg / cap) * cap
    top = np.argsort(-lg[:unpadded])[:args.topk]
    print(f"\nfinal top{args.topk}: "
          f"{[(int(i), round(float(lg[i]), 4)) for i in top]}")
    print(f"|hh| = {np.linalg.norm(hh):.6g}")
    np.save(Path(args.vault) / "ff_logits.npy", lg[:unpadded].astype(np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main())
