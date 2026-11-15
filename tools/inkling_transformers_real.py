#!/usr/bin/env python3
"""Runs the RELEASE's own `transformers` implementation on REAL weights.

Every other reference in this tree (tools/inkling_layer_ref.py,
tools/inkling_full_forward.py) is my own re-derivation from the artifact, so the
three of them agreeing proves self-consistency, not correctness: a wrong reading
of a tensor's role would be reproduced by all of them.

This tool closes that gap. It builds `Inkling2TextModel` from the release's
modeling code (transformers 5.23) with a TRUNCATED stack — the first N layers —
loads the real dense weights from the converted artifact, and dequantizes only
the experts the router actually selects (so a 594B model can be checked in a few
tens of GB). The hidden state after layer N-1 is then compared against the
engine's parity trace (row N), which is the state after the same layer.

Usage:
  PYTHONPATH=/tmp/tfnew/x:/tmp/tfk python3 tools/inkling_transformers_real.py \
      <vault> --trace <trace.bin> --layers 6

The trace must have been produced by the engine for the SAME token ids, passed
with --tokens (comma separated).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)


def read_arrow(path):
    with pa.memory_map(str(path), "r") as s:
        return ipc.open_file(s).read_all()


def col_bytes(col, i):
    """Row `i` of a large_binary OR list<uint8> column as a byte array."""
    s = col[i]
    if hasattr(s, "as_buffer"):
        return np.frombuffer(s.as_buffer(), dtype=np.uint8)
    return s.values.to_numpy(zero_copy_only=False).astype(np.uint8)


def e4m3_to_float(b):
    b = b.astype(np.int32)
    sign = np.where((b >> 7) & 1, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    man = b & 0x7
    val = np.where(exp == 0, man * (2.0 ** -9),
                   (1.0 + man / 8.0) * (2.0 ** (exp - 7)))
    return sign * val


def dequant_nvfp4(packed, scales, gs, rows, cols):
    b = packed.view(np.uint8).reshape(rows, cols // 2)
    lo = (b & 0x0F).astype(np.int32)
    hi = (b >> 4).astype(np.int32)
    nib = np.empty((rows, cols), dtype=np.int32)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    # The fourth bit of each nibble is the SIGN; E2M1 holds the 8 magnitudes.
    q = E2M1[nib & 7] * np.where(nib & 8, -1.0, 1.0)
    sc = e4m3_to_float(scales.reshape(rows, cols // 16))
    return q * np.repeat(sc, 16, axis=1) * float(gs)


def raw_tensors(path):
    """name -> float64 tensor for a pinned plane (BF16 or F32)."""
    b = read_arrow(path)
    out = {}
    for i in range(b.num_rows):
        name = b.column(0)[i].as_py()
        shape = tuple(b.column(1)[i].as_py())
        dt = b.column(2)[i].as_py()
        buf = b.column(3)[i].as_buffer()
        if dt == "BF16":
            u = np.frombuffer(buf, dtype=np.uint16)
            out[name] = (u.astype(np.uint32) << 16).view(np.float32).astype(
                np.float64).reshape(shape)
        elif dt == "F32":
            out[name] = np.frombuffer(buf, dtype=np.float32).astype(
                np.float64).reshape(shape)
        else:
            out[name] = np.zeros(shape, dtype=np.float64)
    return out


def expert_planes(base, l, e, I, H):
    """(gate [I,H], down [H,I], up [I,H]) from the artifact's expert record."""
    b = read_arrow(base / "experts" / f"layer_{l}" / f"exp_{e}.arrow")
    cols = [col_bytes(b.column(c), 0) for c in range(6)]
    gs = [float(b.column(c)[0].as_py()) for c in (6, 7, 8)]
    if cols[0].size == I * H:            # MXFP8 (E4M3 + ue8m0/16 duplicated)
        def dec(w, s, rows, c):
            w = w.reshape(rows, c)
            s = s.reshape(rows, c // 16).astype(np.float64)
            return e4m3_to_float(w) * np.repeat(2.0 ** (s - 127.0), 16, axis=1)
        return (dec(cols[0], cols[3], I, H), dec(cols[1], cols[4], H, I),
                dec(cols[2], cols[5], I, H))
    return (dequant_nvfp4(cols[0], cols[3], gs[0], I, H),
            dequant_nvfp4(cols[1], cols[4], gs[1], H, I),
            dequant_nvfp4(cols[2], cols[5], gs[2], I, H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vault")
    ap.add_argument("--trace", required=True, help="engine parity trace (.bin)")
    ap.add_argument("--tokens", required=True, help="comma separated token ids")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    import torch
    from transformers.models.inkling2 import configuration_inkling2 as C
    from transformers.models.inkling2 import modeling_inkling2 as M

    vault = Path(args.vault)
    base = vault / "logestix"
    idx = json.loads((base / "model.index.json").read_text())
    cfg_json = json.loads((vault / "config.json").read_text())
    tc = C.Inkling2TextConfig(**cfg_json["text_config"])

    N = args.layers
    tc.num_hidden_layers = N
    tc.layer_types = tc.layer_types[:N]
    tc.local_layer_ids = [l for l in tc.local_layer_ids if l < N]
    H = tc.hidden_size

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    print(f"building Inkling2TextModel with {N} layers, dtype={args.dtype}",
          flush=True)
    with torch.device("meta"):
        model = M.Inkling2TextModel(tc)
    model.to_empty(device="cpu")
    model = model.to(dtype)
    model.eval()

    I = idx["moe_intermediate_size"]
    E = idx["num_experts"]

    # ── load the dense weights of the first N layers ──────────────────────
    g = raw_tensors(base / "pinned" / "global.arrow")
    sd = {}
    sd["embed_tokens.weight"] = g["model.llm.embed.weight"]
    sd["embed_tokens.embed_norm.weight"] = g["model.llm.embed_norm.weight"]

    ATTN = {
        "attn.wq_du.weight": "self_attn.q_proj.weight",
        "attn.wk_dv.weight": "self_attn.k_proj.weight",
        "attn.wv_dv.weight": "self_attn.v_proj.weight",
        "attn.wr_du.weight": "self_attn.r_proj.weight",
        "attn.wo_ud.weight": "self_attn.o_proj.weight",
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
        "attn_sconv.weight": "post_attention_convolution.conv1d.weight",
        "mlp_sconv.weight": "post_mlp_convolution.conv1d.weight",
        "attn_norm.weight": "input_layernorm.weight",
        "q_input_norm.weight": "q_input_layernorm.weight",
        "k_input_norm.weight": "k_input_layernorm.weight",
        "v_input_norm.weight": "v_input_layernorm.weight",
        "mlp_norm.weight": "post_attention_layernorm.weight",
        "mlp.gate.weight": "mlp.gate.weight",
    }
    for l in range(N):
        lw = raw_tensors(base / "pinned" / f"layer_{l}.arrow")
        for src, dst in ATTN.items():
            key = f"model.llm.layers.{l}.{src}"
            if key in lw:
                sd[f"layers.{l}.{dst}"] = lw[key]
        sd[f"layers.{l}.mlp.gate.e_score_correction_bias"] = lw[
            f"model.llm.layers.{l}.mlp.gate.bias"]
        sd[f"layers.{l}.mlp_input.query"] = g[
            f"model.llm.attn_res.mlp_queries.{l}"]
        if l > 0:
            # The checkpoint's attention bank is numbered by ordinal (0..46), and
            # layer 0 has no attention_input at all, so bank k belongs to layer
            # k+1. The converter re-indexed it, so the artifact key is the layer.
            sd[f"layers.{l}.attention_input.query"] = g[
                f"model.llm.attn_res.attn_queries.{l}"]

    missing, unexpected = model.load_state_dict(
        {k: torch.tensor(np.ascontiguousarray(v), dtype=dtype)
         for k, v in sd.items()}, strict=False)
    skip = [n for n in unexpected if "gate_up_proj" not in n and "down_proj" not in n]
    print(f"loaded {len(sd)} tensors; unloaded expert params stay on meta",
          flush=True)
    if skip:
        print(f"WARNING unexpected keys: {skip[:4]}", flush=True)
    still_meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    print(f"still meta: {len(still_meta)} (expert slabs expected)", flush=True)

    # ── on-demand experts ─────────────────────────────────────────────────
    cache = {}

    def expert_weights(l, e):
        key = (l, e)
        if key not in cache:
            gate, down, up = expert_planes(base, l, e, I, H)
            # gate_up_proj is [E, 2I, H] with gate rows first (the release's
            # `_apply_gate` chunks it in half), down_proj is [E, H, I].
            gu = np.concatenate([gate, up], axis=0)
            cache[key] = (torch.tensor(gu, dtype=dtype),
                          torch.tensor(down, dtype=dtype))
        return cache[key]

    for l in range(N):
        mod = model.layers[l].mlp.experts
        mod.layer_idx = l

        def fwd(hidden_states, top_k_index, top_k_weights, _self=mod):
            final = torch.zeros_like(hidden_states, dtype=torch.float32)
            for slot in range(top_k_index.shape[-1]):
                for tok in range(top_k_index.shape[0]):
                    e = int(top_k_index[tok, slot])
                    gu, dn = expert_weights(_self.layer_idx, e)
                    st = hidden_states[tok:tok + 1].to(dtype)
                    act = _self._apply_gate(torch.nn.functional.linear(st, gu))
                    out = torch.nn.functional.linear(act, dn)
                    final[tok] += (out[0].float()
                                   * top_k_weights[tok, slot].float())
            return final.to(hidden_states.dtype)

        mod.forward = fwd

    # ── run and compare ───────────────────────────────────────────────────
    ids = [int(x) for x in args.tokens.split(",") if x.strip()]
    captured = {}

    def hook(_m, _inp, out):
        captured["h"] = out[0].detach().to(torch.float32).numpy()[0] \
            if isinstance(out, tuple) else out.detach().to(torch.float32).numpy()[0]

    handle = model.layers[N - 1].register_forward_hook(hook)
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], dtype=torch.long), use_cache=False)
    handle.remove()

    tr = np.fromfile(args.trace, dtype=np.float32).astype(np.float64)
    tr = tr.reshape(idx["num_layers"] + 1, H)
    eng = tr[N]
    for pos in range(len(ids)):
        ref = captured["h"][pos].astype(np.float64)
        e = eng if pos == len(ids) - 1 else None
        if e is None:
            print(f"pos {pos}: |ref|={np.linalg.norm(ref):.6g} (engine trace "
                  f"holds only the LAST position)")
            continue
        rel = np.linalg.norm(ref - e) / (np.linalg.norm(ref) + 1e-30)
        print(f"pos {pos}: relL2={rel:.6g}  |release|={np.linalg.norm(ref):.6g} "
              f"|engine|={np.linalg.norm(e):.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
