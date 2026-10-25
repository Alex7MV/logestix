#!/usr/bin/env python3
"""Compare the engine's per-layer hidden states against the checkpoint's own
reference math (DeepSeek-V3 MLA + MoE), with SOURCE weights: BF16 attention and
INT4 group-32 experts dequantized only for the experts the router selects.

This is the instrument that would have caught the RoPE pairing and attention
scale defects: it does not share a line of code with the engine, and it reads
the published safetensors, not the converted tree.

Engine side: run a generation with KIMI_DUMP_LAYERS=0,1,30,60 (writes
/tmp/kimi_layer_<L>.bin = int32 seq, int32 hidden, seq*hidden bf16 rows).

Usage:
  KIMI_DUMP_LAYERS=0,1,30,60 LSX_MODEL_ROOT=/mnt/fast_storage/huggingface \
      ./build/linux-release-znver4-aocc/tests/test_kimi_moe_selfcheck
  python3 tools/kimi_layer_oracle.py \
      --model /mnt/fast_storage/huggingface/Kimi-K2.7-Code \
      --ids "The capital of France is" --layers 0,1,30,60 --dump-dir /tmp
"""
import argparse
import glob
import json
import math
import os
import struct

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


class Shards:
    """Lazy safetensors access (mmap; only the requested tensor is paged in)."""

    def __init__(self, model_dir):
        self.index = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    self.index[name] = path

    def get(self, name, dtype=torch.bfloat16):
        with safe_open(self.index[name], framework="pt", device="cpu") as f:
            return f.get_tensor(name).to(dtype)


def dequant_int4(packed, scale, cols):
    """compressed-tensors pack-quantized symmetric int4: unsigned nibble with
    implicit zero point 8 (offset binary), low nibble first."""
    words = packed.to(torch.int32)
    shifts = torch.arange(8, dtype=torch.int32) * 4
    nib = (words.unsqueeze(-1) >> shifts) & 0xF
    vals = nib.reshape(words.shape[0], cols).to(torch.float32) - 8.0
    return vals * scale.repeat_interleave(32, dim=1).to(torch.float32)


def rms_norm(x, w, eps):
    v = x.to(torch.float32)
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.to(torch.float32)).to(torch.bfloat16)


def yarn_inv_freq(dim=64, base=50000.0, factor=64.0, orig=4096.0,
                  beta_fast=32.0, beta_slow=1.0):
    def corr(rot):
        return dim * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    lo, hi = math.floor(corr(beta_fast)), math.ceil(corr(beta_slow))
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    extra = 1.0 / (base ** (i / dim))
    inter = extra / factor
    ramp = torch.clamp((i / 2 - lo) / (hi - lo), 0.0, 1.0)
    return inter * ramp + extra * (1.0 - ramp)   # mask = 1 - ramp


def rope(x, positions, inv):
    """Adjacent-pair (interleaved) RoPE: (2p, 2p+1) rotate by pos*inv[p]."""
    out = x.clone().to(torch.float32)
    for p in range(inv.numel()):
        th = positions.to(torch.float32) * inv[p]
        c, s = torch.cos(th), torch.sin(th)
        a, b = out[:, 2 * p].clone(), out[:, 2 * p + 1].clone()
        out[:, 2 * p] = a * c - b * s
        out[:, 2 * p + 1] = a * s + b * c
    return out.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids", required=True, help="prompt text")
    ap.add_argument("--dump-dir", default="/tmp")
    ap.add_argument("--layers", default="0,1,30,60")
    ap.add_argument("--input-ids", default=None,
                    help="optional JSON list of ids (must match the engine)")
    args = ap.parse_args()
    want = [int(x) for x in args.layers.split(",")]

    cfg = json.load(open(os.path.join(args.model, "config.json")))["text_config"]
    if args.input_ids:
        ids = json.loads(args.input_ids)
    else:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        ids = tok(args.ids).input_ids
    seq = len(ids)
    print("ids:", ids)

    sh = Shards(args.model)
    H = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    eps = cfg["rms_norm_eps"]
    top_k = cfg["num_experts_per_tok"]
    scale = (192 ** -0.5) * (0.1 * math.log(64.0) + 1.0) ** 2

    def t(name):
        return sh.get("language_model.model." + name)

    def lin(x, w):
        # Checkpoint weights are [out, in]; torch linear is x @ w.T.
        return torch.nn.functional.linear(x, w)

    x = t("embed_tokens.weight")[torch.tensor(ids)]
    inv = yarn_inv_freq()
    pos = torch.arange(seq)
    causal = torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1)
    print("layer   relL2     cos")

    for layer in range(0, max(want) + 1):
        p = f"layers.{layer}."
        # ---- MLA attention ----
        h = rms_norm(x, t(p + "input_layernorm.weight"), eps)
        q = lin(lin(h, t(p + "self_attn.q_a_proj.weight")),
                t(p + "self_attn.q_b_proj.weight"))
        q = lin(rms_norm(lin(h, t(p + "self_attn.q_a_proj.weight")),
                         t(p + "self_attn.q_a_layernorm.weight"), eps),
                t(p + "self_attn.q_b_proj.weight"))
        q = q.view(seq, heads, 192)
        q_nope, q_pe = q[..., :128], q[..., 128:]
        kv = lin(h, t(p + "self_attn.kv_a_proj_with_mqa.weight"))
        latent, k_pe = kv[:, :512], kv[:, 512:]
        latent = rms_norm(latent, t(p + "self_attn.kv_a_layernorm.weight"), eps)
        kvb = lin(latent, t(p + "self_attn.kv_b_proj.weight")).view(seq, heads, 256)
        k_nope, v = kvb[..., :128], kvb[..., 128:]
        q_pos = pos.repeat_interleave(heads)
        q_pe = rope(q_pe.reshape(seq * heads, 64), q_pos, inv).view(seq, heads, 64)
        k_pe = rope(k_pe, pos, inv)
        attn = torch.zeros(seq, heads, 128, dtype=torch.float32)
        for hd in range(heads):
            qh = torch.cat([q_nope[:, hd], q_pe[:, hd]], dim=-1).to(torch.float32)
            kh = torch.cat([k_nope[:, hd], k_pe], dim=-1).to(torch.float32)
            s = qh @ kh.T * scale
            s = s.masked_fill(causal, float("-inf"))
            attn[:, hd] = torch.softmax(s, dim=-1) @ v[:, hd].to(torch.float32)
        o = lin(attn.reshape(seq, 8192).to(torch.bfloat16),
                t(p + "self_attn.o_proj.weight"))
        x = x + o.to(torch.bfloat16)

        # ---- FFN / MoE ----
        h = rms_norm(x, t(p + "post_attention_layernorm.weight"), eps)
        if layer < cfg["first_k_dense_replace"]:
            g = lin(h, t(p + "mlp.gate_proj.weight"))
            u = lin(h, t(p + "mlp.up_proj.weight"))
            d = lin((torch.nn.functional.silu(g.to(torch.float32)) *
                     u.to(torch.float32)).to(torch.bfloat16),
                    t(p + "mlp.down_proj.weight"))
        else:
            sg = lin(h, t(p + "mlp.shared_experts.gate_proj.weight"))
            su = lin(h, t(p + "mlp.shared_experts.up_proj.weight"))
            d = lin((torch.nn.functional.silu(sg.to(torch.float32)) *
                     su.to(torch.float32)).to(torch.bfloat16),
                    t(p + "mlp.shared_experts.down_proj.weight"))
            logits = h.to(torch.float32) @ t(p + "mlp.gate.weight").to(torch.float32).T
            scores = torch.sigmoid(logits)
            bias = t(p + "mlp.gate.e_score_correction_bias", torch.float32)
            top = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
            w = torch.gather(scores, 1, top)
            w = w / (w.sum(-1, keepdim=True) + 1e-20) * cfg["routed_scaling_factor"]
            for ei in range(top.shape[1]):
                for ti in range(seq):
                    e = int(top[ti, ei])
                    base = f"{p}mlp.experts.{e}."
                    gg = dequant_int4(
                        sh.get("language_model.model." + base + "gate_proj.weight_packed"),
                        sh.get("language_model.model." + base + "gate_proj.weight_scale"),
                        H)
                    uu = dequant_int4(
                        sh.get("language_model.model." + base + "up_proj.weight_packed"),
                        sh.get("language_model.model." + base + "up_proj.weight_scale"),
                        H)
                    dd = dequant_int4(
                        sh.get("language_model.model." + base + "down_proj.weight_packed"),
                        sh.get("language_model.model." + base + "down_proj.weight_scale"),
                        2048)
                    hf = h[ti].to(torch.float32)
                    y = (torch.nn.functional.silu(gg @ hf) * (uu @ hf)) @ dd.T
                    d[ti] += (float(w[ti, ei]) * y).to(torch.bfloat16)
        x = x + d.to(torch.bfloat16)

        if layer in want:
            path = os.path.join(args.dump_dir, f"kimi_layer_{layer}.bin")
            if not os.path.exists(path):
                print(f"{layer:5d}   NO ENGINE DUMP ({path})")
                continue
            with open(path, "rb") as f:
                n_seq, n_hidden = struct.unpack("<ii", f.read(8))
                raw = f.read(n_seq * n_hidden * 2)
            eng = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(
                n_seq, n_hidden)
            if n_seq != seq or n_hidden != H:
                print(f"{layer:5d}   SHAPE MISMATCH engine={n_seq}x{n_hidden} ref={seq}x{H}")
                continue
            ref = x.to(torch.float32)
            e = eng.to(torch.float32)
            rel = float((e - ref).norm() / ref.norm())
            cos = float(torch.nn.functional.cosine_similarity(
                e.reshape(-1), ref.reshape(-1), dim=0))
            print(f"{layer:5d}   {rel:.5f}  {cos:.6f}")


if __name__ == "__main__":
    main()
