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

import pyarrow as pa
import torch
from safetensors import safe_open
from transformers import AutoTokenizer


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e4m3_to_float(b):
    """OCP/NVIDIA E4M3FN block scale (bit 7 is the reserved sign)."""
    b = b & 0x7F
    e, m = (b >> 3) & 0xF, b & 0x7
    if e == 0:
        return m * (2.0 ** -9)
    if e == 15 and m == 7:
        return float("nan")
    return (2.0 ** (e - 7)) * (1.0 + m / 8.0)


def dequant_nvfp4(packed, scales, rows, cols, gs):
    """w = E2M1(nibble) * e4m3(block scale) * global_scale, row-major, per-16
    blocks along the flattened tensor (the engine's kv_b plane layout)."""
    b = torch.frombuffer(bytearray(packed), dtype=torch.uint8)
    lo = E2M1[(b & 0x7).long()]
    hi = E2M1[((b >> 4) & 0x7).long()]
    sign_lo = torch.where((b & 0x8) != 0, -1.0, 1.0)
    sign_hi = torch.where((b & 0x80) != 0, -1.0, 1.0)
    nib = torch.empty(b.numel() * 2, dtype=torch.float32)
    nib[0::2] = lo * sign_lo
    nib[1::2] = hi * sign_hi
    lut = torch.tensor([e4m3_to_float(int(x)) for x in range(128)],
                       dtype=torch.float32)
    sc = lut[torch.frombuffer(bytearray(scales), dtype=torch.uint8).long() & 0x7F]
    return (nib.view(rows, cols) *
            sc.repeat_interleave(16)[: rows * cols].view(rows, cols) * gs)


class Pinned:
    """Converted pinned-layer records (name -> (shape, dtype, raw bytes))."""

    def __init__(self, model_dir, layer):
        self.t = {}
        path = os.path.join(model_dir, "logestix", "pinned",
                            f"layer_{layer}.arrow")
        tb = pa.ipc.open_file(path).read_all()
        for r in tb.to_pylist():
            self.t[r["name"]] = (r["shape"], r["dtype"], r["data"])
        idx = json.load(open(os.path.join(model_dir, "logestix",
                                          "model.index.json")))
        self.gs = idx["mla_nvfp4"][str(layer)]

    def bf16(self, name, shape):
        raw = self.t[name][2]
        x = torch.frombuffer(bytearray(raw), dtype=torch.uint16)
        return x.view(torch.bfloat16).to(torch.bfloat16).view(*shape)

    def proj(self, full_prefix, src_shape):
        """Dequantize one MLA projection ("<...>.q_a_proj" -> packed/scale)."""
        rows, cols = src_shape
        pk = self.t[full_prefix + ".weight_packed"][2]
        sc = self.t[full_prefix + ".weight_scale"][2]
        base = full_prefix.split(".")[-1]
        return dequant_nvfp4(pk, sc, rows, cols,
                             float(self.gs[base])).to(torch.bfloat16)


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
    ap.add_argument("--ids", default="", help="prompt text")
    ap.add_argument("--dump-dir", default="/tmp")
    ap.add_argument("--layers", default="0,1,30,60")
    ap.add_argument("--trace", action="store_true",
                    help="print the per-layer hidden norms")
    ap.add_argument("--weights", choices=["source", "converted"], default="source",
                    help="source BF16 safetensors or the converted NVFP4 tree")
    ap.add_argument("--stage", choices=["layer", "attn"], default="layer",
                    help="compare the layer output or the post-attention residual")
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

    def t(name, dtype=torch.bfloat16):
        return sh.get("language_model.model." + name, dtype)

    def lin(x, w):
        # Checkpoint weights are [out, in]; torch linear is x @ w.T.
        return torch.nn.functional.linear(x, w)

    x0 = t("embed_tokens.weight")[torch.tensor(ids)]
    x = x0.clone()
    inv = yarn_inv_freq()
    pos = torch.arange(seq)
    causal = torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1)
    print("layer   relL2     cos")

    for layer in range(0, max(want) + 1):
        p = f"layers.{layer}."
        pinned = Pinned(args.model, layer) if args.weights == "converted" else None

        def w(name):
            """A projection weight in the requested precision."""
            full = "language_model.model." + name
            if pinned is None:
                return sh.get(full)
            prefix = full[: -len(".weight")]
            return pinned.proj(prefix, sh.get(full).shape)

        def norm_w(name):
            return pinned.bf16("language_model.model." + name, sh.get(
                "language_model.model." + name).shape) if pinned else t(name)

        # ---- MLA attention ----
        h = rms_norm(x, norm_w(p + "input_layernorm.weight"), eps)
        q = lin(rms_norm(lin(h, w(p + "self_attn.q_a_proj.weight")),
                         norm_w(p + "self_attn.q_a_layernorm.weight"), eps),
                w(p + "self_attn.q_b_proj.weight"))
        q = q.view(seq, heads, 192)
        q_nope, q_pe = q[..., :128], q[..., 128:]
        kv = lin(h, w(p + "self_attn.kv_a_proj_with_mqa.weight"))
        latent, k_pe = kv[:, :512], kv[:, 512:]
        latent = rms_norm(latent,
                          norm_w(p + "self_attn.kv_a_layernorm.weight"), eps)
        kvb = lin(latent, w(p + "self_attn.kv_b_proj.weight")).view(seq, heads, 256)
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
                w(p + "self_attn.o_proj.weight"))
        x = x + o.to(torch.bfloat16)
        attn_stage = x.clone()

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
            if args.trace:
                print(f"[trace]   attn_stage={[round(float(v),2) for v in attn_stage.float().norm(dim=1)]} "
                      f"h={[round(float(v),2) for v in h.float().norm(dim=1)]} "
                      f"d_shared={[round(float(v),2) for v in d.float().norm(dim=1)]}")
            for ei in range(top.shape[1]):
                for ti in range(seq):
                    e = int(top[ti, ei])
                    base = f"{p}mlp.experts.{e}."
                    gg = dequant_int4(
                        sh.get("language_model.model." + base + "gate_proj.weight_packed",
                               torch.int32),
                        sh.get("language_model.model." + base + "gate_proj.weight_scale"),
                        H)
                    uu = dequant_int4(
                        sh.get("language_model.model." + base + "up_proj.weight_packed",
                               torch.int32),
                        sh.get("language_model.model." + base + "up_proj.weight_scale"),
                        H)
                    dd = dequant_int4(
                        sh.get("language_model.model." + base + "down_proj.weight_packed",
                               torch.int32),
                        sh.get("language_model.model." + base + "down_proj.weight_scale"),
                        2048)
                    hf = h[ti].to(torch.float32)
                    y = (torch.nn.functional.silu(gg @ hf) * (uu @ hf)) @ dd.T
                    d[ti] += (float(w[ti, ei]) * y).to(torch.bfloat16)
        if args.trace:
            print(f"[trace]   d_total={[round(float(v),2) for v in d.float().norm(dim=1)]}")
        x = x + d.to(torch.bfloat16)
        if args.trace:
            print(f"[trace] layer {layer:2d} |x|={[round(float(v),2) for v in x.float().norm(dim=1)]}")

        if layer in want:
            tag = "layer" if args.stage == "layer" else "attn"
            path = os.path.join(args.dump_dir, f"kimi_{tag}_{layer}.bin")
            if not os.path.exists(path):
                print(f"{layer:5d}   NO ENGINE DUMP ({path})")
                continue
            with open(path, "rb") as f:
                n_seq, n_hidden = struct.unpack("<ii", f.read(8))
                raw = f.read(n_seq * n_hidden * 2)
            eng = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(
                n_seq, n_hidden)
            if n_hidden != H:
                print(f"{layer:5d}   SHAPE MISMATCH engine={n_seq}x{n_hidden} ref={seq}x{H}")
                continue
            if seq == 1 and n_seq > 1:
                # Causal prefill: row 0 of the engine dump is exactly the
                # single-token computation (position 0, attends to itself).
                eng = eng[0].unsqueeze(0)
            ref = (x if args.stage == "layer" else attn_stage).to(torch.float32)
            e = eng.to(torch.float32)
            rel = float((e - ref).norm() / ref.norm())
            cos = float(torch.nn.functional.cosine_similarity(
                e.reshape(-1), ref.reshape(-1), dim=0))
            # Bracket: is the engine row closer to the embedding (attention
            # contribution missing) or to our reference?
            emb = x0[0].to(torch.float32)
            rel_emb = float((e[0] - emb).norm() / emb.norm())
            cos_emb = float(torch.nn.functional.cosine_similarity(
                e[0].reshape(-1), emb.reshape(-1), dim=0))
            print(f"{layer:5d}   {rel:.5f}  {cos:.6f}   "
                  f"|o|/|x|={float((ref[0]-emb).norm()/emb.norm()):.4f} "
                  f"eng_vs_embed rel={rel_emb:.4f} cos={cos_emb:.5f}")


if __name__ == "__main__":
    main()
