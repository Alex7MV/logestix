#!/usr/bin/env python3
"""CPU reference for Recursive Flint (39.8B-A2.5B).

Flint has no transformers modeling code and its vendor runner (SGLang + FA4)
needs a >=141 GB GPU, so the only oracle available on a 32 GB card is a CPU
execution of the published BF16 weights. This module reconstructs the decoder
from `config.json` and the safetensors shards, reusing upstream primitives for
everything that is not a Flint delta:

  * RMSNorm, rotary embedding and rotate-half application come from
    transformers' Qwen3-MoE implementation (the family the vendor subclasses);
  * the five deltas are transcribed from the vendor plugin
    (`flint_sglang/model.py`, `tool_grammar.py`, `fa4-learned-sink-fix.patch`):
    learned per-head sinks as an extra softmax COLUMN, sliding-window layers
    with RoPE, full-attention layers with NO positional encoding, sigmoid
    top-k routing with renormalized weights, and one ungated shared expert.

It exposes the tiny surface the capture scripts need (`load()` returning an
object with `.generate`, `.eval`, `.parameters`, `.device`), plus a CLI for
probing one prompt and dumping the first-step logits so the converted engine can
be compared against it directly.

    python3 tools/flint_oracle.py --model /path/to/flint \
        --prompt "The capital of France is" --frame --max-new 32
"""
import argparse
import json
import os
import struct
import sys

import torch
import torch.nn.functional as F


# ── checkpoint access ────────────────────────────────────────────────────────

class Shards:
    """Reads tensors straight out of the shard files (file order + offsets)."""

    def __init__(self, model_dir):
        self.dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as fh:
            self.weight_map = json.load(fh)["weight_map"]
        self._headers = {}

    def _header(self, shard):
        if shard not in self._headers:
            with open(os.path.join(self.dir, shard), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._headers[shard] = json.loads(fh.read(n))
        return self._headers[shard]

    def get(self, name, dtype=torch.bfloat16):
        shard = self.weight_map[name]
        header = self._header(shard)
        info = header[name]
        start, end = info["data_offsets"]
        with open(os.path.join(self.dir, shard), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            fh.seek(8 + n + start)
            raw = fh.read(end - start)
        if info["dtype"] == "BF16":
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
        elif info["dtype"] == "F32":
            t = torch.frombuffer(bytearray(raw), dtype=torch.float32)
        else:
            raise SystemExit("unsupported dtype %s for %s" % (info["dtype"], name))
        return t.reshape(info["shape"]).to(dtype).clone()


# ── model ────────────────────────────────────────────────────────────────────

class Layer:
    """One Flint decoder layer: attention (+sink) then dense MLP or MoE."""

    def __init__(self, cfg, layer_id, shards, dtype):
        self.layer_id = layer_id
        self.is_full = cfg["layer_types"][layer_id] == "full_attention"
        self.skip_rope = layer_id in set(cfg["no_rope_layer_indices"])
        self.sliding_window = cfg["sliding_window"]
        self.is_moe = layer_id not in set(cfg["mlp_only_layers"])
        p = "model.layers.%d." % layer_id
        self.input_norm = shards.get(p + "input_layernorm.weight", torch.float32)
        self.post_norm = shards.get(p + "post_attention_layernorm.weight", torch.float32)
        self.q_proj = shards.get(p + "self_attn.q_proj.weight", dtype)
        self.k_proj = shards.get(p + "self_attn.k_proj.weight", dtype)
        self.v_proj = shards.get(p + "self_attn.v_proj.weight", dtype)
        self.o_proj = shards.get(p + "self_attn.o_proj.weight", dtype)
        self.q_norm = shards.get(p + "self_attn.q_norm.weight", torch.float32)
        self.k_norm = shards.get(p + "self_attn.k_norm.weight", torch.float32)
        self.sink = shards.get(p + "self_attn.softmax_offset", torch.float32)
        if self.is_moe:
            self.router = shards.get(p + "mlp.gate.weight", torch.float32)
            self.shared_gate = shards.get(p + "mlp.shared_expert.gate_proj.weight", dtype)
            self.shared_up = shards.get(p + "mlp.shared_expert.up_proj.weight", dtype)
            self.shared_down = shards.get(p + "mlp.shared_expert.down_proj.weight", dtype)
            n = cfg["num_experts"]
            gate = [shards.get(p + "mlp.experts.%d.gate_proj.weight" % e, dtype)
                    for e in range(n)]
            up = [shards.get(p + "mlp.experts.%d.up_proj.weight" % e, dtype)
                  for e in range(n)]
            down = [shards.get(p + "mlp.experts.%d.down_proj.weight" % e, dtype)
                    for e in range(n)]
            self.exp_gate = torch.stack(gate)
            self.exp_up = torch.stack(up)
            self.exp_down = torch.stack(down)
            del gate, up, down
        else:
            self.mlp_gate = shards.get(p + "mlp.gate_proj.weight", dtype)
            self.mlp_up = shards.get(p + "mlp.up_proj.weight", dtype)
            self.mlp_down = shards.get(p + "mlp.down_proj.weight", dtype)


class FlintLM(torch.nn.Module):
    def __init__(self, model_dir, dtype=torch.bfloat16):
        super().__init__()
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            self.cfg = json.load(fh)
        self.dtype = dtype
        shards = Shards(model_dir)
        c = self.cfg
        self.heads = c["num_attention_heads"]
        self.kv_heads = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.scale = self.head_dim ** -0.5
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"]
        self.top_k = c["num_experts_per_tok"]
        self.eos = c["eos_token_id"]
        if isinstance(self.eos, int):
            self.eos = [self.eos]
        self.embed = shards.get("model.embed_tokens.weight", dtype)
        self.head = shards.get("lm_head.weight", torch.float32)
        self.final_norm = shards.get("model.norm.weight", torch.float32)
        self.layers = [
            Layer(c, i, shards, dtype) for i in range(c["num_hidden_layers"])
        ]
        # Rotary tables (full rotary over every head dim, rotate-half pairing).
        half = self.head_dim // 2
        inv = 1.0 / (self.theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv, persistent=False)

    # -- primitives (same math as transformers' Qwen3-MoE RMSNorm/rotary) ----
    def _rms(self, x, w):
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (w * xf).to(x.dtype)

    def _rope(self, q, k, pos):
        # q: [heads, T, HD]; rotate the first `half` dims against the second.
        T = q.shape[1]
        ang = torch.arange(pos, pos + T, dtype=torch.float32)[:, None] * self.inv_freq[None, :]
        cos = ang.cos()[None, :, :]   # [1, T, half] (broadcast over heads)
        sin = ang.sin()[None, :, :]
        def rot(x):
            x1 = x[..., : self.head_dim // 2].float()
            x2 = x[..., self.head_dim // 2:].float()
            o1 = x1 * cos - x2 * sin
            o2 = x1 * sin + x2 * cos
            return torch.cat([o1, o2], dim=-1).to(x.dtype)
        return rot(q), rot(k)

    def _moe(self, layer, x):
        # Sigmoid router (no expert bias in this checkpoint), top-k by score,
        # renormalized weights, then the ungated shared expert added on top.
        T = x.shape[0]
        logits = F.linear(x.float(), layer.router.float())
        score = torch.sigmoid(logits)
        top = torch.topk(score, self.top_k, dim=-1).indices
        w = score.gather(1, top)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        out = torch.zeros(T, x.shape[-1], dtype=torch.float32)
        for t in range(T):
            for j in range(self.top_k):
                e = int(top[t, j])
                h = F.linear(x[t], layer.exp_gate[e])
                h = F.silu(h) * F.linear(x[t], layer.exp_up[e])
                out[t] += w[t, j] * F.linear(h, layer.exp_down[e]).float()
        sh = F.silu(F.linear(x, layer.shared_gate)) * F.linear(x, layer.shared_up)
        out = out + F.linear(sh, layer.shared_down).float()
        return out.to(x.dtype)

    def forward(self, ids, past=None):
        """ids: [T] int64. Returns (logits [T, V] float32, cache)."""
        pos = 0 if past is None else past["len"]
        x = F.embedding(ids, self.embed)
        cache = {"len": pos + ids.shape[0], "k": [], "v": []}
        trace = getattr(self, "trace", False)
        for layer in self.layers:
            if trace:
                print("  layer %d in max %.6g" % (layer.layer_id,
                      float(x.abs().max())))
            h = self._rms(x, layer.input_norm)
            T = h.shape[0]
            q = F.linear(h, layer.q_proj).view(T, self.heads, self.head_dim).transpose(0, 1)
            k = F.linear(h, layer.k_proj).view(T, self.kv_heads, self.head_dim).transpose(0, 1)
            v = F.linear(h, layer.v_proj).view(T, self.kv_heads, self.head_dim).transpose(0, 1)
            q = self._rms(q, layer.q_norm)
            k = self._rms(k, layer.k_norm)
            if not layer.skip_rope:
                q, k = self._rope(q, k, pos)
            if past is not None:
                k = torch.cat([past["k"][len(cache["k"])], k], dim=1)
                v = torch.cat([past["v"][len(cache["k"])], v], dim=1)
            cache["k"].append(k)
            cache["v"].append(v)
            rep = self.heads // self.kv_heads
            kx = k.repeat_interleave(rep, dim=0)
            vx = v.repeat_interleave(rep, dim=0)
            scores = torch.matmul(q.float(), kx.float().transpose(1, 2)) * self.scale
            total = k.shape[1]
            rows = torch.arange(pos, pos + T)[:, None]
            cols = torch.arange(total)[None, :]
            dist = rows - cols
            mask = dist < 0
            if not layer.is_full:
                mask |= dist >= layer.sliding_window
            scores = scores.masked_fill(mask[None], float("-inf"))
            sink = layer.sink.float()[:, None, None]
            m = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink)
            p = torch.exp(scores - m)
            denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink - m)
            weights = p / denom.to(p.dtype)
            attn = torch.matmul(weights, vx.float()).transpose(0, 1).reshape(T, -1)
            x = x + F.linear(attn.to(x.dtype), layer.o_proj)
            h2 = self._rms(x, layer.post_norm)
            mlp = self._moe(layer, h2) if layer.is_moe else F.linear(
                F.silu(F.linear(h2, layer.mlp_gate)) * F.linear(h2, layer.mlp_up),
                layer.mlp_down)
            x = x + mlp
            if trace:
                print("  layer %d out max %.6g" % (layer.layer_id,
                      float(x.abs().max())))
        h = self._rms(x, self.final_norm)
        logits = F.linear(h.float(), self.head)
        return logits, cache

    @property
    def device(self):
        return self.embed.device

    def parameters(self):
        return iter(())

    def eval(self):
        return self

    def to(self, device):
        # CPU-only oracle; the capture scripts call .to() unconditionally.
        if str(device) != "cpu":
            raise SystemExit("flint_oracle runs on CPU only")
        return self

    def generate(self, ids, do_sample=False, max_new_tokens=64, **kwargs):
        assert not do_sample, "the oracle implements greedy decoding only"
        ids = ids[0].tolist() if ids.dim() == 2 else ids.tolist()
        out = list(ids)
        cache = None
        cur = ids
        for _ in range(max_new_tokens):
            logits, cache = self.forward(torch.tensor(cur, dtype=torch.long), cache)
            nxt = int(torch.argmax(logits[-1]))
            out.append(nxt)
            if nxt in self.eos:
                break
            cur = [nxt]
        return torch.tensor([out], dtype=torch.long)


# ── capture-script entry point ───────────────────────────────────────────────

def load(model_dir, device_map=None, dtype=torch.bfloat16):
    return FlintLM(model_dir, dtype=dtype)


def _frame(model_dir, text):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    msgs = [{"role": "user", "content": text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _logits_report(model, ids):
    logits, _ = model.forward(torch.tensor(ids, dtype=torch.long))
    row = logits[-1]
    top = torch.topk(row, 5)
    return {
        "first_step_top5": [[int(i), float(v)] for v, i in zip(top.values, top.indices)],
        "argmax": int(top.indices[0]),
        "logit_sum": float(row.double().sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--frame", action="store_true",
                    help="apply the checkpoint's chat template to the prompt")
    ap.add_argument("--max-new", type=int, default=0)
    ap.add_argument("--ids", default=None,
                    help="comma-separated token ids (skips tokenization)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--dump-logits", default=None,
                    help="write the first-step top-5 logits to this JSON file")
    args = ap.parse_args()

    model = load(args.model, dtype=getattr(torch, args.dtype))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x]
    else:
        text = _frame(args.model, args.prompt) if args.frame else args.prompt
        ids = tok.encode(text, add_special_tokens=False)
    print("prompt_ids:", ids)
    if args.dump_logits:
        with open(args.dump_logits, "w", encoding="utf-8") as fh:
            json.dump(_logits_report(model, ids), fh, indent=1)
    if args.max_new > 0:
        out = model.generate(torch.tensor([ids]), max_new_tokens=args.max_new)
        gen = out[0][len(ids):].tolist()
        print("gen_ids:", gen)
        print("gen_text:", repr(tok.decode(gen, skip_special_tokens=False)))


if __name__ == "__main__":
    main()
