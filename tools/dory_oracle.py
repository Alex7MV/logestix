#!/usr/bin/env python3
"""Source reference for NVIDIA-Nemotron-Labs-Dory1.0-4B (nGPT looped LM).

transformers has no `DoryForCausalLM` (installed 5.15.1), so the checkpoint
cannot be built with AutoModelForCausalLM. This module reimplements the
forward pass EXACTLY as the vendor reference does (vllm-project/vllm#60149,
`vllm/model_executor/models/dory.py`), loads the BF16 safetensors shards into
it, and exposes a `generate()` compatible with the capture tools
(`tools/{fsi,code,det}_capture.py`).

Architecture (config.json + reference forward):
  * 72 physical "layers", each ONE mixer (attention OR mlp) with its own
    nGPT residual: no RMSNorm anywhere, the residual stream is L2-normalized
    by `justnorm` after every mixer.
  * layer_types[layer] in {sliding_attention, full_attention, mlp}; the stack
    is partitioned into input (0..n_in-1), recurrent (n_in..n_in+n_rec-1) and
    output blocks.
  * the recurrent block is applied `n_recurrent_loops` times, re-injecting the
    input block's output: state = justnorm(state + input_h) before every loop,
    state = justnorm(h) after it (latent recurrence).
  * attention: GQA, RoPE by layer type (sliding theta vs full theta), q scaled
    by (sqk * sqrt(head_dim)/initializer_range)^2 AFTER a per-head justnorm,
    k plain per-head justnorm, output gate sigmoid(g_proj * sg) with
    sg = (s_gate * sqrt(hidden_size^0.5)/initializer_range)^2, then a
    justnorm over the full head stack before o_proj.
  * mlp: SwiGLU with the suv_* per-dimension scales and a justnorm before
    down_proj.
  * logits: h @ lm_head^T, then scaled per vocabulary entry by
    lm_head.logit_scale / initializer_range. No final norm.

The numerics mirror the reference: weights stay BF16, `justnorm`/slerp/softmax
and the logit scale accumulate in FP32, and the softmax output is cast back to
the activation dtype.
"""

import lsx_paths
import glob
import json
import math
import os

import torch

EPS = 1e-6


def justnorm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """NGPT normalization: x * rsqrt(sum(x^2) + eps) in FP32."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).sum(dim=dim, keepdim=True) + EPS)
    return y.to(x.dtype)


def slerp_alpha(a, b, alpha, alpha_scale):
    """nGPT residual: normalize both, move by |alpha * alpha_scale|, renormalize."""
    ya, yb = justnorm(a).float(), justnorm(b).float()
    lr = (alpha.float() * alpha_scale).abs()
    return justnorm(ya + lr * (yb - ya)).to(a.dtype)


def rope_neox(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotate-half (NeoX) RoPE over the full head dimension."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32,
                                       device=x.device) / d))
    freqs = torch.outer(pos.float(), inv)
    cos = freqs.cos()[:, None, :]
    sin = freqs.sin()[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class DoryConfig:
    def __init__(self, raw):
        self.raw = raw
        self.vocab_size = raw["vocab_size"]
        self.hidden_size = raw["hidden_size"]
        self.intermediate_size = raw["intermediate_size"]
        self.num_layers = raw["num_hidden_layers"]
        self.num_heads = raw["num_attention_heads"]
        self.num_kv_heads = raw["num_key_value_heads"]
        self.head_dim = raw.get("head_dim") or (
            self.hidden_size // self.num_heads)
        self.max_position_embeddings = raw.get("max_position_embeddings", 8192)
        self.initializer_range = raw.get("initializer_range", 0.02)
        self.sliding_window = raw.get("sliding_window")
        self.layer_types = raw["layer_types"]
        self.rope = raw["rope_parameters"]
        self.ngpt_alpha_init = raw.get("ngpt_alpha_init", 0.05)
        self.gate_init = raw.get("attention_output_gate_logit_scale_init")
        self.n_input = raw.get("n_input_layers", 0)
        self.n_recurrent = raw.get(
            "n_recurrent_layers",
            self.num_layers - self.n_input - raw.get("n_output_layers", 0))
        self.n_output = raw.get("n_output_layers", 0)
        self.n_loops = raw.get("n_recurrent_loops", 1)
        self.kv_mode = raw.get("recurrent_kv_cache_mode", "per_loop")
        assert len(self.layer_types) == self.num_layers
        assert self.n_input + self.n_recurrent + self.n_output \
            == self.num_layers

    def is_attention(self, layer: int) -> bool:
        return self.layer_types[layer] != "mlp"

    def rope_theta(self, layer: int) -> float:
        kind = self.layer_types[layer]
        return float(self.rope[kind]["rope_theta"])

    def window(self, layer: int):
        if self.layer_types[layer] == "sliding_attention":
            return self.sliding_window
        return None


class DoryReference:
    """BF16 reference forward + greedy/sampled generation."""

    def __init__(self, model_path: str, device: str = "cpu"):
        from safetensors.torch import load_file

        with open(os.path.join(model_path, "config.json"),
                  encoding="utf-8") as fh:
            self.cfg = DoryConfig(json.load(fh))
        self.device = torch.device(device)
        self.w = {}
        for shard in sorted(glob.glob(os.path.join(model_path,
                                                   "*.safetensors"))):
            for key, tensor in load_file(shard).items():
                self.w[key] = tensor.to(self.device)
        cfg = self.cfg
        self.std = cfg.initializer_range
        # nGPT scale factors (reference constants, NOT re-derived).
        self.sqk_factor = math.sqrt(cfg.head_dim) / self.std
        g0 = cfg.gate_init
        if g0 is None:
            g0 = math.sqrt(cfg.hidden_size)
        self.sgate_factor = math.sqrt(g0) / self.std
        self.suv_factor = math.sqrt(cfg.hidden_size)
        self.alpha_scale = cfg.ngpt_alpha_init / self.std
        self.attn_scale = 1.0 / math.sqrt(cfg.head_dim)
        self._cache = None

    # ── torch-module compatibility for the capture tools ───────────────────
    def to(self, device):
        """No-op: the weights are placed by `load()`; the capture tools call
        `.to()` unconditionally on whatever loader returned."""
        return self

    def eval(self):
        return self

    # ── weights ────────────────────────────────────────────────────────────
    def p(self, name):
        return self.w[name]

    def parameters(self):
        return iter(self.w.values())

    # ── pieces ─────────────────────────────────────────────────────────────
    def _attention(self, layer, h, pos, cache, key):
        cfg = self.cfg
        base = f"backbone.layers.{layer}."
        nh, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        t = h.shape[0]
        q = h @ self.p(base + "mixer.q_proj.weight").t()
        k = h @ self.p(base + "mixer.k_proj.weight").t()
        v = h @ self.p(base + "mixer.v_proj.weight").t()
        gate = h @ self.p(base + "mixer.g_proj.weight").t()

        v = v.view(t, nkv, hd)

        q = rope_neox(q.view(t, nh, hd), pos, cfg.rope_theta(layer))
        k = rope_neox(k.view(t, nkv, hd), pos, cfg.rope_theta(layer))

        sqk = (self.p(base + "mixer.sqk").float() * self.sqk_factor)
        sqk = sqk.view(1, nh, hd)
        q = (sqk.square() * justnorm(q).float()).to(q.dtype)
        k = justnorm(k)

        if key in cache:
            pk, pv = cache[key]
            kk = torch.cat([pk, k], dim=0)
            vv = torch.cat([pv, v], dim=0)
        else:
            kk, vv = k, v
        cache[key] = (kk, vv)

        # GQA attention, FP32 softmax, output cast back to the activation dtype.
        rep = nh // nkv
        total = kk.shape[0]                                     # cached positions
        kx = kk.repeat_interleave(rep, dim=1).permute(1, 0, 2)  # [nh,pos,hd]
        vx = vv.repeat_interleave(rep, dim=1).permute(1, 0, 2)
        qf = q.float()
        scores = torch.einsum("thd,hpd->htp", qf, kx.float()) * self.attn_scale
        key_pos = torch.arange(total, device=q.device)
        qpos = pos.reshape(-1)[:, None]
        mask = key_pos[None, :] <= qpos
        window = cfg.window(layer)
        if window is not None:
            mask &= key_pos[None, :] > qpos - window
        scores = scores.masked_fill(~mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("htp,hpd->thd", probs, vx.float())
        out = out.to(v.dtype)                          # [t, nh, hd]

        sg = (self.p(base + "mixer.s_gate").float() * self.sgate_factor)
        sg = sg.square().view(1, nh * hd)
        out = (out.reshape(t, nh * hd).float()
               * torch.sigmoid(gate.float() * sg)).to(out.dtype)
        out = justnorm(out)
        return out @ self.p(base + "mixer.o_proj.weight").t()

    def _mlp(self, layer, h):
        base = f"backbone.layers.{layer}."
        g = h @ self.p(base + "mixer.gate_proj.weight").t()
        u = h @ self.p(base + "mixer.up_proj.weight").t()
        g = (g.float() * (self.p(base + "mixer.suv_gate").float()
                          * self.suv_factor)).to(g.dtype)
        u = (u.float() * (self.p(base + "mixer.suv_up").float()
                          * self.suv_factor)).to(u.dtype)
        m = (torch.nn.functional.silu(g.float()) * u.float()).to(g.dtype)
        m = justnorm(m)
        return m @ self.p(base + "mixer.down_proj.weight").t()

    def _layer(self, layer, h, pos, cache, loop):
        cfg = self.cfg
        base = f"backbone.layers.{layer}."
        if cfg.is_attention(layer):
            alpha = self.p(base + "attn_alpha")
            out = self._attention(layer, h, pos, cache, (layer, loop))
        else:
            alpha = self.p(base + "mlp_alpha")
            out = self._mlp(layer, h)
        return slerp_alpha(h, out, alpha, self.alpha_scale)

    # ── forward ────────────────────────────────────────────────────────────
    def forward(self, input_ids, cache=None, pos0=0):
        cfg = self.cfg
        ids = torch.as_tensor(input_ids, device=self.device,
                              dtype=torch.long).reshape(-1)
        pos = torch.arange(pos0, pos0 + ids.shape[0], device=self.device,
                           dtype=torch.long)
        h = self.p("backbone.embeddings.weight")[ids]
        if cache is None:
            cache = {}

        for layer in range(cfg.n_input):
            h = self._layer(layer, h, pos, cache, 0)
        input_h = h

        state = input_h
        for loop in range(cfg.n_loops):
            h = justnorm(state + input_h)
            for layer in range(cfg.n_input, cfg.n_input + cfg.n_recurrent):
                h = self._layer(layer, h, pos, cache, loop)
            state = justnorm(h)

        h = state
        for layer in range(cfg.n_input + cfg.n_recurrent, cfg.num_layers):
            h = self._layer(layer, h, pos, cache, 0)
        return h

    def logits(self, h):
        scale = self.p("lm_head.logit_scale").float() / self.std
        logits = h @ self.p("lm_head.weight").t()
        return logits.float() * scale

    # ── generation ─────────────────────────────────────────────────────────
    def reset_cache(self):
        self._cache = {}

    @torch.no_grad()
    def generate(self, input_ids, do_sample=False, max_new_tokens=64,
                 temperature=1.0, top_p=0.95, top_k=0, eos_token_id=11,
                 **kwargs):
        if not do_sample:
            temperature = 0.0
        ids = torch.as_tensor(input_ids).reshape(1, -1)
        self.reset_cache()
        pos = 0
        h = self.forward(ids[0].tolist(), self._cache, pos)
        pos += ids.shape[1]
        out = [ids[0].tolist()]
        eos = eos_token_id if isinstance(eos_token_id, (list, tuple)) \
            else [eos_token_id]
        eos = [e for e in eos if e is not None]
        for _ in range(max_new_tokens):
            logits = self.logits(h[-1:])[0]
            if temperature <= 0:
                nxt = int(torch.argmax(logits).item())
            else:
                scaled = logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                if top_k and top_k > 0:
                    kth = torch.topk(probs, top_k).values[-1]
                    probs = torch.where(probs < kth,
                                        torch.zeros_like(probs), probs)
                if top_p and top_p < 1.0:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    cut = cum - sp > top_p
                    sp = torch.where(cut, torch.zeros_like(sp), sp)
                    probs = torch.zeros_like(probs).scatter(0, si, sp)
                probs = probs / probs.sum()
                nxt = int(torch.multinomial(probs, 1).item())
            out[0].append(nxt)
            if nxt in eos:
                break
            h = self.forward([nxt], self._cache, pos)
            pos += 1
        return torch.tensor(out, dtype=torch.long)


def load(model_path: str, device_map=None):
    """Load the checkpoint as a reference LM (GPU when it fits)."""
    device = "cpu"
    if device_map in (None, "auto") and torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        need = 0
        for shard in glob.glob(os.path.join(model_path, "*.safetensors")):
            need += os.path.getsize(shard)
        if need < int(free * 0.85):
            device = "cuda"
    return DoryReference(model_path, device)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    # org-nested store: the registry knows the org
                    default=str(lsx_paths.source_dir("dory-1.0-4b")))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--raw", action="store_true",
                    help="no chat framing (raw continuation)")
    ap.add_argument("--thinking", dest="thinking", action="store_true",
                    default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    args = ap.parse_args()

    from transformers import PreTrainedTokenizerFast
    # AutoTokenizer cannot build this checkpoint: `model_type: dory` has no
    # transformers config class, and the config-validation path raises before the
    # tokenizer is touched. The tokenizer itself is a plain byte-level BPE.
    tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(args.model, "tokenizer.json"),
        bos_token="<s>", eos_token="<|im_end|>", pad_token="<|im_end|>",
        unk_token="<unk>")
    if not getattr(tok, "chat_template", None):
        jinja = os.path.join(args.model, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja, encoding="utf-8") as fh:
                tok.chat_template = fh.read()
    if args.raw:
        text = args.prompt
    else:
        text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=args.thinking)
    ids = tok(text, return_tensors="pt").input_ids[0].tolist()
    lm = DoryReference(args.model, args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", lm.device, "prompt ids:", len(ids))
    out = lm.generate(ids, max_new_tokens=args.max_new_tokens)
    gen = out[0][len(ids):].tolist()
    print("gen ids:", gen)
    print("gen text:", repr(tok.decode(gen, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
