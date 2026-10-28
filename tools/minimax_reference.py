#!/usr/bin/env python3
"""MiniMax-M3 host reference over the SOURCE safetensors (external oracle).

The 428B/796 GB checkpoint cannot be loaded by transformers on this card, so
this script re-implements the text backbone from `modeling_minimax_m3_vl.py`
directly over the source BF16 tensors, loading ONLY the dense graph and the
experts the router actually selects. That keeps the working set at ~25 GB of
host RAM and makes the reference runnable, which is what turns "the engine
answers something" into "the engine answers the model's answer".

What it implements (and therefore what it can catch):
  * Gemma-form (1+w) RMSNorm, at full row and per head
  * GQA with per-head QK norm + partial RoPE on the first rotary_dim dims
  * GPT-OSS-style SwiGLU-OAI with the alpha/limit clamps
  * sigmoid router + e_score_correction_bias, renormalized top-k, routed
    scale, ungated shared expert
  * MSA block selection (engaged only past topk_blocks*block_size keys)

Usage:
    tools/minimax_reference.py --model <dir> --prompt "The capital of France is" \\
        --max-new 16 [--thinking disabled] [--json out.json]

Prints the prompt ids and the greedy continuation. Compare against the
engine's `gen_tokens` (e.g. the selfcheck log line).
"""
import argparse
import json
import os
import sys

import torch


def _load_index(model_dir):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as fh:
        return json.load(fh)["weight_map"]


class SourceModel:
    def __init__(self, model_dir, config):
        from safetensors import safe_open
        self.dir = model_dir
        self.cfg = config
        self.index = _load_index(model_dir)
        self._files = {}
        self._cache = {}
        self.NL = config["num_hidden_layers"]
        self.I = config["intermediate_size"]
        self.SI = config["shared_intermediate_size"]
        self.DI = config["dense_intermediate_size"]
        self.E = config["num_local_experts"]
        self.K = config["num_experts_per_tok"]
        self.moe_freq = config["moe_layer_freq"]
        self.sparse_freq = config["sparse_attention_config"][
            "sparse_attention_freq"]

    def _file(self, shard):
        from safetensors import safe_open
        if shard not in self._files:
            self._files[shard] = safe_open(
                os.path.join(self.dir, shard), framework="pt", device="cpu")
        return self._files[shard]

    def get(self, name):
        if name in self._cache:
            return self._cache[name]
        shard = self.index[name]
        t = self._file(shard).get_tensor(name).to(torch.float32)
        if t.ndim == 2:
            # Row-major [N,K] weights: y = x @ W.T
            t = t.contiguous()
        self._cache[name] = t
        return t

    def drop_experts(self):
        for k in [k for k in self._cache if ".experts." in k]:
            del self._cache[k]


def rms_norm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * (
        1.0 + w.float())


def head_norm(x, w, eps):
    # x: [heads, head_dim]
    return rms_norm(x, w, eps)


def swiglu_oai(gate, up, alpha, limit):
    g = torch.clamp(gate, max=limit)
    u = torch.clamp(up, min=-limit, max=limit)
    return (u + 1.0) * g * torch.sigmoid(alpha * g)


def rope(x, rotary_dim, theta, pos):
    # x: [head_dim] (one head); rotates dims [0, rotary_dim) with the HF
    # rotate_half pairing (j, j + rotary_dim/2).
    half = rotary_dim // 2
    out = x.clone()
    j = torch.arange(half, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (j / half))
    ang = pos * inv_freq
    c, s = torch.cos(ang), torch.sin(ang)
    x1 = x[:half]
    x2 = x[half:rotary_dim]
    out[:half] = x1 * c - x2 * s
    out[half:rotary_dim] = x1 * s + x2 * c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--thinking", default="disabled")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump-all", default=None,
                    help="write the first 16 hidden values after EVERY layer "
                         "(last prompt token) for kernel bisection")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open(os.path.join(args.model, "config.json")) as fh:
        root = json.load(fh)
    cfg = root["text_config"]

    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, thinking_mode=args.thinking)
    ids = tok(text, add_special_tokens=False)["input_ids"]

    sm = SourceModel(args.model, cfg)
    H = cfg["hidden_size"]
    NH = cfg["num_attention_heads"]
    NKV = cfg["num_key_value_heads"]
    HD = cfg["head_dim"]
    rdim = cfg["rotary_dim"]
    theta = cfg["rope_theta"]
    eps = cfg["rms_norm_eps"]
    alpha = cfg["swiglu_alpha"]
    limit = cfg["swiglu_limit"]
    rscale = cfg["routed_scaling_factor"]
    sa = cfg["sparse_attention_config"]
    block = sa["sparse_block_size"]
    topk_blocks = sa["sparse_topk_blocks"]
    local_blocks = sa["sparse_local_block"]
    scale = HD ** -0.5

    LAST_POS = [len(ids) - 1]
    embed = sm.get("language_model.model.embed_tokens.weight")
    head = sm.get("language_model.lm_head.weight")
    final_norm = sm.get("language_model.model.norm.weight")

    # KV cache per layer (plain python lists of tensors).
    kcache = [[] for _ in range(sm.NL)]
    vcache = [[] for _ in range(sm.NL)]
    layer_dump = {}

    def step(token, pos):
        nonlocal layer_dump
        h = embed[token].clone()
        for l in range(sm.NL):
            p = f"language_model.model.layers.{l}."
            hn = rms_norm(h, sm.get(p + "input_layernorm.weight"), eps)
            q = (hn @ sm.get(p + "self_attn.q_proj.weight").T).view(NH, HD)
            k = (hn @ sm.get(p + "self_attn.k_proj.weight").T).view(NKV, HD)
            v = (hn @ sm.get(p + "self_attn.v_proj.weight").T).view(NKV, HD)
            qn = sm.get(p + "self_attn.q_norm.weight")
            kn = sm.get(p + "self_attn.k_norm.weight")
            q = torch.stack([head_norm(q[i], qn, eps) for i in range(NH)])
            k = torch.stack([head_norm(k[i], kn, eps) for i in range(NKV)])
            hn_dbg = [round(float(x), 5) for x in hn[:6]]
            q = torch.stack([rope(q[i], rdim, theta, pos) for i in range(NH)])
            k = torch.stack([rope(k[i], rdim, theta, pos) for i in range(NKV)])
            kcache[l].append(k)
            vcache[l].append(v)
            keys = torch.stack(kcache[l])          # [kv, NKV, HD]
            vals = torch.stack(vcache[l])
            rep = NH // NKV
            keys_r = keys.repeat_interleave(rep, dim=1)  # [kv, NH, HD]
            vals_r = vals.repeat_interleave(rep, dim=1)
            scores = torch.einsum("hd,thd->ht", q, keys_r) * scale
            if sm.sparse_freq[l] and len(keys) > topk_blocks * block:
                # MSA: indexer block selection, identical to the engine.
                iq = (hn @ sm.get(p + "self_attn.index_q_proj.weight").T).view(
                    sa["sparse_num_index_heads"], sa["sparse_index_dim"])
                ik = (hn @ sm.get(p + "self_attn.index_k_proj.weight").T).view(
                    1, sa["sparse_index_dim"])
                iq_n = sm.get(p + "self_attn.index_q_norm.weight")
                ik_n = sm.get(p + "self_attn.index_k_norm.weight")
                iq = torch.stack([head_norm(iq[i], iq_n, eps)
                                  for i in range(iq.shape[0])])
                ik = torch.stack([head_norm(ik[i], ik_n, eps)
                                  for i in range(ik.shape[0])])
                # indexer key history per layer
                if not hasattr(step, "ik_hist"):
                    step.ik_hist = [[] for _ in range(sm.NL)]
                step.ik_hist[l].append(ik[0])
                ik_all = torch.stack(step.ik_hist[l])       # [kv, dim]
                idx_scores = torch.einsum("hd,kd->hk", iq, ik_all)  # [IH, kv]
                nb = (len(keys) + block - 1) // block
                pooled = torch.full((iq.shape[0], nb), float("-inf"))
                for b in range(nb):
                    lo, hi = b * block, min((b + 1) * block, len(keys))
                    pooled[:, b] = idx_scores[:, lo:hi].amax(dim=-1)
                qb = pos // block
                for i in range(local_blocks):
                    pooled[:, max(qb - i, 0)] = float("inf")
                keep = torch.topk(pooled, min(topk_blocks, nb), dim=-1).indices
                mask = torch.zeros(nb, dtype=torch.bool)
                mask[keep.unique()] = True
                for b in range(nb):
                    if not mask[b]:
                        scores[:, b * block:min((b + 1) * block, len(keys))] = \
                            float("-inf")
            probs = torch.softmax(scores, dim=-1)
            attn = torch.einsum("ht,thd->hd", probs, vals_r)
            attn = attn.reshape(NH * HD)
            attn_out = attn @ sm.get(p + "self_attn.o_proj.weight").T
            if os.environ.get("MM_DEBUG") and l == 0 and pos == LAST_POS[0]:
                print("REF input_norm", hn_dbg)
                print("REF attn", [round(float(x), 5) for x in attn_out[:6]])
            h = h + attn_out
            hn = rms_norm(h, sm.get(p + "post_attention_layernorm.weight"), eps)
            if sm.moe_freq[l] == 0:
                g = hn @ sm.get(p + "mlp.gate_proj.weight").T
                u = hn @ sm.get(p + "mlp.up_proj.weight").T
                d = sm.get(p + "mlp.down_proj.weight")
                mlp_out = swiglu_oai(g, u, alpha, limit) @ d.T
                if os.environ.get("MM_DEBUG") and l == 0 and pos == LAST_POS[0]:
                    print("REF mlp", [round(float(x), 5) for x in mlp_out[:6]])
                h = h + mlp_out
            else:
                gate_w = sm.get(p + "block_sparse_moe.gate.weight")  # [E,H]
                bias = sm.get(p + "block_sparse_moe.e_score_correction_bias")
                logits = hn @ gate_w.T
                scores_r = torch.sigmoid(logits.float())
                choice = scores_r + bias
                sel = torch.topk(choice, sm.K).indices.sort().values
                w = scores_r[sel]
                w = w / w.sum()
                routed = torch.zeros(H)
                for kk, e in enumerate(sel.tolist()):
                    ew = sm.get(p + "block_sparse_moe.experts.%d.w1.weight" % e)
                    uw = sm.get(p + "block_sparse_moe.experts.%d.w3.weight" % e)
                    dw = sm.get(p + "block_sparse_moe.experts.%d.w2.weight" % e)
                    g = ew @ hn
                    u = uw @ hn
                    routed = routed + w[kk] * (swiglu_oai(g, u, alpha, limit)
                                               @ dw.T)
                sg = hn @ sm.get(
                    p + "block_sparse_moe.shared_experts.gate_proj.weight").T
                su = hn @ sm.get(
                    p + "block_sparse_moe.shared_experts.up_proj.weight").T
                sd = sm.get(
                    p + "block_sparse_moe.shared_experts.down_proj.weight")
                shared = swiglu_oai(sg, su, alpha, limit) @ sd.T
                h = h + (routed * rscale + shared)
                sm.drop_experts()
            if pos == LAST_POS[0]:
                layer_dump[l] = [round(float(x), 6) for x in h]
        hn = rms_norm(h, final_norm, eps)
        logits = head @ hn
        return logits

    with torch.no_grad():
        logits = None
        for i, t in enumerate(ids):
            logits = step(t, i)
        pos = len(ids)
        print("prompt_ids:", ids)
        print("first logits top5:", torch.topk(logits, 5).indices.tolist(),
              [round(float(x), 4) for x in torch.topk(logits, 5).values])
        gen = []
        for _ in range(args.max_new):
            nxt = int(torch.argmax(logits))
            gen.append(nxt)
            logits = step(nxt, pos)
            pos += 1
            if nxt == 200020:
                break
        print("gen_tokens:", gen)
        print("text:", tok.decode(gen, skip_special_tokens=False))
        out = {"prompt_ids": ids, "gen_tokens": gen,
               "final_logits_top5": torch.topk(logits, 5).indices.tolist()}
        if args.dump_all:
            with open(args.dump_all, "w") as fh:
                for l in sorted(layer_dump):
                    fh.write("layer %d " % l +
                             " ".join("%.6f" % x for x in layer_dump[l]) + "\n")
            print("wrote", args.dump_all)
        if args.json:
            with open(args.json, "w") as fh:
                json.dump(out, fh)
            print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
