#!/usr/bin/env python3
"""Hy4-flash EXPLORATORY numpy oracle — superseded by tools/hy4_flash_reference.py.

STATUS: exploratory, NOT the reference of record. Stage 0 of this ingestion
concluded that no published implementation of the architecture existed and wrote
this file to have something to compare against. That conclusion was WRONG: the
checkpoint's own README pins the implementation ("Support for Hy4 flash will be
merged into the transformers main branch soon. Until then, install from the
commit below") and huggingface/transformers PR #50903 IS that implementation.
`tools/hy4_flash_reference.py` runs it, and THAT is the reference the gates are
captured from.

This file is kept because it is cheap to read and it documents, function by
function, the semantics an independent implementation has to get right — but it
disagrees with the engine on the real weights and the disagreement is not yet
explained (docs/OPEN_TASKS.md, hy4-flash row). Do not capture a golden from it.

The original docstring follows.

Hy4-flash CPU reference oracle — one-token forward on the REAL weights.

No published implementation of this architecture exists (transformers ships
`hy_v4`, not `hy_v4_flash`: Engram, the 64-head sliding layers, LatentMoE and
the compressed indexer are all absent), so this file IS the reference of record
for the engine's parity gates and for the FSI/CODE/DET goldens. It is written
import lsx_paths
from the checkpoint's own tensor names, its config.json and its model card, and
every semantic decision it makes is named in a comment with the evidence.

Design rules, all of which exist to keep it honest:

  * it reads the checkpoint's safetensors directly (mmap, per-tensor), never a
    converted artifact — a golden captured from the engine would only test the
    engine against itself;
  * fp32 throughout, so a disagreement with the engine is a semantic difference
    and not a rounding one;
  * one token at a time, so a 411 GB tree costs ~17 GB of reads per step;
  * it prints per-layer diagnostics next to the logits, so a divergence can be
    LOCALISED instead of guessed at.

    python3 tools/hy4_flash_oracle.py --model <source root>/tencent/Hy4-flash \
        --prompt "The capital of France is" --tokens 4
"""
import argparse
import json
import math
import os
import struct
import sys

import numpy as np


# ── checkpoint reader (mmap, one tensor at a time) ─────────────────────────

class Shards:
    """Reads tensors out of a sharded safetensors tree.

    Keeps one open mmap per file, so a step reads only the tensors it touches
    instead of 411 GB.
    """

    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self.headers = {}
        self.files = {}

    def _header(self, name):
        fn = self.map[name]
        if fn not in self.headers:
            path = os.path.join(self.root, fn)
            fh = open(path, "rb")
            n = struct.unpack("<Q", fh.read(8))[0]
            self.headers[fn] = (json.loads(fh.read(n)), 8 + n)
            self.files[fn] = fh
            fh.seek(0)
        return fn, self.headers[fn]

    def info(self, name):
        fn, (hdr, _off) = self._header(name)
        e = hdr[name]
        return e["dtype"], e["shape"]

    def get_rows(self, name, row, nrows=1, dtype=np.float32):
        """Reads `nrows` rows of a 2-D or 3-D tensor without materializing the rest.

        The routed-expert tensors are 3-D and huge (gate_up is 4.8 GB per layer),
        and a decode step touches 16 of the 512 experts, so an unsliced read would
        pull 4.8 GB per layer per token.
        """
        fn, (hdr, data_off) = self._header(name)
        e = hdr[name]
        sv = e["dtype"]
        width = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1,
                 "F8_E4M3": 1}[sv]
        shape = e["shape"]
        inner = int(np.prod(shape[1:]))
        row_bytes = inner * width
        s, _eo = e["data_offsets"]
        fh = self.files[fn]
        fh.seek(data_off + s + row * row_bytes)
        raw = fh.read(nrows * row_bytes)
        if sv == "BF16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16)
            arr = arr.view(np.float32)
        elif sv == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif sv == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            arr = np.frombuffer(raw, dtype=np.int32)
        return np.asarray(arr, dtype=dtype).reshape((nrows,) + tuple(shape[1:]))

    def get(self, name, dtype=np.float32):
        fn, (hdr, data_off) = self._header(name)
        e = hdr[name]
        s, eo = e["data_offsets"]
        nbytes = eo - s
        sv = e["dtype"]
        width = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1,
                 "F8_E4M3": 1}[sv]
        numel = nbytes // width
        fh = self.files[fn]
        fh.seek(data_off + s)
        raw = fh.read(nbytes)
        if sv == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            arr = arr.view(np.float32)
        elif sv == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif sv == "F32":
            arr = np.frombuffer(raw, dtype=np.float32)
        elif sv == "I32":
            arr = np.frombuffer(raw, dtype=np.int32)
        else:
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return np.asarray(arr, dtype=dtype).reshape(e["shape"])


# ── the model ──────────────────────────────────────────────────────────────

def rms_norm_unweighted(x, eps):
    x = x.astype(np.float32)
    return x * (1.0 / np.sqrt(np.mean(x * x) + eps))


def rms_norm(x, w, eps):
    x = x.astype(np.float32)
    return w * (x * (1.0 / np.sqrt(np.mean(x * x) + eps)))


def rms_inv(x, eps):
    """Unweighted RMSNorm that returns the INVERSE (iHC's `input_norm`)."""
    x = x.astype(np.float32)
    return 1.0 / np.sqrt(np.mean(x * x) + eps)


def silu(x):
    return x / (1.0 + np.exp(-x))


def swiglu_clamp(gate, up, limit):
    """The reference clamps BEFORE the activation; the gate has no lower bound."""
    g = np.minimum(gate, limit)
    u = np.clip(up, -limit, limit)
    return silu(g) * u


def rope_tables(rope_dim, theta, n):
    half = rope_dim // 2
    inv = theta ** (-2.0 * np.arange(half) / rope_dim)
    pos = np.arange(n, dtype=np.float64)[:, None]
    ang = pos * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rotate_half_interleaved(x):
    """The reference `rotate_half`: pairs (2j, 2j+1), i.e. stack((-x_odd, x_even)).

    `apply_rotary_pos_emb` then broadcasts `cos` as `repeat_interleave(2)` of the
    first half, so both lanes of a pair share one frequency — which makes the
    rotation the standard pair rotation, NOT the half-split rotate_half of
    Llama/HF-default. Getting this wrong rotates every head into a different
    subspace and the model still emits fluent text.
    """
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return np.stack((-odd, even), axis=-1).reshape(x.shape)


class Hy4Flash:
    """One-token (decode) reference. `verbose` prints per-layer diagnostics."""

    def __init__(self, root, verbose=False):
        self.sd = Shards(root)
        with open(os.path.join(root, "config.json")) as f:
            cfg = json.load(f)
        t = cfg["text_config"]
        self.cfg = t
        self.hidden = t["hidden_size"]
        self.layers = t["num_hidden_layers"]
        self.vocab = t["vocab_size"]
        self.heads_dsa = t["num_attention_heads"]
        self.heads_swa = t["swa_num_attention_heads"]
        self.qk_nope = t["qk_nope_head_dim"]
        self.qk_rope = t["qk_rope_head_dim"]
        self.v_head = t["v_head_dim"]
        self.qk_head = self.qk_nope + self.qk_rope
        self.q_lora = t["q_lora_rank"]
        self.kv_lora = t["kv_lora_rank"]
        self.moe_inter = t["moe_intermediate_size"]
        self.latent = t["routed_expert_hidden_size"]
        self.n_routed = t["n_routed_experts"]
        self.topk = t["num_experts_per_tok"]
        self.hc = t["hc_mult"]
        self.eps = t["rms_norm_eps"]
        self.hc_eps = t["hc_eps"]
        self.hc_mag = t["hc_magnitude"]
        self.route_scale = t["routed_scaling_factor"]
        self.swiglu_limit = t["swiglu_limit"]
        self.window = t["sliding_window"]
        self.index_topk = t["index_topk"]
        self.layer_types = t["layer_types"]
        self.mlp_types = t["mlp_layer_types"]
        rp = t["rope_parameters"]
        self.theta = {"dsa": rp["deepseek_sparse_attention"]["rope_theta"],
                      "swa": rp["sliding_attention"]["rope_theta"]}
        self.verbose = verbose
        self.kv = None      # [layers][pos][kv_lora + rope] fp32
        self._cache = {}
        self.reset()

    def tbl(self, name):
        """Cached read of a large table that is reused every step."""
        if name not in self._cache:
            self._cache[name] = self.sd.get(name)
        return self._cache[name]

    # -- session ----------------------------------------------------------
    def reset(self):
        self.kv = np.zeros((self.layers, 0, self.kv_lora + self.qk_rope),
                           dtype=np.float32)
        self.rope = {}
        self._idx_keys = {}
        self._x_now = {}

    def ensure_kv(self, pos):
        """Grows the latent cache to hold `pos` (decode appends one row)."""
        if self.kv.shape[1] <= pos:
            grow = max(64, pos + 64)
            nk = np.zeros((self.layers, grow, self.kv_lora + self.qk_rope),
                          dtype=np.float32)
            nk[:, :self.kv.shape[1]] = self.kv
            self.kv = nk

    def is_dsa(self, l):
        return self.layer_types[l] == "deepseek_sparse_attention"

    def is_moe(self, l):
        return self.mlp_types[l] == "sparse"

    def heads(self, l):
        return self.heads_dsa if self.is_dsa(l) else self.heads_swa

    def rope_for(self, kind, n):
        if kind not in self.rope or self.rope[kind][0].shape[0] < n:
            self.rope[kind] = rope_tables(self.qk_rope, self.theta[kind],
                                          max(n, 4096))
        return self.rope[kind]

    # -- sublayers ---------------------------------------------------------
    def ihc(self, x, layer, which):
        """iHC: returns (agg, pre, post) for one sub-block.

        x is the [hc, hidden] stream stack. The reference computes
          flat   = x.flatten()                       [hc*hidden]
          mixes  = fn @ flat * rsqrt(mean(flat^2)+eps)
          pre_j  = sigmoid(mixes_j*scale[0] + base_j)            + eps
          post_j = magnitude*sigmoid(mixes_{hc+j}*scale[1]+base_{hc+j}) + eps
          agg    = sum_j pre_j * x_j
        Note there is NO sinkhorn and NO comb matrix in this model (the model card
        calls it "identity Hyper-Connections"); the checkpoint ships only
        hc_fn/hc_base/hc_scale.
        """
        stem = "model.layers.%d.%s.hc_pre." % (layer, which)
        fn = self.sd.get(stem + "hc_fn")
        base = self.sd.get(stem + "hc_base")
        scale = self.sd.get(stem + "hc_scale")
        flat = x.reshape(-1)
        rs = rms_inv(flat, self.eps)
        mixes = (fn @ flat) * rs
        pre = 1.0 / (1.0 + np.exp(-(mixes[:self.hc] * scale[0] +
                                   base[:self.hc]))) + self.hc_eps
        post = self.hc_mag * (1.0 / (1.0 + np.exp(-(
            mixes[self.hc:] * scale[1] + base[self.hc:])))) + self.hc_eps
        agg = np.einsum("j,ji->i", pre, x)
        return agg.astype(np.float32), pre.astype(np.float32), post.astype(np.float32)

    def attention(self, x, layer, pos):
        """MLA with a sigmoid gate, a learned per-head sink and (sparse layers)
        IndexPool selection.

        The score is computed in the LATENT space: k_nope = kv_b_nope @ latent,
        so q_nope . k_nope = (kv_b_nope^T q_nope) . latent. That is algebraically
        identical to expanding k and is what makes the cache 576 wide instead of
        heads*448.
        """
        heads = self.heads(layer)
        if not hasattr(self, "_x_now"):
            self._x_now = {}
        self._x_now[layer] = x
        stem = "model.layers.%d.self_attn." % layer
        is_dsa = self.is_dsa(layer)
        qr = rms_norm(self.sd.get(stem + "q_a_proj.weight") @ x,
                      self.sd.get(stem + "q_a_layernorm.weight"), self.eps)
        q = (self.sd.get(stem + "q_b_proj.weight") @ qr).reshape(heads, self.qk_head)
        kv = self.sd.get(stem + "kv_a_proj_with_mqa.weight") @ x
        kv_pass, k_rot = kv[:self.kv_lora], kv[self.kv_lora:]
        kv_pass = rms_norm(kv_pass, self.sd.get(stem + "kv_a_layernorm.weight"),
                           self.eps)
        cos, sin = self.rope_for("dsa" if is_dsa else "swa", pos + 2)
        c, s = cos[pos], sin[pos]

        def rope_row(v):
            # c, s are the 32 halved frequencies; repeat_interleave(2) broadcasts
            # them onto the 64 lanes and the rotation pairs (2j, 2j+1).
            cc = np.repeat(c, 2)
            ss = np.repeat(s, 2)
            return v * cc + rotate_half_interleaved(v) * ss

        q_rot = np.stack([rope_row(q[h, self.qk_nope:]) for h in range(heads)])
        k_rot = rope_row(k_rot)
        self.ensure_kv(pos)
        self.kv[layer, pos] = np.concatenate([kv_pass, k_rot])
        gate = self.sd.get(stem + "linear_gate.weight") @ x
        sink = self.sd.get(stem + "learnable_sink_param")

        # Visible positions.
        if is_dsa and pos + 1 > self.index_topk:
            vis = self.index_pool(layer, qr, x, pos, heads)
        elif is_dsa:
            vis = np.arange(0, pos + 1)
        else:
            vis = np.arange(max(0, pos - self.window + 1), pos + 1)

        lat = self.kv[layer, vis]                       # [n, kv+rope]
        w_nope = self.sd.get(stem + "kv_b_proj.weight")      # [(nope+v)*heads, kv]
        w_nope = w_nope.reshape(heads, self.qk_nope + self.v_head, self.kv_lora)
        qk = np.einsum("hnd,hr->hd", w_nope[:, :self.qk_nope], q[:, :self.qk_nope])
        scores = (qk @ lat[:, :self.kv_lora].T +
                  q_rot @ lat[:, self.kv_lora:].T) / math.sqrt(self.qk_head)
        # Sink: one extra logit per head, dropped after the softmax, so the
        # retained probability mass is < 1.
        allv = np.concatenate([scores, sink[:, None]], axis=1)
        allv = allv - allv.max(axis=1, keepdims=True)
        p = np.exp(allv)
        p /= p.sum(axis=1, keepdims=True)
        p = p[:, :-1]
        acc = p @ lat[:, :self.kv_lora]                  # [heads, kv_lora]
        v = np.einsum("hnd,hd->hn", w_nope[:, self.qk_nope:], acc)
        out = (v.reshape(-1) * (1.0 / (1.0 + np.exp(-gate))))
        out = self.sd.get(stem + "o_proj.weight") @ out
        out = rms_norm(out, self.sd.get(stem + "post_proj_norm.weight"), self.eps)
        return out.astype(np.float32), vis

    def index_pool(self, layer, qr, x, pos, heads):
        """IndexPool: pooled keys per 4-token group, top-k groups by score.

        Reconstructed from the checkpoint's tensor names and the config's
        hyperparameters (compress_ratio 4, group_topk 512, index_topk 2048,
        skip_fine_select). See docs/model-reports for the recall measurement.
        """
        stem = "model.layers.%d.self_attn.compressed_indexer." % layer
        ratio = self.cfg["compressed_indexer_compress_ratio"]
        dim = self.cfg["index_head_dim"]
        n_complete = (pos + 1) // ratio
        w_tok = self.sd.get(stem + "token_key_projection.weight")
        tok_norm = self.sd.get(stem + "token_key_norm.weight")
        w_kv = self.sd.get(stem + "key_value_projection.weight")
        w_gate = self.sd.get(stem + "key_gate_projection.weight")
        k_norm = self.sd.get(stem + "key_norm.weight")
        ape = self.sd.get(stem + "intra_group_ape")
        # Token keys are a function of this layer's block input at each position,
        # so they are cached per (layer, position) as the session advances.
        if not hasattr(self, "_idx_keys"):
            self._idx_keys = {}
        keys = self._idx_keys.setdefault(layer, {})
        keys[pos] = rms_norm(w_tok @ self._x_now[layer], tok_norm, self.eps)
        pooled = []
        for g in range(n_complete):
            ks = np.stack([keys[g * ratio + i] for i in range(ratio)])
            values = ks @ w_kv.T                      # [ratio, dim]
            gates = ks @ w_gate.T + ape               # [ratio, dim]
            gmax = gates.max(axis=0, keepdims=True)
            e = np.exp(gates - gmax)
            w = e / e.sum(axis=0, keepdims=True)      # softmax over the group
            pooled.append(rms_norm((values * w).sum(0), k_norm, self.eps))
        pooled = np.stack(pooled)
        q = (self.sd.get(stem + "query_projection.weight") @ qr).reshape(heads, dim)
        wh = self.sd.get(stem + "head_weight_projection.weight") @ x
        wh = wh * (heads ** -0.5) * (dim ** -0.5)
        sc = np.einsum("h,hd,gd->g", wh, q, pooled)
        sc = np.maximum(sc, 0.0)
        keep = min(self.cfg["compressed_indexer_group_topk"], n_complete)
        top = np.argpartition(-sc, keep - 1)[:keep]
        vis = np.sort(np.concatenate([top[:, None] * ratio +
                                      np.arange(ratio)[None, :]]).reshape(-1))
        tail = np.arange(n_complete * ratio, pos + 1)
        return np.unique(np.concatenate([vis, tail]))

    # -- Engram (the n-gram memory; eight columns of the model card) ----------
    def engram_init(self):
        """Per-layer constants for the four Engram memories.

        The hash MULTIPLIERS are not shipped as weights: the reference draws them
        from `np.random.default_rng(seed + 10007*layer_id)` bounded by the size of
        the compressed vocabulary, and the per-head vocabulary sizes come from
        `engram_layer_head_vocab_sizes`. So the memory IS readable without
        guessing — which is why this engine applies it.
        """
        t = self.cfg
        self.engram_ids = t["engram_layer_ids"]
        self.engram_max_ngram = t["engram_max_ngram_size"]
        self.engram_nheads = t["engram_n_head_per_ngram"]
        self.engram_nembed = t["engram_n_embed_per_ngram"]
        self.engram_pad = t["engram_pad_id"]
        seed = t["engram_seed"]
        self.engram_head_vocab = {
            int(k): [int(v2) for order in v for v2 in order]
            for k, v in t["engram_layer_head_vocab_sizes"].items()}
        self.engram_mult = {}
        for lid in self.engram_ids:
            table = self.sd.get("model.layers.%d.engram_layer.lookup_table"
                                % (lid - 1))
            size = int(table.shape[0])
            half_bound = max(1, (2 ** 63 - 1) // size // 2)
            rng = np.random.default_rng(seed + 10007 * int(lid))
            drawn = rng.integers(0, half_bound, size=self.engram_max_ngram,
                                 dtype=np.int64)
            self.engram_mult[int(lid)] = (drawn * 2 + 1)

    def engram(self, streams, layer, token, pos, ngram_tokens):
        """Returns the residual added to the [hc, hidden] stream stack."""
        lid = layer + 1
        if lid not in self.engram_ids:
            return 0.0
        stem = "model.layers.%d.engram_layer." % layer
        table = self.sd.get(stem + "lookup_table")
        tsize = int(table.shape[0])
        vocab = self.engram_head_vocab[lid]
        head_dim = self.engram_nembed // self.engram_nheads
        mult = self.engram_mult[lid]

        ng = np.asarray(ngram_tokens, dtype=np.int64)          # [max_ngram]
        compressed = table[np.clip(ng, 0, tsize - 1)].astype(np.int64)
        compressed = np.where(ng < tsize, compressed,
                              np.int64(table[self.engram_pad]))
        mixed = (compressed * mult).astype(np.int64)
        hash_ids = []
        for n in range(2, self.engram_max_ngram + 1):
            mix = mixed[0].copy()
            for off in range(1, n):
                mix = np.bitwise_xor(mix, mixed[off])
            start = (n - 2) * self.engram_nheads
            hash_ids.extend(int(mix % vocab[start + h])
                            for h in range(self.engram_nheads))
        # One flat table per layer, heads separated by an offset.
        emb_table = stem + "multi_head_embedding.embedding.weight"
        offsets = np.concatenate([[0], np.cumsum(vocab[:-1])])
        rows, keep = [], []
        for h, hid in enumerate(hash_ids):
            row = int(hid + offsets[h])
            rows.append(self.sd.get_rows(emb_table, row, 1)[0])
            keep.append(hid >= 0)
        emb = np.stack(rows).astype(np.float32)                # [16, 96]
        for h in range(self.engram_nheads):
            w = self.sd.get(stem + "lookup_per_head_norms.%d.weight" % h)
            emb[h] = rms_norm(emb[h], w, 1e-5)                 # nn.RMSNorm default
        flat = emb.reshape(-1)                                  # [1536]

        value = (self.sd.get(stem + "value_proj.weight") @ flat +
                 self.sd.get(stem + "value_proj.bias"))
        gamma = self.sd.get(stem + "unified_pre_gamma")
        main = gamma * rms_norm_unweighted(value, self.eps)
        branch = np.stack([self.sd.get(stem + "unified_per_hc_scales.%d" % i) * main
                           for i in range(self.hc)])
        out = np.zeros_like(streams)
        for i in range(self.hc):
            key = rms_norm(self.sd.get(stem + "key_projs.%d.weight" % i) @ flat +
                           self.sd.get(stem + "key_projs.%d.bias" % i),
                           self.sd.get(stem + "norm_key.%d.weight" % i), 1e-5)
            query = rms_norm(streams[i],
                             self.sd.get(stem + "norm_query.%d.weight" % i),
                             1e-5)
            dot = float((key * query).sum()) * self.hidden ** -0.5
            g = 1.0 / (1.0 + math.exp(-(math.copysign(1.0, dot) *
                                        math.sqrt(max(abs(dot), 1e-6)))))
            out[i] = g * main + branch[i]
        return out

    def moe(self, x, layer):
        """LatentMoE: experts read/write `latent`, then routed_expert_up_proj maps
        back to hidden; the shared expert reads the FULL-width block input.
        The routed sum is weighted in fp32 and the two branches are added before
        `combine_output_norm` (the "combined output" the name refers to).
        """
        stem = "model.layers.%d.mlp." % layer
        gate_w = self.sd.get(stem + "gate.weight")
        bias = self.sd.get("model.layers.%d.mlp.gate.e_score_correction_bias"
                           % layer)
        logits = gate_w @ x
        scores = 1.0 / (1.0 + np.exp(-logits))
        choice = scores + bias                     # selection only, never scales
        idx = np.argsort(-choice, kind="stable")[:self.topk]
        w = scores[idx]
        w = w / (w.sum() + 1e-20) * self.route_scale
        lat = self.sd.get(stem + "routed_expert_down_proj.weight") @ x
        acc = np.zeros(self.latent, dtype=np.float32)
        for k, e in enumerate(idx):
            # Read ONLY this expert's slice: the layer's gate_up tensor is 4.8 GB
            # and a step uses 16 of its 512 experts.
            gu = lat @ self.sd.get_rows(stem + "experts.gate_up_proj",
                                        int(e), 1)[0].T
            act = swiglu_clamp(gu[:self.moe_inter], gu[self.moe_inter:],
                               self.swiglu_limit)
            dn = self.sd.get_rows(stem + "experts.down_proj", int(e), 1)[0]
            acc += w[k] * (dn @ act)
        routed = self.sd.get(stem + "routed_expert_up_proj.weight") @ acc
        shared = (self.sd.get(stem + "shared_experts.down_proj.weight") @
                  swiglu_clamp(self.sd.get(stem + "shared_experts.gate_proj.weight") @ x,
                               self.sd.get(stem + "shared_experts.up_proj.weight") @ x,
                               self.swiglu_limit))
        total = routed + shared
        return rms_norm(total, self.sd.get(stem + "combine_output_norm.weight"),
                        self.eps).astype(np.float32)

    def dense_mlp(self, x, layer):
        stem = "model.layers.%d.mlp." % layer
        y = (self.sd.get(stem + "down_proj.weight") @
             swiglu_clamp(self.sd.get(stem + "gate_proj.weight") @ x,
                          self.sd.get(stem + "up_proj.weight") @ x,
                          self.swiglu_limit))
        return rms_norm(y, self.sd.get(stem + "post_fc2_norm.weight"), self.eps)

    # -- forward -----------------------------------------------------------
    def forward(self, token, pos, ids_all=()):
        x = self.sd.get_rows("model.embed_tokens.weight", token, 1)[0]
        streams = np.tile(x.astype(np.float32), (self.hc, 1))

        window = self.engram_max_ngram if hasattr(self, "engram_max_ngram") else 0
        history = list(ids_all)
        for l in range(self.layers):
            if window:
                ng = [history[pos - off] if pos - off >= 0 else self.engram_pad
                      for off in range(window)]
                streams = streams + self.engram(streams, l, token, pos, ng)
            agg, _pre, post = self.ihc(streams, l, "hc_attn_layer")
            xn = rms_norm(agg, self.sd.get("model.layers.%d.input_layernorm.weight" % l),
                          self.eps)
            attn, _vis = self.attention(xn, l, pos)
            streams = post[:, None] * attn[None, :] + streams
            if self.verbose:
                print("  L%-2d attn |y|=%.4f |streams|=%.4f" %
                      (l, np.linalg.norm(attn), np.linalg.norm(streams)),
                      flush=True)

            agg, _pre, post = self.ihc(streams, l, "hc_mlp_layer")
            xn = rms_norm(agg, self.sd.get(
                "model.layers.%d.post_attention_layernorm.weight" % l), self.eps)
            y = self.moe(xn, l) if self.is_moe(l) else self.dense_mlp(xn, l)
            streams = post[:, None] * y[None, :] + streams
            if self.verbose:
                print("  L%-2d ffn  |y|=%.4f |streams|=%.4f" %
                      (l, np.linalg.norm(y), np.linalg.norm(streams)), flush=True)

        # hc_head collapse, then the shared final norm.
        stem = "model.hc_head."
        fn = self.sd.get(stem + "hc_head_fn")
        base = self.sd.get(stem + "hc_head_base")
        scale = self.sd.get(stem + "hc_head_scale")
        flat = streams.reshape(-1)
        mixes = (fn @ flat) * rms_inv(flat, self.eps)
        pre = 1.0 / (1.0 + np.exp(-(mixes * scale[0] + base))) + self.hc_eps
        collapsed = np.einsum("j,ji->i", pre, streams)
        h = rms_norm(collapsed, self.sd.get("model.norm.weight"), self.eps)
        head = self.tbl("lm_head.weight")
        logits = head @ h
        return logits.astype(np.float32), h


def topk(logits, k=8):
    idx = np.argsort(-logits)[:k]
    return [(int(i), float(logits[i])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(lsx_paths.source_dir("hy4-flash")))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ids", default="", help="comma-separated input ids")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = ([int(x) for x in args.ids.split(",")] if args.ids
           else tok.encode(args.prompt, add_special_tokens=False))
    print("prompt ids:", ids, flush=True)

    m = Hy4Flash(args.model, verbose=args.verbose)
    if os.environ.get("LSX_HY4_ENGRAM", "1") == "1":
        m.engram_init()
    hist = []
    for step in range(len(ids) - 1 + args.tokens):
        token = ids[step] if step < len(ids) else nxt
        if step < len(ids):
            hist.append(token)
        logits, h = m.forward(token, step, hist)
        if step + 1 >= len(ids):
            nxt = int(np.argmax(logits))
            print("step %d -> %d %r  top8=%s" %
                  (step, nxt, tok.decode([nxt]), topk(logits)), flush=True)


if __name__ == "__main__":
    main()
