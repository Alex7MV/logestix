#!/usr/bin/env python3
"""Source-checkpoint oracle loader for Mistral-Small-4-119B-2603-NVFP4.

The published checkpoint is the Mistral "consolidated" layout: `params.json`
(with different field names than an HF config), no `config.json` and no
`model_type`, plus NVFP4-packed experts in `consolidated-*.safetensors` whose
index is named `consolidated.safetensors.index.json`. `AutoConfig` therefore
cannot construct it from the files, which is why the family's FSI/CODE/DET
goldens are declared debt rather than captured.

This module synthesizes the HF config so the reference model CAN run, which is
what a source-faithful golden needs. Three details are load-bearing and each
fails differently without them:

1. `partial_rotary_factor` (rope) - the rotary embedding is built as
   `head_dim * partial_rotary_factor` and attention applies it to the rotary
   half only. Left at the default 1.0 the cos/sin tensor is built at the full
   head width (128) against a 64-wide `q_rot`, and the forward pass raises
   "The size of tensor a (32) must match the size of tensor b (64)".
2. `llama_4_scaling_beta` (rope) - the Llama-4 position-dependent logit
   temperature is read from `rope_parameters`, NOT from `params.json`'s
   `llama_4_scaling` block. Missing, the first forward pass raises
   "unsupported operand type(s) for *: 'NoneType' and 'Tensor'".
3. `rope_type: "yarn"` with `attention_factor: 1.0` - `params.json` sets
   `yarn.apply_scale = false`, so the tables must not be pre-multiplied by the
   mscale factor. `hidden_dim` in `params.json` is the DENSE MLP width
   (12288), not the model width (4096).

Used by tools/{fsi,code,det}_capture.py (`loader: "mistral_small4"`). Nothing
here is part of the engine: this is the external parity oracle.
"""
import json
import os
import sys


def mistral_small4_config_dict(params):
    """params.json -> a Mistral4Config-shaped dict.

    `params` is the parsed contents of the checkpoint's params.json.
    """
    moe = params["moe"]
    qk_nope = params["qk_nope_head_dim"]
    qk_rope = params["qk_rope_head_dim"]
    yarn = params.get("yarn", {})
    llama4 = params.get("llama_4_scaling", {})
    return {
        "architectures": ["Mistral4ForCausalLM"],
        "model_type": "mistral4",
        "vocab_size": params["vocab_size"],
        "hidden_size": params["dim"],
        "intermediate_size": params["hidden_dim"],
        "moe_intermediate_size": moe["expert_hidden_dim"],
        "num_hidden_layers": params["n_layers"],
        "num_attention_heads": params["n_heads"],
        "num_key_value_heads": params["n_kv_heads"],
        "n_shared_experts": moe["num_shared_experts"],
        "n_routed_experts": moe["num_experts"],
        "num_experts_per_tok": moe["num_experts_per_tok"],
        "routed_scaling_factor": moe["routed_scale"],
        "n_group": moe["num_expert_groups"],
        "topk_group": moe["num_expert_groups_per_tok"],
        "first_k_dense_replace": moe["first_k_dense_replace"],
        "kv_lora_rank": params["kv_lora_rank"],
        "q_lora_rank": params["q_lora_rank"],
        "qk_rope_head_dim": qk_rope,
        "qk_nope_head_dim": qk_nope,
        "v_head_dim": params["v_head_dim"],
        "max_position_embeddings": params["max_position_embeddings"],
        "rms_norm_eps": params["norm_eps"],
        "pad_token_id": 11,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": params["tied_embeddings"],
        "torch_dtype": "bfloat16",
        "quantization_config": params["quantization_config"],
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": params["rope_theta"],
            "factor": yarn.get("factor", 128.0),
            "original_max_position_embeddings": yarn.get(
                "original_max_position_embeddings", 8192),
            "beta_fast": yarn.get("beta", 32.0),
            "beta_slow": yarn.get("beta_slow", 1.0),
            "attention_factor": 1.0,          # yarn.apply_scale = false
            "partial_rotary_factor": qk_rope / (qk_nope + qk_rope),
            "llama_4_scaling_beta": llama4.get("beta", 0.1),
        },
    }


def prepare_dir(src, dst):
    """Materialize <dst> as symlinks to the source shards + a synthesized config.

    The source shards are 66 GB; symlinks keep this free. transformers resolves
    a sharded checkpoint through `model.safetensors.index.json`, which the
    consolidated layout does not have, so both index names are provided.
    """
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "params.json")) as fh:
        cfg = mistral_small4_config_dict(json.load(fh))
    with open(os.path.join(dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)

    src_index = os.path.join(src, "consolidated.safetensors.index.json")
    with open(src_index) as fh:
        index = json.load(fh)
    for name in set(index["weight_map"].values()):
        link = os.path.join(dst, name)
        if not os.path.exists(link):
            os.symlink(os.path.join(src, name), link)
    for alias in ("model.safetensors.index.json",
                  "consolidated.safetensors.index.json"):
        link = os.path.join(dst, alias)
        if not os.path.exists(link):
            os.symlink(src_index, link)
    for extra in ("tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "tekken.json"):
        s = os.path.join(src, extra)
        d = os.path.join(dst, extra)
        if os.path.exists(s) and not os.path.exists(d):
            os.symlink(s, d)
    return cfg


# Names transformers' Mistral4ForCausalLM expects vs the consolidated
# checkpoint's own names. They differ in every component, so a plain
# from_pretrained() initializes the whole model RANDOMLY and says so only in a
# MISSING/UNEXPECTED table. Documented so the next session does not lose an hour
# to a forward pass over random weights (which produces NaN logits and looks
# like a numeric defect).
NAME_GAP = {
    "layers.N.attention.wq_a": "model.layers.N.self_attn.q_a_proj",
    "layers.N.attention.wq_b": "model.layers.N.self_attn.q_b_proj",
    "layers.N.attention.wkv_a_with_mqa":
        "model.layers.N.self_attn.kv_a_proj_with_mqa",
    "layers.N.attention.wkv_b": "model.layers.N.self_attn.kv_b_proj",
    "layers.N.attention.wo": "model.layers.N.self_attn.o_proj",
    "layers.N.attention.q_a_norm": "model.layers.N.self_attn.q_a_layernorm",
    "layers.N.attention.kv_a_norm": "model.layers.N.self_attn.kv_a_layernorm",
    "layers.N.attention_norm": "model.layers.N.input_layernorm",
    "layers.N.ffn_norm": "model.layers.N.post_attention_layernorm",
    "layers.N.gate": "model.layers.N.mlp.gate",
    "tok_embeddings": "model.embed_tokens",
    "output": "lm_head",
    "norm": "model.norm",
    # Layout (not just names): the checkpoint stores one expert per index as
    # `layers.N.experts.E.w{1,2,3}`, the class wants the FUSED stacked
    # `model.layers.N.mlp.experts.{gate_up_proj,down_proj}`.
    "layers.N.experts.E.w1/w2/w3":
        "model.layers.N.mlp.experts.gate_up_proj/down_proj (fused, stacked)",
}


def report_name_gap(src):
    """Print the checkpoint-vs-class naming gap for one layer. Diagnostics only."""
    with open(os.path.join(src, "consolidated.safetensors.index.json")) as fh:
        index = json.load(fh)
    ckpt = sorted(n for n in index["weight_map"]
                  if n.startswith("layers.0.") and ".experts." not in n)
    print("checkpoint layer-0 names:")
    for n in ckpt:
        print("  ", n)
    print("\nexpected (Mistral4ForCausalLM) equivalents:")
    for k, v in NAME_GAP.items():
        print(f"   {k:36s} -> {v}")
    return ckpt


def load(model_path, work_dir=None, allow_missing=False):
    """Load the reference model for capture.

    Returns `(tokenizer, model)`. `model_path` is either the consolidated source
    directory or an already-prepared work directory; `work_dir` (default: a
    sibling `.hf_oracle` directory) receives the symlinks + synthesized config.

    REFUSES to return a model whose parameters did not all come from the
    checkpoint (unless `allow_missing=True`). This checkpoint's tensor names do
    not match `Mistral4ForCausalLM`'s expectations (see NAME_GAP), so an
    unguarded load silently yields a randomly-initialized 119B model; a golden
    captured from it would be a different model's answers, which is exactly what
    `docs/OPEN_ENDEDNESS.md` forbids.

    Note the tokenizer regex warning: this checkpoint predates the corrected
    Mistral pre-tokenizer regex. The goldens' `input_ids` must come from THIS
    repository's tokenizer arrows, not from here, so a vendor tokenizer
    difference cannot silently move the prompt.
    """
    from transformers import AutoTokenizer, Mistral4ForCausalLM

    work_dir = work_dir or os.path.join(model_path, ".hf_oracle")
    prepare_dir(model_path, work_dir)
    tok = AutoTokenizer.from_pretrained(work_dir, fix_mistral_regex=True)
    model, info = Mistral4ForCausalLM.from_pretrained(
        work_dir, dtype="bfloat16", device_map="cpu", low_cpu_mem_usage=True,
        output_loading_info=True)
    model.eval()
    missing = info.get("missing_keys") or []
    if missing and not allow_missing:
        first = sorted(missing)[:8]
        raise RuntimeError(
            f"{len(missing)} parameters were NOT loaded from the checkpoint, so "
            f"the model is partially random and cannot serve as a parity "
            f"oracle. The consolidated tensor names do not match "
            f"Mistral4ForCausalLM's expectations (see NAME_GAP) and the "
            f"experts are stored per-expert rather than fused. A name+repack "
            f"mapping is required first (the vendor `mistral-inference` loader "
            f"is the reference implementation). First missing keys: {first}")
    return tok, model


if __name__ == "__main__":
    src = sys.argv[1]
    work = sys.argv[2] if len(sys.argv) > 2 else None
    tok, model = load(src, work)
    prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"
    import torch
    ids = tok(prompt, return_tensors="pt").input_ids
    print("ids:", ids.tolist())
    with torch.inference_mode():
        out = model(ids)
    top = torch.topk(out.logits[0, -1], 5)
    print("top5:", top.indices.tolist(), top.values.tolist())
    print("decode:", repr(tok.decode([top.indices[0].item()])))
