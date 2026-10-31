#!/usr/bin/env python3
"""Source-checkpoint oracle loader for Hy-Memory-30B-A3B (hy_v3).

The published checkpoint does NOT load into the stock
`transformers.models.hy_v3.HYV3ForCausalLM`: the weights are stored per expert
(`model.layers.N.mlp.experts.E.{gate,up,down}_projection`) and the shared expert
is `mlp.shared_mlp.*`, while the class expects the fused
`mlp.experts.gate_up_proj` / `mlp.experts.down_proj` 3-D tensors and
`mlp.shared_experts.*`. The router is `mlp.router.gate.weight` +
`mlp.expert_bias`, the class wants `mlp.gate.weight` +
`mlp.e_score_correction_bias`.

It also carries its own expert width: `expert_hidden_dim` (768) is the real
tensor width while the vestigial `moe_intermediate_size` says 1536. The loader
sets `moe_intermediate_size = expert_hidden_dim` before constructing the config,
which is what makes `HYV3Experts` and the shared expert line up with the
checkpoint.

Used by tools/{fsi,code,det}_capture.py (`loader: "hy_v3"`). Nothing here is
part of the engine: this is the external parity oracle.
"""
import glob
import json
import os

import torch


def _hy_config(model_path):
    from transformers import HYV3Config

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    raw = dict(raw)
    expert_width = raw.pop("expert_hidden_dim", raw.get("moe_intermediate_size"))
    raw["moe_intermediate_size"] = int(expert_width)
    raw.pop("eod_token_id", None)
    raw.pop("enable_lm_head_fp32", None)
    raw.pop("enable_moe_fp32_combine", None)
    raw.pop("moe_router_enable_expert_bias", None)
    raw.pop("moe_router_use_sigmoid", None)
    raw.pop("qk_norm", None)
    raw.pop("route_norm", None)
    raw.pop("sep_token_id", None)
    raw.pop("use_grouped_mm", None)
    raw.pop("attn_impl", None)
    raw.pop("num_shared_experts", None)
    raw.pop("num_experts", None)          # config maps num_local_experts
    raw["num_local_experts"] = int(json.load(
        open(os.path.join(model_path, "config.json"),
             encoding="utf-8"))["num_experts"])
    return HYV3Config(**raw)


def load(model_path, device_map=None, dtype=None):
    """Builds HYV3ForCausalLM and fills it from the per-expert checkpoint."""
    from safetensors.torch import load_file
    from transformers import HYV3ForCausalLM

    cfg = _hy_config(model_path)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16 if dtype in (None, "bf16") else dtype)
    lm = HYV3ForCausalLM(cfg)
    lm.eval()
    params = dict(lm.named_parameters())
    bufs = dict(lm.named_buffers())

    def put(name, tensor):
        if name in params:
            with torch.no_grad():
                params[name].data.copy_(tensor)
            return True
        if name in bufs:
            with torch.no_grad():
                bufs[name].copy_(tensor)
            return True
        return False

    num_layers = cfg.num_hidden_layers
    experts = cfg.num_local_experts
    inter = cfg.moe_intermediate_size

    # Expert stacks are assembled per layer (all experts of a layer live in one
    # shard in the published layout, but that is not guaranteed for a
    # republished tree, so the buffers are filled as shards stream in).
    gu = {l: torch.zeros((experts, 2 * inter, cfg.hidden_size),
                         dtype=torch.bfloat16)
          for l in range(num_layers)}
    dn = {l: torch.zeros((experts, cfg.hidden_size, inter),
                         dtype=torch.bfloat16)
          for l in range(num_layers)}
    gu_seen = {l: 0 for l in range(num_layers)}

    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        for key, tensor in load_file(shard).items():
            # Hy-SimulMT stores a speech encoder beside the same text backbone;
            # the oracle runs the TEXT decoder only, so that ballast is skipped.
            if key.startswith("audio_tower."):
                continue
            if key.startswith("model.layers.") and ".mlp.experts." in key:
                rest = key.split(".mlp.experts.", 1)[1]
                eid_s, leaf = rest.split(".", 1)
                lid = int(key.split(".")[2])
                eid = int(eid_s)
                if leaf == "gate_proj.weight":
                    gu[lid][eid, :inter] = tensor
                elif leaf == "up_proj.weight":
                    gu[lid][eid, inter:] = tensor
                elif leaf == "down_proj.weight":
                    dn[lid][eid] = tensor
                gu_seen[lid] += 1
                continue
            name = (key.replace(".mlp.router.gate.weight", ".mlp.gate.weight")
                       .replace(".mlp.expert_bias",
                                ".mlp.e_score_correction_bias")
                       .replace(".mlp.shared_mlp.", ".mlp.shared_experts."))
            if not put(name, tensor):
                raise SystemExit("hy oracle: unmapped tensor %r" % key)

    for l in range(num_layers):
        if cfg.mlp_layer_types[l] != "sparse":
            continue
        if gu_seen[l] != 3 * experts:
            raise SystemExit(
                "hy oracle: layer %d saw %d expert tensors, want %d"
                % (l, gu_seen[l], 3 * experts))
        put("model.layers.%d.mlp.experts.gate_up_proj" % l, gu[l])
        put("model.layers.%d.mlp.experts.down_proj" % l, dn[l])
    torch.set_default_dtype(prev)

    if device_map:
        from accelerate import dispatch_model, infer_auto_device_map
        dm = infer_auto_device_map(
            lm, max_memory={0: "28GiB", "cpu": "640GiB"},
            no_split_module_classes=["HYV3DecoderLayer", "HYV3MoE"])
        lm = dispatch_model(lm, device_map=dm)
    return lm


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--tokens", type=int, default=24)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    lm = load(args.model, args.device_map)
    tok = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": "The capital of France is"}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(lm.device)
    with torch.no_grad():
        out = lm.generate(ids, do_sample=False, max_new_tokens=args.tokens)
    print(repr(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
    print("prompt_ids:", ids[0].tolist())
