#!/usr/bin/env python3
"""Authoritative per-layer oracle for MiniMax-M3, from the HF reference code.

`tools/minimax_reference.py` is a hand-written re-implementation; this script
runs the CHECKPOINT'S OWN modeling code (transformers' `minimax_m3_vl`) for ONE
decoder layer on the meta device, loads only that layer's weights from the
source safetensors, and feeds the prompt token by token. It is what decides
whether a mismatch lives in the engine or in the hand-written reference.

Usage:
    tools/minimax_hf_layer_oracle.py --model <dir> --layer 0 \\
        --prompt "The capital of France is" [--dump <file>]

Prints the first 16 hidden values after the layer for the LAST prompt token,
in the same format the engine writes with LSX_MINIMAX_DUMP_LAYERS=<file>.
"""
import argparse
import json
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--thinking", default="disabled")
    ap.add_argument("--dump", default=None)
    ap.add_argument("--full-logits", action="store_true",
                    help="also run the final norm + lm_head for the last "
                         "token (needs the head, 2.4 GB)")
    args = ap.parse_args()

    from safetensors import safe_open
    from transformers import AutoTokenizer
    from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as M
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import (
        MiniMaxM3VLTextConfig)

    with open(os.path.join(args.model, "config.json")) as fh:
        root = json.load(fh)
    cfg = MiniMaxM3VLTextConfig(**root["text_config"])

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, thinking_mode=args.thinking)
    ids = tok(text, add_special_tokens=False)["input_ids"]

    with open(os.path.join(args.model, "model.safetensors.index.json")) as fh:
        index = json.load(fh)["weight_map"]

    files = {}

    def get(name):
        if name not in files:
            files[name] = safe_open(os.path.join(args.model, index[name]),
                                    framework="pt", device="cpu")
        return files[name].get_tensor(name)

    prefix = "language_model.model.layers.%d." % args.layer
    with torch.device("meta"):
        layer = M.MiniMaxM3VLDecoderLayer(cfg, args.layer)
    state = {}
    for name, shard in index.items():
        if not name.startswith(prefix):
            continue
        leaf = name[len(prefix):]
        # The checkpoint stores the dense MLP unfused; the module is fused.
        state[leaf] = get(name).to(torch.float32)
    gate = state.pop("mlp.gate_proj.weight", None)
    up = state.pop("mlp.up_proj.weight", None)
    if gate is not None and up is not None:
        state["mlp.gate_up_proj.weight"] = torch.cat([gate, up], dim=0)
    missing, unexpected = layer.load_state_dict(state, strict=False,
                                                assign=True)
    real_missing = [k for k in missing if "indexer" not in k]
    if real_missing:
        print("MISSING:", real_missing, file=sys.stderr)
        return 2
    if unexpected:
        print("UNEXPECTED:", unexpected, file=sys.stderr)
        return 2
    layer = layer.to(torch.float32).eval()
    rotary = M.MiniMaxM3VLRotaryEmbedding(cfg)

    embed = get("language_model.model.embed_tokens.weight")
    past = None
    cache_cls = M.DynamicCache
    past = cache_cls(config=cfg)
    out = None
    dbg = {}
    if os.environ.get("MM_DEBUG"):
        def hook(name):
            def fn(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                dbg[name] = [round(float(v), 5)
                             for v in t.view(-1)[:6]]
            return fn
        layer.self_attn.register_forward_hook(hook("attn"))
        layer.mlp.register_forward_hook(hook("mlp"))
        layer.input_layernorm.register_forward_hook(hook("input_norm"))
    with torch.no_grad():
        for pos, tid in enumerate(ids):
            x = embed[tid].to(torch.float32).view(1, 1, -1)
            pos_ids = torch.tensor([[pos]])
            cos, sin = rotary(x, position_ids=pos_ids)
            out = layer(x, attention_mask=None, position_ids=pos_ids,
                        past_key_values=past, position_embeddings=(cos, sin))
    h = out[0] if isinstance(out, tuple) else out
    if dbg:
        for k, v in dbg.items():
            print("HF", k, v[:6])
    vals = [round(float(v), 6) for v in h.view(-1)[:16]]
    line = "layer %d " % args.layer + " ".join("%.6f" % v for v in vals)
    print(line)
    if args.dump:
        with open(args.dump, "w") as fh:
            fh.write(line + "\n")
    if args.full_logits:
        # Final norm + head on THIS layer's output is not the model's logits;
        # this flag is only for shape/format debugging of a single-layer run.
        print("hidden[:8]:", vals[:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
