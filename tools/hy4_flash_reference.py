#!/usr/bin/env python3
"""Run the AUTHORITATIVE Hy4-flash reference (transformers PR #50903) on the
checkpoint, with the same cut set as the converted artifact.

This is the reference of record: the checkpoint's README pins PR #50903 as the
implementation ("Support for Hy4 flash will be merged into the transformers main
branch soon. Until then, install from the commit below"), so goldens are captured
here and nowhere else.

The README pins the implementation: "Support for Hy4 flash will be merged into the
transformers main branch soon. Until then, install from the commit below:
  pip install transformers @ git+https://github.com/huggingface/transformers.git@refs/pull/50903/head"

This script loads ONLY the text backbone, and skips exactly the tensors the
artifact cuts (Engram, the ViT tower, the DFly2 drafter), so its logits are the
reference the engine must reproduce. It runs on CPU in bf16 with a streamed
state_dict, because the full tree is 411 GB and no single GPU here can hold it.

    /mnt/fast_storage/tmp/hy4venv2/bin/python tools/hy4_flash_reference.py \
        --model /mnt/fast_storage/huggingface/Hy4-flash --prompt "The capital of France is"
"""
import argparse
import json
import os
import struct
import sys

import torch


# The checkpoint's leaf names are NOT the reference model's module names. The
# mapping is the PR's own `conversion_mapping.py` hy_v4_flash list, copied
# verbatim: a reference run with unmapped names silently leaves 422 parameters at
# their random initialisation (that is what the first attempt did).
RENAMES = [
    (".self_attn.learnable_sink_param", ".self_attn.sinks"),
    (".self_attn.linear_gate.weight", ".self_attn.gate_proj.weight"),
    (".mlp.post_fc2_norm.weight", ".post_mlp_layernorm.weight"),
    (".mlp.routed_expert_down_proj.weight", ".mlp.pre_expert_proj.weight"),
    (".mlp.routed_expert_up_proj.weight", ".mlp.post_expert_proj.weight"),
    (".hc_attn_layer.hc_pre.hc_fn", ".attn_hc.fn"),
    (".hc_attn_layer.hc_pre.hc_base", ".attn_hc.base"),
    (".hc_attn_layer.hc_pre.hc_scale", ".attn_hc.scale"),
    (".hc_mlp_layer.hc_pre.hc_fn", ".ffn_hc.fn"),
    (".hc_mlp_layer.hc_pre.hc_base", ".ffn_hc.base"),
    (".hc_mlp_layer.hc_pre.hc_scale", ".ffn_hc.scale"),
    (".hc_head.hc_head_fn", ".hc_head.hc_fn"),
    (".hc_head.hc_head_base", ".hc_head.hc_base"),
    (".hc_head.hc_head_scale", ".hc_head.hc_scale"),
    ("hc_head.hc_head_fn", "hc_head.hc_fn"),
    ("hc_head.hc_head_base", "hc_head.hc_base"),
    ("hc_head.hc_head_scale", "hc_head.hc_scale"),
]


def rename(name):
    for src, dst in RENAMES:
        name = name.replace(src, dst)
    return name


ENGRAM = "engram_layer."


def iter_tensors(root, skip_engram, skip_vision, skip_dfly):
    """Yields (name, tensor) reading one shard at a time."""
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    by_file = {}
    for name, fn in wmap.items():
        if skip_engram and ENGRAM in name:
            continue
        if skip_vision and ("vit." in name):
            continue
        if skip_dfly and ("dfly_layers." in name):
            continue
        by_file.setdefault(fn, []).append(name)
    for fn, names in by_file.items():
        path = os.path.join(root, fn)
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            base = 8 + n
            for name in names:
                e = hdr[name]
                s, eo = e["data_offsets"]
                dt = {"BF16": torch.bfloat16, "F16": torch.float16,
                      "F32": torch.float32, "I32": torch.int32}[e["dtype"]]
                fh.seek(base + s)
                raw = fh.read(eo - s)
                t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(e["shape"])
                yield name, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/fast_storage/huggingface/Hy4-flash")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--ids", default="")
    ap.add_argument("--engram", action="store_true",
                    help="keep the Engram tables (needs ~130 GB more RAM)")
    ap.add_argument("--out", default="/mnt/fast_storage/tmp/hy4-ref-step.json")
    ap.add_argument("--frame", action="store_true",
                    help="wrap the prompt in the checkpoint's chat template")
    args = ap.parse_args()

    from transformers import AutoTokenizer, HYV4FlashTextConfig
    from transformers.models.hy_v4_flash import modeling_hy_v4_flash as R

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(os.path.join(args.model, "config.json")) as f:
        full = json.load(f)
    text_cfg = dict(full["text_config"])
    if not args.engram:
        # Same cut set as the converted artifact.
        text_cfg["engram_layer_ids"] = None
        text_cfg.pop("engram_layer_head_vocab_sizes", None)
    cfg = HYV4FlashTextConfig(**text_cfg)
    cfg._attn_implementation = "eager"

    # bf16 params: a float32 instantiation is 310 GB and this box refuses the
    # commit (CommitLimit is 402 GB), which killed the first two attempts.
    torch.set_default_dtype(torch.bfloat16)
    model = R.HYV4FlashTextModel._from_config(cfg)
    model.eval()

    # Copy the backbone in TENSOR BY TENSOR: one 155 GB model plus one source
    # tensor is the only footprint the commit limit allows, so materializing a
    # whole state_dict (even with assign=True) is not an option.
    params = dict(model.named_parameters())
    loaded = set()
    for name, t in iter_tensors(args.model, skip_engram=not args.engram,
                                skip_vision=True, skip_dfly=True):
        key = name[len("model."):] if name.startswith("model.") else name
        key = rename(key)
        p = params.get(key)
        if p is None:
            continue
        with torch.no_grad():
            p.copy_(t)
        loaded.add(key)
        del t
    missing = [n for n in params if n not in loaded]
    print("loaded:", len(loaded), "missing:", len(missing), missing[:6],
          flush=True)

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    elif args.frame:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, tokenize=True, return_dict=False,
            reasoning_effort="no_think")
    else:
        ids = tok.encode(args.prompt, add_special_tokens=False)
    print("prompt ids:", ids, flush=True)

    # lm_head is not part of the text model: it is a wrapper-level nn.Linear with
    # the checkpoint's [vocab, hidden] table.
    head = None
    for name, t in iter_tensors(args.model, skip_engram=True, skip_vision=True,
                                skip_dfly=True):
        if name == "lm_head.weight":
            head = t.bfloat16()
            break
    assert head is not None, "lm_head.weight not found"
    print("lm_head dtype", head.dtype, tuple(head.shape), flush=True)

    # Decode by feeding the WHOLE prefix every step (`use_cache=False`): the
    # dynamic cache path threads per-layer-type M-RoPE position planes through
    # `DynamicCache`, and a hand-rolled one would be a second implementation to
    # trust. O(n^2) is affordable at golden-capture lengths.
    nxt = None
    hist = list(ids)
    with torch.no_grad():
        for step in range(len(ids) - 1 + args.tokens):
            if step >= len(ids):
                hist.append(nxt)
            prefix = hist[:step + 1]
            out = model(input_ids=torch.tensor([prefix]), use_cache=False,
                        return_dict=True)
            h = out.last_hidden_state[0, -1]
            logits = h.float() @ head.float().T
            if step + 1 >= len(ids):
                nxt = int(torch.argmax(logits))
                top = torch.topk(logits, 8)
                print("gen %d: seq=%d -> %d %r top8=%s" %
                      (step - len(ids) + 1, len(prefix), nxt,
                       tok.decode([nxt]),
                       [(int(i), round(float(v), 4))
                        for i, v in zip(top.indices, top.values)]), flush=True)
                json.dump({"ids": prefix, "next": nxt,
                           "top8": [(int(i), float(v)) for i, v in
                                    zip(top.indices, top.values)]},
                          open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
