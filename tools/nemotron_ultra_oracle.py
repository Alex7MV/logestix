#!/usr/bin/env python3
"""Slice oracle for NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16.

The full checkpoint is 1.1 TB of BF16 and cannot be assembled on this host (750 GB
RAM, and `from_pretrained` needs the whole model plus activations). So this script
builds the reference from a SLICE instead: a real `NemotronHForCausalLM` whose
`layers_block_type` is a hand-picked subset of the checkpoint's own layers, with
the real `backbone.embeddings`, `backbone.norm_f` and `lm_head`. That is enough to
execute — and therefore to be an oracle for — every piece of math this model adds
over the 30B sibling:

  * the latent MoE bridge (`fc1_latent_proj` -> experts at 2048 -> `fc2_latent_proj`)
  * the sigmoid top-22 router with `e_score_correction_bias` and
    `routed_scaling_factor` 5.0
  * Mamba-2 at 256 heads x 64 (d_inner 16384, gated-norm group 2048)
  * NoPE full attention at 64 query / 2 KV heads

What it produces
----------------
`--out` receives `layers_slice.json` with, for each selected layer and each
position of the prompt, a summary of that layer's hidden state (L2 norm plus the
first `--dims` values). The summary is what makes the file small enough to commit
and still sharp enough to catch a wrong router, a missing latent projection or a
rotary applied where the reference applies none: all three change the norm and the
leading dims far beyond the 4-bit noise floor.

Usage
-----
  python3 tools/nemotron_ultra_oracle.py \
      --checkpoint <source root>/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 \
      --layers 0,1,7 \
      --prompt "The capital of France is" \
      --out tests/fixtures/nemotron_ultra/layers_slice.json

Only tensors belonging to the selected layers are loaded, so peak RSS is roughly
`2 * params(selected layers)`: layers 0 (mamba), 1 (moe) and 7 (attention) are
about 4 GB in BF16.

Layer choice note: layer indices come from `layers_block_type`, so `0` is a mamba
block, `1` a MoE block and `7` the first full-attention block. Selecting a subset
re-indexes the blocks inside the built model (they become blocks 0,1,2), which is
why each record carries both `source_layer` and `block_index`.
"""
import argparse
import json
import os
import sys

import torch


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise SystemExit("--layers is empty")
    return out


def build_sliced_model(checkpoint, layers, dtype):
    """Assembles a NemotronHForCausalLM from the selected real layers."""
    from transformers import NemotronHConfig, NemotronHForCausalLM

    with open(os.path.join(checkpoint, "config.json")) as f:
        raw = json.load(f)
    block_types = raw["layers_block_type"]
    for l in layers:
        if l < 0 or l >= len(block_types):
            raise SystemExit(f"layer {l} out of range (model has {len(block_types)})")

    sliced = [block_types[l] for l in layers]

    cfg = dict(raw)
    cfg["layers_block_type"] = sliced
    cfg.pop("num_hidden_layers", None)
    # The MTP head is ballast the reference itself drops on load.
    cfg["num_nextn_predict_layers"] = 0
    cfg.pop("mtp_layers_block_type", None)
    config = NemotronHConfig(**cfg)

    with torch.device("meta"):
        model = NemotronHForCausalLM(config)
    model = model.to_empty(device="cpu")
    model = model.to(dtype)
    return model, raw, sliced


def load_sliced_weights(model, checkpoint, layers, sliced):
    """Loads only the selected layers' tensors plus the real globals."""
    from safetensors import safe_open

    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # The checkpoint's own layout is NOT the modeling class's state dict:
    # transformers' `conversion_mapping` for nemotron_h does two things this has
    # to reproduce (conversion_mapping.py, NemotronH entry around line 1145):
    #   * rename the `backbone.` prefix to `model.`;
    #   * MERGE the per-expert `mixer.experts.<E>.{up,down}_proj.weight` tensors
    #     into the single 3-D `mixer.experts.{up,down}_proj` parameter the
    #     reference declares as [num_experts, out, in].
    # Without the merge the routed experts stay uninitialised and the slice
    # produces garbage logits (top1 = 0), i.e. an oracle that agrees with nothing.
    remap = {src: new for new, src in enumerate(layers)}

    def keep_and_rename(name):
        if name.startswith("mtp."):
            return None
        if name.startswith("backbone."):
            name = "model." + name[len("backbone."):]
        if name.startswith("model.layers."):
            rest = name[len("model.layers."):]
            dot = rest.find(".")
            src = int(rest[:dot])
            if src not in remap:
                return None
            return f"model.layers.{remap[src]}{rest[dot:]}"
        return name

    wanted = {}
    merged = {}  # (new_layer, which) -> {expert: (shard, old_name)}
    for name, shard in weight_map.items():
        new = keep_and_rename(name)
        if new is None:
            continue
        parts = new.split(".")
        if "experts" in parts and parts[parts.index("experts") + 1].isdigit():
            i = parts.index("experts")
            layer_prefix = ".".join(parts[:i])
            expert = int(parts[i + 1])
            which = parts[i + 2]  # up_proj / down_proj
            merged.setdefault((layer_prefix, which), {})[expert] = (shard, name)
            continue
        wanted.setdefault(shard, []).append((name, new))

    # Preallocate each merged 3-D parameter and copy experts straight into it, so
    # the peak is one copy of the layer rather than one dict of 512 tensors.
    merge_targets = {}
    for (layer_prefix, which), by_expert in merged.items():
        key = f"{layer_prefix}.experts.{which}"
        ref = model.get_parameter(key)
        buf = torch.empty_like(ref)
        merge_targets[key] = (buf, by_expert)

    state = {}
    shard_ids = sorted(wanted)
    for i, shard in enumerate(shard_ids, 1):
        print(f"[{i}/{len(shard_ids)}] {shard}: {len(wanted[shard])} tensors",
              file=sys.stderr)
        with safe_open(os.path.join(checkpoint, shard), framework="pt") as f:
            for old, new in wanted[shard]:
                state[new] = f.get_tensor(old)

    for key, (buf, by_expert) in merge_targets.items():
        print(f"merging {len(by_expert)} experts into {key} {tuple(buf.shape)}",
              file=sys.stderr)
        by_shard = {}
        for expert, (shard, old) in by_expert.items():
            by_shard.setdefault(shard, []).append((expert, old))
        for shard, items in by_shard.items():
            with safe_open(os.path.join(checkpoint, shard), framework="pt") as f:
                for expert, old in items:
                    buf[expert].copy_(f.get_tensor(old))
        state[key] = buf

    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    if real_missing:
        print(f"WARNING: {len(real_missing)} tensors not loaded, e.g. "
              f"{real_missing[:5]}", file=sys.stderr)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected tensors, e.g. "
              f"{unexpected[:5]}", file=sys.stderr)
    return model


E4M3_LEVELS = None


def nvfp4_roundtrip(w, block=16):
    """Applies the NVFP4 codec (E2M1 + per-16 E4M3 block scale) to a 2-D weight.

    This is the same encoding the converter applies (docs/NVIDIA_DOCS §1): block
    amax -> scale = amax/6, scale stored as E4M3, elements as E2M1 levels
    {0,.5,1,1.5,2,3,4,6} (sign-magnitude). The GLOBAL scale is folded into the
    block scales here because it is a power-of-two-free float that cancels in the
    round trip; the point of this function is the RESOLUTION loss, which is what
    separates a runtime defect from 4-bit noise.
    """
    import torch

    orig = w.detach().to(torch.float32)
    out_shape = orig.shape
    flat = orig.reshape(-1, orig.shape[-1])
    rows, cols = flat.shape
    pad = (-cols) % block
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    blocks = flat.reshape(rows, -1, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / 6.0
    # E4M3 round-trip of the block scale: 3 mantissa bits, exponent bias 7.
    import math
    e = torch.floor(torch.log2(scale))
    e = torch.clamp(e, min=-6.0, max=8.0)
    base = torch.pow(2.0, e)
    mant = torch.round((scale / base - 1.0) * 8.0) / 8.0
    mant = torch.clamp(mant, min=0.0, max=1.875)
    scale_q = base * (1.0 + mant)
    # E2M1 levels.
    norm = blocks / scale_q
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mag = norm.abs().clamp(max=6.0)
    idx = torch.bucketize(mag, (levels[1:] + levels[:-1]) / 2.0)
    q = levels[idx] * torch.sign(norm)
    deq = q * scale_q
    deq = deq.reshape(rows, -1)[:, :cols]
    return deq.reshape(out_shape).to(w.dtype)


def quantize_like_converter(model, block_types_selected):
    """NVFP4-quantizes every weight the converter stores as NVFP4.

    Mirrors `ultra_tensor_is_nvfp4`: mixer in/out projections, attention q/k/v/o,
    the latent bridge, the shared expert, and the routed experts. Norms, biases,
    A_log/D/dt_bias, conv1d, the router gate and embed/head stay raw — quantizing
    them here would make the reference disagree with the engine for reasons the
    engine never does.
    """
    import torch

    n = 0
    for name, p in list(model.named_parameters()):
        if not name.startswith("model.layers."):
            continue
        tail = name.split(".", 3)[-1]
        if not tail.endswith(".weight"):
            continue
        if any(k in tail for k in ("in_proj", "out_proj", "q_proj", "k_proj",
                                   "v_proj", "o_proj", "fc1_latent_proj",
                                   "fc2_latent_proj", "shared_experts.up_proj",
                                   "shared_experts.down_proj", "experts.")):
            with torch.no_grad():
                p.copy_(nvfp4_roundtrip(p.data))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layers", default="0,1,7",
                    help="source layer indices to assemble (default 0,1,7 = "
                         "mamba, moe, attention)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ids-file", help="whitespace-separated token ids to use "
                                       "instead of encoding --prompt")
    ap.add_argument("--dims", type=int, default=8,
                    help="leading hidden dimensions recorded per layer (default 8)")
    ap.add_argument("--quantize-nvfp4", action="store_true",
                    help="apply the engine's NVFP4 codec to the reference's own "
                         "weights before running, so the comparison isolates "
                         "runtime error from 4-bit quantization noise")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    dtype = torch.bfloat16
    model, raw, sliced = build_sliced_model(args.checkpoint, layers, dtype)
    model = load_sliced_weights(model, args.checkpoint, layers, sliced)
    model.eval()

    if args.quantize_nvfp4:
        n = quantize_like_converter(model, sliced)
        print(f"NVFP4 round-trip applied to {n} weight tensors", file=sys.stderr)

    # Tokenize with the checkpoint's own tokenizer.json (the family golden is
    # captured from it too, so ids match the engine's bridge).
    if args.ids_file:
        with open(args.ids_file) as f:
            ids = [int(x) for x in f.read().split()]
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(args.checkpoint, "tokenizer.json"))
        ids = tok.encode(args.prompt, add_special_tokens=False).ids

    input_ids = torch.tensor([ids], dtype=torch.long)
    print(f"prompt ids ({len(ids)}): {ids}", file=sys.stderr)

    captured = {}

    def hook(name, src_layer):
        def fn(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.float()
            last = h[0]  # [seq, hidden]
            captured[name] = {
                "source_layer": src_layer,
                "l2": [float(x) for x in last.norm(dim=-1).tolist()],
                "lead": [[float(v) for v in row] for row in
                         last[:, : args.dims].tolist()],
            }
        return fn

    handles = []
    for new_idx, blk in enumerate(model.model.layers):
        handles.append(blk.register_forward_hook(hook(f"block_{new_idx}", layers[new_idx])))

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    for h in handles:
        h.remove()

    logits = out.logits[0, -1].float()
    record = {
        "checkpoint": os.path.basename(args.checkpoint.rstrip("/")),
        "nvfp4_reference": bool(args.quantize_nvfp4),
        "num_layers": raw["layers_block_type"].__len__(),
        "layers": layers,
        "layer_types": sliced,
        "prompt": args.prompt,
        "prompt_ids": ids,
        "dims": args.dims,
        "hidden_size": raw["hidden_size"],
        "top1_id": int(logits.argmax()),
        "top1_logit": float(logits.max()),
        "final_logits_head": [float(v) for v in logits[:16].tolist()],
        "blocks": captured,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print(f"wrote {args.out}: {len(captured)} blocks, "
          f"{len(ids)} positions, top1={record['top1_id']}", file=sys.stderr)


if __name__ == "__main__":
    main()
