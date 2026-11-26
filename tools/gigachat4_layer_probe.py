#!/usr/bin/env python3
"""GigaChat4-37B-A4B layer-prefix probe (ingestion debug aid).

The full source oracle (`gigachat4_oracle.py`) has to dequantize all 39 GB of
FP8 body, which is slow. When the engine and the source disagree, the useful
question is *which layer diverges first*, and that needs only the embedding plus
a prefix of decoder layers. This script loads exactly that prefix, runs the
reference forward on a short id prefix and writes the residual stream after
layer `--layer` at position `--pos`.

The engine side is `LSX_GC4_DUMP_LAYERS=<dir>` (see GcPipeline::Step): it writes
F32 `layer_<idx>.bin` for every layer at position 0.

Usage:
  python3 tools/gigachat4_layer_probe.py --model <dir> --ids-file ids.txt \
      --layer 2 --pos 0 --out /tmp/gc4/ref_layer_02.bin [--compare /tmp/gc4/layer_02.bin]
"""
import argparse
import json
import os
import shutil
import struct
import sys
import tempfile

import numpy as np
import torch


def shard_map(model_dir):
    return json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]


def read_raw(model_dir, weight_map, name):
    shard = weight_map[name]
    with open(os.path.join(model_dir, shard), "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        m = header[name]
        f.seek(8 + n + m["data_offsets"][0])
        return m, f.read(m["data_offsets"][1] - m["data_offsets"][0])


def dequant(model_dir, weight_map, name, dev="cuda"):
    m, raw = read_raw(model_dir, weight_map, name)
    shape = tuple(m["shape"])
    if m["dtype"] == "F8_E4M3":
        sm, sraw = read_raw(model_dir, weight_map, name + "_scale_inv")
        w = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(shape).to(dev)
        wf = w.view(torch.float8_e4m3fn).to(torch.float32)
        sc = torch.frombuffer(bytearray(sraw), dtype=torch.float32).to(dev).view(tuple(sm["shape"]))
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[: shape[0], : shape[1]]
        return (wf * sc).to(torch.bfloat16)
    if m["dtype"] == "BF16":
        return (torch.frombuffer(bytearray(raw), dtype=torch.uint16).to(dev).to(torch.int32) << 16) \
            .to(torch.int32).view(torch.float32).view(shape).to(torch.bfloat16)
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).view(shape).to(dev).to(torch.bfloat16)


def build_skeleton(model_dir):
    tmp = tempfile.mkdtemp(prefix="gc4cfg_")
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    cfg.pop("quantization_config", None)
    json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
    for f in ("modeling_gigachat4.py", "configuration_gigachat4.py"):
        shutil.copy(os.path.join(model_dir, f), tmp)
    open(os.path.join(tmp, "__init__.py"), "w").close()
    sys.path.insert(0, os.path.dirname(tmp))
    from importlib import import_module
    mod = import_module(os.path.basename(tmp) + ".modeling_gigachat4")
    conf = mod.GigaChat4Config(**cfg)
    conf._attn_implementation = "eager"
    torch.set_grad_enabled(False)
    return mod, mod.GigaChat4ForCausalLM(conf).to(torch.bfloat16).eval(), tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--max-ids", type=int, default=24)
    args = ap.parse_args()

    ids = [int(x) for x in open(args.ids_file).read().replace(",", " ").split()][: args.max_ids]
    wm = shard_map(args.model)
    mod, model, tmp = build_skeleton(args.model)

    # A tensor is needed when it belongs to the embedding, the final norm, the
    # MTP layer is never applied, or its layer index is <= --layer.
    def needed(name):
        if name.startswith("model.layers."):
            idx = int(name.split(".")[2])
            return idx <= args.layer
        return name in ("model.embed_tokens.weight", "lm_head.weight") or name.startswith("model.norm")

    names = [n for n in sorted(wm) if needed(n) and not n.endswith("_scale_inv")]
    print("loading %d tensors for layers 0..%d" % (len(names), args.layer), flush=True)
    sd = {}
    for i, n in enumerate(names):
        sd[n] = dequant(args.model, wm, n)
        if (i + 1) % 200 == 0:
            print("  %d/%d" % (i + 1, len(names)), flush=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("loaded; missing %d unexpected %d" % (len(missing), len(unexpected)), flush=True)
    if missing:
        print("first missing:", list(missing)[:5], flush=True)

    # The reference returns a plain tuple (return_dict is false in the config),
    # so capture the per-layer states with a forward pre-hook on each layer.
    states = {}

    def grab(idx):
        def hook(_module, inputs):
            states[idx] = inputs[0].detach()
        return hook

    for i in range(args.layer + 1):
        model.model.layers[i].register_forward_pre_hook(grab(i))
    # Stage taps of layer 0, the same points the engine dumps under
    # LSX_GC4_DUMP_LAYERS (s0_embed / s0_in_norm / s0_attn / s0_pre_mlp / s0_mlp).
    stage = {}

    def tap(name):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            stage[name] = t.detach()
        return hook

    def pre(name):
        def hook(_m, inputs):
            stage[name] = inputs[0].detach()
        return hook

    if args.layer >= 0:
        model.model.layers[0].input_layernorm.register_forward_hook(tap("s0_in_norm"))
        model.model.layers[0].self_attn.register_forward_hook(tap("s0_attn"))
        model.model.layers[0].post_attn_norm_dummy = None
        model.model.layers[0].post_attention_layernorm.register_forward_hook(
            tap("s0_pre_mlp"))
        model.model.layers[0].mlp.register_forward_hook(tap("s0_mlp"))
        model.model.layers[0].input_layernorm.register_forward_pre_hook(
            pre("s0_embed"))
        model.model.layers[0].post_self_attn_layernorm.register_forward_hook(
            tap("s0_postself"))
        model.model.layers[0].mlp.register_forward_pre_hook(pre("s0_pre_mlp_ref"))
    with torch.no_grad():
        model(input_ids=torch.tensor([ids], dtype=torch.long))
    # states[i] is the INPUT to layer i, so the OUTPUT of `--layer` is
    # states[layer + 1] when it exists, else recompute from the next layer's
    # input (we only ever need layer <= args.layer + 1, so hook one extra).
    if (args.layer + 1) not in states:
        model.model.layers[args.layer + 1].register_forward_pre_hook(
            grab(args.layer + 1))
        with torch.no_grad():
            model(input_ids=torch.tensor([ids], dtype=torch.long))
    ref = states[args.layer + 1][0, args.pos].float().cpu().numpy()
    ref.tofile(args.out)
    print("reference layer %d pos %d: std=%.5f" % (args.layer, args.pos, float(np.std(ref))))

    for name, t in stage.items():
        v = t[0, args.pos].float().cpu().numpy()
        print("  ref stage %-14s std=%.5f max=%.4f" %
              (name, float(np.std(v)), float(np.abs(v).max())))

    # Per-stage comparison, when the engine dumps are present.
    d = os.path.dirname(args.compare or "")
    for name, t in stage.items():
        path = os.path.join(d, name + ".bin")
        if not os.path.exists(path):
            continue
        eng = np.fromfile(path, dtype=np.float32)
        ref_t = t[0, args.pos].float().cpu().numpy()
        n = min(len(eng), len(ref_t))
        e, r = eng[:n].astype(np.float64), ref_t[:n].astype(np.float64)
        cos = float(np.dot(e, r) / np.sqrt((e * e).sum() * (r * r).sum() + 1e-30))
        print("  stage %-12s cos=%.5f std_engine=%.5f std_ref=%.5f" %
              (name, cos, float(np.std(eng)), float(np.std(ref_t))))

    if args.compare and os.path.exists(args.compare):
        eng = np.fromfile(args.compare, dtype=np.float32)
        n = min(len(eng), len(ref))
        e, r = eng[:n].astype(np.float64), ref[:n].astype(np.float64)
        cos = float(np.dot(e, r) / np.sqrt((e * e).sum() * (r * r).sum() + 1e-30))
        rel = float(np.sqrt(((e - r) ** 2).mean()) / (np.sqrt((r * r).mean()) + 1e-30))
        print("engine vs reference: cos=%.5f relL2=%.5f std_engine=%.5f" %
              (cos, rel, float(np.std(eng))))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
