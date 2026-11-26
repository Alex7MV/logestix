#!/usr/bin/env python3
"""GigaChat4-37B-A4B source oracle probe (Model Ingestion Protocol stage 9).

The operator chose template-derived FSI/CODE/DET goldens plus ONE short
source-side probe (2026-11-26), because running the FP8 source needs a
host-side dequant to BF16 (74 GB) over the 39 GB checkpoint. This script is
that probe:

  1. dequantizes the FP8 block-128 body with numpy (BF16 for everything else),
  2. builds `modeling_gigachat4.GigaChat4ForCausalLM` in BF16 on the CPU,
  3. teacher-forces a short id prefix and writes the LAST-position logits,
  4. optionally compares them with the engine's (`--engine-logits`), applying
     the same `final_logit_softcapping` to both.

The engine side is produced by `test_gigachat_moe_37b_model` with
LSX_GC4_PROBE_IDS / LSX_GC4_PROBE_OUT set (see tests/test_gigachat_moe_37b_model.cpp).

Usage:
  python3 tools/gigachat4_oracle.py \
      --model /mnt/fast_storage/huggingface/GigaChat4-37B-A4B \
      --ids-file /mnt/fast_storage/tmp/gc4/probe_ids.txt \
      --out /mnt/fast_storage/tmp/gc4/source_logits.bin \
      [--engine-logits /mnt/fast_storage/tmp/gc4/engine_logits.bin] \
      [--max-ids 24] [--dtype bfloat16]
"""
import argparse
import json
import os
import shutil
import struct
import sys
import tempfile

import numpy as np


def _shard_map(model_dir):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    return idx["weight_map"]


def _read_raw(model_dir, weight_map, name):
    shard = weight_map[name]
    path = os.path.join(model_dir, shard)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        f.seek(8 + n + meta["data_offsets"][0])
        raw = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
    return meta, raw


def _e4m3_to_float(u):
    s = ((u >> 7) & 1).astype(np.float32)
    e = ((u >> 3) & 0xF).astype(np.float32)
    m = (u & 0x7).astype(np.float32)
    out = np.where(e == 0, m / 8.0 * np.exp2(-6.0), (1.0 + m / 8.0) * np.exp2(e - 7.0))
    return np.where(s > 0, -out, out)


def _dequant_fp8_torch(torch, fp8_bytes, shape, scale_bytes, scale_shape):
    """FP8 E4M3 block-128 dequant on the GPU when one is present.

    On this box the dequant is the whole cost of the oracle (39 GB of FP8) and
    the 5090 does it two orders of magnitude faster than numpy."""
    if not torch.cuda.is_available():
        return None
    w = torch.frombuffer(bytearray(fp8_bytes), dtype=torch.uint8).view(shape).to("cuda")
    wf = w.view(torch.float8_e4m3fn).to(torch.float32)
    sc = torch.frombuffer(bytearray(scale_bytes), dtype=torch.float32).to("cuda")
    sc = sc.view(scale_shape).repeat_interleave(128, 0).repeat_interleave(128, 1)
    tiled = sc[: shape[0], : shape[1]]
    return (wf * tiled).to(torch.bfloat16).to("cpu").numpy()


def load_state_dict_bf16(model_dir):
    """Every tensor as a BF16 numpy view (FP8 dequantized with its scale tiles)."""
    import torch

    weight_map = _shard_map(model_dir)
    names = sorted(weight_map)
    state = {}
    for i, name in enumerate(names):
        meta, raw = _read_raw(model_dir, weight_map, name)
        if name.endswith("_scale_inv"):
            continue
        dtype = meta["dtype"]
        shape = meta["shape"]
        if dtype == "F8_E4M3":
            scale_name = name + "_scale_inv"
            smeta, sraw = _read_raw(model_dir, weight_map, scale_name)
            arr = _dequant_fp8_torch(torch, raw, shape, sraw, smeta["shape"])
            if arr is None:
                scales = np.frombuffer(sraw, dtype=np.float32).reshape(smeta["shape"])
                fp8 = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
                R, C = shape
                tile = np.repeat(np.repeat(scales, 128, axis=0)[:R], 128, axis=1)[:, :C]
                arr = (_e4m3_to_float(fp8) * tile).astype(np.float32)
            state[name] = arr.astype(np.float32)
            continue
        elif dtype == "BF16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
            arr = arr.reshape(shape)
        elif dtype == "F32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        else:
            raise SystemExit("unsupported dtype %s for %s" % (dtype, name))
        state[name] = arr.astype(np.float32)
        if (i + 1) % 5000 == 0:
            print("  dequantized %d/%d tensors" % (i + 1, len(names)), flush=True)
    return state


def build_model(model_dir, dtype):
    import torch

    # A stripped copy of the checkpoint: the config must not carry the FP8
    # quantization block, or transformers constructs its finegrained quantizer
    # (which needs CUDA _scaled_mm) instead of a plain BF16 module.
    tmp = tempfile.mkdtemp(prefix="gc4cfg_")
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"))
    for f in ("modeling_gigachat4.py", "configuration_gigachat4.py"):
        shutil.copy(os.path.join(model_dir, f), tmp)
    sys.path.insert(0, os.path.dirname(tmp))
    pkg = os.path.basename(tmp)
    open(os.path.join(tmp, "__init__.py"), "w").close()

    from importlib import import_module
    mod = import_module(pkg + ".modeling_gigachat4")
    conf = mod.GigaChat4Config(**{k: v for k, v in cfg.items()
                                  if k not in ("torch_dtype",)})
    conf._attn_implementation = "eager"
    torch.set_grad_enabled(False)
    model = mod.GigaChat4ForCausalLM(conf)
    model = model.to(dtype).eval()
    return model, tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ids-file", required=True,
                    help="whitespace-separated token ids (comma or newline ok)")
    ap.add_argument("--out", required=True, help="raw float32 logits of the last position")
    ap.add_argument("--engine-logits", default=None)
    ap.add_argument("--max-ids", type=int, default=24)
    ap.add_argument("--softcap", type=float, default=30.0)
    args = ap.parse_args()

    import torch

    text = open(args.ids_file).read().replace(",", " ")
    ids = [int(x) for x in text.split()][: args.max_ids]
    print("probe ids (%d): %s" % (len(ids), ids), flush=True)

    model, tmp = build_model(args.model, torch.bfloat16)
    print("dequantizing weights ...", flush=True)
    state = load_state_dict_bf16(args.model)
    print("loading state dict (%d tensors) ..." % len(state), flush=True)
    missing, unexpected = model.load_state_dict(
        {k: torch.from_numpy(v) for k, v in state.items()}, strict=False)
    print("missing:", list(missing)[:8], "unexpected:", list(unexpected)[:8],
          flush=True)
    del state

    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, logits_to_keep=1)
    logits = out.logits[0, -1].float().numpy()
    if args.softcap and args.softcap > 0:
        logits = np.tanh(logits / args.softcap) * args.softcap
    np.asarray(logits, dtype=np.float32).tofile(args.out)
    top = np.argsort(-logits)[:8]
    print("source top-8:", [(int(t), float(logits[t])) for t in top], flush=True)

    if args.engine_logits and os.path.exists(args.engine_logits):
        eng = np.fromfile(args.engine_logits, dtype=np.float32)
        n = min(len(eng), len(logits))
        e, s = eng[:n].astype(np.float64), logits[:n].astype(np.float64)
        cos = float(np.dot(e, s) / np.sqrt((e * e).sum() * (s * s).sum()))
        print("engine vs source: cos=%.5f  max|d|=%.3f  argmax %d vs %d" %
              (cos, float(np.abs(e - s).max()), int(np.argmax(eng)), int(np.argmax(logits))))
        etop = np.argsort(-eng)[:8]
        print("engine top-8:", [(int(t), float(eng[t])) for t in etop])
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
