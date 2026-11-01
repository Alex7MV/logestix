#!/usr/bin/env python3
"""CPU BF16 oracle for NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.

The published checkpoint is a ModelOpt 0.44.0rc5 `MIXED_PRECISION` tree
(W4A16_NVFP4 experts + lm_head, per-tensor FP8 Mamba in/out projections, BF16
attention) that the installed `transformers` cannot load (no modelopt
quantizer). This script undoes those encodings with torch and assembles a plain
BF16 `NemotronHForCausalLM`, so the *source* model can be executed as the
external oracle the parity gates need.

Decodings (mirroring libs/lsxcommon/src/nemotron_moe_nvfp4_dequant.cu):

  W4A16_NVFP4  `<base>.weight` U8 [rows, cols/2] nibbles (even element = LOW
               nibble), `<base>.weight_scale` F8_E4M3 [rows, cols/16],
               `<base>.weight_scale_2` F32 scalar:
                   w = e2m1(nibble) * e4m3(block_scale) * weight_scale_2
  per-tensor FP8  `F8_E4M3` weight + scalar F32 `weight_scale`:
                   w = e4m3(byte) * weight_scale[0]
  BF16 / F32    passed through (attention projections, router gate, norms,
               conv1d, A_log, D, dt_bias, embeddings)
  dropped       `*.input_scale`, `*.k_scale`, `*.v_scale` (activation / FP8
               KV-cache scales, not weight factors) and `mtp.*` (speculative
               ballast the engine stores but never applies).

Usage:
  python3 tools/nemotron_nvfp4_oracle.py build \
      --model /mnt/fast_storage/huggingface/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
      --out /mnt/fast_storage/tmp/nemotron_work/ref_bf16

  python3 tools/nemotron_nvfp4_oracle.py run \
      --ref /mnt/fast_storage/tmp/nemotron_work/ref_bf16 \
      --ids-file /path/to/prompt_ids.txt --max-new 32

`--ids-file` takes the prompt ids as whitespace-separated integers (so the same
ids can be fed to the engine); `--prompt` encodes text with the checkpoint's own
tokenizer instead. `--top5` prints the first-step logits.

Diagnostic flags (`--quant-attn`, `--quant-mamba`, `--quant-experts`,
`--quant-shared`, `--quant-head`, `--quant-router`) apply an offline NVFP4
round-trip to that weight class after loading, which is how the 2026-11-01
regression separated "the conversion is lossy" from "the runtime is wrong"
(`docs/OPEN_TASKS.md` §R13): with every linear quantized offline the reference
still made the tool call, so the converted weights were not the cause.
`REF_DTYPE=fp16` swaps the whole model to FP16 to test activation range.
"""
import argparse
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32)


def dequant_nvfp4(packed, block_scale, global_scale):
    """U8 [rows, cols/2] + F8_E4M3 [rows, cols/16] + F32 scalar -> BF16."""
    rows, half = packed.shape
    cols = half * 2
    b = packed.to(torch.int16)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    nib = torch.empty((rows, cols), dtype=torch.int64)
    nib[:, 0::2] = lo.to(torch.int64)
    nib[:, 1::2] = hi.to(torch.int64)
    vals = E2M1[nib]
    scale = block_scale.to(torch.float32)
    assert scale.shape == (rows, cols // 16), (scale.shape, rows, cols)
    scale = scale.repeat_interleave(16, dim=1)
    w = vals * scale * float(global_scale)
    return w.to(torch.bfloat16)


def dequant_fp8(weight, scale):
    return (weight.to(torch.float32) * float(scale)).to(torch.bfloat16)


def build(model_dir, out_dir):
    from transformers import NemotronHConfig

    cfg_d = json.load(open(os.path.join(model_dir, "config.json")))
    cfg_d.pop("quantization_config", None)
    cfg_d["num_nextn_predict_layers"] = 0
    cfg_d["mtp_layers_block_type"] = []
    cfg_d["use_mamba_kernels"] = False
    cfg_d["torch_dtype"] = "bfloat16"
    config = NemotronHConfig(**cfg_d)

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shard_files = sorted(set(weight_map.values()))
    handles = {}

    def get(name, shard=None):
        f = shard or weight_map[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(model_dir, f), framework="pt")
        return handles[f].get_tensor(name)

    def target(src):
        if src.startswith("backbone."):
            return "model." + src[len("backbone."):]
        return src

    sd = {}
    skipped = {"mtp": 0, "activation_scale": 0}
    # Dense tensors first, experts stacked afterwards.
    expert_prefixes = set()
    for name in weight_map:
        if name.startswith("mtp."):
            skipped["mtp"] += 1
            continue
        if name.endswith((".input_scale", ".k_scale", ".v_scale")):
            skipped["activation_scale"] += 1
            continue
        if ".experts." in name:
            continue
        if name.endswith((".weight_scale", ".weight_scale_2")):
            continue
        tgt = target(name)
        t = get(name)
        if t.dtype == torch.uint8:
            base = name[: -len(".weight")]
            packed = t
            block = get(base + ".weight_scale")
            glob = get(base + ".weight_scale_2")
            sd[tgt] = dequant_nvfp4(packed, block, glob.item())
        elif t.dtype == torch.float8_e4m3fn:
            scale = get(name[: -len(".weight")] + ".weight_scale")
            sd[tgt] = dequant_fp8(t, scale.item())
        else:
            sd[tgt] = t.to(torch.bfloat16) if t.dtype == torch.float32 else t
        if ".experts." in name and ".experts." not in tgt:
            expert_prefixes.add(name)

    # Routed experts: source is per-expert, the HF class wants [E, out, in].
    n_routed = int(cfg_d["n_routed_experts"])
    layers = [i for i, b in enumerate(cfg_d["layers_block_type"]) if b == "moe"]
    for l in layers:
        for proj in ("up_proj", "down_proj"):
            stacked = []
            for e in range(n_routed):
                base = f"backbone.layers.{l}.mixer.experts.{e}.{proj}"
                stacked.append(dequant_nvfp4(
                    get(base + ".weight"),
                    get(base + ".weight_scale"),
                    get(base + ".weight_scale_2").item()))
            sd[f"model.layers.{l}.mixer.experts.{proj}"] = torch.stack(stacked)
            del stacked
        print(f"layer {l}: experts stacked", flush=True)

    print("tensors:", len(sd), "skipped:", skipped, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    # Shard on the way out: a single 61 GB safetensors write needs the whole
    # dict resident alongside the mmap.
    pending, pending_bytes, shard = {}, 0, 0
    for k, v in sd.items():
        pending[k] = v
        pending_bytes += v.numel() * v.element_size()
        if pending_bytes > 6 << 30:
            save_file(pending, os.path.join(out_dir, f"ref-{shard:05d}.safetensors"))
            shard += 1
            pending, pending_bytes = {}, 0
    if pending:
        save_file(pending, os.path.join(out_dir, f"ref-{shard:05d}.safetensors"))
        shard += 1
    cfg_d["architectures"] = ["NemotronHForCausalLM"]
    json.dump(cfg_d, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "generation_config.json"):
        src = os.path.join(model_dir, f)
        if os.path.exists(src):
            os.system(f"cp -f '{src}' '{out_dir}/'")
    # The HF loader needs an index for a sharded tree.
    wm = {}
    for i in range(shard):
        fn = f"ref-{i:05d}.safetensors"
        with safe_open(os.path.join(out_dir, fn), framework="pt") as fh:
            for k in fh.keys():
                wm[k] = fn
    json.dump({"metadata": {"total_size": 0}, "weight_map": wm},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"))
    print("wrote", shard, "shards to", out_dir, flush=True)


def quant_nvfp4(w, block=16):
    """Offline NVFP4 round-trip (per-16 E4M3 block scale + per-tensor global).

    Mirrors the engine codec closely enough for a go/no-go bisection: the block
    scale is rounded to the E4M3 grid and the values to the nearest E2M1 code.
    """
    orig = w.shape
    rows = w.reshape(-1, w.shape[-1]).to(torch.float32)
    cols = rows.shape[1]
    pad = (-cols) % block
    if pad:
        rows = torch.nn.functional.pad(rows, (0, pad))
    blocks = rows.reshape(rows.shape[0], -1, block)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (amax / 6.0).to(torch.float8_e4m3fn).to(torch.float32)
    scale = scale.clamp_min(torch.finfo(torch.float8_e4m3fn).tiny)
    q = blocks / scale
    # nearest E2M1 code
    idx = (q.abs().unsqueeze(-1) - E2M1[:8].abs()).abs().argmin(dim=-1)
    mag = E2M1[idx.long()]
    val = torch.where(q < 0, -mag, mag) * scale
    out = val.reshape(rows.shape[0], -1)[:, :cols].to(torch.bfloat16)
    return out.reshape(orig)


def run(ref_dir, ids_file, max_new, prompt_text=None, quant_attn=False,
        quant_mamba=False, top5=False, quant_experts=False,
        quant_shared=False, quant_head=False, quant_router=False):
    from transformers import AutoTokenizer, NemotronHForCausalLM

    tok = AutoTokenizer.from_pretrained(ref_dir)
    if ids_file and os.path.exists(ids_file):
        ids = [int(x) for x in open(ids_file).read().split()]
    else:
        ids = tok.encode(prompt_text, add_special_tokens=False)
    print("prompt ids:", len(ids), flush=True)
    dtype = torch.float16 if os.environ.get("REF_DTYPE") == "fp16" else torch.bfloat16
    model = NemotronHForCausalLM.from_pretrained(
        ref_dir, dtype=dtype, low_cpu_mem_usage=True)
    model.eval()
    if (quant_attn or quant_mamba or quant_experts or quant_shared or quant_head
            or quant_router):
        cfg = model.config
        done = 0
        for i, block in enumerate(cfg.layers_block_type):
            if block == "full_attention" and quant_attn:
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    p = getattr(model.model.layers[i].mixer, proj).weight
                    p.data = quant_nvfp4(p.data)
                    done += 1
            if block == "linear_attention" and quant_mamba:
                for proj in ("in_proj", "out_proj"):
                    p = getattr(model.model.layers[i].mixer, proj).weight
                    p.data = quant_nvfp4(p.data)
                    done += 1
        if quant_experts or quant_shared:
            for i, block in enumerate(cfg.layers_block_type):
                if block != "moe":
                    continue
                if quant_experts:
                    ex = model.model.layers[i].mixer.experts
                    ex.up_proj.data = quant_nvfp4(ex.up_proj.data)
                    ex.down_proj.data = quant_nvfp4(ex.down_proj.data)
                    done += 2
                if quant_shared:
                    sh = model.model.layers[i].mixer.shared_experts
                    sh.up_proj.weight.data = quant_nvfp4(sh.up_proj.weight.data)
                    sh.down_proj.weight.data = quant_nvfp4(sh.down_proj.weight.data)
                    done += 2
        if quant_router:
            for i, block in enumerate(cfg.layers_block_type):
                if block != "moe":
                    continue
                g = model.model.layers[i].mixer.gate.weight
                g.data = g.data.to(torch.float16).to(torch.bfloat16)
                done += 1
        if quant_head:
            model.lm_head.weight.data = quant_nvfp4(model.lm_head.weight.data)
            done += 1
        print("offline-quantized tensors:", done, flush=True)
    if top5:
        with torch.no_grad():
            logits = model(torch.tensor([ids], dtype=torch.long)).logits[0, -1]
        v, i = logits.topk(5)
        print("top5:", [(int(a), round(float(b), 3)) for a, b in zip(i, v)],
              flush=True)
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.tensor([ids], dtype=torch.long),
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.pad_token_id or 0)
    gen = out[0, len(ids):].tolist()
    print("gen_ids:", " ".join(map(str, gen)), flush=True)
    print("gen_text:", tok.decode(gen), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--ref", required=True)
    r.add_argument("--ids-file")
    r.add_argument("--prompt")
    r.add_argument("--max-new", type=int, default=32)
    r.add_argument("--quant-attn", action="store_true")
    r.add_argument("--quant-mamba", action="store_true")
    r.add_argument("--top5", action="store_true")
    r.add_argument("--quant-experts", action="store_true")
    r.add_argument("--quant-shared", action="store_true")
    r.add_argument("--quant-head", action="store_true")
    r.add_argument("--quant-router", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.model, args.out)
    else:
        run(args.ref, args.ids_file, args.max_new, args.prompt,
            args.quant_attn, args.quant_mamba, args.top5,
            args.quant_experts, args.quant_shared, args.quant_head,
            args.quant_router)


if __name__ == "__main__":
    main()
