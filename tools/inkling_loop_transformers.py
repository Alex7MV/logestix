#!/usr/bin/env python3
"""Runs the RELEASE's own transformers 5.24 `inkling2_loop` implementation on
the REAL Inkling-2-Mini weights (CPU, bf16), greedy-decoding a prompt.

This is the SOURCE side of the parity contract: the engine's goldens and the
Paris probe are measured against what this script produces, never against the
engine itself. The checkpoint's NVFP4/MXFP8 expert slabs are dequantized by
the release's own conversion ops (Inkling2LoopDequantizeExperts + Interleave),
so this reference does not read the engine's artifact interpretation.

Usage:
  PYTHONPATH=/mnt/fast_storage/tmp/inkling2-mini/tf/deps:/mnt/fast_storage/tmp/inkling2-mini/tf/x \
      python3 tools/inkling_loop_transformers.py \
      --vault /mnt/fast_storage/lsx_model/thinkingmachines/Inkling-2-Mini \
      --prompt "The capital of France is" --max-gen 32 \
      --out /mnt/fast_storage/tmp/inkling2-mini/probes/paris.json

--raw feeds the prompt unframed (knowledge probes); the default frames it as
the native user turn exactly like chat_template.jinja (effort line included).
--ids feeds comma-separated token ids directly (no tokenizer).
"""
import argparse
import json
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default="/mnt/fast_storage/lsx_model/thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--ids", default=None, help="comma separated token ids (overrides --prompt)")
    ap.add_argument("--raw", action="store_true", help="do not frame the prompt")
    ap.add_argument("--max-gen", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-logits", action="store_true",
                    help="save the LAST step's full logits as .npy next to --out")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer
    from transformers.models.inkling2_loop.configuration_inkling2_loop import (
        Inkling2LoopTextConfig)
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForCausalLM)

    vault = Path(args.vault)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(vault))

    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
        framed = None
    elif args.raw:
        ids = tok.encode(args.prompt, add_special_tokens=False)
        framed = None
    else:
        # The native frame: tool-less chat_template.jinja render of one user
        # message. The template appends NO assistant opener; the model writes
        # its own (<|message_model|><|content_thinking|>...).
        msgs = [{"role": "user", "content": args.prompt}]
        framed = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok.encode(framed, add_special_tokens=False)
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)

    cfg_full = json.loads((vault / "config.json").read_text())
    cfg = Inkling2LoopTextConfig(**cfg_full["text_config"])
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"building model (dtype={dtype}) ...", flush=True)
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(cfg)
    model = model.to_empty(device="cpu")
    model.config._attn_implementation = "eager"

    # Load via the release's own conversion mapping (renames + dequant ops).
    from transformers.core_model_loading import convert_and_load_state_dict_in_model
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.modeling_utils import LoadStateDictConfig
    sd = {}
    from safetensors import safe_open
    with safe_open(str(vault / "model-00001-of-00001.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                continue
            sd[k] = f.get_tensor(k)
    print(f"source tensors: {len(sd)}; converting ...", flush=True)
    mapping = get_model_conversion_mapping(model, None, None)
    load_cfg = LoadStateDictConfig(dtype=dtype, device_map={"": "cpu"},
                                   weight_mapping=mapping,
                                   dtype_plan=model._get_dtype_plan(dtype))
    convert_and_load_state_dict_in_model(model, sd, load_cfg)
    # The reference keeps routers, controllers' gates, convs and query banks in
    # FP32 (_keep_in_fp32_modules_strict). The loader applies the plan during
    # conversion; re-assert it by module-name substring so nothing slips to
    # bf16 (a bf16 router weight fails F.linear against fp32 activations).
    strict = model._keep_in_fp32_modules_strict or []
    def want_fp32(name):
        return any(s in name for s in strict)
    for name, p in model.named_parameters():
        if want_fp32(name) and p.dtype != torch.float32:
            p.data = p.data.float()
    for name, b in model.named_buffers():
        if want_fp32(name) and b.dtype != torch.float32:
            b.data = b.data.float()
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)

    input_ids = torch.tensor([ids], dtype=torch.long)
    gen = []
    last_logits = None
    step_times = []
    with torch.no_grad():
        out = model(input_ids, use_cache=True)
        cache = out.cache
        logits = out.logits[0, -1]
        for step in range(args.max_gen):
            t1 = time.time()
            nxt = int(logits.argmax())
            gen.append(nxt)
            last_logits = logits.float()
            if nxt in (cfg.eos_token_id, 199999):
                break
            out = model(torch.tensor([[nxt]], dtype=torch.long), cache=cache, use_cache=True)
            cache = out.cache
            logits = out.logits[0, -1]
            step_times.append(time.time() - t1)
            if step < 3 or step % 8 == 0:
                print(f"step {step}: {nxt} {tok.decode([nxt])!r} "
                      f"({step_times[-1]:.2f}s)", flush=True)

    text = tok.decode(gen, skip_special_tokens=False)
    result = {
        "prompt": args.prompt,
        "framed": framed,
        "prompt_ids": ids,
        "generated_ids": gen,
        "generated_text": text,
        "dtype": args.dtype,
        "max_gen": args.max_gen,
        "eos_terminated": bool(gen and gen[-1] in (cfg.eos_token_id, 199999)),
        "seconds": time.time() - t0,
        "step_seconds": step_times,
        "source": str(vault),
        "reference": "transformers 5.24.0 models/inkling2_loop (release code, real weights)",
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=1))
    if args.save_logits and last_logits is not None:
        import numpy as np
        np.save(outp.with_suffix(".logits.npy"), last_logits.numpy())
    print(f"generated: {text!r}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()