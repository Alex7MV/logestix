#!/usr/bin/env python3
"""The SOURCE-side oracle for Inkling-2-Mini golden captures.

Loads the release's own `inkling2_loop` implementation (transformers 5.24) on
the real 53 GB BF16 checkpoint, on the HOST CPU, and greedy-decodes prompts.
`tools/{fsi,code,det}_capture.py --type inkling-2-mini` reach this module
through their per-model override (`"loader": "inkling_loop"`), so the goldens
are model-defined: the release's own answer, captured by the release's own
code, never by the engine.

Why a wrapper instead of `lm.generate(...)`: the release's model keeps its
per-pass KV/conv state in cache layers it APPENDS to the DynamicCache on the
first forward, and `generate` re-enters that path on every step. The manual
greedy loop below is the one already validated against the release in the
stage-0 work (`tools/inkling_loop_transformers.py`, Paris probe), and it makes
the stop condition explicit -- this checkpoint's stream does NOT reach EOS
inside a 32-token window, so a capture that relied on EOS would truncate.

Requires the 5.24 reference environment on PYTHONPATH:

  PYTHONPATH=/mnt/fast_storage/tmp/inkling2-mini/tf/deps:/mnt/fast_storage/tmp/inkling2-mini/tf/x \
  python3 tools/fsi_capture.py --model /mnt/fast_storage/huggingface/Inkling-2-Mini \
      --type inkling-2-mini --out tests/fixtures/fsi/inkling-2-mini.json
"""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_VAULT = "/mnt/fast_storage/huggingface/Inkling-2-Mini"
# The reference's default sampling stop set: `eos_token_id` plus <|endoftext|>.
STOP_IDS = (200006, 199999)


def _require_reference_env():
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 24):
        raise SystemExit(
            "inkling_loop oracle: transformers %s is too old; the release's "
            "inkling2_loop model needs 5.24.x. Re-run with\n"
            "  PYTHONPATH=/mnt/fast_storage/tmp/inkling2-mini/tf/deps:"
            "/mnt/fast_storage/tmp/inkling2-mini/tf/x"
            % transformers.__version__)
    return transformers


class GreedyDecoder:
    """Minimal `.generate` shim over the release's model (see module docstring)."""

    def __init__(self, model, tok, stop_ids=STOP_IDS):
        self.model = model
        self.tok = tok
        self.stop_ids = tuple(i for i in stop_ids if i is not None)
        self.device = next(model.parameters()).device

    def parameters(self):
        """The wrapped model's parameters.

        The capture scripts print `next(lm.parameters()).device` to report where
        the source ran; this shim is what they hold, so it answers for the model
        instead of forcing a call-site special case.
        """
        return self.model.parameters()

    def generate(self, input_ids, do_sample=False, max_new_tokens=128,
                 eos_token_id=STOP_IDS, **kwargs):
        import torch

        if do_sample:
            raise SystemExit("inkling_loop oracle: only greedy decoding is "
                             "supported (a sampled golden is not reproducible)")
        stop = tuple(self.stop_ids) if eos_token_id is not None else ()
        gen = []
        with torch.no_grad():
            out = self.model(input_ids, use_cache=True)
            cache = out.cache
            logits = out.logits[0, -1]
            for _ in range(int(max_new_tokens)):
                nxt = int(logits.argmax())
                gen.append(nxt)
                if stop and nxt in stop:
                    break
                out = self.model(torch.tensor([[nxt]], dtype=torch.long),
                                 cache=cache, use_cache=True)
                cache = out.cache
                logits = out.logits[0, -1]
        new = torch.tensor([gen], dtype=torch.long)
        return torch.cat([input_ids, new], dim=1)


def load(model_dir, dtype=None, threads=64):
    """Loads the tokenizer and the text tower on CPU bf16.

    Returns `(tokenizer, GreedyDecoder)`, the pair the capture scripts use.
    """
    torch = __import__("torch")
    torch.set_num_threads(int(threads))
    _require_reference_env()
    from transformers import AutoTokenizer
    from transformers.models.inkling2_loop.configuration_inkling2_loop import (
        Inkling2LoopTextConfig)
    from transformers.models.inkling2_loop.modeling_inkling2_loop import (
        Inkling2LoopForCausalLM)

    vault = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(str(vault))
    if not getattr(tok, "chat_template", None):
        jinja = vault / "chat_template.jinja"
        if jinja.exists():
            tok.chat_template = jinja.read_text(encoding="utf-8")

    torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
    cfg_full = json.loads((vault / "config.json").read_text())
    cfg = Inkling2LoopTextConfig(**cfg_full["text_config"])
    with torch.device("meta"):
        model = Inkling2LoopForCausalLM(cfg)
    model = model.to_empty(device="cpu")
    model.config._attn_implementation = "eager"

    # Load through the release's own conversion mapping (renames + the NVFP4 /
    # MXFP8 expert dequant ops): the oracle must read the checkpoint the way the
    # release does, or it would be a second interpretation to compare against.
    from transformers.core_model_loading import convert_and_load_state_dict_in_model
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.modeling_utils import LoadStateDictConfig
    from safetensors import safe_open

    shards = sorted(vault.glob("model-*.safetensors"))
    sd = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    continue
                sd[k] = f.get_tensor(k)
    mapping = get_model_conversion_mapping(model, None, None)
    load_cfg = LoadStateDictConfig(dtype=torch_dtype, device_map={"": "cpu"},
                                   weight_mapping=mapping,
                                   dtype_plan=model._get_dtype_plan(torch_dtype))
    convert_and_load_state_dict_in_model(model, sd, load_cfg)
    # The reference keeps routers, the controllers' gates, convs and the query
    # banks in FP32; re-assert the plan by module-name substring (a bf16 router
    # weight fails F.linear against fp32 activations).
    strict = model._keep_in_fp32_modules_strict or []
    for name, p in model.named_parameters():
        if any(s in name for s in strict) and p.dtype != torch.float32:
            p.data = p.data.float()
    for name, b in model.named_buffers():
        if any(s in name for s in strict) and b.dtype != torch.float32:
            b.data = b.data.float()
    model.eval()
    return tok, GreedyDecoder(model, tok)


def capture_opts():
    """Capture facts this checkpoint's own stream dictates.

    `ignore_eos`: the framed stream does not reach EOS inside a 32-token window
    (measured: 32 tokens, eos_terminated=false), so the budget must be fixed or
    a longer answer line would be cut at whatever the sampler stopped on.
    """
    return {"ignore_eos": True}


if __name__ == "__main__":  # smoke test: load, frame, decode 4 tokens
    import time

    tok, lm = load(DEFAULT_VAULT)
    msgs = [{"role": "user", "content": "The capital of France is"}]
    framed = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=False)
    ids = tok.encode(framed, add_special_tokens=False)
    t0 = time.time()
    out = lm.generate(__import__("torch").tensor([ids]), max_new_tokens=4,
                      eos_token_id=None)
    gen = out[0][len(ids):].tolist()
    print("prompt ids:", ids)
    print("gen:", gen, "in %.1fs" % (time.time() - t0))
    print("text:", tok.decode(gen, skip_special_tokens=False))
