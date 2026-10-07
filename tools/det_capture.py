#!/usr/bin/env python3
"""DETSelfCheck reference capture (determinism + near-tie gate).

Captures the DET golden answers from an original safetensors model with HF
transformers. The logestix side only verifies the converted model against the
committed golden (libs/lsxcommon/src/det_selfcheck.cpp). Run the self-test with
no model present to validate the prompt set and the extraction mirror.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "det_prompts.json")

# Per-model capture overrides keyed by model type. Absent -> generic defaults.
OVERRIDES = {
    # ornith-1.5-35b-a3b is served with a leading system turn (its model
    # class injects one when the caller omits it); without it the safety
    # alignment refuses the FSI-02 financial prompt. Thinking stays off so
    # the answer is the short direct continuation.
    "ornith-1.5-35b-a3b": {"system": "You are a helpful assistant."},
    "granite-4.2-30b": {"enable_thinking": True, "low_effort": True},
    "qwen3.8-flash-next": {"loader": "qwen4_exp"},
    # qwen3.8-27b is a Qwen3_5 VLM checkpoint; the text language model is
    # loaded directly (loader remaps the model.language_model prefix) and
    # sharded GPU+CPU because 27B bf16 does not fit 32 GiB VRAM.
    "qwen3.8-27b": {"loader": "qwen3_5", "device_map": "auto"},
    # qwen3.5-9b is a Qwen3_5 VLM checkpoint; bf16 fits in 32 GiB so the
    # accelerate device map keeps every layer on the GPU.
    "qwen3.5-9b": {"loader": "qwen3_5", "device_map": "auto"},
    # deepseek-v4.1-flash ships no HF chat template; frame with the release's
    # own encoding/encoding.py, in the reasoning model's default thinking mode.
    "deepseek-v4.1-flash": {"framing": "deepseek_v41",
                            "thinking_mode": "thinking"},
}
GENERIC = {"trust_remote_code": True, "max_new_tokens": 512,
           "device": None, "dtype": None, "enable_thinking": False,
           "device_map": None, "loader": None, "system": None}

_STRESS_RUNS = 3

_QWEN4_NGRAM_ROWS = 2500012  # rows per PLE n-gram shard (128 shards)


def _load_qwen4_exp(model_path):
    """Load an experimental qwen4_exp checkpoint with the installed
    transformers (config allowlist + cache mapping patched, PLE shards fused).
    Same helper as fsi_capture.py / code_capture.py."""
    import glob

    import torch
    from safetensors.torch import load_file
    import transformers.configuration_utils as cu
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    from transformers import cache_utils as cu_cache

    cu.ALLOWED_ATTN_LAYER_TYPES = tuple(cu.ALLOWED_ATTN_LAYER_TYPES) + (
        "qwen_sparse_attention",)
    cu_cache.DYNAMIC_LAYER_TYPE_MAPPING.setdefault(
        "qwen_sparse_attention", cu_cache.DynamicIndexedLayer)
    cu_cache.STATIC_LAYER_TYPE_MAPPING.setdefault(
        "qwen_sparse_attention", cu_cache.StaticIndexedLayer)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    raw = json.load(open(os.path.join(model_path, "config.json"),
                        encoding="utf-8"))["text_config"]
    raw["mtp_num_hidden_layers"] = 0
    tc = Qwen4ExpTextConfig(**raw)
    lm = Qwen4ExpForCausalLM(tc)
    lm.eval()
    params = dict(lm.named_parameters())
    bufs = dict(lm.named_buffers())
    ngram = params.get(
        "model.layers.1.ple.ple_embedding.ngram_embedding.weight")
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        for key, tensor in load_file(shard).items():
            if key.startswith("model.visual."):
                continue
            if "ngram_embedding.shard_" in key and ngram is not None:
                idx = int(re.search(r"shard_(\d+)\.weight", key).group(1))
                off = idx * _QWEN4_NGRAM_ROWS
                with torch.no_grad():
                    ngram.data[off:off + tensor.shape[0]].copy_(tensor)
                continue
            name = key.replace("model.language_model.", "model.")
            if name in params:
                with torch.no_grad():
                    params[name].data.copy_(tensor)
            elif name in bufs:
                with torch.no_grad():
                    bufs[name].copy_(tensor)
    torch.set_default_dtype(prev_dtype)
    return lm


def _remap_qwen3_5_key(key):
    """Map a Qwen3.8-27B VLM checkpoint key to a text-only
    Qwen3_5ForCausalLM key. Returns None for vision/MTP tensors the text model
    does not own."""
    if key.startswith("model.visual.") or key.startswith("mtp."):
        return None
    return key.replace("model.language_model.", "model.")


def _load_qwen3_5(model_path):
    """Load the Qwen3.8-27B VLM checkpoint as a text-only causal LM.

    A plain AutoModelForCausalLM load of this checkpoint is wrong: `qwen3_5`
    maps to Qwen3_5ForCausalLM, whose weights are prefixed `model.`, while the
    checkpoint stores them under `model.language_model.`. Without the remap the
    language weights are silently dropped and the model is random.
    """
    import glob

    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    raw = json.load(open(os.path.join(model_path, "config.json"),
                         encoding="utf-8"))["text_config"]
    raw = dict(raw)
    raw["mtp_num_hidden_layers"] = 0
    tc = Qwen3_5TextConfig(**raw)

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    lm = Qwen3_5ForCausalLM(tc)
    lm.eval()
    params = dict(lm.named_parameters())
    bufs = dict(lm.named_buffers())
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        for key, tensor in load_file(shard).items():
            name = _remap_qwen3_5_key(key)
            if name is None:
                continue
            if name in params:
                with torch.no_grad():
                    params[name].data.copy_(tensor)
            elif name in bufs:
                with torch.no_grad():
                    bufs[name].copy_(tensor)
    torch.set_default_dtype(prev_dtype)
    return lm


def _dispatch_auto(lm, no_split):
    """Shard a model too large for VRAM across GPU + CPU (accelerate)."""
    from accelerate import dispatch_model, infer_auto_device_map

    dm = infer_auto_device_map(
        lm, max_memory={0: "28GiB", "cpu": "640GiB"},
        no_split_module_classes=no_split)
    return dispatch_model(lm, device_map=dm)


def load_prompts():
    with open(PROMPTS, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["format"] == "logestix-det-prompts", doc["format"]
    return doc["prompts"]


def _unfence(raw):
    s = raw.strip()
    open_at = s.find("```")
    if open_at == -1:
        return s
    begin = open_at + 3
    nl = s.find("\n", begin)
    if nl != -1 and re.fullmatch(r"[A-Za-z0-9_+-]*", s[begin:nl]):
        begin = nl + 1
    close = s.find("```", begin)
    inner = s[begin:] if close == -1 else s[begin:close]
    return inner.strip()


def _preprocess(raw):
    text = _unfence(raw)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text


def _is_word_char(c):
    return c.isalnum() or c == "_"


def _extract_token(text):
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if not ("A" <= c <= "Z"):
            i += 1
            continue
        if i > 0 and _is_word_char(text[i - 1]):
            i += 1
            continue
        j = i + 1
        while j < n and (("A" <= text[j] <= "Z") or ("0" <= text[j] <= "9")
                         or text[j] == "_"):
            j += 1
        if j - i < 2:
            i = j
            continue
        if j < n and _is_word_char(text[j]):
            i = j
            continue
        return text[i:j]
    return ""


def extract(kind, raw):
    text = _preprocess(raw)
    if kind == "token":
        return _extract_token(text)
    raise ValueError("unknown kind: %r" % (kind,))


def selftest():
    prompts = load_prompts()
    assert [p["id"] for p in prompts] == ["DET-TEST-01", "DET-TEST-02",
                                          "DET-TEST-03"]
    for p in prompts:
        got = extract(p["kind"], p["expected"])
        assert got == p["expected"], (p["id"], got, p["expected"])
    assert extract("token", "The answer is CATASTROPHIC.") == "CATASTROPHIC"
    assert extract("token", "lowercase only") == ""
    # qwen3.8-27b: the VLM -> text-only key remap (vision/MTP dropped).
    assert _remap_qwen3_5_key("model.language_model.embed_tokens.weight") \
        == "model.embed_tokens.weight", "language prefix remap"
    assert _remap_qwen3_5_key("lm_head.weight") == "lm_head.weight"
    assert _remap_qwen3_5_key("model.visual.blocks.0.weight") is None
    assert _remap_qwen3_5_key("mtp.fc.weight") is None
    # qwen3.5-9b is the same Qwen3_5 VLM checkpoint shape as the 27B.
    assert OVERRIDES.get("qwen3.5-9b", {}).get("loader") == "qwen3_5", \
        "qwen3.5-9b must use the qwen3_5 text-only loader"
    print("det selftest OK: token templates valid")


def _frame_messages(opts, body):
    """Chat messages for a prompt: an optional leading system turn (some
    models are only served with one and refuse otherwise) plus the user turn."""
    messages = []
    if opts.get("system"):
        messages.append({"role": "system", "content": opts["system"]})
    messages.append({"role": "user", "content": body})
    return messages


def frame_prompt(tok, body, opts):
    """Apply the model's chat template when present, else the raw body."""
    prompt_text = body
    if getattr(tok, "chat_template", None):
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if opts.get("enable_thinking") is not None:
            template_kwargs["enable_thinking"] = opts["enable_thinking"]
        if opts.get("low_effort"):
            template_kwargs["low_effort"] = True
        messages = _frame_messages(opts, body)
        try:
            prompt_text = tok.apply_chat_template(
                messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            prompt_text = tok.apply_chat_template(
                messages, **template_kwargs)
        except Exception as exc:  # noqa: BLE001
            print("chat_template failed, using raw prompt:", exc,
                  file=sys.stderr)
            prompt_text = body
    return prompt_text


def _frame_deepseek_v41(model_dir, body, thinking_mode):
    """Frame a prompt with the DeepSeek-V4.1 release's own encoding stack
    (encoding/encoding.py); this model ships no HF chat template."""
    import importlib.util

    enc_path = os.path.join(model_dir, "encoding", "encoding.py")
    spec = importlib.util.spec_from_file_location("dsv41_encoding", enc_path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    return enc.encode_messages([{"role": "user", "content": body}],
                               thinking_mode=thinking_mode)


def frame(model, tok, body, opts):
    """Frame a prompt: per-model custom framing when configured, else the
    tokenizer's chat template / raw body."""
    if opts.get("framing") == "deepseek_v41":
        return _frame_deepseek_v41(model, body,
                                   opts.get("thinking_mode", "thinking"))
    return frame_prompt(tok, body, opts)


def _load_tokenizer(model, opts):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=opts["trust_remote_code"])
    if not getattr(tok, "chat_template", None):
        jinja = os.path.join(model, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja, encoding="utf-8") as fh:
                tok.chat_template = fh.read()
    return tok


def template_golden(model, model_type, out, max_new_tokens):
    """Template/API-derived golden for a source that cannot run locally: frame
    each prompt with the model's own chat template + tokenizer and record the
    prompt's Expected Template as `expected`. The `source` field marks the
    deviation."""
    import transformers

    opts = dict(GENERIC)
    opts.update(OVERRIDES.get(model_type, {}))
    opts["max_new_tokens"] = max_new_tokens
    tok = _load_tokenizer(model, opts)

    cases = []
    for p in load_prompts():
        prompt_text = frame(model, tok, p["body"], opts)
        ids = tok(prompt_text).input_ids
        got = extract(p["kind"], p["expected"])
        assert got == p["expected"], (p["id"], got, p["expected"])
        cases.append({"id": p["id"], "kind": p["kind"],
                      "input_ids": list(ids),
                      "max_new_tokens": max_new_tokens,
                      "stress_runs": _STRESS_RUNS, "body": p["body"],
                      "expected": p["expected"], "raw": ""})

    golden = {"format": "logestix-det-golden", "version": 1,
              "model_type": model_type,
              "source": ("prompt-template (encoding/encoding.py %s framing; "
                         "template-defined canonical expected)"
                         % opts.get("thinking_mode", "thinking")
                         if opts.get("framing") == "deepseek_v41"
                         else "prompt-template (tokenizer.json + "
                              "chat_template.jinja; template-defined "
                              "canonical expected)"),
              "runtime": ("encoding/encoding.py + tokenizer.json "
                          "(transformers %s)" % transformers.__version__
                          if opts.get("framing") == "deepseek_v41"
                          else "tokenizer.json + chat_template.jinja "
                               "(transformers %s)" % transformers.__version__),
              "prompts": cases}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    print("wrote", out)


def capture(model, model_type, out, max_new_tokens, device_map=None):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    opts = dict(GENERIC)
    opts.update(OVERRIDES.get(model_type, {}))
    opts["max_new_tokens"] = max_new_tokens
    if device_map:
        opts["device_map"] = device_map
    tok = _load_tokenizer(model, opts)

    load_kwargs = {"trust_remote_code": opts["trust_remote_code"]}
    if opts.get("dtype"):
        load_kwargs["dtype"] = opts["dtype"]
    if opts.get("device_map"):
        load_kwargs["device_map"] = opts["device_map"]
    if opts.get("loader") == "qwen3_5":
        lm = _load_qwen3_5(model)
        if opts.get("device_map"):
            lm = _dispatch_auto(lm, ["Qwen3_5DecoderLayer"])
    elif opts.get("loader") == "qwen4_exp":
        lm = _load_qwen4_exp(model)
    else:
        lm = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        lm.eval()

    device = opts.get("device")
    if (device is None and not opts.get("device_map")
            and torch.cuda.is_available()):
        nbytes = sum(p.numel() * p.element_size() for p in lm.parameters())
        free, _total = torch.cuda.mem_get_info()
        if nbytes < int(free * 0.9):
            device = "cuda"
    if device:
        lm = lm.to(device)
    print("device:", next(lm.parameters()).device)

    cases = []
    for p in load_prompts():
        prompt_text = frame(model, tok, p["body"], opts)
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(lm.device)
        with torch.no_grad():
            out_ids = lm.generate(ids, do_sample=False,
                                  max_new_tokens=max_new_tokens)
        gen = out_ids[0][ids.shape[1]:].tolist()
        raw = tok.decode(gen, skip_special_tokens=True)
        canonical = extract(p["kind"], p["expected"])
        assert canonical == p["expected"], (p["id"], canonical, p["expected"])
        got = extract(p["kind"], raw)
        if not got:
            print("WARNING %s: source emitted no canonical token (raw=%r); "
                  "using the prompt template" % (p["id"], raw), file=sys.stderr)
        elif got != canonical:
            print("WARNING %s: source answer differs from the prompt template: "
                  "%r != %r" % (p["id"], got, canonical), file=sys.stderr)
        cases.append({"id": p["id"], "kind": p["kind"],
                      "input_ids": ids[0].tolist(),
                      "max_new_tokens": max_new_tokens,
                      "stress_runs": _STRESS_RUNS, "body": p["body"],
                      "expected": canonical, "raw": raw})

    golden = {"format": "logestix-det-golden", "version": 1,
              "model_type": model_type, "source": model,
              "runtime": "transformers %s / torch %s"
                         % (transformers.__version__, torch.__version__),
              "prompts": cases}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--type")
    ap.add_argument("--out")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--template-golden", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model and args.type and args.out):
        ap.error("--model, --type and --out are required unless --selftest")
    if args.template_golden:
        template_golden(args.model, args.type, args.out, args.max_new_tokens)
    else:
        capture(args.model, args.type, args.out, args.max_new_tokens,
                args.device_map)


if __name__ == "__main__":
    main()
