#!/usr/bin/env python3
"""CODESelfCheck reference capture (separate from SelfCheck and FSISelfCheck).

Captures the CODE (compiler syntactic strictness) golden answers from an
original safetensors model with HF transformers. The logestix side only verifies
the converted model against the committed golden
(libs/lsxcommon/src/code_selfcheck.cpp). Run the self-test with no model present
to validate the prompt set and the extraction mirror.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "code_prompts.json")

# Per-model capture overrides keyed by model type. Absent -> generic defaults.
OVERRIDES = {
    # Mistral-Small-4-119B ships the Mistral "consolidated" layout: params.json
    # (no config.json, no model_type) plus consolidated-*.safetensors with
    # NVFP4-packed MoE experts. transformers cannot construct it, so the golden
    # is template-derived and NO golden is committed (declared debt in
    # test_model_suite_guard). To capture it for real, pass one of: an HF-format
    # bf16 export of the same model (no change needed here - the auto loader
    # handles it), or the vendor stack (mistral-inference + mistral_common),
    # which replaces the template-only capture with a `loader: "mistral"` branch
    # on the qwen4_exp/qwen3_5 pattern and drops this override.
    "mistral-small-4-119b": {
        "unrunnable": "consolidated layout (params.json, no config.json) + "
                      "NVFP4-packed experts",
        # Frame exactly as the engine does (RenderText: BOS + [INST]...[/INST],
        # no vendor system turn). The checkpoint's default system prompt tells
        # the model to deflect every data-shaped request to "use tools", which
        # suppresses all three answer gates regardless of the weights.
        "chat_template": "{%- for m in messages %}{%- if m.role == 'user' %}{{ bos_token }}[INST] {{ m.content }} [/INST]{%- endif %}{%- endfor %}",
    },
    # ornith-1.5-35b-a3b is served with a leading system turn (its model
    # class injects one when the caller omits it); without it the safety
    # alignment refuses the FSI-02 financial prompt. Thinking stays off so
    # the answer is the short direct continuation.
    "ornith-1.5-35b-a3b": {"system": "You are a helpful assistant."},
    # granite-4.2 is a reasoning model; low-effort thinking is the reliable
    # mode for short exact-token answers. Its chat template lives in
    # chat_template.jinja (not tokenizer_config).
    "granite-4.2-30b": {"enable_thinking": True, "low_effort": True},
    # qwen3.8-flash-next is an experimental "qwen4_exp" arch (same patches as
    # the FSI capture): layer-type allowlist + KV-cache mapping + PLE shards.
    "qwen3.8-flash-next": {"loader": "qwen4_exp"},
    # qwen3.8-27b is a Qwen3_5 VLM checkpoint; the text language model is
    # loaded directly (loader remaps the model.language_model prefix) and
    # sharded GPU+CPU because 27B bf16 does not fit 32 GiB VRAM.
    "qwen3.8-27b": {"loader": "qwen3_5", "device_map": "auto"},
    # qwen3.5-9b is a Qwen3_5 VLM checkpoint; bf16 fits in 32 GiB so the
    # accelerate device map keeps every layer on the GPU.
    "qwen3.5-9b": {"loader": "qwen3_5", "device_map": "auto"},
    # DeepSeek-V4-Pro-0813 ships no HF chat template; frame with its own
    # encoding/encoding_dsv4.py in chat mode. Template-only: the TP-4 FP8
    # source runner cannot execute on one 32 GB card.
    "deepseek-v4-pro-0813": {
        "framing": "deepseek_v4", "thinking_mode": "chat",
        "unrunnable": "TP-4 FP8 reference (inference/generate.py, MP=4) over an 892.7 GB checkpoint; this 32 GB card cannot execute it",
    },
}

# MiniMax-M3 (428B / 796 GB BF16, 60-layer MSA + MoE): the source model does
# not fit this card and no quantized reference is published, so the golden is
# the prompt's Expected Template (template-only mode) framed with the
# checkpoint's own chat template and thinking disabled (the engine's own
# generation prefix). Recorded deviation, not a silent one.
OVERRIDES["minimax-m3"] = {
    "template_kwargs": {"thinking_mode": "disabled"},
    "unrunnable": "428B/796 GB BF16 checkpoint: no 32 GB-card execution path "
                  "(vendor stack is TP-8 over the full BF16 tree)",
}
def _apply_template_kwargs(template_kwargs, opts):
    """Per-model extra chat-template variables (e.g. MiniMax's thinking_mode)."""
    extra = opts.get("template_kwargs") or {}
    for k, v in extra.items():
        template_kwargs[k] = v
    return template_kwargs

GENERIC = {"trust_remote_code": True, "max_new_tokens": 512,
           "device": None, "dtype": None, "enable_thinking": False,
           "device_map": None, "loader": None, "system": None,
           "chat_template": None}

_QWEN4_NGRAM_ROWS = 2500012  # rows per PLE n-gram shard (128 shards)


def _load_qwen4_exp(model_path):
    """Load an experimental qwen4_exp checkpoint with the installed
    transformers (config allowlist + cache mapping patched, PLE shards fused)."""
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
    assert doc["format"] == "logestix-code-prompts", doc["format"]
    return doc["prompts"]


def normalize(s):
    out = "".join(ch for ch in s if not ch.isspace())
    return out.rstrip(".,;")


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


def _extract_json(text, anchor):
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    esc = False
    end = None
    i = start
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        i += 1
    if end is None:
        return ""
    sub = text[start:end]
    out = []
    in_str = False
    esc = False
    for c in sub:
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
                out.append(c)
            elif not c.isspace():
                out.append(c)
    res = "".join(out)
    if anchor and anchor not in res:
        return ""
    return res


def _find_assignment_anchor(s, anchor):
    pos = s.find(anchor)
    while pos != -1:
        i = pos + len(anchor)
        while i < len(s) and s[i].isspace():
            i += 1
        if i < len(s) and s[i] == "=":
            return pos
        pos = s.find(anchor, pos + 1)
    return -1


def _extract_assignment(text, anchor):
    if not anchor:
        return ""
    a = _find_assignment_anchor(text, anchor)
    if a == -1:
        return ""
    line_start = text.rfind("\n", 0, a) + 1
    line_end = text.find("\n", a)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    i = a - line_start
    start = i
    depth = 0
    first = True
    prev_op = True
    while i < len(line):
        c = line[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            depth += 1
            prev_op = True
            first = False
            i += 1
            continue
        if c == ")":
            if depth == 0:
                break
            depth -= 1
            prev_op = False
            first = False
            i += 1
            continue
        if c in "+-*/=":
            prev_op = True
            first = False
            i += 1
            continue
        if c.isalnum() or c in "_.":
            if (not first) and depth == 0 and (not prev_op):
                break
            while i < len(line) and (line[i].isalnum() or line[i] in "_."):
                i += 1
            prev_op = False
            first = False
            continue
        break
    if i <= start:
        return ""
    return normalize(line[start:i])


def _is_regex_token(c):
    return c.isalnum() or c in "\\[]{}()|.*+?^$/-"


def _extract_regex(text, anchor):
    if not anchor:
        return ""
    a = text.find(anchor)
    if a == -1:
        return ""
    line_start = text.rfind("\n", 0, a) + 1
    line_end = text.find("\n", a)
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end]
    l = a - line_start
    r = l + len(anchor)
    while r < len(line) and _is_regex_token(line[r]):
        r += 1
    while l > 0 and _is_regex_token(line[l - 1]):
        l -= 1
    if r <= l:
        return ""
    return normalize(line[l:r])


def extract(kind, anchor, raw):
    text = _preprocess(raw)
    if kind == "json":
        return _extract_json(text, anchor)
    if kind == "assignment":
        return _extract_assignment(text, anchor)
    if kind == "regex":
        return _extract_regex(text, anchor)
    raise ValueError("unknown kind: %r" % (kind,))


def selftest():
    prompts = load_prompts()
    assert [p["id"] for p in prompts] == ["CODE-TEST-01", "CODE-TEST-02",
                                          "CODE-TEST-03"]
    for p in prompts:
        got = extract(p["kind"], p["anchor"], p["expected"])
        assert got == p["expected"], (p["id"], got, p["expected"])
    # CODE-01 is valid minified JSON with the required nesting.
    obj = json.loads(prompts[0]["expected"])
    assert obj == {"root": {"nodes": {"leaf": True}}}, obj
    # CODE-02 is the midpoint formula for the stated boundary split.
    assert eval("l+((h-l)/2)", {"__builtins__": {}}, {"l": 0, "h": 8}) == 4
    # CODE-03 compiles and matches uppercase hex only, bounded.
    pat = re.compile(prompts[2]["expected"])
    assert pat.search("ABCD") is not None
    assert pat.search("abcd") is None
    assert pat.search("ABCDE") is None
    # qwen3.8-27b: the VLM -> text-only key remap (vision/MTP dropped).
    assert _remap_qwen3_5_key("model.language_model.embed_tokens.weight") \
        == "model.embed_tokens.weight", "language prefix remap"
    assert _remap_qwen3_5_key("lm_head.weight") == "lm_head.weight"
    assert _remap_qwen3_5_key("model.visual.blocks.0.weight") is None
    assert _remap_qwen3_5_key("mtp.fc.weight") is None
    # qwen3.5-9b is the same Qwen3_5 VLM checkpoint shape as the 27B.
    assert OVERRIDES.get("qwen3.5-9b", {}).get("loader") == "qwen3_5", \
        "qwen3.5-9b must use the qwen3_5 text-only loader"
    print("code selftest OK: json/assignment/regex templates valid")


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
        _apply_template_kwargs(template_kwargs, opts)
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


def _frame_deepseek_v4(model_dir, body, thinking_mode):
    import importlib.util

    enc_path = os.path.join(model_dir, "encoding", "encoding_dsv4.py")
    spec = importlib.util.spec_from_file_location("dsv4_encoding", enc_path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    return enc.encode_messages([{"role": "user", "content": body}],
                               thinking_mode=thinking_mode)


def frame(model, tok, body, opts):
    if opts.get("framing") == "deepseek_v4":
        return _frame_deepseek_v4(model, body,
                                  opts.get("thinking_mode", "chat"))
    return frame_prompt(tok, body, opts)


def template_golden(model, model_type, out, max_new_tokens):
    """Template/API-derived golden for a source that cannot run locally: frame
    each prompt with the model's own chat template + tokenizer and record the
    prompt's Expected Template as `expected` (same stopgap as the FSI
    GLM-5.3-Flash golden). The `source` field marks the deviation."""
    import transformers
    from transformers import AutoTokenizer

    opts = dict(GENERIC)
    opts.update(OVERRIDES.get(model_type, {}))
    opts["max_new_tokens"] = max_new_tokens
    prompts = load_prompts()

    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=opts["trust_remote_code"])
    if not getattr(tok, "chat_template", None):
        jinja = os.path.join(model, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja, encoding="utf-8") as fh:
                tok.chat_template = fh.read()
    if opts.get("chat_template"):
        tok.chat_template = opts["chat_template"]

    cases = []
    for p in prompts:
        prompt_text = frame(model, tok, p["body"], opts)
        ids = tok(prompt_text).input_ids
        got = extract(p["kind"], p["anchor"], p["expected"])
        assert got == p["expected"], (p["id"], got, p["expected"])
        cases.append({"id": p["id"], "kind": p["kind"], "anchor": p["anchor"],
                      "input_ids": list(ids), "max_new_tokens": max_new_tokens,
                      "body": p["body"], "expected": p["expected"], "raw": ""})

    golden = {"format": "logestix-code-golden", "version": 1,
              "model_type": model_type,
              "source": "prompt-template (source not runnable locally: %s)"
                        % (opts.get("unrunnable") or
                           "custom architecture/layout"),
              "runtime": ("encoding/encoding_dsv4.py + tokenizer.json "
                          "(transformers %s)" % transformers.__version__)
                         if opts.get("framing") == "deepseek_v4" else
                         ("tokenizer.json + chat_template.jinja "
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    opts = dict(GENERIC)
    opts.update(OVERRIDES.get(model_type, {}))
    opts["max_new_tokens"] = max_new_tokens
    if device_map:
        opts["device_map"] = device_map
    prompts = load_prompts()

    tok = AutoTokenizer.from_pretrained(
        model, trust_remote_code=opts["trust_remote_code"])
    if not getattr(tok, "chat_template", None):
        jinja = os.path.join(model, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja, encoding="utf-8") as fh:
                tok.chat_template = fh.read()

    if opts.get("loader") == "qwen3_5":
        lm = _load_qwen3_5(model)
        if opts.get("device_map"):
            lm = _dispatch_auto(lm, ["Qwen3_5DecoderLayer"])
    elif opts.get("loader") == "qwen4_exp":
        lm = _load_qwen4_exp(model)
    else:
        load_kwargs = {"trust_remote_code": opts["trust_remote_code"]}
        if opts.get("dtype"):
            load_kwargs["dtype"] = opts["dtype"]
        if opts.get("device_map"):
            load_kwargs["device_map"] = opts["device_map"]
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
    for p in prompts:
        prompt_text = frame_prompt(tok, p["body"], opts)
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(lm.device)
        with torch.no_grad():
            out_ids = lm.generate(ids, do_sample=False,
                                  max_new_tokens=max_new_tokens)
        gen = out_ids[0][ids.shape[1]:].tolist()
        raw = tok.decode(gen, skip_special_tokens=True)
        got = extract(p["kind"], p["anchor"], raw)
        if not got:
            raise SystemExit("%s: empty extraction (raw=%r)" % (p["id"], raw))
        if p["kind"] == "json":
            try:
                json.loads(got)
            except ValueError as exc:
                raise SystemExit("%s: extracted answer is not valid JSON: %r "
                                 "(raw=%r)" % (p["id"], got, raw)) from exc
        if got != p["expected"]:
            print("WARNING %s: model answer differs from the prompt template: "
                  "%r != %r" % (p["id"], got, p["expected"]), file=sys.stderr)
        cases.append({"id": p["id"], "kind": p["kind"], "anchor": p["anchor"],
                      "input_ids": ids[0].tolist(),
                      "max_new_tokens": max_new_tokens,
                      "body": p["body"], "expected": got, "raw": raw})

    golden = {"format": "logestix-code-golden", "version": 1,
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
    ap.add_argument("--device-map", default=None,
                    help="transformers device_map (requires accelerate); use "
                         "'auto' for large models on a big-VRAM host")
    ap.add_argument("--template-golden", action="store_true",
                    help="stopgap for an unrunnable source: frame with the "
                         "tokenizer's chat template and record the prompt "
                         "Expected Template as expected (no model run)")
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
