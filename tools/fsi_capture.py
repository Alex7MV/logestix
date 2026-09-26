#!/usr/bin/env python3
"""FSISelfCheck reference capture (separate functionality from SelfCheck).

Captures the FSI golden answers from an original safetensors model with HF
transformers. The logestix side only verifies the converted model against the
committed golden (libs/lsxcommon/src/fsi_selfcheck.cpp). Run the self-test with
no model present to validate the prompt/formula set.
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "fsi_prompts.json")

# Per-model capture overrides keyed by model type. Absent -> generic defaults.
# `device`: "cuda" / "cpu" / None (auto: cuda when the full checkpoint fits in
# free VRAM, else cpu). `dtype`: torch dtype name or None (checkpoint default).
OVERRIDES = {
    # qwen3.8-flash-next is an experimental "qwen4_exp" arch: the installed
    # transformers ships Qwen4Exp* but its layer-type allowlist and KV-cache
    # mapping predate "qwen_sparse_attention", and the PLE n-gram embedding is
    # shipped as 128 shards. The custom loader patches both.
    "qwen3.8-flash-next": {"loader": "qwen4_exp"},
}
GENERIC = {"trust_remote_code": True, "max_new_tokens": 512,
           "device": None, "dtype": None, "enable_thinking": False,
           "device_map": None, "loader": None}

_QWEN4_NGRAM_ROWS = 2500012  # rows per PLE n-gram shard (128 shards)


def _load_qwen4_exp(model_path):
    """Load an experimental qwen4_exp checkpoint with the installed
    transformers (config allowlist + cache mapping patched, PLE shards fused)."""
    import glob
    import re

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



def load_prompts():
    with open(PROMPTS, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["format"] == "logestix-fsi-prompts", doc["format"]
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


def extract(kind, anchor, raw):
    text = _unfence(raw)
    lines = text.splitlines() or [text]
    line = None
    for cand in lines:
        a = cand.find(anchor)
        if a == -1:
            continue
        i = a + len(anchor)
        while i < len(cand) and cand[i].isspace():
            i += 1
        if i >= len(cand):
            continue
        if kind == "arithmetic" and cand[i] == "=":
            line = cand
            break
        if kind == "comparison" and cand[i] in "<>=!":
            line = cand
            break
    if line is None:
        for cand in lines:
            if anchor in cand:
                line = cand
                break
    if line is None:
        line = text
    if kind == "comparison":
        pat = re.compile(re.escape(anchor) + r"\s*[<>=!]+\s*[A-Za-z_]\w*")
        hit = pat.search(line)
        return normalize(hit.group(0)) if hit else ""
    a = line.find(anchor)
    if a == -1:
        return ""
    eq = line.find("=", a + len(anchor))
    if eq == -1:
        return ""
    i = eq + 1
    depth = 0
    first = True
    prev_op = True
    n = len(line)
    while i < n:
        ch = line[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "(":
            depth += 1
            prev_op = True
            first = False
            i += 1
            continue
        if ch == ")":
            if depth == 0:
                break
            depth -= 1
            prev_op = False
            first = False
            i += 1
            continue
        if ch in "+-*/":
            prev_op = True
            first = False
            i += 1
            continue
        if ch.isalnum() or ch in "_.":
            if (not first) and depth == 0 and (not prev_op):
                break
            while i < n and (line[i].isalnum() or line[i] in "_."):
                i += 1
            prev_op = False
            first = False
            continue
        break
    return normalize(line[a:i])


def selftest():
    prompts = load_prompts()
    assert [p["id"] for p in prompts] == ["FSI-TEST-01", "FSI-TEST-02",
                                          "FSI-TEST-03"]
    for p in prompts:
        got = extract(p["kind"], p["anchor"], p["expected"])
        assert got == p["expected"], (p["id"], got)
    # Formula checks: FSI-03 is False at the stated boundary; both arithmetic
    # formulas evaluate to positive finite values with the stated parameters.
    assert (14.999991 > 15.000000) is False
    v1 = eval("CF*((1-exp(-r*n))/(1-exp(-r)))", {"__builtins__": {}},
              {"CF": 2500000, "n": 7, "r": 0.0835, "exp": math.exp})
    v2 = eval("(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C",
              {"__builtins__": {}},
              {"Rate_A": 1.083541, "Slip_A": 0.0005, "Rate_B": 83.152402,
               "Slip_B": 0.0005, "Rate_C": 0.011104, "Slip_C": 0.0005})
    assert math.isfinite(v1) and v1 > 0
    assert math.isfinite(v2) and v2 > 0
    print("fsi selftest OK: pv=%.6f arb=%.6f" % (v1, v2))


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
    if opts.get("loader") == "qwen4_exp":
        lm = _load_qwen4_exp(model)
        print("device:", next(lm.parameters()).device)
    else:
        load_kwargs = {"trust_remote_code": opts["trust_remote_code"]}
        if opts.get("dtype"):
            load_kwargs["dtype"] = opts["dtype"]
        if opts.get("device_map"):
            load_kwargs["device_map"] = opts["device_map"]
        lm = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        lm.eval()

        device = opts.get("device")
        if device is None and not opts.get("device_map") and torch.cuda.is_available():
            nbytes = sum(p.numel() * p.element_size() for p in lm.parameters())
            free, _total = torch.cuda.mem_get_info()
            if nbytes < int(free * 0.9):
                device = "cuda"
        if device:
            lm = lm.to(device)
        print("device:", next(lm.parameters()).device)

    cases = []
    for p in prompts:
        prompt_text = p["body"]
        if getattr(tok, "chat_template", None):
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if opts.get("enable_thinking") is not None:
                template_kwargs["enable_thinking"] = opts["enable_thinking"]
            try:
                prompt_text = tok.apply_chat_template(
                    [{"role": "user", "content": p["body"]}], **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                prompt_text = tok.apply_chat_template(
                    [{"role": "user", "content": p["body"]}], **template_kwargs)
            except Exception as exc:  # noqa: BLE001
                print("chat_template failed, using raw prompt:", exc,
                      file=sys.stderr)
                prompt_text = p["body"]
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(lm.device)
        with torch.no_grad():
            out_ids = lm.generate(ids, do_sample=False,
                                  max_new_tokens=max_new_tokens)
        gen = out_ids[0][ids.shape[1]:].tolist()
        raw = tok.decode(gen, skip_special_tokens=True)
        got = extract(p["kind"], p["anchor"], raw)
        if not got:
            raise SystemExit("%s: empty extraction (raw=%r)" % (p["id"], raw))
        if got.count("(") != got.count(")"):
            raise SystemExit(
                "%s: unbalanced parentheses in extracted answer %r (raw=%r)"
                % (p["id"], got, raw))
        if got != p["expected"]:
            print("WARNING %s: model answer differs from the prompt template: "
                  "%r != %r" % (p["id"], got, p["expected"]), file=sys.stderr)
        cases.append({"id": p["id"], "kind": p["kind"], "anchor": p["anchor"],
                      "input_ids": ids[0].tolist(),
                      "max_new_tokens": max_new_tokens,
                      "expected": got, "raw": raw})

    golden = {"format": "logestix-fsi-golden", "version": 1,
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
                         "'auto' for FP8/large models on a big-VRAM host")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model and args.type and args.out):
        ap.error("--model, --type and --out are required unless --selftest")
    capture(args.model, args.type, args.out, args.max_new_tokens,
            args.device_map)


if __name__ == "__main__":
    main()
