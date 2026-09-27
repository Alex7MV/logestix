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
    # granite-4.2 is a reasoning model; low-effort thinking is the reliable
    # mode for short exact-token answers. Its chat template lives in
    # chat_template.jinja (not tokenizer_config).
    "granite-4.2-30b": {"enable_thinking": True, "low_effort": True},
}
GENERIC = {"trust_remote_code": True, "max_new_tokens": 512,
           "device": None, "dtype": None, "enable_thinking": False,
           "device_map": None}


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
    print("code selftest OK: json/assignment/regex templates valid")


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
            if opts.get("low_effort"):
                template_kwargs["low_effort"] = True
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
                      "expected": got, "raw": raw})

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
