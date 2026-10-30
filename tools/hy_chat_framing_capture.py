#!/usr/bin/env python3
"""Capture Hy-Memory-30B-A3B chat-framing goldens from the checkpoint's own
tokenizer + chat_template.jinja, so the engine's native renderer
(libs/lsxcommon/src/hy_moe_tool_format.cpp) can be compared against the
reference instead of against itself.

Only the tokenizer is loaded (no weights), so this is fast.

Output: JSON with one entry per case:

    {"name": ..., "messages": [...], "tools": [...], "ids": [...]}
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

# The bodies the parity gates actually run, plus the control prompt, plus a
# multi-turn and a tool conversation.
CASES = [
    {"name": "control", "messages": [{"role": "user",
                                      "content": "The capital of France is"}]},
    {"name": "system_user", "messages": [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Hi"}]},
    {"name": "multi_turn", "messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"}]},
    {"name": "tools", "messages": [{"role": "user",
                                    "content": "Weather in Paris?"}],
     "tools": TOOLS},
    {"name": "tool_roundtrip", "messages": [
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": "sunny 20C"}],
     "tools": TOOLS},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import transformers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if not getattr(tok, "chat_template", None):
        with open(os.path.join(args.model, "chat_template.jinja"),
                  encoding="utf-8") as fh:
            tok.chat_template = fh.read()

    # The gate prompt bodies, so the renderer is pinned on exactly the texts
    # the parity tests feed it.
    for fixture in ("fsi_prompts.json", "code_prompts.json", "det_prompts.json"):
        path = os.path.join(HERE, fixture)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        for p in doc["prompts"]:
            CASES.append({"name": "%s:%s" % (fixture.split("_")[0], p["id"]),
                          "messages": [{"role": "user", "content": p["body"]}]})

    out = {"format": "logestix-hy-chat-framing", "version": 1,
           "source": args.model,
           "runtime": "transformers %s" % transformers.__version__,
           "cases": []}
    for case in CASES:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if case.get("tools"):
            kwargs["tools"] = case["tools"]
        text = tok.apply_chat_template(case["messages"], **kwargs)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        out["cases"].append({"name": case["name"],
                             "messages": case["messages"],
                             "tools": case.get("tools", []),
                             "ids": ids,
                             "text": text})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print("wrote", args.out, "with", len(out["cases"]), "cases")


if __name__ == "__main__":
    sys.exit(main())
