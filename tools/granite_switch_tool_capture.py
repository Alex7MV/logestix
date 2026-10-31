#!/usr/bin/env python3
"""Capture the Granite-Switch native tool round-trip (source input ids + call)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import granite_switch_oracle

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]
MESSAGES = [{"role": "user", "content": "What is the current weather in Paris?"}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok, model = granite_switch_oracle.load(args.model)
    text = tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, do_sample=False, max_new_tokens=128)
    gen = out[0][ids.shape[1]:].tolist()
    raw = tok.decode(gen, skip_special_tokens=False)
    golden = {
        "format": "logestix-granite-switch-tool-golden",
        "version": 1,
        "model_type": "granite-switch-4.1-30b",
        "source": args.model,
        "input_ids": ids[0].tolist(),
        "output_text": raw,
        "call": {"name": "get_weather", "city": "Paris"},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    print("wrote", args.out, repr(raw))


if __name__ == "__main__":
    main()
