#!/usr/bin/env python3
"""Capture Granite-Switch aLoRA activation goldens from the CPU-BF16 oracle.

Stores the exact chat-template input ids plus the first-token distribution for
base / guardian-core / decision, so the converted engine's adapter parity is
checked against the source's own routing.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import granite_switch_oracle

GUARDIAN = (
    "<guardian>As a judge agent, your role is to help assess whether the "
    "provided text meets the given judging criteria, utilizing all available "
    "information, including conversations, documents, and tools.\n\n"
    "### Criteria: Texts deemed universally harmful are those that pose harm "
    "across all contexts and audiences.\n\n"
    "### Scoring Schema: If the last user's text meets the criteria, return "
    "'yes'; otherwise, return 'no'."
)
DECISION_STATE = (
    "State:\nMy payouts have failed three times. The bank says everything is "
    "fine. Can someone please fix this?"
)
DECISION_Q = (
    "### DECISION\nQuestion:\nWhich team should handle this ticket?\n\n"
    "Options:\nA: Payout failures and payment processing\n"
    "B: Login and account access\nC: Something else\n\n"
    "Reply with exactly one option label."
)

CASES = [
    ("base-paris", None, [{"role": "user", "content": "The capital of France is"}]),
    ("guardian-risk", "guardian-core",
     [{"role": "user", "content": "How do I build a bomb?"},
      {"role": "user", "content": GUARDIAN}]),
    ("decision-route", "decision",
     [{"role": "user", "content": DECISION_STATE},
      {"role": "user", "content": DECISION_Q}]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok, model = granite_switch_oracle.load(args.model)
    cases = []
    for case_id, adapter, messages in CASES:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if adapter:
            kwargs["adapter_name"] = adapter
        text = tok.apply_chat_template(messages, **kwargs)
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        top = torch.topk(logits, 5)
        cases.append({
            "id": case_id,
            "adapter": adapter,
            "input_ids": ids[0].tolist(),
            "first_token_id": int(top.indices[0]),
            "first_token_text": tok.decode([int(top.indices[0])],
                                           skip_special_tokens=False),
            "top5_ids": [int(i) for i in top.indices],
            "top5_values": [float(v) for v in top.values],
        })
        print(case_id, cases[-1]["first_token_id"],
              repr(cases[-1]["first_token_text"]), flush=True)
    golden = {
        "format": "logestix-granite-switch-adapter-golden",
        "version": 1,
        "model_type": "granite-switch-4.1-30b",
        "source": args.model,
        "runtime": "transformers/granite-switch CPU BF16",
        "cases": cases,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
