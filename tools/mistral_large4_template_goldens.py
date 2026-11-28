#!/usr/bin/env python3
"""Template-derived FSI/CODE/DET goldens for Mistral-Large-4-1T-A52B.

The 988 GB FP8 source is a multi-node FP8 runner and cannot execute on this
32 GB card, so no MODEL-DEFINED golden can be captured. This script takes the
sanctioned deviation (GLM-5.3-Flash / GigaChat / DeepSeek-Pro precedent): the
`expected` value is the prompt set's own **Expected Template** (an external
oracle of the task), and `input_ids` come from the checkpoint's tokenizer plus
the exact frame the engine renders (`<s>` + [MODEL_SETTINGS] + [INST]...[/INST]).

The gate therefore still measures the converted model: if it answers something
other than the Expected Template, the stage is RED and that is a finding, not a
golden to rewrite.

Usage:
    python3 tools/mistral_large4_template_goldens.py \
        <source root>/mistralai/Mistral-Large-4-1T-A52B
"""
import lsx_paths
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

MODEL_TYPE = "mistral-large-4-1t"
SETTINGS = '[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]'
SOURCE = (
    "prompt-template (source not runnable locally: 988 GB FP8-E4M3 "
    "block-128 multi-node runner; the expected value is the prompt set's own "
    "Expected Template)"
)
RUNTIME = "tokenizer.json (tokenizers 0.22.2) + the engine's Large-4 frame " \
          "(<s>[MODEL_SETTINGS]{\"reasoning_effort\": \"high\"}[/MODEL_SETTINGS]" \
          "[INST]body[/INST])"

SETS = [
    ("fsi", "tools/fsi_prompts.json", "logestix-fsi-golden"),
    ("code", "tools/code_prompts.json", "logestix-code-golden"),
    ("det", "tools/det_prompts.json", "logestix-det-golden"),
]


def frame(body: str) -> str:
    return f"<s>{SETTINGS}[INST]{body}[/INST]"


def main() -> int:
    if len(sys.argv) > 2:
        print(__doc__)
        return 2
    # The store directory comes from the registry (org-nested), so an operator
    # can run this without remembering the org.
    model_dir = (Path(sys.argv[1]) if len(sys.argv) > 1
                 else lsx_paths.source_dir("mistral-large-4-1t"))
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    repo = Path(__file__).resolve().parent.parent

    for stage, prompts_path, fmt in SETS:
        src = json.load(open(repo / prompts_path))
        prompts = []
        for p in src["prompts"]:
            text = frame(p["body"])
            ids = tok.encode(text, add_special_tokens=False).ids
            if not ids or ids[0] != 1:
                raise SystemExit(f"{stage}/{p['id']}: frame does not start with <s>")
            entry = {
                "id": p["id"],
                "kind": p["kind"],
                "input_ids": ids,
                "max_new_tokens": 512,
                # [/THINK]: the decoder drops special ids, so the harness
                # cuts the reasoning chain by id and keeps the final answer.
                "answer_start_token_id": 35,
                "prefer_last_answer": stage == "fsi",
                "expected": p["expected"],
            }
            if stage == "fsi":
                # reasoning_effort=high: the chain of thought is ~700-1000
                # tokens before the one-line answer.
                entry["max_new_tokens"] = 2048
            if "anchor" in p:
                entry["anchor"] = p["anchor"]
            if "title" in p:
                entry["title"] = p["title"]
            if "extract" in p:
                entry["extract"] = p["extract"]
            prompts.append(entry)
        out = {
            "format": fmt,
            "version": 1,
            "model_type": MODEL_TYPE,
            "source": SOURCE,
            "runtime": RUNTIME,
            "prompts": prompts,
        }
        dst = repo / "tests" / "fixtures" / stage / f"{MODEL_TYPE}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
        print(f"wrote {dst} ({len(prompts)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
