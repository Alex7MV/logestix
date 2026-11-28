#!/usr/bin/env python3
"""One-shot: capture the MCP golden for every manifest row (operator helper).

Not a tool the gate runs — it is the loop that produced
tests/fixtures/mcp/*.json, kept so the capture is reproducible instead of a
transcript of shell history. Failures are collected, not fatal: a model whose
tokenizer cannot be framed declares `golden_debt.mcp` in its manifest.
"""
import glob
import json
import os
import subprocess
import sys

ROOT = "/mnt/fast_storage/lsx_model"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main():
    only = sys.argv[1:] or None
    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "models", "*", "*.json"))):
        doc = json.load(open(path))
        model_type = doc["model_type"]
        if only and model_type not in only:
            continue
        vault = doc.get("vault", {}).get("dir", "")
        hits = glob.glob(os.path.join(ROOT, "*", vault))
        if not hits:
            rows.append((model_type, "", "no checkpoint directory"))
            continue
        # A placeholder directory exists but is EMPTY: the source store creates
        # the row before the download lands (docs/open-tasks infra-migrations).
        # Saying "no tokenizer" here sends the next operator after a missing
        # Python package when the real answer is "the weights are not on disk".
        has_checkpoint = any(
            name in ("config.json", "params.json") or
            name.endswith(".safetensors") for name in os.listdir(hits[0]))
        if not has_checkpoint:
            rows.append((model_type, "", "empty placeholder (weights not downloaded)"))
            continue
        rows.append((model_type, hits[0], ""))
    failed = []
    for model_type, checkpoint, why in rows:
        if not checkpoint:
            print("SKIP %-32s %s" % (model_type, why), flush=True)
            failed.append((model_type, why))
            continue
        out = os.path.join(REPO, "tests", "fixtures", "mcp",
                           model_type + ".json")
        cmd = [sys.executable, os.path.join(HERE, "mcp_capture.py"),
               "--model", checkpoint, "--type", model_type, "--out", out]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
            print("FAIL %-32s %s" % (model_type, tail[0][:140]), flush=True)
            failed.append((model_type, tail[0][:140]))
        else:
            print("OK   %-32s" % model_type, flush=True)
    print("\n%d ok, %d failed" % (len(rows) - len(failed), len(failed)))
    for model_type, why in failed:
        print("  failed: %s (%s)" % (model_type, why))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
