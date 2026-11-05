#!/usr/bin/env python3
"""Template-derived FSI/CODE/DET goldens for Kimi-K3.

The 1.5 TB MXFP4 source needs `fla-core` plus a multi-GPU runner, so no
MODEL-DEFINED golden can be captured on this host. This script takes the
sanctioned deviation (GLM-5.3-Flash / GigaChat / Mistral-Large-4 precedent):
the `expected` value is the prompt set's own **Expected Template** (an external
oracle of the task), and `input_ids` come from the checkpoint's own tokenizer
plus the exact XTML frame the engine renders (encoding_k3.py's
`build_chat_segments` with one user turn and the thinking generation prompt).

The gate therefore still measures the converted model: if it answers something
other than the Expected Template, the stage is RED and that is a finding, not a
golden to rewrite.

Frame (encoding_k3.py, thinking on):
  <|open|>message role="user"<|sep|>{body}<|close|>message<|sep|><|end_of_msg|>
  <|open|>message role="assistant"<|sep|><|open|>think<|sep|>
`answer_start_token_id` is <|sep|> (163589): the decoder's answer starts in the
response channel, after <|open|>response<|sep|>.

Kimi-K3.1-Pro is the same tower with Gated DSA on the full-attention layers:
identical tokenizer (byte-identical `tiktoken.model`), identical XTML wire, so
its goldens are the same frames keyed to its own model type (`--model`).

Usage:
    python3 tools/kimi_k3_template_goldens.py /mnt/fast_storage/huggingface/Kimi-K3
    python3 tools/kimi_k3_template_goldens.py --model kimi-k3.1-pro \
        /mnt/fast_storage/huggingface/Kimi-K3.1-Pro
"""
import json
import sys
from pathlib import Path

MODEL_TYPE = "kimi-k3"
KNOWN_MODELS = ("kimi-k3", "kimi-k3.1-pro")
SOURCE = (
    "prompt-template (source not runnable locally: 1.5 TB MXFP4 KimiLinear "
    "checkpoint needing fla-core and a multi-GPU runner; the expected value is "
    "the prompt set's own Expected Template)"
)
RUNTIME = (
    "tiktoken.model + the checkpoint's pat_str, framed exactly as "
    "encoding_k3.py build_chat_segments (XTML, thinking generation prompt)"
)

PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

OPEN, CLOSE, SEP, EOM = "<|open|>", "<|close|>", "<|sep|>", "<|end_of_msg|>"

SETS = [
    ("fsi", "tools/fsi_prompts.json", "logestix-fsi-golden"),
    ("code", "tools/code_prompts.json", "logestix-code-golden"),
    ("det", "tools/det_prompts.json", "logestix-det-golden"),
]


def load_encoding(model_dir):
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe

    config = json.load(open(model_dir / "tokenizer_config.json"))
    decoder = config["added_tokens_decoder"]
    ranks = load_tiktoken_bpe(str(model_dir / "tiktoken.model"))
    specials = {}
    for i in range(len(ranks), len(ranks) + 256):
        entry = decoder.get(str(i))
        specials[entry["content"] if entry else f"<|reserved_token_{i}|>"] = i
    return tiktoken.Encoding(
        name="kimi_k3", pat_str=PAT_STR, mergeable_ranks=ranks,
        special_tokens=specials,
    ), specials


def encode_segments(enc, specials, segments):
    ids = []
    for text, allow_special in segments:
        if not text:
            continue
        if allow_special:
            ids.append(specials[text])
        else:
            ids.extend(enc.encode(text, disallowed_special=()))
    return ids


def frame(enc, specials, body):
    segs = [
        (OPEN, True), ("message", False), (" role", False), ('="', False),
        ("user", False), ('"', False), (SEP, True),
        (body, False),
        (CLOSE, True), ("message", False), (SEP, True), (EOM, True),
        (OPEN, True), ("message", False), (" role", False), ('="', False),
        ("assistant", False), ('"', False), (SEP, True),
        (OPEN, True), ("think", False), (SEP, True),
    ]
    return encode_segments(enc, specials, segs)


def main() -> int:
    global MODEL_TYPE
    args = [a for a in sys.argv[1:]]
    if len(args) == 3 and args[0] == "--model":
        MODEL_TYPE = args[1]
        if MODEL_TYPE not in KNOWN_MODELS:
            print(f"unknown --model {MODEL_TYPE}; expected one of "
                  f"{', '.join(KNOWN_MODELS)}")
            return 2
        args = args[2:]
    if len(args) != 1:
        print(__doc__)
        return 2
    model_dir = Path(args[0])
    enc, specials = load_encoding(model_dir)
    repo = Path(__file__).resolve().parent.parent
    sep_id = specials[SEP]

    for stage, prompts_path, fmt in SETS:
        src = json.load(open(repo / prompts_path))
        prompts = []
        for p in src["prompts"]:
            ids = frame(enc, specials, p["body"])
            entry = {
                "id": p["id"],
                "kind": p["kind"],
                "input_ids": ids,
                "max_new_tokens": 512,
                "answer_start_token_id": sep_id,
                "prefer_last_answer": stage == "fsi",
                "expected": p["expected"],
            }
            if stage == "fsi":
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
        dest = repo / "tests" / "fixtures" / stage / f"{MODEL_TYPE}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        print(f"wrote {dest} ({len(prompts)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
