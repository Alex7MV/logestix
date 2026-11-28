#!/usr/bin/env python3
"""MCPSelfCheck golden capture (separate from SelfCheck/FSI/CODE/DET).

MCP is the one parity gate whose expected answer is NOT model-defined: a
JSON-RPC 2.0 payload is fixed by the protocol, so `expected` is the canonical
payload from tools/mcp_prompts.json and no source model is executed. What this
script captures per model is therefore only the FRAMED PROMPT — the
`input_ids` the source checkpoint would have seen — and it does that with the
checkpoint's own tokenizer and chat template, through the SAME framing helper
the FSI gate uses (tools/fsi_capture.py), so a per-model framing override
cannot exist in one gate and be missing in the other.

The logestix side verifies the converted model against the committed golden
(libs/lsxcommon/src/mcp_selfcheck.cpp). Run --selftest with no model present to
validate the prompt set and the protocol shape of every expected payload.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "mcp_prompts.json")
sys.path.insert(0, HERE)

# The framing facts (chat-template kwargs, vendor system turns, DeepSeek release
# encoders, tokenizer quirks) live in one table, in fsi_capture.py. Importing it
# is deliberate: a copy here would drift, and a drifted frame means the gate
# replays a prompt the served model never sees.
import fsi_capture  # noqa: E402

KINDS = ("tool_call", "resource_read", "error_response")

SOURCE = ("prompt-template (JSON-RPC 2.0 payload from tools/mcp_prompts.json; "
          "tokenizer-only capture, no source-model forward pass)")


def load_prompts():
    with open(PROMPTS, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["format"] == "logestix-mcp-prompts", doc["format"]
    return doc["prompts"]


def _canonical(obj):
    """Key-sorted, unspaced form — the comparison the C++ harness makes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _check_payload(pid, expected):
    """The protocol contract of one expected payload, as the gate will read it.

    Mirrors McpSelfCheck's criterion: the golden must itself be a valid
    JSON-RPC 2.0 member, otherwise the gate would compare the model against
    something that is not a protocol message.
    """
    try:
        doc = json.loads(expected)
    except ValueError as exc:
        raise SystemExit("%s: expected is not JSON: %s" % (pid, exc))
    if not isinstance(doc, dict):
        raise SystemExit("%s: expected is not a JSON object" % pid)
    if doc.get("jsonrpc") != "2.0":
        raise SystemExit("%s: jsonrpc must be \"2.0\"" % pid)
    if "id" not in doc:
        raise SystemExit("%s: missing id" % pid)
    has_method = "method" in doc
    has_error = "error" in doc
    if has_method == has_error:
        raise SystemExit("%s: exactly one of method/error is required" % pid)
    if has_error:
        err = doc["error"]
        if not isinstance(err, dict) or "code" not in err or "message" not in err:
            raise SystemExit("%s: error needs code and message" % pid)
        if not isinstance(err["code"], int):
            raise SystemExit("%s: error.code must be an integer" % pid)
    if has_method and not isinstance(doc["method"], str):
        raise SystemExit("%s: method must be a string" % pid)
    # The payload must already be minified: the prompt asks for it, and a
    # golden that carries whitespace would make `exact` unreachable for every
    # model. (The gate itself is order- and whitespace-insensitive.)
    if expected != _canonical(doc) and expected != json.dumps(
            doc, separators=(",", ":")):
        raise SystemExit("%s: expected is not the minified payload: %r"
                         % (pid, expected))
    return doc


def selftest():
    prompts = load_prompts()
    ids = [p["id"] for p in prompts]
    assert ids == ["MCP-TEST-01", "MCP-TEST-02", "MCP-TEST-03"], ids
    shapes = []
    for p in prompts:
        if p["kind"] not in KINDS:
            raise SystemExit("%s: unknown kind %r" % (p["id"], p["kind"]))
        doc = _check_payload(p["id"], p["expected"])
        if p["anchor"] not in p["expected"]:
            raise SystemExit("%s: anchor %r is not in the expected payload"
                             % (p["id"], p["anchor"]))
        if doc.get("method") and p["kind"] == "error_response":
            raise SystemExit("%s: error_response must not carry a method" % p["id"])
        if doc.get("error") and p["kind"] != "error_response":
            raise SystemExit("%s: %s must not carry an error" % (p["id"], p["kind"]))
        # The body must show the template it is asking for, or the prompt and
        # the golden have drifted apart.
        if p["expected"] not in p["body"]:
            raise SystemExit("%s: body does not contain its Expected Template"
                             % p["id"])
        shapes.append((p["id"], p["kind"], p["anchor"], len(p["expected"])))
    for pid, kind, anchor, size in shapes:
        print("%s %-15s anchor=%-15r payload=%d bytes" % (pid, kind, anchor, size))
    print("mcp selftest OK: %d prompts, 3 protocol shapes" % len(prompts))


def _tokenizer(model, opts):
    from transformers import AutoTokenizer
    if opts.get("tokenizer") == "fast":
        tok = fsi_capture._load_dory_tokenizer(model)
    else:
        tok = AutoTokenizer.from_pretrained(
            model, trust_remote_code=opts["trust_remote_code"])
    if not getattr(tok, "chat_template", None):
        jinja = os.path.join(model, "chat_template.jinja")
        if os.path.exists(jinja):
            with open(jinja, encoding="utf-8") as fh:
                tok.chat_template = fh.read()
    if opts.get("chat_template"):
        tok.chat_template = opts["chat_template"]
    return tok


def _frame(tok, opts, model, body):
    """The source-visible prompt text, framed exactly as the FSI capture does."""
    framed = fsi_capture.frame(model, tok, body, opts)
    if framed:
        return framed
    return fsi_capture._frame_prompt(tok, opts, body)


def capture(model, model_type, out, max_new_tokens, only=None, resume=False):
    import transformers

    opts = dict(fsi_capture.GENERIC)
    opts.update(fsi_capture.OVERRIDES.get(model_type, {}))
    tok = _tokenizer(model, opts)
    prompts = load_prompts()
    if only:
        prompts = [p for p in prompts if p["id"] in only]
        if not prompts:
            raise SystemExit("--only matched no prompt id")
    frames = [(p, _frame(tok, opts, model, p["body"])) for p in prompts]
    # Never silently drop the model's marking of its own prompt boundaries: the
    # engine's replay uses the ids verbatim, so whatever the tokenizer does here
    # is what the converted model sees.
    add_special = opts.get("add_special_tokens", True)

    cases = []
    for p, text in frames:
        ids = tok(text, add_special_tokens=add_special).input_ids
        if not ids:
            raise SystemExit("%s: empty input_ids" % p["id"])
        case = {
            "id": p["id"],
            "kind": p["kind"],
            "anchor": p["anchor"],
            "input_ids": [int(i) for i in ids],
            "max_new_tokens": max_new_tokens,
            "ignore_eos": bool(opts.get("ignore_eos")),
        }
        if opts.get("answer_channel"):
            case["answer_channel"] = opts["answer_channel"]
            case["reasoning_channel"] = opts.get("reasoning_channel") or ""
        case["expected"] = p["expected"]
        # Empty on purpose: the payload is protocol-defined, not captured from
        # a running model. test_model_suite_guard cross-checks this against the
        # `source` prefix, so the deviation cannot rot into a silent claim.
        case["raw"] = ""
        cases.append(case)
        print("[mcp-capture] %s ids=%d framed=%d chars"
              % (p["id"], len(ids), len(text)), file=sys.stderr, flush=True)

    if resume and os.path.exists(out):
        with open(out, "r", encoding="utf-8") as fh:
            prev = json.load(fh)
        if prev.get("format") != "logestix-mcp-golden":
            raise SystemExit("--resume: %s is not a logestix-mcp-golden" % out)
        by_id = {c["id"]: c for c in prev.get("prompts") or []}
        for c in cases:
            by_id[c["id"]] = c
        order = {p["id"]: i for i, p in enumerate(load_prompts())}
        cases = sorted(by_id.values(),
                       key=lambda c: (order.get(c["id"], len(order)), c["id"]))

    golden = {
        "format": "logestix-mcp-golden",
        "version": 1,
        "model_type": model_type,
        "source": SOURCE,
        "runtime": "transformers %s" % transformers.__version__,
        "prompts": cases,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, out)
    print("wrote %s (%d cases)" % (out, len(cases)), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--type")
    ap.add_argument("--out")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="replay budget recorded in the golden (reasoning "
                         "models burn tokens before the payload)")
    ap.add_argument("--only", default=None,
                    help="capture only these prompt ids (comma-separated)")
    ap.add_argument("--resume", action="store_true",
                    help="merge into the cases an earlier run already wrote")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model and args.type and args.out):
        ap.error("--model, --type and --out are required unless --selftest")
    only = None
    if args.only:
        only = [s.strip() for s in args.only.split(",") if s.strip()]
    capture(args.model, args.type, args.out, args.max_new_tokens, only,
            args.resume)


if __name__ == "__main__":
    main()
