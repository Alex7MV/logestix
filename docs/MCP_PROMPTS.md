# MCP-test prompts — where they came from, and what "correct" means here

The operator supplied three MCP validation prompts (`MCP-TEST-01/02/03`) with an
`Expected Template` each. They are in `tools/mcp_prompts.json` verbatim, and the
expected payloads are those templates.

## The one thing worth writing down

Unlike FSI/CODE/DET, the expected answer is **not** whatever a source model
happens to say. A JSON-RPC 2.0 message is fixed by its specification: `jsonrpc`,
`method`, `params`, `id` for a request, `jsonrpc`, `error{code,message}`, `id`
for an error response. So:

- `expected` is the protocol-defined payload, committed upstream in
  `tools/mcp_prompts.json`;
- no source model is executed during capture — only its tokenizer and chat
  template, to freeze the framed prompt (`input_ids`);
- the gate compares **JSON documents**, not bytes, because the specification
  says nothing about key order. `exact` records whether a model also matched the
  template's layout, and gates nothing.

Rationale and measurements: `docs/superpowers/specs/2026-11-28-mcp-selfcheck-design.md`
§3 and `docs/MCP_SELFCHECK_FINDINGS.md`.

## What the three prompts exercise

| id | what a client library must get right |
|---|---|
| MCP-TEST-01 | a tool invocation: `method`, the tool `name`, and a nested `arguments` object with the caller's `node_id` |
| MCP-TEST-02 | a resource read: a `uri` parameter with a custom scheme (`schema://`), which is the case most likely to be mangled into a filesystem path or re-quoted |
| MCP-TEST-03 | a standard protocol error: numeric code `-32602` (not a string), the canonical `message`, and the session `id` echoed |

## Not covered, on purpose

- **No live server.** The gate validates payloads offline; it never opens a
  socket, so it runs on a host with no MCP server configured.
- **No transport framing.** `Content-Length` headers, SSE chunking and
  `initialize`/`notifications/*` handshakes are transport and session concerns,
  not payload correctness, and a conversion-parity gate cannot observe them.
- **No result shapes.** `tools/list` and `tools/call` *responses* would double
  the prompt set for less signal than the three request/error shapes; add them
  as `MCP-TEST-04+` if a family is ever found to mangle results specifically.
