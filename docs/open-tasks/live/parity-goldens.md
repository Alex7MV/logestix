# Parity goldens — declared debt and how a golden is captured

### K. Golden-debt: the declared list, and why it is not a hard failure (2026-10-06)

Vector 2 of the three "will it recur" holes is now closed. `test_model_suite_guard`
carries a new case that pairs every `Suites()` row against `tests/fixtures/{fsi,code,det}`
and is machine-enforced in **both** directions:

- a parity golden that is missing **and not declared** in `kGoldenDebt` fails;
- a `kGoldenDebt` row whose golden now exists fails too, so the list cannot outlive
  the problem and rot;
- a `kGoldenDebt` row matching no live gap fails, so renames do not leave dead entries.

Current state: **39 declared debt entries** across thirteen model rows that have
none (or only some) of the three goldens. That number is the honest coverage figure, and
until now it was only obtainable by diffing two directories by hand.

The design choice worth recording: this is **declared debt, not a hard requirement**.
A hard requirement would paint nine families red on the next fast run and demand
capturing goldens for every one — including models with no local safetensors, which
is hours of work plus vault access we do not have here. Declared debt gets the
property that actually matters (a new family cannot be added quietly without
goldens, and the remaining gap is enumerable from test output) without pretending the
existing debt is smaller than it is. A missing golden stays a SKIP; what changes is
that it can no longer be invisible.

The last single-stage entry here, `granite-4.2-30b:code`, closed 2026-11-03
(its golden had existed since 2026-10-07; archived with the inference defect
found alongside it). The mechanism remains: a family can look fully wired across
all four parity stages and still have a hole in one of them.

### L. MCP gate is a pilot on two rows — the rollout is the open work (2026-11-28)

`mcp::RunMcpSelfCheck` shipped 2026-11-28 with three JSON-RPC 2.0 prompts
(`MCP-TEST-01/02/03`) and goldens for **two** models: `phi-4` and `qwen3.5-9b` —
the pair whose converted artifacts the codec-contract branch verified end to end.
Both pass 3/3 on real weights, byte-exact against the template.

The stage is therefore in `tools/gen_model_manifest.py`'s `OPTIONAL_STAGES` and
deliberately absent from `test_model_suite_guard`'s `kRequired`. This is a
rollout decision expressed in machinery, not a quality caveat:

- absent stage → no target, no golden, no complaint. The 45 other rows are
  neither reported broken nor claimed as covered (`docs/MODELS.md` says
  `mcp: no` for them);
- declared stage → every existing check applies: target registered,
  `SKIP_RETURN_CODE 4`, golden present or declared as `golden_debt.mcp`, GPU
  lock, `inference` label.

**What closing it needs, per model:** one `tools/mcp_capture.py` run (tokenizer
only, seconds — 38 of the 47 rows captured during the pilot sweep, and those are
exactly the 38 with weights on disk), a manifest row, and one
`lsx_add_parity_gate(... KIND mcp)` line. No harness change. When the last row
lands, move `mcp` out of `OPTIONAL_STAGES` and into `kRequired` **in that same
commit**, so the stage cannot be dropped quietly afterwards.

**Capture blockers seen, so they are not rediscovered as mystery:** nine of the
ten first-sweep failures are rows whose source-store directory exists but is
**empty** — no weights downloaded — and transformers reports that as a
SentencePiece/tiktoken error, which sends a reader after a Python package. Only
`alice` had weights and a genuine host gap: `alice`'s SentencePiece
`tokenizer.model` needs `protobuf` (without it transformers falls back to a
TikToken extractor and reports `Error parsing line b'\x0e'`). With `protobuf`
installed it captured cleanly, i.e. **every row with weights on disk captures**
(38 of 38). `tools/mcp_capture_all.py` now distinguishes the two cases.

The gate itself is `docs/MCP_SELFCHECK_FINDINGS.md`; the criterion (JSON
DOCUMENTS, key order free, `exact` recorded but not gating) is
`docs/superpowers/specs/2026-11-28-mcp-selfcheck-design.md` §3.
