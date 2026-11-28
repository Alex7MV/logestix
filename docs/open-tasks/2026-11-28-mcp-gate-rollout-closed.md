# MCP gate rollout — closed 2026-11-28

Snapshot of `docs/open-tasks/live/parity-goldens.md` §L, resolved the same day by
operator directive: **MCP is not a pilot stage — it is a lifecycle gate and runs
like SelfCheck/FSI/CODE/DET.**

## What §L said (verbatim, before the closure)

> ### L. MCP gate is a pilot on two rows — the rollout is the open work (2026-11-28)
>
> `mcp::RunMcpSelfCheck` shipped 2026-11-28 with three JSON-RPC 2.0 prompts
> (`MCP-TEST-01/02/03`) and goldens for **two** models: `phi-4` and `qwen3.5-9b` —
> the pair whose converted artifacts the codec-contract branch verified end to end.
> Both pass 3/3 on real weights, byte-exact against the template.
>
> The stage is therefore in `tools/gen_model_manifest.py`'s `OPTIONAL_STAGES` and
> deliberately absent from `test_model_suite_guard`'s `kRequired`. This is a
> rollout decision expressed in machinery, not a quality caveat:
>
> - absent stage → no target, no golden, no complaint. The 45 other rows are
>   neither reported broken nor claimed as covered (`docs/MODELS.md` says
>   `mcp: no` for them);
> - declared stage → every existing check applies: target registered,
>   `SKIP_RETURN_CODE 4`, golden present or declared as `golden_debt.mcp`, GPU
>   lock, `inference` label.
>
> **What closing it needs, per model:** one `tools/mcp_capture.py` run (tokenizer
> only, seconds — 38 of the 47 rows captured during the pilot sweep, and those are
> exactly the 38 with weights on disk), a manifest row, and one
> `lsx_add_parity_gate(... KIND mcp)` line. No harness change. When the last row
> lands, move `mcp` out of `OPTIONAL_STAGES` and into `kRequired` **in that same
> commit**, so the stage cannot be dropped quietly afterwards.
>
> **Capture blockers seen, so they are not rediscovered as mystery:** nine of the
> ten first-sweep failures are rows whose source-store directory exists but is
> **empty** — no weights downloaded — and transformers reports that as a
> SentencePiece/tiktoken error, which sends a reader after a Python package. Only
> `alice` had weights and a genuine host gap: `alice`'s SentencePiece
> `tokenizer.model` needs `protobuf` (without it transformers falls back to a
> TikToken extractor and reports `Error parsing line b'\x0e'`). With `protobuf`
> installed it captured cleanly, i.e. **every row with weights on disk captures**
> (38 of 38). `tools/mcp_capture_all.py` now distinguishes the two cases.

## Resolution

The rollout was completed in one commit, so the "move `mcp` in the same commit as
the last target" condition is satisfied by construction.

**Machinery (the policy, not a per-model toggle):**

- `tools/gen_model_manifest.py`: `OPTIONAL_STAGES = ("perf",)` — only `perf`
  remains optional, because a baseline exists only once a model is measured.
  `mcp` is no longer relaxed by "an empty stage is a gap unless declared".
- `tests/test_model_suite_guard.cpp`: `kRequired` gained `"mcp"`, so a row that
  drops the stage fails "every suite row declares the full stage set" like any
  other lifecycle stage.
- The golden-coverage guard already listed `mcp` in `kGoldenStages`, so the same
  two-way rule now applies to it: an undeclared gap FAILS, and a stale debt row
  FAILS. No new enforcement path was written — the existing one was made to
  cover a required stage instead of a pilot.

**Rows (all 48):**

- every manifest declares `stages.mcp` = `test_<family>_mcp_selfcheck` plus the
  `{"form": "parity", "kind": "mcp"}` override. The target name is derived from
  the row's own `selfcheck` target by suffix (`*_selfcheck` →
  `*_mcp_selfcheck`), which reproduces the three names that already existed
  (`test_phi_dense_mcp_selfcheck`, `test_qwen_dense_mcp_selfcheck`,
  `test_gemma_dense_12b_mcp_selfcheck`) and is unique across the table;
- 45 new `lsx_add_parity_gate(... KIND mcp)` lines in `tests/CMakeLists.txt`
  (48 total). The `add_test` / label / `SKIP_RETURN_CODE 4` / GPU-lock half still
  comes from the generated `model_targets.cmake`, so it cannot drift;
- goldens: **39 captured**, 9 declared as `golden_debt.mcp`.

**The 9 debt rows are the empty source directories**, confirmed by the canonical
tool rather than by inspection:

```
$ LSX_SOURCE_ROOT=/mnt/fast_storage/lsx_model python3 tools/mcp_capture_all.py
39 ok, 9 failed
  minimax-m3, flint, mimo-v2.6-flash-rl, ling, mistral-large-4-1t,
  kimi-k3, beam, step-5-preview, glm-5.2
  — "empty placeholder (weights not downloaded)" each
```

No converted artifact exists for any of them either, so the framed prompt
genuinely cannot be produced: `mcp_capture.py` runs no forward pass, but the
`input_ids` must still come from the checkpoint's own tokenizer and chat
template. This is the same treatment FSI/CODE/DET already get on those rows —
the stage and its target are declared, the gap is measured and named, and the
gate SKIPs (exit 4) rather than pretending to pass. Re-capture is one command
once a checkpoint is restored.

**A defect found in the helper while doing this.** `tools/mcp_capture_all.py`
crashed with `AttributeError: 'PosixPath' object has no attribute 'endswith'`:
it iterated `checkpoint.iterdir()` (Path objects) and called string methods on
them, so the "is this a real checkpoint or an empty placeholder" test could
never have run. Fixed to read `entry.name`; that is why the 39/9 split above is
now the tool's own output rather than a hand count.

## Measured after the change

```
ctest -R "_mcp_selfcheck" -j 1        → 49 tests: 4 passed, 45 skipped, 0 failed
  test_phi_dense_mcp_selfcheck         Passed  6.53 s
  test_qwen_dense_mcp_selfcheck        Passed  3.72 s
  test_gemma_dense_12b_mcp_selfcheck   Passed  7.63 s
  test_mcp_selfcheck (host criterion)  Passed  0.01 s
```

The 45 skips are the rows with no converted artifact on this host — the same
reason their FSI/CODE/DET targets skip. Every skip message names the model and
the directory it looked in, so a skip cannot be mistaken for a pass.
