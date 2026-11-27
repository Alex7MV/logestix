# Mistral-Small-4-119B-2603-NVFP4 — re-ingestion and full lifecycle verification (2026-11-06)

## Intent

The `mistral-small-4-119b` family landed on 2026-10-11/12 (stages 0–5 + 4b,
real-weight conversion, five decode defects found and fixed). The converted
vault tree is **no longer present** on this machine
(`/mnt/fast_storage/lsx_model/mistralai/Mistral-Small-4-119B-2603-NVFP4/logestix/`
does not exist while 24 other families do have theirs), so every real-weight
claim about this family is currently unverifiable. The goal is to re-establish
it: convert the source again and execute **every** model-lifecycle stage
against the resulting tree, fixing (TDD) anything that is red, and record the
measured numbers.

## Goals

- G1 Convert the real checkpoint with `lsxforge --mistral` into the protocol
  path (`<model>/logestix/`), and verify the artifact structurally: format tag
  `logestix-mistral-nvfp4`, 36 layers × 128 experts = 4608 expert arrows,
  vision ballast stripped **and counted**, tokenizer arrows written, `vram`
  block present.
- G2 Execute all lifecycle stages on real weights and record results:
  convert → load → model → kernels → selfcheck (Paris ×3, bit-exact) →
  tool framing / round-trip → streaming + cancel → perf (`perf` and `r3` rows
  re-seeded through the harness) → FSI/CODE/DET (declared debt, reason
  re-verified).
- G3 Any defect found: failing test first, then the fix, with the guard
  registries updated in the same commit.
- G4 Documentation: `docs/OPEN_TASKS.md` §T, `docs/SELFCHECK_MATRIX.md` row,
  `docs/model-reports/2026-11-06-Mistral-Small-4-119B-2603-NVFP4.md`, and the
  `AGENTS.md` numbers if a measured value changes.

## Non-goals

- No FSI/CODE/DET golden capture. Operator decision 2026-11-06: keep the three
  gates as declared debt and re-verify the documented reason (the consolidated
  checkpoint has `params.json`, no `config.json`/`model_type`, and packed NVFP4
  experts, so no source oracle can be run here). Shipping a template-derived or
  self-certified golden is explicitly against `docs/OPEN_ENDEDNESS.md`.
- No new kernels, no numeric/recipe change, unless a gate proves a defect.
- No Windows build, no `mistral-large-4-1t` work.

## Constraints

- Frozen geometry stays in `mistral_moe_constants.h`; a mismatch must fail
  `Load`, not be repaired at runtime.
- Host storage stays the neutral `ExpertStore` (R1/R2): the 56 GiB routed set is
  mmap-viewed + slot LRU, never a family-private owning arena.
- One GPU workload at a time; `test_perf_baseline` runs last, on the AOCC
  Release tree, because it writes the committed reference.
- `ulimit -l unlimited` is required for the pinned dense graph.
- Temporary/intermediate work goes to `/mnt/fast_storage/tmp`.

## Approaches considered

- **A. Convert in place into the vault** (`--output` = model root, converter
  appends `logestix/`). This is the protocol path every other family uses and
  the one every loader/selfcheck/perf target resolves through
  `lsx_model_models_dir()`. **Chosen.**
- B. Convert into `/mnt/fast_storage/tmp` and point `LSX_MODEL_ROOT` at it.
  Rejected: the perf harness pins the vault subdir, and a tree outside the vault
  would make the measured rows non-reproducible for the next operator.
- C. Skip conversion and only run the synthetic stages. Rejected: that verifies
  nothing that is currently unverified — the synthetic stages were already green
  when the family landed.

## Design

1. **Stage 1 (convert).** `lsxforge --input <src> --output <src> --mistral`.
   Validate the index + expert count + arrow files + tokenizer arrows; dump the
   conversion log for the report.
2. **Stage 3 (load) and stage 5 (decode).** Drive them through the family's own
   targets and through `lsxfabric --validate` (engine path), and record
   load time / resident bytes.
3. **Stage 8 (gates).** `test_mistral_moe_selfcheck` (Paris ×3 bit-exact,
   `LSX_TEST_MISTRAL_*` env), `test_mistral_moe_tool_framing`, and a real-weight
   tool round-trip through `lsxfabric --testprompt` with a tool prompt;
   streaming + cancel are exercised by the model stage's pre-set-abort case.
   `test_mistral_moe_{fsi,code,det}_selfcheck` are run to confirm they SKIP with
   the debt reason, not to hide them.
4. **Stage 8 (perf).** `test_perf_baseline_mistral_small_4` seeds both rows
   (`perf` = kAuto, `r3` = kOff) at the canonical 2048-token protocol through
   the harness, with its DRAM probe as the environment oracle. Rows are never
   hand-edited.
5. **Stage 9 (report).** Written after the gates, numbers copied from the gate
   output.

Failure handling: a red gate is treated as a defect, not as a baseline to move.
Reproduce with the smallest failing case, add the test, then fix.

## What is left to the builder

Ordering inside the gate phase, the exact env-var stems, and any defect triage
that appears once the tree exists.
