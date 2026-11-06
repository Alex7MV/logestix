# Plan — Mistral-Small-4-119B-2603-NVFP4 re-ingestion (2026-11-06)

Design: `docs/superpowers/specs/2026-11-06-mistral-small-4-reingest-design.md`.
Branch: `feat/mistral-small-4-reingest`.

Conventions for every step: build → run → read the evidence → only then check
the box.

## 0. Prep

- [x] 0.1 Branch `feat/mistral-small-4-reingest` (done, from `main` @ `46c32367`).
- [x] 0.2 Build the AOCC Release tree (`build/linux-release-znver4-aocc`) —
      it is the perf reference and the tree the gates run on.
      Verify: `ldd` shows 0 `not found`; `CMAKE_CXX_FLAGS_RELEASE` carries
      `-DNDEBUG`.
- [x] 0.3 `ulimit -l unlimited` in the shell used for the model runs.

## 1. Baseline: what is already green without weights

- [x] 1.1 `ctest --preset linux-debug` (fast suite + guards).
- [ ] 1.2 `ctest --preset linux-debug-convert -R mistral`.
- [x] 1.3 `ctest --test-dir build/linux-release-znver4-aocc -R test_mistral_moe`
      — loader/model/kernels/tool-framing are synthetic; record which SKIP
      (weights absent) and which pass, so a post-conversion red is attributable.

## 2. Stage 1 — convert real weights

- [x] 2.1 `lsxforge --input /mnt/fast_storage/huggingface/Mistral-Small-4-119B-2603-NVFP4
      --output <same> --mistral` (log to `/mnt/fast_storage/tmp/mistral_convert.log`).
- [x] 2.2 Structural verification of `logestix/`:
      `model.index.json` format tag, `36 × 128 = 4608` expert arrows,
      vision tensors stripped + counted, tokenizer arrows, `vram` block.
      Any mismatch = a converter defect → test first (TDD), then fix.
- [ ] 2.3 Disk/RSS sanity: tree size vs source size, and the pinned-graph cost.

## 3. Stage 3/5 — load and decode through the engine

- [ ] 3.1 `lsxfabric --localmodel <root> --validate` (engine load path).
- [x] 3.2 `lsxbenchmark --model <root> --selfcheck` — Paris, logits hash.
- [ ] 3.3 Real-weight tool round-trip (`lsxfabric --testprompt` with a tool
      prompt) → must return `get_weather {"city":"Paris"}`.

## 4. Stage 8 — gates

- [ ] 4.1 `test_mistral_moe_selfcheck` with `LSX_TEST_MISTRAL_*` env, ≥3 cycles,
      bit-exact determinism.
- [x] 4.2 `test_mistral_moe_{loader,model,kernels,tool_framing}` re-run on the
      converted tree (`inference` label).
- [ ] 4.3 Streaming + cancel: the model stage's pre-set-abort case and the
      `kLifecycle` guard.
- [x] 4.4 FSI/CODE/DET: run the three targets; confirm they SKIP for the
      documented reason (source unrunnable), and re-verify that reason from the
      checkpoint (`params.json`, no `config.json`, packed experts).

## 5. Stage 8 — perf (last, GPU-exclusive)

- [ ] 5.1 `ctest --test-dir build/linux-release-znver4-aocc -R
      test_perf_baseline_mistral_small_4 --output-on-failure`.
- [ ] 5.2 Confirm both rows (`perf`, `r3`) plus a fresh `__gpu_probe__` in
      `tests/perf/perf_baselines.json`, and that no other model's row moved.
- [ ] 5.3 Commit the JSON change exactly as the harness wrote it.

## 6. Stage 9 — documentation

- [ ] 6.1 `docs/OPEN_TASKS.md` §T: tree restored, gate results, what stays debt.
- [ ] 6.2 `docs/SELFCHECK_MATRIX.md` row for `mistral-small-4-119b`.
- [ ] 6.3 `docs/model-reports/2026-11-06-Mistral-Small-4-119B-2603-NVFP4.md`
      (publish-ready post + evidence), and the `README.md` index row.
- [ ] 6.4 `AGENTS.md`: update the family's numbers only if a measured value
      changed.

## 7. Closing verification

- [ ] 7.1 `ctest --preset linux-debug` green (guards included).
- [ ] 7.2 The family's full suite green or SKIP-with-debt, documented.
- [ ] 7.3 Commit(s) in `feat/...` prefix style; no golden or perf row rewritten
      by hand.
