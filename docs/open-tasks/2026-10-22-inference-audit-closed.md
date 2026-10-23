# Inference audit 2026-10-22 — findings closed

Verbatim snapshot of §X from `docs/OPEN_TASKS.md` as written before the fixes,
with the resolution log appended. Full evidence:
[`../INFERENCE_AUDIT_2026-10-22.md`](../INFERENCE_AUDIT_2026-10-22.md).
Branch `fix/inference-audit-2026-10-22`.

## X. Inference audit 2026-10-22 — findings to close

Full evidence: [`INFERENCE_AUDIT_2026-10-22.md`](INFERENCE_AUDIT_2026-10-22.md).
Branch `fix/inference-audit-2026-10-22`; each item gets a failing test first
(TDD) and moves to the archive when green.

- **X1 (P1) — Mistral/Xing continuation writes past the KV cache.** The cache
  is sized by the first request and never grown; `MistralDecodeStep` writes
  `K/V` and `scores` at `pos == capacity` (`mistral_moe_inference.cu:249,276`),
  Xing writes KV before `LaunchXingAttention`'s capacity check
  (`xing_moe_inference.cu:999`, `xing_moe_kernels.cu:802`). Reproduced with
  `compute-sanitizer` (146 errors). Fix: capacity growth or a pre-write guard.
- **X2 (P1) — Continuation semantics diverge.** Mistral passes
  `session_->length - 1` as `kv_last_tok` (a position, not a token id);
  Granite prefills the continuation token at pos 0 and resets `kv_rows`;
  Xing/Mistral re-feed a last token that is already decoded. `session_tail_`
  also differs (gen_tokens vs prompt+gen). Fix: one contract + tests.
- **X3 (P1) — The E1 guard is not process-global.** `FastMathDepth` is a
  TU-local anonymous-namespace static (66 copies in one binary); scopes leak
  and FSI/DET print false `VIOLATION: 2..9`. Fix: one external-linkage object,
  thread-aware nesting.
- **X4 (P1) — `--bit-exact` uses `gemv_large`.** kOff does not clear the
  large-N streaming GEMV / `LSX_GEMV_HEAD` path; the 9B selfcheck hash changes
  with `LSX_GEMV_LARGE` (`c58a9e99…` vs `3e44e44d…`). Fix: gate both on
  fast-math and clear the policy knob in kOff; re-capture r3 hashes.
- **X5 (P1) — Fused w13 writes 2I into an I-sized `d_yg`.** Mistral's
  `d_yg`/`d_yu` are separate `cudaMalloc`s; the fused GEMM spills past `d_yg`
  (memcheck: 68 errors at `lsx_tc_gemm.cu:720`) and the up half is read from
  whatever the allocator placed next. Fix: make the up half explicit
  (`d_yg + I`), require a 2I buffer, update all callers.
- **X6 (P2) — `linux-release-znver4-aocc` RPATH.** Relative `$ORIGIN` resolves
  to `apps/_deps` for every app binary; 12 fast tests fail without
  `LD_LIBRARY_PATH`. Fix: add the three `$ORIGIN/.../_deps` entries.
- **X7 (P3) — Non-split attention accepts non-pow2 `head_dim`.** The sum
  reduction silently drops elements for e.g. 192; split-KV already rejects it.
- **X8 (P3) — GatedDeltaNet state race for `seq > 1` + `col_seed`.** No
  grid barrier between the read and the `t == seq-1` write. Current pipelines
  only use `col_seed` with `seq == 1`; pin the contract.
- **X9 (P2) — Process-global caches in GLM-5.3-Flash/Kimi/Gemma-MoE are never
  reset.** `d_embed_h` is keyed only by element count; KDA/DSA/arena and
  `g_expert_db`/`g_gen_ctx` survive reloads and second instances.
- **X10 (P2) — MoE router handshake busy-polls and drops CUDA errors.** Six
  `cudaEventQuery` spins have no timeout/abort and `(void)q` discards failure.
- **X11 (P2) — Per-token/per-layer allocations and `getenv` on the decode
  path.** Split-KV attention + router `cudaMallocAsync` per layer/token; a
  `std::vector pos` per layer in granite/phi/gemma; Kimi reads `getenv` per
  token.
- **X12 (P3) — SwiGLU fallback double-activates.** After the prequant kernel
  overwrites `d_gate`, a `NotSupported/InvalidValue` falls through and applies
  silu twice. Unreachable on CC 12 today; gate the prequant path on
  `LsxTcRuntimeSupported()` and stop falling through.
- **X13 (P3) — `g_splitk_partials` is shared across streams.** Grow-only
  process buffer, `cudaFree` while an earlier kernel may still write, and the
  comment calls per-GEMM regions disjoint when they are not.
- **X14 (P3) — §V drift.** §V says qwen3.8 is on views at 10.9 GB; §V1 and the
  code say it is back on the 63.6 GB pinned arena (`LoadQwen4ExpExpertStore`,
  two passes over 24 576 files).
- **X15 (P3) — Zero-copy Arrow views are not range-checked.** `GetView`
  pointers are stored without an `in_range` check against the mapping
  (`schema.cpp:831-945`).

## Resolution log (2026-10-22)

Build `linux-release-znver4-aocc`; fast suite **97/97** (no
`LD_LIBRARY_PATH`), including the new guards below. Dense parity after the
fixes: qwen3.5-9b, qwen3.8-27b, granite-4.2-30b, gemma-4-31b CODE/DET all
green; only the pre-existing known reds remain (phi DET-02, gemma-4-31b
FSI-01, gemma-4-26b FSI/CODE).

- **X1 (fixed).** `MistralSmall4MoeModel::Infer` and
  `XingMoeModel::Infer` now check `length + 1 + max_gen_tokens <= capacity`
  before decoding and fall back to a fresh prefill (freeing/reallocating the
  session) when it would overflow. No KV write happens at `pos == capacity`.
  Regression: `test_mistral_moe_model` / `test_xing_moe_model`
  "continuation matches a fresh prefill". `compute-sanitizer --tool memcheck`
  on `test_mistral_moe_model`: **0 memory errors** (the 146-error legacy run
  also hid a real teardown bug: `h_slots` is `cudaMallocHost` memory but was
  released with `cudaFree`; now `cudaFreeHost`). Four duplicate-kernel-name
  API notices remain, pre-existing and unrelated to memory safety.
- **X2 (fixed).** One continuation contract, pinned by the same tests:
  `MistralSession::last_tok` holds the last SAMPLED-but-not-decoded token and
  is the decode seed; Mistral/Xing no longer decode that token twice. Granite
  prefills at `pos0 = kv_rows`, grows to `pos0 + P + max_new`, and sets
  `kv_rows = pos0 + seq` instead of resetting the cache. Granite Paris hash is
  unchanged (`3d1604aa75c99a1f`); Mistral/Xing selfchecks stay green.
- **X3 (fixed).** The E1 depth counter is one external-linkage object
  (`inline std::atomic<int> g_fast_math_depth`) plus a thread-local nesting
  counter in `lsx_inference_nvfp4.h`; the cross-TU regression
  `test_fast_math_guard_cross_tu` (with `test_fast_math_guard_other_tu`)
  proves the guard is shared. gemma FSI/phi DET no longer print false
  `VIOLATION`.
- **X4 (fixed).** `ResolvePerfKnobs(kOff)` clears `gemv_large_default`;
  `LsxFastMathScope(kOff)` clears/restores `LsxGemmLarge`; the dispatcher
  gates `gemv_large`/`gemv_head` on `LsxTcFastMath()`. Pinned by
  `test_gpu_perf` and `test_lsx_nvfp4_dispatch`. `--bit-exact` is now
  independent of `LSX_GEMV_LARGE`; the 9B hashes differ from the fast-math
  run by construction, so the r3 baselines are re-capture candidates.
- **X5 (fixed).** The fused w13 GEMM writes the up half to `d_yg + I`;
  every caller (qwen MoE, ornith, qwen MoE Exp, Mistral) allocates `2I`.
  Pinned by `test_qwen_moe_kernels` and `test_ornith_moe_kernels`; Mistral
  memcheck no longer reports the `TcGemmKernel` OOB write.
- **X6 (fixed).** `CMakePresets.json` RPATH now has the three
  `$ORIGIN/{..,../..,../../..}/_deps` forms. All app tests pass without
  `LD_LIBRARY_PATH`; `ldd` reports no missing library.
- **X7 (fixed).** `RunAttnFullAttnKernel` rejects non-power-of-two
  `head_dim` and inconsistent GQA before launching; regression added to
  `test_qwen_dense_full_attn`. (That suite still has a pre-existing
  split-KV tolerance failure, now tracked as X16.)
- **X8 (fixed).** Both linear-attention entries reject
  `d_gd_state && seq > 1`; regression added to
  `test_qwen_dense_fused_linear_attn`. (That suite still has pre-existing
  NaN failures at synthetic geometry `nk=2 kd=8`, unrelated to the guard.)
- **X10 (fixed).** All six busy-poll sites use
  `lsxcommon::cuda::WaitEventChecked` (`cuda_event_wait.h`): bounded wait,
  abort-aware, and the sticky CUDA error is propagated instead of discarded.
  New `test_cuda_event_wait` covers success, a pending event timing out, and
  an unrecorded event. Remaining `cudaEventQuery` calls in `libs/`: none.
- **X12 (fixed by inspection; not unit-tested).** The prequant branch is
  entered only when `lsx_tc::LsxTcRuntimeSupported()`, and any GEMM failure
  is returned instead of falling through to a second SwiGLU. The failure mode
  needs an injected TC error, which is not reachable on CC 12.x; a test would
  have to fake `LsxTcRuntimeSupported`, so none was added.
- **X13 (fixed).** The persistent split-K partials scratch is now a
  per-stream grow-only buffer (`std::unordered_map<cudaStream_t, ...>` under
  the existing mutex) instead of one process-global region; two streams
  decoding concurrently can no longer finalize over each other's partials.
  The old buffer is released on the grow path only (cudaFree synchronizes the
  device, so no in-flight kernel can still read it). Covered by the existing
  split-K dispatch tests (`test_lsx_nvfp4_dispatch`,
  `test_granite_dense_kernels`, `test_phi_dense_kernels`); a dedicated
  two-stream case was not added.
- **X14 (fixed).** §V now says qwen3.8-flash-next stays on the 63.6 GB pinned
  arena (`LoadQwen4ExpExpertStore`, two passes over 24 576 files), matching
  §V1 and the code.

Still open after this pass: **X9** (process-global caches in
GLM-Flash/Kimi/Gemma-MoE), **X11** (per-token/per-layer allocations and
`getenv` on the decode path), **X15** (zero-copy Arrow views not
range-checked), and **X16** (pre-existing split-KV attention tolerance
failure).
