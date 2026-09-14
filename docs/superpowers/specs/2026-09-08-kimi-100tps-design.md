# Kimi-K2.7-Code — 100 tok/s decode on RTX 5090 — Design

Date: 2026-09-08
Branch: `feat/kimi-100tps`
Status: approved design (approach A, amended with model-independence correction)
Reference: Qwen3.5-35B-A3B 101 t/s round (`2026-09-01-qwen-moe-80tps-design.md`),
GLM-5.3-Flash dual-stream staging, NVIDIA docs reference
(`docs/NVIDIA_DOCS_NVFP4_MEMORY.md` §B4, §B5).

## Goal

Raise the DEFAULT production decode throughput of Kimi-K2.7-Code on the
RTX 5090 (Linux, release-znver4/AOCC, LRU expert cache) to **>= 100 tok/s**,
measured by the standard protocol: real model from the vault
(`/mnt/llm_models/models/Kimi-K2.7-Code`), FAST_MATH=1, 128 generated
tokens, >= 3 alternating runs, within-config spread < 1%. Baseline is
measured FIRST (Phase 0) — the current t/s after the split-phase decode
round is unknown and must be recorded before any lever is engaged.

## Geometry & stream facts (fixed)

- 61 layers (layer 0 dense MLP 18432, layers 1..60 MoE), 384 routed
  experts/layer, top-8, expert slab [w1|w3|w2] = 3 × (7168×2048) NVFP4,
  ~22 MiB packed per expert.
- Total routed weight stream: 23424 experts ≈ 515 GiB — full residency
  impossible on 32 GiB; the generation loop runs against the LRU device
  slot cache with mmap streaming of misses.
- Worst-case miss traffic per token: top-8 × 61 = 488 experts ≈ 10.7 GiB
  H2D — cache warmth dominates; no kernel-level lever can reach 100 tok/s
  on a fully cold cache.
- MLA attention (absorbed decode form) is NOT a target of this iteration.

## User-approved decisions

| Decision | Choice |
|---|---|
| Correctness contract | Determinism + selfcheck (Qwen amendment 2026-09-02 pattern); bitwise parity with the legacy GEMV chain is NOT required for the TC path. |
| Profiling | Measure the actual baseline first; prioritize levers by the measured profile. |
| Residency | Hybrid NVFP4-TC + VRAM reserve revision (no full checkpoint re-conversion). |
| Branch / reference | `feat/kimi-100tps`, Qwen round architecture. |

## Correctness contract (STRICT)

- **BINDING** — run-to-run determinism: selfcheck "The capital of France
  is" → "Paris" ×3 cycles bit-exact AND a two-call Infer `final_logits`
  memcmp gate. Both must pass after every phase that changes the numeric
  path (Phase 1 TC batch, Phase 4 device feed).
- **BINDING** — EOS truncation and `kv_slots_used` remain correct under
  device-side decode.
- **BINDING** — model independence: `qwen_tc_gemm` renamed to
  `lsx_tc_gemm` + namespace `lsx_tc` BEFORE Kimi consumes the TC path
  (no kimi→qwen symbol references); Kimi opens its own FastMathScope;
  kimi-scoped VRAM reserve (shared governor constant untouched).
- **NOT required** — bitwise equality of the TC batch with the legacy
  dequant→BF16 GEMV chain (split-K slice composition diverges; Qwen
  precedent). On the real geometry the gate is determinism, not parity.
- **Invariant** — the legacy per-call staging path is fully preserved as
  the fallback (cache disabled / slot miss / unfused geometry /
  non-sm_12x device).
- Numeric class changes: W4A4 TC fast_math (the growth source), accepted.
- Model independence (operator directive, verified against
  `test_model_independence_guard.cpp`): the TC batch entry points are
  consumed ONLY through the model-neutral NVFP4 layer (`lsx_nvfp4`);
  Kimi defines no family-local fast-math toggle; `KimiK27CodeModel::Infer`
  opens `LsxFastMathScope(cfg_.fast_math, ...)` so no other model's policy
  can leak into Kimi kernels and vice versa.

## Architecture (6 phases, TDD order)

### Phase 0 — Baseline profile (FIRST step of the plan)

Measure current decode t/s on the real model (selfcheck + perf pattern of
`test_blackwell_perf_moe`, FAST_MATH=1, 128 tokens, 3 runs) and split the
GPU step by CUDA events: attn / router / expert-FFN / head / host gaps.
Record numbers in the findings registry. Lever priorities below are
re-ordered if the measured profile contradicts the Qwen analogy.

### Phase 1 — Batched TC routed-expert chain (model-neutral)

- **Neutral layer consumption**: the batched TC entry points live in
  `lsx_tc_gemm.h` (renamed, Phase 6) and are called directly by Kimi —
  the neutral `lsx_nvfp4` dispatcher gains no kimi-specific wrapper;
  `lsx_tc_gemm` carries no family stem (AGENTS.md NVFP4 GEMM rule:
  single neutral surface, no family-owned copies).
- **BF16 act-quant**: Kimi hidden states are `__nv_bfloat16`; the
  neutral layer gains `LsxTcQuantizeActsNvfp4Bf16` (same per-16
  pow2-E4M3 + E2M1 codec as the FP16 variant; parity test pins
  bit-equal output for values representable in both).
- New Kimi launcher `RunKimiRoutedTcBatch` in `kimi_moe_inference.cu`:
  - act-quant: FP16/BF16 hidden row → NVFP4 (per-16 pow2-E4M3 + E2M1),
    reusing `QuantizeActsNvfp4` semantics;
  - w13-batch: ONE launch, grid `(ceil(4096/64), 1, split_k·8)`,
    M=1, N=2I=4096, K=H=7168; per-slot weight base = slot pointer from
    the device slot table; `{gs1|gs3}` segment split at N/2 in the
    epilogue;
  - batched swiglu-quant (K×I = 16384 elements) → NVFP4 acts for down;
  - w2-batch: M=1, N=H=7168, K=I=2048 into per-slot partials;
  - ascending k=0..7 coeff reduction into the accumulator (round-19b
    order).
- Kernel class: warp-level `mma.sync.aligned...kind::mxf4nvf4.block_scale`
  (the proven CC 12.x route; tcgen05 tensor-memory MMA is NOT available on
  sm_120a — docs §B4). Runtime probe `LsxTcRuntimeSupported()` gates
  engagement; `cudaErrorNotSupported` → legacy path.
- The expert slab layout in cache slots is the EXISTING checkpoint NVFP4
  packing (no repack at upload for Phase 1 — the TC kernel consumes the
  repo layout, like Qwen).

### Phase 2 — Host-built slot table (router fence stays, 128-byte H2D)

AMENDED (planning round): Qwen's device-composed no-handshake table
requires FULL residency (dev_slot == flat index forever, no LRU). Kimi
streams 515 GiB through an LRU cache — slot assignment is a host-side
LRU decision, so:

- The per-layer event-scoped router fence (`KimiRouteBegin/Finish`,
  Round F-11) REMAINS — it is event-scoped and does not drain the stream.
- Per-flat-expert TC-meta `{gs1, gs3, gs2}` is staged at slot insert
  into the cache slot table (mirrors Qwen round-19c, host-side).
- The 8-entry per-layer slot table `{dev_slot, gs1, gs3, gs2, coeff}`
  is composed host-side right after `KimiRouteFinish` and moved with ONE
  128-byte async H2D (pinned staging, reused per generation).
- Engagement gate: cache active AND TC fast-math AND `I % 16 == 0` AND
  `LsxTcRuntimeSupported()`. Any anomaly → the driver falls back for
  that generation to the legacy per-call staging path (bit-identical
  fallback, already the code path today).

### Phase 3 — Dual-stream expert prefetch (GLM-Flash pattern)

- Prefetch the predicted top-8 of layer l+1 (route from the previous
  step — top-8 changes slowly between decode steps) onto `copy_stream`
  while the expert FFN of layer l computes; event-plumbed readiness.
- Upload of a miss batch stays `UploadKimiMoeExpertBatch` (unchanged
  contract); only its scheduling point moves earlier.
- Determinism: weights arrive earlier but are byte-identical; two-run
  token-stream memcmp gate.

### Phase 4 — Device-side token feed (kimi round-25 completion)

- Argmax stays on device; `kimi_feed_token_kernel` (already present)
  feeds `*d_best` into the next embed gather; `gen.tokens` drained by ONE
  bulk D2H after the loop; TTFT via events; EOS host-truncate over the
  drained tokens via `Tokenizer().eos_id` (already threaded).
- The per-step event busy-poll at the argmax readback
  (`kimi_moe_inference.cu:1408-1419`) is removed.
- Race gate (round-25 lesson): if the two-call memcmp gate fails, token
  feed reverts to event-scoped readback and the lever stays latent.

### Phase 5 — Hybrid residency + Kimi-scoped VRAM reserve

- Kimi-scoped reserve: the shared `lifecycle::kExpertCacheVramReserveBytes`
  (4 GiB) is NOT touched (model-independence rule). `KimiMoeExpertCacheSlots`
  gains a kimi-namespace reserve constant (re-measured, tighter) used only
  by Kimi's cache sizing; the `LSX_KIMI_EXPERT_CACHE_SLOTS` override stays.
- TC-meta (`{gs1,gs3,gs2}` + slot binding) staged at insert — a tiny
  per-slot on-upload transform, bit-exact conversion test pinned.
- Hot-expert hinting (runtime tuner already exists) may bias LRU
  bookkeeping — optional, only if the measured hit-rate is the residual
  bottleneck.

### Phase 6 — Neutral rename prerequisite: qwen_tc_gemm → lsx_tc_gemm

Audit finding (verified 2026-09-08): `qwen_tc_gemm.{h,cu}` is de-facto
cross-family infrastructure — consumed by qwen (dense+moe), deepseek
(`deepseek_moe_inference_moe.cu`), the neutral layer dispatcher
(`lsx_inference_nvfp4.cu`) — with ZERO qwen-specific content (no qwen
constants, headers, or geometry; only measurement comments in prose).
Done BEFORE Phase 1 so Kimi never references a qwen-named symbol:

- Files renamed: `qwen_tc_gemm.h` → `lsx_tc_gemm.h`,
  `qwen_tc_gemm.cu` → `lsx_tc_gemm.cu`.
- Namespace `qwen::tc` → `lsxcommon::lsx_tc`.
- Symbols: `RunQwenTcNvfp4Gemm` → `RunLsxTcNvfp4Gemm`,
  `RunQwenTcGemmPreQuant` → `RunLsxTcGemmPreQuant`,
  `RunQwenTcGemmBatch13` → `RunLsxTcGemmBatch13`,
  `RunQwenTcGemmBatchDown` → `RunLsxTcGemmBatchDown`,
  `QwenTcBatchSlot` → `LsxTcBatchSlot`,
  `QwenTcBatchSplitK` → `LsxTcBatchSplitK`,
  `QuantizeActsNvfp4` → `LsxTcQuantizeActsNvfp4`,
  `TcRuntimeSupported` → `LsxTcRuntimeSupported`,
  `ProbeTmaBulkCopy` → `LsxProbeTmaBulkCopy`,
  `BuildFusedGemmWeights` → `LsxTcBuildFusedGemmWeights`,
  `TcFusedOutputScratch` → `LsxTcFusedOutputScratch`.
- CMake OBJECT lib `lsxcommon_qwen_tc_cu` → `lsxcommon_tc_cu` (its
  `sm_120a/sm_121a` generate-code set is carried over unchanged).
- Consumers updated mechanically: `qwen_inference.cu`,
  `qwen_moe_inference.cu` (incl. the layout static_assert),
  `deepseek_moe_inference_moe.cu`, `lsx_inference_nvfp4.cu`, tests
  (`test_blackwell_tc_mma.cu`, `test_blackwell_micro_gemm.cu`,
  `test_qwen_moe_kernels.cu`, `test_deepseek_moe_inference_kernels.cu`).
- Naming guard: allowlist entry `qwen_tc_gemm` removed
  (`test_code_naming_guard.cpp`); no qwen-specific code needs
  extraction into a separate file — none exists.

### Model-independence amendments (guard tests, same commit)

- `tests/test_model_independence_guard.cpp`:
  - `consumers` += `kimi_moe_model.cpp` (must open
    `LsxFastMathScope(cfg_.fast_math, ...)` in Infer);
  - fast-math toggle check: kimi TUs define no `g_kimi_fast_math`-style
    state; the single definition stays in `lsx_inference_nvfp4.cu`.
- `tests/test_code_naming_guard.cpp`: allowlist entry `qwen_tc_gemm`
  removed; no new family-prefixed files outside the kimi family (the
  renamed neutral files carry no family stem and pass the scheme scan).

## Error handling

- Fail-fast with the named-stage `fail(e, "...")` pattern.
- CUDA error → generation aborts; weights/cache survive (idempotent
  Unload preserved).
- Fallback chain: TC unavailable / cache miss → legacy per-call staging
  (current default); device-table compose failure → per-layer host
  readback (current behavior).
- Any red determinism test → revert the corresponding default (kimi
  round-25 precedent); levers remain opt-in/latent.

## Tests (TDD, plan order)

0. `[rename]` Phase 6 rename is regression-pinned: full `ctest -R
   "kimi|qwen|deepseek|blackwell"` suite green after the mechanical
   rename; naming guard green with `qwen_tc_gemm` removed from the
   allowlist; `rg "QwenTc\|qwen::tc\|qwen_tc_gemm"` returns zero hits
   in libs/ and tests/.
1. `[tc-batch]` synthetic: batched TC chain vs pinned legacy GEMV chain —
   tolerance parity (NOT bit) + adversarial inputs: duplicate experts,
   zero coeffs, alternating scale bytes (round-19 lesson).
2. `[tc-batch]` fallback parity: cache-miss path bit-identical to the
   current default staging path.
3. `[slot-table]` host-bookkeeping device compose unit tests (pattern of
   `test_kimi_moe_expert_cache.cpp`, host-only parts without CUDA).
4. `[neutral-layer]` guard: kimi calls only `lsx_nvfp4` entry points;
   no kimi-named dispatcher state; guard test extension compiles red
   before the FastMathScope amendment.
5. `[staging]` dual-stream prefetch race gate: two runs, token-stream
   memcmp.
6. `[decode]` two-call Infer `final_logits` memcmp (round-25 gate) + EOS
   truncation correctness under device feed.
7. `[selfcheck]` real "Paris" ×3, bit-exact (`LSX_TEST_KIMI_SELFCHECK_CYCLES`,
   SKIP_RETURN_CODE 4).
8. Perf: ≥ 100 tok/s by the standard protocol; before/after recorded in
   the findings registry.
- Mandatory regression: all `test_kimi_*` green, naming guard green,
  `test_absl_format_guard` green, `test_model_independence_guard` green.

## Non-goals

- No checkpoint re-conversion (format `logestix-kimi` untouched).
- No CUDA Graph default (Round 23 measured neutral on Linux).
- No GEMV-experts default, no head-GEMV experiments.
- No MLA attention changes; no numeric-class experiments beyond W4A4 TC.
- Windows: not exercised by this iteration (model tests are Linux-only
  per AGENTS.md OS restriction).

---

## Findings (execution round, 2026-09-08, feat/kimi-100tps)

### Result: goal re-scoped — 100 tok/s is physically unreachable for this
### model on this GPU; iteration delivered bug fixes + latent fast paths.

**Baseline (Phase 0):** 1.4 tok/s (128 tokens, staging path) — the expert
device cache was silently DISABLED in production by a uniformity-check bug
(comparing strides against the zero-entry of dense layer 0), forcing the
legacy per-call staging path. A second pre-existing bug (pass-3 slot
resolution ignoring first-touch `missing_slot`) broke the cached path the
moment it engaged. Both fixed on this branch.

**H2D wall (measured):** 515 GiB of routed experts through 32 GiB VRAM.
One decode token touches top-8 × 61 = 488 experts ≈ 11.2 GiB. With the
LRU cache warm (414–520 slots, governor-capped), measured upload traffic
stays ~480 experts/token (top-8 token-to-token reuse ~10–50%) → ~625
ms/token of PCIe time dominates → **cached decode ceiling ≈ 4–15 tok/s**
regardless of kernels. Qwen3.6's 105 tok/s was enabled by FULL expert
residency (all experts fit in VRAM) — not achievable for Kimi's 515 GiB.
Reaching 100 tok/s would need ~100+ GiB VRAM or an order-of-magnitude
faster interconnect.

**Delivered on this branch:**
1. `fix:` cache uniformity check tolerates dense-layer zero entries
   (root cause of the 1.4 t/s staging fallback).
2. `fix:` cached path resolves first-touch slots via `missing_slot`.
3. `refactor:` `qwen_tc_gemm` → `lsx_tc_gemm` (model-neutral rename;
   qwen/deepseek/neutral-layer consumers; naming-guard allowlist cleaned).
4. `feat:` `LsxTcQuantizeActsNvfp4Bf16` (neutral-layer BF16 act-quant).
5. `feat:` `RunKimiRoutedTcBatch` — batched TC routed-expert chain +
   neutral-layer batch-down per-slot activation offset fix (latent bug
   found in the Qwen batch-down: single shared activation row for all
   slots; pinned by new tests).
6. `feat:` `LsxFastMathScope` consumer registration for kimi + guard
   extension.
7. Device-side token feed was implemented then RE-PARKED: the fully-async
   decode reproduced the round-25b cross-cycle `final_logits` divergence
   (two-call memcmp gate red at cycles 2-3, nondeterministic across
   processes). Per-step event-scoped readback restored; determinism gate
   green x3 again (round-25 precedent: red determinism test reverts the
   lever).

**Parked (honest engineering call):** the device expert cache is gated
behind an explicit `LSX_KIMI_EXPERT_CACHE_SLOTS` override: with the cache
engaged, generation produces wrong logits (selfcheck "Domino!" vs
"Paris"; bypass staging is bit-correct). Slot bytes verified byte-correct
right after upload yet read back different later — an unresolved writer
between upload-retire and read; needs a dedicated compute-sanitizer
session. Because the cache NEVER ran in production before (bug #1), this
is a pre-existing latent defect exposed by fix #2, not a regression.
The TC batch path (correctness-proven on synthetic geometry) stays latent
behind the same gate and selfcheck passes x3 bit-exact on the default
staging path.

**Final default-path perf:** 1.4 tok/s staging (unchanged — H2D-bound;
the TC path cannot help until the cache is un-parked and upload traffic
is eliminated).

**Regression status:** full suite 127/127 green (serial run; the parallel
run's six failures were VRAM contention between 17+ GB models). Skips are
missing checkpoints for other families.

### Next steps (follow-up iteration)
1. compute-sanitizer (racecheck/memcheck) session on the kimi cache
   writer; un-park the cache.
2. Re-engage TC batch (already wired, selfcheck-gated).
3. Optional: NVFP4-quantize pinned MLA weights (17.3 → ~6 GiB) to widen
   the slot budget.
4. Optional: two-token streaming reuse analysis to size the slot budget
   for a target hit rate.

### Interim checkpoint (2026-09-08, end of session — deps bumps + fixture OOB fix)

Commit `3ed7979` (deps) + `34daa37` (CCCL), all pushed to `feat/kimi-100tps`:

- **Dependencies bumped (TDD, version-pinned in `test_core.cpp`
  "pinned dependency versions"):** spdlog 1.15.0 → **1.17.0**, Catch2
  3.7.0 → **3.16.0** (FetchContent GIT_TAGs), simdjson v4.4.2 →
  **v4.6.11** (submodule), NVIDIA CCCL → **tag v3.4.2** (submodule; prior
  HEAD was a master commit self-reporting 3.5.0-dev, not a release).
  AGENTS.md/README.md dependency tables updated.
- **Flaky-test root cause fixed (NOT a kernel race):** the
  ~40% `RoutedGemmVBatch` memcmp flakes across the 12 `test_qwen_dense_*`
  executables were an OOB in the `RunBatch` fixture itself
  (`tests/test_qwen_nvfp4_gemv.cu`): hardcoded `eids = {3,0,3,1}`
  (duplicate-expert coverage) with `S=2` cases read `dev_slot=3` past the
  2-slot `d_pack`/`d_scale` allocations → garbage heap bytes →
  nondeterministic bits. compute-sanitizer: 98 invalid reads; after the
  `eids[k] % S` wrap (slot table AND ground-truth loop — the second OOB
  was in the reference path) 0 errors. `RunQwenMoeRoutedGemmVBatch` was
  correct all along; the determinism invariant holds.
  Lesson: the 12 executables share one OBJECT lib — any test-source fix
  requires a full rebuild, partial target rebuilds keep stale code.
- **Verification:** compute-sanitizer 0 errors on the suite; stress 40+
  serial runs green; full ctest **127/127** (serial, fresh, on `34daa37`).
- Unchanged parked items for the next iteration: kimi device-cache slot
  corruption (needs compute-sanitizer session) → un-park cache → re-engage
  TC batch (wired, selfcheck-gated); optional MLA NVFP4 requant.

---

## Findings (execution round, 2026-09-14) — copy/compute overlap + shared-expert residency

### Result

**3.10 -> 5.10 tok/s @2048** (RTX 5090, Linux, `linux-release-znver4-aocc`,
LRU device cache), reproduced twice: 398.4 s / 399.5 s. TTFT 4012 -> 1421 ms.
Intermediates: 3.8 t/s after the pipeline, 4.7 t/s @128 after residency.
Commits: `e255048` (perf), `9a6c237` (docs). Gates: selfcheck Paris x3
bit-exact (18 assertions), 12/12 kimi ctest, fast label 45/45.

Contradicts the earlier "cached decode ceiling 4-15 tok/s regardless of
kernels" pessimism only in degree: the wall was never the bus alone, it was
the bus **plus** a fully serial compute stream that the batch-wide upload
fence prevented from overlapping.

### Phase 0 profile (the decisive measurement)

`nsys profile -t cuda --stats=false ... --max-gen 64`, then
`nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum`.
Per decode token: **142 ms of kernels, 204 ms of H2D, wall = their sum**
(333 ms at 3.0 t/s) — i.e. zero overlap.

Kernels (9.088 s / 64 tok):

| share | kernel | note |
|---|---|---|
| 71.4% | `Nvfp4DequantKernel<__nv_bfloat16>` | 111,841 calls = 1747/token, 58 us each; materializes 29.4 MB of bf16 per projection, then cuBLAS re-reads it (~648 GB/s effective) |
| 8.2% + 2.8% + 0.2% | cuBLAS `nvjet` bf16 GEMMs | the routed/shared FFN projections |
| 3.2% | `kimi_router_topk_kernel` | 70 us x 61 layers |
| 3.1% | `kimi_mla_attn_kernel` | |
| 2.6% | `lsx_nvfp4_gemv_warprow_kernel` | MLA absorbed projections |

H2D = 8.60 GB/token, and the per-stream split was the root cause:

| stream | bytes/token | ms/token | rate | what |
|---|---|---|---|---|
| copy (14) | 6.86 GiB | 120 | **57 GB/s** | 277 routed-expert misses x 24.78 MiB ([w1|w3] 14.68 MB, [s1|s3] 1.835 MB, w2 7.34 MB, s2 0.917 MB, 4 copies/miss) |
| compute (13) | 1.42 GiB | 84 | **18.8 GB/s** | the 61 shared experts (3 projections each) re-streamed EVERY token from a **pageable** mmap, host-blocking, serially ahead of the routed FFN |

The copy stream is already at the Gen5 bus ceiling, so the only H2D levers
are (a) fewer misses and (b) overlap. The compute-stream pageable blob was
pure waste: identical bytes, every token. Cache: 809 slots x 23.6 MiB
against 21.2 GiB free before it.

### Change 1 — per-slice upload pipeline (+27%)

`KimiMoeCacheBatch::wait_miss_pos[]` (`kimi_moe_expert_cache.h`) now states,
per batch entry, *which* missing entry's upload delivers its slot: a miss
waits on its own upload, a same-batch duplicate inherits the first
occurrence's position, and a pre-resident expert needs no wait at all. The
CUDA side (`kimi_moe_expert_cache.cu`) records one readiness event per
missing entry (`ready_slot[8]`) and gates the upload on the last-read event
of the slot it evicts (`slot_read[slots]`, published by
`KimiMoeSlotMarkRead` after the FFN) instead of a batch-wide fence. The old
fence made the copy stream wait for ALL already-enqueued compute, which
serialized upload against compute by construction. `upload_fence` is gone;
the routed loop waits per entry and the copy engine keeps uploading while
the SMs compute.

### Change 2 — shared-expert device arena (+24%, and TTFT x2.5)

One `[w1|w3|w2]` / `[s1|s3|s2]` slab per MoE layer (61 x 1441 MiB),
uploaded once per model, consumed through `RunKimiExpertFfnCached`. The
upload and the inference-time view both call `KimiSharedSlabLayout`, so a
layout disagreement (silent corruption of every shared expert) is
impossible by construction. Sized BEFORE the routed cache so the slot budget
sees the reduced free VRAM: slots 809 -> 748, which costs ~8% more routed
misses and is more than paid back by the 84 ms that disappears.

### TDD artifacts

Each test was watched fail first (compile error for a missing API counts,
but the behavioural tests were also seen red at runtime):

- `tests/test_kimi_moe_expert_cache.cpp`
  `"Kimi cache batch: per-slice upload dependency plan"` — pins the
  dependency plan (pre-resident -> -1, miss -> own position, duplicate ->
  first occurrence's position, disabled cache -> all -1).
- `tests/test_kimi_moe_expert_cache.cpp`
  `"Kimi shared-expert slab layout: 256-aligned, gap-free"` — pins the
  single layout function shared by upload and view.
- The overlap itself is a scheduling property with no host-observable
  state; its TDD proxy is the dependency plan above plus the E2E bench.

### Rejected: LFU cache policy (measured, then reverted)

Frequency-based victim selection with periodic count ageing: **+15% at 128
tokens** (4.7 -> 5.4 t/s), the exact direction the trace simulation
predicted (283 -> 219 misses/token at 748 slots). But at 2048 tokens it
regressed to **3.7 t/s and stopped at 1386 tokens (early EOS)** — an output
divergence, and functionally a much worse cache. Structural cause: LFU
evicts recently read *low-count* slots, so the copy stream repeatedly waits
on `slot_read` of the immediately preceding compute, which is exactly the
dependency the pipeline exists to avoid. Reverting to LRU reproduced 2048
tokens @5.1 t/s (398.4 s vs 399.5 s, 0.3%).

Offline bound worth knowing before retrying any policy: Belady's MIN on the
captured 125-token routing trace at 748 slots is **158 misses/token vs LRU's
283** — at this capacity recency is *not* the limiting factor, so a policy
retry needs a fence-aware victim rule (account for last-read time in the
victim metric) and MUST be validated at 2048 tokens, not 128.

### Remaining levers, ranked

1. **Eliminate the dequant (largest).** 101 ms/token, 71% of kernel time, is
   pure overhead: it materializes bf16 that is immediately re-read. The
   model-neutral `lsx_nvfp4` layer already streams packed NVFP4 weights with
   fp32 accumulate (`RunLsxNvfp4GemmSeq1GemmV`,
   `RunLsxNvfp4GemmGateUpPair`), and Kimi's slot layout is already that
   container (`[w1|w3|w2]` + per-16 pow2 E4M3 scales + global scale). What is
   missing is a bf16 activation path (the neutral entries take `__half`);
   precedent for a bf16 twin exists (`LsxTcQuantizeActsNvfp4Bf16`). If the
   routed+shared FFN moves off dequant, kernels fall 142 -> ~45 ms/token and
   the wall becomes the 120 ms copy stream: **~7-8 t/s** (estimate, unvalidated).
   Numeric caution: bf16-rounded dequant + cuBLAS becomes fp32 dequant with a
   different summation order — re-run the determinism gate and selfcheck
   (the design doc already exempts parity with the legacy GEMV chain, but
   determinism is binding).
2. **Free VRAM for slots** — the only way to cut the 120 ms of compulsory
   PCIe once compute is cheap. Per-expert NVFP4 (not bf16) on the *device*
   MLA projections is the historical candidate (~17.3 -> ~6 GiB host; the
   device copies are what matters here) for ~+470 slots (748 -> ~1218, +63%),
   on top of embed+head at 4.7 GiB bf16.
3. **TC batch (W4A4)** — still the biggest single jump, still blocked by the
   2026-09-12 root cause (swiglu output incompatible with per-16 E2M1
   pow2 quantization, QUANT hq rel = 120%). Per NVIDIA reference §B4 the
   higher-precision activation route is `kind::mxf8f6f4` with UE8M0 vec-32
   scales; `kind::mxf4nvf4` is E2M1-only, so this also means repacking the
   weights away from per-16 E4M3.
4. **Host-side per-layer sync** — 61 `KimiRouteFinish` busy-polls per token
   plus the per-step argmax readback. The router kernel itself is 70 us/layer.
   The pipeline now wants the routing as early as possible, so quantify the
   host timeline (nsys `cuda_api_sum`/OS-runtime) before changing it.
5. **Shared-expert FFN compute** — after residency its upload is gone but it
   still runs 3 dequant + 3 cuBLAS per token; subsumed by lever 1.

Dead ends, do not retry as-is: making the copy faster (57 GB/s is the bus
ceiling); prefetching the next layer from the previous token's routing
(measured reuse ~46%, and prefetch would inflate traffic that is already
bus-bound); LFU/ageing without a fence-aware victim rule.

### Reproduction

```bash
ninja -C build/linux-release-znver4-aocc lsxbenchmark test_kimi_moe_selfcheck
./build/linux-release-znver4-aocc/apps/lsxbenchmark/lsxbenchmark \
    --model /mnt/llm_models/models/Kimi-K2.7-Code --bench --max-gen 2048 \
    --prompt "The capital of France is"          # expect 2048 @ ~5.1 t/s, ttft ~1.4 s
./build/linux-release-znver4-aocc/tests/test_kimi_moe_selfcheck   # Paris x3
./build/linux-debug/tests/test_kimi_moe_expert_cache              # host-only policy/layout
# routing trace for offline cache-policy work (125 tokens -> 300 KB):
KIMI_TRACE_FILE=/tmp/kimi_trace.bin ./.../lsxbenchmark ... --max-gen 128
# profile (delay/duration windows silently produce NO report on this box):
nsys profile -t cuda --stats=false -o /tmp/kimi ./.../lsxbenchmark ... --max-gen 64
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format table /tmp/kimi.nsys-rep
```

Offline cache-policy analysis costs seconds and needs no GPU: the
`KIMI_TRACE_FILE` capture is a flat array of 10 int32 records
`{turn, layer, top8[8]}`, one per decode layer, so any policy can be
replayed by re-implementing `KimiMoeResolveBatch`'s loop (slot array +
`last_use` guard for LRU; slot array + per-slot count + same-batch guard for
LFU) in a script; Belady's MIN over the same stream is the optimal bound.
Keep the replay semantics identical to the shipped header — the first LFU
attempt was tuned on a subtly different (unguarded) simulator and its +15%
did not survive at 2048 tokens.

Environment notes: every process re-pins the 544 GiB routed store (~300 s,
34 x 16 GiB chunks; needs `ulimit -l unlimited`); `lsxbenchmark` is
GPU-exclusive; `ncu` counters are unavailable on this box, so kernel
arguments come from `nsys` plus standalone event-timed `.cu` probes.
