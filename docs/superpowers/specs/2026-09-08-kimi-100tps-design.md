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

- **Neutral-layer migration**: `QuantizeActsNvfp4`, `RunQwenTcGemmPreQuant`,
  `RunQwenTcGemmBatch13`, `RunQwenTcGemmBatchDown`, `QwenTcBatchSlot` gain
  neutral `lsx_nvfp4` wrappers (single definition site in
  `lsx_inference_nvfp4.h/.cu`); `qwen::tc` remains the internal
  implementation. Qwen and Kimi both call the neutral API. No kimi-named
  dispatcher copies (AGENTS.md NVFP4 GEMM rule).
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
  sm_120a — docs §B4). Runtime probe `TcRuntimeSupported()` gates
  engagement; `cudaErrorNotSupported` → legacy path.
- The expert slab layout in cache slots is the EXISTING checkpoint NVFP4
  packing (no repack at upload for Phase 1 — the TC kernel consumes the
  repo layout, like Qwen).

### Phase 2 — Device slot table (router fence elimination)

- `d_routes` (top-8 ids on device) + per-flat-expert TC-meta
  `{dev_slot, gs1, gs3, gs2, coeff}` staged at slot insert into
  `KimiMoeExpertDeviceCache` (mirrors Qwen `d_gemv_meta` / round-19c).
- The per-slot table is composed ON DEVICE from `d_routes`; in
  steady-state the host never reads the router back →
  `KimiRouteBegin/Finish` busy-poll leaves the hot path.
- Engagement gate: cache active AND TC fast-math AND fused slot layout
  AND `I % 16 == 0`. Any miss breaks steady-state → the driver falls
  back for that generation to the legacy per-call staging path
  (bit-identical fallback, already the code path today).

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

### Phase 5 — Hybrid residency + VRAM reserve revision

- Governor reserve for Kimi is re-measured and tightened (the
  `LSX_KIMI_EXPERT_CACHE_SLOTS` override stays; the default comes from
  `lifecycle::ExpertCacheSlots`) → more device slots.
- TC-meta (`{gs1,gs3,gs2}` + slot binding) staged at insert — a tiny
  per-slot on-upload transform, bit-exact conversion test pinned.
- Hot-expert hinting (runtime tuner already exists) may bias LRU
  bookkeeping — optional, only if the measured hit-rate is the residual
  bottleneck.

### Model-independence amendments (guard tests, same commit)

- `tests/test_model_independence_guard.cpp`:
  - `consumers` += `kimi_moe_model.cpp` (must open `LsxFastMathScope(cfg_.fast_math, ...)` in Infer);
  - fast-math toggle check: kimi TUs define no `g_kimi_fast_math`-style
    state; the single definition stays in `lsx_inference_nvfp4.cu`.
- `tests/test_code_naming_guard.cpp` registries unchanged (no new files
  outside the kimi family; neutral wrappers live in
  `lsx_inference_nvfp4.*`).

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
