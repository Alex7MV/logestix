# Kimi-K2.7-Code round 2: un-park expert cache + MLA NVFP4 requant + TC batch engage

**Date:** 2026-09-08
**Branch:** `feat/kimi-100tps` (continuation; prior spec `2026-09-08-kimi-100tps-design.md`)
**Model:** `kimi-k2.7-code` (DeepSeek-V3-style MLA + MoE, 61 layers, 384 experts, top-8, H=7168)
**Build/test:** `build/linux-release-znver4-aocc`; build `cmake --build build/linux-release-znver4-aocc -j 160`, tests `ctest --test-dir build/linux-release-znver4-aocc -R <pattern>` (serial for inference tests).

## Goal

Raise default-path Kimi decode throughput from 1.4 t/s (H2D staging wall)
to the physically reachable ceiling, removing env-gates:

- un-park the device expert cache (root-cause slot corruption),
- free ~8 GiB VRAM via MLA NVFP4 requant (wider slot budget),
- re-engage the correctness-proven TC batch path.

Realistic target: prior measured analysis puts the cached ceiling at
4-15 t/s; the MLA requant adds ~342 slots to push toward the upper band.

## Operator decisions (binding)

1. Scope: cache un-park + MLA NVFP4 requant + TC batch engage
   (one iteration, strict order A -> B -> C).
2. Correctness contract: run-to-run determinism + valid "Paris" top-1.
   The selfcheck bit-exact reference is re-baselined ONCE after step B
   (NVFP4 requant changes logits legitimately); afterwards bit-exactness
   is enforced strictly again (round-25 precedent).
3. Requant scope: ALL MLA projections (q_a, q_b, kv_a, kv_b, o_proj);
   norms/biases/router/embeddings/lm_head stay BF16/F32 (AGENTS.md
   precision invariant).
4. Requant pipeline: converter path (lsxforge re-reads checkpoint,
   writes MLA projections as NVFP4 directly).
5. Format: hard break — new format rev; old rev is rejected with
   "reconvert required" (AGENTS.md no-backward-compatibility policy).

## Step A — Un-park the device expert cache

### Known facts (prior round)

- Cache slot bytes are bit-correct immediately after upload; read back
  differently later -> an unresolved writer between upload-retire and
  read. With the cache engaged, selfcheck generates "Domino!" vs "Paris";
  the bypass-staging path is bit-correct.
- Cache is parked behind `LSX_KIMI_EXPERT_CACHE_SLOTS` (default 0 =
  legacy per-call staging); `LSX_KIMI_CACHE_BYPASS` forces bypass.
- Suspects (in order): (1) copy-stream upload overwrites a slot the
  compute-stream is still reading (missing/incorrect event fencing on
  eviction + immediate re-upload of the same slot); (2) LRU eviction of
  a slot mid-read by the current Resident-GEMV batch; (3) stale
  gather-table meta cached against a re-uploaded slot.

### Diagnostics plan

- compute-sanitizer `racecheck` + `synccheck` on
  `tests/test_kimi_moe_experts` (cache-path case) and the selfcheck run
  with `LSX_KIMI_EXPERT_CACHE_SLOTS` override engaged.
- If the writer is still not exposed: targeted event-graph
  instrumentation (per-slot upload/read event log), one run, then removed
  (round-25 pattern: no committed diag hacks).

### Fix (root-cause driven only)

Expected shape: per-slot state machine
`free -> uploading -> ready -> retiring` with event fencing —
`cudaEventRecord` on the copy-stream after each slot upload +
`cudaStreamWaitEvent` on the compute-stream before the slot is read;
eviction is deferred until the slot retires (no overwrite of a
mid-generation slot). The exact fix is committed only after the
sanitizer pinpoints the site — no speculative fences.

**Gate A:** with the cache engaged by default, compute-sanitizer
0 errors + selfcheck "Paris" x3 bit-exact + `[kimi][cache][slot-bytes]`
byte-compare green.

## Step B — MLA NVFP4 requant (converter path, hard break)

### Geometry (frozen, do not re-derive)

Per layer: q_a [1536, 7168], q_b [12288, 1536], kv_a [576, 7168],
kv_b [16384, 512], o_proj [7168, 8192]. 61 layers.

- Today: BF16 pinned = 11.49 GiB total.
- NVFP4: ~3.59 GiB (E2M1 nibbles + pow2-E4M3 per-16 scales +
  float global_scale) -> frees ~7.9 GiB = ~342 extra expert slots
  (slot = 24,774,858 B).

### Converter (`apps/lsxforge --kimi`)

- MLA projection tensors are classified, quantized on GPU through the
  existing engine (`safetensors_gpu.cu`: E2M1 + pow2-E4M3 block scales +
  global_scale), and written into `pinned/layer_N.arrow` as
  `*_packed` (uint8 nibbles) + `*_scale` (uint8 E4M3 per-16) records;
  `global_scale` per tensor goes into `model.index.json`.
- Norms (q_a/kv_a/input/post_attn RMS), router + e_score_correction_bias
  (F32), embeddings, lm_head, final_norm remain BF16/F32.
- Format rev bumped; loader rejects the old rev with a strict
  "reconvert required" error (strict loader conventions).

### Loader (`kimi_moe_load.cpp`)

- The MLA tensor classifier recognizes the new `*_packed`/`*_scale`
  records; projections upload as NVFP4 records (same pinned-ring
  machinery); layer arenas account the new (smaller) sizes.
- Strict errors on missing tensors and rev mismatch stay.

### Kernels (`kimi_moe_inference.cu`)

- The 5 attention GEMV calls (q_a, q_b, kv_a, kv_b, o) switch from the
  BF16 scalar path to the model-neutral NVFP4 layer:
  `RunLsxNvfp4Gemm` / `RunLsxNvfp4GemmSeq1GemmV` (seq==1 decode ->
  streaming GEMV; attention GEMV speed improves as a side effect).
- RMSNorm pre/post, rope, absorbed-attention math unchanged.

**Gate B:** converter policy tests (MLA tensors NVFP4, norms untouched,
global_scale present), loader roundtrip, GEMV parity vs BF16 reference
within the TC-test tolerance class (rel-L2 < 0.35, synthetic), and the
ONCE-re-baselined selfcheck "Paris" x3 bit-exact recorded as the new
reference (test asserts bit-exactness across cycles vs itself).

## Step C — Re-engage TC batch + default-enabled cache

- `LSX_KIMI_EXPERT_CACHE_SLOTS` default flips from 0 to
  cache-enabled-from-free-VRAM (Qwen/glm-flash pattern) with a fixed
  reserve sized on step B measurements; the override remains as a
  debug tumbler.
- `RunKimiRoutedTcBatch` becomes the production routed path
  (`LSX_KIMI_TC_OFF` stays as a debug tumbler, default off).
- `LSX_KIMI_CACHE_BYPASS` remains a debug tumbler, default off.

**Gate C:** perf harness `test_kimi_moe_perf` logs tok/s (asserting no
silent regression: tok/s >= 3x the 1.4 staging baseline) + full suite
green + independence/naming guards green.

## Test plan (TDD)

| Step | Test | Gate |
|---|---|---|
| A | `[kimi][cache][slot-bytes]` byte-compare + selfcheck x3 with cache engaged | sanitizer 0 errors; Paris x3 bit-exact |
| A | new failing test first if the sanitizer exposes a fence-order defect (red on unfixed state) | red -> fix -> green |
| B | converter: MLA NVFP4 policy (classifier, global_scale, norms not requantized) | policy test red before converter change |
| B | loader roundtrip: new rev loads, old rev hard-rejects | roundtrip red before loader change |
| B | MLA GEMV parity vs BF16 reference (synthetic, tolerance class) | parity red before kernel switch |
| C | `test_kimi_moe_perf` threshold assert (>= 3x 1.4 t/s) | threshold red if perf regresses |
| final | `test_model_independence_guard`, `test_code_naming_guard`, full suite | 127+/127+ green serially |

Order is strict: A completes before B starts (the requant re-baselines
logits — diagnostics must run on stable bits). B completes before C
(the TC batch gate needs the widened slot budget to be measurable).

## Error handling

- Sanitizer findings root-cause first; no speculative fixes.
- Converter strictness: missing checkpoint tensors and geometry mismatches
  hard-fail (existing conventions).
- Loader rev mismatch: hard error naming the required reconvert.
- Any step whose gate fails twice is parked honestly with findings in
  the spec (round-25/25b precedent), no silent workarounds.

## Non-goals

- No new fast-math numerics beyond the NVFP4 requant itself.
- No two-token streaming reuse redesign (separate lever, next round).
- No backward compatibility for the old format rev.
- No parallel/VRAM-risky test runs (AGENTS.md inference exception).
