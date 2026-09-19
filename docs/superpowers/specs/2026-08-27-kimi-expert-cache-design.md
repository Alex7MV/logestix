# Spec: Kimi MoE — expert device cache + batch pipeline (unified VRAM policy)

Date: 2026-08-27
Status: approved by the operator
Premise: the centralized expert residency policy
(`lifecycle::ExpertCacheSlots`, `kExpertCacheVramReserveBytes = 4 GiB` in
`resource_governor.h`) is wired into qwen-moe, gemma, deepseek. Kimi streams
experts per-call without a device cache — this spec closes the last gap.

## 1. Problem

Kimi (61 layers × 384 experts × top-8, expert ≈ 22 MB INT4 pack, total
≈ 515 GB) loads expert weights from the host mmap (`ExpertStore`) to the device
on every token: 488 experts × 22 MB H2D copy + 3 GEMV, synchronously one at a
time (`RunKimiExpertFfn`, loop in `kimi_moe_inference.cu:947-959`).
≈ 10 GB of PCIe traffic per token — the dominant decode bottleneck.

## 2. Solution

### 2.1 Device slot cache (`kimi_moe_expert_cache`)

- `include/lsxcommon/kimi_moe_expert_cache.h` — host bookkeeping **without CUDA**:
  - slots from `lifecycle::ExpertCacheSlots(free, per_expert, total)`
    (per_expert = w1+w3+w2+scales ≈ 22 MB, total = 61×384 = 23424);
  - an explicit override `LSX_KIMI_EXPERT_CACHE_SLOTS` (parity with the other
    families); value 0 = cache disabled, legacy per-call staging path;
  - LRU eviction, flat key `layer * 384 + expert`;
  - decision API `ResolveBatch(layer, top8_ids)` → `{resident[slot], missing[]}`
    — residency of the whole top-8 layer in one call; a repeated call for the same
    experts is idempotent (all resident);
  - the decision API allocates nothing (zero-alloc contract).
- `src/kimi_moe_expert_cache.cu` — the CUDA part: slot arena allocation
  (pack + scale strides), batched async upload of missing on a copy stream,
  an event fence before reading the slot, LRU eviction with copy-stream
  synchronization (the in-flight DMA must complete before the slot is overwritten).
- A slot starts as prewarm zeros: before the first fill the slot is invalid.

### 2.2 Batch pipeline in generation

The top-8 layer loop (`kimi_moe_inference.cu:947-959`) is replaced with:

1. Pass 1 — `ResolveBatch(layer, top8_ids)` for all 8 experts;
2. Pass 2 — batched async upload of all missing on the copy stream in one packet
   (w1|w3|w2 + scales into cache slots); concurrently on the compute stream
   GEMVs of already-resident experts run;
3. Pass 3 — event-wait for the upload to finish, GEMV of the rest from the slots.

The GEMV accumulation order in `acc` is preserved (determinism; the top-k
processing order does not affect the result — addition into the FP32 accumulator
is performed in a fixed order of k indices, not in the order the copies finish).

### 2.3 Fallback and degradation

- Cache allocation at startup: shrink-retry by half (qwen/gemma parity);
  0 slots → legacy per-call staging path (current behaviour) + warn.
- An explicit `LSX_KIMI_EXPERT_CACHE_SLOTS=0` — the same legacy path, no warn.

## 3. Bit-exactness

The GEMV kernel reads identical packed bytes from the cache slot instead of the
staging buffer — the input is bit-for-bit identical, the kernel is the same ⇒ the
outputs are bit-exact. The device copy does not transform the data. The best
proof is the real selfcheck `test_kimi_moe_selfcheck` (on machines with the
model): bit-exact determinism of `final_logits` is preserved (the test SKIPs
without a model).

## 4. Tests (TDD)

- New `tests/test_kimi_moe_expert_cache.cpp` (host-only, Catch2):
  - policy wiring: slots from `lifecycle::ExpertCacheSlots`, clamp to 23424;
  - env override: explicit > policy; `0` = disable;
  - LRU: eviction order on overflow, reuse of the hot one;
  - `ResolveBatch`: resident/missing classification, idempotency,
    handling of invalid indices (negative, out of range → invalid);
  - the zero-alloc contract of the decision API;
  - batch upload plan: the missing byte ranges cover all expert records
    (w1, w3, w2, scales) and do not overlap.
- Regression: existing host tests `test_kimi_*` green; naming guard
  green (new names follow the scheme, the allowlist is not extended).

## 5. Build

- `src/kimi_moe_expert_cache.cu` → OBJECT `lsxcommon_kimi_cu`
  (`libs/lsxcommon/CMakeLists.txt`), the object flags are unchanged
  (`--extended-lambda; -fmad=false`).
- The new test in `tests/CMakeLists.txt` (host-only, without lsxcommon CUDA targets —
  only the bookkeeping header).
- Full windows-debug rebuild exit=0.

## 6. Definition of done

- `test_kimi_moe_expert_cache` green; governor policy tests green;
  naming guard green; the full host set without regressions.
- A parity log line at startup: `[kimi-infer] expert cache: free=...
  per_expert=... total=... slots=...`.
- On machines with the Kimi model: selfcheck bit-exact for 3 cycles (outside the
  scope of this machine, noted as follow-up verification).
- An explicit limitation recorded in logs/comment: with 16 GiB VRAM the cache
  covers ~2% of the experts — the effect depends on routing hotness; the main win
  is in any case the batched upload instead of 8 synchronous H2D per layer.
