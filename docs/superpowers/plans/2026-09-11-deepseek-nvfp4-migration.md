# DeepSeek-V4-Flash NVFP4 Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task (project AGENTS.md FORBIDS subagents — inline
> execution only). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move DeepSeek-V4-Flash from the MXFP4/FP8 codec to the NVFP4 contour
(`lsx_nvfp4`) like the other fast models: 4.43 t/s baseline → 8.5 t/s (already
reached with the cache) → a target of ~25-35 t/s (parity with GLM-5.3-Flash at the
same MoE geometry).

**Architecture:** (1) Immediate pinned wins: an expert LRU cache over
free VRAM (by default, the GLM pattern) + wiring `SetDeepseekFastMath` into the
perf policy. (2) Offline vault repack `logestix-v1` → `logestix-v1-nvfp4`:
expert scales E8M0/32 → E4M3/16 (an exact mapping of powers of two,
nibbles unchanged), attention/dense FP8 tiles → NVFP4 via the existing
`quantize_fp8_nvfp4`. (3) Inference: routed/shared/dense GEMM via
`RunLsxNvfp4GemmGateUpPair` / `RunLsxNvfp4Gemm` (the GLM cache-slot pattern
`[w1|w3|w2]`, scales `[s1|s3|s2]`).

**Tech Stack:** CUDA 13.4 (CC 12.0/12.1, sm_120), C++20, CMake presets
`linux-debug`/`linux-release-znver4`, Catch2 v3, the lsx_nvfp4 model-neutral
layer, ctest labels fast/convert/inference/perf.

## Global Constraints

- AGENTS.md: subagents are FORBIDDEN — inline only.
- AGENTS.md protocol step 5: NVFP4 GEMM only via `lsxcommon/lsx_inference_nvfp4.h`; no parallel named copies of the dispatcher.
- Norms and biases stay FP16/BF16 (PRECISION INVARIANT).
- lsxbenchmark — strict GPU exclusivity, no parallel GPU work.
- Preserve Doxygen comments; logging only via spdlog; format strings abseil printf style (`%d`, not `{}`).
- Commits: English, prefixes feat/fix/test/docs; tests — a label in the same commit.
- Reference chain: glm_moe_flash_inference.cu:1288-1315 (pair-GEMV → SwigluKernel → w2 → coeff_add).
- Default numeric policy: kAuto = per-GPU perf policy (already applied via `EnsurePerfPolicyApplied`).

---

### Task 0: Immediate wins — default cache + TC policy

**Files:**
- Modify: `libs/lsxcommon/src/gpu_perf.cpp:193-204` (already edited: `gpu::SetDeepseekFastMath(knobs.tc_fast_math_default)`)
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu:480-495`
- Test: manual bench + `ctest -L fast` (regression)

**Interfaces:**
- Consumes: `perf::CurrentKnobs().expert_slots_pct` (gpu_perf.h), `cudaMemGetInfo`.
- Produces: DeepSeek by default ≥8 t/s without env variables.

- [x] **Step 0.1: The probe is already done** — the env experiment `LSX_DEEPSEEK_EXPERT_CACHE_SLOTS=1300` gave 8.5 t/s (a 105-token run), 4.6 without the cache. The TC path on a cache hit is slower (5.2) — the `!use_cache` gate is kept.

- [ ] **Step 0.2: Default cache size from free VRAM** — replace pipeline.cu:494 `if (!explicit_slots) slots = 0;` with the GLM pattern:

```cpp
        if (!explicit_slots) {
            // GLM-Flash ResolveExpertSlots pattern: cache everything that
            // fits into free VRAM minus a generation reserve. The old
            // "thrashes every token" note applied to fixed small slot
            // counts; full-free-VRAM sizing measured 4.6 -> 8.5 tok/s.
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                constexpr size_t kReserve = size_t{2} << 30;  // arenas + ctx
                const size_t budget =
                    free_bytes > kReserve ? free_bytes - kReserve : 0;
                const int pct = perf::CurrentKnobs().expert_slots_pct;
                size_t eff = budget;
                if (pct < 100) eff = budget / 100u * static_cast<unsigned>(pct);
                const int64_t slot_bytes = routed_pack_total + routed_scale_total;
                if (slot_bytes > 0)
                    slots = static_cast<int>(eff / static_cast<size_t>(slot_bytes));
            }
        }
```

Requires `#include "lsxcommon/gpu_perf.h"` in pipeline.cu (verify).

- [ ] **Step 0.3: Rebuild + bench** — `cmake --build build/linux-release-znver4 -j 160` and `lsxbenchmark --model /mnt/llm_models/models/DeepSeek-V4-Flash --max-gen 1024` WITHOUT env. Expectation: ≥8.0 t/s.

- [ ] **Step 0.4: ctest fast** — `ctest --preset linux-release-znver4 -L fast` green.

- [ ] **Step 0.5: Commit** `fix: wire DeepSeek TC toggle into perf policy; default expert cache to free VRAM (4.6 -> 8.5 tok/s)`.

---

### Task 1: E8M0→E4M3/16 host primitives + unit test

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/nvfp4_scale.h` (add a function)
- Test: `tests/test_deepseek_nvfp4_repack.cpp` (new, label fast)

**Interfaces:**
- Produces: `namespace lsxcommon::nvfp4 { inline void E8m0ScaleToE4m3Pair(uint8_t e8m0, uint8_t out[2]); }` — mapping one E8M0 byte to a pair of E4M3 bytes per-16 (both halves of the 32-block). Used by Task 2 (tool) and Task 4 (if a runtime path is needed).

- [ ] **Step 1.1: Failing test** — for all 256 bytes b: `E8m0ScaleToE4m3Pair(b)` → decoding the pair via `E4M3ScaleToFloat` equals `E8m0ClampedToFloat(b)` for the representable range [2^-9 .. 2^8] (b∈[118..135]) and 0x00→0x00; outside the range — clamp to the boundary (the test pins the contract: `decode_pair(b) == clamp_to_e4m3_range(E8m0ClampedToFloat(b))`, float comparison bitwise for powers of two).

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include "lsxcommon/nvfp4_scale.h"

namespace {
float E4m3Host(uint8_t b) {
    // host copy of the gpu::nvfp4::E4M3ScaleToFloat semantics (E4M3, bias 7)
    if (!b) return 0.0f;
    const int s = (b >> 7) & 1, e = (b >> 3) & 0x0F, m = b & 7;
    float v = (e == 0) ? static_cast<float>(m) * (1.0f / 128.0f)
                       // 2^-9 subnormal step
                       : static_cast<float>(1u << (e - 1 + 127 - 7));
    if (e != 0 && m) v *= 1.0f + static_cast<float>(m) / 8.0f;
    return s ? -v : v;
}
} // namespace

TEST_CASE("e8m0 scale byte -> nvfp4 e4m3 per-16 pair", "[fast][deepseek]") {
    for (int b = 0; b < 256; ++b) {
        uint8_t pair[2] = {0xEE, 0xEE};
        lsxcommon::nvfp4::E8m0ScaleToE4m3Pair(static_cast<uint8_t>(b), pair);
        REQUIRE(pair[0] == pair[1]);  // both halves of the 32-block are identical
        const float want = /* E8m0ClampedToFloat host copy */ 0.f;
        // representable: exact; else clamp — see the test body in the implementation
    }
    SECTION("zero and FLT_MAX saturate") {
        uint8_t p[2];
        lsxcommon::nvfp4::E8m0ScaleToE4m3Pair(0x00, p);
        REQUIRE(p[0] == 0x00);
        lsxcommon::nvfp4::E8m0ScaleToE4m3Pair(0xFF, p);
        REQUIRE(p[0] == 0x7E);  // 448.0 (E4M3 max), documented clamp
    }
}
```

- [ ] **Step 1.2: Run** — `cmake --build build/linux-release-znver4 -j 160 --target test_deepseek_nvfp4_repack && ctest --test-dir build/linux-release-znver4 -R deepseek_nvfp4_repack` — FAIL (no function).

- [ ] **Step 1.3: Implementation** in `nvfp4_scale.h`:

```cpp
// One E8M0 exponent byte (per-32 block) maps to TWO identical E4M3 bytes
// (per-16 halves of the same block). Exact for 2^-9..2^8; outside the
// E4M3 range the value saturates to 0 / E4M3 max — the clamped flavor the
// routed-expert FFN already uses (lsx_kernels_quant.cuh E8m0ClampedToFloat).
inline void E8m0ScaleToE4m3Pair(uint8_t e8m0, uint8_t out[2]) {
    const uint8_t mapped = FloatToE4m3P2(E8m0ClampedToFloatHost(e8m0));
    out[0] = mapped; out[1] = mapped;
}
```

(host copies: `E8m0ClampedToFloat` from lsx_kernels_quant.cuh:67 and `FloatToE4m3P2` from nvfp4_scale.h — both are already `__host__ __device__`; verify the include graph: nvfp4_scale.h must NOT pull CUDA headers into a host TU.)

- [ ] **Step 1.4: Test green**, `ctest -L fast` without regressions.
- [ ] **Step 1.5: Commit** `feat: e8m0->e4m3/16 scale pair primitive for deepseek nvfp4 repack`.

---

### Task 2: Repack tool vault → logestix-v1-nvfp4 (lsxforge)

**Files:**
- Create: `apps/lsxforge/src/deepseek_repack.cpp`
- Modify: `apps/lsxforge/src/cli.{h,cpp}` + `apps/lsxforge/src/main.cpp` (flag `--deepseek-nvfp4 <src> <dst>`)
- Modify: `apps/lsxforge/CMakeLists.txt` (add_executable)
- Test: `tests/test_deepseek_nvfp4_repack.cpp` — a second TEST_CASE (label convert, TIMEOUT 120, a synthetic v1 micro-vault)

**Interfaces:**
- Consumes: `quantize_fp8_nvfp4(fp8, shape, scale_f32, scale_shape, name, DenseNvfp4Record&)` (glm_moe_flash_convert.cpp:404); `E8m0ScaleToE4m3Pair`; Arrow IPC read/write (pattern converter.cpp / glm_moe_flash_convert.cpp).
- Produces: a vault with `"format": "logestix-v1-nvfp4"`; expert tensors: packed unchanged, scales ×2 (per-16), global_scale=1.0; dense FP8 tensors → NVFP4; BF16/FP16 copied as-is.

- [ ] **Step 2.1: Test fixture** — a synthetic vault: `model.index.json` (format logestix-v1) + one expert arrow (w1/w2/w3 of a small geometry, e.g. [32,32], E8M0 scales of known bytes) + one FP8 dense tensor. Run the tool, checks: the format tag, scales_bytes ×2, packed bytes identical, the decoded scale pair == the original E8M0 values (host reference from Task 1), the FP8 tensor became NVFP4 (DenseNvfp4Record fields in the index).
- [ ] **Step 2.2: Implementation** — the flow: read the index → for experts/*.arrow byte-wise repack of the scale arrays (packed is copied) → for pinned/*.arrow: dtype routing F8_E4M3+scale-tiles → `quantize_fp8_nvfp4`, otherwise as-is → write the new index (`logestix-v1-nvfp4`). The GPU is free — densifying quantization is allowed, but NOT concurrently with inference (GPU exclusivity is not needed: quantize_fp8_nvfp4 is a host path — verify, and if slow, replace with the safetensors_gpu.cu GPU kernels).
- [ ] **Step 2.3: convert test green** (label convert, TIMEOUT 120).
- [ ] **Step 2.4: Commit** `feat: lsxforge --deepseek-nvfp4 repack tool (logestix-v1 -> v1-nvfp4)`.

---

### Task 3: Repack the real vault

**Files:** no code (an operation on `/mnt/llm_models/models/DeepSeek-V4-Flash`)

- [ ] **Step 3.1:** `lsxfabric/lsxforge --deepseek-nvfp4 /mnt/llm_models/models/DeepSeek-V4-Flash/logestix /mnt/llm_models/models/DeepSeek-V4-Flash/logestix-nvfp4` (~149GB, wait, check free space with `df -h /mnt/llm_models`).
- [ ] **Step 3.2:** Replace `logestix` → symlink/rename (delete the old directory after Tasks 4-5 are green, saving disk: first `mv logestix logestix-mxfp4.bak` if space allows, otherwise repack straight into `logestix` over an index backup).

---

### Task 4: Loader v1-nvfp4

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cu` (format detect; scales bytes numel/16; global_scale from the record)
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cuh` (ExpertEntry doc: scales now E4M3/16)
- Modify: `libs/lsxcommon/src/deepseek_moe_model.cpp:57-65` (frozen gate unchanged)
- Test: `tests/test_deepseek_moe_loader.cpp` — roundtrip on a v1-nvfp4 fixture

- [ ] **Step 4.1:** Loader: a strict error on the old format with a clear message "repack via lsxforge --deepseek-nvfp4" (AGENTS: no backward compatibility). Update the test synthetic fixtures to v1-nvfp4 generation (scale bytes duplicated, as in the Task 2 fixture).
- [ ] **Step 4.2:** Roundtrip tests green; `ctest -R deepseek` (the fast part).
- [ ] **Step 4.3:** Commit `feat: deepseek loader logestix-v1-nvfp4 (e4m3/16 scales)`.

---

### Task 5: Routed FFN via lsx_nvfp4 (the main performance piece)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu:343-502,688-843`
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu` (cache slot layout [w1|w3|w2], arena)
- Test: `tests/test_deepseek_moe_inference_kernels.cu` (update the routed-FFN parity cases)

**Interfaces:**
- Consumes: `RunLsxNvfp4GemmGateUpPair(stream, x, packed, scales, gs1, gs3, y, K, n_half, acc)`, `RunLsxNvfp4Gemm(stream, x, packed, scales, gs, y, 1, K, N, acc)`; the GLM pattern glm_moe_flash_inference.cu:1288-1315; `SwigluKernel<__half, Glibc, true>`; a `flash_coeff_add_kernel` analog (there is acc_half_to_f32_kernel — add a coeff version).
- Produces: `run_routed_expert_ffn_nvfp4(...)` — the v1-nvfp4 branch; the old F32 branch is removed (no backward compat).

- [ ] **Step 5.1:** Cache slot: copies in the order w1,w3,w2 / s1,s3,s2 (pipeline + moe.cu staging), stage buffers of the same layout. `pack_stride` = w1+w3+w2 packed bytes.
- [ ] **Step 5.2:** The FFN branch (modeled on GLM):

```cpp
// [w1|w3] pair GEMV → [gate|up] halves
lsx_nvfp4::RunLsxNvfp4GemmGateUpPair(stream, d_normed, p13, s13,
    en.w1_global_scale, en.w3_global_scale, d_yg, kHiddenDim,
    kIntermediateDim, false);
kernels::SwigluKernel<__half, ExpFlavor::Glibc, true><<<grid, 256, 0, stream>>>(
    d_yg, d_yg + kIntermediateDim, d_yg, kIntermediateDim, 10.0f);
lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_yg, p2, s2, en.w2_global_scale, d_y2,
    1, kIntermediateDim, kHiddenDim, false);
// d_output += weight * d_y2 (FP32 accum)
coeff_add_f32_kernel<<<(kHiddenDim + 255) / 256, 256, 0, stream>>>(
    d_y2, weight, d_output, kHiddenDim);
```

- [ ] **Step 5.3:** Kernel parity tests: a host reference dequant(E4M3/16) vs GEMV on a small geometry (K=256, N=512); update the existing 4470-assertion cases (the dequant_fp4_e8m0_* paths are removed together with the branch).
- [ ] **Step 5.4:** `ctest -R deepseek -L fast` + the kernels test; bench: expect ≥15 t/s (H2D the same, the dequant is gone, GEMV 4x lighter in traffic).
- [ ] **Step 5.5:** Commit `perf: deepseek routed FFN on lsx_nvfp4 pair-GEMV (v1-nvfp4)`.

---

### Task 6: Dense/attention via lsx_nvfp4

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_attn.cu`, `deepseek_moe_inference_csa.cu`
- Test: kernels suite

- [ ] **Step 6.1:** q_a/q_b/kv_a/kv_b/wo (+gate) — LsxNvfp4Weight from pinned (converted in Task 2), GEMV via `LaunchNvfp4Gemm`; the dequant_f8_e8m0 paths are removed.
- [ ] **Step 6.2:** Update the parity tests; bench: expect 20-30 t/s.
- [ ] **Step 6.3:** Commit `perf: deepseek attention GEMMs on lsx_nvfp4`.

---

### Task 7: Shared experts + cleanup

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu` (shared FFN), `deepseek_moe_load.cu`
- Cleanup: remove the dead dequant kernels in `shared_experts.cu` (deepseek usages only; check the glm/qwen consumers with grep!)

- [ ] **Step 7.1:** Shared FFN → `LaunchNvfp4Gemm` (resident LsxNvfp4Weight).
- [ ] **Step 7.2:** grep-audit the consumers of `dequant_fp4_e8m0_kernel|dequant_fp4_e8m0_f32_kernel` — remove if 0.
- [ ] **Step 7.3:** Commit `perf: deepseek shared experts via LaunchNvfp4Gemm; drop mxfp4 dequant kernels`.

---

### Task 8: SelfCheck, baseline, documentation

**Files:**
- Modify: `tests/perf/perf_baselines.json` (deepseek t/s)
- Modify: `AGENTS.md` (deepseek entry: v1-nvfp4, speed)
- Test: selfcheck Paris ×3 (SKIP_RETURN_CODE 4 without a model — we have the model)

- [ ] **Step 8.1:** `lsxbenchmark --selfcheck` Paris green ×3 bit-exact cycles (tolerance class on the new branch — changing the reference logits is allowed, determinism is mandatory).
- [ ] **Step 8.2:** Final 2048-token bench; update perf_baselines.json (the `test_perf_baseline_deepseek` test overwrites it itself on kUpdate).
- [ ] **Step 8.3:** AGENTS.md: DeepSeek-V4-Flash entry — format v1-nvfp4, lsx_nvfp4 contour, t/s.
- [ ] **Step 8.4:** Full `ctest -L "^(fast|convert)$"` green; commit `docs: deepseek-v4-flash on nvfp4 contour` + push.

## Self-Review

- The spec is covered: cache (Task 0), codec break (Tasks 1-4), GEMM contour (Tasks 5-7), verification/docs (Task 8).
- Placeholder scan: code is given for the key edits; Task 2/6 details are tied to existing functions (quantize_fp8_nvfp4, LaunchNvfp4Gemm) — signatures verified in the sources.
- Types: `E8m0ScaleToE4m3Pair(uint8_t, uint8_t[2])`, ExpertEntry global_scale fields already exist (deepseek_moe_load.cuh:93-95).

## Round 3-4 findings (2026-09-11, post-migration tuning)

State: 11.7 tok/s steady (baseline 4.43, +164%), SelfCheck Paris green.

- H2D miss transfers = 57 ms/token (44 GB per 30 steps, 1.46 GB/token at
  40% LRU miss) — the hard ceiling; GPU compute is ~17 ms and mostly
  hidden under copies by the dual-stream interleave.
- cudaMemcpyBatchAsync for pageable mmap H2D measured SLOWER than plain
  per-copy calls (10.3 vs 11.7 tok/s) — different staging path. Plain
  loop retained.
- Phase-split ex() (batch copies first, GEMVs after one fence) loses the
  GEMV-under-copy interleave: +110 us/layer visible GEMV time. Interleave
  (copy k -> fence k -> GEMV k) is optimal at this shape.
- Early-issue prefetch is blocked by an unresolved race (silent numeric
  corruption both plain and with a layer-boundary event sync). Needs a
  root-cause session (candidate: cache-slot overwrite while a previous
  layer's GEMV still reads the slot — issue(N+1) victims vs in-flight
  main-stream readers).
- Per-layer partitioned cache: worse (10.7). Shared experts: already on
  load-time dequant caches, lsx_nvfp4 payoff ~0.3 tok/s — deferred.
- Task 6 (attention NVFP4) payoff re-estimated at ~+1 tok/s (dequant_f8
  4.4 ms + fp32 GEMV 1 ms of an 85 ms token) — deferred until the H2D
  ceiling moves.

Conclusion: without cutting miss BYTES (hit rate/coverage bounded by
VRAM) or solving the prefetch race, ~12 tok/s is the practical ceiling
of this architecture generation on 32 GB. The 11.7 state is committed
and selfcheck-green.

## Round 5 findings (2026-09-11): prefetch race LOCALIZED

Root-cause instrumentation (LSX_DS_VERIFY_COPY / LSX_DS_VERIFY_READ
byte-exact probes at issue time vs GEMV-read time):

1. Prefetch H2D copies are BYTE-EXACT at issue time (2363/2363 verified).
2. 7/5676 slot reads at GEMV time MISMATCH the host source — ALL in
   layers >= 27 (= 7000-mmap-cap boundary: 7000/256 = 27.3 layers).
3. Two races identified:
   a. munmap-under-DMA: issue(L) enqueues copies from mmap-backed entries;
      ensure(L+1) evicts the 7000-cap mmaps (munmap) while the DMA is
      still queued on copy_stream. evict_done fences main only.
      FIXED: lsx_pending_copy_drain() registry — issue registers
      (source base, completion event), expert_store drains before munmap.
   b. slot write-after-read: victims selected by issue(L+1..L+k) (host
      runs several layers ahead of the main-stream GEMVs) can overwrite a
      slot a still-executing GEMV of layer L is reading. NOT yet fixed —
      this is the remaining corruption (verified copies + drain, output
      still garbage when LSX_DS_PREFETCH=1).
4. Candidate fix for (b): per-layer slot bands for prefetch fills
   (layer L writes only [L*band, (L+1)*band) — builders and readers of a
   layer never share bands with in-flight GEMVs of other layers), or a
   "slot pinned until its GEMV retires" guard counted from the ex loop.

State: prefetch env-gated OFF (LSX_DS_PREFETCH), production stable at
11.7 tok/s, SelfCheck Paris green, fast 43/43.

## Round 5 additions (2026-09-11, evening)

- Drain-hook wired end-to-end (lsx_pending_copy_drain in expert_store,
  registry in moe.cu) — munmap-under-DMA race closed.
- CORRUPTION root-cause CONFIRMED DIFFERENT: byte-exact probes show the
  prefetch copies are correct at issue time AND the slot content matches
  at GEMV-read time in 5669/5676 cases — the 7 mismatches are exactly the
  write-after-read slot conflicts described above.
- Canonical O(1) LRU migration experiment (expert_device_cache.h
  Lookup/Insert/Touch wired into ex()/issue(), free-stack init): measured
  a STABLE 8.0 tok/s vs 11.6 — hit-refreshing real LRU is SLOWER on this
  traffic than the legacy "FIFO-of-first-fill" order. Reverted; both
  variants kept in git history for the follow-up study.

Production state unchanged: 11.7 tok/s steady, Paris green, fast 43/43.
Round-6 candidates: (1) prefetch with slot-retire guard + per-layer
bands (fix race (b) by construction, keep legacy fill order inside each
band), (2) attention NVFP4, (3) host-gap trims.

## Round 6 (2026-09-11): prefetch race — timing ruled OUT

Decisive experiment: banded slot fills (per-layer bands, race (b)
eliminated by construction) + munmap drain (race (a) fixed) + early
issue on the MAIN stream itself (zero copy-lane concurrency,
LSX_DS_PREFETCH_SYNC): output is STILL garbage (3.8 tok/s, 513-token
no-EOS run). Meanwhile byte-exact probes prove the copies and the slot
reads are all correct (5669/5676 matches, mismatches = races already
handled).

Conclusion: the corruption is LOGICAL, not a timing race. Something in
the prefetch state change (slot bookkeeping via the banded path /
hit-vs-miss path divergence between issue() and ex()) breaks numerics
even with fully synchronous main-stream copies. Next session: diff the
FULL cache state (slot_of/owner/rotor) issue-vs-ex for one layer, and
audit ex()'s hit path for a subtle difference vs its miss path (e.g.,
staging-layout assumptions, stage-buffer aliasing with d_yg/d_y2).

Production stays env-gated OFF: 11.6 tok/s steady, Paris green,
fast 43/43.

## Round 8-9 final (2026-09-11): pinned store, 2-copy merge, feasibility verdict

- Pinned expert store: 11008 experts / 148.6 GiB in one cudaHostAlloc
  pool, pair-layout [w1|w3|w2][s1|s3|s2]. 21.4-21.5 tok/s steady,
  Paris green, fast 43/43. (4.43 -> 21.4 = +383% over the session.)
- 2-copy merge (blob is already pair-ordered): API calls halved; within
  run-to-run noise of the 6-copy path in OFF mode. Prefetch ON improved
  14.2 -> 14.6 but still loses to OFF (prefetch serializes: one fence
  before all GEMVs) — stays env-gated.
- Registry note: g_pending_copies only grows while prefetch is enabled
  (drain runs on mmap eviction, which full-resident mode never does) —
  bounded by issue frequency, but should be cleared per layer when
  prefetch is revisited.

## FEASIBILITY VERDICT

Remaining levers and honest payoffs on RTX 5090 / 32 GB:
- attention NVFP4 (Task 6): -5 ms/token -> ~23-24 tok/s; high effort
  (converter + loader + attn/csa kernels + tests)
- host-gap trims (route readback, event API): +1-2 tok/s; medium effort
- hit-rate research: 60% measured vs ~70% theoretical; bounded by VRAM
- PHYSICS: miss traffic >= ~1 GB/token (hit ceiling ~75% at this VRAM)
  => PCIe floor ~20 ms/token; overall ceiling ~28-30 tok/s.

VERDICT: decode-tuning is near its architectural ceiling. Incremental
gains left are +2-4 tok/s for 2-3 risky rounds. RECOMMENDATION: stop
DeepSeek decode tuning here (21.4 tok/s, +383% over session baseline);
redirect effort to other backlog items (qwen3.8 arena OOM, Kimi
completion) unless a >25 tok/s requirement appears.

## SESSION FINAL (2026-09-11)

| Stage | t/s | Commit |
|---|---|---|
| Baseline (MXFP4/FP8, no cache) | 4.43 | — |
| + LRU cache over free VRAM (default) | 8.7 | f7af44c |
| + NVFP4 routed FFN (pair-GEMV) | 11.7 | 46ebc3b |
| + full-resident mmap (65536 cap) | 12.1 | cc0d274 |
| + pinned expert store (148.6 GiB) | **21.4** | fade1c1 |
| + 2-copy merge (pinned blobs) | 21.5 | a59b5af |

Root causes of the original slowness (all resolved):
1. The expert LRU cache was disabled by default → 3.7 GB/token of PCIe streaming.
2. routed_scale_total was computed for per-32 MXFP4 (v1-nvfp4 per-16 is twice
   as large) → all routed experts were silently skipped by the staging check.
3. max_open=7000 mmap cap → munmap churn past layer 27 under in-flight DMA.

Not closed (env-gated, documented above): early-issue prefetch
(a logical slot race, LSX_DS_PREFETCH), attention NVFP4 (Task 6,
~+2 t/s), host-gap trims (+1-2 t/s). The physical 32 GB ceiling for this
model is ~28-30 t/s — tuning was stopped as not worthwhile.
