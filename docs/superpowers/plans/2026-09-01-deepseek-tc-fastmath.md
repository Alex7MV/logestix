# DeepSeek TC W4A4 fast-math Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (subagent dispatch is FORBIDDEN by AGENTS.md — execute inline). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise DeepSeek-V4-Flash decode throughput on RTX 5090 by routing GEMMs through Blackwell tensor cores: TC W4A4 routed-expert FFN under `fast_math`, bit-exact dequant caching for shared/attention F8 sites, and fixing the LRU-cache thrash that currently costs 4.3x (F-12).

**Architecture:** Reuse the proven `RunQwenTcNvfp4Gemm` mxf4nvf4 kernel via load/stage-time scale repack (E8M0 per-32 → UE4M3 per-16, E2M1 nibbles unchanged). Toggle = existing `ModelInitConfig::fast_math` (process-wide, default OFF = bit-exact legacy). All GEMM sites covered in two phases: routed FFN first (dominant), then shared expert + attention dequant caching (always-on, bit-exact).

**Tech Stack:** CUDA 13.3, sm_120a PTX `mma.sync...kind::mxf4nvf4.block_scale` (via existing qwen tc kernels), Catch2 v3, CMake preset `linux-release-znver4-aocc`.

## Global Constraints

- Branch: `feat/nohandshake-graphs-5090` (already checked out; all commits here).
- Build dir: `build/linux-release-znver4-aocc` (AOCC 5.2.0, `-march=znver4`, LTO). Build single targets with `cmake --build build/linux-release-znver4-aocc --target <t> -j 160`.
- Default (fast_math=OFF) routing MUST stay bit-exact: `test_deepseek_selfcheck` 3-cycle byte-identical `final_logits`.
- Subagents forbidden (AGENTS.md). All work inline in this session.
- No `std::cout`/`printf` — spdlog only. No comments unless adapting existing Doxygen.
- Naming guard: no new files under `libs/lsxcommon/{src,include}` — all edits inside existing `deepseek_moe_*` / `shared_experts.*` units (`tests/test_code_naming_guard` must stay green).
- NVFP4 facts (docs/NVIDIA_DOCS_NVFP4_MEMORY.md §B2): `kind::mxf4nvf4` SF=UE4M3 vec-16; UE4M3 decode `2^(E-7)`, `E = e8 − 120`; scale byte range exact for E∈[1..15] i.e. e8∈[121..135]; E=0 decodes 0.
- DeepSeek routed weights: row-major [rows, cols], packed E2M1 nibbles [rows, cols/2] (low nibble = even element), E8M0 scale per 32 elements, flat scale index `i >> 5` (cols % 32 == 0). Geometries: kHiddenDim=4096, kIntermediateDim=2048, kTopK=6, 43 layers × 256 experts.
- Inference runs on real weights are resource-critical (AGENTS.md): never run two concurrently; GPU must be idle before starting (`nvidia-smi` shows ~33 MiB used).

---

### Task 0: Default DeepSeek expert cache to streaming (slots=0)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu:473-497`

**Interfaces:**
- Consumes: existing env override `LSX_DEEPSEEK_EXPERT_CACHE_SLOTS` (verbatim semantics preserved).
- Produces: default `slots = 0` (cache disabled, pure staging path) unless env set. Later tasks unaffected (cache path code stays functional when env forces slots>0).

Rationale (measured 2026-09-01, RTX 5090, 8-token run): working set is 43×6 experts/token × ~13.4 MB ≈ 143 GB — no VRAM can hold it; the partial cache thrashes and measures **0.6 tok/s / ttft 20.6 s** vs **2.6 tok/s / ttft 1.8 s** with slots=0.

- [ ] **Step 1: Read the current slot-selection block** (`deepseek_moe_inference_pipeline.cu:473-500`) and confirm structure:

```cpp
// LSX_DEEPSEEK_EXPERT_CACHE_SLOTS (0 disables). Each slot holds one
bool explicit_slots = false;
int slots = 0;
if (const char* env = std::getenv("LSX_DEEPSEEK_EXPERT_CACHE_SLOTS")) {
    slots = std::atoi(env);
    explicit_slots = true;   // verbatim: 0 legitimately disables
} else {
    slots = lifecycle::ExpertCacheSlots(...);   // ← replace this branch
}
```

- [ ] **Step 2: Change the non-explicit branch to disable the cache.** The `lifecycle::ExpertCacheSlots(...)` call computes a slot count that can never cover the routed working set (143 GB), so a partial cache only adds LRU bookkeeping + slab-copy overhead on top of staging. Replace the `else` body with:

```cpp
} else {
    // Routed working set (kTopK experts x layers x ~13.4 MB) exceeds any
    // single-GPU VRAM by orders of magnitude; a partial LRU cache thrashes
    // every token (measured 0.6 vs 2.6 tok/s on RTX 5090). Stream experts
    // from pinned host memory by default; LSX_DEEPSEEK_EXPERT_CACHE_SLOTS
    // still opts into a fixed cache for experiments.
    slots = 0;
}
```

- [ ] **Step 3: Build the test target**

Run: `cmake --build build/linux-release-znver4-aocc --target test_blackwell_perf_deepseek -j 160`
Expected: succeeds, no new warnings.

- [ ] **Step 4: Measure (real-model, single run, nothing else on GPU)**

Run: `cd build/linux-release-znver4-aocc/tests && LSX_TEST_PERF_GEN_TOKENS=32 ./test_blackwell_perf_deepseek 2>&1 | grep "PERF]"`
Expected: tok/s ≥ 2.0, ttft_ms ≤ 3000 (baseline 0.6 / 20588).

- [ ] **Step 5: Run deepseek test group**

Run: `ctest --test-dir build/linux-release-znver4-aocc -R deepseek --output-on-failure`
Expected: all PASS (selfcheck keeps bit-exact default; staging path unchanged).

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu
git commit -m "fix: default deepseek expert cache to streaming (partial LRU thrashes 4.3x)"
```

---

### Task 1: E8M0 → UE4M3 scale repack kernel (TDD)

**Files:**
- Modify: `libs/lsxcommon/src/shared_experts.cu` (add kernel + wrapper next to `dequant_fp4_e8m0` at line ~558)
- Modify: `libs/lsxcommon/include/lsxcommon/shared_experts.h` (declare wrapper)
- Test: `tests/test_deepseek_moe_inference_kernels.cu`

**Interfaces:**
- Consumes: nothing new (plain CUDA kernels alongside existing dequant kernels in the same TU).
- Produces (used by Tasks 3+):
```cpp
// In namespace lsxcommon::gpu, shared_experts.h:
// Repack flat E8M0 scales (1 byte per 32 elements, [rows, cols/32]) into
// repo-canonical UE4M3 pow2 scales (1 byte per 16 elements, [rows, cols/16]).
// Mapping: e4 = e8 - 120 for e8 in [121..135]; e8 <= 120 -> 0 (decodes as
// zero scale, matching FP16 subnormal underflow of 2^(e8-127)); e8 > 135 is
// clamped to 135 by the kernel and must be rejected earlier by the
// load-time validator (ValidateE8m0Range). Returns cudaErrorInvalidValue on
// null args or (cols % 32) != 0.
cudaError_t RepackE8m0ToUe4m3(cudaStream_t stream,
                              const uint8_t *d_e8m0, int64_t num_elements,
                              uint8_t *d_ue4m3);
// Host-side range scan over PINNED source scales (load-time, before H2D).
// Returns false and fills *bad_byte with the first offending value when any
// scale byte exceeds 135 (value class outside the UE4M3 pow2 contract).
bool ValidateE8m0Range(const uint8_t *h_scales, int64_t scale_bytes,
                       int64_t *bad_offset);
```

- [ ] **Step 1: Write the failing tests** — append to `tests/test_deepseek_moe_inference_kernels.cu` (mirror the includes/setup of an existing case in that file; device helpers `CudaMalloc`-style used there):

```cpp
TEST_CASE("e8m0 repack: exact pow2 mapping and boundaries", "[deepseek][repack]")
{
    // 64 elements -> 2 E8M0 bytes -> 4 UE4M3 bytes.
    // e8=127 -> 2^0 -> E=7; e8=120 -> E=0; e8=135 -> E=15; e8=200 clamped 135.
    const uint8_t e8[2] = {127, 120};
    uint8_t *d_in, *d_out;
    cudaMalloc(&d_in, 2); cudaMalloc(&d_out, 4);
    cudaMemcpy(d_in, e8, 2, cudaMemcpyHostToDevice);
    REQUIRE(lsxcommon::gpu::RepackE8m0ToUe4m3(0, d_in, 64, d_out) == cudaSuccess);
    uint8_t out[4] = {};
    cudaMemcpy(out, d_out, 4, cudaMemcpyDeviceToHost);
    CHECK(out[0] == 7);   // (127-120)
    CHECK(out[1] == 7);   // same 32-block
    CHECK(out[2] == 0);   // e8=120 -> E=0
    CHECK(out[3] == 0);
    cudaFree(d_in); cudaFree(d_out);
}

TEST_CASE("e8m0 repack value identity vs dequant", "[deepseek][repack]")
{
    // scale factor identity: UE4M3 decode (E+120)<<23 == E8M0 2^(e8-127).
    for (uint8_t e8 = 121; e8 <= 135; ++e8) {
        const float ue4m3 = lsxcommon::nvfp4::E4M3ScaleToFloat(
            static_cast<uint8_t>(e8 - 120));
        const float e8m0 = std::ldexp(1.0f, static_cast<int>(e8) - 127);
        REQUIRE(ue4m3 == e8m0);
    }
}

TEST_CASE("e8m0 range validator", "[deepseek][repack]")
{
    const uint8_t ok[4] = {0, 120, 127, 135};
    int64_t off = -1;
    CHECK(lsxcommon::gpu::ValidateE8m0Range(ok, 4, &off));
    const uint8_t bad[3] = {127, 136, 130};
    CHECK_FALSE(lsxcommon::gpu::ValidateE8m0Range(bad, 3, &off));
    CHECK(off == 1);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build/linux-release-znver4-aocc --target test_deepseek_moe_inference_kernels -j 160`
Expected: FAIL — `RepackE8m0ToUe4m3` / `ValidateE8m0Range` not declared.

- [ ] **Step 3: Implement.** In `shared_experts.cu` after `dequant_fp4_e8m0_half` (~line 583):

```cpp
// ── E8M0 → UE4M3 repack (TC W4A4 prep, F-12) ──────────────────────────────
// One output byte per 16 elements: block j covers elements [16j, 16j+16) and
// reads source scale (16j) >> 5 = j >> 1 — both halves of the 32-element
// E8M0 block share one source byte (cols % 32 == 0 keeps row boundaries out).
__global__ void repack_e8m0_ue4m3_kernel(const uint8_t* __restrict__ src,
                                         int64_t n_out,
                                         uint8_t* __restrict__ dst)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
    if (i >= n_out) return;
    const unsigned e8 = src[i >> 1];
    const unsigned e = e8 > 135u ? 15u
                     : (e8 > 120u ? e8 - 120u : 0u);
    dst[i] = static_cast<uint8_t>(e);
}

cudaError_t RepackE8m0ToUe4m3(cudaStream_t stream,
                              const uint8_t *d_e8m0, int64_t num_elements,
                              uint8_t *d_ue4m3)
{
    if (!d_e8m0 || !d_ue4m3 || num_elements <= 0 || (num_elements & 31))
        return cudaErrorInvalidValue;
    const int64_t n_out = num_elements / 16;
    const int block = 256;
    const int blocks = static_cast<int>((n_out + block - 1) / block);
    repack_e8m0_ue4m3_kernel<<<blocks, block, 0, stream>>>(d_e8m0, n_out,
                                                           d_ue4m3);
    return cudaGetLastError();
}

bool ValidateE8m0Range(const uint8_t *h_scales, int64_t scale_bytes,
                       int64_t *bad_offset)
{
    if (!h_scales || scale_bytes <= 0) return false;
    for (int64_t i = 0; i < scale_bytes; ++i) {
        if (h_scales[i] > 135) {
            if (bad_offset) *bad_offset = i;
            return false;
        }
    }
    return true;
}
```

Declare both in `include/lsxcommon/shared_experts.h` next to `dequant_fp4_e8m0_half` (same namespace `lsxcommon::gpu`, same doc-comment style).

- [ ] **Step 4: Run tests until green**

Run: `cmake --build build/linux-release-znver4-aocc --target test_deepseek_moe_inference_kernels -j 160 && build/linux-release-znver4-aocc/tests/test_deepseek_moe_inference_kernels "[deepseek][repack]"`
Expected: PASS (3 new cases).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/shared_experts.cu libs/lsxcommon/include/lsxcommon/shared_experts.h tests/test_deepseek_moe_inference_kernels.cu
git commit -m "feat: E8M0->UE4M3 scale repack kernel + load-time range validator"
```

---

### Task 2: fast_math toggle plumbing

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cuh` (declare toggle next to the gpu:: constants, ~line 23)
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu` (define toggle at top of gpu namespace)
- Modify: `libs/lsxcommon/src/deepseek_moe_model.cpp:30-34` (set from `ModelInitConfig`)

**Interfaces:**
- Produces:
```cpp
// deepseek_moe_load.cuh, namespace lsxcommon::gpu:
void SetDeepseekFastMath(bool on);
bool DeepseekFastMath();
```

- [ ] **Step 1: Define the toggle** in `deepseek_moe_inference_moe.cu` above `run_shared_experts_ffn` (mirror of `SetQwenTcFastMath`, `qwen_inference.cu:31-42`):

```cpp
// Blackwell TC W4A4 toggle (F-12). Process-wide; flipped from
// ModelInitConfig::fast_math at model construction. Default OFF keeps the
// dequant+cuBLAS routes bit-exact.
static std::atomic<bool> g_deepseek_fast_math{false};

void SetDeepseekFastMath(bool on)
{
    g_deepseek_fast_math.store(on, std::memory_order_relaxed);
}

bool DeepseekFastMath()
{
    return g_deepseek_fast_math.load(std::memory_order_relaxed);
}
```

(add `#include <atomic>` if absent; declare the two functions in `deepseek_moe_load.cuh` inside `namespace lsxcommon::gpu` next to `kTopK`.)

- [ ] **Step 2: Wire from model config** — `deepseek_moe_model.cpp` constructor:

```cpp
DeepSeekV4FlashModel::DeepSeekV4FlashModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)}
{
    gpu::SetDeepseekFastMath(cfg_.fast_math);
}
```

- [ ] **Step 3: Build + bit-exact guard**

Run: `cmake --build build/linux-release-znver4-aocc --target test_deepseek_moe_inference_kernels test_deepseek_moe_loader -j 160 && ctest --test-dir build/linux-release-znver4-aocc -R deepseek --output-on-failure`
Expected: all PASS (toggle default OFF — no behavioral change).

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_load.cuh libs/lsxcommon/src/deepseek_moe_inference_moe.cu libs/lsxcommon/src/deepseek_moe_model.cpp
git commit -m "feat: deepseek fast_math toggle plumbed from ModelInitConfig"
```

---

### Task 3: TC W4A4 routed-expert FFN dispatch

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu` (`run_routed_expert_ffn` ~line 306-419; staging/cached scale plumbing ~line 640-740)
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu` (stage-slab sizing for TC scales)
- Test: `tests/test_deepseek_moe_inference_kernels.cu`

**Interfaces:**
- Consumes: `qwen::tc::RunQwenTcNvfp4Gemm` / `RunQwenTcGemmPreQuant` / `QuantizeActsNvfp4` (`include/lsxcommon/qwen_tc_gemm.h` — weights in repo layout [N,K] nibbles + [N,K/16] UE4M3 scales, y=__half, epilogue `*w_global_scale`, requires sm_12x); `RepackE8m0ToUe4m3` (Task 1); `ValidateE8m0Range` (Task 1); `DeepseekFastMath()` (Task 2).
- Produces: TC body inside `run_routed_expert_ffn` with 3 trailing optional params appended to its signature (mirrors round-20b optional-scratch pattern):
```cpp
// run_routed_expert_ffn(...) gains, after the existing params:
//     const uint8_t *d_w1_scales_tc, const uint8_t *d_w2_scales_tc,
//     const uint8_t *d_w3_scales_tc   // UE4M3 per-16 slabs; nullptr = legacy
// Callers (run_routed_experts_ex / run_routed_experts) pass the repacked
// slabs when DeepseekFastMath() && TC device, else nullptr.
```

Numeric-class note (binding): the TC path changes accumulation order and quantizes activations per-16 — it is tolerance-class, gated strictly behind `fast_math`. The FP32-W2 overflow comment at `deepseek_moe_inference_moe.cu:391-399` documents the LEGACY path rationale and stays untouched.

- [ ] **Step 1: Write the failing parity test** — append to `tests/test_deepseek_moe_inference_kernels.cu`. It builds one random-ish expert (deterministic LCG fill), runs legacy `run_routed_expert_ffn` vs the TC body, and bounds the W4A4 error by the NVFP4 quantization bound (max |Δ| ≤ 2× block absmax/7 per quantized operand — empirical bound used by the qwen TC tests; assert ≤ 5% of |y|max):

```cpp
TEST_CASE("routed expert TC W4A4 parity vs legacy", "[deepseek][tc]")
{
    // Skip without sm_12x: TC kernel returns cudaErrorNotSupported.
    if (!lsxcommon::gpu::DeepseekFastMath()) {
        lsxcommon::gpu::SetDeepseekFastMath(true);
    }
    // Deterministic fill: E2M1 nibble bytes from LCG; E8M0 scales 121..135.
    // Allocate host+device buffers for w1/w3 [2048,4096], w2 [4096,2048].
    // 1) legacy = run_routed_expert_ffn(..., nullptr,nullptr,nullptr)
    // 2) tc     = run_routed_expert_ffn(..., tc_scale_slabs)
    // 3) CHECK(max_abs_diff(tc, legacy) <= 0.05f * max_abs(legacy));
    // Cleanup: SetDeepseekFastMath(false);
}
```

(Write the full body following the allocation helpers already used in this test file; the assertion values above are the contract.)

- [ ] **Step 2: Run to verify failure** — TC slab pointers do not exist yet.

Run: `cmake --build build/linux-release-znver4-aocc --target test_deepseek_moe_inference_kernels -j 160`
Expected: FAIL (compile error on new params).

- [ ] **Step 3: Implement the TC body.** Inside `run_routed_expert_ffn`, before the legacy block:

```cpp
const bool use_tc = DeepseekFastMath() && d_w1_scales_tc && d_w2_scales_tc &&
                    d_w3_scales_tc;
if (use_tc) {
    // W4A4: activations quantized per-16 pow2; weights stay packed.
    // Scratch: carve half views from the existing float arena slices
    // (each float slot holds one half — halves never exceed floats).
    __half* const gate_h = reinterpret_cast<__half*>(d_gate_f32);
    __half* const up_h   = reinterpret_cast<__half*>(d_up_f32);
    // y_tc carved from d_dequant: the TC path performs NO weight dequant, so
    // the (max w*_el)-float dequant arena slice is free; H halves always fit.
    __half* const y_tc   = reinterpret_cast<__half*>(d_dequant);
    // W1: y = W1 @ x  (N=I, K=H)
    if (qwen::tc::RunQwenTcNvfp4Gemm(stream, d_hidden, d_w1, d_w1_scales_tc,
                                     1.0f, gate_h, 1, kHiddenDim,
                                     kIntermediateDim) != cudaSuccess)
        return cudaGetLastError();
    // W3: same shape
    if (qwen::tc::RunQwenTcNvfp4Gemm(stream, d_hidden, d_w3, d_w3_scales_tc,
                                     1.0f, up_h, 1, kHiddenDim,
                                     kIntermediateDim) != cudaSuccess)
        return cudaGetLastError();
    // SwiGLU FP16 (same clamp semantics as the FP32 reference path):
    // silu_mul(up, gate, n, 10) — in-place on up_h (kernel writes its first
    // arg), leaving gate_h intact.
    silu_mul_kernel<<<(kIntermediateDim + 255) / 256, 256, 0, stream>>>(
        up_h, gate_h, kIntermediateDim, 10.0f);
    // W2 with the routing weight folded into the TC epilogue scale
    // (w_global_scale == weight — post-GEMM scaling, mathematically equal
    // to the legacy pre-W2 FP32 scale_float_kernel).
    if (qwen::tc::RunQwenTcNvfp4Gemm(stream, up_h, d_w2, d_w2_scales_tc,
                                     weight, y_tc, 1, kIntermediateDim,
                                     kHiddenDim) != cudaSuccess)
        return cudaGetLastError();
    // Accumulate y_tc into FP32 d_output.
    acc_half_to_f32_kernel<<<(kHiddenDim + 255) / 256, 256, 0, stream>>>(
        d_output, y_tc, kHiddenDim, accumulate);
    return cudaGetLastError();
}
```

Add the tiny epilogue kernel next to `scale_float_kernel`:

```cpp
// TC epilogue: fold __half expert output into the FP32 routed accumulator.
__global__ void acc_half_to_f32_kernel(float* __restrict__ dst,
                                       const __half* __restrict__ src,
                                       int n, bool accumulate)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = __half2float(src[i]);
    dst[i] = accumulate ? dst[i] + v : v;
}
```

Includes: `#include "lsxcommon/qwen_tc_gemm.h"` at the top of the TU (safe: host+device header, no absl).

- [ ] **Step 4: Thread the TC scale slabs through the callers.**
  1. In `run_routed_experts_ex` (same file, ~line 640-745): for BOTH the cache-hit slab path and the stage path, compute the UE4M3 pointers:
     - cache path: slab layout per slot becomes `[s1e8|s2e8|s3e8|s1u|s2u|s3u]`; pass `s1 + (s1b + s2b + s3b)` etc. when `DeepseekFastMath()`.
     - stage path: after the six `cudaMemcpyAsync` calls, when `DeepseekFastMath()`, launch `RepackE8m0ToUe4m3(stream, s1, w1_el, s1_tc)` for w1/w2/w3 into a TC stage region (`d_scale_stage + scale_stage_bytes` — see Step 4b allocation).
  2. In `deepseek_moe_inference_pipeline.cu`: wherever `d_scale_stage` is allocated, add `d_scale_stage_tc` of the same size when `gpu::DeepseekFastMath()` (arena slice; disable if allocation would exceed the arena budget — fallback: TC off, legacy path).
  3. Cache geometry: in the same file where `expert_cache.scale_stride` is set (~line 490), multiply the scale stride by 2 when `DeepseekFastMath()` (E8M0 half first, UE4M3 second half) and fill the UE4M3 half with `RepackE8m0ToUe4m3` at expert-insert (the same place the six scale `cudaMemcpyAsync` calls land in the cache).
  4. Load-time validation (real weights): in the expert-loading loop of `deepseek_moe_load.cu`, when `DeepseekFastMath()`, run `ValidateE8m0Range` over each expert's pinned scale bytes; on failure return a load error naming layer/expert index (spdlog error + `cudaErrorInvalidValue`).

- [ ] **Step 5: Run the new test + full kernels suite**

Run: `cmake --build build/linux-release-znver4-aocc --target test_deepseek_moe_inference_kernels -j 160 && build/linux-release-znver4-aocc/tests/test_deepseek_moe_inference_kernels`
Expected: all cases PASS (4470+ assertions incl. 3 new repack + new parity).

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_inference_moe.cu libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu libs/lsxcommon/src/deepseek_moe_load.cu tests/test_deepseek_moe_inference_kernels.cu
git commit -m "feat: TC W4A4 routed-expert FFN under fast_math (repacked UE4M3 slabs)"
```

---

### Task 3b: end-to-end fast_math measurement (real model)

**Files:** none (measurement + registry row).

- [ ] **Step 1: Rebuild all affected test targets**

Run: `cmake --build build/linux-release-znver4-aocc --target test_deepseek_moe_inference_kernels test_deepseek_moe_loader test_blackwell_perf_deepseek -j 160`

- [ ] **Step 2: Selfcheck OFF (bit-exact) and ON (Paris)**

Run: `ctest --test-dir build/linux-release-znver4-aocc -R "deepseek" --output-on-failure`
Expected: PASS. Then `cd build/linux-release-znver4-aocc/tests && LSX_TEST_FAST_MATH=1 LSX_TEST_PERF_GEN_TOKENS=8 ./test_blackwell_perf_deepseek 2>&1 | grep "PERF]"`
Expected: text starts with " Paris", tok/s strictly greater than the Task-0 result (2.6); record the number.

- [ ] **Step 3: Record results** in `docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md` (new Round 24 subsection: F-12 phase-1 before/after table) and update the F-12 row status to `in progress (phase-1 landed; shared/attention caching pending)`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md
git commit -m "docs: F-12 phase-1 (routed TC W4A4) before/after measurements"
```

---

### Task 4: Shared-expert + attention F8 dequant caching (bit-exact, always-on)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_moe_load.cu` (allocate per-layer FP16/FP32 dequant caches in the device arena next to `d_attn_dequant` ~line 944)
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_moe.cu` (`run_shared_experts_ffn` ~line 188-296: accept optional cached weights)
- Modify: `libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu` (thread cache pointers through `run_shared_experts_ffn` call ~line 649 and the attention dequant sites ~line 575-582)
- Test: `tests/test_deepseek_moe_inference_kernels.cu`

**Interfaces:**
- Consumes: `dequant_shared_f8_half` / `dequant_shared_f8` (existing, `shared_experts.cu`).
- Produces: trailing optional params on `run_shared_experts_ffn`:
```cpp
// Append: const __half *d_w1_h_cache, const __half *d_w3_h_cache,
//         const float *d_w2_f32_cache   // nullptr = dequant per call (legacy)
// Bit-exactness contract: cache content produced by the SAME dequant kernels
// at load; when cache pointers are non-null the per-call dequant launches are
// skipped and GEMMs consume the cached pointers directly.
```

- [ ] **Step 1: Write the failing bit-exact test** in `tests/test_deepseek_moe_inference_kernels.cu`:

```cpp
TEST_CASE("shared expert dequant cache is bit-exact", "[deepseek][cache]")
{
    // Fill w1/w2/w3 (F8_E4M3 bytes) + E8M0 scales deterministically.
    // 1) legacy: run_shared_experts_ffn(..., nullptr caches) -> out_a
    // 2) cached: dequant once into cache slabs, run with caches -> out_b
    // 3) REQUIRE(memcmp(out_a, out_b) == 0);   // bit-exact, no tolerance
}
```

- [ ] **Step 2: Run to verify failure** (cache params do not exist).

Run: `cmake --build build/linux-release-znver4-aocc --target test_deepseek_moe_inference_kernels -j 160`
Expected: FAIL.

- [ ] **Step 3: Implement:**
  1. `run_shared_experts_ffn` gains the three trailing cache params (default `nullptr`); each dequant launch is wrapped: `if (!d_w1_h_cache) { dequant_shared_f8_half(...); }` and GEMMs read the cache pointer instead of the scratch when provided. W2 FP32 GEMM reads `d_w2_f32_cache` when provided (dequant_shared_f8 output is FP32 — cache it whole).
  2. Load-time (`deepseek_moe_load.cu`): allocate and fill the caches once per layer right where `d_attn_dequant` arena is sized: shared w1/w3 → FP16 (`dequant_shared_f8_half`), shared w2 → FP32 (`dequant_shared_f8`), attention `wq_b`/`wo_a`/`wo_b` → FP16. Total ≈ 2.5-3 GB (within VRAM governor budget; if the governor refuses, fall back to legacy per-call dequant and log once).
  3. Pipeline: pass the cache pointers at the two call sites (~575, ~649).
- [ ] **Step 4: Run to green** (same command as Task 3 Step 5). Bit-exactness pinned by the new test; selfcheck 3-cycle stays byte-identical.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/deepseek_moe_load.cu libs/lsxcommon/src/deepseek_moe_inference_moe.cu libs/lsxcommon/src/deepseek_moe_inference_pipeline.cu tests/test_deepseek_moe_inference_kernels.cu
git commit -m "feat: load-time FP16/FP32 dequant caches for deepseek shared expert + attention F8"
```

---

### Task 5: Acceptance gate + final measurements

**Files:**
- Modify: `docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md` (F-12 closure)
- Modify: `docs/NVIDIA_DOCS_NVFP4_MEMORY.md` (only if new hardware facts surface — none expected)

- [ ] **Step 1: Full deepseek + related suites**

Run: `ctest --test-dir build/linux-release-znver4-aocc -R "deepseek|blackwell|expert" --output-on-failure`
Expected: all PASS (selfcheck OFF bit-exact 3 cycles; ON "Paris").

- [ ] **Step 2: Final perf table (32 tokens, sequential runs, GPU idle)**

```bash
cd build/linux-release-znver4-aocc/tests
LSX_TEST_PERF_GEN_TOKENS=32 ./test_blackwell_perf_deepseek 2>&1 | grep "PERF]"          # default (legacy)
LSX_TEST_FAST_MATH=1 LSX_TEST_PERF_GEN_TOKENS=32 ./test_blackwell_perf_deepseek 2>&1 | grep "PERF]"  # TC W4A4
```
Record both rows (plus Task 0's 2.6 baseline) into the Round 24 registry section.

- [ ] **Step 3: Registry + naming guard**

Run: `cmake --build build/linux-release-znver4-aocc --target test_code_naming_guard -j 160 && build/linux-release-znver4-aocc/tests/test_code_naming_guard`
Expected: PASS (no new files). Update F-12 row: `fixed (phase-1+2) — see Round 24`.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-23-blackwell-cuda-audit-findings.md
git commit -m "docs: F-12 acceptance gate — deepseek TC fast-math final measurements"
```
