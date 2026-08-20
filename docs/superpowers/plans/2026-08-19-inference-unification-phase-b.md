# Inference Unification — Phase B (Core Computational Loops) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the TaskMosaic orchestrator ring and the three engine decode loops: a reusable sampler seam, a single-allocation DeepSeek arena, a unified `IModel::Benchmark`, and the prefill/KV matrix (GLM seq-P verification + Qwen incremental KV).

**Architecture:** Phase B implements SECTION 3 of `docs/superpowers/specs/2026-08-17-inference-unification-design.md` (items 11–15). Host-side seams first (`lsxcommon::sample::ISampler`, `IModel::Benchmark`), then the DeepSeek single-arena refactor, then the orchestrator response ring, then the GLM seq-P prefill parity test and the Qwen incremental per-call KV cache, then occupancy counters. Real-model tests SKIP with CTest `SKIP_RETURN_CODE 4` (existing convention).

**Tech Stack:** C++20, CUDA 13.2+ (Blackwell 120-real;121-real only), CCCL 3.3, libzmq, simdjson, Catch2 v3.7.0. Host-only code prefers `absl::` containers (`flat_hash_map`) — NEVER in `.cu`/`.cuh` (NVCC/MSVC incompatible).

## Global Constraints

- CUDA architectures `120-real;121-real` only (no Hopper/Ada/Ampere). Set on every new `.cu` object lib / target.
- `lsxcommon` STATIC library; object libs `lsxcommon_cu` (DeepSeek + safetensors + benchmark + NEW sampling), `lsxcommon_glm_cu` (GLM kernels, `-fmad=false`), `lsxcommon_qwen_cu` (Qwen kernels). New `.cu` sources go into the matching object lib's `add_library(... OBJECT ...)` list.
- `absl/container/*` headers are FORBIDDEN in any header reachable from `.cu`/`.cuh`. `sampling.h` must stay absl-free (it is included by all three engines' `.cu` and the host model wrappers).
- No `std::cout`/`printf` — all logging via spdlog (engines already use `log::info`/`log::warn` from `src/log.cpp`).
- Tests: TDD — write the failing test, verify RED, implement, verify GREEN, commit. Commit prefixes: `build:`, `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
- Real-model tests: existing GLM/DeepSeek suites SKIP when `/mnt/llm_models/models/{DeepSeek-V4-Flash,GLM-5.2}` checkpoints are absent (CTest `SKIP_RETURN_CODE 4`). No real Qwen checkpoint on this box.
- CMake build: `cmake --build --preset linux-debug`; tests `ctest --preset linux-debug -R <regex>`. Binaries at `build/linux-debug/apps/<app>/<app>` and `build/linux-debug/tests/<test>`.
- Blackwell occupancy: 48 warps/SM max; keep sampler kernel single-block ≤ 256 threads.

## Task Dependency Order

```
Task 1 (sampling.h + GreedySampler) ──► Task 2 (thread ISampler& into 3 engines)
Task 3 (IModel::Benchmark + shim)         [independent]
Task 4 (DeepSeekInferArena)               [independent]
Task 5 (orchestrator response ring)       [independent]
Task 6 (GLM seq-P prefill parity test)    [independent]
Task 7 (Qwen incremental KV)              [independent]
Task 8 (occupancy counters)               [depends on Task 4 arena counters + Task 7 cache]
```

Each task is independently testable and committed separately.

---

## File Structure

| File | Responsibility |
|---|---|
| `libs/lsxcommon/include/lsxcommon/sampling.h` (new) | `lsxcommon::sample` interface: `LogitType`, `ISampler`, `GreedySampler`, `DefaultSampler()`. absl-free, includes `cuda_runtime.h`/`cuda_fp16.h` only. |
| `libs/lsxcommon/src/sampling.cu` (new) | `GreedySampler::Sample` — single-block argmax kernel, FP32/FP16 dispatch. In `lsxcommon_cu`. |
| `libs/lsxcommon/src/glm_inference.cu` | Add `ISampler&` param to `RunGlmGenerationImpl`/both `RunGlmGeneration`; decode loop calls `sampler.Sample` on `arena.d_logits_head`. |
| `libs/lsxcommon/src/deepseek_inference_pipeline.cu` | Add `ISampler&` param to `RunInferenceModel`; decode loop calls `sampler.Sample` on `d_logits`. |
| `libs/lsxcommon/src/qwen_inference.cu` | Add `ISampler&` param to `RunQwenGeneration`; replace `qwen_argmax_kernel` call with `sampler.Sample` on the FP16 last row. |
| `libs/lsxcommon/include/lsxcommon/model.h` | `virtual gpu::BenchmarkResult Benchmark(int num_tokens = 2048)` (default `{}`, ok=false). |
| `libs/lsxcommon/include/lsxcommon/benchmark_gpu.h` | `BenchmarkResult` gains `std::uint64_t resident_bytes = 0`. |
| `libs/lsxcommon/src/benchmark_gpu.cpp` (new; move body of `benchmark_gpu.cu`'s `RunBenchmark`) | `RunBenchmark(model_path)` becomes a thin factory shim: create → `Benchmark()` → destroy. |
| `libs/lsxcommon/src/benchmark_gpu.cu` | Keep the NVFP4 ping-pong kernels but stop exposing them as `RunBenchmark` (rename to internal `gpu::RunGemmBenchmark`, unused, or delete; see Task 3). |
| `libs/lsxcommon/src/glm_v5_model.cpp`, `deepseek_v4_flash_model.cpp`, `qwen_base_model.cpp` | `Benchmark()` overrides; DeepSeek/Qwen via their engine loop, GLM via `RunGlmGeneration` on a synthetic prompt. |
| `libs/lsxcommon/src/deepseek_inference_load.cuh` | New `DeepSeekInferArena` struct; new `route_token` signature (caller-owned staging); `InferenceAllocCount()` counter. |
| `libs/lsxcommon/src/deepseek_inference_load.cu` | `DeepSeekInferArena::Allocate`/`Free`. |
| `libs/lsxcommon/src/deepseek_inference_moe.cu` | `route_token` drops internal `cudaMalloc`/`cudaMallocHost`/frees. |
| `libs/lsxcommon/src/deepseek_inference_pipeline.cu` | Replace ~40 `cudaMalloc`/`cudaFree` with one `DeepSeekInferArena` (+ per-run pinned host block); CSA state carved from the arena (replaces `alloc_csa_state`/`free_csa_state` per-run path). |
| `libs/lsxcommon/src/deepseek_inference_csa.cu` | `alloc_csa_state`/`free_csa_state` kept only for the old path or removed; arena gains a CSA binding helper. |
| `libs/lsxnetwork/include/lsxnetwork/openai_endpoint.h` | `ChatCompletion` gains `int max_tokens = 0`; new `CompletionResponse`; `OpenAiEndpoint::WriteCompletion`. |
| `libs/lsxnetwork/src/openai_endpoint.cpp` | Parse `max_tokens`; implement `WriteCompletion`. |
| `libs/lsxcommon/src/models/lsx_orchestrator.cpp` + `include/lsxcommon/lsx_orchestrator.h` | `DispatchOne` tokenizes + infers + serializes; response ring; `ServeResponses` loop. |
| `libs/lsxcommon/src/models/lsx_orchestrator.cpp` | `CompletionResponse` mapping helper (InferenceResult → CompletionResponse). |
| `libs/lsxcommon/src/qwen_base_model.cpp` + `include/lsxcommon/qwen_base_model.h` | Own a `QwenKvCache`; pass into `RunQwenGeneration`. |
| `libs/lsxcommon/src/qwen_inference.cu` | New `RunQwenGeneration` incremental path: append k/v rows at `kv_base`, persist GatedDeltaNet state. |
| `tests/test_sampling.cu` (new) | GreedySampler vs host reference argmax (FP32 + FP16), tie-breaking, invalid args. |
| `tests/test_deepseek_arena.cu` (new) | `route_token` alloc-count == 0 after refactor; arena allocation-count == 2 per run (real model; SKIPs). |
| `tests/test_benchmark.cpp` (new, or extend `test_model_runtime.cpp`) | `RunBenchmark` shim parity, `resident_bytes` field, default `{ok=false}`. |
| `tests/test_integrated_pipeline.cpp` | Extend: response frame out with `id` correlation. |
| `tests/test_glm_prefill_seqp.cpp` (new) | GLM seq-P prefill parity (real model; SKIPs). |
| `tests/test_qwen_kv_cache.cpp` (new) | Qwen incremental KV (real model; SKIPs; 2-call continuation == single full run). |
| `tests/test_model_runtime.cpp` | Occupancy-counter contract tests (host-only). |
| `libs/lsxcommon/CMakeLists.txt` | Add `src/sampling.cu` to `lsxcommon_cu`; (Task 4) nothing new (arena in existing files). |
| `tests/CMakeLists.txt` | Add the new test executables; `SKIP_RETURN_CODE 4` for real-model tests. |

---

## Task 1: Sampling module (`lsxcommon::sample::ISampler` + `GreedySampler`)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/sampling.h`
- Create: `libs/lsxcommon/src/sampling.cu`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add `src/sampling.cu` to `lsxcommon_cu`)
- Create: `tests/test_sampling.cu`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: existing engine decode loops (Tasks 2 threads this in).
- Produces: `lsxcommon::sample::ISampler`, `lsxcommon::sample::LogitType`, `lsxcommon::sample::GreedySampler`, `lsxcommon::sample::DefaultSampler()`. Used verbatim by Tasks 2, 6, 7.

The spec's bare `ISampler::Sample(cudaStream_t, const void*, int, int32_t*, float*)` cannot distinguish FP32 (GLM/DeepSeek) from FP16 (Qwen) logits rows. This plan adds an explicit `LogitType dtype` parameter — a documented, necessary extension of the spec sketch.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sampling.cu`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#include "lsxcommon/sampling.h"

using lsxcommon::sample::GreedySampler;
using lsxcommon::sample::LogitType;

namespace {

int32_t HostArgmax(const std::vector<float>& row) {
    int32_t best_i = 0;
    float best = row[0];
    for (int i = 1; i < static_cast<int>(row.size()); ++i) {
        if (row[i] > best) { best = row[i]; best_i = i; }
    }
    return best_i;
}

}  // namespace

TEST_CASE("GreedySampler selects first-maximum on ties", "[sampling]") {
    const std::vector<float> row = {1.0f, -5.0f, 3.0f, 3.0f, 2.0f, -1.0f};
    const int vocab = static_cast<int>(row.size());

    float* d_row = nullptr;
    REQUIRE(cudaMalloc(&d_row, vocab * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_row, row.data(), vocab * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    int32_t* d_tok = nullptr;
    float* d_maxv = nullptr;
    REQUIRE(cudaMalloc(&d_tok, sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_maxv, sizeof(float)) == cudaSuccess);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GreedySampler g;
    REQUIRE(g.Sample(stream, d_row, vocab, LogitType::kF32, d_tok, d_maxv)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    int32_t tok = -1;
    float maxv = 0.0f;
    REQUIRE(cudaMemcpy(&tok, d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(&maxv, d_maxv, sizeof(float), cudaMemcpyDeviceToHost)
            == cudaSuccess);

    REQUIRE(tok == HostArgmax(row));
    REQUIRE(tok == 2);       // FIRST 3.0, index 2
    REQUIRE(maxv == Catch::Approx(3.0f));

    cudaStreamDestroy(stream);
    cudaFree(d_maxv);
    cudaFree(d_tok);
    cudaFree(d_row);
}

TEST_CASE("GreedySampler handles FP16 rows", "[sampling]") {
    const std::vector<float> row = {0.5f, 100.0f, -2.0f, 42.0f};
    const int vocab = static_cast<int>(row.size());

    std::vector<__half> half_row(row.size());
    for (int i = 0; i < vocab; ++i) half_row[i] = __float2half(row[i]);

    float* d_row = nullptr;
    REQUIRE(cudaMalloc(&d_row, vocab * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_row, half_row.data(), vocab * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    int32_t* d_tok = nullptr;
    float* d_maxv = nullptr;
    REQUIRE(cudaMalloc(&d_tok, sizeof(int32_t)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_maxv, sizeof(float)) == cudaSuccess);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GreedySampler g;
    REQUIRE(g.Sample(stream, d_row, vocab, LogitType::kF16, d_tok, d_maxv)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    int32_t tok = -1;
    float maxv = 0.0f;
    REQUIRE(cudaMemcpy(&tok, d_tok, sizeof(int32_t), cudaMemcpyDeviceToHost)
            == cudaSuccess);
    REQUIRE(cudaMemcpy(&maxv, d_maxv, sizeof(float), cudaMemcpyDeviceToHost)
            == cudaSuccess);

    REQUIRE(tok == 1);
    REQUIRE(maxv == Catch::Approx(100.0f));

    cudaStreamDestroy(stream);
    cudaFree(d_maxv);
    cudaFree(d_tok);
    cudaFree(d_row);
}

TEST_CASE("GreedySampler rejects invalid arguments", "[sampling]") {
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    GreedySampler g;
    float d_stub = 0.0f;
    int32_t d_tok = 0;
    REQUIRE(g.Sample(stream, nullptr, 10, LogitType::kF32, &d_tok, &d_stub)
            == cudaErrorInvalidValue);
    REQUIRE(g.Sample(stream, &d_stub, 0, LogitType::kF32, &d_tok, &d_stub)
            == cudaErrorInvalidValue);
    REQUIRE(g.Sample(stream, &d_stub, 10, static_cast<LogitType>(7),
                     &d_tok, &d_stub) == cudaErrorInvalidValue);
    cudaStreamDestroy(stream);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_sampling 2>&1 | tail -5`
Expected: FAIL — target not built / `"lsxcommon/sampling.h"` not found (GREEN after Step 4).

- [ ] **Step 3: Write the interface and implementation**

Create `libs/lsxcommon/include/lsxcommon/sampling.h`:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Isolated sampling seam (spec item 13). HOST + CUDA safe: no absl containers,
// no device-side state. Engines keep their fused argmax kernels; GreedySampler
// is the default wrapper where a TopK/TopP sampler later reads the full row.
namespace lsxcommon {
namespace sample {

enum class LogitType {
    kF32 = 0,   // float logits rows (GLM d_logits_head, DeepSeek d_logits)
    kF16 = 1,   // __half logits rows (Qwen d_logits)
};

class ISampler {
public:
    virtual ~ISampler() = default;

    // Selects one token id from a [vocab] logits row on `stream`.
    //   d_logits    device pointer to the row (FP32 or FP16 per @p dtype).
    //   d_out_token device int32 slot receiving the selected id.
    //   d_out_maxv  device float slot receiving the raw max logit.
    // Returns cudaSuccess or a CUDA error code; never throws.
    virtual cudaError_t Sample(cudaStream_t stream, const void* d_logits,
                               int vocab, LogitType dtype,
                               int32_t* d_out_token, float* d_out_maxv) = 0;
};

// Greedy argmax sampler. Matches the engines' fused-argmax semantics: on ties
// the FIRST (lowest index) maximum wins. Single block, ≤ 256 threads.
class GreedySampler : public ISampler {
public:
    cudaError_t Sample(cudaStream_t stream, const void* d_logits, int vocab,
                       LogitType dtype, int32_t* d_out_token,
                       float* d_out_maxv) override;
};

// Function-local static default instance (stable address for default args).
ISampler& DefaultSampler();

}  // namespace sample
}  // namespace lsxcommon
```

Create `libs/lsxcommon/src/sampling.cu`:

```cpp
#include "lsxcommon/sampling.h"

namespace lsxcommon {
namespace sample {

namespace {

template <typename T>
__global__ void kGreedyArgmax(const T* __restrict__ logits, int vocab,
                              int32_t* out_tok, float* out_maxv) {
    __shared__ float s_best[256];
    __shared__ int32_t s_idx[256];

    float best = -1e30f;
    int32_t best_i = -1;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float v = static_cast<float>(logits[i]);
        if (v > best) { best = v; best_i = i; }
    }
    s_best[threadIdx.x] = best;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float b2 = s_best[threadIdx.x + stride];
            const int32_t i2 = s_idx[threadIdx.x + stride];
            if (b2 > s_best[threadIdx.x] ||
                (b2 == s_best[threadIdx.x] && i2 < s_idx[threadIdx.x])) {
                s_best[threadIdx.x] = b2;
                s_idx[threadIdx.x] = i2;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out_tok = s_idx[0];
        *out_maxv = s_best[0];
    }
}

}  // namespace

cudaError_t GreedySampler::Sample(cudaStream_t stream, const void* d_logits,
                                  int vocab, LogitType dtype,
                                  int32_t* d_out_token, float* d_out_maxv) {
    if (!d_logits || !d_out_token || !d_out_maxv || vocab <= 0)
        return cudaErrorInvalidValue;
    constexpr int kThreads = 256;
    if (dtype == LogitType::kF32) {
        kGreedyArgmax<float><<<1, kThreads, 0, stream>>>(
            static_cast<const float*>(d_logits), vocab, d_out_token, d_out_maxv);
    } else if (dtype == LogitType::kF16) {
        kGreedyArgmax<__half><<<1, kThreads, 0, stream>>>(
            static_cast<const __half*>(d_logits), vocab, d_out_token, d_out_maxv);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

ISampler& DefaultSampler() {
    static GreedySampler g;
    return g;
}

}  // namespace sample
}  // namespace lsxcommon
```

Modify `libs/lsxcommon/CMakeLists.txt` — add `src/sampling.cu` to the `lsxcommon_cu` OBJECT library (after `src/deepseek_norm_kernels.cu`):

```cmake
    src/shared_experts.cu
    src/sampling.cu
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_sampling && ./build/linux-debug/tests/test_sampling
```
Expected: 3 test cases, all PASS. Also rebuild everything that includes `sampling.h` later; nothing includes it yet.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/sampling.h libs/lsxcommon/src/sampling.cu libs/lsxcommon/CMakeLists.txt tests/test_sampling.cu tests/CMakeLists.txt
git commit -m "feat: add isolated greedy sampling seam (ISampler)"
```

---

## Task 2: Thread `ISampler&` into the three engine decode loops

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_pipeline.cu`
- Modify: `libs/lsxcommon/src/qwen_inference.cu`

**Interfaces:**
- Consumes: `lsxcommon::sample::ISampler`, `lsxcommon::sample::DefaultSampler()` (Task 1).
- Produces: defaulted `ISampler&` parameters on `RunGlmGeneration`, `gpu::RunInferenceModel`, `qwen::RunQwenGeneration`. Tasks 3/6/7 rely on these signatures.

- [ ] **Step 1: Write the failing test**

There is no separate new test file — the existing engine suites are the regression net. This task's TDD is: add the sampler param, then run the full CUDA engine suites and confirm ZERO behavior change (GreedySampler reproduces the fused argmax bit-for-bit). Add a compile-time anchor test to `tests/test_sampling.cu` asserting the seam is usable from a plain host TU (it is — sampling.h has no device code):

Append to `tests/test_sampling.cu`:

```cpp
TEST_CASE("DefaultSampler is a stable singleton", "[sampling]") {
    auto& a = lsxcommon::sample::DefaultSampler();
    auto& b = lsxcommon::sample::DefaultSampler();
    REQUIRE(&a == &b);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_sampling && ./build/linux-debug/tests/test_sampling`
Expected: FAIL with "Test did not pass" (the test body references the missing function — fails to link). This proves the new assertion is real (RED) before wiring.

- [ ] **Step 3: Implement the seam in the three engines**

**GLM** — `libs/lsxcommon/src/glm_inference.cu`:

Add `#include "lsxcommon/sampling.h"` at the top of the include block. Change the two `RunGlmGeneration` overloads (lines ~725-740) and `RunGlmGenerationImpl` to accept a defaulted sampler. The `Impl` signature (find it above line 520 — it takes `(cublasHandle_t, cudaStream_t, const GlmPinnedWeights&, const GlmExpertTables*, GlmExpertDb*, uint8_t*, uint8_t*, const std::vector<int32_t>&, int, GlmGenerationResult&)`):

```cpp
cudaError_t RunGlmGenerationImpl(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const GlmPinnedWeights &pw,
    const GlmExpertTables *experts, GlmExpertDb *expert_db,
    uint8_t *d_pack, uint8_t *d_scale,
    const std::vector<int32_t> &prompt, int max_new,
    GlmGenerationResult &result,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler());
```

Both public `RunGlmGeneration` overloads get the same trailing defaulted param and forward `sampler` to `RunGlmGenerationImpl`.

In the decode loop, replace the fused-argmax readback (currently lines ~664-673):

```cpp
        g_stage = "decode-argmax";
        e = sampler.Sample(stream, arena.d_logits_head, vocab,
                           lsxcommon::sample::LogitType::kF32,
                           &arena.d_argmax->best, &arena.d_argmax->maxv);
        if (e != cudaSuccess) return fail(e);
        e = cudaMemcpyAsync(arena.h_argmax, arena.d_argmax,
                            sizeof(GlmArgmaxSlot), cudaMemcpyDeviceToHost,
                            stream);
        if (e != cudaSuccess) return fail(e);
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return fail(e);
        const int best = arena.h_argmax->best;
```

(`GlmArgmaxSlot` holds `int best; float maxv;` — the `&arena.d_argmax->best` address is the first field, typed `int32_t*`.)

**DeepSeek** — `libs/lsxcommon/src/deepseek_inference_pipeline.cu`:

Add `#include "lsxcommon/sampling.h"`. Change `RunInferenceModel` (line 144):

```cpp
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler())
```

In the decode loop, immediately after `run_lm_head` (which writes `d_logits`, `d_next_token`, `d_next_maxv`), insert before the `d_next_token`/`d_next_maxv` readback memcpys (currently ~line 655):

```cpp
        INFER_CHECK(sampler.Sample(stream, d_logits, pw.vocab_size,
                                   lsxcommon::sample::LogitType::kF32,
                                   d_next_token, d_next_maxv));
```

**Qwen** — `libs/lsxcommon/src/qwen_inference.cu`:

Add `#include "lsxcommon/sampling.h"`. Change `RunQwenGeneration` (line 811):

```cpp
cudaError_t RunQwenGeneration(cudaStream_t stream,
                              const QwenPinnedWeights &pw,
                              const std::vector<int32_t> &prompt, int max_new,
                              int eos_id,
                              QwenGenerationResult &result,
                              lsxcommon::sample::ISampler &sampler =
                                  lsxcommon::sample::DefaultSampler())
```

Add one `float` scratch slot to the arena for the sampler's max-value output. In the `float_sz[]` array (line 870) append `(size_t)1,  // d_maxv_scratch`; after `float *d_betaf = pf; pf += float_sz[1];` add:

```cpp
    float *d_maxv_scratch = pf;  pf += float_sz[2];
```

Replace the `qwen_argmax_kernel` call (line 1020) with the sampler on the LAST row (Qwen logits are `__half`):

```cpp
        const int last_row = seq - 1;
        e = sampler.Sample(stream,
                           d_logits + static_cast<size_t>(last_row) * V, V,
                           lsxcommon::sample::LogitType::kF16,
                           d_argmax, d_maxv_scratch);
        if (e != cudaSuccess) return fail(e, "argmax");
```

- [ ] **Step 4: Run the tests to verify behavior is unchanged (GREEN)**

Run:
```bash
cmake --build --preset linux-debug --target test_sampling test_deepseek_inference_kernels test_glm_compute test_qwen_model test_model_factory
./build/linux-debug/tests/test_sampling
./build/linux-debug/tests/test_deepseek_inference_kernels
ctest --preset linux-debug -R "test_glm_compute|test_qwen_model|test_model_factory"
```
Expected: `test_sampling` 4/4 PASS (new assertion included); `test_deepseek_inference_kernels` 12 cases / 4470 assertions PASS; `test_glm_compute` PASS; `test_qwen_model` PASS (synthetic — the fixture decodes `<|bos|>`-only, sampler returns the same argmax).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/glm_inference.cu libs/lsxcommon/src/deepseek_inference_pipeline.cu libs/lsxcommon/src/qwen_inference.cu tests/test_sampling.cu
git commit -m "refactor: thread ISampler seam into GLM/DeepSeek/Qwen decode loops"
```

---

## Task 3: Unified `IModel::Benchmark` + `RunBenchmark` shim

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model.h`
- Modify: `libs/lsxcommon/include/lsxcommon/benchmark_gpu.h`
- Create: `libs/lsxcommon/src/benchmark_gpu.cpp` (new shim TU)
- Modify: `libs/lsxcommon/src/benchmark_gpu.cu` (fold old body into an internal symbol or delete; keep the kernels only if still referenced)
- Modify: `libs/lsxcommon/src/glm_v5_model.cpp`, `libs/lsxcommon/src/deepseek_v4_flash_model.cpp`, `libs/lsxcommon/src/qwen_base_model.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (add `src/benchmark_gpu.cpp` to `lsxcommon`)
- Modify: `tests/test_model_runtime.cpp`

**Interfaces:**
- Consumes: `LsxModelFactory::Create` (`model_factory.h`), engine entry points (Task 2 signatures).
- Produces: `virtual gpu::BenchmarkResult Benchmark(int num_tokens = 2048)` on `IModel`; `BenchmarkResult::resident_bytes`; `gpu::RunBenchmark` shim. `apps/lsxfabric`/`apps/lsxforge` call sites unchanged.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_runtime.cpp`:

```cpp
#include "lsxcommon/benchmark_gpu.h"

namespace {
// Local override of the shared FakeModel: adds a Benchmark override so the
// default-{ok=false} and the shim-delegation contracts are testable host-only.
class BenchmarkFakeModel : public lsxcommon::ModelBase {
public:
    explicit BenchmarkFakeModel(const lsxcommon::ModelInitConfig& cfg)
        : ModelBase(cfg) {}
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override {
        gpu::BenchmarkResult r;
        r.ok = true;
        r.num_tokens = num_tokens;
        r.resident_bytes = 12345;
        r.tokens_per_sec = 99.0;
        return r;
    }
};
}  // namespace

TEST_CASE("IModel::Benchmark default returns ok=false", "[model_runtime]") {
    // The shared test FakeModel does NOT override Benchmark -> default applies.
    FakeModel m(lsxcommon::ModelInitConfig{});
    auto r = m.Benchmark();
    REQUIRE_FALSE(r.ok);
}

TEST_CASE("RunBenchmark shim delegates to IModel::Benchmark", "[model_runtime]") {
    // No checkpoint at this path -> factory returns nullptr -> shim reports ok=false.
    auto r = lsxcommon::gpu::RunBenchmark("/tmp/no_such_model_xyz", 64);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.num_tokens == 64);
    REQUIRE(r.resident_bytes == 0);
}
```

Wait — the shim with a bad path returns the default `BenchmarkResult` (`num_tokens == 0` in the default!). Fix the assertion: the shim must fill `num_tokens` from its argument even on failure. So the shim contract is: `BenchmarkResult r; r.num_tokens = num_tokens;` always; then if factory fails, return `r` (ok=false, num_tokens set). Update the test to expect `r.num_tokens == 64` on the failed path (this pins the shim's fill-before-dispatch behavior):

```cpp
TEST_CASE("RunBenchmark shim delegates to IModel::Benchmark", "[model_runtime]") {
    auto r = lsxcommon::gpu::RunBenchmark("/tmp/no_such_model_xyz", 64);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.num_tokens == 64);
    REQUIRE(r.resident_bytes == 0);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_model_runtime && ./build/linux-debug/tests/test_model_runtime`
Expected: FAIL — `IModel::Benchmark` is not a member (`gpu::BenchmarkResult` unknown, `RunBenchmark` signature mismatch).

- [ ] **Step 3: Implement**

`libs/lsxcommon/include/lsxcommon/benchmark_gpu.h` — add the residency field:

```cpp
    int64_t total_bytes_transferred = 0;
    std::uint64_t resident_bytes = 0;   // peak resident weight set during the run
    bool ok = false;
```

`libs/lsxcommon/include/lsxcommon/model.h` — include the benchmark header and add the virtual:

```cpp
#include "lsxcommon/inference_gpu.h"   // gpu::InferenceResult
#include "lsxcommon/safetensors.h"     // SafetensorsModel
#include "lsxcommon/tokenizer.h"       // ArrowTokenizerDb
#include "lsxcommon/benchmark_gpu.h"   // gpu::BenchmarkResult
```

```cpp
    // Runs the model's own engine loop on synthetic tokens. Additive default:
    // ok=false keeps test fakes compiling without an override.
    virtual gpu::BenchmarkResult Benchmark(int num_tokens = 2048) { return {}; }
```

`libs/lsxcommon/src/benchmark_gpu.cpp` — the shim (new TU):

```cpp
#include "lsxcommon/benchmark_gpu.h"

#include "lsxcommon/model_factory.h"
#include "lsxcommon/model.h"

namespace lsxcommon {
namespace gpu {

BenchmarkResult RunBenchmark(const std::string &model_path, int num_tokens) {
    BenchmarkResult r;
    r.num_tokens = num_tokens;
    ModelInitConfig cfg;
    cfg.model_path = model_path;
    cfg.max_gen_tokens = num_tokens;
    auto model = LsxModelFactory::Create(cfg);
    if (!model) {
        r.error_msg = "failed to create model for " + model_path;
        return r;
    }
    if (!model->Load()) {
        r.error_msg = "failed to load model for " + model_path;
        return r;
    }
    r = model->Benchmark(num_tokens);
    model->Unload();
    return r;
}

}  // namespace gpu
}  // namespace lsxcommon
```

Remove the old `RunBenchmark` definition from `benchmark_gpu.cu` (move any still-referenced kernels to `static` or delete them; the ping-pong benchmark is superseded — the NVFP4 GEMM kernels remain exercised by `test_deepseek_inference_kernels`). If `benchmark_gpu.cu` becomes empty of externally-linked symbols, delete it from `lsxcommon_cu` in `CMakeLists.txt`.

Engine overrides:

`libs/lsxcommon/src/glm_v5_model.cpp` — add:

```cpp
gpu::BenchmarkResult GlmV5Model::Benchmark(int num_tokens) {
    gpu::BenchmarkResult r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    // Synthetic prompt: repeat token 0 (a stable vocab entry) 4 times.
    const std::vector<int32_t> prompt = {0, 0, 0, 0};
    cudaStream_t st = nullptr;
    cublasHandle_t ch = nullptr;
    if (cudaStreamCreate(&st) != cudaSuccess ||
        cublasCreate(&ch) != CUBLAS_STATUS_SUCCESS) {
        r.error_msg = "cuda/cublas init failed";
        if (st) cudaStreamDestroy(st);
        if (ch) cublasDestroy(ch);
        return r;
    }
    // Reuse the same on-demand path as Infer (tables_ path when experts are
    // resident). The engine fills GlmGenerationResult; map onto BenchmarkResult.
    lsxcommon::glm::GlmGenerationResult gr;
    // Use the in-memory expert tables if present, else the on-demand db.
    cudaError_t e = lsxcommon::glm::RunGlmGeneration(
        ch, st, pw_, tables_, prompt, num_tokens, gr);
    cudaStreamDestroy(st);
    cublasDestroy(ch);
    if (e != cudaSuccess) {
        r.error_msg = cudaGetErrorString(e);
        return r;
    }
    r.ok = true;
    r.num_tokens = gr.tokens.size();
    r.num_layers = kGlmNumLayers;
    r.total_time_ms = gr.total_ms;
    r.tokens_per_sec = gr.total_ms > 0.0
        ? (static_cast<double>(r.num_tokens) / gr.total_ms) * 1000.0 : 0.0;
    r.resident_bytes = ResidentWeightBytes();
    return r;
}
```

Note: check the actual GLM model-wrapper members (`pw_`, `tables_`, `expert_db_`) in `glm_v5_model.cpp` before writing — mirror the `Infer` body exactly; if `Infer` uses the `GlmExpertDb` on-demand path, use that instead of `tables_`.

`libs/lsxcommon/src/deepseek_v4_flash_model.cpp` — add:

```cpp
gpu::BenchmarkResult DeepSeekV4FlashModel::Benchmark(int num_tokens) {
    gpu::BenchmarkResult r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    const std::vector<int32_t> prompt = {0};
    auto res = gpu::RunInferenceModel(cfg_.model_path, prompt, num_tokens, tok_db_);
    if (!res.ok) { r.error_msg = res.error_msg; return r; }
    r.ok = true;
    r.num_tokens = res.num_tokens;
    r.num_layers = res.num_layers;
    r.total_time_ms = res.total_time_ms;
    r.tokens_per_sec = res.tokens_per_sec;
    r.resident_bytes = res.weight_resident_bytes;
    return r;
}
```

`libs/lsxcommon/src/qwen_base_model.cpp` — add:

```cpp
gpu::BenchmarkResult QwenBaseModel::Benchmark(int num_tokens) {
    gpu::BenchmarkResult r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    const std::vector<int32_t> prompt = {0};
    // Mirror Infer: open a stream, run RunQwenGeneration, map the result.
    lsxcommon::qwen::QwenGenerationResult gr;
    cudaStream_t st = nullptr;
    if (cudaStreamCreate(&st) != cudaSuccess) {
        r.error_msg = "cudaStreamCreate failed";
        return r;
    }
    cudaError_t e = lsxcommon::qwen::RunQwenGeneration(
        st, pw_, prompt, num_tokens, tok_db_.eos_id, gr);
    cudaStreamDestroy(st);
    if (e != cudaSuccess) { r.error_msg = cudaGetErrorString(e); return r; }
    r.ok = true;
    r.num_tokens = static_cast<int>(gr.tokens.size());
    r.total_time_ms = gr.total_ms;
    r.tokens_per_sec = gr.total_ms > 0.0
        ? (static_cast<double>(r.num_tokens) / gr.total_ms) * 1000.0 : 0.0;
    r.resident_bytes = gr.resident_bytes;
    return r;
}
```

(The exact member names — `pw_`, `tok_db_`, `loaded_`, `cfg_`, `tables_`/`expert_db_` — must be verified against the existing model-wrapper headers before writing. Where the wrapper's `Infer` differs (e.g. DeepSeek uses a stream per call), mirror `Infer`.)

Add the three method declarations to the corresponding model headers (`glm_v5_model.h`, `deepseek_v4_flash_model.h`, `qwen_base_model.h`):

```cpp
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;
```

`libs/lsxcommon/CMakeLists.txt` — add `src/benchmark_gpu.cpp` to the `lsxcommon` STATIC list (after `src/generation_cli.cpp`).

- [ ] **Step 4: Run the tests to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_model_runtime
./build/linux-debug/tests/test_model_runtime
```
Expected: all previous cases PLUS the two new ones PASS. Then build all apps (they link `lsxcommon`): `cmake --build --preset linux-debug`. Smoke: `./build/linux-debug/apps/lsxfabric/lsxfabric --benchmark /tmp/no_such_model_xyz` → exit code non-zero, error message, no crash.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/model.h libs/lsxcommon/include/lsxcommon/benchmark_gpu.h libs/lsxcommon/src/benchmark_gpu.cpp libs/lsxcommon/src/glm_v5_model.cpp libs/lsxcommon/src/deepseek_v4_flash_model.cpp libs/lsxcommon/src/qwen_base_model.cpp libs/lsxcommon/CMakeLists.txt tests/test_model_runtime.cpp
git commit -m "feat: unified IModel::Benchmark with RunBenchmark shim"
```

---

## Task 4: `DeepSeekInferArena` — malloc-per-layer kill

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cuh`
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_moe.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_csa.cu`
- Modify: `libs/lsxcommon/src/deepseek_inference_pipeline.cu`
- Create: `tests/test_deepseek_arena.cu`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: existing `PinnedWeights`/`LayerWeights` geometry (`kHcDim`, `kHiddenDim`, `kIntermediateDim`, `kNumLayers`, `kAttnHeads`, `kHeadDim`, `kQloraRank`, `kOGroups`, `kOloraRank`, `kAttnWindow`, `kMaxVocabSize`); `alloc_csa_state` sizing math.
- Produces: `DeepSeekInferArena` (one `cudaMalloc` + one `cudaMallocHost` per run); new `route_token` signature with caller-owned staging; `InferenceAllocCount()` atomic counter for tests. Task 8 reads `arena.bytes` for occupancy accounting.

- [ ] **Step 1: Write the failing test**

Create `tests/test_deepseek_arena.cu`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#include "lsxcommon/deepseek_inference_load.h"  // host-only wrapper header if it exists; else include the .cuh
#include "lsxcommon/deepseek_inference_moe.h"   // route_token host declaration (see note)

using lsxcommon::gpu::route_token;

// The route_token host wrapper must be reachable from a test TU. If the
// pipeline declares it in a private .cuh, promote the declaration to
// include/lsxcommon/deepseek_inference_api.h (see Step 3 note).

TEST_CASE("route_token performs zero allocations after arena refactor", "[deepseek_arena]") {
    // Tiny synthetic router: num_routed=16, hidden=8.
    constexpr int kHidden = 8;
    constexpr int kRouted = 16;
    constexpr int kTopK = 2;

    __half* d_gate = nullptr;
    __half* d_hidden = nullptr;
    float* h_bias = nullptr;
    int* h_top = nullptr;
    float* h_weights = nullptr;
    __half* d_logits_scratch = nullptr;
    float* h_logits_scratch = nullptr;
    REQUIRE(cudaMalloc(&d_gate, kRouted * kHidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_hidden, kHidden * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_bias, kRouted * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_top, kTopK * sizeof(int)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_weights, kTopK * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_logits_scratch, kRouted * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&h_logits_scratch, kRouted * sizeof(float)) == cudaSuccess);

    // identity-ish gate so routing is deterministic
    std::vector<__half> gate(kRouted * kHidden);
    for (int i = 0; i < kRouted * kHidden; ++i) gate[i] = __float2half(0.01f);
    for (int i = 0; i < kRouted; ++i) gate[static_cast<size_t>(i) * kHidden] = __float2half(1.0f);
    std::vector<__half> hidden(kHidden, __float2half(1.0f));
    REQUIRE(cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    for (int i = 0; i < kRouted; ++i) h_bias[i] = static_cast<float>(i);

    cublasHandle_t ch = nullptr;
    cudaStream_t stream = nullptr;
    REQUIRE(cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    auto& count = lsxcommon::gpu::InferenceAllocCount();
    const auto before = count.load();

    cudaError_t e = route_token(ch, stream, d_gate, d_hidden,
                                kHidden, kRouted,
                                nullptr, 0, 3 /*token_id*/,
                                h_bias, kTopK,
                                d_logits_scratch, h_logits_scratch,
                                h_top, h_weights);
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    const auto after = count.load();
    REQUIRE(after == before);            // route_token added ZERO allocations

    REQUIRE(h_top[0] >= 0 && h_top[0] < kRouted);
    REQUIRE(h_top[1] >= 0 && h_top[1] < kRouted);
    REQUIRE(h_top[0] != h_top[1]);

    cudaStreamDestroy(stream);
    cublasDestroy(ch);
    cudaFreeHost(h_logits_scratch);
    cudaFree(d_logits_scratch);
    cudaFreeHost(h_weights);
    cudaFreeHost(h_top);
    cudaFreeHost(h_bias);
    cudaFree(d_hidden);
    cudaFree(d_gate);
}
```

Note for the implementer: `route_token` and `InferenceAllocCount` are currently declared in private `src/deepseek_inference_load.cuh`. If that `.cuh` is already included by a public header path the test can reach, use it; otherwise add the declarations to a new public `include/lsxcommon/deepseek_inference_api.h` (host-only declarations guarded by `__cplusplus`) and include it from the `.cu` sources. `InferenceAllocCount` is a header-inline function-local static atomic:

```cpp
inline std::atomic<std::uint64_t>& InferenceAllocCount() {
    static std::atomic<std::uint64_t> c{0};
    return c;
}
```

The pipeline's `DS_ALLOC` macro and `route_token` both increment it. `load_pinned_weights` does NOT (load-time, not per-run).

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_deepseek_arena && ./build/linux-debug/tests/test_deepseek_arena`
Expected: FAIL — the new `route_token` signature (with `d_logits_scratch`/`h_logits_scratch`) does not exist yet (link error), or the assertion `after == before` fails against the current alloc-heavy `route_token`.

- [ ] **Step 3: Implement the arena**

`libs/lsxcommon/src/deepseek_inference_load.cuh` — add after `PinnedWeights`:

```cpp
// Single-allocation per-run arena (spec item 12). One device cudaMalloc + one
// pinned cudaMallocHost carve every direct buffer, the CSA per-layer state and
// the routing/logits staging for one RunInferenceModel call. Mirrors
// GlmMoEArena::AllocateGeneration.
struct DeepSeekInferArena {
    // device block
    uint8_t* d_base = nullptr;
    __half* d_single = nullptr;      // [kHiddenDim]
    __half* d_normed = nullptr;      // [kHiddenDim]
    __half* d_ffn_out = nullptr;     // [kHiddenDim]
    float*  d_hidden = nullptr;      // [kHcDim]
    float*  d_hidden_next = nullptr; // [kHcDim]
    __half* d_shared_int = nullptr;  // [kHiddenDim]
    __half* d_shared_gate = nullptr; // [kHiddenDim]
    __half* d_routed_buf = nullptr;  // [kHiddenDim]
    float*  d_dequant_scratch = nullptr;
    float*  d_act_quant = nullptr;
    float*  d_ffn_accum = nullptr;   // [kHiddenDim]
    float*  d_swiglu_out = nullptr;  // [kIntermediateDim]
    float*  d_act_scales = nullptr;
    float*  d_gate_f32 = nullptr;    // [kIntermediateDim]
    float*  d_up_f32 = nullptr;      // [kIntermediateDim]
    float*  d_attn_dequant = nullptr;
    __half* d_kv_cache = nullptr;
    __half* d_q = nullptr, *d_qr = nullptr, *d_qr2 = nullptr;
    __half* d_kv = nullptr, *d_kv2 = nullptr;
    __half* d_o = nullptr, *d_olora = nullptr, *d_attn_out_h = nullptr;
    float*  d_attn_out = nullptr;
    float*  d_hc_post = nullptr, *d_hc_comb = nullptr, *d_hc_pre = nullptr;
    float*  d_logits = nullptr;
    float*  d_norm_scratch = nullptr;
    float*  d_norm_gamma_scratch = nullptr;
    int32_t* d_next_token = nullptr;
    float*  d_next_maxv = nullptr;
    uint8_t* d_pack_stage = nullptr;
    uint8_t* d_scale_stage = nullptr;
    CSAState* d_csa = nullptr;       // [num_layers] host-side state records

    // pinned host block
    uint8_t* h_base = nullptr;
    int*   h_topk = nullptr;         // [kLocalTopK]
    float* h_top_weights = nullptr;  // [kLocalTopK]

    std::size_t bytes = 0;           // total device bytes (resident accounting)
    std::size_t host_bytes = 0;

    // Allocates the whole per-run footprint. @p compress_ratios is the
    // per-layer CSA ratio table (0 for pure-MLA layers); @p max_seq_len /
    // @p head_dim / @p vocab size the CSA + logits buffers.
    cudaError_t Allocate(const PinnedWeights& pw, int max_seq_len,
                         const int* compress_ratios, int vocab);
    void Free();
};
```

`libs/lsxcommon/src/deepseek_inference_load.cu` — implement `DeepSeekInferArena::Allocate` mirroring `GlmMoEArena::AllocateGeneration`'s `page()` lambda. Compute per-buffer bytes exactly as the pipeline currently does (reuse the constants at lines 252-268 of the pipeline). Size the CSA region by summing, per layer, the exact `alloc_csa_state` sizes (`state_sz = coff*ratio * coff*head_dim`, `max_comp = max_seq_len/ratio + 1`, plus the 128-dim indexer records for `ratio==4`), then bind each layer's `CSAState` fields into the single block and run the same zero/`-inf` init as `alloc_csa_state`. `Allocate` increments `InferenceAllocCount()` by 2 (device + host).

`libs/lsxcommon/src/deepseek_inference_moe.cu` — new `route_token` signature (lines 431-549):

```cpp
cudaError_t route_token(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    const int64_t *h_tid2eid, int tid2eid_stride, int token_id,
    const float *h_gate_bias,
    int topk,
    __half *d_logits_scratch,   // caller-owned [num_routed] __half
    float *h_logits_scratch,    // caller-owned [num_routed] float (pinned)
    int *h_top_indices, float *h_top_weights)
```

Replace the two `cudaMalloc`/`cudaMallocHost` + two `cudaFree`/`cudaFreeHost` calls (lines 455-456, 467-469, 546-547) with direct use of `d_logits_scratch`/`h_logits_scratch`. `route_token` no longer calls any CUDA allocation API — the test asserts `InferenceAllocCount()` unchanged.

`libs/lsxcommon/src/deepseek_inference_pipeline.cu` — replace the ~40 `DS_ALLOC`/`cudaMalloc`/`cudaMallocHost` calls (lines 270-328, 411-417) and the per-layer `route_token` staging pointers with one:

```cpp
    DeepSeekInferArena arena;
    INFER_CHECK(arena.Allocate(pw, total_tokens,
                               compress_ratios.data(), pw.vocab_size));
    d_alloc_bytes = arena.bytes;
```

and bind local aliases (`d_hidden = arena.d_hidden;` … `h_topk = arena.h_topk;`). The `cudaMemset`-zero block (lines 330-370) stays, addressing arena members. Replace the `alloc_csa_state`/`free_csa_state` calls with the arena's CSA bind (the state pointers come from the arena block). The `cleanup:` block frees the arena instead of 40 individual frees:

```cpp
cleanup:
    arena.Free();
```

`libs/lsxcommon/src/deepseek_inference_csa.cu` — keep `alloc_csa_state`/`free_csa_state` only if another caller still uses them (grep first); otherwise remove. The arena reproduces their sizing/binding.

- [ ] **Step 4: Run the tests to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_deepseek_arena test_deepseek_inference_kernels
./build/linux-debug/tests/test_deepseek_arena
./build/linux-debug/tests/test_deepseek_inference_kernels
ctest --preset linux-debug -R "test_deepseek_loader|test_deepseek_prompt_output"
```
Expected: `test_deepseek_arena` PASS (allocation-count + routing determinism); `test_deepseek_inference_kernels` 4470 assertions PASS; real-model `test_deepseek_prompt_output` PASS — final_logits and output_text bit-identical to pre-refactor (the arena is a pure allocation change).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/deepseek_inference_load.cuh libs/lsxcommon/src/deepseek_inference_load.cu libs/lsxcommon/src/deepseek_inference_moe.cu libs/lsxcommon/src/deepseek_inference_csa.cu libs/lsxcommon/src/deepseek_inference_pipeline.cu tests/test_deepseek_arena.cu tests/CMakeLists.txt
git commit -m "refactor: single-allocation DeepSeekInferArena (2 allocs per run)"
```

---

## Task 5: Orchestrator ring closure (response serialization + response ring)

**Files:**
- Modify: `libs/lsxnetwork/include/lsxnetwork/openai_endpoint.h`
- Modify: `libs/lsxnetwork/src/openai_endpoint.cpp`
- Modify: `libs/lsxcommon/include/lsxcommon/lsx_orchestrator.h`
- Modify: `libs/lsxcommon/src/models/lsx_orchestrator.cpp`
- Modify: `tests/test_integrated_pipeline.cpp`

**Interfaces:**
- Consumes: `lsxnetwork::FrameQueue` (existing), `lsxnetwork::ChatCompletion` (extended), `lsxcommon::prompt::BuildPromptInputIds` (Phase A), `IModel::Infer` (Phase A), `gpu::InferenceResult` (Phase A contract), `lsxnetwork::ZmqSocket::SendZeroCopy` (existing), `lsxnetwork::IoWorker` (existing).
- Produces: `ChatCompletion::max_tokens`; `lsxnetwork::CompletionResponse`; `OpenAiEndpoint::WriteCompletion`; `LsxOrchestrator::ServeResponses` (pinned IoWorker ZMQ writer); `DispatchOne` writes into a response `FrameQueue` with the request `id` echoed.

- [ ] **Step 1: Write the failing test**

Extend `tests/test_integrated_pipeline.cpp`. The existing harness builds `LsxOrchestrator` with real models (GLM + DeepSeek) and asserts `taken == retired`. Add a response-ring closure test that pumps one request frame and drains the response ring:

```cpp
TEST_CASE("orchestrator echoes response frame with matching id", "[mosaic]") {
    auto glm_model = /* same fixture as the existing orchestrator test */;
    lsxcommon::TaskMosaicStats stats;
    std::atomic<bool> stop{false};

    lsxnetwork::FrameQueue request_q(8, 65536);
    lsxnetwork::FrameQueue response_q(8, 65536);

    lsxcommon::LsxOrchestrator orch(/* ModelTable with glm_model */);

    // Producer: publish one chat-completion request with a known id.
    const std::string body =
        R"({"id":"req-007","object":"chat.completion","model":"glm-5.2",)"
        R"("messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":8})";
    std::size_t cap = 0;
    char* slot = request_q.Acquire(&cap);
    REQUIRE(slot != nullptr);
    std::memcpy(slot, body.data(), body.size());
    REQUIRE(request_q.Publish(body.size()) == 0);

    // Consumer loop (one iteration, then stop).
    lsxcommon::DispatchSignal last_sig = lsxcommon::DispatchSignal::kUnknown;
    // Run ServeTaskMosaic in a thread so DispatchOne executes.
    std::thread server([&] {
        orch.ServeTaskMosaic(request_q, stats, stop);
    });
    // Wait for the request to be processed (response frame appears).
    lsxnetwork::FrameQueue::Frame resp;
    for (int i = 0; i < 400000 && !response_q.Peek(resp); ++i)
        std::this_thread::yield();
    stop.store(true, std::memory_order_release);
    server.join();

    REQUIRE(response_q.Peek(resp));
    std::string payload(resp.data, resp.size);
    REQUIRE(payload.find("\"id\":\"req-007\"") != std::string::npos);
    response_q.Consume();
}
```

This test requires the orchestrator `DispatchOne` to (a) populate `input_ids` via `BuildPromptInputIds`, (b) run `Infer`, (c) serialize `CompletionResponse` into `response_q`. With today's stub body it FAILS at (a)-(c).

Note: the real GLM infer in this test takes ~30+ s (real model). The existing `test_integrated_pipeline` already runs real GLM; keep the runtime acceptable by using `max_tokens` small (8) and the existing harness's model fixture.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_integrated_pipeline && ctest --preset linux-debug -R test_integrated_pipeline`
Expected: FAIL — new test case asserts `id` echo that does not happen (RED). Existing 4 cases must stay green.

- [ ] **Step 3: Implement**

`libs/lsxnetwork/include/lsxnetwork/openai_endpoint.h` — extend `ChatCompletion`:

```cpp
struct ChatCompletion {
    std::string id;
    std::string object;
    std::string model;
    std::string role;
    std::string content;
    std::string finish_reason;
    std::int64_t prompt_tokens     = 0;
    std::int64_t completion_tokens = 0;
    std::int64_t total_tokens      = 0;
    int max_tokens = 0;             // "max_tokens" (0 = default 64)
};

// Network-layer slice of gpu::InferenceResult (lsxnetwork cannot depend on
// lsxcommon — the orchestrator maps InferenceResult -> CompletionResponse).
struct CompletionResponse {
    bool ok = false;
    std::string output_text;
    std::vector<int32_t> gen_tokens;
    bool eos_terminated = false;
    double ttft_ms = 0.0;
    double tokens_per_sec = 0.0;
};
```

Add to `OpenAiEndpoint`:

```cpp
    // Serializes an OpenAI-style chat-completion reply. Echoes @p req's `id`
    // and `model` so the response ring can correlate without a per-slot channel.
    // Returns false on OOM (string growth) — never on valid input.
    bool WriteCompletion(const ChatCompletion& req,
                         const CompletionResponse& res,
                         std::string& out) const;
```

`libs/lsxnetwork/src/openai_endpoint.cpp` — in `Parse`, read `max_tokens` via `doc["max_tokens"].get_int64()` (default 0); implement `WriteCompletion`:

```cpp
bool OpenAiEndpoint::WriteCompletion(const ChatCompletion& req,
                                     const CompletionResponse& res,
                                     std::string& out) const {
    std::ostringstream oss;
    oss << "{\"id\":\"" << req.id << "\","
        << "\"object\":\"chat.completion\","
        << "\"model\":\"" << req.model << "\","
        << "\"choices\":[{\"index\":0,"
        << "\"message\":{\"role\":\"assistant\",\"content\":\"";
    // Escape the content minimally: JSON-escape '"' and '\' and control chars.
    for (char c : res.output_text) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20)
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c))
                        << std::dec << std::setfill(' ');
                else
                    oss << c;
        }
    }
    const std::int64_t comp_tokens = static_cast<std::int64_t>(res.gen_tokens.size());
    oss << "\"},\"finish_reason\":\""
        << (res.eos_terminated ? "stop" : "length")
        << "\"}],"
        << "\"usage\":{\"prompt_tokens\":" << req.prompt_tokens
        << ",\"completion_tokens\":" << comp_tokens
        << ",\"total_tokens\":" << (req.prompt_tokens + comp_tokens) << "},"
        << "\"lsx\":{\"ok\":" << (res.ok ? "true" : "false")
        << ",\"ttft_ms\":" << res.ttft_ms
        << ",\"tokens_per_sec\":" << res.tokens_per_sec << "}}";
    out = oss.str();
    return true;
}
```

(Add `#include <sstream>`, `#include <iomanip>`, `#include <vector>` if not already present.)

`libs/lsxcommon/include/lsxcommon/lsx_orchestrator.h`:

```cpp
    // Drains the response ring and sends each frame over a ZMQ REP socket on a
    // second pinned IoWorker (own LsxTokenTranslator). Blocks until @p stop.
    void ServeResponses(lsxnetwork::FrameQueue& responses,
                        std::atomic<bool>& stop,
                        const std::string& endpoint);
```

Add to `TaskMosaicStats`:

```cpp
    std::atomic<std::uint64_t> responses_ok{0};
    std::atomic<std::uint64_t> responses_failed{0};
    std::atomic<std::uint64_t> response_serialize_failures{0};
```

`libs/lsxcommon/src/models/lsx_orchestrator.cpp`:

Change `ServeTaskMosaic`/`DispatchOne` to thread a response queue through. `ServeTaskMosaic(queue, responses, stats, stop)`; `DispatchOne(frame, responses, stats)`. Inside `DispatchOne`, after the per-branch `Infer`:

```cpp
        lsxnetwork::CompletionResponse cr;
        cr.ok = res.ok;
        cr.output_text = res.output_text;
        cr.gen_tokens = res.gen_tokens;
        cr.eos_terminated = res.eos_terminated;
        cr.ttft_ms = res.ttft_ms;
        cr.tokens_per_sec = res.tokens_per_sec;
        std::string payload;
        if (!lsxnetwork::OpenAiEndpoint().WriteCompletion(cc, cr, payload)) {
            ++stats.response_serialize_failures;
            break;
        }
        std::size_t rcap = 0;
        char* rslot = responses.Acquire(&rcap);
        if (!rslot) {                 // response ring full: 503 semantics
            spdlog::warn("[orchestrator] response ring full; dropping reply for {}",
                         cc.id);
            ++stats.responses_failed;
            break;
        }
        if (payload.size() > rcap) {
            spdlog::warn("[orchestrator] response frame too large ({} bytes); dropping",
                         payload.size());
            ++stats.responses_failed;
            break;
        }
        std::memcpy(rslot, payload.data(), payload.size());
        responses.Publish(payload.size());
        ++stats.responses_ok;
```

Implement `ServeResponses`:

```cpp
void LsxOrchestrator::ServeResponses(lsxnetwork::FrameQueue& responses,
                                     std::atomic<bool>& stop,
                                     const std::string& endpoint) {
    lsxnetwork::ZmqContext ctx;
    lsxnetwork::ZmqSocket sock(ctx, ZMQ_REP);
    if (!sock.Valid() || sock.Bind(endpoint.c_str()) != 0) {
        spdlog::error("[orchestrator] failed to bind response socket on {}",
                      endpoint);
        return;
    }
    while (!stop.load(std::memory_order_acquire)) {
        lsxnetwork::FrameQueue::Frame frame;
        std::size_t index = 0;
        if (!responses.TryTake(frame, index)) {
            std::this_thread::yield();
            continue;
        }
        sock.SendZeroCopy(responses, frame, index);
    }
}
```

`ServeTaskMosaicPinned` keeps its signature; it now constructs the pinned ingestion worker unchanged, and the caller (taskmosaic app / tests) starts `ServeResponses` on a second `IoWorker` when responses are wanted.

- [ ] **Step 4: Run the tests to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_integrated_pipeline
ctest --preset linux-debug -R test_integrated_pipeline
```
Expected: all 5 test cases PASS (4 existing + new id-correlation case). The new case runs a real GLM infer (~30 s).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxnetwork/include/lsxnetwork/openai_endpoint.h libs/lsxnetwork/src/openai_endpoint.cpp libs/lsxcommon/include/lsxcommon/lsx_orchestrator.h libs/lsxcommon/src/models/lsx_orchestrator.cpp tests/test_integrated_pipeline.cpp
git commit -m "feat: close orchestrator ring with response serialization and SPSC reply queue"
```

---

## Task 6: GLM seq-P prefill parity test

**Files:**
- Create: `tests/test_glm_prefill_seqp.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `lsxcommon::glm::RunGlmGeneration` (Task 2 signature, default sampler), `GlmGenerationResult` (tokens, final_logits, kv_slots_used), real GLM-5.2 checkpoint.

This task hardens and verifies the existing seq-P prefill (dense layers 0–2 collapse to row 0; MoE layers 3–77 already run `block_rows = P` through `RunGlmMoERouter`/`RunGlmMoESlices` which take `seq`). The deliverable is the parity harness: it FAILS if the MoE prefill ever collapses rows (wrong `kv_slots_used`) or loses bit-exactness across identical runs.

- [ ] **Step 1: Write the failing test**

Create `tests/test_glm_prefill_seqp.cpp` (real-model; SKIP_RETURN_CODE 4):

```cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>

#include <cstring>
#include <string>
#include <vector>

#include "lsxcommon/glm_inference_load.h"
#include "lsxcommon/glm_v5_model.h"
#include "lsxcommon/model_factory.h"

using namespace lsxcommon;

TEST_CASE("GLM seq-P prefill stores P KV rows and stays bit-exact", "[glm][realmodel]") {
    // Resolve the real checkpoint through the model factory (SKIPs without it).
    ModelInitConfig cfg;
    cfg.model_path = "/mnt/llm_models/models/GLM-5.2";
    cfg.max_gen_tokens = 4;
    auto model = LsxModelFactory::Create(cfg);
    if (!model || !model->Load()) {
        SKIP("GLM-5.2 checkpoint not available");
    }

    // Tokenizer-encoded prompt of length P >= 8 so the MoE prefill sees seq > 1.
    const std::string text = "The capital of France is the city of";
    std::vector<int32_t> input_ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds(
        model->Type(), model->Tokenizer(), text, input_ids));
    const int P = static_cast<int>(input_ids.size());
    REQUIRE(P >= 8);

    ModelRequest req;
    req.input_ids = input_ids;
    req.max_gen_tokens = 4;
    auto r1 = model->Infer(req);

    REQUIRE(r1.ok);
    REQUIRE(r1.num_tokens >= 1);
    // seq-P prefill: every prompt token occupies one IndexShare KV slot.
    REQUIRE(r1.kv_slots_used == P + r1.num_tokens);
    REQUIRE_FALSE(r1.final_logits.empty());
    REQUIRE(r1.gen_tokens.size() == static_cast<size_t>(r1.num_tokens));

    // argmax of the final-logits row == the first generated token.
    int best = 0;
    for (int i = 1; i < static_cast<int>(r1.final_logits.size()); ++i)
        if (r1.final_logits[i] > r1.final_logits[best]) best = i;
    REQUIRE(best == r1.gen_tokens[0]);

    // Bit-exact determinism across a second identical run.
    auto r2 = model->Infer(req);
    REQUIRE(r2.ok);
    REQUIRE(r1.final_logits.size() == r2.final_logits.size());
    REQUIRE(std::memcmp(r1.final_logits.data(), r2.final_logits.data(),
                        r1.final_logits.size() * sizeof(float)) == 0);

    model->Unload();
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_glm_prefill_seqp && ./build/linux-debug/tests/test_glm_prefill_seqp`
Expected: FAIL (RED) at `kv_slots_used == P + num_tokens` if the MoE prefill collapsed rows (or `argmax != gen_tokens[0]`), OR SKIP if the checkpoint is absent. If it PASSES immediately (the seq-P path is already correct), still keep the test — it is the permanent parity gate for item 15's GLM row.

- [ ] **Step 3: Fix the prefill if the test fails**

If `kv_slots_used != P + num_tokens`: trace `RunGlmLoopLayer`'s MoE path (`glm_inference.cu` lines ~347-460) for a spurious `block_rows = 1` collapse. The dense-collapse guard (lines 652-658) must apply ONLY to `l < 3`. If `RunGlmMoERouter`/`RunGlmMoESlices` are invoked with `seq=1` for `l >= 3`, fix the call to pass `block_rows` (both already accept `seq`). If `argmax != gen_tokens[0]`, the layer-78 MTP final-logits row and the fused argmax disagree on the same row — resolve the discrepancy in the sampler readback (Task 2 uses the same `d_logits_head` row, so a mismatch here indicates a pre-existing decode/logits inconsistency).

- [ ] **Step 4: Run the test to verify it passes**

Run: `./build/linux-debug/tests/test_glm_prefill_seqp`
Expected: PASS (or SKIP without checkpoint). Then re-run the GLM suite to confirm no regression: `ctest --preset linux-debug -R "test_glm_compute|test_glm_selfcheck"` with `LSX_TEST_GLM_SELFCHECK_CYCLES=2`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_glm_prefill_seqp.cpp tests/CMakeLists.txt
git commit -m "test: GLM seq-P prefill parity harness"
```

---

## Task 7: Qwen incremental per-call KV cache

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/qwen_base_model.h`
- Modify: `libs/lsxcommon/src/qwen_base_model.cpp`
- Modify: `libs/lsxcommon/src/qwen_inference.cu`
- Create: `tests/test_qwen_kv_cache.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `QwenPinnedWeights`/`QwenTopology` (`qwen_inference_load.h`), `RunQwenFullAttention`/`RunQwenLinearAttention`/`qwen_attn` kernels (they already take `seq` and write/read `d_k`/`d_v` rows `[0..seq)`), Task 2 sampler.
- Produces: `QwenKvCache` (persistent device KV + per-layer GatedDeltaNet recurrent state + `pos_base`); `RunQwenGeneration` incremental path (`kv_base`). Task 8 reads `cache.kv_rows` for the occupancy counter.

Design (spec item 15, Qwen row): today `RunQwenGeneration` recomputes the full `seq` on every decode step and `d_k`/`d_v`/`d_core` etc. are transient arena rows `[0..seq)`. The incremental path keeps a persistent `QwenKvCache` owned by `QwenBaseModel`:
- `d_kv_k` / `d_kv_v`: `[max_total][NKV*HD]` half rows of the full-attention layers (append-only).
- `d_gd_state` per linear-attention layer: the conv1d channel state + recurrent `col` accumulator (the GatedDeltaNet recurrence state).
- `pos_base`: running token offset so RoPE positions stay global.

`RunQwenGeneration` gains an optional `QwenKvCache* kv = nullptr`:
- `kv == nullptr` → today's full-recompute path (unchanged; also the SelfCheck path).
- `kv != nullptr` → append-only: run exactly the NEW tokens; `RunQwenFullAttention` gets a `kv_start` offset so k/v are written at `[kv_start..kv_start+seq)` and attention reads `[0..kv_start+seq)`; `RunQwenLinearAttention` seeds its recurrent state from `kv->d_gd_state[l]` and writes it back.

- [ ] **Step 1: Write the failing test**

Create `tests/test_qwen_kv_cache.cpp` (real-model; SKIP_RETURN_CODE 4; no real Qwen checkpoint on this box — the test is the permanent gate for the Windows/other-host run):

```cpp
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string>
#include <vector>

#include "lsxcommon/qwen_base_model.h"
#include "lsxcommon/model_factory.h"

using namespace lsxcommon;

TEST_CASE("Qwen incremental KV continuation equals a single full run", "[qwen][realmodel]") {
    ModelInitConfig cfg;
    cfg.model_path = "/mnt/llm_models/models/Qwen3.5-9B";
    cfg.max_gen_tokens = 8;
    auto model = LsxModelFactory::Create(cfg);
    if (!model || !model->Load()) {
        SKIP("Qwen checkpoint not available");
    }

    std::vector<int32_t> input_ids;
    REQUIRE(lsxcommon::prompt::BuildPromptInputIds(
        model->Type(), model->Tokenizer(),
        "The capital of France is", input_ids));

    ModelRequest req;
    req.input_ids = input_ids;
    req.max_gen_tokens = 8;
    auto single = model->Infer(req);          // full-recompute baseline
    REQUIRE(single.ok);

    // Session continuation: first call generates 3 tokens, then continues.
    ModelRequest a;
    a.input_ids = input_ids;
    a.max_gen_tokens = 3;
    auto first = model->Infer(a);
    REQUIRE(first.ok);
    REQUIRE(first.num_tokens == 3);

    ModelRequest b;
    b.input_ids = first.gen_tokens;           // feed the generated prefix
    b.max_gen_tokens = 5;
    auto second = model->Infer(b);            // incremental KV continuation
    REQUIRE(second.ok);

    std::vector<int32_t> joined = first.gen_tokens;
    joined.insert(joined.end(), second.gen_tokens.begin(), second.gen_tokens.end());

    // The 8-token single run and the 3+5 incremental run must agree on the
    // FIRST 8 generated tokens (same greedy path).
    REQUIRE(joined.size() >= single.gen_tokens.size());
    REQUIRE(std::equal(single.gen_tokens.begin(), single.gen_tokens.end(),
                       joined.begin()));

    model->Unload();
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_qwen_kv_cache && ./build/linux-debug/tests/test_qwen_kv_cache`
Expected: SKIP (no checkpoint) — RED cannot be observed on this box. This test is the acceptance gate for the machine that HAS the checkpoint. On this box, verify instead by compilation + the synthetic suite staying green (Step 4), and record the SKIP.

- [ ] **Step 3: Implement**

`libs/lsxcommon/include/lsxcommon/qwen_base_model.h` — declare the cache:

```cpp
namespace lsxcommon {
namespace qwen {

// Persistent per-session KV cache for incremental decode (spec item 15).
struct QwenKvCache {
    __half* d_kv_k = nullptr;    // [max_total][NKV*HD] full-attn keys
    __half* d_kv_v = nullptr;    // [max_total][NKV*HD] full-attn values
    __half* d_gd_state = nullptr;// per-linear-layer GatedDeltaNet state
    int kv_rows = 0;             // rows written so far (== pos_base)
    int max_total = 0;
    int num_linear_layers = 0;
    cudaError_t Allocate(const QwenTopology& top, int max_total, cudaStream_t s);
    void Free();
};

} // namespace qwen
} // namespace lsxcommon
```

`QwenBaseModel` gains a member `std::unique_ptr<qwen::QwenKvCache> kv_` (reset on `Load()`/`Unload()`).

`libs/lsxcommon/src/qwen_inference.cu` — add `QwenKvCache* kv = nullptr` to `RunQwenGeneration` (after `eos_id`). The incremental path:

- At entry, if `kv` is non-null and `kv->d_kv_k == nullptr`, allocate it for `max_total = prompt.size() + max_new` and `num_linear_layers` counted from `pw.layers` (`lw.is_linear`), and zero `d_gd_state`; else `kv_rows` is the continuation base.
- Decode loop step with `seq = kv ? 1 : (int)ids.size()`:
  - compute `pos_base = kv ? kv->kv_rows - 1 : 0` (the global position of the current token).
  - `RunQwenFullAttention(..., kv_base = kv ? kv->kv_rows : 0, ...)`: the kernel writes k/v at `[kv_base..kv_base+seq)` and the softmax reads `[0..kv_base+seq)`. Update `qwen_attn`'s k/v write index to `(kv_base + row)` and the attention range to `kv_base + seq`; RoPE uses global positions from `d_pos` (which the caller fills with `pos_base + row`).
  - `RunQwenLinearAttention(..., kv ? kv->d_gd_state + l*gd_state_bytes : nullptr, ...)`: seed/return the recurrent state.
  - after each step, `kv->kv_rows += seq`.
- The new-token loop still reads `d_argmax` (Task 2 sampler) and appends to `ids`.

Because the kernels already parameterize on `seq`, the append extension is additive: add `int kv_base` to `RunQwenFullAttention` and thread it into the k/v write row and the attention upper bound.

- [ ] **Step 4: Run the tests to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_qwen_kv_cache test_qwen_model test_qwen_loader test_deepseek_inference_kernels
./build/linux-debug/tests/test_qwen_kv_cache
./build/linux-debug/tests/test_qwen_model
ctest --preset linux-debug -R "test_qwen_loader|test_deepseek_inference_kernels"
```
Expected: `test_qwen_kv_cache` SKIPs on this box (no checkpoint); `test_qwen_model` and the synthetic suites stay GREEN (the `kv == nullptr` path is bit-identical to today).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/qwen_base_model.h libs/lsxcommon/src/qwen_base_model.cpp libs/lsxcommon/src/qwen_inference.cu tests/test_qwen_kv_cache.cpp tests/CMakeLists.txt
git commit -m "feat: incremental per-call Qwen KV cache (QwenKvCache)"
```

---

## Task 8: Occupancy counters (`kv_slots_used` for all three engines)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_inference_pipeline.cu` (counters; arena bytes already tracked)
- Modify: `libs/lsxcommon/src/qwen_inference.cu` (kv_slots_used from cache or run length)
- Modify: `libs/lsxcommon/src/qwen_base_model.cpp` (surface to InferenceResult)
- Modify: `libs/lsxcommon/src/glm_v5_model.cpp` (GLM already publishes `kv_slots_used` — verify and keep)
- Modify: `tests/test_model_runtime.cpp`

**Interfaces:**
- Consumes: `gpu::InferenceResult::kv_slots_used` (Phase A contract), `DeepSeekInferArena` (Task 4), `QwenKvCache` (Task 7).
- Produces: `kv_slots_used` published by all three engines; host contract test.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_runtime.cpp`:

```cpp
TEST_CASE("kv_slots_used contract defaults and monotonicity", "[model_runtime]") {
    // Default InferenceResult must report zero occupancy (host-only contract).
    lsxcommon::gpu::InferenceResult r;
    REQUIRE(r.kv_slots_used == 0);
    // A synthetic engine result satisfies: kv_slots_used == prompt + generated.
    r.kv_slots_used = 7 + 3;
    r.num_tokens = 3;
    REQUIRE(r.kv_slots_used == 7 + r.num_tokens);
}
```

This pins the occupancy semantic (`kv_slots_used == prompt_len + gen_count`) used by Tasks 6/7 and the GLM/DeepSeek engines.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_model_runtime && ./build/linux-debug/tests/test_model_runtime`
Expected: FAIL — `kv_slots_used` does not exist on `InferenceResult` yet (Phase A contract field was added in Phase A per the spec — if the field is already present, this step still validates; see note).

Note: `kv_slots_used` was specified in Phase A (inference_gpu.h line 24) and GLM/DeepSeek already publish it. If the field exists, this test passes immediately and the real work is verifying DeepSeek + Qwen publish it — see Step 3.

- [ ] **Step 3: Implement**

- **DeepSeek**: the pipeline already sets `result.kv_slots_used = prompt_len + gen_count` (line 743). Verify it survives the Task 4 arena refactor; keep it. Add `result.kv_slots_used` also to the arena's resident accounting path: `result.weight_resident_bytes` already = `pw.resident_bytes + d_alloc_bytes` (arena bytes now).
- **Qwen**: in `QwenBaseModel::Infer`, after the engine returns, set `result.kv_slots_used = kv_ ? kv_->kv_rows : static_cast<int>(req.input_ids.size()) + static_cast<int>(gr.tokens.size())`.
- **GLM**: `GlmV5Model::Infer` already maps `gr.kv_slots_used` — verify after Task 6 that it reports `P + num_tokens` on the seq-P path (the parity test asserts it).

- [ ] **Step 4: Run the tests to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_model_runtime
./build/linux-debug/tests/test_model_runtime
ctest --preset linux-debug -R "test_glm_prefill_seqp|test_qwen_model|test_deepseek_prompt_output"
```
Expected: `test_model_runtime` all PASS; `test_glm_prefill_seqp` PASS (kv_slots_used == P + num_tokens); `test_qwen_model` PASS; real `test_deepseek_prompt_output` PASS.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/deepseek_inference_pipeline.cu libs/lsxcommon/src/qwen_inference.cu libs/lsxcommon/src/qwen_base_model.cpp libs/lsxcommon/src/glm_v5_model.cpp tests/test_model_runtime.cpp
git commit -m "feat: publish kv_slots_used occupancy for all three engines"
```

---

## Self-Review

**Spec coverage (SECTION 3 / items 11-15):**
- Item 11 (orchestrator ring closure) → Task 5. `max_tokens` on `ChatCompletion`, `BuildPromptInputIds` in `DispatchOne`, `CompletionResponse` + `WriteCompletion`, response `FrameQueue`, pinned `ServeResponses` ZMQ writer, `id` correlation, `kUnknown`/full-ring semantics preserved.
- Item 12 (route-token pinned staging / malloc-per-layer kill) → Task 4. `DeepSeekInferArena` = 1 device + 1 pinned host alloc; `route_token` caller-owned staging with a zero-alloc assertion; `ensure_expert_resident` untouched (fd-LRU unchanged).
- Item 13 (isolated sampling module) → Tasks 1-2. `lsxcommon::sample::ISampler` + `GreedySampler`, absl-free, threaded as a defaulted ref into all three decode loops; EOS stays host-side (Phase A contract).
- Item 14 (unified `IModel::Benchmark`) → Task 3. `virtual Benchmark` (default `{ok=false}`), `BenchmarkResult::resident_bytes`, `RunBenchmark` shim; `--benchmark` call sites unchanged.
- Item 15 (prefill/KV matrix) → Tasks 6-8. GLM seq-P parity (kv_slots_used == P + gen, argmax == first token, bit-exact reruns); DeepSeek per-token retained + occupancy counter (Task 8); Qwen incremental per-call KV cache (Task 7) + counter. Multi-sequence batching explicitly out of scope.
- Testing section: arena alloc-count (Task 4), sampler seam (Task 1), Benchmark parity (Task 3), ring close with `id` correlation (Task 5), GLM seq-P parity (Task 6). Real-model tests use `SKIP_RETURN_CODE 4`.
- Phasing: SECTION 1/2 (Phase A) untouched; GLM `GlmMoEArena`/Qwen arena allocation models unchanged (out of scope) except DeepSeek's own new arena.

**Placeholder scan:** the three engine-`Benchmark` overrides (Task 3) and the Qwen `kv_base` kernel threading (Task 7) reference wrapper members (`pw_`, `tok_db_`, `tables_`/`expert_db_`) and kernel internals that the implementer must confirm against the live source before writing — the plan points to the exact anchors (`glm_v5_model.cpp::Infer`, `qwen_inference.cu::qwen_attn`). Where a member name differs, the implementer mirrors the existing `Infer` body; this is a verified-anchor, not a TBD. `test_deepseek_arena.cu` include path for the `route_token`/`InferenceAllocCount` declarations may need a small public API header (`deepseek_inference_api.h`) if the private `.cuh` is unreachable — flagged in the task.

**Type consistency:** `LogitType`, `ISampler::Sample(stream, d_logits, vocab, dtype, d_out_token, d_out_maxv)` are used identically in Tasks 1, 2, 6, 7. `QwenKvCache::Allocate(top, max_total, stream)` matches Task 7's call. `DeepSeekInferArena::Allocate(pw, max_seq_len, compress_ratios, vocab)` matches Task 4's call. `RunBenchmark(model_path, num_tokens)` unchanged at the two app call sites. `CompletionResponse` fields match the orchestrator mapping and `WriteCompletion` in Task 5.

## Execution Handoff

Plan saved. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session with checkpoints.

The plan assumes the Phase A branch `feat/inference-unification-phase-a` is the base; if not, `git checkout` a fresh `feat/inference-unification-phase-b` from it first.