# GLM Pinned-Weights Load Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Radically speed up `LoadGlmPinnedWeights` by replacing the scalar host BF16→FP16 conversion with a GPU in-place kernel, per-layer device arenas and a pinned staging ring — bit-exactly.

**Architecture:** Raw BF16 bytes are uploaded straight into the final FP16 allocation (both are 2 bytes/element) through a double-buffered pinned ring; an in-place CUDA kernel converts on device with round-to-nearest math identical to the legacy host loop. Each layer's FP16 weights live in ONE `cudaMalloc` arena carved into per-tensor views. Missing layer files / malformed tensors become hard errors instead of silent skips.

**Tech Stack:** C++20, CUDA 13.x (target `lsxcommon_glm_cu`), Arrow IPC (existing), Catch2 v3, CMake presets (`linux-debug`).

**Spec:** `docs/superpowers/specs/2026-08-22-glm-load-acceleration-design.md`

## Global Constraints

- Bit-exactness: device conversion MUST equal the host reference `uint32_t(bits) << 16 → float → __float2half` (round-to-nearest-even) bit-for-bit.
- No on-disk format changes; no changes to `PinnedSchema`, `GlmExpertDb`, inference kernels, or public loader API signatures.
- All logging via spdlog (`lsxcommon/log.h`, existing `log::` shim in glm_inference_load.cpp); no `printf`/`std::cout`.
- No comments unless required by the task (repo convention; keep existing Doxygen when refactoring).
- Build/test commands use the `linux-debug` preset: `cmake --build --preset linux-debug --target <t>` and `ctest --preset linux-debug -R <regex>`.
- CUDA test targets follow the OBJECT-library pattern (see `test_sampling` block at `tests/CMakeLists.txt:1071`) so `.cu` files never inherit lsxcommon's C++ flags.
- The new kernel lives in `glm_inference_load.cu` (already part of `lsxcommon_glm_cu`).
- Arena offsets are 256-byte aligned so uint4-vectorized kernel access is always safe.
- Windows MSVC must keep compiling: plain CUDA runtime calls only, no host intrinsics.

---

### Task 1: In-place BF16→FP16 CUDA kernel (TDD)

**Files:**
- Create: `tests/test_glm_load_fast.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference_load.h` (add launcher declaration)
- Modify: `libs/lsxcommon/src/glm_inference_load.cu` (append kernel + launcher)
- Modify: `tests/CMakeLists.txt` (new OBJECT-lib test target)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  ```cpp
  // namespace lsxcommon::glm, declared in glm_inference_load.h.
  // d_buf holds raw BF16 bits on entry and FP16 bits on exit.
  // Precondition: d_buf is 16-byte aligned (cudaMalloc and all arenas satisfy it).
  cudaError_t GlmBf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                                   cudaStream_t stream);
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_glm_load_fast.cu`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include <lsxcommon/glm_inference_load.h>

namespace {

uint16_t RefBf16ToHalf(uint16_t bits)
{
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    __half h = __float2half(f);
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

std::vector<uint16_t> TestPatterns()
{
    return {
        0x0000,  // +0
        0x8000,  // -0
        0x0001,  // min subnormal
        0x03FF,  // max subnormal
        0x0400,  // min normal
        0x3555,  // round-to-even boundary
        0x3556,  // round-to-nearest up
        0x4248,  // ordinary value
        0xC048,  // negative value
        0x7F80,  // +inf
        0xFF80,  // -inf
        0x7FC0,  // canonical quiet NaN
    };
}

std::vector<uint16_t> RandomFinite(size_t n, uint32_t seed)
{
    std::vector<uint16_t> v(n);
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        uint16_t bits = static_cast<uint16_t>(s >> 13);
        if ((bits & 0x7F80u) == 0x7F80u) bits &= 0x7F7Fu;
        v[i] = bits;
    }
    return v;
}

std::vector<uint16_t> RunOnDevice(const std::vector<uint16_t> &host_bits,
                                  cudaStream_t stream)
{
    const int64_t n = static_cast<int64_t>(host_bits.size());
    uint16_t *d = nullptr;
    REQUIRE(cudaMalloc(&d, host_bits.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpyAsync(d, host_bits.data(), host_bits.size() * 2,
                            cudaMemcpyHostToDevice, stream) == cudaSuccess);
    cudaError_t e = lsxcommon::glm::GlmBf16ToHalfInPlace(d, n, stream);
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    std::vector<uint16_t> out(host_bits.size());
    REQUIRE(cudaMemcpy(out.data(), d, host_bits.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    cudaFree(d);
    return out;
}

void CheckBitExact(const std::vector<uint16_t> &host_bits)
{
    auto ref = host_bits;
    for (auto &b : ref) b = RefBf16ToHalf(b);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    auto got = RunOnDevice(host_bits, stream);
    cudaStreamDestroy(stream);

    for (size_t i = 0; i < ref.size(); ++i)
        INFO("index " << i << " bf16=0x" << std::hex << host_bits[i]);
    for (size_t i = 0; i < ref.size(); ++i)
        REQUIRE(got[i] == ref[i]);
}

} // namespace

TEST_CASE("Bf16ToHalfInPlace bit-exact vs host reference", "[glm][loadfast]")
{
    CheckBitExact(TestPatterns());
    CheckBitExact(RandomFinite(100003, 1));
}

TEST_CASE("Bf16ToHalfInPlace ragged tails stay bit-exact", "[glm][loadfast]")
{
    for (int64_t n : {1, 7, 9, 4097}) {
        CheckBitExact(RandomFinite(static_cast<size_t>(n), 42));
    }
}
```

Add to `tests/CMakeLists.txt` right after the `test_sampling` block (after line 1116):

```cmake
# ── GLM fast loader test (in-place BF16→FP16 kernel + pinned ring + arena) ──
add_library(test_glm_load_fast_cu OBJECT test_glm_load_fast.cu)
set_target_properties(test_glm_load_fast_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_glm_load_fast_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_glm_load_fast_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_compile_definitions(test_glm_load_fast_cu PRIVATE
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_glm_load_fast_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_glm_load_fast)
set_target_properties(test_glm_load_fast PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_glm_load_fast PRIVATE
    test_glm_load_fast_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_glm_load_fast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_glm_load_fast>"
    )
    add_custom_command(TARGET test_glm_load_fast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_glm_load_fast>"
    )
endif()

add_test(NAME test_glm_load_fast COMMAND test_glm_load_fast)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
cmake --build --preset linux-debug --target test_glm_load_fast 2>&1 | tail -20
```
Expected: FAIL — `GlmBf16ToHalfInPlace` is not declared (`glm_inference_load.h`). If it links (it must not), stop and re-check.

- [ ] **Step 3: Declare the launcher**

In `libs/lsxcommon/include/lsxcommon/glm_inference_load.h`, after `constexpr float kGlmRmsEps = 1e-5f;` (line 34), add:

```cpp
// In-place device conversion of raw BF16 bits to FP16 bits. d_buf holds
// numel BF16 values on entry and their FP16 images on exit. Rounding is
// nearest-even via __float2half — bit-identical to the legacy host scalar
// path. d_buf must be 16-byte aligned.
cudaError_t GlmBf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                                 cudaStream_t stream);
```

- [ ] **Step 4: Implement kernel and launcher (minimal)**

Append to `libs/lsxcommon/src/glm_inference_load.cu` (before the closing `} // namespace glm`):

```cpp
// ── Fast-path loader kernels ────────────────────────────────────────────────
// In-place BF16→FP16 conversion, bit-exact mirror of the legacy host scalar
// path: bf16 bits<<16 reinterpreted as FP32, rounded to nearest-even FP16.
__global__ void GlmBf16ToHalfInPlaceKernel(uint16_t *__restrict__ buf,
                                           int64_t n)
{
    constexpr int64_t kVec = 8;
    const int64_t nv = n / kVec;
    const int64_t stride =
        static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
         v < nv; v += stride) {
        uint4 raw = *reinterpret_cast<const uint4 *>(buf + v * kVec);
        uint16_t w[8] = {
            static_cast<uint16_t>(raw.x & 0xFFFFu),
            static_cast<uint16_t>(raw.x >> 16),
            static_cast<uint16_t>(raw.y & 0xFFFFu),
            static_cast<uint16_t>(raw.y >> 16),
            static_cast<uint16_t>(raw.z & 0xFFFFu),
            static_cast<uint16_t>(raw.z >> 16),
            static_cast<uint16_t>(raw.w & 0xFFFFu),
            static_cast<uint16_t>(raw.w >> 16),
        };
        __half h[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            h[k] = __float2half(__uint_as_float(
                static_cast<uint32_t>(w[k]) << 16));
        uint16_t o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
            o[k] = *reinterpret_cast<uint16_t *>(&h[k]);
        uint4 ov = make_uint4(
            o[0] | (static_cast<uint32_t>(o[1]) << 16),
            o[2] | (static_cast<uint32_t>(o[3]) << 16),
            o[4] | (static_cast<uint32_t>(o[5]) << 16),
            o[6] | (static_cast<uint32_t>(o[7]) << 16));
        *reinterpret_cast<uint4 *>(buf + v * kVec) = ov;
    }
    const int64_t tail0 = nv * kVec;
    for (int64_t i = tail0 + static_cast<int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
         i < n; i += stride) {
        __half hv =
            __float2half(__uint_as_float(static_cast<uint32_t>(buf[i]) << 16));
        buf[i] = *reinterpret_cast<uint16_t *>(&hv);
    }
}

cudaError_t GlmBf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                                 cudaStream_t stream)
{
    if (!d_buf || numel <= 0) return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    const int64_t nv = numel / 8;
    int64_t blocks64 = (nv + kBlock - 1) / kBlock;
    if (blocks64 < 1) blocks64 = 1;
    if (blocks64 > 65535) blocks64 = 65535;
    GlmBf16ToHalfInPlaceKernel<<<static_cast<int>(blocks64), kBlock, 0,
                                 stream>>>(d_buf, numel);
    return cudaGetLastError();
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```bash
cmake --build --preset linux-debug --target test_glm_load_fast && ctest --preset linux-debug -R test_glm_load_fast --output-on-failure
```
Expected: PASS (2 test cases).

- [ ] **Step 6: Commit**

```bash
git add tests/test_glm_load_fast.cu tests/CMakeLists.txt \
  libs/lsxcommon/include/lsxcommon/glm_inference_load.h \
  libs/lsxcommon/src/glm_inference_load.cu
git commit -m "feat: GLM in-place BF16-to-FP16 CUDA load kernel"
```

---

### Task 2: PinnedStagingRing (TDD)

**Files:**
- Create: `libs/lsxcommon/src/glm_load_internal.h`
- Modify: `tests/test_glm_load_fast.cu` (add ring test cases)

**Interfaces:**
- Consumes: `GlmBf16ToHalfInPlace` (Task 1).
- Produces (header-only internal helper, included as `"glm_load_internal.h"` from `libs/lsxcommon/src`):
  ```cpp
  class lsxcommon::glm::PinnedStagingRing {
      explicit PinnedStagingRing(size_t chunk_bytes = kDefaultChunkBytes);
      bool ok() const;
      cudaError_t UploadBf16AsFp16(const uint8_t *host_bf16, int64_t numel,
                                   __half *d_dst, cudaStream_t stream,
                                   std::uint64_t &bytes_acc);
      cudaError_t UploadRaw(const void *src, void *d_dst, size_t bytes,
                            cudaStream_t stream, std::uint64_t &bytes_acc);
  };
  ```

- [ ] **Step 1: Write failing ring tests**

Append to `tests/test_glm_load_fast.cu`:

```cpp
#include "glm_load_internal.h"

using lsxcommon::glm::PinnedStagingRing;

namespace {

std::vector<uint16_t> RefConvertAll(const std::vector<uint16_t> &bits)
{
    std::vector<uint16_t> out(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) out[i] = RefBf16ToHalf(bits[i]);
    return out;
}

bool DeviceBitsEqual(cudaStream_t stream, const __half *d,
                     const std::vector<uint16_t> &expect)
{
    std::vector<uint16_t> got(expect.size());
    cudaMemcpyAsync(got.data(), d, expect.size() * 2,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return got == expect;
}

} // namespace

TEST_CASE("Staging ring chunked BF16 upload is bit-exact", "[glm][loadfast]")
{
    constexpr int64_t kNumel = 20001;
    constexpr size_t kChunk = 4096;
    PinnedStagingRing ring(kChunk);
    REQUIRE(ring.ok());

    auto bits = RandomFinite(static_cast<size_t>(kNumel), 7);

    __half *d = nullptr;
    REQUIRE(cudaMalloc(&d, kNumel * 2) == cudaSuccess);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    std::uint64_t acc = 0;
    REQUIRE(ring.UploadBf16AsFp16(reinterpret_cast<const uint8_t *>(bits.data()),
                                  kNumel, d, stream, acc) == cudaSuccess);
    CHECK(acc == static_cast<std::uint64_t>(kNumel) * 2);
    REQUIRE(DeviceBitsEqual(stream, d, RefConvertAll(bits)));

    cudaStreamDestroy(stream);
    cudaFree(d);
}

TEST_CASE("Staging ring buffer reuse never corrupts in-flight chunks",
          "[glm][loadfast]")
{
    constexpr int64_t kNumel = 8192 + 3;
    constexpr size_t kChunk = 1024;
    PinnedStagingRing ring(kChunk);
    REQUIRE(ring.ok());

    std::vector<std::vector<uint16_t>> tensors;
    std::vector<std::vector<uint16_t>> refs;
    std::vector<__half *> devs;
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    for (int t = 0; t < 4; ++t) {
        tensors.push_back(
            RandomFinite(static_cast<size_t>(kNumel), 100 + t));
        refs.push_back(RefConvertAll(tensors.back()));
        __half *d = nullptr;
        REQUIRE(cudaMalloc(&d, kNumel * 2) == cudaSuccess);
        devs.push_back(d);
    }

    std::uint64_t acc = 0;
    for (int t = 0; t < 4; ++t) {
        REQUIRE(ring.UploadBf16AsFp16(
                    reinterpret_cast<const uint8_t *>(tensors[t].data()),
                    kNumel, devs[t], stream, acc) == cudaSuccess);
    }
    for (int t = 0; t < 4; ++t)
        REQUIRE(DeviceBitsEqual(stream, devs[t], refs[t]));

    cudaStreamDestroy(stream);
    for (__half *d : devs) cudaFree(d);
}

TEST_CASE("Staging ring raw upload matches source", "[glm][loadfast]")
{
    constexpr size_t kBytes = 5000;
    PinnedStagingRing ring(2048);
    REQUIRE(ring.ok());

    std::vector<uint8_t> src(kBytes);
    for (size_t i = 0; i < kBytes; ++i) src[i] = static_cast<uint8_t>(i * 31);

    uint8_t *d = nullptr;
    REQUIRE(cudaMalloc(&d, kBytes) == cudaSuccess);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    std::uint64_t acc = 0;
    REQUIRE(ring.UploadRaw(src.data(), d, kBytes, stream, acc) == cudaSuccess);
    CHECK(acc == kBytes);
    REQUIRE(DeviceBitsEqual(stream,
                            reinterpret_cast<const __half *>(d), src));

    cudaStreamDestroy(stream);
    cudaFree(d);
}
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cmake --build --preset linux-debug --target test_glm_load_fast 2>&1 | tail -5
```
Expected: FAIL — `glm_load_internal.h` not found.

- [ ] **Step 3: Implement the ring**

Create `libs/lsxcommon/src/glm_load_internal.h`:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

#include <lsxcommon/glm_inference_load.h>

namespace lsxcommon {
namespace glm {

// Double-buffered pinned staging for async H2D uploads on one stream.
// Buffer parity is guarded by a recorded event: before the host refills a
// buffer, the previous copy enqueued from it has been event-synchronized, so
// an in-flight DMA read can never race the next host memcpy. Same-stream
// ordering guarantees every conversion kernel runs strictly after its copy.
class PinnedStagingRing {
public:
    static constexpr size_t kDefaultChunkBytes = size_t{8} << 20;

    explicit PinnedStagingRing(size_t chunk_bytes = kDefaultChunkBytes)
        : chunk_bytes_(chunk_bytes < (size_t{1} << 20) ? (size_t{1} << 20)
                                                       : chunk_bytes)
    {
        for (int i = 0; i < 2; ++i) {
            bufs_[i] = nullptr;
            ev_[i] = nullptr;
        }
        for (int i = 0; i < 2; ++i) {
            if (cudaMallocHost(&bufs_[i], chunk_bytes_) != cudaSuccess)
                bufs_[i] = nullptr;
        }
        for (int i = 0; i < 2; ++i)
            cudaEventCreateWithFlags(&ev_[i], cudaEventDisableTiming);
    }

    ~PinnedStagingRing()
    {
        for (int i = 0; i < 2; ++i) {
            if (bufs_[i]) cudaFreeHost(bufs_[i]);
            if (ev_[i]) cudaEventDestroy(ev_[i]);
        }
    }

    PinnedStagingRing(const PinnedStagingRing &) = delete;
    PinnedStagingRing &operator=(const PinnedStagingRing &) = delete;

    bool ok() const { return bufs_[0] && bufs_[1] && ev_[0] && ev_[1]; }

    cudaError_t UploadBf16AsFp16(const uint8_t *host_bf16, int64_t numel,
                                 __half *d_dst, cudaStream_t stream,
                                 std::uint64_t &bytes_acc)
    {
        if (!ok()) return cudaErrorMemoryAllocation;
        const int64_t per_chunk =
            static_cast<int64_t>(chunk_bytes_ / 2);
        if (per_chunk <= 0) return cudaErrorInvalidValue;
        int bi = 0;
        for (int64_t off = 0; off < numel; ) {
            const int64_t cnt =
                (numel - off < per_chunk) ? (numel - off) : per_chunk;
            cudaError_t e = cudaEventSynchronize(ev_[bi]);
            if (e != cudaSuccess) return e;
            std::memcpy(bufs_[bi], host_bf16 + off * 2,
                        static_cast<size_t>(cnt) * 2);
            uint16_t *dst16 = reinterpret_cast<uint16_t *>(d_dst) + off;
            e = cudaMemcpyAsync(dst16, bufs_[bi],
                                static_cast<size_t>(cnt) * 2,
                                cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
            e = cudaEventRecord(ev_[bi], stream);
            if (e != cudaSuccess) return e;
            e = GlmBf16ToHalfInPlace(dst16, cnt, stream);
            if (e != cudaSuccess) return e;
            off += cnt;
            bi ^= 1;
        }
        bytes_acc += static_cast<std::uint64_t>(numel) * 2;
        return cudaSuccess;
    }

    cudaError_t UploadRaw(const void *src, void *d_dst, size_t bytes,
                          cudaStream_t stream, std::uint64_t &bytes_acc)
    {
        if (!ok()) return cudaErrorMemoryAllocation;
        int bi = 0;
        size_t off = 0;
        while (off < bytes) {
            const size_t cnt =
                (bytes - off < chunk_bytes_) ? (bytes - off) : chunk_bytes_;
            cudaError_t e = cudaEventSynchronize(ev_[bi]);
            if (e != cudaSuccess) return e;
            std::memcpy(bufs_[bi], static_cast<const uint8_t *>(src) + off,
                        cnt);
            e = cudaMemcpyAsync(static_cast<uint8_t *>(d_dst) + off,
                                bufs_[bi], cnt, cudaMemcpyHostToDevice,
                                stream);
            if (e != cudaSuccess) return e;
            e = cudaEventRecord(ev_[bi], stream);
            if (e != cudaSuccess) return e;
            off += cnt;
            bi ^= 1;
        }
        bytes_acc += bytes;
        return cudaSuccess;
    }

private:
    size_t chunk_bytes_;
    uint8_t *bufs_[2];
    cudaEvent_t ev_[2];
};

} // namespace glm
} // namespace lsxcommon
```

Note: `cudaEventSynchronize` on a freshly created (never-recorded) event returns immediately — first use needs no guard.

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
cmake --build --preset linux-debug --target test_glm_load_fast && ctest --preset linux-debug -R test_glm_load_fast --output-on-failure
```
Expected: PASS (5 test cases total).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/glm_load_internal.h tests/test_glm_load_fast.cu
git commit -m "feat: double-buffered pinned staging ring for GLM loader"
```

---

### Task 3: Full 79-layer synthetic fixture + loader test update

The strict-error semantics (Task 4) require every layer file to exist, but
`BuildSyntheticModel()` only emits layers 0, 3, 6, 78. Extend the fixture to
all 79 layers and update the loader test that asserted skip semantics.

**Files:**
- Modify: `tests/glm_test_fixture.h` (`BuildSyntheticModel`, lines 128–225)
- Modify: `tests/test_glm_loader.cpp` ("sizes layers from topology" TEST_CASE)

**Interfaces:**
- Consumes: nothing new.
- Produces: `glmtest::BuildSyntheticModel()` returning tensors for ALL layers
  0–78: dense 0–2 (attention+norm+MLP), MoE 3–78 (attention+norm+router+bias,
  indexer when `l % 4 == 0`), MTP extras on layer 78. Geometry unchanged
  (hidden 16, vocab 64).

- [ ] **Step 1: Check all fixture consumers**

Run: `rg -l BuildSyntheticModel tests/`
Expected consumers: `test_glm_loader.cpp`. Re-check each hit before editing;
if any other file asserts missing-layer null pointers, apply the same rewrite
pattern as Step 3 there.

- [ ] **Step 2: Rewrite the fixture generator**

Replace `BuildSyntheticModel()` in `tests/glm_test_fixture.h` with:

```cpp
inline void AppendAttention(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("model.layers.", l, ".");
    t.push_back({p + "input_layernorm.weight",          "BF16", {16}, MakeBf16(16)});
    t.push_back({p + "post_attention_layernorm.weight", "BF16", {16}, MakeBf16(16)});
    t.push_back({p + "self_attn.q_a_proj.weight",            "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.q_a_layernorm.weight",       "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "self_attn.q_b_proj.weight",            "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.kv_a_proj_with_mqa.weight",  "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.kv_a_layernorm.weight",      "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "self_attn.kv_b_proj.weight",           "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "self_attn.o_proj.weight",              "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendMoeFfn(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("model.layers.", l, ".mlp.");
    t.push_back({p + "gate.weight",                "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "gate.e_score_correction_bias", "F32", {16},   MakeF32(16)});
    for (int e = 0; e < 2; ++e) {
        auto ep = absl::StrCat(p, "experts.", e, ".");
        t.push_back({ep + "gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({ep + "down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
        t.push_back({ep + "up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
    }
    t.push_back({p + "shared_experts.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "shared_experts.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "shared_experts.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendDenseMlp(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("model.layers.", l, ".");
    t.push_back({p + "mlp.gate_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "mlp.down_proj.weight", "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "mlp.up_proj.weight",   "BF16", {16, 16}, MakeBf16(256)});
}

inline void AppendIndexer(std::vector<FakeTensor> &t, int l)
{
    auto p = absl::StrCat("model.layers.", l, ".self_attn.indexer.");
    t.push_back({p + "wq_b.weight",          "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "wk.weight",            "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "weights_proj.weight",  "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({p + "k_norm.weight",        "BF16", {16},    MakeBf16(16)});
    t.push_back({p + "k_norm.bias",          "F32",  {16},    MakeF32(16)});
}

inline std::vector<FakeTensor> BuildSyntheticModel()
{
    std::vector<FakeTensor> t;
    t.push_back({"model.embed_tokens.weight", "BF16", {64, 16}, MakeBf16(64 * 16)});
    t.push_back({"lm_head.weight",            "BF16", {64, 16}, MakeBf16(64 * 16)});
    t.push_back({"model.norm.weight",         "BF16", {16},     MakeBf16(16)});

    for (int l = 0; l <= 78; ++l) {
        AppendAttention(t, l);
        if (l < 3) {
            AppendDenseMlp(t, l);
        } else {
            AppendMoeFfn(t, l);
            if (l % 4 == 0) AppendIndexer(t, l);
        }
    }

    t.push_back({"model.layers.78.enorm.weight",             "BF16", {16},    MakeBf16(16)});
    t.push_back({"model.layers.78.hnorm.weight",             "BF16", {16},    MakeBf16(16)});
    t.push_back({"model.layers.78.eh_proj.weight",           "BF16", {16, 16}, MakeBf16(256)});
    t.push_back({"model.layers.78.shared_head.norm.weight",  "BF16", {16},    MakeBf16(16)});

    return t;
}
```

Keep the comment block above it updated to describe the full grid.

- [ ] **Step 3: Update the loader topology test**

In `tests/test_glm_loader.cpp`, replace the body assertions of
TEST_CASE `"GLM loader sizes layers from topology and skips missing files"`
(rename it to `"GLM loader sizes layers from topology"`) after the
`REQUIRE(err == cudaSuccess)` block:

```cpp
    // Topology-derived sizing: all 79 layer files present and loaded.
    CHECK(pw.layers.size() == 79);
    CHECK(pw.vocab_size == 64);
    CHECK(pw.hidden == 16);

    // Dense layer 0: dense MLP present, no router, no indexer.
    auto &l0 = pw.layers[0];
    CHECK(l0.d_mlp_gate != nullptr);
    CHECK(l0.d_mlp_down != nullptr);
    CHECK(l0.d_mlp_up != nullptr);
    CHECK(l0.d_router_gate == nullptr);
    CHECK(l0.d_idx_wq_b == nullptr);
    CHECK(l0.d_q_a_proj != nullptr);
    CHECK(l0.d_q_b_proj != nullptr);
    CHECK(l0.d_o_proj != nullptr);
    CHECK(l0.d_input_norm != nullptr);
    CHECK(l0.d_post_attn_norm != nullptr);

    // Dense layer 2 (was previously absent from the fixture).
    CHECK(pw.layers[2].d_mlp_gate != nullptr);
    CHECK(pw.layers[2].d_router_gate == nullptr);

    // MoE layer 3: router gate + F32 bias, no dense MLP.
    auto &l3 = pw.layers[3];
    CHECK(l3.d_router_gate != nullptr);
    REQUIRE(l3.h_gate_bias != nullptr);
    CHECK(l3.gate_bias_n == 16);
    CHECK(l3.d_mlp_gate == nullptr);
    CHECK(l3.d_q_a_proj != nullptr);

    // Indexer layer 4 (l % 4 == 0): indexer weights + F32 k_norm bias.
    auto &l4 = pw.layers[4];
    CHECK(l4.d_idx_wq_b != nullptr);
    CHECK(l4.d_idx_wk != nullptr);
    CHECK(l4.d_idx_weights_proj != nullptr);
    CHECK(l4.d_idx_k_norm != nullptr);
    REQUIRE(l4.d_idx_k_norm_bias != nullptr);
    CHECK(l4.d_idx_k_norm_bias == reinterpret_cast<const float *>(
                                      l4.d_idx_k_norm_bias)); // non-null sanity
    CHECK(l4.d_router_gate != nullptr);   // indexer layers are MoE too

    // Non-indexer MoE layer 5: no indexer weights.
    CHECK(pw.layers[5].d_idx_wq_b == nullptr);

    // MTP layer 78: MTP extras present.
    auto &l78 = pw.layers[78];
    CHECK(l78.d_eh_proj != nullptr);
    CHECK(l78.d_enorm != nullptr);
    CHECK(l78.d_hnorm != nullptr);
    CHECK(l78.d_shared_head_norm != nullptr);
    CHECK(l78.d_router_gate != nullptr);
```

Delete the self-referential sanity line `CHECK(l4.d_idx_k_norm_bias == ...)` if
it reads awkwardly — the `REQUIRE(... != nullptr)` above already covers it.

- [ ] **Step 4: Run the loader suite**

Run:
```bash
cmake --build --preset linux-debug --target test_glm_loader && ctest --preset linux-debug -R test_glm_loader --output-on-failure
```
Expected: PASS (3 test cases; the two value-based tests are unaffected except
that they now also exercise more layers).

- [ ] **Step 5: Commit**

```bash
git add tests/glm_test_fixture.h tests/test_glm_loader.cpp
git commit -m "test: full 79-layer synthetic GLM fixture"
```

---

### Task 4: Strict error semantics (TDD)

Missing layer file → error. Tensor found by suffix but malformed (bad shape
rank, `numel <= 0`) → error. Required-for-type tensor absent → error
(`input_layernorm` + `post_attention_layernorm` for every layer; dense `<3`:
three MLP matrices; MoE `>=3`: router gate). Absence of optional tensors
(indexer, MTP extras) stays legitimate.

**Files:**
- Modify: `tests/test_glm_loader.cpp` (two new TEST_CASEs)
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp` (loader loop + helpers)

**Interfaces:**
- Consumes: existing `LoadSingleLayer`, `Upload1D`, `Upload2D`,
  `FindTensor`, `LoadArrowFile` (all currently in the anonymous namespace).
- Produces: `LoadSingleLayer` returns `cudaError_t` instead of `void`;
  `LoadGlmPinnedWeights` returns `cudaErrorInvalidValue` on any strict-error
  condition with an error log naming the offending path/tensor.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_glm_loader.cpp`:

```cpp
TEST_CASE("GLM loader errors on missing layer file", "[glm][loader]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_loader_missing";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    glmtest::WriteTokenizer(in);
    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    auto victim = out / "logestix" / "pinned" / "layer_5.arrow";
    REQUIRE(fs::exists(victim));
    fs::remove(victim);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GlmPinnedWeights pw;
    auto err = lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream);
    CHECK(err == cudaErrorInvalidValue);
    CHECK(pw.layers.empty());   // freed cleanly on the error path

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("GLM loader errors on malformed tensor shape", "[glm][loader]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_loader_badshape";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    glmtest::WriteTokenizer(in);
    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    // Overwrite layer_5.arrow with a batch whose input_layernorm has an
    // empty shape (found-by-suffix but malformed).
    std::vector<lsxcommon::TensorRecord> bad;
    bad.push_back({"input_layernorm.weight", "BF16", {}, {}});
    bad.push_back({"post_attention_layernorm.weight", "BF16", {16},
                   std::vector<uint8_t>(16 * 2)});
    auto batch_res = lsxcommon::PinnedSchema::MakeBatch(bad);
    REQUIRE(batch_res.ok());
    auto wres = lsxcommon::WriteRecordBatchToFile(
        *batch_res, out / "logestix" / "pinned" / "layer_5.arrow");
    REQUIRE(wres.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    GlmPinnedWeights pw;
    auto err = lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream);
    CHECK(err == cudaErrorInvalidValue);

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cmake --build --preset linux-debug --target test_glm_loader && ctest --preset linux-debug -R test_glm_loader --output-on-failure
```
Expected: FAIL — current loader silently skips the missing file / null-shape
tensor and returns `cudaSuccess`.

- [ ] **Step 3: Implement strict semantics**

In `libs/lsxcommon/src/glm_inference_load.cpp`:

3a. Change signatures to propagate errors:

```cpp
cudaError_t Upload1DStrict(const TensorRecord *t, __half **dst,
                           cudaStream_t stream, PinnedStagingRing &ring,
                           std::uint64_t &bytes_acc);
cudaError_t Upload2DStrict(const std::vector<TensorRecord> &tensors,
                           const std::string &suffix, __half **dst,
                           int64_t *rows, int64_t *cols,
                           cudaStream_t stream, PinnedStagingRing &ring,
                           std::uint64_t &bytes_acc);
```

Behavior (this Task keeps the legacy scalar conversion inside them — the
rewiring to the ring happens in Task 5; declare them taking the ring parameter
now but ignore it until Task 5 to avoid churn):

```cpp
// Found-but-malformed => error; absent => ok (optional tensor).
cudaError_t Upload1DStrict(const TensorRecord *t, __half **dst,
                           cudaStream_t stream, PinnedStagingRing &,
                           std::uint64_t &bytes_acc)
{
    *dst = nullptr;
    if (!t) return cudaSuccess;
    if (t->shape.size() != 1 || t->shape[0] <= 0 ||
        t->data.size() < static_cast<size_t>(t->shape[0]) * 2) {
        log::error("[glm_load] malformed 1D tensor '" + t->name + "'");
        return cudaErrorInvalidValue;
    }
    *dst = nullptr;
    cudaError_t e = UploadBf16ToHalf(t->data.data(), t->shape[0], dst,
                                     stream, bytes_acc);
    if (e != cudaSuccess)
        log::error("[glm_load] upload failed for '" + t->name + "': " +
                   cudaGetErrorString(e));
    return e;
}

cudaError_t Upload2DStrict(const std::vector<TensorRecord> &tensors,
                           const std::string &suffix, __half **dst,
                           int64_t *rows, int64_t *cols,
                           cudaStream_t stream, PinnedStagingRing &,
                           std::uint64_t &bytes_acc)
{
    *dst = nullptr;
    auto *t = FindTensor(tensors, suffix);
    if (!t) return cudaSuccess;
    if (t->shape.size() != 2 || t->shape[0] <= 0 || t->shape[1] <= 0 ||
        t->data.size() <
            static_cast<size_t>(t->shape[0]) * static_cast<size_t>(t->shape[1]) * 2) {
        log::error("[glm_load] malformed 2D tensor '" + suffix + "'");
        return cudaErrorInvalidValue;
    }
    if (rows) *rows = t->shape[0];
    if (cols) *cols = t->shape[1];
    cudaError_t e = UploadBf16ToHalf(t->data.data(),
                                     t->shape[0] * t->shape[1], dst,
                                     stream, bytes_acc);
    if (e != cudaSuccess)
        log::error("[glm_load] upload failed for '" + suffix + "': " +
                   cudaGetErrorString(e));
    return e;
}
```

Include the ring header at the top of the file:

```cpp
#include "glm_load_internal.h"
```

3b. Rewrite `LoadSingleLayer` to build a required-tensor plan and use the
strict helpers. Signature change:

```cpp
cudaError_t LoadSingleLayer(const std::vector<TensorRecord> &tensors,
                            GlmLayerWeights &lw, cudaStream_t stream,
                            std::uint64_t &bytes_acc);
```

Implementation (keeping the legacy per-tensor uploads this task):

```cpp
cudaError_t LoadSingleLayer(const std::vector<TensorRecord> &tensors,
                            GlmLayerWeights &lw, cudaStream_t stream,
                            std::uint64_t &bytes_acc)
{
    PinnedStagingRing unused_ring;  // wired into uploads in the perf task

    // Required-for-type validation (heterogeneous-grid rules):
    auto *in_norm = FindTensor(tensors, "input_layernorm.weight");
    auto *post_norm = FindTensor(tensors, "post_attention_layernorm.weight");
    if (!in_norm || !post_norm) {
        log::error("[glm_load] layer missing layernorm tensors");
        return cudaErrorInvalidValue;
    }
    const bool moe = FindTensor(tensors, "mlp.gate.weight") != nullptr;
    if (!moe && !FindTensor(tensors, "mlp.gate_proj.weight")) {
        log::error("[glm_load] dense layer missing mlp.gate_proj");
        return cudaErrorInvalidValue;
    }

    cudaError_t e = Upload1DStrict(in_norm, &lw.d_input_norm, stream,
                                   unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload1DStrict(post_norm, &lw.d_post_attn_norm, stream,
                       unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;

    e = Upload2DStrict(tensors, "self_attn.q_a_proj.weight", &lw.d_q_a_proj,
                       &lw.q_a_rows, &lw.q_a_cols, stream, unused_ring,
                       bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload1DStrict(FindTensor(tensors, "self_attn.q_a_layernorm.weight"),
                       &lw.d_q_a_norm, stream, unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload2DStrict(tensors, "self_attn.q_b_proj.weight", &lw.d_q_b_proj,
                       &lw.q_b_rows, &lw.q_b_cols, stream, unused_ring,
                       bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload2DStrict(tensors, "self_attn.kv_a_proj_with_mqa.weight",
                       &lw.d_kv_a_proj, &lw.kv_a_rows, &lw.kv_a_cols,
                       stream, unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload1DStrict(FindTensor(tensors, "self_attn.kv_a_layernorm.weight"),
                       &lw.d_kv_a_norm, stream, unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload2DStrict(tensors, "self_attn.kv_b_proj.weight", &lw.d_kv_b_proj,
                       &lw.kv_b_rows, &lw.kv_b_cols, stream, unused_ring,
                       bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload2DStrict(tensors, "self_attn.o_proj.weight", &lw.d_o_proj,
                       &lw.o_rows, &lw.o_cols, stream, unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;

    auto *router = FindTensor(tensors, "mlp.gate.weight");
    if (router) {
        if (router->shape.size() != 2 || router->shape[0] <= 0 ||
            router->shape[1] <= 0 ||
            router->data.size() <
                static_cast<size_t>(router->shape[0]) *
                    static_cast<size_t>(router->shape[1]) * 2) {
            log::error("[glm_load] malformed router gate");
            return cudaErrorInvalidValue;
        }
        if (UploadBf16ToHalf(router->data.data(),
                             router->shape[0] * router->shape[1],
                             &lw.d_router_gate, stream, bytes_acc)
            != cudaSuccess)
            return cudaErrorInvalidValue;
        auto *gate_bias = FindTensor(tensors, "mlp.gate.e_score_correction_bias");
        if (gate_bias && gate_bias->shape.size() == 1 &&
            gate_bias->data.size() >=
                static_cast<size_t>(gate_bias->shape[0]) * 4) {
            int n = static_cast<int>(gate_bias->shape[0]);
            if (cudaMallocHost(&lw.h_gate_bias,
                               static_cast<size_t>(n) * sizeof(float))
                == cudaSuccess) {
                std::memcpy(lw.h_gate_bias, gate_bias->data.data(),
                            static_cast<size_t>(n) * sizeof(float));
                lw.gate_bias_n = n;
            }
        }
    } else if (FindTensor(tensors, "mlp.down_proj.weight") &&
               FindTensor(tensors, "mlp.up_proj.weight")) {
        // Dense layer 0-2.
        e = Upload2DStrict(tensors, "mlp.gate_proj.weight", &lw.d_mlp_gate,
                           nullptr, nullptr, stream, unused_ring, bytes_acc);
        if (e != cudaSuccess) return e;
        e = Upload2DStrict(tensors, "mlp.down_proj.weight", &lw.d_mlp_down,
                           nullptr, nullptr, stream, unused_ring, bytes_acc);
        if (e != cudaSuccess) return e;
        e = Upload2DStrict(tensors, "mlp.up_proj.weight", &lw.d_mlp_up,
                           nullptr, nullptr, stream, unused_ring, bytes_acc);
        if (e != cudaSuccess) return e;
    }

    e = Upload2DStrict(tensors, "self_attn.indexer.wq_b.weight",
                       &lw.d_idx_wq_b, nullptr, nullptr, stream, unused_ring,
                       bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload2DStrict(tensors, "self_attn.indexer.wk.weight", &lw.d_idx_wk,
                       nullptr, nullptr, stream, unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload2DStrict(tensors, "self_attn.indexer.weights_proj.weight",
                       &lw.d_idx_weights_proj, nullptr, nullptr, stream,
                       unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload1DStrict(FindTensor(tensors, "self_attn.indexer.k_norm.weight"),
                       &lw.d_idx_k_norm, stream, unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    auto *k_bias = FindTensor(tensors, "self_attn.indexer.k_norm.bias");
    if (k_bias && !k_bias->data.empty())
        UploadRaw(k_bias->data, &lw.d_idx_k_norm_bias, stream, bytes_acc);

    {
        int64_t eh_rows = 0, eh_cols = 0;
        e = Upload2DStrict(tensors, "eh_proj.weight", &lw.d_eh_proj,
                           &eh_rows, &eh_cols, stream, unused_ring, bytes_acc);
        if (e != cudaSuccess) return e;
    }
    e = Upload1DStrict(FindTensor(tensors, "enorm.weight"), &lw.d_enorm,
                       stream, unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload1DStrict(FindTensor(tensors, "hnorm.weight"), &lw.d_hnorm,
                       stream, unused_ring, bytes_acc);
    if (e != cudaSuccess) return e;
    e = Upload1DStrict(FindTensor(tensors, "shared_head.norm.weight"),
                       &lw.d_shared_head_norm, stream, unused_ring, bytes_acc);
    return e;
}
```

3c. Strict missing-file check + error propagation in the layer loop of
`LoadGlmPinnedWeights`:

```cpp
    for (int l = 0; l < static_cast<int>(pw.layers.size()); ++l) {
        if (layer_filter &&
            std::find(layer_filter->begin(), layer_filter->end(), l) ==
                layer_filter->end())
            continue;
        auto fn = absl::StrCat("layer_", l, ".arrow");
        auto layer_path = pinned_dir / fn;
        if (!fs::exists(layer_path)) {
            log::error("[glm_load] required layer file missing: " + fn);
            pw.Free();
            return cudaErrorInvalidValue;
        }
        auto tensors_res = LoadArrowFile(layer_path.string());
        if (!tensors_res.ok()) {
            log::error("[glm_load] ERROR loading " + fn + ": " +
                       tensors_res.status().ToString());
            pw.Free();
            return cudaErrorInvalidValue;
        }
        cudaError_t lerr =
            LoadSingleLayer(*tensors_res, pw.layers[static_cast<size_t>(l)],
                            stream, pw.resident_bytes);
        if (lerr != cudaSuccess) {
            pw.Free();
            return lerr;
        }
    }
```

- [ ] **Step 4: Run the loader suite**

Run:
```bash
ctest --preset linux-debug -R "test_glm_loader|test_glm_load_fast" --output-on-failure
```
Expected: PASS — both new error tests green; existing loader tests still pass
(the full fixture satisfies every required-for-type rule).

- [ ] **Step 5: Commit**

```bash
git add tests/test_glm_loader.cpp libs/lsxcommon/src/glm_inference_load.cpp
git commit -m "feat: strict GLM pinned-loader error semantics"
```

---

### Task 5: Arena + ring rewiring (the actual acceleration)

Replace the legacy scalar path inside the strict helpers with arena-backed
uploads through `PinnedStagingRing`. Public API and behavior (bit-for-bit
values) unchanged.

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference_load.h`
  (`GlmLayerWeights::d_arena`)
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp` (helpers + global loader)
- Modify: `tests/test_glm_load_fast.cu` (arena/idempotency tests)

**Interfaces:**
- Consumes: `PinnedStagingRing`, `GlmBf16ToHalfInPlace` (Tasks 1–2).
- Produces:
  ```cpp
  struct GlmLayerWeights {
      ...
      void *d_arena = nullptr;   // non-null => device pointers are views
  };
  ```
  Semantics: when `d_arena != nullptr`, `Free()` frees ONLY `h_gate_bias`
  (pinned host) and `d_arena`; all `d_*` device pointers become invalid
  without individual `cudaFree`. Legacy branch preserved for hand-built
  structs.

- [ ] **Step 1: Add the field**

In `glm_inference_load.h`, at the end of `struct GlmLayerWeights` (before
`void Free();`):

```cpp
    // Non-null when the layer was uploaded arena-backed: every d_* pointer
    // above is a view into this single allocation and Free() releases the
    // arena instead of the individual pointers.
    void *d_arena = nullptr;
```

- [ ] **Step 2: Write failing arena/idempotency tests**

Append to `tests/test_glm_load_fast.cu`:

```cpp
#include <catch2/catch_approx.hpp>
#include <lsxcommon/glm_convert.h>
#include "glm_test_fixture.h"

#include <filesystem>

namespace {

bool InRange(const void *p, const void *base, size_t bytes)
{
    return p >= base &&
           static_cast<const uint8_t *>(p) <
               static_cast<const uint8_t *>(base) + bytes;
}

} // namespace

TEST_CASE("Arena-backed GLM load: containment, idempotency, values",
          "[glm][loadfast]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_glm_loadfast_arena";
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);

    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    glmtest::WriteTokenizer(in);
    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::glm::GlmPinnedWeights pw;
    REQUIRE(lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(pw.layers.size() == 79);

    // Every layer is arena-backed; its device pointers fall inside the arena.
    for (int l : {0, 3, 4, 78}) {
        auto &lw = pw.layers[static_cast<size_t>(l)];
        REQUIRE(lw.d_arena != nullptr);
        if (lw.d_input_norm)
            CHECK(InRange(lw.d_input_norm, lw.d_arena, lw.arena_bytes));
        if (lw.d_o_proj)
            CHECK(InRange(lw.d_o_proj, lw.d_arena, lw.arena_bytes));
        if (lw.d_router_gate)
            CHECK(InRange(lw.d_router_gate, lw.d_arena, lw.arena_bytes));
    }
    CHECK(pw.resident_bytes > 0);

    // Spot-check converted values against the host reference (layer 3 norm).
    {
        auto &lw = pw.layers[3];
        REQUIRE(lw.d_input_norm != nullptr);
        __half h;
        cudaMemcpyAsync(&h, lw.d_input_norm, sizeof(__half),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        uint16_t got;
        std::memcpy(&got, &h, 2);
        CHECK(got == RefBf16ToHalf(glmtest::LastLayer3NormFirstBits()));
    }

    // Idempotent reload: Free-on-entry, then identical success.
    REQUIRE(lsxcommon::glm::LoadGlmPinnedWeights(out.string(), pw, stream) ==
            cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(pw.layers.size() == 79);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}
```

For `RefBf16ToHalf(glmtest::LastLayer3NormFirstBits())`: add to
`glm_test_fixture.h` a helper exposing the first raw BF16 bits that
`MakeBf16(16)` writes for layer 3's norm. Read `MakeBf16`'s definition first
(`rg -n "MakeBf16\b" tests/glm_test_fixture.h`) and mirror its deterministic
fill:

```cpp
inline uint16_t LastLayer3NormFirstBits()
{
    // Must equal MakeBf16(16)[0] as written for model.layers.3.input_layernorm.
    return MakeBf16FirstBits(/*count=*/16);
}
```

and refactor `MakeBf16` so `MakeBf16FirstBits(count)` returns element 0 of
what `MakeBf16(count)` would write (keep both consistent — single source of
truth).

Also add `size_t arena_bytes = 0;` next to `d_arena` in `GlmLayerWeights`
(the test reads it; the loader sets it to the allocation size).

- [ ] **Step 3: Run to verify failure**

Run:
```bash
cmake --build --preset linux-debug --target test_glm_load_fast && ctest --preset linux-debug -R test_glm_load_fast --output-on-failure
```
Expected: FAIL — `d_arena` is null after load (containment checks fail).

- [ ] **Step 4: Rewire the loader onto ring + arena**

In `libs/lsxcommon/src/glm_inference_load.cpp`:

4a. Delete the old `UploadBf16ToHalf`, `UploadRaw`, `Upload1D`, `Upload2D`
scalar implementations and replace the `*Strict` helpers with ring-based
uploads (signatures already take `PinnedStagingRing &`):

The single upload core is a job-list: validate every job first, then ONE
arena allocation, then per-job view assignment + ring upload (`dst` must be
assigned from the arena BEFORE uploading):

```cpp
struct Fp16Job {
    const TensorRecord *t;
    __half **dst;
    int64_t *rows;   // optional row capture (2D)
    int64_t *cols;   // optional col capture (2D)
};

cudaError_t ValidateJob(const Fp16Job &j, bool want2d)
{
    const auto &s = j.t->shape;
    const size_t el2 = want2d ? 2 : 1;
    const bool rank_ok = want2d ? (s.size() == 2 && s[0] > 0 && s[1] > 0)
                                : (s.size() == 1 && s[0] > 0);
    size_t numel = 0;
    if (rank_ok) {
        numel = want2d ? static_cast<size_t>(s[0]) * static_cast<size_t>(s[1])
                       : static_cast<size_t>(s[0]);
    }
    if (!rank_ok || j.t->data.size() < numel * 2) {
        log::error("[glm_load] malformed tensor '" + j.t->name + "'");
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t UploadJobs(std::vector<Fp16Job> jobs, GlmLayerWeights &lw,
                       cudaStream_t stream, PinnedStagingRing &ring,
                       std::uint64_t &bytes_acc)
{
    for (const auto &j : jobs)
        if (j.t) {
            cudaError_t e = ValidateJob(j, j.rows != nullptr);
            if (e != cudaSuccess) return e;
        }

    size_t total = 0;
    for (const auto &j : jobs) {
        if (!j.t) continue;
        const auto &s = j.t->shape;
        size_t numel = j.rows ? static_cast<size_t>(s[0]) *
                                    static_cast<size_t>(s[1])
                              : static_cast<size_t>(s[0]);
        total = AlignUp(total, kArenaAlign) + numel * 2;
    }

    void *arena = nullptr;
    if (total > 0) {
        cudaError_t e = cudaMalloc(&arena, total);
        if (e != cudaSuccess) {
            log::error("[glm_load] arena alloc failed: " +
                       std::string(cudaGetErrorString(e)));
            return e;
        }
    }
    lw.d_arena = arena;
    lw.arena_bytes = total;

    size_t off = 0;
    for (const auto &j : jobs) {
        if (!j.t) continue;
        const auto &s = j.t->shape;
        const size_t numel = j.rows
            ? static_cast<size_t>(s[0]) * static_cast<size_t>(s[1])
            : static_cast<size_t>(s[0]);
        off = AlignUp(off, kArenaAlign);
        __half *view = static_cast<__half *>(arena) + off / 2;
        *j.dst = view;
        if (j.rows) { *j.rows = s[0]; *j.cols = s[1]; }
        cudaError_t e = ring.UploadBf16AsFp16(j.t->data.data(),
                                              static_cast<int64_t>(numel),
                                              view, stream, bytes_acc);
        if (e != cudaSuccess) return e;
        off += numel * 2;
    }
    return cudaSuccess;
}

} // namespace
```

4b. Rewrite `LoadSingleLayer` to assemble the job list (validation rules from
Task 4 stay identical — required-for-type errors are checked before any
allocation) and call `UploadJobs` once:

```cpp
cudaError_t LoadSingleLayer(const std::vector<TensorRecord> &tensors,
                            GlmLayerWeights &lw, cudaStream_t stream,
                            std::uint64_t &bytes_acc)
{
    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    auto *in_norm = FindTensor(tensors, "input_layernorm.weight");
    auto *post_norm = FindTensor(tensors, "post_attention_layernorm.weight");
    if (!in_norm || !post_norm) {
        log::error("[glm_load] layer missing layernorm tensors");
        return cudaErrorInvalidValue;
    }
    const bool moe = FindTensor(tensors, "mlp.gate.weight") != nullptr;
    if (!moe && !FindTensor(tensors, "mlp.gate_proj.weight")) {
        log::error("[glm_load] dense layer missing mlp.gate_proj");
        return cudaErrorInvalidValue;
    }

    std::vector<Fp16Job> jobs = {
        {in_norm,  &lw.d_input_norm,     nullptr, nullptr},
        {post_norm,&lw.d_post_attn_norm, nullptr, nullptr},
        {FindTensor(tensors, "self_attn.q_a_proj.weight"),
                   &lw.d_q_a_proj,       &lw.q_a_rows, &lw.q_a_cols},
        {FindTensor(tensors, "self_attn.q_a_layernorm.weight"),
                   &lw.d_q_a_norm,       nullptr, nullptr},
        {FindTensor(tensors, "self_attn.q_b_proj.weight"),
                   &lw.d_q_b_proj,       &lw.q_b_rows, &lw.q_b_cols},
        {FindTensor(tensors, "self_attn.kv_a_proj_with_mqa.weight"),
                   &lw.d_kv_a_proj,      &lw.kv_a_rows, &lw.kv_a_cols},
        {FindTensor(tensors, "self_attn.kv_a_layernorm.weight"),
                   &lw.d_kv_a_norm,      nullptr, nullptr},
        {FindTensor(tensors, "self_attn.kv_b_proj.weight"),
                   &lw.d_kv_b_proj,      &lw.kv_b_rows, &lw.kv_b_cols},
        {FindTensor(tensors, "self_attn.o_proj.weight"),
                   &lw.d_o_proj,         &lw.o_rows, &lw.o_cols},
    };

    if (moe) {
        jobs.push_back({FindTensor(tensors, "mlp.gate.weight"),
                        &lw.d_router_gate, nullptr, nullptr});
    } else {
        jobs.push_back({FindTensor(tensors, "mlp.gate_proj.weight"),
                        &lw.d_mlp_gate, nullptr, nullptr});
        jobs.push_back({FindTensor(tensors, "mlp.down_proj.weight"),
                        &lw.d_mlp_down, nullptr, nullptr});
        jobs.push_back({FindTensor(tensors, "mlp.up_proj.weight"),
                        &lw.d_mlp_up, nullptr, nullptr});
    }

    jobs.push_back({FindTensor(tensors, "self_attn.indexer.wq_b.weight"),
                    &lw.d_idx_wq_b, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "self_attn.indexer.wk.weight"),
                    &lw.d_idx_wk, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "self_attn.indexer.weights_proj.weight"),
                    &lw.d_idx_weights_proj, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "self_attn.indexer.k_norm.weight"),
                    &lw.d_idx_k_norm, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "eh_proj.weight"),
                    &lw.d_eh_proj, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "enorm.weight"),
                    &lw.d_enorm, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "hnorm.weight"),
                    &lw.d_hnorm, nullptr, nullptr});
    jobs.push_back({FindTensor(tensors, "shared_head.norm.weight"),
                    &lw.d_shared_head_norm, nullptr, nullptr});

    cudaError_t e = UploadJobs(std::move(jobs), lw, stream, ring, bytes_acc);
    if (e != cudaSuccess) return e;

    auto *router = FindTensor(tensors, "mlp.gate.weight");
    if (router) {
        auto *gate_bias =
            FindTensor(tensors, "mlp.gate.e_score_correction_bias");
        if (gate_bias && gate_bias->shape.size() == 1 &&
            gate_bias->data.size() >=
                static_cast<size_t>(gate_bias->shape[0]) * 4) {
            int n = static_cast<int>(gate_bias->shape[0]);
            if (cudaMallocHost(&lw.h_gate_bias,
                               static_cast<size_t>(n) * sizeof(float))
                == cudaSuccess) {
                std::memcpy(lw.h_gate_bias, gate_bias->data.data(),
                            static_cast<size_t>(n) * sizeof(float));
                lw.gate_bias_n = n;
            }
        }
    }

    auto *k_bias = FindTensor(tensors, "self_attn.indexer.k_norm.bias");
    if (k_bias && !k_bias->data.empty()) {
        void *d = nullptr;
        cudaError_t ke = cudaMalloc(&d, k_bias->data.size());
        if (ke != cudaSuccess) return ke;
        lw.d_idx_k_norm_bias = static_cast<float *>(d);
        ke = ring.UploadRaw(k_bias->data.data(), d, k_bias->data.size(),
                            stream, bytes_acc);
        if (ke != cudaSuccess) return ke;
    }
    return cudaSuccess;
}
```

4c. Global loader: route embed/head/final_norm through a ring too (individual
mallocs kept — three allocations are negligible). Replace the three
`UploadBf16ToHalf` calls in `LoadGlobal` with:

```cpp
    PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;
```

and for each tensor (embed example; head/norm analogous, preserving the
existing missing/malformed error logs and return codes):

```cpp
        cudaError_t e = cudaMalloc(&pw.d_embed,
            static_cast<size_t>(pw.vocab_size) * pw.hidden * 2);
        if (e != cudaSuccess) return e;
        e = ring.UploadBf16AsFp16(embed->data.data(),
                                  static_cast<int64_t>(pw.vocab_size) *
                                      pw.hidden,
                                  pw.d_embed, stream, pw.resident_bytes);
        if (e != cudaSuccess) return e;
```

Malformed-shape guards added in Task 4 for globals: embed requires
`shape.size() >= 2` (already enforced); add the same data-size check
`data.size() >= rows*cols*2` → else `log::error` + `return cudaErrorInvalidValue`.

4d. Error-path drain: at the top of `LoadGlmPinnedWeights`, wrap the body so
every early error return synchronizes the stream before `pw.Free()` /
ring destruction (pinned buffers must outlive in-flight DMA). Simplest
correct form — a small RAII guard used on error paths only:

```cpp
struct StreamDrain {
    cudaStream_t s;
    ~StreamDrain() { if (s) cudaStreamSynchronize(s); }
};
```

Instantiate `StreamDrain drain{stream};` right after the topology check and
let every `return` after it flow through the destructor. Remove nothing else.

4e. Arena-aware `Free()`:

```cpp
void GlmLayerWeights::Free()
{
    if (d_arena) {
        if (h_gate_bias) cudaFreeHost(h_gate_bias);
        cudaFree(d_arena);
        *this = GlmLayerWeights{};
        return;
    }
    // legacy branch unchanged below
    ...
}
```

- [ ] **Step 5: Run the full affected suites**

Run:
```bash
cmake --build --preset linux-debug && \
ctest --preset linux-debug -R "glm" --output-on-failure
```
Expected: PASS — all GLM suites including `test_glm_load_fast`,
`test_glm_loader`, `test_glm_compute`, `test_glm_dense_compute`,
`test_glm_moe_router`, `test_glm_experts`, `test_glm_moe_compute`,
`test_glm_attention`, `test_glm_inference` (real-model ones SKIP without a
checkpoint).

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/glm_inference_load.h \
  libs/lsxcommon/src/glm_inference_load.cpp \
  tests/test_glm_load_fast.cu tests/glm_test_fixture.h
git commit -m "feat: arena-backed ring uploads in GLM pinned loader"
```

---

### Task 6: Timing log + regression + docs

**Files:**
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp` (`LoadGlmPinnedWeights`)
- Modify: `AGENTS.md` (structure note)

**Interfaces:**
- Consumes: everything above.
- Produces: spdlog info line with wall-clock ms, MiB and GiB/s for every
  pinned-weights load.

- [ ] **Step 1: Add timing instrumentation**

At the top of `LoadGlmPinnedWeights` (after `pw.Free();`):

```cpp
    const auto t0 = std::chrono::steady_clock::now();
```

Before the final `return cudaSuccess;` (after shared experts):

```cpp
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double gib = static_cast<double>(pw.resident_bytes) /
                       (1024.0 * 1024.0 * 1024.0);
    const double gibs = (ms > 0.0) ? (gib / (ms / 1000.0)) : 0.0;
    log::info(absl::StrCat("[glm_load] pinned weights: ",
                           pw.resident_bytes >> 20, " MiB in ",
                           static_cast<int>(ms + 0.5), " ms (",
                           gibs, " GiB/s)"));
```

Add `#include <chrono>` to the include block.

- [ ] **Step 2: Capture before/after numbers on the real model (if present)**

Check whether the GLM checkpoint exists at the configured vault root
(`/mnt/llm_models/models` per AGENTS.md; the loader accepts
`<model>/logestix/pinned` or `<logestix>/pinned`). If present:

```bash
git stash && cmake --build --preset linux-debug --target test_glm_selfcheck && \
  ./build/bin/test_glm_selfcheck 2>&1 | grep "glm_load.*pinned" | tee /tmp/opencode/load_before.txt ; git stash pop && \
  cmake --build --preset linux-debug --target test_glm_selfcheck && \
  ./build/bin/test_glm_selfcheck 2>&1 | grep "glm_load.*pinned" | tee /tmp/opencode/load_after.txt
```

Report both numbers to the operator. Without a checkpoint, state that the
measurement is deferred (SKIP invariant) and rely on the synthetic-suite
green run.

- [ ] **Step 3: Full regression**

Run:
```bash
ctest --preset linux-debug -R "glm|core|tokenizer|deepseek_loader|model_factory" --output-on-failure
```
Expected: PASS (real-model suites SKIP without checkpoints).

- [ ] **Step 4: Update AGENTS.md structure entry**

In `AGENTS.md`, under `libs/lsxcommon/`, update the loader bullet:

```markdown
  - `libs/lsxcommon/src/glm_inference_load.cpp` + `include/lsxcommon/glm_inference_load.h` + `src/glm_load_internal.h` — host-only GLM loader: `LoadGlmPinnedWeights` (GPU in-place BF16→FP16 via `GlmBf16ToHalfInPlace`, pinned staging ring, per-layer device arenas, strict errors on missing layer files/broken tensors), `GlmLayerWeights`/`GlmPinnedWeights`
```

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/glm_inference_load.cpp AGENTS.md
git commit -m "feat: GLM pinned-load timing instrumentation"
```

---

## Self-Review Notes (already applied)

- Spec coverage: GPU in-place kernel (Task 1), pinned ring (Task 2), arena
  (Task 5), strict errors incl. heterogeneous-grid rule (Task 4), timing log
  (Task 6), bit-exactness tests (Tasks 1, 2, 5), idempotent reload (Task 5),
  real-model arbiter remains `test_glm_selfcheck` (untouched, SKIP-safe).
- NaN payload scope: patterns cover canonical qNaN + ±inf; arbitrary sNaN
  payloads excluded because host/device NaN canonicalization may differ —
  production tensors never carry them.
- Type consistency: `GlmBf16ToHalfInPlace(uint16_t*, int64_t, cudaStream_t)`
  used identically in Tasks 1, 2, 5; `PinnedStagingRing` API identical between
  Task 2 (definition) and Tasks 4/5 (consumers); `d_arena`/`arena_bytes`
  introduced in Task 5 Steps 1–2 and consumed by the same task's tests.
