# DeepSeek-V4-Flash Load Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the GLM-5.2 fast weight-loading path 1:1 to DeepSeek-V4-Flash: GPU in-place BF16→FP16, pinned staging ring, per-layer device arena, strict errors, topology-driven layer sizing, model-owned weights.

**Architecture:** Shared `lsxcommon::fastload` primitives (`Bf16ToHalfInPlace` kernel + `PinnedStagingRing`) extracted from the GLM loader into `lsxcommon_cu`; GLM delegates to them unchanged (regression-guarded by the existing GLM suites). The DeepSeek loader (`load_pinned_weights`) is rewritten around a per-layer arena upload plan (`ValidateJob` → one `cudaMalloc` per layer → ring uploads), strict error semantics and topology-driven sizing. `DeepSeekV4FlashModel` owns its `PinnedWeights` and loads once; `RunInferenceModel` consumes them without reloading.

**Tech Stack:** CUDA 13.3 (Blackwell sm_120/121 only), C++20, Arrow 25 IPC pinned schema, Catch2 v3.7, spdlog, CMake presets.

**Spec:** `docs/superpowers/specs/2026-08-22-deepseek-load-acceleration-design.md`

## Global Constraints

- Branch: `feat/deepseek-load-acceleration` (already created).
- Build/test via CMakePresets: `cmake --preset linux-debug && cmake --build --preset linux-debug`; run tests with `ctest --preset linux-debug -R <regex>` (from repo root).
- CUDA architectures are fixed at `120-real;121-real` — never add sm_80/89/90.
- NO absl container headers in anything reachable from `.cu`/`.cuh` TUs (NVCC prohibition). `fast_load_internal.h` uses only cuda/std headers.
- No `std::cout`/`printf` logging — spdlog via `#include "lsxcommon/log.h"` (`log::info/error/warn`).
- All loading on ONE `cudaStream_t`; no `cudaDeviceSynchronize()` in the load path; synchronization = staging-ring event parity + one final `cudaStreamSynchronize`.
- Bit-exactness invariant: GPU BF16→FP16 must equal host reference `(uint32(bits) << 16) -> float -> __float2half`.
- Comments: keep Doxygen-style comments; new code follows surrounding style. English commit messages with prefixes (`feat:`, `test:`, `refactor:`).
- Existing GLM tests (`ctest -R glm`) must stay green after every task.

---

### Task 1: Shared fast-load primitives + GLM delegation

**Files:**
- Create: `tests/test_deepseek_load_fast.cu`
- Create: `libs/lsxcommon/include/lsxcommon/fast_load.h`
- Create: `libs/lsxcommon/src/bf16_convert.cu`
- Create: `libs/lsxcommon/src/fast_load_internal.h`
- Modify: `libs/lsxcommon/src/glm_inference_load.cu:1580-1645` (kernel block)
- Modify: `libs/lsxcommon/src/glm_load_internal.h` (whole file → alias)
- Modify: `libs/lsxcommon/CMakeLists.txt:86-99` (add source)
- Modify: `tests/CMakeLists.txt` (append test target)

**Interfaces:**
- Consumes: nothing new (CUDA runtime only).
- Produces:
  - `cudaError_t lsxcommon::fastload::Bf16ToHalfInPlace(uint16_t *d_buf, int64_t numel, cudaStream_t stream);` (declared in `include/lsxcommon/fast_load.h`)
  - `class lsxcommon::fastload::PinnedStagingRing` (header-only in `src/fast_load_internal.h`; ctor `explicit PinnedStagingRing(size_t chunk_bytes = kDefaultChunkBytes)`, `bool ok() const`, `cudaError_t UploadBf16AsFp16(const uint8_t *host_bf16, int64_t numel, __half *d_dst, cudaStream_t stream, std::uint64_t &bytes_acc)`, `cudaError_t UploadRaw(const void *src, void *d_dst, size_t bytes, cudaStream_t stream, std::uint64_t &bytes_acc)`)
  - `lsxcommon::glm::PinnedStagingRing` becomes an alias of the fastload class; `GlmBf16ToHalfInPlace` delegates to `Bf16ToHalfInPlace` (existing signatures unchanged).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_deepseek_load_fast.cu`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include <lsxcommon/fast_load.h>

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
        0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3555,
        0x3556, 0x4248, 0xC048, 0x7F80, 0xFF80, 0x7FC0,
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
    cudaError_t e = lsxcommon::fastload::Bf16ToHalfInPlace(d, n, stream);
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

TEST_CASE("fastload Bf16ToHalfInPlace bit-exact vs host reference",
          "[deepseek][loadfast]")
{
    CheckBitExact(TestPatterns());
    CheckBitExact(RandomFinite(100003, 1));
}

TEST_CASE("fastload Bf16ToHalfInPlace ragged tails stay bit-exact",
          "[deepseek][loadfast]")
{
    for (int64_t n : {1, 7, 9, 4097}) {
        CheckBitExact(RandomFinite(static_cast<size_t>(n), 42));
    }
}

#include "fast_load_internal.h"

using lsxcommon::fastload::PinnedStagingRing;

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

TEST_CASE("Staging ring chunked BF16 upload is bit-exact",
          "[deepseek][loadfast]")
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
          "[deepseek][loadfast]")
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

TEST_CASE("Staging ring raw upload matches source", "[deepseek][loadfast]")
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
    std::vector<uint16_t> expect(kBytes / 2);
    std::memcpy(expect.data(), src.data(), expect.size() * 2);
    REQUIRE(DeviceBitsEqual(stream,
                            reinterpret_cast<const __half *>(d), expect));

    cudaStreamDestroy(stream);
    cudaFree(d);
}
```

- [ ] **Step 2: Register the test target**

Append to `tests/CMakeLists.txt` (copying the `test_deepseek_arena` OBJECT-lib pattern from lines 1040-1090):

```cmake
# test_deepseek_load_fast — shared fast-load primitives (GPU BF16->FP16 +
# pinned staging ring). Same OBJECT-lib pattern as test_deepseek_arena_cu.
add_library(test_deepseek_load_fast_cu OBJECT test_deepseek_load_fast.cu)
set_target_properties(test_deepseek_load_fast_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_load_fast_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
target_include_directories(test_deepseek_load_fast_cu PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src"
    "${CMAKE_BINARY_DIR}/libs/lsxcommon/include"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/thrust"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/cub"
    "${CMAKE_SOURCE_DIR}/third_party/cccl/libcudacxx/include"
    "${CMAKE_BINARY_DIR}/_deps/arrow-src/cpp/src"
    "${CMAKE_BINARY_DIR}/_deps/arrow-build/src"
    "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include"
)
target_compile_definitions(test_deepseek_load_fast_cu PRIVATE
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP
    CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    LOGESTIX_HAS_CUDA
)
target_link_libraries(test_deepseek_load_fast_cu PRIVATE Catch2::Catch2WithMain)

add_executable(test_deepseek_load_fast)
set_target_properties(test_deepseek_load_fast PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_load_fast PRIVATE
    test_deepseek_load_fast_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_load_fast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_load_fast>"
    )
    add_custom_command(TARGET test_deepseek_load_fast POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_load_fast>"
    )
endif()

add_test(NAME test_deepseek_load_fast COMMAND test_deepseek_load_fast)
```

- [ ] **Step 3: Run build to verify it fails**

Run: `cmake --preset linux-debug && cmake --build --preset linux-debug --target test_deepseek_load_fast 2>&1 | tail -20`
Expected: FAIL — fatal error: `lsxcommon/fast_load.h: No such file or directory`.

- [ ] **Step 4: Implement the shared header**

Create `libs/lsxcommon/include/lsxcommon/fast_load.h`:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lsxcommon {
namespace fastload {

// In-place device conversion of raw BF16 bits to FP16 bits. d_buf holds
// numel BF16 values on entry and their FP16 images on exit. Rounding is
// nearest-even via __float2half — bit-identical to the legacy host scalar
// path (bits<<16 reinterpreted as FP32). d_buf must be 16-byte aligned.
cudaError_t Bf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                              cudaStream_t stream);

} // namespace fastload
} // namespace lsxcommon
```

- [ ] **Step 5: Implement the kernel TU**

Create `libs/lsxcommon/src/bf16_convert.cu`:

```cpp
#include "lsxcommon/fast_load.h"

namespace lsxcommon {
namespace fastload {
namespace {

// In-place BF16→FP16 conversion, bit-exact mirror of the legacy host scalar
// path: bf16 bits<<16 reinterpreted as FP32, rounded to nearest-even FP16.
__global__ void bf16_to_half_in_place_kernel(uint16_t *__restrict__ buf,
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

} // namespace

cudaError_t Bf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                              cudaStream_t stream)
{
    if (!d_buf || numel <= 0) return cudaErrorInvalidValue;
    constexpr int kBlock = 256;
    const int64_t nv = numel / 8;
    int64_t blocks64 = (nv + kBlock - 1) / kBlock;
    if (blocks64 < 1) blocks64 = 1;
    if (blocks64 > 65535) blocks64 = 65535;
    // Drain any stale thread-local CUDA error left by unrelated earlier
    // calls, so cudaGetLastError() below reflects THIS launch only.
    (void)cudaGetLastError();
    bf16_to_half_in_place_kernel<<<static_cast<int>(blocks64), kBlock, 0,
                                   stream>>>(d_buf, numel);
    return cudaGetLastError();
}

} // namespace fastload
} // namespace lsxcommon
```

Add the source to `libs/lsxcommon/CMakeLists.txt` inside `add_library(lsxcommon_cu OBJECT ...)` (after `src/deepseek_inference_pipeline.cu`):

```cmake
    src/bf16_convert.cu
```

- [ ] **Step 6: Move the staging ring**

Create `libs/lsxcommon/src/fast_load_internal.h`:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

#include "lsxcommon/fast_load.h"

namespace lsxcommon {
namespace fastload {

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
        // Both events were recorded right after each buffer's latest copy:
        // synchronizing them proves no DMA read from bufs_[i] is still in
        // flight before cudaFreeHost (a freed-while-in-flight pinned buffer
        // poisons the CUDA context for every later call).
        for (int i = 0; i < 2; ++i)
            if (ev_[i]) cudaEventSynchronize(ev_[i]);
        for (int i = 0; i < 2; ++i) {
            if (bufs_[i]) cudaFreeHost(bufs_[i]);
            if (ev_[i]) cudaEventDestroy(ev_[i]);
        }
    }

    PinnedStagingRing(const PinnedStagingRing &) = delete;
    PinnedStagingRing &operator=(const PinnedStagingRing &) = delete;

    bool ok() const { return bufs_[0] && bufs_[1] && ev_[0] && ev_[1]; }

    // Uploads numel BF16 values to d_dst, converting them to FP16 on device
    // (bit-exact legacy path). d_dst must be 16-byte aligned.
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
            e = Bf16ToHalfInPlace(dst16, cnt, stream);
            if (e != cudaSuccess) return e;
            off += cnt;
            bi ^= 1;
        }
        bytes_acc += static_cast<std::uint64_t>(numel) * 2;
        return cudaSuccess;
    }

    // Plain chunked byte upload (F32/F8 tensors etc.), same event-guarded ring.
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

} // namespace fastload
} // namespace lsxcommon
```

Replace the ENTIRE content of `libs/lsxcommon/src/glm_load_internal.h` with:

```cpp
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>

#include <lsxcommon/glm_inference_load.h>
#include "fast_load_internal.h"

namespace lsxcommon {
namespace glm {

// GLM now shares the model-neutral fast-load primitives; the glm:: name is
// kept so all existing call sites (loader + tests) compile unchanged.
using PinnedStagingRing = lsxcommon::fastload::PinnedStagingRing;

} // namespace glm
} // namespace lsxcommon
```

- [ ] **Step 7: Delegate the GLM symbol**

In `libs/lsxcommon/src/glm_inference_load.cu`: add `#include "lsxcommon/fast_load.h"` next to the other includes, then REPLACE the whole block from the comment line `// ── Fast-path loader kernels ───...` through the end of `GlmBf16ToHalfInPlace` (lines 1580-1645) with:

```cpp
// ── Fast-path loader kernels ────────────────────────────────────────────────
// The bit-exact BF16→FP16 conversion kernel moved to the model-neutral
// fastload TU (bf16_convert.cu) so the DeepSeek fast loader can share it.
// The glm:: entry point stays for API compatibility.
cudaError_t GlmBf16ToHalfInPlace(uint16_t *d_buf, int64_t numel,
                                 cudaStream_t stream)
{
    return fastload::Bf16ToHalfInPlace(d_buf, numel, stream);
}
```

- [ ] **Step 8: Build and verify GREEN**

Run:
```bash
cmake --preset linux-debug && cmake --build --preset linux-debug
ctest --preset linux-debug -R "deepseek_load_fast|glm_load_fast|glm_loader" --output-on-failure
```
Expected: `test_deepseek_load_fast` PASS (5 test cases), `test_glm_load_fast` PASS (unchanged), `test_glm_loader` PASS.

- [ ] **Step 9: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/fast_load.h \
        libs/lsxcommon/src/bf16_convert.cu \
        libs/lsxcommon/src/fast_load_internal.h \
        libs/lsxcommon/src/glm_inference_load.cu \
        libs/lsxcommon/src/glm_load_internal.h \
        libs/lsxcommon/CMakeLists.txt \
        tests/test_deepseek_load_fast.cu tests/CMakeLists.txt
git commit -m "refactor: extract shared fastload Bf16ToHalfInPlace + PinnedStagingRing from GLM loader"
```

---

### Task 2: Synthetic DeepSeek pinned-model fixture + failing loader tests

**Files:**
- Create: `tests/deepseek_test_fixture.h`
- Modify: `tests/test_deepseek_loader.cpp` (append includes + 2 test cases)
- Modify: `tests/CMakeLists.txt:22-23` (add src include dir to test_deepseek_loader)

**Interfaces:**
- Consumes: `lsxcommon::PinnedSchema::MakeBatch`, `lsxcommon::WriteRecordBatchToFile` (`include/lsxcommon/schema.h`), `lsxcommon::gpu::load_pinned_weights(model_path, pw, stream)` (`src/deepseek_inference_load.cuh`).
- Produces: `dsfixtest::BuildSyntheticModel(fs::path)` → `arrow::Status`; constants `dsfixtest::kVocab=128, kHidden=256, kNumLayers=3, kNumRouted=8`. Task 3's containment/malformed tests reuse these.

- [ ] **Step 1: Write the fixture header**

Create `tests/deepseek_test_fixture.h`:

```cpp
#pragma once

#include <arrow/api.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <lsxcommon/schema.h>

// Synthetic minimal DeepSeek-V4-Flash converted-model layout for loader tests:
//   <model_dir>/logestix/model.index.json     (topology + vram block)
//   <model_dir>/logestix/pinned/global.arrow  (embed/head/norm + HC head F32)
//   <model_dir>/logestix/pinned/layer_<n>.arrow (pure-MLA layer tensor set)
// No shared_experts.arrow: that file stays optional-if-absent by design.
namespace dsfixtest {

constexpr int kVocab = 128;
constexpr int kHidden = 256;
constexpr int kNumLayers = 3;
constexpr int kNumRouted = 8;

inline void AppendU16(std::vector<uint8_t> &data, uint16_t v)
{
    const auto off = data.size();
    data.resize(off + 2);
    std::memcpy(data.data() + off, &v, 2);
}

inline lsxcommon::TensorRecord Bf16Record(const std::string &name,
                                          std::vector<int64_t> shape,
                                          uint16_t fill_bits)
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = "BF16";
    t.shape = std::move(shape);
    int64_t numel = 1;
    for (int64_t d : t.shape) numel *= d;
    t.data.reserve(static_cast<size_t>(numel) * 2);
    for (int64_t i = 0; i < numel; ++i) AppendU16(t.data, fill_bits);
    return t;
}

inline lsxcommon::TensorRecord F32Record(const std::string &name,
                                         std::vector<int64_t> shape,
                                         float fill)
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = "F32";
    t.shape = std::move(shape);
    int64_t numel = 1;
    for (int64_t d : t.shape) numel *= d;
    t.data.resize(static_cast<size_t>(numel) * 4);
    for (size_t off = 0; off < t.data.size(); off += 4)
        std::memcpy(t.data.data() + off, &fill, 4);
    return t;
}

inline lsxcommon::TensorRecord ByteRecord(const std::string &name,
                                          std::vector<int64_t> shape,
                                          uint8_t fill,
                                          const char *dtype = "F8_E4M3")
{
    lsxcommon::TensorRecord t;
    t.name = name;
    t.dtype = dtype;
    t.shape = std::move(shape);
    int64_t numel = 1;
    for (int64_t d : t.shape) numel *= d;
    t.data.resize(static_cast<size_t>(numel), fill);
    return t;
}

inline void WriteIndexJson(const std::filesystem::path &model_dir)
{
    std::filesystem::create_directories(model_dir / "logestix");
    std::ofstream f(model_dir / "logestix" / "model.index.json");
    f << "{\n"
      << "  \"model\": \"DeepSeek-V4-Flash\",\n"
      << "  \"num_layers\": " << kNumLayers << ",\n"
      << "  \"experts_per_layer\": 2,\n"
      << "  \"total_experts\": " << (kNumLayers * 2) << ",\n"
      << "  \"vram\": {\n"
      << "    \"min_bytes\": 1024,\n"
      << "    \"optimal_bytes\": 2048,\n"
      << "    \"max_bytes\": 4096,\n"
      << "    \"context_tokens\": 2048\n"
      << "  }\n"
      << "}\n";
}

inline arrow::Status WritePinnedFile(
    const std::filesystem::path &path,
    const std::vector<lsxcommon::TensorRecord> &tensors)
{
    auto batch = lsxcommon::PinnedSchema::MakeBatch(tensors);
    if (!batch.ok()) return batch.status();
    return lsxcommon::WriteRecordBatchToFile(*batch, path);
}

inline std::vector<lsxcommon::TensorRecord> GlobalRecords()
{
    std::vector<lsxcommon::TensorRecord> g;
    g.push_back(Bf16Record("model.embed_tokens.weight", {kVocab, kHidden},
                           0x4248));
    g.push_back(Bf16Record("lm_head.weight", {kVocab, kHidden}, 0x3555));
    g.push_back(Bf16Record("model.norm.weight", {kHidden}, 0x3C00));
    g.push_back(F32Record("hc_head_fn", {4, kHidden * 4}, 1.0f));
    g.push_back(F32Record("hc_head_base", {4}, 0.5f));
    g.push_back(F32Record("hc_head_scale", {1}, 2.0f));
    return g;
}

inline std::vector<lsxcommon::TensorRecord> LayerRecords(int l)
{
    const std::string p = "layers." + std::to_string(l) + ".";
    std::vector<lsxcommon::TensorRecord> v;
    v.push_back(Bf16Record(p + "ffn_norm.weight", {kHidden}, 0x3C00));
    v.push_back(Bf16Record(p + "ffn.gate.weight", {kNumRouted, kHidden},
                           0x4248));
    v.push_back(F32Record(p + "ffn.gate.bias", {kNumRouted}, 0.25f));
    v.push_back(ByteRecord(p + "attn.wq_a.weight", {512, kHidden}, 0xAA));
    v.push_back(ByteRecord(p + "attn.wq_a.scale", {4, 4}, 0x38,
                           "F8_E8M0"));
    v.push_back(Bf16Record(p + "attn.q_norm.weight", {64}, 0x3C00));
    v.push_back(Bf16Record(p + "attn.kv_norm.weight", {32}, 0x3C00));
    v.push_back(Bf16Record(p + "attn_norm.weight", {kHidden}, 0x3C00));
    v.push_back(F32Record(p + "attn.attn_sink", {8}, 1.0f));
    v.push_back(F32Record(p + "hc_attn_fn", {24, kHidden * 4}, 1.0f));
    v.push_back(F32Record(p + "hc_attn_base", {24}, 0.5f));
    v.push_back(F32Record(p + "hc_attn_scale", {3}, 2.0f));
    v.push_back(F32Record(p + "hc_ffn_fn", {24, kHidden * 4}, 1.0f));
    v.push_back(F32Record(p + "hc_ffn_base", {24}, 0.5f));
    v.push_back(F32Record(p + "hc_ffn_scale", {3}, 2.0f));
    return v;
}

inline arrow::Status BuildSyntheticModel(const std::filesystem::path &model_dir)
{
    const auto pinned = model_dir / "logestix" / "pinned";
    std::filesystem::create_directories(pinned);
    WriteIndexJson(model_dir);
    if (auto st = WritePinnedFile(pinned / "global.arrow", GlobalRecords());
        !st.ok())
        return st;
    for (int l = 0; l < kNumLayers; ++l) {
        auto fn = pinned / ("layer_" + std::to_string(l) + ".arrow");
        if (auto st = WritePinnedFile(fn, LayerRecords(l)); !st.ok())
            return st;
    }
    return arrow::Status::OK();
}

} // namespace dsfixtest
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_deepseek_loader.cpp`:

```cpp
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "deepseek_inference_load.cuh"
#include "deepseek_test_fixture.h"

namespace {

uint16_t RefDsBf16ToHalf(uint16_t bits)
{
    uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    __half h = __float2half(f);
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

} // namespace

TEST_CASE("Fast pinned load uploads synthetic model bit-exact",
          "[deepseek][loader][fastload]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_fast";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    CHECK(pw.vocab_size == dsfixtest::kVocab);
    CHECK(pw.d_embed != nullptr);
    CHECK(pw.d_head != nullptr);
    CHECK(pw.d_final_norm != nullptr);
    CHECK(pw.resident_bytes > 0);

    // embed[0] must equal BF16(0x4248) converted to FP16 bit-exactly.
    {
        uint16_t got = 0;
        REQUIRE(cudaMemcpy(&got, pw.d_embed, 2, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        CHECK(got == RefDsBf16ToHalf(0x4248));
    }
    {
        auto &lw = pw.layers[1];
        REQUIRE(lw.d_ffn_norm != nullptr);
        uint16_t got = 0;
        REQUIRE(cudaMemcpy(&got, lw.d_ffn_norm, 2, cudaMemcpyDeviceToHost)
                == cudaSuccess);
        CHECK(got == RefDsBf16ToHalf(0x3C00));
    }

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("DeepSeek loader sizes layers from topology",
          "[deepseek][loader][fastload]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_topo";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    // Topology-driven sizing (index.json says 3; was hardcoded 43).
    CHECK(pw.layers.size() == dsfixtest::kNumLayers);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("DeepSeek loader errors on missing layer file",
          "[deepseek][loader][strict]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_missing";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());
    fs::remove(model / "logestix" / "pinned" / "layer_1.arrow");

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    CHECK(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
          == cudaErrorInvalidValue);
    pw.Free();

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}
```

Modify `tests/CMakeLists.txt` — extend the `test_deepseek_loader` target (after line 23's `target_link_libraries`) so the private headers resolve:

```cmake
target_include_directories(test_deepseek_loader PRIVATE
    "${CMAKE_SOURCE_DIR}/libs/lsxcommon/src")
```

- [ ] **Step 3: Run to verify RED (exact expected failures)**

Run: `cmake --build --preset linux-debug --target test_deepseek_loader && ctest --preset linux-debug -R deepseek_loader --output-on-failure`
Expected FAILURES (do not commit yet):
- `"Fast pinned load uploads synthetic model bit-exact"` — FAILS at `REQUIRE(load == cudaSuccess)`: current loader returns success but leaves layers null? No — current loader loads layers 0-2 fine; this case should actually PASS except it may pass fully. If it passes, fine (it is the baseline).
- `"DeepSeek loader sizes layers from topology"` — FAIL: `pw.layers.size()` is 43, expected 3.
- `"DeepSeek loader errors on missing layer file"` — FAIL: current loader silently skips missing layer files and returns `cudaSuccess`.

- [ ] **Step 4: Leave the tree in RED state (no commit yet)**

Task 3 turns these green and commits fixture+tests+implementation together.

---

### Task 3: Rewrite `load_pinned_weights` — arena + ring + strict errors + topology

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cuh:226-297` (LayerWeights arena fields)
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cu:25-124` (helpers), `:302-347` (Free), `:361-771` (main flow)
- Modify: `tests/test_deepseek_loader.cpp` (containment + malformed-tensor tests)

**Interfaces:**
- Consumes: `lsxcommon::fastload::PinnedStagingRing` (Task 1), `dsfixtest::BuildSyntheticModel` (Task 2).
- Produces: `LayerWeights::d_arena`/`arena_bytes` fields (public struct members, consumed by pipeline unchanged — pipeline only reads existing pointer fields); `load_pinned_weights` keeps signature `(const std::string&, PinnedWeights&, cudaStream_t)`; new behavior: idempotent (calls `pw.Free()` first), strict errors, topology-sized `pw.layers`.

- [ ] **Step 1: Write the failing containment/malformed tests**

Append to `tests/test_deepseek_loader.cpp`:

```cpp
namespace {

bool DsInRange(const void *p, const void *base, size_t bytes)
{
    return p >= base &&
           static_cast<const uint8_t *>(p) <
               static_cast<const uint8_t *>(base) + bytes;
}

} // namespace

TEST_CASE("Arena-backed DeepSeek load: containment + idempotency",
          "[deepseek][loader][fastload]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_arena";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    REQUIRE(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(pw.layers.size() == dsfixtest::kNumLayers);

    for (int l = 0; l < dsfixtest::kNumLayers; ++l) {
        auto &lw = pw.layers[l];
        REQUIRE(lw.d_arena != nullptr);
        REQUIRE(lw.arena_bytes > 0);
        CHECK(DsInRange(lw.d_ffn_norm, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_gate_weight, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_wq_a, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_wq_a_scales, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_hc_attn_fn, lw.d_arena, lw.arena_bytes));
        CHECK(DsInRange(lw.d_attn_sink, lw.d_arena, lw.arena_bytes));
    }

    // Idempotent re-load: frees the previous set internally, no leak.
    REQUIRE(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
            == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    CHECK(pw.layers.size() == dsfixtest::kNumLayers);

    pw.Free();
    cudaStreamDestroy(stream);
    fs::remove_all(base);
}

TEST_CASE("DeepSeek loader errors on malformed tensor",
          "[deepseek][loader][strict]")
{
    namespace fs = std::filesystem;
    auto base = fs::temp_directory_path() / "logestix_ds_loader_badtensor";
    fs::remove_all(base);
    auto model = base / "model";
    REQUIRE(dsfixtest::BuildSyntheticModel(model).ok());

    // Corrupt global.arrow: truncate model.norm.weight payload below its
    // declared shape size.
    {
        auto recs = dsfixtest::GlobalRecords();
        recs[2].data.resize(2);   // shape says kHidden elements
        REQUIRE(dsfixtest::WriteGlobalArrow(model, recs).ok());
    }

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    lsxcommon::gpu::PinnedWeights pw;
    CHECK(lsxcommon::gpu::load_pinned_weights(model.string(), pw, stream)
          == cudaErrorInvalidValue);
    pw.Free();

    cudaStreamDestroy(stream);
    fs::remove_all(base);
}
```

For the malformed test, expose the fixture writer by adding at the end of `tests/deepseek_test_fixture.h` (inside the namespace):```cpp
inline arrow::Status WriteGlobalArrow(
    const std::filesystem::path &model_dir,
    const std::vector<lsxcommon::TensorRecord> &records)
{
    return WritePinnedFile(model_dir / "logestix" / "pinned" /
                               "global.arrow",
                           records);
}
```

…and use `dsfixtest::WriteGlobalArrow(model, recs)` instead of `lsxfix_write_pinned(...)` in the malformed test above.

Run: `cmake --build --preset linux-debug --target test_deepseek_loader 2>&1 | tail -10`
Expected: FAIL to compile — `'d_arena': is not a member of 'lsxcommon::gpu::LayerWeights'` (this is the RED state).

- [ ] **Step 2: Add arena fields to LayerWeights**

In `libs/lsxcommon/src/deepseek_inference_load.cuh`, inside `struct LayerWeights` just before `void Free();` (line ~296), add:

```cpp
    // Non-null when the layer was uploaded arena-backed: every d_* device
    // pointer above is a view into this single allocation and Free()
    // releases the arena instead of the individual pointers. Host-pinned
    // router state (h_tid2eid/h_gate_bias) is NOT part of the arena.
    void *d_arena = nullptr;
    size_t arena_bytes = 0;
```

- [ ] **Step 3: Arena-first Free path**

In `libs/lsxcommon/src/deepseek_inference_load.cu`, replace the beginning of `LayerWeights::Free()` (line 302):

```cpp
void LayerWeights::Free()
{
    if (d_arena) {
        if (h_tid2eid)   cudaFreeHost(h_tid2eid);
        if (h_gate_bias) cudaFreeHost(h_gate_bias);
        cudaFree(d_arena);
        *this = LayerWeights{};
        return;
    }
    if (d_ffn_norm)     cudaFree(d_ffn_norm);
    // ... rest of the existing body unchanged ...
}
```

- [ ] **Step 4: Replace upload helpers with arena job machinery**

In `libs/lsxcommon/src/deepseek_inference_load.cu`:

(a) Add include at top (next to the other lsxcommon includes):

```cpp
#include "lsxcommon/fast_load.h"
#include "fast_load_internal.h"
#include <chrono>
```

(b) DELETE the anonymous-namespace helpers `upload_bf16_to_half` (:30-52), `upload_bf16_checked` (:57-67), `upload_f8_bytes` (:93-107), `upload_f32_bytes` (:110-124), `load_attn_weight` (:127-169), `load_hc_f32` (:172-181).

(c) ADD in the same anonymous namespace:

```cpp
constexpr size_t kArenaAlign = 256;

size_t AlignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

// Drains the stream on scope exit so pinned staging buffers never die while
// a DMA transfer is still in flight (mirrors the GLM loader guard).
struct StreamDrain {
    cudaStream_t s;
    ~StreamDrain() { if (s) cudaStreamSynchronize(s); }
};

// One planned upload into the per-layer arena. to_fp16=true uploads BF16
// through the staging ring with an on-device BF16→FP16 conversion; otherwise
// raw bytes are uploaded verbatim (F32 / F8_E4M3 / F8_E8M0 / I64 payloads).
struct ArenaJob {
    const TensorRecord *t;
    void **dst;
    bool to_fp16;
    int64_t *bytes_cap = nullptr;   // optional: uploaded byte count
    int *rows_cap = nullptr;        // optional: shape[0] when 2D
    int *cols_cap = nullptr;        // optional: shape[1] when 2D
};

cudaError_t ValidateArenaJob(const ArenaJob &j, size_t &upload_bytes)
{
    const auto &s = j.t->shape;
    if (j.to_fp16) {
        if (s.size() == 1 && s[0] > 0)
            upload_bytes = static_cast<size_t>(s[0]) * 2;
        else if (s.size() == 2 && s[0] > 0 && s[1] > 0)
            upload_bytes = static_cast<size_t>(s[0])
                         * static_cast<size_t>(s[1]) * 2;
        else {
            log::error("[load_pinned] malformed BF16 tensor '" + j.t->name
                       + "'");
            return cudaErrorInvalidValue;
        }
        if (j.t->data.size() < upload_bytes) {
            log::error("[load_pinned] truncated BF16 tensor '" + j.t->name
                       + "'");
            return cudaErrorInvalidValue;
        }
    } else {
        upload_bytes = j.t->data.size();
        if (upload_bytes == 0) {
            log::error("[load_pinned] empty byte tensor '" + j.t->name + "'");
            return cudaErrorInvalidValue;
        }
    }
    return cudaSuccess;
}

// Validates every job, issues ONE arena allocation sized to the layer's total
// upload footprint and streams each tensor through the pinned staging ring
// into its 256-byte-aligned arena view. On success lw.d_arena/arena_bytes own
// the block and every *dst points inside it. Any failure aborts the whole
// layer (caller frees the weights).
cudaError_t UploadLayerJobs(std::vector<ArenaJob> jobs, LayerWeights &lw,
                            cudaStream_t stream,
                            lsxcommon::fastload::PinnedStagingRing &ring,
                            std::uint64_t &bytes_acc)
{
    std::vector<size_t> sz(jobs.size(), 0);
    size_t total = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].t) continue;
        cudaError_t e = ValidateArenaJob(jobs[i], sz[i]);
        if (e != cudaSuccess) return e;
        total = AlignUp(total, kArenaAlign) + sz[i];
    }

    void *arena = nullptr;
    if (total > 0) {
        cudaError_t e = cudaMalloc(&arena, total);
        if (e != cudaSuccess) {
            log::error(std::string("[load_pinned] layer arena alloc failed: ")
                       + cudaGetErrorString(e));
            return e;
        }
    }
    lw.d_arena = arena;
    lw.arena_bytes = total;

    uint8_t *base = static_cast<uint8_t *>(arena);
    size_t off = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const auto &j = jobs[i];
        if (!j.t) continue;
        off = AlignUp(off, kArenaAlign);
        void *view = base + off;
        *j.dst = view;
        if (j.bytes_cap) *j.bytes_cap = static_cast<int64_t>(sz[i]);
        if ((j.rows_cap || j.cols_cap) && j.t->shape.size() == 2) {
            if (j.rows_cap) *j.rows_cap = static_cast<int>(j.t->shape[0]);
            if (j.cols_cap) *j.cols_cap = static_cast<int>(j.t->shape[1]);
        }
        cudaError_t e = j.to_fp16
            ? ring.UploadBf16AsFp16(j.t->data.data(),
                  static_cast<int64_t>(sz[i] / 2),
                  static_cast<__half *>(view), stream, bytes_acc)
            : ring.UploadRaw(j.t->data.data(), view, sz[i], stream,
                             bytes_acc);
        if (e != cudaSuccess) {
            log::error("[load_pinned] upload failed for '" + j.t->name
                       + "': " + cudaGetErrorString(e));
            return e;
        }
        off += sz[i];
    }
    return cudaSuccess;
}
```

- [ ] **Step 5: Rewrite the main load flow**

Rewrite `load_pinned_weights` (`:361-771`). Full replacement body:

```cpp
cudaError_t load_pinned_weights(
    const std::string &model_path,
    PinnedWeights &pw,
    cudaStream_t stream)
{
    namespace fs = std::filesystem;

    // Idempotent re-load: drop any previously uploaded set first so calling
    // Load twice cannot leak the old device allocations.
    pw.Free();

    const auto t0 = std::chrono::steady_clock::now();

    auto pinned_dir = fs::path(model_path) / "logestix" / "pinned";

    // Drains the stream on every error exit below so pinned staging never
    // dies while a DMA transfer is still in flight.
    StreamDrain drain{stream};

    lsxcommon::fastload::PinnedStagingRing ring;
    if (!ring.ok()) return cudaErrorMemoryAllocation;

    // ── Topology: layer count comes from model.index.json (was hardcoded) ──
    auto topo = load_topology((fs::path(model_path) / "logestix"
                               / "model.index.json").string());
    if (!topo.error_msg.empty() || topo.num_layers <= 0) {
        log::error("[load_pinned] topology error: "
                   + (topo.error_msg.empty()
                          ? std::string("num_layers <= 0")
                          : topo.error_msg));
        return cudaErrorInvalidValue;
    }

    // Per-layer compress_ratio table (decoder grid of DeepSeek-V4-Flash:
    // pure MLA=0, CSA>0; ratio==4 uses coff=2 overlap).
    static const int compress_ratios[] = {
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4
    };
    constexpr int kRatioTableSize =
        static_cast<int>(sizeof(compress_ratios) / sizeof(compress_ratios[0]));
    const int num_layers = topo.num_layers;
    if (num_layers > kRatioTableSize) {
        log::error("[load_pinned] topology num_layers="
                   + std::to_string(num_layers)
                   + " exceeds supported decoder grid (" +
                   std::to_string(kRatioTableSize) + ")");
        return cudaErrorInvalidValue;
    }
    pw.layers.resize(static_cast<size_t>(num_layers));

    std::uint64_t rb = 0;   // device-resident byte total (weights only)

    // ── global.arrow ──────────────────────────────────────────────────────
    auto global_path = pinned_dir / "global.arrow";
    auto global_tensors_res = load_arrow_file(global_path.string());
    if (!global_tensors_res.ok()) {
        log::error("[load_pinned] ERROR loading global.arrow: "
                   + global_tensors_res.status().ToString());
        return cudaErrorInvalidValue;
    }
    auto &gt = *global_tensors_res;

    // Strict global set: embed/norm are mandatory (the model cannot run
    // without them); head is mandatory too (tied-weight-free layout).
    auto *embed_t = find_tensor_suffix(gt, "embed.weight");
    auto *head_t  = find_tensor_suffix(gt, "head.weight");
    auto *norm_t  = find_tensor_suffix(gt, "norm.weight");
    if (!embed_t || embed_t->shape.size() != 2 ||
        embed_t->data.size() <
            static_cast<size_t>(embed_t->shape[0]) *
                static_cast<size_t>(embed_t->shape[1]) * 2) {
        log::error("[load_pinned] embed.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }
    if (!head_t || head_t == embed_t || head_t->shape.size() != 2 ||
        head_t->data.size() <
            static_cast<size_t>(head_t->shape[0]) *
                static_cast<size_t>(head_t->shape[1]) * 2) {
        log::error("[load_pinned] head.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }
    if (!norm_t || norm_t->shape.size() != 1 ||
        norm_t->data.size() < static_cast<size_t>(norm_t->shape[0]) * 2) {
        log::error("[load_pinned] norm.weight missing/malformed in "
                   "global.arrow");
        return cudaErrorInvalidValue;
    }

    pw.vocab_size = static_cast<int>(embed_t->shape[0]);
    {
        int64_t numel =
            static_cast<int64_t>(embed_t->shape[0]) * embed_t->shape[1];
        cudaError_t ge = cudaMalloc(&pw.d_embed, static_cast<size_t>(numel) * 2);
        if (ge != cudaSuccess) return ge;
        rb += static_cast<std::uint64_t>(numel) * 2;
        ge = ring.UploadBf16AsFp16(embed_t->data.data(), numel,
                                   pw.d_embed, stream, rb);
        if (ge != cudaSuccess) {
            log::error("[load_pinned] embed upload failed: "
                       + cudaGetErrorString(ge));
            return ge;
        }
    }
    {
        int64_t numel =
            static_cast<int64_t>(head_t->shape[0]) * head_t->shape[1];
        cudaError_t ge = cudaMalloc(&pw.d_head, static_cast<size_t>(numel) * 2);
        if (ge != cudaSuccess) return ge;
        rb += static_cast<std::uint64_t>(numel) * 2;
        ge = ring.UploadBf16AsFp16(head_t->data.data(), numel,
                                   pw.d_head, stream, rb);
        if (ge != cudaSuccess) {
            log::error("[load_pinned] head upload failed: "
                       + cudaGetErrorString(ge));
            return ge;
        }
    }
    {
        int64_t numel = norm_t->shape[0];
        cudaError_t ge = cudaMalloc(&pw.d_final_norm,
                                    static_cast<size_t>(numel) * 2);
        if (ge != cudaSuccess) return ge;
        rb += static_cast<std::uint64_t>(numel) * 2;
        ge = ring.UploadBf16AsFp16(norm_t->data.data(), numel,
                                   pw.d_final_norm, stream, rb);
        if (ge != cudaSuccess) {
            log::error("[load_pinned] final_norm upload failed: "
                       + cudaGetErrorString(ge));
            return ge;
        }
    }

    // HC head (FP32) — strict status checks.
    for (auto [suffix, dst] : std::initializer_list<
             std::pair<const char *, float **>>{
             {"hc_head_fn", &pw.d_hc_head_fn},
             {"hc_head_base", &pw.d_hc_head_base},
             {"hc_head_scale", &pw.d_hc_head_scale}}) {
        auto *t = find_tensor_suffix(gt, suffix);
        if (!t || t->data.empty()) {
            log::error(std::string("[load_pinned] hc head tensor missing: ")
                       + suffix);
            return cudaErrorInvalidValue;
        }
        cudaError_t he = cudaMalloc(dst, t->data.size());
        if (he != cudaSuccess) return he;
        rb += t->data.size();
        he = ring.UploadRaw(t->data.data(), *dst, t->data.size(), stream, rb);
        if (he != cudaSuccess) {
            log::error(std::string("[load_pinned] upload failed for ")
                       + suffix + ": " + cudaGetErrorString(he));
            return he;
        }
    }
```

Continue with the layer loop (replacing old lines 431-699):

```cpp
    // ── Per-layer files (strict: every layer file MUST exist) ─────────────
    for (int l = 0; l < num_layers; ++l) {
        auto fn = "layer_" + std::to_string(l) + ".arrow";
        auto layer_path = pinned_dir / fn;
        if (!fs::exists(layer_path)) {
            log::error("[load_pinned] required layer file missing: " + fn);
            pw.Free();
            return cudaErrorInvalidValue;
        }
        auto tensors_res = load_arrow_file(layer_path.string());
        if (!tensors_res.ok()) {
            log::error("[load_pinned] ERROR loading " + fn + ": "
                       + tensors_res.status().ToString());
            pw.Free();
            return cudaErrorInvalidValue;
        }
        auto &lt = *tensors_res;
        auto &lw = pw.layers[l];

        lw.compress_ratio = compress_ratios[l];
        lw.rope_theta = (lw.compress_ratio > 0) ? 160000.0f : 10000.0f;
        lw.comp_coff = (lw.compress_ratio == 4) ? 2 : 1;

        std::vector<ArenaJob> jobs = {
            {find_tensor_substr(lt, "ffn_norm.weight"), &lw.d_ffn_norm, true},
            {find_tensor_substr(lt, "ffn.gate.weight"),
             &lw.d_gate_weight, true},
            {find_tensor_suffix(lt, "attn.q_norm.weight"),
             &lw.d_q_norm, true},
            {find_tensor_suffix(lt, "attn.kv_norm.weight"),
             &lw.d_kv_norm, true},
            {find_tensor_suffix(lt, "attn_norm.weight"),
             &lw.d_attn_norm, true},
        };

        // CSA Compressor (ratio>0): wkv/wgate BF16, ape F32, norm BF16.
        if (lw.compress_ratio > 0) {
            jobs.push_back({find_tensor_suffix(lt,
                 "attn.compressor.wkv.weight"), &lw.d_comp_wkv, true});
            jobs.push_back({find_tensor_suffix(lt,
                 "attn.compressor.wgate.weight"), &lw.d_comp_wgate, true});
            jobs.push_back({find_tensor_suffix(lt, "attn.compressor.ape"),
                            &lw.d_comp_ape, false});
            jobs.push_back({find_tensor_suffix(lt,
                 "attn.compressor.norm.weight"), &lw.d_comp_norm, true});

            // CSA Indexer (ratio==4 only).
            if (lw.compress_ratio == 4) {
                auto *idx_w = find_tensor_suffix(lt,
                    "attn.indexer.wq_b.weight");
                auto *idx_s = find_tensor_suffix(lt,
                    "attn.indexer.wq_b.scale");
                if (idx_w) {
                    if (!idx_s) {
                        log::error("[load_pinned] indexer wq_b scale "
                                   "missing in " + fn);
                        pw.Free();
                        return cudaErrorInvalidValue;
                    }
                    jobs.push_back({idx_w, &lw.d_idx_wq_b, false,
                                    &lw.idx_wq_b_n, &lw.idx_wq_b_rows,
                                    &lw.idx_wq_b_cols});
                    jobs.push_back({idx_s, &lw.d_idx_wq_b_scales, false});
                    lw.idx_wq_b_rows = 8192;   // canonical geometry
                    lw.idx_wq_b_cols = 1024;
                }
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.weights_proj.weight"),
                     &lw.d_idx_weights_proj, true});
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.compressor.wkv.weight"),
                     &lw.d_idx_comp_wkv, true});
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.compressor.wgate.weight"),
                     &lw.d_idx_comp_wgate, true});
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.compressor.ape"),
                     &lw.d_idx_comp_ape, false});
                jobs.push_back({find_tensor_suffix(lt,
                     "attn.indexer.compressor.norm.weight"),
                     &lw.d_idx_comp_norm, true});
            }
        }

        // MLA attention weights (F8_E4M3 + F8_E8M0 scales, strict pairing).
        const std::pair<const char *, const char *> attn_pairs[] = {
            {"attn.wq_a.weight", "attn.wq_a.scale"},
            {"attn.wq_b.weight", "attn.wq_b.scale"},
            {"attn.wkv.weight",  "attn.wkv.scale"},
            {"attn.wo_a.weight", "attn.wo_a.scale"},
            {"attn.wo_b.weight", "attn.wo_b.scale"},
        };
        const std::tuple<const uint8_t **, const uint8_t **,
                         int64_t *, int64_t *, int *, int *> attn_fields[] = {
            {&lw.d_wq_a, &lw.d_wq_a_scales, &lw.wq_a_n, &lw.wq_a_s_n,
             &lw.wq_a_rows, &lw.wq_a_cols},
            {&lw.d_wq_b, &lw.d_wq_b_scales, &lw.wq_b_n, &lw.wq_b_s_n,
             &lw.wq_b_rows, &lw.wq_b_cols},
            {&lw.d_wkv,  &lw.d_wkv_scales,  &lw.wkv_n,  &lw.wkv_s_n,
             &lw.wkv_rows, &lw.wkv_cols},
            {&lw.d_wo_a, &lw.d_wo_a_scales, &lw.wo_a_n, &lw.wo_a_s_n,
             &lw.wo_a_rows, &lw.wo_a_cols},
            {&lw.d_wo_b, &lw.d_wo_b_scales, &lw.wo_b_n, &lw.wo_b_s_n,
             &lw.wo_b_rows, &lw.wo_b_cols},
        };
        for (int p = 0; p < 5; ++p) {
            auto *wt = find_tensor_suffix(lt, attn_pairs[p].first);
            if (!wt) continue;   // absent attention slot stays legal
            auto *st = find_tensor_suffix(lt, attn_pairs[p].second);
            if (!st) {
                log::error("[load_pinned] scale missing for '"
                           + std::string(attn_pairs[p].first) + "' in "
                           + fn);
                pw.Free();
                return cudaErrorInvalidValue;
            }
            auto [dw, dsc, wn, sn, rw, cw] = attn_fields[p];
            jobs.push_back({wt, reinterpret_cast<void **>(
                                    const_cast<uint8_t **>(dw)), false,
                            wn, rw, cw});
            jobs.push_back({st, reinterpret_cast<void **>(
                                    const_cast<uint8_t **>(dsc)), false,
                            sn});
        }

        // Attention sink + Hyper-Connections (FP32).
        jobs.push_back({find_tensor_suffix(lt, "attn.attn_sink"),
                        &lw.d_attn_sink, false});
        for (auto [suffix, dst] :
             std::initializer_list<std::pair<const char *, void **>>{
                 {"hc_attn_fn", reinterpret_cast<void **>(&lw.d_hc_attn_fn)},
                 {"hc_attn_base",
                  reinterpret_cast<void **>(&lw.d_hc_attn_base)},
                 {"hc_attn_scale",
                  reinterpret_cast<void **>(&lw.d_hc_attn_scale)},
                 {"hc_ffn_fn", reinterpret_cast<void **>(&lw.d_hc_ffn_fn)},
                 {"hc_ffn_base",
                  reinterpret_cast<void **>(&lw.d_hc_ffn_base)},
                 {"hc_ffn_scale",
                  reinterpret_cast<void **>(&lw.d_hc_ffn_scale)}}) {
            jobs.push_back({find_tensor_suffix(lt, suffix), dst, false});
        }

        cudaError_t le = UploadLayerJobs(std::move(jobs), lw, stream, ring,
                                         rb);
        if (le != cudaSuccess) {
            pw.Free();
            return le;
        }

        // Router selection bias (F32) + hash table (I64) — host pinned.
        auto *gate_bias = find_tensor_substr(lt, "ffn.gate.bias");
        if (gate_bias && gate_bias->shape.size() == 1 &&
            gate_bias->dtype == "F32") {
            int n = static_cast<int>(gate_bias->shape[0]);
            if (cudaMallocHost(&lw.h_gate_bias,
                               static_cast<size_t>(n) * sizeof(float))
                == cudaSuccess) {
                std::memcpy(lw.h_gate_bias, gate_bias->data.data(),
                            static_cast<size_t>(n) * sizeof(float));
                lw.gate_bias_n = n;
            }
        }
        auto *tid2eid = find_tensor_substr(lt, "ffn.gate.tid2eid");
        if (tid2eid && tid2eid->shape.size() == 2 &&
            tid2eid->dtype == "I64") {
            int64_t numel = tid2eid->shape[0] * tid2eid->shape[1];
            if (cudaMallocHost(&lw.h_tid2eid,
                               static_cast<size_t>(numel) * sizeof(int64_t))
                == cudaSuccess) {
                std::memcpy(lw.h_tid2eid, tid2eid->data.data(),
                            static_cast<size_t>(numel) * sizeof(int64_t));
                lw.tid2eid_vocab = static_cast<int>(tid2eid->shape[0]);
                lw.tid2eid_topk  = static_cast<int>(tid2eid->shape[1]);
            }
        }
    }
```

Shared experts section (replacing old lines 701-767): keep the structure but make every failure fatal and check statuses:

```cpp
    // Shared experts: NVFP4-packed metadata from pinned/shared_experts.arrow.
    // Absent file stays legal; present-but-broken file is fatal.
    {
        auto shared_path = pinned_dir / "shared_experts.arrow";
        if (fs::exists(shared_path)) {
            auto mmap_res = arrow::io::MemoryMappedFile::Open(
                shared_path.string(), arrow::io::FileMode::READ);
            if (!mmap_res.ok()) {
                log::error("[load_pinned] shared_experts open failed: "
                           + mmap_res.status().ToString());
                pw.Free();
                return cudaErrorInvalidValue;
            }
            SharedExpertDb db;
            auto st = load_shared_experts_metadata(*mmap_res, db);
            if (!st.ok()) {
                log::error("[load_pinned] shared_experts metadata: "
                           + st.ToString());
                pw.Free();
                return cudaErrorInvalidValue;
            }
            for (int l = 0; l < db.num_layers() &&
                            l < static_cast<int>(pw.layers.size()); ++l) {
                auto &lw = pw.layers[l];
                auto upload = [&](const SharedExpertDescriptor &desc,
                                  uint8_t *&d_buf, uint8_t *&d_scales,
                                  float &gs, int64_t &nbytes) -> cudaError_t {
                    nbytes = desc.packed_bytes;
                    if (nbytes <= 0) return cudaSuccess;
                    cudaError_t ue = cudaMalloc(
                        &d_buf, static_cast<size_t>(desc.packed_bytes));
                    if (ue != cudaSuccess) return ue;
                    rb += static_cast<std::uint64_t>(desc.packed_bytes);
                    ue = ring.UploadRaw(desc.host_ptr, d_buf,
                                        static_cast<size_t>(desc.packed_bytes),
                                        stream, rb);
                    if (ue != cudaSuccess) return ue;
                    if (desc.scale_bytes > 0) {
                        ue = cudaMalloc(&d_scales,
                                        static_cast<size_t>(desc.scale_bytes));
                        if (ue != cudaSuccess) return ue;
                        rb += static_cast<std::uint64_t>(desc.scale_bytes);
                        ue = ring.UploadRaw(desc.scale_host_ptr, d_scales,
                                            static_cast<size_t>(desc.scale_bytes),
                                            stream, rb);
                        if (ue != cudaSuccess) return ue;
                    }
                    gs = desc.global_scale;
                    return cudaSuccess;
                };
                cudaError_t ue = upload(db.w1[l], lw.d_shared_w1,
                                        lw.d_shared_w1_scales,
                                        lw.shared_w1_global_scale,
                                        lw.shared_w1_bytes);
                if (ue == cudaSuccess)
                    ue = upload(db.w2[l], lw.d_shared_w2,
                                lw.d_shared_w2_scales,
                                lw.shared_w2_global_scale,
                                lw.shared_w2_bytes);
                if (ue == cudaSuccess)
                    ue = upload(db.w3[l], lw.d_shared_w3,
                                lw.d_shared_w3_scales,
                                lw.shared_w3_global_scale,
                                lw.shared_w3_bytes);
                if (ue != cudaSuccess) {
                    log::error("[load_pinned] shared expert upload failed "
                               "(layer " + std::to_string(l) + "): "
                               + cudaGetErrorString(ue));
                    pw.Free();
                    return ue;
                }
            }
        }
    }
```

Final block (replacing old lines 769-770):

```cpp
    pw.resident_bytes = rb;

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    const double gib =
        static_cast<double>(rb) / (1024.0 * 1024.0 * 1024.0);
    const double gibs = (ms > 0.0) ? (gib / (ms / 1000.0)) : 0.0;
    log::info("[load_pinned] pinned weights: " + std::to_string(rb >> 20)
              + " MiB in " + std::to_string(static_cast<int>(ms + 0.5))
              + " ms (" + std::to_string(gibs) + " GiB/s)");

    return cudaSuccess;
}
```

Note: the debug logging blocks (first-10-global-tensors hex dump, per-layer first-5 names, layer-0/layer-2 info logs) are dropped — the timing summary replaces them.

- [ ] **Step 6: Build and verify GREEN**

Run:
```bash
cmake --build --preset linux-debug
ctest --preset linux-debug -R "deepseek_loader|deepseek_load_fast" --output-on-failure
```
Expected: ALL PASS — including the four Task 2 cases (values/topology/missing-file) and the two new cases (containment/idempotency, malformed tensor).

If `"Fast pinned load ..."` fails on `layers[1].d_ffn_norm`: check `find_tensor_substr` matched the right record — fixture names use `layers.<l>.ffn_norm.weight` whose suffix `.ffn_norm.weight` matches.

- [ ] **Step 7: Full DeepSeek + GLM regression**

Run:
```bash
ctest --preset linux-debug -R deepseek --output-on-failure
ctest --preset linux-debug -R glm --output-on-failure
```
Expected: deepseek suite green (real-checkpoint tests SKIP with exit 4 where no checkpoint present), glm suite green (regression guard for the Task 1 refactor).

- [ ] **Step 8: Commit Tasks 2+3 together**

```bash
git add tests/deepseek_test_fixture.h tests/test_deepseek_loader.cpp \
        tests/CMakeLists.txt \
        libs/lsxcommon/src/deepseek_inference_load.cuh \
        libs/lsxcommon/src/deepseek_inference_load.cu
git commit -m "feat: DeepSeek fast pinned-weight load (per-layer arena, staging ring, strict errors, topology-driven layers)"
```

---

### Task 4: Model-owned weights — load once in `Load()`, remove per-call reload

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_v4_flash_model.h` (members)
- Modify: `libs/lsxcommon/src/deepseek_v4_flash_model.cpp` (Load/Unload/Infer/Benchmark)
- Modify: `libs/lsxcommon/src/deepseek_inference_pipeline.cu:139-146, 233, 287-295, 832` (signature + local pw removal)

**Interfaces:**
- Consumes: `load_pinned_weights` (Task 3), `gpu::PinnedWeights`.
- Produces: `RunInferenceModel(const std::string&, const std::vector<int32_t>&, int, ArrowTokenizerDb&, gpu::PinnedWeights&, ISampler& = DefaultSampler())` — callers must pass a loaded `PinnedWeights`. `DeepSeekV4FlashModel::Infer/Benchmark` pass `pw_`.

- [ ] **Step 1: Change the pipeline signature and drop the per-call load**

In `libs/lsxcommon/src/deepseek_inference_pipeline.cu`:

(a) Change the definition signature (:139-145) to:

```cpp
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db,
    gpu::PinnedWeights &pw,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler())
```

(b) DELETE the local `PinnedWeights pw;` declaration (:233).

(c) REPLACE the load block (:287-295) with a guard (weights arrive pre-loaded):

```cpp
    // 5. Pinned weights are loaded ONCE by the owning model
    //    (DeepSeekV4FlashModel::Load) and handed in here.
    if (!pw.d_embed || pw.layers.empty()) {
        result.error_msg = "pinned weights not loaded";
        goto cleanup;
    }
```

(The old `INFER_CHECK(cudaDeviceSynchronize());` disappears — `load_pinned_weights` already ends synchronized via its StreamDrain.)

(d) In the `cleanup:` block, DELETE the line `pw.Free();` (:832) — ownership belongs to the model now. Keep every other cleanup line.

(e) Keep the later uses of `pw` (vocab_size guard, arena sizing, metrics) untouched — they now refer to the parameter.

- [ ] **Step 2: Update the model header**

In `libs/lsxcommon/src/deepseek_v4_flash_model.h`:

```cpp
#pragma once

#include <cuda_runtime.h>

#include <string>

#include "lsxcommon/model_base.h"
#include "deepseek_inference_load.cuh"

namespace lsxcommon {

class DeepSeekV4FlashModel final : public ModelBase {
public:
    static const std::string kType;
    explicit DeepSeekV4FlashModel(ModelInitConfig cfg);
    ~DeepSeekV4FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return kType; }

private:
    gpu::PinnedWeights pw_;
    cudaStream_t stream_ = nullptr;
};

} // namespace lsxcommon
```

- [ ] **Step 3: Update the model implementation**

In `libs/lsxcommon/src/deepseek_v4_flash_model.cpp`:

(a) Update the forward declaration (:18-24):

```cpp
InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db,
    gpu::PinnedWeights &pw,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler());
```

(b) Replace `Load()`:

```cpp
bool DeepSeekV4FlashModel::Load()
{
    if (!BeginLoad(cfg_.model_path)) return false;
    auto topo = gpu::load_topology(cfg_.model_path);
    if (!topo.error_msg.empty()) {
        lsxcommon::log::error(absl::StrCat(
            "[deepseek-model] topology error: ", topo.error_msg));
        return false;
    }
    vram_ = topo.vram;

    // Fast-load the pinned weights ONCE (GLM-parity ownership model):
    // Infer/Benchmark reuse them instead of paying full load cost per call.
    if (!stream_) {
        if (cudaStreamCreate(&stream_) != cudaSuccess) {
            lsxcommon::log::error("[deepseek-model] stream create failed");
            return false;
        }
    }
    if (gpu::load_pinned_weights(cfg_.model_path, pw_, stream_)
        != cudaSuccess) {
        lsxcommon::log::error("[deepseek-model] pinned weight load failed");
        pw_.Free();
        return false;
    }
    FinishLoad();
    return true;
}
```

(c) Replace `Infer()` and `Benchmark()` call sites:

```cpp
gpu::InferenceResult DeepSeekV4FlashModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    return gpu::RunInferenceModel(cfg_.model_path, req.input_ids,
                                  req.max_gen_tokens, *tok_db_, pw_);
}
```

and in `Benchmark()`:

```cpp
    auto res = gpu::RunInferenceModel(cfg_.model_path, prompt, num_tokens,
                                      *tok_db_, pw_);
```

(d) Replace `Unload()`:

```cpp
void DeepSeekV4FlashModel::Unload()
{
    pw_.Free();
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    tok_db_.reset();
    loaded_ = false;
}
```

- [ ] **Step 4: Build and run regression**

Run:
```bash
cmake --build --preset linux-debug
ctest --preset linux-debug -R "deepseek|glm|integrated" --output-on-failure
```
Expected: everything compiles; all non-checkpoint DeepSeek tests PASS; `test_deepseek_prompt_output` / `test_deepseek_selfcheck` / integrated-pipeline SKIP-or-PASS depending on checkpoint presence (SKIP_RETURN_CODE 4). GLM suites stay green.

Also smoke-check any direct `RunInferenceModel` callers compile: `grep -rn "RunInferenceModel" apps libs --include=*.cpp --include=*.cu` — every call site gains the `pw` argument (expected: only deepseek_v4_flash_model.cpp calls it).

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/src/deepseek_v4_flash_model.h \
        libs/lsxcommon/src/deepseek_v4_flash_model.cpp \
        libs/lsxcommon/src/deepseek_inference_pipeline.cu
git commit -m "feat: model-owned DeepSeek pinned weights (single load in Load(), no per-Infer reload or device sync)"
```

---

## Final verification

- [ ] Full relevant suites:

```bash
cmake --build --preset linux-debug
ctest --preset linux-debug -R "deepseek|glm" --output-on-failure
```

Expected: all green (checkpoint-gated tests may SKIP via exit 4). Then commit any remaining stray changes (there should be none).

## Self-review notes

- Spec coverage: shared primitives (Task 1), fast-load arena/ring/strict/topology (Tasks 2-3), model-owned weights + no post-load `cudaDeviceSynchronize` (Task 4). Qwen unification explicitly out of scope.
- Type consistency: `ArenaJob.dst` is `void**`; MLA const-pointer members use `reinterpret_cast` through `const_cast<uint8_t**>` exactly as the old `Free()` does for `cudaFree`.
- Known risk: `attn_pairs`/`attn_fields` parallel arrays rely on positional pairing — verified against `LayerWeights` member list (`deepseek_inference_load.cuh:271-282`).
