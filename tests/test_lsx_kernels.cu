// TDD harness for the shared lsx_kernels_*.cuh library (Phase A unification).
// Each cluster task adds red/green equivalence tests here: a verbatim
// reference copy of the kernel being replaced vs. the unified template.
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#include "lsx_kernels_quant.cuh"

namespace {

__host__ inline void RequireCudaOk(cudaError_t e) {
    REQUIRE(e == cudaSuccess);
}

template <typename T>
std::vector<T> ToHost(const T* dev, size_t n) {
    std::vector<T> h(n);
    RequireCudaOk(
        cudaMemcpy(h.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

template <typename T>
void RequireDeviceEqual(const T* a, const T* b, size_t n) {
    auto ha = ToHost(a, n);
    auto hb = ToHost(b, n);
    REQUIRE(std::memcmp(ha.data(), hb.data(), n * sizeof(T)) == 0);
}

// Deterministic LCG fill so failures are reproducible.
inline uint32_t LcgNext(uint32_t& s) { return s = s * 1664525u + 1013904223u; }

} // namespace

TEST_CASE("harness smoke", "[lsxk]") {
    int* d = nullptr;
    RequireCudaOk(cudaMalloc(&d, 4));
    RequireCudaOk(cudaMemset(d, 0, 4));
    cudaFree(d);
    REQUIRE(true);
}

// ── Task 2: quant cluster references (verbatim ports) ──────────────────────
namespace ref {

// Elementwise NVFP4 dequant, kimi_inference.cu body, __half output.
__global__ void nvfp4_dequant_half(const uint8_t* __restrict__ packed,
                                   const uint8_t* __restrict__ scales,
                                   float global_scale, __half* __restrict__ out,
                                   int64_t numel) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= numel) return;
    const uint8_t byte = packed[i >> 1];
    const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
    out[i] = __float2half(lsxcommon::gpu::nvfp4::E2M1Value(nib) *
                          lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(scales[i >> 4]) *
                          global_scale);
}

// gemma_inference.cu body: bf16 via __float2bfloat16_rn.
__global__ void nvfp4_dequant_bf16(const uint8_t* __restrict__ packed,
                                   const uint8_t* __restrict__ scales,
                                   float global_scale,
                                   __nv_bfloat16* __restrict__ out,
                                   int64_t numel) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= numel) return;
    const uint8_t byte = packed[i >> 1];
    const int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
    out[i] = __float2bfloat16_rn(lsxcommon::gpu::nvfp4::E2M1Value(nib) *
                                 lsxcommon::gpu::nvfp4::E4M3ScaleToFloat(scales[i >> 4]) *
                                 global_scale);
}

// safetensors_gpu.cu dev_f8e4m3_to_float verbatim.
__host__ __device__ float f8e4m3(uint8_t b) {
    if (!b || b == 0x80) return 0.0f;
    if ((b & 0x7F) == 0x7F) {
        const uint32_t nan_bits = 0x7FC00000u;
        float nan_v;
        std::memcpy(&nan_v, &nan_bits, sizeof(nan_v));
        return nan_v;
    }
    int s = (b >> 7) & 1;
    int e = (b >> 3) & 0x0F;
    int m = b & 0x07;
    if (e == 0) {
        float v = static_cast<float>(m) * 0.001953125f;
        return s ? -v : v;
    }
    uint32_t f32 = (static_cast<uint32_t>(s) << 31)
                 | (static_cast<uint32_t>(e + 127 - 7) << 23)
                 | (static_cast<uint32_t>(m) << 20);
    float v;
    std::memcpy(&v, &f32, sizeof(v));
    return v;
}

// safetensors_gpu.cu dev_quantize_nvfp4 verbatim (nearest loop, ties lower).
__host__ __device__ uint8_t quantize_e2m1_nearest(float val) {
    if (val == 0.0f) return 0;
    int sign = val < 0 ? 1 : 0;
    float abs_val = fabsf(val);
    static const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int best = 0;
    float best_diff = fabsf(abs_val - levels[0]);
    for (int i = 1; i < 8; i++) {
        float d = fabsf(abs_val - levels[i]);
        if (d < best_diff) { best_diff = d; best = i; }
    }
    return static_cast<uint8_t>((sign << 3) | best);
}

// qwen_tc_gemm.cu QuantizeE2m1 verbatim (threshold chain, ties up).
__host__ __device__ uint8_t quantize_e2m1_tie_up(float v) {
    const bool neg = v < 0.0f;
    const float a = fabsf(v);
    uint8_t mag;
    if (a < 0.25f) mag = 0;
    else if (a < 0.75f) mag = 1;
    else if (a < 1.25f) mag = 2;
    else if (a < 1.75f) mag = 3;
    else if (a < 2.5f) mag = 4;
    else if (a < 3.5f) mag = 5;
    else if (a < 5.0f) mag = 6;
    else mag = 7;
    return neg ? static_cast<uint8_t>(mag | 8) : mag;
}

} // namespace ref

namespace {

constexpr int kDequantN = 4096;

void FillNvfp4Input(std::vector<uint8_t>& packed, std::vector<uint8_t>& scales,
                    float& gs) {
    uint32_t s = 12345;
    packed.resize(kDequantN / 2);
    scales.resize(kDequantN / 16);
    for (auto& b : packed) b = static_cast<uint8_t>(LcgNext(s) & 0xFF);
    for (auto& b : scales) b = static_cast<uint8_t>(LcgNext(s) & 0xFF);
    gs = 0.25f + static_cast<float>(LcgNext(s) % 100) / 1000.f;
}

template <typename OutT>
void RunBoth(void (*ref_kernel)(const uint8_t*, const uint8_t*, float, OutT*, int64_t),
             float gs) {
    std::vector<uint8_t> packed, scales;
    FillNvfp4Input(packed, scales, gs);
    uint8_t *d_packed, *d_scales;
    OutT *d_ref, *d_uni;
    cudaMalloc(&d_packed, packed.size());
    cudaMalloc(&d_scales, scales.size());
    cudaMalloc(&d_ref, kDequantN * sizeof(OutT));
    cudaMalloc(&d_uni, kDequantN * sizeof(OutT));
    cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice);

    constexpr int blk = 256;
    const int grid = (kDequantN + blk - 1) / blk;
    ref_kernel<<<grid, blk>>>(d_packed, d_scales, gs, d_ref, kDequantN);
    lsxcommon::kernels::Nvfp4DequantKernel<OutT>
        <<<grid, blk>>>(d_packed, d_scales, gs, d_uni, kDequantN);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    RequireDeviceEqual(d_ref, d_uni, kDequantN);

    cudaFree(d_packed); cudaFree(d_scales); cudaFree(d_ref); cudaFree(d_uni);
}

} // namespace

TEST_CASE("nvfp4 dequant unified matches elementwise refs", "[lsxk][quant]") {
    RunBoth<__half>(ref::nvfp4_dequant_half, 0.42f);
    RunBoth<__nv_bfloat16>(ref::nvfp4_dequant_bf16, 0.17f);
}

TEST_CASE("f8e4m3/e8m0 decoders match over all bytes", "[lsxk][quant]") {
    for (uint32_t b = 0; b < 256; ++b) {
        const float got = lsxcommon::kernels::F8E4m3ToFloat(static_cast<uint8_t>(b));
        const float want = ref::f8e4m3(static_cast<uint8_t>(b));
        if (std::isnan(want)) {
            REQUIRE(std::isnan(got));
        } else {
            REQUIRE(got == want);
        }
        // e8m0: canonical shift decode.
        float expected = (b == 0) ? 0.0f : [&] {
            uint32_t bits = b << 23; float v; std::memcpy(&v, &bits, sizeof(v)); return v;
        }();
        REQUIRE(lsxcommon::kernels::E8m0ToFloat(static_cast<uint8_t>(b)) == expected);
    }
}

TEST_CASE("e2m1 encode ties: tie-up and nearest both replicated", "[lsxk][quant]") {
    // Dense sweep including exact midpoints +- ulp.
    for (int i = -13000; i <= 13000; ++i) {
        float v = static_cast<float>(i) / 1000.0f;
        REQUIRE(lsxcommon::kernels::QuantizeE2m1TieUp(v) ==
                ref::quantize_e2m1_tie_up(v));
        REQUIRE(lsxcommon::kernels::QuantizeE2m1Nearest(v) ==
                ref::quantize_e2m1_nearest(v));
        REQUIRE(lsxcommon::kernels::QuantizeE2m1Nearest(-v) ==
                ref::quantize_e2m1_nearest(-v));
    }
}
