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
#include "nvfp4_scale.h"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_act.cuh"
#include "lsx_kernels_norm.cuh"
#include "glm_moe_device_math.h"

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

// Elementwise NVFP4 dequant, kimi_moe_inference.cu body, __half output.
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

// gemma_moe_inference.cu body: bf16 via __float2bfloat16_rn.
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

// lsx_tc_gemm.cu QuantizeE2m1 verbatim (threshold chain, ties up).
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

// ── Coverage: pow2-E4M3 scale ENCODER edge classes (docs §1 deviation) ──────
// Reference: nvidia-cuda-docs MCP (see docs/NVIDIA_DOCS.md) — ceil-rounding pow2-only
// payload, e4 clamped to [1,14], exact powers of two map onto themselves,
// subnormals clamp to the minimum normal scale 2^-6 (byte 0x08).
TEST_CASE("FloatToE4m3P2 edge classes, identity and ceil property",
          "[lsxk][quant][scale]")
{
    using lsxcommon::nvfp4::FloatToE4m3P2;
    using lsxcommon::nvfp4::E4m3P2ToFloat;

    // Zero (incl. -0) -> reserved zero byte 0x00.
    REQUIRE(FloatToE4m3P2(0.0f) == 0x00);
    REQUIRE(FloatToE4m3P2(-0.0f) == 0x00);

    // Float subnormals clamp to the minimum normal scale 2^-6 (byte 0x08).
    REQUIRE(FloatToE4m3P2(1e-45f) == 0x08);
    REQUIRE(FloatToE4m3P2(1e-40f) == 0x08);
    // Normal values below 2^-6 also clamp to 2^-6 (e4 floor 1).
    REQUIRE(FloatToE4m3P2(0.0078125f) == 0x08);   // 2^-7

    // Exact powers of two map onto themselves (2^-6 .. 2^7).
    const struct { float v; uint8_t b; } pow2[] = {
        {0.015625f, 0x08}, {0.125f, 0x20}, {1.0f, 0x38},
        {2.0f, 0x40}, {64.0f, 0x68}, {128.0f, 0x70},
    };
    for (const auto& c : pow2) {
        INFO("v=" << c.v);
        REQUIRE(FloatToE4m3P2(c.v) == c.b);
        REQUIRE(E4m3P2ToFloat(c.b) == c.v);   // identity
    }

    // Ceil rounding: smallest power of two >= s.
    REQUIRE(FloatToE4m3P2(0.1f) == 0x20);    // -> 2^-3
    REQUIRE(FloatToE4m3P2(1.5f) == 0x40);    // -> 2^1
    REQUIRE(FloatToE4m3P2(6.5f) == 0x50);    // -> 2^3
    REQUIRE(FloatToE4m3P2(300.0f) == 0x70);  // > 2^7 band -> ceiling clamp

    // e4 clamps to 14 (max scale 2^7); never enters the E4M3 NaN slot.
    REQUIRE(FloatToE4m3P2(256.0f) == 0x70);
    REQUIRE(FloatToE4m3P2(1e30f) == 0x70);

    // Property sweep: every finite positive input yields a pow2 scale that
    // never saturates (decoded >= s inside the representable band; decoded/2
    // < s), and the clamp band edges behave exactly as documented.
    uint32_t st = 20260910u;
    // Normals only via the exponent field (e in [-126, 127]); float
    // subnormals are covered by the explicit clamp cases above.
    for (int e = -126; e <= 127; ++e) {
        for (int mi = 0; mi < 4; ++mi) {
            const uint32_t mant = (mi == 0) ? 0u
                                : (mi == 1) ? 0x400000u
                                : (LcgNext(st) & 0x7FFFFFu);
            const uint32_t bits = (static_cast<uint32_t>(e + 127) << 23) | mant;
            float s;
            std::memcpy(&s, &bits, sizeof(s));
            const uint8_t b = FloatToE4m3P2(s);
            const float d = E4m3P2ToFloat(b);
            // Decoded scale is a power of two (mantissa field empty) or the
            // reserved zero (only for s == 0, impossible in this sweep).
            INFO("s=" << s << " byte=0x" << std::hex << int(b) << std::dec);
            REQUIRE(b != 0x00);
            REQUIRE(((b >> 3) & 0x0F) >= 1);
            REQUIRE(((b >> 3) & 0x0F) <= 14);
            REQUIRE((b & 0x07) == 0);
            const uint32_t dbits = [&] { uint32_t x; std::memcpy(&x, &d, sizeof(x)); return x; }();
            REQUIRE((dbits & 0x7FFFFFu) == 0);   // pow2 payload
            if (s <= 128.0f && s >= 0.015625f) {
                REQUIRE(d >= s);
                REQUIRE(d < 2.0f * s);   // ceil, not round-up-two
            } else if (s < 0.015625f) {
                REQUIRE(d == 0.015625f);   // 2^-6 floor
            } else {
                REQUIRE(d == 128.0f);      // 2^7 ceiling
            }
        }
    }
}

// Coverage: the clamped E8M0 decode flavor used by the routed-expert FFN
// paths (shared_experts.cu). Saturation slot 0xFF yields FLT_MAX (not +Inf)
// and 0x00 yields 0; everywhere else it agrees with the canonical shift
// decode E8m0ToFloat. (Deliberate deviation note: E8m0ToFloat decodes 0x00
// as 0.0f, whereas PTX ue8m0 would read 2^-127 — numerically indistinguishable
// in FP16 outputs; see docs/NVIDIA_DOCS.md (nvidia-cuda-docs MCP Part A §3).)
TEST_CASE("E8m0ClampedToFloat clamps the saturation slot", "[lsxk][quant][scale]")
{
    using lsxcommon::kernels::E8m0ClampedToFloat;
    using lsxcommon::kernels::E8m0ToFloat;

    REQUIRE(E8m0ClampedToFloat(0x00) == 0.0f);
    REQUIRE(E8m0ClampedToFloat(0xFF) == FLT_MAX);
    REQUIRE(E8m0ClampedToFloat(0x7F) == 1.0f);
    for (uint32_t b = 0; b <= 0xFE; ++b)
        REQUIRE(E8m0ClampedToFloat(static_cast<uint8_t>(b)) ==
                E8m0ToFloat(static_cast<uint8_t>(b)));
}

// Coverage: exhaustive E2M1 decode table against the PTX-spec value levels
// (sign(1) | exp(2, bias=1) | mant(1); levels {0, .5, 1, 1.5, 2, 3, 4, 6}),
// plus the device-side lo-nibble-first packing order via Nvfp4DequantKernel.
TEST_CASE("E2M1Value exhaustive table and lo-nibble packing order",
          "[lsxk][quant][e2m1]")
{
    using lsxcommon::gpu::nvfp4::E2M1Value;
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int nib = 0; nib < 16; ++nib) {
        const float v = E2M1Value(nib);
        INFO("nib=" << nib);
        REQUIRE(v == ((nib & 8) ? -levels[nib & 7] : levels[nib & 7]));
        if (nib == 8) REQUIRE(std::signbit(v));   // -0.0f keeps the sign bit
    }

    // Device packing order: byte 0x21 = low nibble 1 (element 0 -> 0.5),
    // high nibble 2 (element 1 -> 1.0); scale byte 0x38 = 2^0, gs = 1.
    const uint8_t hp[1] = {0x21}, hs[1] = {0x38};
    uint8_t *dp, *ds; __half *dout;
    REQUIRE(cudaMalloc(&dp, 1) == cudaSuccess);
    REQUIRE(cudaMalloc(&ds, 1) == cudaSuccess);
    REQUIRE(cudaMalloc(&dout, 2 * sizeof(__half)) == cudaSuccess);
    REQUIRE(cudaMemcpy(dp, hp, 1, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ds, hs, 1, cudaMemcpyHostToDevice) == cudaSuccess);
    lsxcommon::kernels::Nvfp4DequantKernel<__half><<<1, 32>>>(
        dp, ds, 1.0f, dout, 2);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    const auto out = ToHost(dout, 2);
    REQUIRE(out[0] == __float2half(0.5f));
    REQUIRE(out[1] == __float2half(1.0f));
    cudaFree(dp); cudaFree(ds); cudaFree(dout);
}

// ── Task 3: math cluster references (verbatim ports) ───────────────────────
namespace ref {

// qwen_inference.cu qwen_add_kernel verbatim.
__global__ void add_half(const __half* __restrict__ src,
                         __half* __restrict__ dst, int64_t n) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = __float2half(__half2float(dst[idx]) + __half2float(src[idx]));
}

// glm_moe_inference.cu GlmEmbedKernel verbatim.
__global__ void embed_half(const __half* __restrict__ embed,
                           const int32_t* __restrict__ ids,
                           __half* __restrict__ out, int seq, int hidden) {
    const int64_t total = static_cast<int64_t>(seq) * hidden;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int t = static_cast<int>(i / hidden);
    const int h = static_cast<int>(i - static_cast<int64_t>(t) * hidden);
    out[i] = embed[static_cast<int64_t>(ids[t]) * hidden + h];
}

// qwen_moe f2h verbatim.
__global__ void f2h(const float* __restrict__ src, __half* __restrict__ dst,
                    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

} // namespace ref

TEST_CASE("math kernels match refs", "[lsxk][math]") {
    constexpr int kN = 2048, kSeq = 4, kHid = 128;
    uint32_t st = 777;
    std::vector<uint16_t> raw(kN);
    for (auto& r : raw) r = static_cast<uint16_t>(LcgNext(st) & 0xFFFF);

    __half *d_a, *d_dst_ref, *d_dst_uni;
    cudaMalloc(&d_a, kN * 2);
    cudaMalloc(&d_dst_ref, kN * 2);
    cudaMalloc(&d_dst_uni, kN * 2);
    cudaMemcpy(d_a, raw.data(), kN * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_ref, d_a, kN * 2, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_dst_uni, d_a, kN * 2, cudaMemcpyDeviceToDevice);

    constexpr int blk = 256;
    const int grid = (kN + blk - 1) / blk;
    ref::add_half<<<grid, blk>>>(d_a, d_dst_ref, kN);
    lsxcommon::kernels::AddInPlaceKernel<__half>
        <<<grid, blk>>>(d_a, d_dst_uni, kN);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    RequireDeviceEqual(d_dst_ref, d_dst_uni, kN);

    // Embed gather.
    std::vector<uint16_t> table_raw(kSeq * kHid);
    for (auto& r : table_raw) r = static_cast<uint16_t>(LcgNext(st) & 0xFFFF);
    std::vector<int32_t> ids = {3, 0, 2, 1};
    __half *d_tab_ref, *d_tab_uni, *d_out_ref, *d_out_uni;
    int32_t* d_ids;
    cudaMalloc(&d_tab_ref, kSeq * kHid * 2);
    cudaMalloc(&d_tab_uni, kSeq * kHid * 2);
    cudaMemcpy(d_tab_ref, table_raw.data(), kSeq * kHid * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tab_uni, d_tab_ref, kSeq * kHid * 2, cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_ids, ids.size() * 4);
    cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_out_ref, kSeq * kHid * 2);
    cudaMalloc(&d_out_uni, kSeq * kHid * 2);
    ref::embed_half<<<(kSeq * kHid + blk - 1) / blk, blk>>>(
        d_tab_ref, d_ids, d_out_ref, kSeq, kHid);
    lsxcommon::kernels::EmbedGatherKernel<__half>
        <<<(kSeq * kHid + blk - 1) / blk, blk>>>(d_tab_uni, d_ids, d_out_uni,
                                                 kSeq, kHid);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    RequireDeviceEqual(d_out_ref, d_out_uni, kSeq * kHid);

    // F32 -> T convert.
    std::vector<float> fsrc(kN);
    for (auto& f : fsrc) f = static_cast<float>(LcgNext(st) % 2000) / 7.f - 100.f;
    float* d_f; __half *d_h_ref, *d_h_uni;
    cudaMalloc(&d_f, kN * 4);
    cudaMemcpy(d_f, fsrc.data(), kN * 4, cudaMemcpyHostToDevice);
    cudaMalloc(&d_h_ref, kN * 2);
    cudaMalloc(&d_h_uni, kN * 2);
    ref::f2h<<<grid, blk>>>(d_f, d_h_ref, kN);
    lsxcommon::kernels::F32toTKernel<__half>
        <<<grid, blk>>>(d_f, d_h_uni, kN);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    RequireDeviceEqual(d_h_ref, d_h_uni, kN);

    cudaFree(d_a); cudaFree(d_dst_ref); cudaFree(d_dst_uni);
    cudaFree(d_tab_ref); cudaFree(d_tab_uni); cudaFree(d_ids);
    cudaFree(d_out_ref); cudaFree(d_out_uni);
    cudaFree(d_f); cudaFree(d_h_ref); cudaFree(d_h_uni);
}

// ── Task 4: act cluster references (verbatim ports) ────────────────────────
namespace ref {

// qwen_moe_inference.cu qwen_moe_swiglu_kernel verbatim (division, Precise).
__global__ void swiglu_div_precise(const __half* __restrict__ g,
                                   const __half* __restrict__ u,
                                   __half* __restrict__ y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = __half2float(g[i]);
    y[i] = __float2half(gv / (1.f + expf(-gv)) * __half2float(u[i]));
}

// glm_moe_inference.cu GlmSwigluClampKernel verbatim (Glibc + clamp).
__global__ void swiglu_glibc_clamp(__half* __restrict__ gate,
                                   const __half* __restrict__ up,
                                   int n, float limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    if (g > limit) g = limit;
    if (u > limit) u = limit;
    if (u < -limit) u = -limit;
    float s = g / (1.0f + lsxcommon::glm::device::GlmExpfGlibc(-g));
    gate[i] = __float2half(s * u);
}

// deepseek swiglu_clamp_kernel verbatim (sig form, Precise).
__global__ void swiglu_sig_precise(__half* __restrict__ d_gate_vec,
                                   const __half* __restrict__ d_up_vec,
                                   int n, float limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(d_gate_vec[i]);
    float u = __half2float(d_up_vec[i]);
    if (g > limit) g = limit;
    if (u >  limit) u =  limit;
    if (u < -limit) u = -limit;
    float sig = 1.0f / (1.0f + expf(-g));
    d_gate_vec[i] = __float2half(sig * g * u);
}

} // namespace ref

namespace {
void FillHalfRand(__half* d, size_t n, uint32_t seed) {
    std::vector<uint16_t> raw(n);
    for (auto& r : raw) r = static_cast<uint16_t>(LcgNext(seed) & 0xFFFF);
    cudaMemcpy(d, raw.data(), n * 2, cudaMemcpyHostToDevice);
}
} // namespace

TEST_CASE("swiglu flavors match refs bit-for-bit", "[lsxk][act]") {
    constexpr int kN = 1024;
    constexpr int blk = 256;
    const int grid = (kN + blk - 1) / blk;

    // Division form, Precise, no clamp.
    {
        __half *d_g, *d_u, *d_r, *d_x;
        cudaMalloc(&d_g, kN * 2); cudaMalloc(&d_u, kN * 2);
        cudaMalloc(&d_r, kN * 2); cudaMalloc(&d_x, kN * 2);
        FillHalfRand(d_g, kN, 42); cudaMemcpy(d_u, d_g, kN * 2, cudaMemcpyDeviceToDevice);
        ref::swiglu_div_precise<<<grid, blk>>>(d_g, d_u, d_r, kN);
        lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::Precise, false>
            <<<grid, blk>>>(d_g, d_u, d_x, kN, 0.f);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        RequireDeviceEqual(d_r, d_x, kN);
        cudaFree(d_g); cudaFree(d_u); cudaFree(d_r); cudaFree(d_x);
    }
    // Glibc clamp (in-place out==gate).
    {
        __half *d_gr, *d_ur, *d_gx, *d_ux;
        cudaMalloc(&d_gr, kN * 2); cudaMalloc(&d_ur, kN * 2);
        cudaMalloc(&d_gx, kN * 2); cudaMalloc(&d_ux, kN * 2);
        FillHalfRand(d_gr, kN, 43); cudaMemcpy(d_ur, d_gr, kN * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_gx, d_gr, kN * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_ux, d_ur, kN * 2, cudaMemcpyDeviceToDevice);
        ref::swiglu_glibc_clamp<<<grid, blk>>>(d_gr, d_ur, kN, 10.f);
        lsxcommon::kernels::SwigluKernel<__half, lsxcommon::kernels::ExpFlavor::Glibc, true>
            <<<grid, blk>>>(d_gx, d_ux, d_gx, kN, 10.f);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        RequireDeviceEqual(d_gr, d_gx, kN);
        cudaFree(d_gr); cudaFree(d_ur); cudaFree(d_gx); cudaFree(d_ux);
    }
    // Sig form, Precise, clamp.
    {
        __half *d_gr, *d_ur, *d_gx, *d_ux;
        cudaMalloc(&d_gr, kN * 2); cudaMalloc(&d_ur, kN * 2);
        cudaMalloc(&d_gx, kN * 2); cudaMalloc(&d_ux, kN * 2);
        FillHalfRand(d_gr, kN, 44); cudaMemcpy(d_ur, d_gr, kN * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_gx, d_gr, kN * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_ux, d_ur, kN * 2, cudaMemcpyDeviceToDevice);
        ref::swiglu_sig_precise<<<grid, blk>>>(d_gr, d_ur, kN, 7.f);
        lsxcommon::kernels::SwigluSigKernel<__half, lsxcommon::kernels::ExpFlavor::Precise, true>
            <<<grid, blk>>>(d_gx, d_ux, d_gx, kN, 7.f);
        REQUIRE(cudaGetLastError() == cudaSuccess);
        RequireDeviceEqual(d_gr, d_gx, kN);
        cudaFree(d_gr); cudaFree(d_ur); cudaFree(d_gx); cudaFree(d_ux);
    }
}

// ── Task 5: norm cluster references ────────────────────────────────────────
namespace ref {

// gemma_rmsnorm_plain_kernel verbatim (tree reduction, no weight).
__global__ void rms_tree_bf16_plain(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ w,
                                    __nv_bfloat16* __restrict__ out, int rows,
                                    int hidden, float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16* xr = x + static_cast<int64_t>(row) * hidden;
    __nv_bfloat16* orow = out + static_cast<int64_t>(row) * hidden;
    extern __shared__ float squares[];
    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = __bfloat162float(xr[j]);
        sum += v * v;
    }
    squares[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) squares[threadIdx.x] += squares[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(squares[0] / static_cast<float>(hidden) + eps);
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        orow[j] = __float2bfloat16(__bfloat162float(xr[j]) * inv);
}

} // namespace ref

TEST_CASE("rmsnorm tree kernel matches refs", "[lsxk][norm]") {
    constexpr int kRows = 3, kHid = 256;
    constexpr int blk = 128;
    std::vector<uint16_t> xr(kRows * kHid), wr(kHid);
    uint32_t st = 99;
    for (auto& r : xr) r = static_cast<uint16_t>(LcgNext(st) & 0xFFFF);
    for (auto& r : wr) r = static_cast<uint16_t>(LcgNext(st) & 0xFFFF);

    __nv_bfloat16 *d_xr, *d_xu, *d_wr, *d_wu, *d_or, *d_ou;
    cudaMalloc(&d_xr, xr.size() * 2); cudaMalloc(&d_xu, xr.size() * 2);
    cudaMemcpy(d_xr, xr.data(), xr.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_xu, d_xr, xr.size() * 2, cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_wr, wr.size() * 2); cudaMalloc(&d_wu, wr.size() * 2);
    cudaMemcpy(d_wr, wr.data(), wr.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_wu, d_wr, wr.size() * 2, cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_or, xr.size() * 2); cudaMalloc(&d_ou, xr.size() * 2);

    ref::rms_tree_bf16_plain<<<kRows, blk, blk * sizeof(float)>>>(
        d_xr, d_wr, d_or, kRows, kHid, 1e-5f);
    lsxcommon::kernels::RmsNormTreeKernel<__nv_bfloat16,
        lsxcommon::kernels::NormWeight::None>
        <<<kRows, blk, blk * sizeof(float) + kHid * int(sizeof(__nv_bfloat16))>>>(
            d_xu, d_wu, d_ou, kRows, kHid, 1e-5f);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    RequireDeviceEqual(d_or, d_ou, kRows * kHid);

    cudaFree(d_xr); cudaFree(d_xu); cudaFree(d_wr); cudaFree(d_wu);
    cudaFree(d_or); cudaFree(d_ou);
}
