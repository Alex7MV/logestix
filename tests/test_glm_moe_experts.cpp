// TDD Cycle 4 - GLM-5.2 MoE expert stack (layers 3-77): NVFP4 block
// dequantization (E2M1 nibbles x per-16 E4M3 block scales x E8M0 global
// modifier) piped straight into the locked w2(swiglu(w1(x) * w3(x)))
// projection path on ONE explicit async stream. The device output must match
// a deterministic CPU reference sequence layout element-for-element, with no
// truncation/zeroing of the FP16 arrays.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "lsxcommon/glm_moe_inference.h"
#include "lsxcommon/glm_moe_load.h"
#include "lsxcommon/schema.h"

using lsxcommon::glm::GlmRoutedSlot;
using lsxcommon::glm::GlmExpertDeviceBlock;

namespace {

constexpr int    kHidden = lsxcommon::glm::kGlmHidden;          // 6144
constexpr int    kInter  = lsxcommon::glm::kGlmMoeIntermediate; // 2048
constexpr int    kTopK   = lsxcommon::glm::kGlmMoETopK;         // 8
constexpr int    kSeq    = 3;
constexpr int64_t kMatEl =
    static_cast<int64_t>(kInter) * kHidden;                     // w1/w2/w3 elems
constexpr int64_t kPackedBytes  = kMatEl / 2;                   // 2 elems/byte
constexpr int64_t kScaleBytes   = kMatEl / lsxcommon::glm::kGlmNvfp4Block;

constexpr int kNRouted = lsxcommon::glm::kGlmNumRouted;

// E2M1 decode table (positive half; nibble bit 3 is the sign).
constexpr float kE2M1Table[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                 2.0f, 3.0f, 4.0f, 6.0f};

inline float E2M1Value(uint8_t nib)
{
    return (nib & 8u) ? -kE2M1Table[nib & 7u] : kE2M1Table[nib & 7u];
}

// Round-to-nearest-even float -> fp16 bits, matching __float2half_rn in the
// normal/subnormal range this fixture produces. Bit-exact for the test data.
inline uint16_t FloatToHalfRne(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
    const int      exp  = static_cast<int>((u >> 23) & 0xFFu);
    const uint32_t mant = u & 0x7FFFFFu;
    if (exp == 0xFF)
        return static_cast<uint16_t>(sign | 0x7C00u | (mant >> 13));
    const int exp2 = exp - 127;
    if (exp2 >= 15)
        return static_cast<uint16_t>(sign | 0x7C00u);   // overflow -> inf
    const uint32_t F = 0x800000u | mant;                // implicit bit 23
    if (exp2 >= -14) {
        int      e  = exp2 + 15;
        uint32_t m10 = (F >> 13) & 0x3FFu;
        const uint32_t rem = F & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (m10 & 1u))) {
            ++m10;
            if (m10 == 0x400u) { m10 = 0; ++e; }
        }
        if (e >= 31)
            return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign |
            (static_cast<uint16_t>(e) << 10) | static_cast<uint16_t>(m10));
    }
    // fp16 subnormal: value = S * 2^-24.
    const int shift = -exp2 - 1;
    if (shift > 24) return sign;
    uint32_t s   = F >> shift;
    const uint32_t rem = F & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (s & 1u))) ++s;
    if (s == 0x400u)
        return static_cast<uint16_t>(sign | 0x0400u);   // rounds to min normal
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(s));
}

inline float HalfToFloat(uint16_t bits)
{
    return __half2float(__ushort_as_half(bits));
}

// Deterministic per-expert NVFP4 nibble pattern (experts differ by id).
// A well-mixed integer hash is required here: a linear function of the flat
// index would be degenerate because both matrix strides (6144, 2048) are
// ≡ 0 (mod 16), so the row index could never reach the low nibble and every
// row would collapse to the same pattern.
inline uint8_t NibbleAt(int64_t i, int eid)
{
    unsigned x = static_cast<unsigned>(i) ^ 0x9E3779B9u;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    x *= 0xC2B2AE35u;
    x ^= x >> 16;
    x += 0x9E3779B9u * static_cast<unsigned>(eid);
    return static_cast<uint8_t>(x & 0x0Fu);
}

// E4M3 power-of-two scale byte representing 2^0 (=1.0): exponent field 7.
constexpr uint8_t kScaleOne = 0x38;

inline std::vector<uint8_t> BuildPacked(int64_t n, int eid)
{
    std::vector<uint8_t> packed(static_cast<size_t>((n + 1) / 2), 0);
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t nib = NibbleAt(i, eid);
        size_t bidx = static_cast<size_t>(i >> 1);
        if (i & 1)
            packed[bidx] = static_cast<uint8_t>((packed[bidx] & 0x0Fu) | (nib << 4));
        else
            packed[bidx] = static_cast<uint8_t>((packed[bidx] & 0xF0u) | nib);
    }
    return packed;
}

inline std::vector<uint8_t> BuildScales(int64_t n)
{
    const int64_t blocks = (n + lsxcommon::glm::kGlmNvfp4Block - 1) /
                           lsxcommon::glm::kGlmNvfp4Block;
    return std::vector<uint8_t>(static_cast<size_t>(blocks), kScaleOne);
}

// Exact device-side dequant: value = E2M1(nibble) * 2^(((s>>3)&0xF)-7) * gs,
// then RNE-rounded to FP16 (bit-identical to the kernel for this fixture).
inline float DequantValue(const std::vector<uint8_t> &packed,
                          const std::vector<uint8_t> &scales,
                          float gs, int64_t i)
{
    const uint8_t sb = scales[static_cast<size_t>(i / lsxcommon::glm::kGlmNvfp4Block)];
    // Canonical block-scale decode (complete E4M3FN; public wrapper of
    // nvfp4_scale.h) — never a local pow2/field-zero copy.
    const float    sc = lsxcommon::e4m3_to_float(sb);
    const uint8_t  b   = packed[static_cast<size_t>(i >> 1)];
    const uint8_t  nib = (i & 1) ? static_cast<uint8_t>(b >> 4)
                                 : static_cast<uint8_t>(b & 0x0Fu);
    return HalfToFloat(FloatToHalfRne(E2M1Value(nib) * sc * gs));
}

struct RefExpert {
    bool active = false;
    std::vector<float> w1;   // [inter, hidden] row-major, FP16-rounded
    std::vector<float> w2;   // [hidden, inter] row-major, FP16-rounded
    std::vector<float> w3;   // [inter, hidden] row-major, FP16-rounded
};

// Full double-precision reference evaluation of the expert stack, matching
// the device's FP16 rounding points exactly:
//   inter_i = RNE_fp16( w1_i . x ),  gate_i = RNE_fp16( w3_i . x )
//   swig_i  = RNE_fp16( silu(inter_i) * gate_i )
//   out_h   = RNE_fp16( acc_h + coeff * RNE_fp16( w2_h . swig ) )
std::vector<float> RefExpertStack(const std::vector<float> &x,
                                  const std::vector<GlmRoutedSlot> &routing,
                                  const std::vector<RefExpert> &experts)
{
    std::vector<float> out(static_cast<size_t>(kSeq) * kHidden, 0.0f);
    for (int t = 0; t < kSeq; ++t) {
        for (int r = 0; r < kTopK; ++r) {
            const GlmRoutedSlot &s = routing[static_cast<size_t>(t) * kTopK + r];
            const float coeff = s.coeff;
            const int   eid   = s.eid;
            if (eid < 0 || eid >= static_cast<int>(experts.size()) ||
                !experts[static_cast<size_t>(eid)].active)
                continue;

            const RefExpert &E = experts[static_cast<size_t>(eid)];
            const float *x_t = x.data() + static_cast<size_t>(t) * kHidden;

            std::vector<float> swig(static_cast<size_t>(kInter), 0.0f);
            for (int i = 0; i < kInter; ++i) {
                double a = 0.0, b = 0.0;
                const float *w1r = E.w1.data() + static_cast<size_t>(i) * kHidden;
                const float *w3r = E.w3.data() + static_cast<size_t>(i) * kHidden;
                for (int k = 0; k < kHidden; ++k) {
                    const double vx = static_cast<double>(x_t[k]);
                    a += static_cast<double>(w1r[k]) * vx;
                    b += static_cast<double>(w3r[k]) * vx;
                }
                const float ih = HalfToFloat(FloatToHalfRne(static_cast<float>(a)));
                const float gh = HalfToFloat(FloatToHalfRne(static_cast<float>(b)));
                const double silu   = ih / (1.0 + std::exp(-static_cast<double>(ih)));
                swig[static_cast<size_t>(i)] =
                    HalfToFloat(FloatToHalfRne(static_cast<float>(silu * gh)));
            }

            float *out_t = out.data() + static_cast<size_t>(t) * kHidden;
            for (int h = 0; h < kHidden; ++h) {
                double acc = 0.0;
                const float *w2r = E.w2.data() + static_cast<size_t>(h) * kInter;
                for (int j = 0; j < kInter; ++j)
                    acc += static_cast<double>(w2r[j]) * swig[static_cast<size_t>(j)];
                const float outm =
                    HalfToFloat(FloatToHalfRne(static_cast<float>(acc)));
                out_t[h] = HalfToFloat(FloatToHalfRne(
                    out_t[h] + coeff * outm));
            }
        }
    }
    return out;
}

inline bool ExpertActive(const GlmExpertDeviceBlock &b)
{
    return b.d_w1_packed && b.d_w1_scales && b.d_w2_packed &&
           b.d_w2_scales && b.d_w3_packed && b.d_w3_scales &&
           b.w1_packed_bytes == kPackedBytes &&
           b.w2_packed_bytes == kPackedBytes &&
           b.w3_packed_bytes == kPackedBytes &&
           b.w1_scale_bytes  == kScaleBytes &&
           b.w2_scale_bytes  == kScaleBytes &&
           b.w3_scale_bytes  == kScaleBytes;
}

} // namespace

TEST_CASE("GLM MoE expert stack: multi-token NVFP4 decode + SwiGLU vs "
          "deterministic FP16 reference", "[glm][moe][experts]")
{
    static_assert(kPackedBytes == 6291456,
                  "GLM expert matrix packing size regression");
    static_assert(kScaleBytes  == 786432,
                  "GLM expert per-16 block scale count regression");
    static_assert(lsxcommon::glm::kGlmNvfp4Block == 16,
                  "Cycle 4 locks the NVFP4 block size to 16");

    cudaFree(0);
    cudaStream_t st;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);

    // ── Input activations: 3 tokens, FP16-exact small values ───────────────
    std::vector<float> hx(static_cast<size_t>(kSeq) * kHidden);
    for (int t = 0; t < kSeq; ++t)
        for (int k = 0; k < kHidden; ++k)
            hx[static_cast<size_t>(t) * kHidden + k] =
                0.0078125f * static_cast<float>(((t * 5 + k * 7) % 9) - 4);
    std::vector<__half> hxh(static_cast<size_t>(kSeq) * kHidden);
    for (size_t i = 0; i < hx.size(); ++i) hxh[i] = __float2half(hx[i]);

    __half *d_act = nullptr;
    __half *d_out = nullptr;
    __half *d_scratch = nullptr;
    REQUIRE(cudaMalloc(&d_act, hxh.size() * sizeof(__half)) == cudaSuccess);
    const int64_t scratch_halves =
        lsxcommon::glm::kGlmExpertMatEl + kHidden + 2 * static_cast<int64_t>(kInter);
    REQUIRE(cudaMalloc(&d_scratch,
                       static_cast<size_t>(scratch_halves) * sizeof(__half)) == cudaSuccess);

    // Output with a guard tail to catch any out-of-grid FP16 writes.
    constexpr size_t kGuard = 8192;
    const size_t out_n = static_cast<size_t>(kSeq) * kHidden;
    std::vector<__half> hout(out_n + kGuard, __float2half(-1.0f));
    REQUIRE(cudaMalloc(&d_out, hout.size() * sizeof(__half)) == cudaSuccess);

    // Pre-fill d_out (output region AND guard tail) with the sentinel so any
    // write beyond [0, out_n) is detectable instead of reading cudaMalloc
    // garbage.
    REQUIRE(cudaMemcpy(d_out, hout.data(), hout.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_act, hxh.data(), hxh.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // ── Persistent VRAM expert blocks: experts 0..3 populated, rest idle ───
    std::vector<GlmExpertDeviceBlock> experts(static_cast<size_t>(kNRouted));
    std::vector<RefExpert> ref_experts(static_cast<size_t>(kNRouted));
    std::vector<void *> allocs;

    struct DevMat { uint8_t *packed = nullptr; uint8_t *scales = nullptr; };
    auto upload_mat = [&](int eid, DevMat &dm,
                          std::vector<float> &ref_w) -> bool {
        auto packed = BuildPacked(kMatEl, eid);
        auto scales = BuildScales(kMatEl);
        for (int64_t i = 0; i < kMatEl; ++i)
            ref_w[static_cast<size_t>(i)] = DequantValue(packed, scales, 1.0f, i);
        if (cudaMalloc(&dm.packed, static_cast<size_t>(kPackedBytes)) != cudaSuccess)
            return false;
        if (cudaMalloc(&dm.scales, static_cast<size_t>(kScaleBytes)) != cudaSuccess)
            return false;
        allocs.push_back(dm.packed);
        allocs.push_back(dm.scales);
        return cudaMemcpy(dm.packed, packed.data(),
                          static_cast<size_t>(kPackedBytes),
                          cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(dm.scales, scales.data(),
                          static_cast<size_t>(kScaleBytes),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    };

    for (int e = 0; e < 4; ++e) {
        auto &blk = experts[static_cast<size_t>(e)];
        auto &ref = ref_experts[static_cast<size_t>(e)];
        ref.w1.assign(static_cast<size_t>(kMatEl), 0.0f);
        ref.w2.assign(static_cast<size_t>(kMatEl), 0.0f);
        ref.w3.assign(static_cast<size_t>(kMatEl), 0.0f);
        DevMat m1{}, m2{}, m3{};
        REQUIRE(upload_mat(e, m1, ref.w1));
        REQUIRE(upload_mat(e, m2, ref.w2));
        REQUIRE(upload_mat(e, m3, ref.w3));
        blk.d_w1_packed = m1.packed; blk.d_w1_scales = m1.scales;
        blk.d_w2_packed = m2.packed; blk.d_w2_scales = m2.scales;
        blk.d_w3_packed = m3.packed; blk.d_w3_scales = m3.scales;
        blk.w1_packed_bytes = kPackedBytes; blk.w1_scale_bytes = kScaleBytes;
        blk.w2_packed_bytes = kPackedBytes; blk.w2_scale_bytes = kScaleBytes;
        blk.w3_packed_bytes = kPackedBytes; blk.w3_scale_bytes = kScaleBytes;
        blk.w1_global_scale = blk.w2_global_scale = blk.w3_global_scale = 1.0f;
        ref.active = true;
        REQUIRE(ExpertActive(blk));
    }

    // ── 64-bit routing grid: distinct active expert layouts per token ──────
    const size_t grid_slots = static_cast<size_t>(kSeq) * kTopK;
    constexpr size_t kGridGuard = 16;
    std::vector<uint64_t> hroute(grid_slots + kGridGuard, 0xdeadbeefdeadbeefULL);
    std::vector<GlmRoutedSlot> slots(grid_slots);
    const int ex_t0[8] = {0, 1, 2, 3, 0, 1, 2, 3};
    const int ex_t1[8] = {1, 2, 3, 0, 250, 1, 2, 3}; // eid 250: inactive skip
    const int ex_t2[8] = {3, 0, 1, 2, 3, 0, 1, 2};
    const float coeffs[8] =
        {0.125f, 0.0625f, 0.125f, 0.0625f, 0.125f, 0.0625f, 0.125f, 0.0625f};
    for (int t = 0; t < kSeq; ++t) {
        const int *ex = (t == 0) ? ex_t0 : (t == 1) ? ex_t1 : ex_t2;
        for (int r = 0; r < kTopK; ++r) {
            GlmRoutedSlot &s = slots[static_cast<size_t>(t) * kTopK + r];
            s.eid   = ex[r];
            s.coeff = coeffs[r];
        }
    }
    std::memcpy(hroute.data(), slots.data(), grid_slots * sizeof(uint64_t));

    uint64_t *d_route = nullptr;
    REQUIRE(cudaMalloc(&d_route, hroute.size() * sizeof(uint64_t)) == cudaSuccess);
    REQUIRE(cudaMemcpy(d_route, hroute.data(), hroute.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    // ── Deterministic CPU reference ─────────────────────────────────────────
    const std::vector<float> ref =
        RefExpertStack(hx, slots, ref_experts);
    float ref_max = 0.0f;
    for (float v : ref) ref_max = std::max(ref_max, std::fabs(v));
    // The fixture must exercise real arithmetic (not a near-zero no-op path).
    CHECK(ref_max > 1.0f);

    // ── Run the expert stack on one explicit async stream (no device sync) ──
    REQUIRE(lsxcommon::glm::RunGlmExpertsStack(
                st,
                reinterpret_cast<GlmRoutedSlot *>(d_route), kSeq,
                d_act, experts.data(), kNRouted,
                d_out, d_scratch) == cudaSuccess);
    REQUIRE(cudaMemcpy(hout.data(), d_out, hout.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);

    // ── Multi-token outputs match the deterministic reference layout ───────
    for (size_t i = 0; i < out_n; ++i) {
        const float got = __half2float(hout[i]);
        const float want = ref[i];
        CHECK(std::isfinite(got));
        // Relative tolerance (cuBLAS Blackwell fp16 tensor-core accumulation
        // differs from the double reference by rounding-order noise) plus an
        // absolute floor covering single-fp16-ulp boundary flips of the
        // largest intermediates. A layout/quantization fault shifts values by
        // far more than this bounded envelope.
        CHECK(std::fabs(got - want) <=
              (1e-2f * (1.0f + std::fabs(want)) + 1.0f));
    }

    // Cast through fp16 the same way the device accumulates: the reference's
    // own elements must be finite too (guards the reference itself).
    for (size_t i = 0; i < out_n; ++i) CHECK(std::isfinite(ref[i]));

    // Guard tails: routing grid untouched, output array wrote exactly out_n.
    for (size_t p = grid_slots; p < grid_slots + kGridGuard; ++p)
        CHECK(hroute[p] == 0xdeadbeefdeadbeefULL);
    for (size_t p = out_n; p < out_n + kGuard; ++p)
        CHECK(__half2float(hout[p]) == -1.0f);

    for (void *a : allocs) cudaFree(a);
    cudaFree(d_act);
    cudaFree(d_out);
    cudaFree(d_scratch);
    cudaFree(d_route);
    cudaStreamDestroy(st);
}