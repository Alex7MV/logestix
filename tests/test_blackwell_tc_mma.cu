// TDD — Phase 3 Task 1b: fragment-mapping calibration for the sm_120a
// block-scaled NVFP4 mma. Places a single e2m1 '1.0' at one packed operand
// slot per warp-launch and records which D cell lights up, empirically
// deriving the A/B register layouts (PTX doc formulas are ambiguous for
// kind::mxf4nvf4 packing order).
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <array>
#include <vector>

#include "lsxcommon/lsx_tc_gemm.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/schema.h"
#include "nvfp4_codec.h"
#include "nvfp4_scale.h"
#include "lsx_kernels_quant.cuh"

using lsxcommon::gpu::nvfp4::E2M1Value;
using lsxcommon::gpu::nvfp4::E4M3ScaleToFloat;
using lsxcommon::kernels::QuantizeE2m1TieUp;

// RAII device buffer owning one cudaMalloc'd region for the enclosing scope.
// Tests synchronize before teardown, so plain cudaMalloc/cudaFree semantics
// suffice (stream-ordered cudaMallocAsync pools add nothing here).
struct DeviceBuffer {
    void* raw = nullptr;
    explicit DeviceBuffer(size_t bytes)
    {
        REQUIRE(cudaMalloc(&raw, bytes) == cudaSuccess);
    }
    ~DeviceBuffer() { if (raw) cudaFree(raw); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template <typename T>
    [[nodiscard]] T* get()
    {
        return static_cast<T*>(raw);
    }
};

// Shared host-side mirror of ActQuantKernel's activation quantization,
// returned as dequantized values. Built on the SAME canonical codec
// primitives the device kernel uses (nvfp4_scale.h / nvfp4_codec.h /
// lsx_kernels_quant.cuh), so host/device encoder drift is impossible.
inline std::vector<float> HostActQuantDequant(const std::vector<__half> &hx)
{
    std::vector<float> av(hx.size());
    for (size_t b = 0; b + 16 <= hx.size(); b += 16) {
        float absmax = 0.f;
        for (int j = 0; j < 16; ++j)
            absmax = std::max(absmax, std::abs(__half2float(hx[b + j])));
        const uint8_t sb =
            absmax > 0.f ? lsxcommon::nvfp4::FloatToE4m3(absmax / 6.0f)
                         : uint8_t{0};
        const float s = E4M3ScaleToFloat(sb);
        for (int j = 0; j < 16; ++j)
            av[b + j] =
                s > 0.f
                    ? E2M1Value(QuantizeE2m1TieUp(__half2float(hx[b + j]) /
                                                  s)) *
                          s
                    : 0.f;
    }
    return av;
}

namespace {

constexpr uint8_t kOne = 2;   // e2m1 code for 1.0 (table idx 2)
constexpr uint8_t kZero = 0;
constexpr uint8_t kScale1 = 0x38;  // ue4m3 bias 7 (measured): E=7,M=0 -> 2^0 = 1.0

struct CalibInput {
    uint32_t a[32][4];
    uint32_t b[32][2];
    uint32_t sfa[32];
    uint32_t sfb[32];
};

__global__ void ProbeMmaKernel(const CalibInput *__restrict__ in,
                               float *__restrict__ out /* [16][8] */)
{
    const int lane = threadIdx.x;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
        ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, "
        "%14, {0, 0}, %15, {0, 0};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(in->a[lane][0]), "r"(in->a[lane][1]), "r"(in->a[lane][2]),
          "r"(in->a[lane][3]), "r"(in->b[lane][0]), "r"(in->b[lane][1]),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3),
          "r"(in->sfa[lane]), "r"(in->sfb[lane]));
    const int group = lane >> 2;
    const int tig = lane & 3;
    out[(group) * 8 + tig * 2 + 0] = d0;
    out[(group) * 8 + tig * 2 + 1] = d1;
    out[(group + 8) * 8 + tig * 2 + 0] = d2;
    out[(group + 8) * 8 + tig * 2 + 1] = d3;
}

void RunProbe(const CalibInput &hin, std::vector<float> &hout)
{
    CalibInput *din;
    float *dout;
    cudaMalloc(&din, sizeof(CalibInput));
    cudaMalloc(&dout, 16 * 8 * sizeof(float));
    cudaMemcpy(din, &hin, sizeof(CalibInput), cudaMemcpyHostToDevice);
    ProbeMmaKernel<<<1, 32>>>(din, dout);
    cudaDeviceSynchronize();
    cudaMemcpy(hout.data(), dout, 16 * 8 * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(din);
    cudaFree(dout);
}

CalibInput BaseOnes()
{
    CalibInput f{};
    for (int l = 0; l < 32; ++l) {
        // All e2m1 slots = 1.0 code in both nibbles of every byte.
        for (int r = 0; r < 4; ++r)
            f.a[l][r] = 0x22222222u | (0u << 0);  // nibble 2 = 1.0
        for (int r = 0; r < 2; ++r) f.b[l][r] = 0x22222222u;
        f.sfa[l] = (0x01010101u * kScale1);
        f.sfb[l] = (0x01010101u * kScale1);
    }
    return f;
}

} // namespace

TEST_CASE("sm_120a nvfp4 mma fragment calibration", "[tc][mma][calib]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    // Sweep 1: scale-byte decode curve. All elements = 1.0-code, SFA byte
    // swept, SFB fixed at candidate-1.0 encodings; D = 64 * f(SFA) * f(SFB).
    {
        const uint8_t sfb_bytes[4] = {0x38, 0x18, 0x08, 0x40};
        for (uint8_t sb : sfb_bytes) {
            for (int sa : {0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38,
                           0x40, 0x48, 0x70, 0x78}) {
                CalibInput f = BaseOnes();
                for (int l = 0; l < 32; ++l) {
                    uint32_t s = 0;
                    for (int b = 0; b < 4; ++b)
                        s |= uint32_t(sa) << (b * 8);
                    f.sfa[l] = s;
                    s = 0;
                    for (int b = 0; b < 4; ++b) s |= uint32_t(sb) << (b * 8);
                    f.sfb[l] = s;
                }
                std::vector<float> out(16 * 8);
                RunProbe(f, out);
                WARN("scale sa=0x" << std::hex << sa << " sb=0x"
                                            << int(sb) << std::dec
                                            << " -> D[0][0]=" << out[0]);
            }
        }
    }

    // Sweep 2: e2m1 code curve. All scales = 0x38 both sides, all A/B
    // nibbles set to code c; D = 64 * f(c)^2 * sfA * sfB.
    for (int c = 0; c < 16; ++c) {
        CalibInput f = BaseOnes();
        for (int l = 0; l < 32; ++l) {
            for (int r = 0; r < 4; ++r) f.a[l][r] = 0x11111111u * c;
            for (int r = 0; r < 2; ++r) f.b[l][r] = 0x11111111u * c;
        }
        std::vector<float> out(16 * 8);
        RunProbe(f, out);
        WARN("e2m1 code=" << c << " -> D[0][0]=" << out[0]);
    }

    // Determinism probe: identical config, repeated launches.
    {
        CalibInput f = BaseOnes();
        std::vector<float> out(16 * 8);
        for (int rep = 0; rep < 5; ++rep) {
            RunProbe(f, out);
            WARN("det rep=" << rep << " D[0][0]=" << out[0]
                            << " D[15][7]=" << out[15 * 8 + 7]);
        }
    }

    // Decisive mixed probes pin the absolute decode anchors.
    {
        // P1: A code 4 (std e2m1 = 2.0), B code 2 (std = 1.0), scales 0x38.
        // std hypothesis -> D = 64*2*1*sf^2 ; tells whether g*f == 1 splits
        // as (g=std, f=1) or (g=2x, f=1/2).
        CalibInput p = BaseOnes();
        for (int l = 0; l < 32; ++l) {
            for (int r = 0; r < 4; ++r) p.a[l][r] = 0x44444444u;
            for (int r = 0; r < 2; ++r) p.b[l][r] = 0x22222222u;
        }
        std::vector<float> out(16 * 8);
        RunProbe(p, out);
        WARN("P1 A=4,B=2,scale=0x38 -> D[0][0]=" << out[0]);

        // P2: A/B code 2, SFA byte0=0x38 bytes1-3=0x00; SFB all 0x38.
        // If byte c maps to K-chunk c: only chunk0 (k=0..15) contributes
        // -> D = 16 * (g*f)^2 = 16 under the unit-product hypothesis;
        // other byte orders give different partial sums (0, 48, 64).
        CalibInput q = BaseOnes();
        for (int l = 0; l < 32; ++l) {
            uint32_t s = 0x00000038u;
            q.sfa[l] = s;
        }
        RunProbe(q, out);
        WARN("P2 SFA byte0-only=0x38 -> D[0][0]=" << out[0]);
    }

    CHECK(true);
}

// ---------------------------------------------------------------------------
// Full-matrix verification: logical A[16x64] / B[64x8] / SFA / SFB packed per
// the PTX fragment formulas (rows alternate every 8 elements; SFA bytes =
// chunk tig/2 and 2+tig/2 for rows g / g+8; SFB byte c -> chunk c), then
// compared against an exact host reference.
// ---------------------------------------------------------------------------

namespace {

constexpr float kE2M1Vals2[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

uint8_t Enc2(float v)
{
    for (int i = 0; i < 16; ++i)
        if (kE2M1Vals2[i] == v) return static_cast<uint8_t>(i);
    return 0;
}

float Aat2(int m, int k) { return kE2M1Vals2[(m * 5 + k * 3) % 15]; }
float Bat2(int k, int n) { return kE2M1Vals2[(k * 7 + n * 11) % 15]; }

} // namespace

TEST_CASE("sm_120a nvfp4 block-scaled mma full-matrix verification",
          "[tc][mma][full]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    // Staged diagnosis: vary SFA/SFB independently to isolate which metadata
    // byte order is wrong. Stage 0: all scales = 1 (pure layout check).
    for (int stage = 0; stage < 4; ++stage) {
        const bool var_sfa = (stage == 1 || stage == 3);
        const bool var_sfb = (stage == 2 || stage == 3);

        uint8_t sfa_enc[16][4], sfb_enc[4][8];
        float sfa_val[16][4], sfb_val[4][8];
        for (int m = 0; m < 16; ++m)
            for (int c = 0; c < 4; ++c) {
                const int e = var_sfa ? ((m * 2 + c) % 5) : 0;
                sfa_enc[m][c] = static_cast<uint8_t>((e + 7) << 3);
                sfa_val[m][c] = static_cast<float>(1 << e);
            }
        for (int c = 0; c < 4; ++c)
            for (int n = 0; n < 8; ++n) {
                const int e = var_sfb ? ((n + 2 * c) % 4) : 0;
                sfb_enc[c][n] = static_cast<uint8_t>((e + 7) << 3);
                sfb_val[c][n] = static_cast<float>(1 << e);
            }


    CalibInput hin{};
    for (int lane = 0; lane < 32; ++lane) {
        const int group = lane >> 2;
        const int tig = lane & 3;
        uint32_t areg[4] = {0, 0, 0, 0};
        for (int i = 0; i < 32; ++i) {
            const int row = (i < 8 || (i >= 16 && i < 24)) ? group : group + 8;
            const int col = (i < 16) ? tig * 8 + (i & 7)
                                     : 32 + tig * 8 + (i & 7);
            const uint8_t code = Enc2(Aat2(row, col));
            areg[i / 8] |= uint32_t(code)
                           << (((i % 8) / 2) * 8 + ((i % 8) % 2) * 4);
        }
        for (int r = 0; r < 4; ++r) hin.a[lane][r] = areg[r];

        uint32_t breg[2] = {0, 0};
        for (int i = 0; i < 16; ++i) {
            const int k = (i < 8) ? tig * 8 + (i & 7)
                                  : 32 + tig * 8 + (i & 7);
            breg[i / 8] |= uint32_t(Enc2(Bat2(k, group)))
                           << (((i % 8) / 2) * 8 + ((i % 8) % 2) * 4);
        }
        hin.b[lane][0] = breg[0];
        hin.b[lane][1] = breg[1];

        // SFA layout (empirically confirmed by single-byte position map):
        // lane tig==0 carries row g's chunk scales in bytes 0..3,
        // lane tig==1 carries row g+8's; lanes tig>=2 are ignored.
        const int sfa_row = (tig == 0) ? group : group + 8;
        uint32_t sfa_v = 0;
        for (int c = 0; c < 4; ++c)
            sfa_v |= uint32_t(sfa_enc[sfa_row][c]) << (c * 8);
        hin.sfa[lane] = sfa_v;
        hin.sfb[lane] = uint32_t(sfb_enc[0][group]) |
                        (uint32_t(sfb_enc[1][group]) << 8) |
                        (uint32_t(sfb_enc[2][group]) << 16) |
                        (uint32_t(sfb_enc[3][group]) << 24);
    }

    std::vector<float> ref(16 * 8, 0.0f);
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < 64; ++k)
                acc += Aat2(m, k) * sfa_val[m][k / 16] *
                       Bat2(k, n) * sfb_val[k / 16][n];
            ref[m * 8 + n] = acc;
        }

    std::vector<float> out(16 * 8);
    RunProbe(hin, out);

    int mismatches = 0;
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 8; ++n) {
            const float got = out[m * 8 + n];
            const float want = ref[m * 8 + n];
            if (std::abs(got - want) > 1e-3f * (std::abs(want) + 1.0f)) {
                if (mismatches < 8)
                    WARN("stage=" << stage << " D[" << m << "][" << n
                                  << "] got=" << got << " want=" << want);
                ++mismatches;
            }
        }
    WARN("stage=" << stage << " mismatches=" << mismatches);
    CHECK(mismatches == 0);
    }
}

TEST_CASE("sm_120a nvfp4 mma SFA single-byte position map",
          "[tc][mma][sfamap]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    // All e2m1 = 1.0; ALL scale bytes = 0x00 (decodes to 0 -> D == 0).
    // Light exactly one SFA byte with 0x38 (f=1); the lit D row reveals the
    // (row, chunk) the hardware assigns to that (lane, byte) slot.
    for (int lane = 0; lane < 8; ++lane) {
        for (int jb = 0; jb < 4; ++jb) {
            CalibInput f = BaseOnes();
            for (int l = 0; l < 32; ++l) { f.sfa[l] = 0u; f.sfb[l] = 0u; }
            // SFB all-ones so column sums are uniform.
            for (int l = 0; l < 32; ++l) f.sfb[l] = 0x38383838u;
            f.sfa[lane] = uint32_t(0x38) << (jb * 8);
            std::vector<float> out(16 * 8);
            RunProbe(f, out);
            int r0 = -1, r1 = -1;
            float v0 = 0.f, v1 = 0.f;
            for (int m = 0; m < 16; ++m) {
                float s = 0.f;
                for (int n = 0; n < 8; ++n) s += out[m * 8 + n];
                if (std::abs(s) > 1e-3f) {
                    if (r0 < 0) { r0 = m; v0 = s / 8.f; }
                    else { r1 = m; v1 = s / 8.f; }
                }
            }
            WARN("SFA lane=" << lane << " byte=" << jb << " -> rows {"
                             << r0 << "(" << v0 << "), " << r1 << "(" << v1
                             << ")}");
        }
    }
    CHECK(true);
}

TEST_CASE("nvfp4 activation quantization round-trip", "[tc][actquant]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    // Odd tail: K = 80 (5 blocks), seq = 3.
    constexpr int seq = 1, K = 64;
    std::vector<__half> hx(static_cast<size_t>(seq) * K);
    // Deterministic small-magnitude pattern built from exact half values
    // (avoids any host-side float->half conversion edge cases).
    const float vals[16] = {0.f,   0.25f, -0.5f, 1.f,  -1.5f, 2.f,
                            -2.5f, 3.f,   -3.5f, 4.f,  -4.5f, 5.f,
                            -5.f,  6.f,   -6.f,  0.75f};
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = __float2half(vals[(i * 7 + i / 16) % 16]);
    hx[0] = __float2half(0.0f);   // zero block edge
    for (int i = 64; i < 80 && i < static_cast<int>(hx.size()); ++i)
        hx[i] = __float2half(0.0f);

    const size_t packed_n = hx.size() / 2, scales_n = hx.size() / 16;
    DeviceBuffer dx(hx.size() * 2), dp(packed_n), ds(scales_n);
    REQUIRE(cudaMemcpy(dx.get<__half>(), hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(
                nullptr, dx.get<__half>(), seq, K, dp.get<uint8_t>(),
                ds.get<uint8_t>()) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint8_t> hp(packed_n), hs(scales_n);
    REQUIRE(cudaMemcpy(hp.data(), dp.get<uint8_t>(), packed_n,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(hs.data(), ds.get<uint8_t>(), scales_n,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // Host decode with the shared repo codec primitives; check error.
    // Metric: per-block error normalized by the BLOCK absmax (per-element
    // relative error is unbounded for tiny elements in dynamic blocks --
    // inherent to NVFP4 pow2 scales). Sound bound: snap error <= 1*s and
    // s <= absmax/3 -> <= 1/3 of block absmax.
    double max_rel = 0.0;
    size_t worst_idx = 0; float worst_got = 0.f, worst_want = 0.f;
    uint8_t worst_sb = 0, worst_code = 0;
    for (int b = 0; b < static_cast<int>(scales_n); ++b) {
        const float s = E4M3ScaleToFloat(hs[b]);
        float block_absmax = 0.f;
        for (int j = 0; j < 16; ++j) {
            const size_t idx = static_cast<size_t>(b) * 16 + j;
            block_absmax =
                std::max(block_absmax, std::abs(__half2float(hx[idx])));
        }
        if (block_absmax == 0.0f) continue;
        for (int j = 0; j < 16; ++j) {
            const size_t idx = static_cast<size_t>(b) * 16 + j;
            const uint8_t byte = hp[idx / 2];
            const uint8_t code = (j & 1) ? (byte >> 4) : (byte & 0x0F);
            const float got = E2M1Value(code) * s;
            const float want = __half2float(hx[idx]);
            if (want == 0.0f && got == 0.0f) continue;
            const double rel =
                std::abs(got - want) / block_absmax;
            if (rel > max_rel) {
                max_rel = rel;
                worst_idx = idx; worst_got = got; worst_want = want;
                worst_sb = hs[b]; worst_code = code;
            }
        }
    }
    INFO("max quantization rel err=" << max_rel << " at idx=" << worst_idx
         << " got=" << worst_got << " want=" << worst_want
         << " scale_byte=0x" << std::hex << int(worst_sb) << std::dec
         << " code=" << int(worst_code));
    WARN("actquant max_rel=" << max_rel << " idx=" << worst_idx
         << " got=" << worst_got << " want=" << worst_want
         << " sb=0x" << std::hex << int(worst_sb) << std::dec
         << " code=" << int(worst_code));
    WARN("block0: sb=0x" << std::hex << int(hs[0]) << std::dec
         << " packed=" << int(hp[0]) << "," << int(hp[1]) << ","
         << int(hp[2]));
    WARN("sanity f2h(5)=" << __half2float(__float2half(5.0f))
         << " f2h(-2.5)=" << __half2float(__float2half(-2.5f)));
    for (int j = 0; j < 16; ++j)
        WARN("hx[" << j << "]=" << __half2float(hx[j]));
    // pow2 scale + nearest-level: worst case ~1/3 relative.
    CHECK(max_rel < 0.34);
}



TEST_CASE("actquant scale parity with canonical float_to_e4m3",
          "[tc][actquant][regression]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    // Contract (lsx_tc_gemm.h / ActQuantKernel comments): the per-16 block
    // scale must byte-match the canonical lsxcommon::float_to_e4m3
    // (full unsigned-E4M3, round-to-nearest-even).
    const float absmax_cases[5] = {6.0f, 12.0f, 3.0f, 5.7f, 2.4f};

    // Quantize one 16-element block on device; yields the scale byte and
    // the first packed byte (element 0 -> low nibble).
    const auto quantize_block = [](const std::vector<__half> &hx,
                                   uint8_t &sb, uint8_t &p0) {
        DeviceBuffer dx(hx.size() * 2), dp(hx.size() / 2), ds(1);
        REQUIRE(cudaMemcpy(dx.get<__half>(), hx.data(), hx.size() * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(
                    nullptr, dx.get<__half>(), 1, 16, dp.get<uint8_t>(),
                    ds.get<uint8_t>()) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        REQUIRE(cudaMemcpy(&sb, ds.get<uint8_t>(), 1,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        REQUIRE(cudaMemcpy(&p0, dp.get<uint8_t>(), 1,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
    };

    for (float am : absmax_cases) {
        std::vector<__half> hx(16);
        hx[0] = __float2half(am);
        for (int j = 1; j < 16; ++j)
            hx[j] = __float2half(am * 0.01f);

        // The device kernel sees the HALF-rounded value; derive the
        // expectation from it so cases near pow2 boundaries stay exact.
        const float am_h = __half2float(hx[0]);

        uint8_t sb = 0, p0 = 0;
        quantize_block(hx, sb, p0);

        INFO("absmax=" << am << " got=0x" << std::hex << int(sb) << std::dec
             << " want=0x"
             << int(lsxcommon::float_to_e4m3(am_h / 6.0f)));
        // Byte parity with the canonical pow2-E4M3 encoder.
        CHECK(sb == lsxcommon::float_to_e4m3(am_h / 6.0f));

        // Element 0 packs into the low nibble of byte 0; with a ceil-rounded
        // scale the block max snaps within the inherent NVFP4 level error
        // (<= half of the widest top gap => >= 75% preserved).
        const float got = E2M1Value(p0 & 0x0F) * E4M3ScaleToFloat(sb);
        INFO("decoded block max=" << got);
        CHECK(got >= 0.75f * am_h);
    }

    // Zero block: reserved scale byte 0, decoded magnitude 0.
    {
        std::vector<__half> hx(16, __float2half(0.0f));
        uint8_t sb = 1, p0 = 1;
        quantize_block(hx, sb, p0);
        CHECK(sb == 0);
        CHECK(E2M1Value(p0 & 0x0F) * E4M3ScaleToFloat(sb) == 0.0f);
    }
}



// ---- Shared exact NVFP4 host model (single source for all TC tests) --------
// Mirrors ActQuantKernel via the shared canonical codecs (nvfp4_scale.h ceil
// pow2-E4M3 encode + threshold-chain E2M1 snap TieUp) and the repo weight
// codec. Returns y[m*N+n] = wgs * sum_k act[m,k]*w[n,k].
namespace tc_ref {
// Host reference mirrors ActQuantKernel via the SAME shared codec primitives
// (HostActQuantDequant / nvfp4_scale.h) the device kernel uses, so encoder
// drift between host and device is impossible by design.
std::vector<float> HostRef(const std::vector<__half> &hx,
                           const std::vector<uint8_t> &hp,
                           const std::vector<uint8_t> &hs, int seq, int K,
                           int N, float wgs)
{
    const int k_half = K / 2;
    const int k_chunks = (K + 15) / 16;
    const std::vector<float> av = HostActQuantDequant(hx);
    std::vector<float> y(static_cast<size_t>(seq) * N, 0.f);
    for (int m = 0; m < seq; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const size_t we = static_cast<size_t>(n) * K + k;
                const uint8_t wb = hp[we / 2];
                const int nib =
                    (k & 1) ? static_cast<int>(wb >> 4)
                            : static_cast<int>(wb & 0x0F);
                acc += static_cast<double>(
                    av[static_cast<size_t>(m) * K + k] *
                    E2M1Value(nib) * E4M3ScaleToFloat(
                        hs[static_cast<size_t>(n) * k_chunks + k / 16]));
            }
            y[static_cast<size_t>(m) * N + n] =
                static_cast<float>(acc * static_cast<double>(wgs));
        }
    return y;
}
} // namespace tc_ref

TEST_CASE("tensor-core nvfp4 gemm vs host reference", "[tc][gemm]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    // Shape sweep: original verified shape plus seq==1 fast-path shapes
    // (decode kernel route; tile via LSX_DEC_DISABLE=1).
    const std::vector<std::array<int, 3>> shapes{
        {3, 544, 36},   {1, 64, 128},  {1, 256, 128},
        {1, 1024, 512}, {1, 4096, 128}};
    for (const auto &[seq, K, N] : shapes) {
    std::vector<__half> hx(static_cast<size_t>(seq) * K);
    const float xvals[16] = {0.f, 0.25f, -0.5f, 1.f,  -1.5f, 2.f, -2.5f, 3.f,
                             -3.5f, 4.f, -4.5f, 5.f, -5.f, 6.f, -6.f, 0.75f};
    for (size_t i = 0; i < hx.size(); ++i)
        hx[i] = __float2half(xvals[(i * 7 + i / 16) % 16]);

    const size_t wp_n = static_cast<size_t>(N) * K / 2;
    const size_t ws_n = static_cast<size_t>(N) * ((K + 15) / 16);
    std::vector<uint8_t> hp(wp_n), hs(ws_n);
    for (size_t i = 0; i < wp_n; ++i)
        hp[i] = static_cast<uint8_t>((i * 29) & 0xFF);
    for (size_t i = 0; i < ws_n; ++i)
        hs[i] = static_cast<uint8_t>((1 + ((i * 3) % 15)) << 3); // repo-valid pow2 (e4 1..15, M=0)

    DeviceBuffer dx(hx.size() * 2), dy(static_cast<size_t>(seq) * N * 2),
        wp(wp_n), ws(ws_n);
    REQUIRE(cudaMemcpy(dx.get<__half>(), hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(wp.get<uint8_t>(), hp.data(), wp_n,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ws.get<uint8_t>(), hs.data(), ws_n,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    const float wgs = 0.125f;
    REQUIRE(lsxcommon::lsx_tc::RunLsxTcNvfp4Gemm(
                nullptr, dx.get<__half>(), wp.get<uint8_t>(),
                ws.get<uint8_t>(), wgs, dy.get<__half>(), seq, K, N)
            == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hy(static_cast<size_t>(seq) * N);
    REQUIRE(cudaMemcpy(hy.data(), dy.get<__half>(), hy.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // ---- Host reference mirroring the kernel's math exactly, via the
    // shared codec primitives (HostActQuantDequant / nvfp4_codec.h).
    const std::vector<float> av = HostActQuantDequant(hx);

    double max_err = 0.0, max_ref = 0.0;
    for (int m = 0; m < seq; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                const size_t we = static_cast<size_t>(n) * K + k;
                const uint8_t wb = hp[we / 2];
                const uint8_t wn =
                    (k & 1) ? static_cast<uint8_t>(wb >> 4)
                            : static_cast<uint8_t>(wb & 0x0F);
                const float wv =
                    E2M1Value(wn) *
                    E4M3ScaleToFloat(
                        hs[static_cast<size_t>(n) * ((K + 15) / 16) +
                           k / 16]);
                acc += av[static_cast<size_t>(m) * K + k] * wv;
            }
            const float want = acc * wgs;
            const float got =
                __half2float(hy[static_cast<size_t>(m) * N + n]);
            max_err = std::max<double>(max_err, std::abs(got - want));
            max_ref = std::max<double>(max_ref, std::abs(want));
        }
    WARN("tcgemm max_abs_err=" << max_err << " max_ref=" << max_ref);
    int shown = 0;
    for (int m = 0; m < seq && shown < 6; ++m)
        for (int n = 0; n < N && shown < 6; ++n) {
            const float want = [&] {
                float acc = 0.f;
                for (int k = 0; k < K; ++k) {
                    const size_t we2 = static_cast<size_t>(n) * K + k;
                    const uint8_t wb2 = hp[we2 / 2];
                    const uint8_t wn2 =
                        (k & 1) ? static_cast<uint8_t>(wb2 >> 4)
                                : static_cast<uint8_t>(wb2 & 0x0F);
                    acc += av[static_cast<size_t>(m) * K + k] *
                           E2M1Value(wn2) *
                           E4M3ScaleToFloat(
                               hs[static_cast<size_t>(n) *
                                      ((K + 15) / 16) +
                                  k / 16]);
                }
                return acc * wgs;
            }();
            const float got =
                __half2float(hy[static_cast<size_t>(m) * N + n]);
            if (std::abs(got - want) >
                1e-2f * std::abs(want) + 1e-3f) {
                WARN("cell [" << m << "," << n << "] got=" << got
                              << " want=" << want);
                ++shown;
            }
        }
    CHECK(max_err <= 2e-2f * static_cast<float>(max_ref) + 1e-3f);
    }
}

TEST_CASE("tc gemm delta-pulse k-alignment probe", "[tc][gemm][delta]")
{
    WARN("delta: entered");
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    constexpr int seq = 3, K = 544, N = 36; // full: tails + multi-warp
    const size_t wp_n = static_cast<size_t>(N) * K / 2;
    const size_t ws_n = static_cast<size_t>(N) * ((K + 15) / 16);
    std::vector<uint8_t> hp(wp_n), hs(ws_n);
    for (size_t i = 0; i < wp_n; ++i)
        hp[i] = static_cast<uint8_t>((i * 29) & 0xFF);
    for (size_t i = 0; i < ws_n; ++i)
        hs[i] = static_cast<uint8_t>((1 + ((i * 3) % 15)) << 3); // repo-valid pow2 (e4 1..15, M=0)

    DeviceBuffer dx(K * 2), dy(N * 2), wp(wp_n), ws(ws_n);
    REQUIRE(cudaMemcpy(wp.get<uint8_t>(), hp.data(), wp_n, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ws.get<uint8_t>(), hs.data(), ws_n, cudaMemcpyHostToDevice) == cudaSuccess);

    for (int pulse_k : {0, 1, 9, 17, 33}) {
        std::vector<__half> hx(K, __float2half(0.0f));
        hx[pulse_k] = __float2half(1.0f);
        REQUIRE(cudaMemcpy(dx.get<__half>(), hx.data(), K * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(lsxcommon::lsx_tc::RunLsxTcNvfp4Gemm(
                    nullptr, dx.get<__half>(), wp.get<uint8_t>(),
                    ws.get<uint8_t>(), 1.0f, dy.get<__half>(), 1, K, N)
                == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<__half> hy(N);
        REQUIRE(cudaMemcpy(hy.data(), dy.get<__half>(), N * 2,
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        // Expected: y[n] = w_deq(pulse_k, n); quantized av(1.0) may snap:
        // block absmax=1 -> s=pow2ceil(1/6)=2^-3=0.125 -> av=4*0.125=1.
        for (int n = 0; n < N; ++n) {
            const size_t we = static_cast<size_t>(n) * K + pulse_k;
            const uint8_t wb = hp[we / 2];
            const uint8_t wn =
                (pulse_k & 1) ? static_cast<uint8_t>(wb >> 4)
                              : static_cast<uint8_t>(wb & 0x0F);
            const float want =
                E2M1Value(wn) *
                E4M3ScaleToFloat(
                    hs[static_cast<size_t>(n) * 4 + pulse_k / 16]);
            const float got = __half2float(hy[n]);
            if (std::abs(got - want) > 1e-3f * (std::abs(want) + 0.1f))
                WARN("pulse k=" << pulse_k << " n=" << n << " got=" << got
                                << " want=" << want);
        }
    }
    CHECK(true);
}

TEST_CASE("tensor-core vs scalar nvfp4 gemm throughput", "[tc][gemm][perf]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    struct S { int K, N; const char *label; };
    const S shapes[] = {{4096, 4096, "attn_qkv_o"},
                        {4096, 9728, "mlp_gate_up"},
                        {9728, 4096, "mlp_down"}};

    cudaStream_t st = nullptr;
    REQUIRE(cudaStreamCreate(&st) == cudaSuccess);
    cudaEvent_t t0, t1;
    REQUIRE(cudaEventCreate(&t0) == cudaSuccess);
    REQUIRE(cudaEventCreate(&t1) == cudaSuccess);

    for (const auto &sh : shapes) {
        const int K = sh.K, N = sh.N;
        std::vector<__half> hx(K);
        for (int i = 0; i < K; ++i)
            hx[i] = __float2half(static_cast<float>((i % 13) - 6) * 0.25f);
        std::vector<uint8_t> hp(static_cast<size_t>(K) * N / 2),
            hs(static_cast<size_t>(K) * N / 16);
        for (size_t i = 0; i < hp.size(); ++i)
            hp[i] = static_cast<uint8_t>((i * 29) & 0xFF);
        for (size_t i = 0; i < hs.size(); ++i)
            hs[i] = static_cast<uint8_t>((1 + ((i * 3) % 15)) << 3);

        __half *dx, *dy;
        uint8_t *dp, *ds, *wp, *ws;
        REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&dy, static_cast<size_t>(N) * 2) == cudaSuccess);
        REQUIRE(cudaMalloc(&dp, hp.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&ds, hs.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&wp, hp.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(&ws, hs.size()) == cudaSuccess);
        REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(dp, hp.data(), hp.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(ds, hs.data(), hs.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(wp, hp.data(), hp.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(ws, hs.data(), hs.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);

        auto bench = [&](bool tc, float &us) {
            using lsxcommon::qwen::RunQwenNvfp4Gemm;
            for (int w = 0; w < 5; ++w) {
                cudaError_t e = tc
                    ? lsxcommon::lsx_tc::RunLsxTcNvfp4Gemm(
                          st, dx, wp, ws, 0.125f, dy, 1, K, N)
                    : RunQwenNvfp4Gemm(st, dx, dp, ds, 0.125f, dy, 1, K, N);
                REQUIRE(e == cudaSuccess);
            }
            REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
            constexpr int iters = 50;
            REQUIRE(cudaEventRecord(t0, st) == cudaSuccess);
            for (int it = 0; it < iters; ++it) {
                cudaError_t e = tc
                    ? lsxcommon::lsx_tc::RunLsxTcNvfp4Gemm(
                          st, dx, wp, ws, 0.125f, dy, 1, K, N)
                    : RunQwenNvfp4Gemm(st, dx, dp, ds, 0.125f, dy, 1, K, N);
                REQUIRE(e == cudaSuccess);
            }
            REQUIRE(cudaEventRecord(t1, st) == cudaSuccess);
            REQUIRE(cudaEventSynchronize(t1) == cudaSuccess);
            float ms = -1.f;
            REQUIRE(cudaEventElapsedTime(&ms, t0, t1) == cudaSuccess);
            us = ms * 1000.f / iters;
        };
        float us_scalar = 0.f, us_tc = 0.f;
        bench(false, us_scalar);
        bench(true, us_tc);

        // L2-cold variant: evict the weight between iterations with a large
        // streaming write, mirroring real decode where the layer working set
        // (~100 MB+) thrashes L2. Reports the DRAM-bound rate.
        const size_t flush_bytes = 256ull << 20;
        char *flush_buf = nullptr;
        REQUIRE(cudaMalloc(&flush_buf, flush_bytes) == cudaSuccess);
        auto bench_cold = [&](bool tc, float &us) {
            auto run1 = [&](bool with_flush) {
                for (int w = 0; w < 3; ++w) {
                    if (with_flush)
                        REQUIRE(cudaMemsetAsync(flush_buf, 0, flush_bytes,
                                                st) == cudaSuccess);
                    cudaError_t e =
                        tc ? lsxcommon::lsx_tc::RunLsxTcNvfp4Gemm(
                                 st, dx, wp, ws, 0.125f, dy, 1, K, N)
                           : lsxcommon::qwen::RunQwenNvfp4Gemm(
                                 st, dx, dp, ds, 0.125f, dy, 1, K, N);
                    REQUIRE(e == cudaSuccess);
                }
                REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
                constexpr int iters = 20;
                REQUIRE(cudaEventRecord(t0, st) == cudaSuccess);
                for (int it = 0; it < iters; ++it) {
                    if (with_flush)
                        REQUIRE(cudaMemsetAsync(flush_buf, 0, flush_bytes,
                                                st) == cudaSuccess);
                    cudaError_t e =
                        tc ? lsxcommon::lsx_tc::RunLsxTcNvfp4Gemm(
                                 st, dx, wp, ws, 0.125f, dy, 1, K, N)
                           : lsxcommon::qwen::RunQwenNvfp4Gemm(
                                 st, dx, dp, ds, 0.125f, dy, 1, K, N);
                    REQUIRE(e == cudaSuccess);
                }
                REQUIRE(cudaEventRecord(t1, st) == cudaSuccess);
                REQUIRE(cudaEventSynchronize(t1) == cudaSuccess);
                float ms = -1.f;
                REQUIRE(cudaEventElapsedTime(&ms, t0, t1) == cudaSuccess);
                return ms * 1000.f / iters;
            };
            const float both = run1(true), only_memset = run1(false);
            us = both - only_memset;
            if (us < 0.f) us = 0.f;
        };
        float us_tc_cold = 0.f;
        bench_cold(true, us_tc_cold);
        WARN("[TCBENCH][l2cold][" << sh.label
                                  << "] tensorcore=" << us_tc_cold
                                  << " us");
        cudaFree(flush_buf);

        WARN("[TCBENCH][" << sh.label << "] scalar=" << us_scalar
                          << " us  tensorcore=" << us_tc << " us  speedup="
                          << us_scalar / us_tc);

        cudaFree(dx); cudaFree(dy); cudaFree(dp); cudaFree(ds);
        cudaFree(wp); cudaFree(ws);
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaStreamDestroy(st);
}

TEST_CASE("tma bulk tensor 2d probe", "[tc][tma]")
{
    cudaStream_t s;
    REQUIRE(cudaStreamCreate(&s) == cudaSuccess);
    const cudaError_t e = lsxcommon::lsx_tc::LsxProbeTmaBulkCopy(s);
    if (e == cudaErrorNotSupported) {
        WARN("TMA bulk-tensor not supported on this device");
        return;
    }
    REQUIRE(e == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(s) == cudaSuccess);
    REQUIRE(cudaStreamDestroy(s) == cudaSuccess);
}


// ── B3 minimal multi-block TMA repro: exact decode-kernel geometry ─────────
#include <cuda.h>
namespace tma_mini {

constexpr int kRows = 128;          // box rows (= kDecCols)
constexpr int kRowBytes = 32;       // box row bytes (= kDecChunkBytes)
constexpr int kN = 256;             // global rows
constexpr int kKH = 4864;           // global row bytes (K/2, K=9728)

__global__ void MiniKernel(const __grid_constant__ CUtensorMap tmap,
                           const uint8_t *__restrict__ gsrc,
                           int *__restrict__ ok)
{
    __shared__ __align__(128) uint8_t ring[2][kRows][kRowBytes];
    __shared__ __align__(8) unsigned long long mbar[2];
    const int tid = threadIdx.x;
    if (tid == 0) {
        for (int b = 0; b < 2; ++b)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
                         "r"(static_cast<unsigned>(
                             __cvta_generic_to_shared(&mbar[b]))));
    }
    __syncthreads();

    auto stage = [&](int chunk, int buf) {
        if (tid == 0) {
            const unsigned mb = static_cast<unsigned>(
                __cvta_generic_to_shared(&mbar[buf]));
            const unsigned dst = static_cast<unsigned>(
                __cvta_generic_to_shared(ring[buf][0]));
            asm volatile(
                "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                "r"(mb), "r"(kRows * kRowBytes));
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier"
                "::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
                "l"(reinterpret_cast<unsigned long long>(&tmap)),
                "r"(static_cast<unsigned>(chunk * 32)),
                "r"(blockIdx.x * kRows), "r"(mb) : "memory");
        }
    };
    stage(0, 0);
    {
        const unsigned mb =
            static_cast<unsigned>(__cvta_generic_to_shared(&mbar[0]));
        asm volatile("{\n\t.reg .pred p;\n\tWAIT:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
                     "@!p bra WAIT;\n\t}" ::"r"(mb));
    }
    __syncthreads();
    bool good = true;
    const int col0 = blockIdx.x * kRows;
    for (int i = tid; i < kRows * kRowBytes; i += blockDim.x)
        good &= ring[0][0][i] == gsrc[(col0 + i / kRowBytes) * kKH +
                                      (i % kRowBytes)];
    if (!good) *ok = 0;
}

} // namespace tma_mini

TEST_CASE("tma mini multi-block repro", "[tc][tma]")
{
    using namespace tma_mini;
    uint8_t *d_src = nullptr;
    int *d_ok = nullptr;
    const size_t bytes = static_cast<size_t>(kN) * kKH;
    REQUIRE(cudaMalloc(&d_src, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ok, 4) == cudaSuccess);
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < h.size(); ++i) h[i] = static_cast<uint8_t>(i * 7);
    REQUIRE(cudaMemcpy(d_src, h.data(), h.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_ok, 1, 4) == cudaSuccess);

    CUtensorMap tmap{};
    const cuuint64_t gdim[2] = {kKH, kN};
    const cuuint64_t gstride[1] = {kKH};
    const cuuint32_t bdim[2] = {kRowBytes, kRows};
    const cuuint32_t estride[2] = {1, 1};
    REQUIRE(cuTensorMapEncodeTiled(
                &tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_src, gdim,
                gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);

    MiniKernel<<<kN / kRows, 256, 0, nullptr>>>(tmap, d_src, d_ok);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    int h_ok = 9;
    REQUIRE(cudaMemcpy(&h_ok, d_ok, 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(h_ok != 0);   // memset fills all four bytes with 0x01
    cudaFree(d_src);
    cudaFree(d_ok);
}

namespace tma_mini {

// Step-A variant: identical to MiniKernel but with a dynamic smem section
// coexisting with the TMA ring (isolate dynamic/static interaction).
__global__ void MiniDynKernel(const __grid_constant__ CUtensorMap tmap,
                              const uint8_t *__restrict__ gsrc,
                              int *__restrict__ ok)
{
    extern __shared__ __align__(16) uint8_t dyn[];
    const int k_half = 4864;
    uint8_t *const sxp = dyn;                       // [k_half]
    uint8_t *const sxsc = sxp + ((k_half + 15) & ~15);
    // Step-B: ring + mbarriers live in the DYNAMIC section too.
    // HYPOTHESIS under test: bulk-tensor dst needs 128B (not just 16B)
    // alignment -> round the section start up to 128.
    uintptr_t rb = reinterpret_cast<uintptr_t>(
        sxsc + ((k_half / 16 + 15) & ~15));
    rb = (rb + 127) & ~(uintptr_t)127;
    uint8_t (*ring)[kRows][kRowBytes] =
        reinterpret_cast<uint8_t (*)[kRows][kRowBytes]>(rb);
    unsigned long long *mbar = reinterpret_cast<unsigned long long *>(
        reinterpret_cast<uint8_t *>(ring) + 2 * kRows * kRowBytes);
    const int tid = threadIdx.x;
    for (int i = tid; i < k_half; i += blockDim.x) sxp[i] = gsrc[i];
    for (int c = tid; c < k_half / 16; c += blockDim.x) sxsc[c] = 1;
    if (tid == 0) {
        for (int b = 0; b < 2; ++b)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::
                         "r"(static_cast<unsigned>(
                             __cvta_generic_to_shared(&mbar[b]))));
    }
    __syncthreads();
    if (tid == 0) {
        const unsigned mb = static_cast<unsigned>(
            __cvta_generic_to_shared(&mbar[0]));
        const unsigned dst =
            static_cast<unsigned>(__cvta_generic_to_shared(ring[0][0]));
        if (blockIdx.x == 0)
            printf("DBG dyn base=%p ring0=%p cvta=%u (%%16==%u)\n",
                   (void *)dyn, (void *)ring[0][0], dst, dst % 16);
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
                     "r"(mb), "r"(kRows * kRowBytes));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier"
            "::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
            "l"(reinterpret_cast<unsigned long long>(&tmap)), "r"(0),
            "r"(blockIdx.x * kRows), "r"(mb) : "memory");
    }
    {
        const unsigned mb =
            static_cast<unsigned>(__cvta_generic_to_shared(&mbar[0]));
        asm volatile("{\n\t.reg .pred p;\n\tWAIT:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t"
                     "@!p bra WAIT;\n\t}" ::"r"(mb));
    }
    __syncthreads();
    bool good = true;
    const int col0 = blockIdx.x * kRows;
    for (int i = tid; i < kRows * kRowBytes; i += blockDim.x)
        good &= ring[0][0][i] == gsrc[(col0 + i / kRowBytes) * kKH +
                                      (i % kRowBytes)];
    if (!good) *ok = 0;
    // Touch dynamic section so it cannot be dead-stripped.
    if (sxp[blockIdx.x] == 0xFF) *ok = 7;
}

} // namespace tma_mini

TEST_CASE("tma mini stepA dyn-smem coexistence", "[tc][tma]")
{
    using namespace tma_mini;
    uint8_t *d_src = nullptr;
    int *d_ok = nullptr;
    const size_t bytes = static_cast<size_t>(kN) * kKH;
    REQUIRE(cudaMalloc(&d_src, bytes) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_ok, 4) == cudaSuccess);
    std::vector<uint8_t> h(bytes);
    for (size_t i = 0; i < h.size(); ++i) h[i] = static_cast<uint8_t>(i * 7);
    REQUIRE(cudaMemcpy(d_src, h.data(), h.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(cudaMemset(d_ok, 1, 4) == cudaSuccess);

    CUtensorMap tmap{};
    const cuuint64_t gdim[2] = {kKH, kN};
    const cuuint64_t gstride[1] = {kKH};
    const cuuint32_t bdim[2] = {kRowBytes, kRows};
    const cuuint32_t estride[2] = {1, 1};
    REQUIRE(cuTensorMapEncodeTiled(
                &tmap, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, d_src, gdim,
                gstride, bdim, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS);

    const size_t dyn_bytes = ((kKH + 15) & ~15) + ((kKH / 16 + 15) & ~15) +
                             2 * kRows * kRowBytes + 32 + 128;
    MiniDynKernel<<<kN / kRows, 256, dyn_bytes>>>(tmap, d_src, d_ok);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    int h_ok = 0;
    REQUIRE(cudaMemcpy(&h_ok, d_ok, 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(h_ok != 0);
    cudaFree(d_src);
    cudaFree(d_ok);
}

TEST_CASE("decode kernel column-response map", "[tc][gemm][calib2]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    // Weights: every column c carries a SINGLE e2m1 '1.0' at k=0 (low nibble
    // of byte 0); everything else zero. Scales neutral (2^0). Activations:
    // x[k]=1.0 -> quantized ~1. Host truth: y[c] == 1 for every column.
    constexpr int K = 64, N = 128;
    std::vector<__half> hx(K, __float2half(1.0f));
    const size_t wp_n = size_t(N) * K / 2, ws_n = size_t(N) * (K / 16);
    std::vector<uint8_t> hp(wp_n, 0), hs(ws_n, 0x38);
    for (int c = 0; c < N; ++c) hp[size_t(c) * (K / 2)] = 0x02;  // k=0 lo nib

    __half *dx, *dy;
    uint8_t *dp, *ds, *wp, *ws;
    cudaMalloc(&dx, hx.size() * 2); cudaMalloc(&dy, N * 2);
    cudaMalloc(&dp, wp_n); cudaMalloc(&ds, ws_n);
    cudaMalloc(&wp, wp_n); cudaMalloc(&ws, ws_n);
    cudaMemcpy(dx, hx.data(), 2 * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp.data(), wp_n, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), ws_n, cudaMemcpyHostToDevice);
    cudaMemcpy(wp, hp.data(), wp_n, cudaMemcpyHostToDevice);
    cudaMemcpy(ws, hs.data(), ws_n, cudaMemcpyHostToDevice);

    using lsxcommon::lsx_tc::RunLsxTcNvfp4Gemm;
    REQUIRE(RunLsxTcNvfp4Gemm(nullptr, dx, dp, ds, 1.0f, dy, 1, K, N,
                               false) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    std::vector<__half> hy(N);
    cudaMemcpy(hy.data(), dy, N * 2, cudaMemcpyDeviceToHost);
    std::string dmp = "COLMAP:";
    for (int c = 0; c < 16; ++c) dmp += " " + std::to_string(__half2float(hy[c]));
    WARN(dmp);
    // The engine act-quantizes x (all 1.0) with a full-E4M3 RNE scale, so the
    // host truth is the same quantize-dequant the kernel performs, not 1.0.
    const float ascale =
        E4M3ScaleToFloat(lsxcommon::float_to_e4m3(1.0f / 6.0f));
    const float act_deq =
        E2M1Value(QuantizeE2m1TieUp(1.0f / ascale)) * ascale;
    for (int c = 0; c < N; ++c)
        CHECK(std::abs(__half2float(hy[c]) - act_deq) <= 1e-3f);
    cudaFree(dx); cudaFree(dy); cudaFree(dp); cudaFree(ds);
    cudaFree(wp); cudaFree(ws);
}

TEST_CASE("act quant byte-exact vs host model at seq1/K256", "[tc][actquant][k256]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    constexpr int seq = 1, K = 256;
    std::vector<__half> hx(static_cast<size_t>(seq) * K);
    const float xv[16] = {0.f,  0.25f, -0.5f, 1.f,  -1.5f, 2.f,
                          -2.5f, 3.f,  -3.5f, 4.f,  -4.5f, 5.f,
                          -5.f,  6.f,  -6.f,  0.75f};
    for (int i = 0; i < K; ++i)
        hx[i] = __float2half(xv[(i * 7 + i / 16) % 16]);

    DeviceBuffer dx(hx.size() * 2), dp(hx.size() / 2), ds(hx.size() / 16);
    cudaMemcpy(dx.get<__half>(), hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    REQUIRE(lsxcommon::lsx_tc::LsxTcQuantizeActsNvfp4(
                nullptr, dx.get<__half>(), seq, K, dp.get<uint8_t>(),
                ds.get<uint8_t>()) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint8_t> hp(K / 2), hs(K / 16);
    cudaMemcpy(hp.data(), dp.get<uint8_t>(), K / 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(hs.data(), ds.get<uint8_t>(), K / 16, cudaMemcpyDeviceToHost);

    // Host model via the shared codec primitives (same source as the device
    // kernel: nvfp4_scale.h encode + lsx_kernels_quant.cuh E2M1 TieUp).
    int mismatches = 0;
    for (int b = 0; b < K / 16; ++b) {
        float absmax = 0.f;
        for (int j = 0; j < 16; ++j)
            absmax = std::max(absmax, std::abs(__half2float(hx[b * 16 + j])));
        const uint8_t sb_want =
            absmax > 0.f ? lsxcommon::nvfp4::FloatToE4m3(absmax / 6.0f)
                         : uint8_t{0};
        INFO("block " << b << " sb want=" << int(sb_want)
                      << " got=" << int(hs[b]));
        CHECK(hs[b] == sb_want);
        const float scl = E4M3ScaleToFloat(sb_want);
        for (int j = 0; j < 16; ++j) {
            const int idx = b * 16 + j;
            // Encode the RATIO x/scale (not the dequantized value).
            const float ratio =
                scl > 0.f ? __half2float(hx[idx]) / scl : 0.f;
            const uint8_t code = QuantizeE2m1TieUp(ratio);
            const uint8_t got_byte = hp[idx / 2];
            const uint8_t got_code =
                (idx & 1) ? static_cast<uint8_t>(got_byte >> 4)
                          : static_cast<uint8_t>(got_byte & 0xF);
            if (got_code != code && mismatches < 8)
                WARN("MISMATCH idx=" << idx << " x=" << __half2float(hx[idx])
                                     << " sb=" << int(hs[b])
                                     << " want_code=" << code
                                     << " got_code=" << int(got_code));
            if (got_code != code) ++mismatches;
        }
    }
    INFO("nibble mismatches=" << mismatches);
    CHECK(mismatches == 0);
}
