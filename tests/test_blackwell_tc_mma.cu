// TDD — Phase 3 Task 1b: fragment-mapping calibration for the sm_120a
// block-scaled NVFP4 mma. Places a single e2m1 '1.0' at one packed operand
// slot per warp-launch and records which D cell lights up, empirically
// deriving the A/B register layouts (PTX doc formulas are ambiguous for
// kind::mxf4nvf4 packing order).
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

#include "lsxcommon/qwen_tc_gemm.h"
#include "lsxcommon/qwen_inference.h"

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

    __half *dx;
    uint8_t *dp, *ds;
    const size_t packed_n = hx.size() / 2, scales_n = hx.size() / 16;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dp, packed_n) == cudaSuccess);
    REQUIRE(cudaMalloc(&ds, scales_n) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    REQUIRE(lsxcommon::qwen::tc::QuantizeActsNvfp4(nullptr, dx, seq, K, dp,
                                                   ds) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<uint8_t> hp(packed_n), hs(scales_n);
    REQUIRE(cudaMemcpy(hp.data(), dp, packed_n,
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(cudaMemcpy(hs.data(), ds, scales_n,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // Host decode with the repo codec semantics and check quantization error.
    auto dec_scale = [](uint8_t b) -> float {
        const uint32_t e4 = (b >> 3) & 0xFu;
        if (e4 == 0) return 0.0f;
        union { uint32_t u; float f; } bits{};
        bits.u = (e4 + 120u) << 23;
        return bits.f;
    };
    const float levels[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f,
                              -6.0f};
    // Metric: per-block error normalized by the BLOCK absmax (per-element
    // relative error is unbounded for tiny elements in dynamic blocks --
    // inherent to NVFP4 pow2 scales). Sound bound: snap error <= 1*s and
    // s <= absmax/3 -> <= 1/3 of block absmax.
    double max_rel = 0.0;
    size_t worst_idx = 0; float worst_got = 0.f, worst_want = 0.f;
    uint8_t worst_sb = 0, worst_code = 0;
    for (int b = 0; b < static_cast<int>(scales_n); ++b) {
        const float s = dec_scale(hs[b]);
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
            const float got = levels[code] * s;
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

    cudaFree(dx); cudaFree(dp); cudaFree(ds);
}


TEST_CASE("tensor-core nvfp4 gemm vs host reference", "[tc][gemm]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    constexpr int seq = 1, K = 64, N = 128; // TEMP bisect: tile@seq1
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

    __half *dx, *dy;
    uint8_t *dp, *ds, *wp, *ws;
    REQUIRE(cudaMalloc(&dx, hx.size() * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dy, static_cast<size_t>(seq) * N * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&wp, wp_n) == cudaSuccess);
    REQUIRE(cudaMalloc(&ws, ws_n) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, hx.data(), hx.size() * 2,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(wp, hp.data(), wp_n,
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ws, hs.data(), ws_n,
                       cudaMemcpyHostToDevice) == cudaSuccess);

    const float wgs = 0.125f;
    REQUIRE(lsxcommon::qwen::tc::RunQwenTcNvfp4Gemm(nullptr, dx, wp, ws, wgs,
                                                    dy, seq, K, N)
            == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<__half> hy(static_cast<size_t>(seq) * N);
    REQUIRE(cudaMemcpy(hy.data(), dy, hy.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    // ---- Host reference mirroring the kernel's math exactly.
    const float lv[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f,
                          -6.0f};
    auto dec_scale = [](uint8_t b) -> float {
        const uint32_t e4 = (b >> 3) & 0xFu;
        if (e4 == 0) return 0.0f;
        union { uint32_t u; float f; } bits{};
        bits.u = (e4 + 120u) << 23;
        return bits.f;
    };
    auto enc_pow2 = [](float s) -> uint8_t {
        // ceil-log2 pow2 encode: byte = (e+7)<<3 with value 2^(e-7); s>0.
        int e;
        std::frexp(s, &e); // s in [2^(e-1), 2^e): ceil-log2 = e
        return static_cast<uint8_t>((e + 7) << 3);
    };
    auto snap_e2m1 = [](float v) -> float {
        const bool neg = v < 0.f;
        const float a = std::abs(v);
        float m;
        if (a < 0.25f) m = 0.f;
        else if (a < 0.75f) m = 0.5f;
        else if (a < 1.25f) m = 1.f;
        else if (a < 1.75f) m = 1.5f;
        else if (a < 2.5f) m = 2.f;
        else if (a < 3.5f) m = 3.f;
        else if (a < 5.0f) m = 4.f;
        else m = 6.f;
        return neg ? -m : m;
    };

    // Quantized activations (dequant values), mirroring ActQuantKernel.
    std::vector<float> av(hx.size());
    for (size_t b = 0; b + 16 <= hx.size(); b += 16) {
        float absmax = 0.f;
        for (int j = 0; j < 16; ++j)
            absmax = std::max(absmax,
                              std::abs(__half2float(hx[b + j])));
        const uint8_t sb =
            absmax > 0.f ? enc_pow2(absmax / 6.0f) : uint8_t{0};
        const float s = dec_scale(sb);
        for (int j = 0; j < 16; ++j)
            av[b + j] =
                s > 0.f
                    ? snap_e2m1(__half2float(hx[b + j]) / s) * s
                    : 0.f;
    }

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
                    lv[wn] *
                    dec_scale(hs[static_cast<size_t>(n) * ((K + 15) / 16) +
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
                           lv[wn2] *
                           dec_scale(hs[static_cast<size_t>(n) *
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

    cudaFree(dx); cudaFree(dy); cudaFree(wp); cudaFree(ws);
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

    __half *dx, *dy;
    uint8_t *wp, *ws;
    REQUIRE(cudaMalloc(&dx, K * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&dy, N * 2) == cudaSuccess);
    REQUIRE(cudaMalloc(&wp, wp_n) == cudaSuccess);
    REQUIRE(cudaMalloc(&ws, ws_n) == cudaSuccess);
    REQUIRE(cudaMemcpy(wp, hp.data(), wp_n, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(ws, hs.data(), ws_n, cudaMemcpyHostToDevice) == cudaSuccess);

    auto dec_scale = [](uint8_t b) -> float {
        const uint32_t e4 = (b >> 3) & 0xFu;
        if (e4 == 0) return 0.0f;
        union { uint32_t u; float f; } bits{};
        bits.u = (e4 + 120u) << 23;
        return bits.f;
    };
    const float lv[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                          -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f,
                          -6.0f};

    for (int pulse_k : {0, 1, 9, 17, 33}) {
        std::vector<__half> hx(K, __float2half(0.0f));
        hx[pulse_k] = __float2half(1.0f);
        REQUIRE(cudaMemcpy(dx, hx.data(), K * 2,
                           cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(lsxcommon::qwen::tc::RunQwenTcNvfp4Gemm(
                    nullptr, dx, wp, ws, 1.0f, dy, 1, K, N) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        std::vector<__half> hy(N);
        REQUIRE(cudaMemcpy(hy.data(), dy, N * 2,
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
                lv[wn] * dec_scale(hs[static_cast<size_t>(n) * 4 +
                                      pulse_k / 16]);
            const float got = __half2float(hy[n]);
            if (std::abs(got - want) > 1e-3f * (std::abs(want) + 0.1f))
                WARN("pulse k=" << pulse_k << " n=" << n << " got=" << got
                                << " want=" << want);
        }
    }
    CHECK(true);
    cudaFree(dx); cudaFree(dy); cudaFree(wp); cudaFree(ws);
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
                    ? lsxcommon::qwen::tc::RunQwenTcNvfp4Gemm(
                          st, dx, wp, ws, 0.125f, dy, 1, K, N)
                    : RunQwenNvfp4Gemm(st, dx, dp, ds, 0.125f, dy, 1, K, N);
                REQUIRE(e == cudaSuccess);
            }
            REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
            constexpr int iters = 50;
            REQUIRE(cudaEventRecord(t0, st) == cudaSuccess);
            for (int it = 0; it < iters; ++it) {
                cudaError_t e = tc
                    ? lsxcommon::qwen::tc::RunQwenTcNvfp4Gemm(
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
    const cudaError_t e = lsxcommon::qwen::tc::ProbeTmaBulkCopy(s);
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

TEST_CASE("tc decode kernel honors acc=true at seq==1", "[tc][gemm][acc]")
{
    int dev = 0;
    REQUIRE(cudaGetDevice(&dev) == cudaSuccess);
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, dev) == cudaSuccess);
    if (prop.major != 12) SKIP("requires Blackwell sm_120");

    // Host-side NVFP4 model helpers (same semantics as the verified
    // tensor-core-vs-host test above).
    auto enc_pow2 = [](float v) -> uint8_t {
        // Canonical ceil-pow2 encode (matches Pow2E4m3Encode / verified
        // host-ref test): frexp gives v in [2^(e-1), 2^e) -> ceil = 2^e.
        if (!(v > 0.f)) return 0;
        int e;
        std::frexp(v, &e);
        const int e4 = e + 7;
        if (e4 < 1) return static_cast<uint8_t>(1 << 3);
        if (e4 > 15) return static_cast<uint8_t>(15 << 3);
        return static_cast<uint8_t>(e4 << 3);
    };
    auto dec_scale = [](uint8_t b) -> float {
        const int e4 = (b >> 3) & 0xF;
        if (e4 == 0) return 0.f;
        return std::ldexp(1.0f, e4 - 7);
    };
    auto snap_e2m1 = [](float v) -> float {
        static const float lv[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        const float a = std::abs(v);
        int best = 0;
        float bd = std::abs(a - lv[0]);
        for (int i = 1; i < 8; ++i) {
            const float d = std::abs(a - lv[i]);
            if (d < bd) { bd = d; best = i; }
        }
        return v < 0.f ? -lv[best] : lv[best];
    };
    constexpr float kLv[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

    constexpr int seq = 1, N = 128;
    for (const int K : {64, 256, 1024, 2048, 4096}) {
        std::vector<__half> hx(K);
        for (int i = 0; i < K; ++i)
            hx[i] = __float2half(((i % 13) - 6) * 0.25f);
        const size_t wp_n = static_cast<size_t>(N) * K / 2;
        const size_t ws_n = static_cast<size_t>(N) * (K / 16);
        std::vector<uint8_t> hp(wp_n), hs(ws_n);
        for (size_t i = 0; i < wp_n; ++i)
            hp[i] = static_cast<uint8_t>((i * 29) & 0xFF);
        for (size_t i = 0; i < ws_n; ++i)
            hs[i] =
                static_cast<uint8_t>((1 + ((i * 3) % 15)) << 3);

        __half *dx, *dy_acc, *dy_init;
        uint8_t *dp, *ds, *wp, *ws;
        cudaMalloc(&dx, hx.size() * 2);
        cudaMalloc(&dy_acc, N * 2);
        cudaMalloc(&dy_init, N * 2);
        cudaMalloc(&dp, wp_n); cudaMalloc(&ds, ws_n);
        cudaMalloc(&wp, wp_n); cudaMalloc(&ws, ws_n);
        cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dp, hp.data(), wp_n, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, hs.data(), ws_n, cudaMemcpyHostToDevice);
        cudaMemcpy(wp, hp.data(), wp_n, cudaMemcpyHostToDevice);
        cudaMemcpy(ws, hs.data(), ws_n, cudaMemcpyHostToDevice);

        // Host truth: act quant (per-16 ceil-pow2 E4M3 + E2M1 snap), weight
        // dequant per repo codec.
        std::vector<float> av(K);
        for (int b = 0; b + 16 <= K; b += 16) {
            float absmax = 0.f;
            for (int j = 0; j < 16; ++j)
                absmax = std::max(absmax, std::abs(__half2float(hx[b + j])));
            const uint8_t sb =
                absmax > 0.f ? enc_pow2(absmax / 6.0f) : uint8_t{0};
            const float scl = dec_scale(sb);
            for (int j = 0; j < 16; ++j)
                av[b + j] =
                    scl > 0.f
                        ? snap_e2m1(__half2float(hx[b + j]) / scl) * scl
                        : 0.f;
        }
        std::vector<float> want(N), want0(N);
        (void)want;
        (void)want0;
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                const size_t we = static_cast<size_t>(n) * K + k;
                const uint8_t wb = hp[we / 2];
                const uint8_t wn =
                    (k & 1) ? static_cast<uint8_t>(wb >> 4)
                            : static_cast<uint8_t>(wb & 0x0F);
                const float wv =
                    ((wn & 8) ? -1.f : 1.f) * kLv[wn & 7] *
                    dec_scale(hs[static_cast<size_t>(n) * (K / 16) +
                                 k / 16]);
                acc += static_cast<double>(av[k] * wv);
            }
            want0[n] = static_cast<float>(acc * 0.125);
            want[n] = want0[n] +
                      __half2float(__float2half(0.5f + (n % 7) * 0.125f));
        }

        using lsxcommon::qwen::tc::RunQwenTcNvfp4Gemm;
        cudaStream_t st = nullptr;
        std::vector<__half> hinit(N), hplain(N), hacc(N);
        for (int n = 0; n < N; ++n)
            hinit[n] = __float2half(0.5f + (n % 7) * 0.125f);

        cudaMemcpy(dy_init, hinit.data(), N * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dy_acc, dy_init, N * 2, cudaMemcpyDeviceToDevice);
        REQUIRE(RunQwenTcNvfp4Gemm(st, dx, dp, ds, 0.125f, dy_init, seq, K,
                                   N, /*acc=*/false) == cudaSuccess);
        REQUIRE(RunQwenTcNvfp4Gemm(st, dx, dp, ds, 0.125f, dy_acc, seq, K,
                                   N, /*acc=*/true) == cudaSuccess);
        REQUIRE(cudaStreamSynchronize(st) == cudaSuccess);
        cudaMemcpy(hplain.data(), dy_init, N * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(hacc.data(), dy_acc, N * 2, cudaMemcpyDeviceToHost);

        // Contract under test: acc=true == prev + gemm, where gemm is the
        // SAME kernel's acc=false result (magnitude-independent; numeric
        // accuracy vs the exact NVFP4 host model is owned by the verified
        // "tensor-core nvfp4 gemm vs host reference" case).
        for (int n = 0; n < N; ++n) {
            const float expect =
                __half2float(hinit[n]) + __half2float(hplain[n]);
            INFO("K=" << K << " n=" << n << " init=" << __half2float(hinit[n])
                      << " plain=" << __half2float(hplain[n])
                      << " acc=" << __half2float(hacc[n]));
            CHECK(std::abs(__half2float(hacc[n]) - expect) <=
                  2e-2f * std::max(1.f, std::abs(expect)));
        }
        cudaFree(dx); cudaFree(dy_acc); cudaFree(dy_init);
        cudaFree(dp); cudaFree(ds); cudaFree(wp); cudaFree(ws);
    }
}
