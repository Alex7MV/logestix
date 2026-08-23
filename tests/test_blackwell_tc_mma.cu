// TDD — Phase 3 Task 1b: fragment-mapping calibration for the sm_120a
// block-scaled NVFP4 mma. Places a single e2m1 '1.0' at one packed operand
// slot per warp-launch and records which D cell lights up, empirically
// deriving the A/B register layouts (PTX doc formulas are ambiguous for
// kind::mxf4nvf4 packing order).
#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

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
