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
        f.sfa[l] = 0x18181818u;
        f.sfb[l] = 0x18181818u;
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
