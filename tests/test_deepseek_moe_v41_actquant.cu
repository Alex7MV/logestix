// tests/test_deepseek_moe_v41_actquant.cu — DeepSeek-V4.1-Flash FP4 activation
// quantize-dequant (E2M1) used by the indexer (E8M0 pow2 scale). Mirrors
// kernel.py:fp4_act_quant and is checked against the host reference.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "deepseek_moe_v41_dequant.h"
#include "deepseek_moe_v41_device_math.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

uint16_t H(float v) { return __half_as_ushort(__float2half_rn(v)); }
float F(uint16_t b) { return __half2float(__ushort_as_half(b)); }

} // namespace

TEST_CASE("deepseek-v41 fp4 e8m0 activation quant matches the host reference",
          "[deepseek-v41][actquant]")
{
    const int rows = 2, n = 64, block = 32;
    std::vector<float> xf(rows * n);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < n; ++j)
            xf[r * n + j] =
                0.7f *
                (static_cast<float>((r * 13 + j * 7) % 19) - 9.0f) / 3.0f;

    std::vector<uint16_t> xh(xf.size());
    for (size_t i = 0; i < xf.size(); ++i) xh[i] = H(xf[i]);

    uint16_t *dx = nullptr;
    REQUIRE(cudaMalloc(&dx, xh.size() * 2) == cudaSuccess);
    REQUIRE(cudaMemcpy(dx, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice) ==
            cudaSuccess);
    REQUIRE(lsxcommon::deepseek_v41::LaunchFp4ActQuantInplace(
                dx, rows, n, block, true, nullptr) == cudaSuccess);
    std::vector<uint16_t> got(xh.size());
    REQUIRE(cudaMemcpy(got.data(), dx, xh.size() * 2,
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int r = 0; r < rows; ++r) {
        std::vector<float> rr(xf.begin() + r * n, xf.begin() + (r + 1) * n);
        lsxcommon::deepseek_v41::Fp4ActQuantRef(rr.data(), n, block, true);
        for (int j = 0; j < n; ++j) {
            const float v = F(got[r * n + j]);
            CHECK(v == Catch::Approx(rr[j]).margin(1e-3f));
            // On the E2M1 x 2^k grid the normalized significand is 1.0 or 1.5.
            const float a = std::fabs(v);
            if (a > 0.0f) {
                const float mant = std::ldexp(a, -std::ilogb(a));
                CHECK((mant == 1.0f || mant == 1.5f));
            }
        }
    }

    cudaFree(dx);
}
