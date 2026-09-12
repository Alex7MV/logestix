#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kMinCudaRuntime = 13020;

} // namespace

TEST_CASE("cuda_toolkit_guard: runtime 13.x, floor 13.2, header/runtime match",
          "[cuda_toolkit_guard]") {
    REQUIRE(CUDART_VERSION >= kMinCudaRuntime);
    REQUIRE(CUDART_VERSION / 1000 == 13);

    int runtime = 0;
    REQUIRE(cudaRuntimeGetVersion(&runtime) == cudaSuccess);
    INFO("CUDART_VERSION=" << CUDART_VERSION << " runtime=" << runtime);
    REQUIRE(runtime == CUDART_VERSION);
}

TEST_CASE("cuda_toolkit_guard: cuda_fp4.h e2m1 RNE, SATFINITE, NaN to +MAXNORM",
          "[cuda_toolkit_guard]") {
    REQUIRE(__nv_cvt_float_to_fp4(6.5f, __NV_E2M1, cudaRoundNearest) == 0x7);
    REQUIRE(__nv_cvt_float_to_fp4(-6.5f, __NV_E2M1, cudaRoundNearest) == 0xF);
    REQUIRE(__nv_cvt_float_to_fp4(std::nanf(""), __NV_E2M1, cudaRoundNearest) == 0x7);

    REQUIRE(__nv_cvt_float_to_fp4(1.25f, __NV_E2M1, cudaRoundNearest) == 0x2);
    REQUIRE(__nv_cvt_float_to_fp4(1.75f, __NV_E2M1, cudaRoundNearest) == 0x4);

    __nv_fp4_e2m1 v;
    v.__x = 0x7;
    REQUIRE(static_cast<float>(v) == 6.0f);
}

TEST_CASE("cuda_toolkit_guard: cuda_fp4.h pack order a high nibble b low nibble",
          "[cuda_toolkit_guard]") {
    const auto ab = __nv_cvt_float2_to_fp4x2(make_float2(1.0f, 0.5f),
                                             __NV_E2M1, cudaRoundNearest);
    const auto ba = __nv_cvt_float2_to_fp4x2(make_float2(0.5f, 1.0f),
                                             __NV_E2M1, cudaRoundNearest);
    INFO("ab=0x" << std::hex << static_cast<int>(ab) << " ba=0x"
                 << static_cast<int>(ba));
    REQUIRE(static_cast<int>(ab) == 0x12);
    REQUIRE(static_cast<int>(ba) == 0x21);
}

TEST_CASE("cuda_toolkit_guard: present devices are Blackwell CC 12.x",
          "[cuda_toolkit_guard]") {
    int ndev = 0;
    const cudaError_t e = cudaGetDeviceCount(&ndev);
    if (e != cudaSuccess || ndev == 0) {
        SKIP("no CUDA device");
    }
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp prop{};
        REQUIRE(cudaGetDeviceProperties(&prop, d) == cudaSuccess);
        INFO("device " << d << ": " << prop.name << " CC " << prop.major << "."
                       << prop.minor);
        REQUIRE(prop.major == 12);
    }
}
