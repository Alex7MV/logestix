#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

constexpr int kMinCudaRuntime = 13020;

// "toolkit=13.5 cudart=13050 nvcc=13.5" -> "13.5"
std::string StampToolkit(const std::string &stamp)
{
    const auto at = stamp.find("toolkit=");
    if (at == std::string::npos) return {};
    const auto from = at + 8;
    const auto to = stamp.find(' ', from);
    return stamp.substr(from, to == std::string::npos ? std::string::npos
                                                      : to - from);
}

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

// The one thing CMake cannot see after the fact: an object built from an older
// toolkit's headers, linked against the newer runtime. Measured 2026-11-05 —
// 1250 of 1460 objects were 13.4 headers against a 13.5 runtime and every build
// step still "succeeded". CUDART_VERSION here is what THIS TU was compiled
// with; the stamp is what the tree was configured with; the linked runtime is
// what the loader got. All three must agree.
TEST_CASE("cuda_toolkit_guard: the build tree matches the headers I compiled with",
          "[cuda_toolkit_guard]") {
    std::FILE *f = std::fopen(LSX_CUDA_TOOLKIT_STAMP, "r");
    if (!f) {
        FAIL("no build-toolkit stamp at " << LSX_CUDA_TOOLKIT_STAMP
             << " — the tree was configured before the stamp existed, or the "
                "file was removed; wipe the build directory and reconfigure");
    }
    char buf[256] = {0};
    const auto n = std::fread(buf, 1, sizeof(buf) - 1, f);
    std::fclose(f);
    const std::string stamp(buf, n);
    INFO("stamp: " << stamp);

    const int major = CUDART_VERSION / 1000;
    const int minor = (CUDART_VERSION % 1000) / 10;
    const std::string here = std::to_string(major) + "." + std::to_string(minor);
    const std::string configured = StampToolkit(stamp);
    INFO("this TU: " << here << " (CUDART_VERSION=" << CUDART_VERSION
                     << "), tree: " << configured);
    REQUIRE(configured == here);

    // And the runtime actually linked must be that same toolkit, not a newer
    // one that the loader picked up around the objects.
    int runtime = 0;
    REQUIRE(cudaRuntimeGetVersion(&runtime) == cudaSuccess);
    REQUIRE(runtime == CUDART_VERSION);
}
