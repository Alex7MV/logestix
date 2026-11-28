#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>
#include <cuda_fp8.h>

// cuda_fp4.h arrived in CUDA 12.8, so a bare include is a COMPILE ERROR on
// JetPack 6.2.1 (CUDA 12.6) — before any SKIP can protect the case. The guard has
// to build on the oldest toolkit this tree supports, so the header and the e2m1
// cases behind it are gated on its presence, which is the fact itself.
//
// LSX_TOOLKIT_GUARD_HIDE_FP4_HEADER is a TEST hook, used by exactly one target
// (test_cuda_toolkit_guard_nofp4): it compiles this same file as if the header
// were absent, which is the only way to exercise the 12.6 branch of a guard
// written on a 13.5 host. It is never defined for the real target.
#if __has_include(<cuda_fp4.h>) && !defined(LSX_TOOLKIT_GUARD_HIDE_FP4_HEADER)
#include <cuda_fp4.h>
#define LSX_HAS_CUDA_FP4 1
#else
#define LSX_HAS_CUDA_FP4 0
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

// The floor and its arithmetic, shared with tests/test_cuda_toolkit_rules.cpp so
// the threshold is checkable where no toolkit is installed.
#include "test_cuda_toolkit_rules.h"

// The supported device set, from the ONE table that owns it (design §2.3):
// {12.0, 12.1} Blackwell consumer, {8.6, 8.7} Ampere GA10x. Asking the table is
// what stops this guard from being a second, hand-written copy that goes stale
// the day a target is added.
#include "lsxcommon/hardware_target.h"

namespace {

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

// The DRIVER half. `cudaRuntimeGetVersion` reports the cudart the binary links
// (i.e. CUDART_VERSION, always equal), so it says nothing about the driver in
// the kernel. CUDA's minor-version compatibility lets a 13.y runtime run on a
// 13.x driver with x <= y, which is the state this host is in (runtime 13.5,
// driver 13.4) and is supported — but only for APIs that existed in 13.x, so it
// has to be VISIBLE rather than assumed. A driver in another major cannot run
// this runtime at all.
// Returns an empty string when the pair is usable.
std::string DriverCompatibility(int driverVersion, int runtimeVersion)
{
    const int d_major = driverVersion / 1000;
    const int r_major = runtimeVersion / 1000;
    if (d_major != r_major)
        return "driver supports CUDA " + std::to_string(d_major) +
               ".x but the runtime is " + std::to_string(r_major) +
               ".x — a driver in another major cannot run this runtime";
    return {};
}

} // namespace

#if defined(LSX_TOOLKIT_GUARD_HIDE_FP4_HEADER)
// The simulated-CUDA-12.6 build (test_cuda_toolkit_guard_nofp4). Its value is
// that THIS FILE COMPILED with no cuda_fp4.h in reach and the rest of the guard
// still ran; the assertion keeps the configuration from going green while
// silently taking the other branch.
TEST_CASE("cuda_toolkit_guard: built without cuda_fp4.h (simulated 12.6)",
          "[cuda_toolkit_guard]") {
    REQUIRE(LSX_HAS_CUDA_FP4 == 0);
}
#endif

TEST_CASE("cuda_toolkit_guard: runtime at or above the 12.6 floor, headers match",
          "[cuda_toolkit_guard]") {
    // No major equality: this tree builds under 12.6 (JetPack 6.2.1, Orin) and
    // 13.5 (x86). "== 13" was a vendor-release pin dressed as a compatibility
    // check, and it is the single line that made the guard fail on JetPack.
    INFO("CUDART_VERSION=" << CUDART_VERSION << " floor="
                           << cuda_toolkit::kMinCudaRuntime);
    const std::string floor_problem =
        cuda_toolkit::RuntimeFloorProblem(CUDART_VERSION);
    INFO(floor_problem);
    REQUIRE(floor_problem.empty());

    int runtime = 0;
    REQUIRE(cudaRuntimeGetVersion(&runtime) == cudaSuccess);
    INFO("CUDART_VERSION=" << CUDART_VERSION << " runtime=" << runtime);
    REQUIRE(runtime == CUDART_VERSION);
    // The LINKED runtime must clear the floor too, not just the compile-time one:
    // an older cudart found at load time is the same unsupported toolkit.
    REQUIRE(cuda_toolkit::RuntimeFloorProblem(runtime).empty());
}

// The e2m1 cases need the header, which is absent before CUDA 12.8. A SKIP with
// the reason, never a silent pass: AGENTS.md is explicit that a skipped gate is
// not a satisfied one, so the log has to say which toolkit lacks what.
TEST_CASE("cuda_toolkit_guard: cuda_fp4.h e2m1 RNE, SATFINITE, NaN to +MAXNORM",
          "[cuda_toolkit_guard]") {
#if !LSX_HAS_CUDA_FP4
    SKIP("cuda_fp4.h is not available to this translation unit (it arrived in "
         "CUDA 12.8; JetPack 6.2.1 is 12.6). CUDART_VERSION="
         << CUDART_VERSION << "; the e2m1 conversion cases need the header");
#else
    REQUIRE(__nv_cvt_float_to_fp4(6.5f, __NV_E2M1, cudaRoundNearest) == 0x7);
    REQUIRE(__nv_cvt_float_to_fp4(-6.5f, __NV_E2M1, cudaRoundNearest) == 0xF);
    REQUIRE(__nv_cvt_float_to_fp4(std::nanf(""), __NV_E2M1, cudaRoundNearest) == 0x7);

    REQUIRE(__nv_cvt_float_to_fp4(1.25f, __NV_E2M1, cudaRoundNearest) == 0x2);
    REQUIRE(__nv_cvt_float_to_fp4(1.75f, __NV_E2M1, cudaRoundNearest) == 0x4);

    __nv_fp4_e2m1 v;
    v.__x = 0x7;
    REQUIRE(static_cast<float>(v) == 6.0f);
#endif
}

TEST_CASE("cuda_toolkit_guard: cuda_fp4.h pack order a high nibble b low nibble",
          "[cuda_toolkit_guard]") {
#if !LSX_HAS_CUDA_FP4
    SKIP("cuda_fp4.h is not available to this translation unit (it arrived in "
         "CUDA 12.8; JetPack 6.2.1 is 12.6). CUDART_VERSION="
         << CUDART_VERSION << "; the fp4x2 packing case needs the header");
#else
    const auto ab = __nv_cvt_float2_to_fp4x2(make_float2(1.0f, 0.5f),
                                             __NV_E2M1, cudaRoundNearest);
    const auto ba = __nv_cvt_float2_to_fp4x2(make_float2(0.5f, 1.0f),
                                             __NV_E2M1, cudaRoundNearest);
    INFO("ab=0x" << std::hex << static_cast<int>(ab) << " ba=0x"
                 << static_cast<int>(ba));
    REQUIRE(static_cast<int>(ab) == 0x12);
    REQUIRE(static_cast<int>(ba) == 0x21);
#endif
}

TEST_CASE("cuda_toolkit_guard: present devices are in the supported set",
          "[cuda_toolkit_guard]") {
    // "prop.major == 12" was a hand-written copy of the supported set, and it
    // fails on the 3090 (8.6) and the Orin (8.7) that this tree now builds for.
    // The capability table owns that fact (design §2.3); the guard asks it, so a
    // target added there cannot leave this check behind.
    int ndev = 0;
    const cudaError_t e = cudaGetDeviceCount(&ndev);
    if (e != cudaSuccess || ndev == 0) {
        SKIP("no CUDA device");
    }
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp prop{};
        REQUIRE(cudaGetDeviceProperties(&prop, d) == cudaSuccess);
        const auto target =
            lsxcommon::hw::TargetFromComputeCapability(prop.major, prop.minor);
        INFO("device " << d << ": " << prop.name << " CC " << prop.major << "."
                       << prop.minor << " -> target '" << target.Id() << "'");
        // A device outside {12.0, 12.1, 8.6, 8.7} is reported BY NAME, because
        // "unsupported device" sends the operator nowhere.
        REQUIRE(target.vendor != lsxcommon::hw::Vendor::kUnknown);
        REQUIRE(lsxcommon::hw::SupportedCodecs(target) != 0);
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

TEST_CASE("cuda_toolkit_guard: driver-compatibility rule", "[cuda_toolkit_guard]") {
    // Same pair, newer driver, older driver in the same major: all usable.
    CHECK(DriverCompatibility(13040, 13050).empty());   // this host
    CHECK(DriverCompatibility(13050, 13050).empty());
    CHECK(DriverCompatibility(13000, 13050).empty());
    // Another major cannot run this runtime.
    CHECK(!DriverCompatibility(12090, 13050).empty());
    CHECK(!DriverCompatibility(14000, 13050).empty());
}

TEST_CASE("cuda_toolkit_guard: the driver can run this runtime",
          "[cuda_toolkit_guard]") {
    // cudaDriverGetVersion reports the DRIVER's CUDA version through the
    // runtime API, so this needs no libcuda link (the target links cudart).
    int driver = 0;
    const cudaError_t rc = cudaDriverGetVersion(&driver);
    if (rc != cudaSuccess) SKIP("no CUDA driver on this host");
    REQUIRE(driver > 0);

    const int d_major = driver / 1000, d_minor = (driver % 1000) / 10;
    const int r_major = CUDART_VERSION / 1000, r_minor = (CUDART_VERSION % 1000) / 10;
    INFO("driver 13.4-style version " << d_major << "." << d_minor
                                      << " (raw " << driver << "), runtime "
                                      << r_major << "." << r_minor);
    if (d_minor < r_minor) {
        // Not a failure — minor-version compatibility — but it means 13.
        // feature-specific APIs may be absent at runtime, so it must be said.
        WARN("running a " << r_major << "." << r_minor << " runtime on a "
                          << d_major << "." << d_minor
                          << " driver: minor-version compatibility, so APIs "
                             "introduced after " << d_major << "." << d_minor
                          << " are unavailable at runtime");
    }
    const std::string problem = DriverCompatibility(driver, CUDART_VERSION);
    INFO(problem);
    REQUIRE(problem.empty());
}
