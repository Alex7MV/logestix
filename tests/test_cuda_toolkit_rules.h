#pragma once

// Rules shared between tests/test_cuda_toolkit_guard.cu (which needs a real
// toolkit) and tests/test_cuda_toolkit_rules.cpp (which can run anywhere).
//
// WHY THIS SPLIT EXISTS
//
// The guard could not be verified where it was written. Its four toolchain
// assumptions live in a `.cu` file, so proving "it now accepts CUDA 12.6" needs a
// 12.6 toolkit and a JetPack machine — neither of which a change to the guard can
// assume. What CAN be proven on any host is the arithmetic: given a CUDART number,
// is it above the floor and inside the supported band? That is the whole content
// of the threshold, so it is a pure function here, tested on the build host, and
// called by the `.cu` case.
//
// Deliberately no CUDA includes: this header is usable from the host test, from a
// .cu TU and from a future CMake probe without dragging a runtime in.

#include <string>

namespace cuda_toolkit {

// The library floor: CUDA 12.6, which is JetPack 6.2.1 on Orin (CUDA 12.6) — the
// oldest toolkit this tree builds for. Below it the tree does not build at all
// (kind::mxf4nvf4 PTX needs 12.8 for the Blackwell path, but that object is only
// built for 12x archs, so the LIBRARY floor stays at the JetPack release).
inline constexpr int kMinCudaRuntime = 12060;

// Where `cuda_fp4.h` first appears. The e2m1 cases are gated on
// `__has_include(<cuda_fp4.h>)` rather than on this number — the header's
// presence is the fact, and a version constant would have to be right about
// NVIDIA's packaging as well as its versioning. Kept for the message a SKIP
// prints, so an operator reading the log learns WHY the cases did not run.
inline constexpr int kFp4HeaderRuntime = 12080;

// "" when `cudartVersion` (CUDART_VERSION's shape, e.g. 13050) is at or above the
// floor; otherwise a message naming both numbers.
//
// No upper bound and no major equality: this tree builds under 12.6 (Orin) and
// 13.5 (x86), and the check exists to catch "the toolkit is older than the code
// requires", not to pin a vendor release. Pinning the major is what made the
// guard fail on JetPack; the version compatibility rules (minor-version
// compatibility, driver skew) are DriverCompatibility's job, not this one's.
inline std::string RuntimeFloorProblem(int cudartVersion)
{
    if (cudartVersion >= kMinCudaRuntime) return {};
    const int major = cudartVersion / 1000;
    const int minor = (cudartVersion % 1000) / 10;
    const int f_major = kMinCudaRuntime / 1000;
    const int f_minor = (kMinCudaRuntime % 1000) / 10;
    return "CUDA runtime " + std::to_string(major) + "." + std::to_string(minor) +
           " is older than the floor " + std::to_string(f_major) + "." +
           std::to_string(f_minor) +
           " (JetPack 6.2.1 / CUDA 12.6 is the oldest supported toolkit)";
}

// True when the e2m1/F2 cases can run in this translation unit. The .cu calls
// this through `__has_include`; a host test cannot answer it, so it is not a
// function there. See the note above for why the constant is not the gate.
inline constexpr bool kFp4HeaderExpectedAt(int cudartVersion)
{
    return cudartVersion >= kFp4HeaderRuntime;
}

} // namespace cuda_toolkit
