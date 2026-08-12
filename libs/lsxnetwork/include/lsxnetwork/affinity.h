#pragma once

#include <cstdint>

namespace lsxnetwork {

/// Hard-pins the calling thread to the host's physical cores 0..3.
///
/// This isolates network I/O worker threads from the downstream CUDA compute
/// paths: the affinity mask is strictly `{0,1,2,3}`, so no executor ever
/// migrates onto cores shared with the inference pipeline.
class CpuAffinity {
public:
    /// Number of host cores the network layer is allowed to run on.
    static constexpr unsigned kCoreCount = 4;

    /// Bitmask of the allowed cores (core i <-> bit i).
    static constexpr std::uint32_t kAllowedMask = 0x0000000Fu;

    /// Pins the current thread to cores 0..3. Returns 0 on success, otherwise a
    /// non-zero OS error code (errno on Linux, GetLastError on Windows).
    static int PinToCores0To3() noexcept;

    /// Returns the bitmask of the cores the calling thread is currently
    /// scheduled onto. TRACE-quality helper for diagnostics and validation.
    static std::uint32_t CurrentAllowedCores() noexcept;
};

}  // namespace lsxnetwork