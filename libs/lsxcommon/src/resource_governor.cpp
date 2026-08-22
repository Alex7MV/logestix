#include "lsxcommon/resource_governor.h"

#include <algorithm>
#include <spdlog/spdlog.h>

namespace lsxcommon {

ResourceGovernor &ResourceGovernor::Instance()
{
    static ResourceGovernor instance;  // Meyers singleton
    return instance;
}

bool ResourceGovernor::Initialize(const GovernorOptions &opts)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (initialized_) {
        spdlog::debug("[governor] already initialized (first options win)");
        return true;
    }
    opts_ = opts;
    hw_ = opts.override_hw ? opts.hw_override : ProbeHardware();
    spdlog::info("[governor] cpu={} ram={}/{} gpu='{}' sm={} vram={}/{}",
                 hw_.cpu_cores, hw_.ram_available_bytes >> 20,
                 hw_.ram_total_bytes >> 20, hw_.gpu_name, hw_.gpu_sm_count,
                 hw_.vram_free_bytes >> 20, hw_.vram_total_bytes >> 20);
    initialized_ = true;
    return true;
}

void ResourceGovernor::Shutdown()
{
    std::lock_guard<std::mutex> lk(mu_);
    initialized_ = false;
    vram_committed_ = 0;
    ram_committed_ = 0;
}

bool ResourceGovernor::IsInitialized() const noexcept
{
    std::lock_guard<std::mutex> lk(mu_);
    return initialized_;
}

const HardwareProfile &ResourceGovernor::hardware() const { return hw_; }
const GovernorOptions &ResourceGovernor::options() const { return opts_; }

std::uint64_t ResourceGovernor::CommittedVramBytes() const
{
    std::lock_guard<std::mutex> lk(mu_);
    return vram_committed_;
}

std::uint64_t ResourceGovernor::CommittedRamBytes() const
{
    std::lock_guard<std::mutex> lk(mu_);
    return ram_committed_;
}

std::uint64_t ResourceGovernor::GrantableVramLocked() const
{
    return hw_.vram_total_bytes > opts_.vram_keep_out_bytes
               ? hw_.vram_total_bytes - opts_.vram_keep_out_bytes
               : 0;
}

// Defined in later tasks (declared now so the header stays stable):
HardwareProfile BuildHardwareProfile(const HardwareQueries &q) { (void)q; return {}; }
HardwareProfile ProbeHardware() { return {}; }
bool IsVramLease(LeaseKind kind) { (void)kind; return true; }
Lease::~Lease() = default;
Lease::Lease(Lease &&) noexcept = default;
Lease &Lease::operator=(Lease &&) noexcept = default;
void Lease::Release() { gov_ = nullptr; }
bool Lease::Resize(std::uint64_t) { return false; }
Lease ResourceGovernor::Acquire(LeaseKind, std::uint64_t, LeasePolicy) { return {}; }
void ResourceGovernor::ReleaseLeaseBytes(LeaseKind, std::uint64_t) {}
bool ResourceGovernor::ResizeLease(Lease &, std::uint64_t) { return false; }
bool ResourceGovernor::ReconcileVram(std::uint64_t, std::uint64_t) { return true; }
bool ResourceGovernor::ReconcileVram() { return true; }

} // namespace lsxcommon
