#include "lsxcommon/resource_governor.h"

#include <cuda_runtime.h>

// windows.h must come AFTER cuda_runtime.h: its macro pollution breaks
// declarations inside the CUDA headers otherwise.
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

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

// ── Hardware probing ────────────────────────────────────────────────────────

HardwareProfile BuildHardwareProfile(const HardwareQueries &q)
{
    HardwareProfile hw;
    if (q.cpu_cores) hw.cpu_cores = q.cpu_cores();
    if (q.ram_bytes) q.ram_bytes(&hw.ram_total_bytes, &hw.ram_available_bytes);
    if (q.gpu_info) q.gpu_info(&hw.gpu_name, &hw.gpu_sm_count);
    if (q.vram_bytes) q.vram_bytes(&hw.vram_total_bytes, &hw.vram_free_bytes);
    return hw;
}

#if defined(_WIN32)
static int QueryCoresWin()
{
    SYSTEM_INFO si{};
    GetSystemInfo(&si);
    return static_cast<int>(si.dwNumberOfProcessors);
}
static void QueryRamWin(std::uint64_t *total, std::uint64_t *available)
{
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    *total = s.ullTotalPhys;
    *available = s.ullAvailPhys;
}
#else
static int QueryCoresPosix()
{
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? static_cast<int>(n) : 0;
}
static void QueryRamPosix(std::uint64_t *total, std::uint64_t *available)
{
    struct sysinfo si{};
    if (sysinfo(&si) != 0) { *total = 0; *available = 0; return; }
    *total = static_cast<std::uint64_t>(si.totalram) * si.mem_unit;
    *available = static_cast<std::uint64_t>(si.freeram) * si.mem_unit;
}
#endif

static void QueryGpuInfo(std::string *name, int *sm_count)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        *name = "";
        *sm_count = 0;
        return;
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        *name = "";
        *sm_count = 0;
        return;
    }
    *name = p.name;
    *sm_count = p.multiProcessorCount;
}
static void QueryVram(std::uint64_t *total, std::uint64_t *free_b)
{
    std::size_t fb = 0, tb = 0;
    if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) { *total = 0; *free_b = 0; return; }
    *total = tb;
    *free_b = fb;
}

HardwareProfile ProbeHardware()
{
#if defined(_WIN32)
    HardwareQueries q{QueryCoresWin, QueryRamWin, QueryGpuInfo, QueryVram};
#else
    HardwareQueries q{QueryCoresPosix, QueryRamPosix, QueryGpuInfo, QueryVram};
#endif
    return BuildHardwareProfile(q);
}

} // namespace lsxcommon
