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

// ── Adaptive resize + drift reconciliation ──────────────────────────────────

bool ResourceGovernor::ResizeLease(Lease &lease, std::uint64_t new_bytes)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (!initialized_ || lease.gov_ != this) return false;
    if (lease.bytes_ == new_bytes) return true;

    new_bytes = std::clamp(new_bytes, lease.min_bytes_, lease.max_bytes_);
    if (new_bytes == lease.bytes_) return true;

    const bool vram = IsVramLease(lease.kind_);
    std::uint64_t &committed = vram ? vram_committed_ : ram_committed_;
    const std::uint64_t grantable =
        vram ? GrantableVramLocked() : hw_.ram_total_bytes;

    if (new_bytes > lease.bytes_ &&
        committed - lease.bytes_ + new_bytes > grantable) {
        spdlog::info("[governor] resize {} -> {} REFUSED (budget occupied)",
                     lease.bytes_, new_bytes);
        return false;  // soft: caller keeps the current size
    }
    committed = committed - lease.bytes_ + new_bytes;
    lease.bytes_ = new_bytes;
    return true;
}

bool ResourceGovernor::ReconcileVram(std::uint64_t queried_free_vram_bytes,
                                     std::uint64_t slack_bytes)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (!initialized_) return false;
    if (queried_free_vram_bytes + vram_committed_ >
        hw_.vram_total_bytes + slack_bytes) {
        spdlog::warn("[governor] VRAM bookkeeping drift: free {} MiB + "
                     "committed {} MiB > total {} MiB + slack {} MiB",
                     queried_free_vram_bytes >> 20, vram_committed_ >> 20,
                     hw_.vram_total_bytes >> 20, slack_bytes >> 20);
        return false;
    }
    return true;
}

bool ResourceGovernor::ReconcileVram()
{
    std::size_t fb = 0, tb = 0;
    if (cudaMemGetInfo(&fb, &tb) != cudaSuccess) return false;
    return ReconcileVram(fb, 256ull << 20);
}

// ── Leases ──────────────────────────────────────────────────────────────────

bool IsVramLease(LeaseKind kind)
{
    switch (kind) {
        case LeaseKind::kPermanentWeights:
        case LeaseKind::kExpertCache:
        case LeaseKind::kInferArena:
            return true;
        case LeaseKind::kHostPinned:
        case LeaseKind::kHostRam:
            return false;
    }
    return true;
}

Lease::~Lease() { Release(); }

Lease::Lease(Lease &&other) noexcept
    : gov_(other.gov_), kind_(other.kind_), bytes_(other.bytes_),
      min_bytes_(other.min_bytes_), max_bytes_(other.max_bytes_)
{
    other.gov_ = nullptr;
    other.bytes_ = 0;
}

Lease &Lease::operator=(Lease &&other) noexcept
{
    if (this != &other) {
        Release();
        gov_ = other.gov_;
        kind_ = other.kind_;
        bytes_ = other.bytes_;
        min_bytes_ = other.min_bytes_;
        max_bytes_ = other.max_bytes_;
        other.gov_ = nullptr;
        other.bytes_ = 0;
    }
    return *this;
}

void Lease::Release()
{
    if (!gov_) return;
    gov_->ReleaseLeaseBytes(kind_, bytes_);
    gov_ = nullptr;
    bytes_ = 0;
}

bool Lease::Resize(std::uint64_t new_bytes)
{
    if (!gov_) return false;
    return gov_->ResizeLease(*this, new_bytes);
}

Lease ResourceGovernor::Acquire(LeaseKind kind, std::uint64_t bytes,
                                LeasePolicy policy)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (!initialized_) {
        spdlog::error("[governor] Acquire before Initialize()");
        return {};
    }
    const bool vram = IsVramLease(kind);
    std::uint64_t &committed = vram ? vram_committed_ : ram_committed_;
    const std::uint64_t grantable =
        vram ? GrantableVramLocked() : hw_.ram_total_bytes;

    std::uint64_t lo = policy.min_bytes, hi = policy.max_bytes;
    if (policy.mode == LeasePolicy::Mode::kAdaptive) {
        if (lo > hi) hi = lo;
        bytes = std::clamp(bytes, lo, hi);
    }

    if (committed + bytes > grantable) {
        spdlog::error("[governor] {} lease REFUSED: requested {} MiB, "
                      "committed {} MiB, grantable {} MiB",
                      vram ? "VRAM" : "RAM", bytes >> 20, committed >> 20,
                      grantable >> 20);
        if (policy.mode == LeasePolicy::Mode::kRequired) {
            if (!opts_.relax_memory_refusal) return {};
            // relaxed (tests only): grant anyway, accounting may exceed
        } else {
            const std::uint64_t avail =
                grantable > committed ? grantable - committed : 0;
            if (avail < lo) return {};  // soft refusal: caller keeps current
            bytes = avail;
        }
    }
    committed += bytes;
    return Lease(this, kind, bytes, lo, hi);
}

void ResourceGovernor::ReleaseLeaseBytes(LeaseKind kind, std::uint64_t bytes)
{
    std::lock_guard<std::mutex> lk(mu_);
    (IsVramLease(kind) ? vram_committed_ : ram_committed_) -= bytes;
}

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
