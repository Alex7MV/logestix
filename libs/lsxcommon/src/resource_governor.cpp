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
    // The memory model decides the budget and the reserve rule, so it is stated
    // rather than inferred by whoever reads the log next.
    if (hw_.memory_model == MemoryModel::kUnified)
        spdlog::info("[governor] unified memory: shared pool={} MiB, "
                     "expert-cache reserve={} MiB (min(8 GiB, 25%))",
                     hw_.unified_shared_bytes >> 20,
                     lifecycle::ExpertCacheReserveBytes(
                         hw_.unified_shared_bytes) >> 20);
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

namespace {

// Which budget a lease of `kind` draws on, given the memory model.
//
// Discrete: two independent pools — VRAM leases against (total - keep-out), host
// leases against RAM. Unified: ONE physical pool, so a host lease is charged
// against the same committed counter the device uses; otherwise a 16 GiB Orin
// happily accepts a 15 GiB model and then OOMs the OS, which is the failure
// design §2.8 exists to prevent.
struct BudgetView {
    bool from_vram_counter = true;   // which `committed` field to charge
    std::uint64_t grantable = 0;
};

BudgetView BudgetFor(const HardwareProfile &hw, const GovernorOptions &opts,
                     LeaseKind kind)
{
    BudgetView v;
    const bool vram = IsVramLease(kind);
    const bool unified = hw.memory_model == MemoryModel::kUnified &&
                         hw.unified_shared_bytes > 0;
    if (!unified) {
        v.from_vram_counter = vram;
        v.grantable = vram ? (hw.vram_total_bytes > opts.vram_keep_out_bytes
                                  ? hw.vram_total_bytes - opts.vram_keep_out_bytes
                                  : 0)
                           : hw.ram_total_bytes;
        return v;
    }
    // One pool: every kind is charged against the same counter, and the budget
    // is the shared system pool.
    v.from_vram_counter = true;
    v.grantable = hw.unified_shared_bytes;
    return v;
}

} // namespace

std::uint64_t ResourceGovernor::GrantableVramLocked() const
{
    // Unified memory: the device draws on system RAM, and `vram_total_bytes`
    // (cudaMemGetInfo) is only the boot carve-out. Budgeting against the
    // carve-out refuses a model that would fit — a 10 GiB model on a 16 GiB Orin
    // against an 8 GiB carve-out — so the shared pool is the budget.
    if (hw_.memory_model == MemoryModel::kUnified &&
        hw_.unified_shared_bytes > 0)
        return hw_.unified_shared_bytes;
    return hw_.vram_total_bytes > opts_.vram_keep_out_bytes
               ? hw_.vram_total_bytes - opts_.vram_keep_out_bytes
               : 0;
}

std::uint64_t ResourceGovernor::GrantableVramBytes() const
{
    std::lock_guard<std::mutex> lk(mu_);
    return GrantableVramLocked();
}

// ── Adaptive resize + drift reconciliation ──────────────────────────────────

bool ResourceGovernor::ResizeLease(Lease &lease, std::uint64_t new_bytes)
{
    std::lock_guard<std::mutex> lk(mu_);
    if (!initialized_ || lease.gov_ != this) return false;
    if (lease.bytes_ == new_bytes) return true;

    new_bytes = std::clamp(new_bytes, lease.min_bytes_, lease.max_bytes_);
    if (new_bytes == lease.bytes_) return true;

    const BudgetView budget = BudgetFor(hw_, opts_, lease.kind_);
    std::uint64_t &committed =
        budget.from_vram_counter ? vram_committed_ : ram_committed_;
    const std::uint64_t grantable = budget.grantable;

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
    const BudgetView budget = BudgetFor(hw_, opts_, kind);
    std::uint64_t &committed =
        budget.from_vram_counter ? vram_committed_ : ram_committed_;
    const std::uint64_t grantable = budget.grantable;

    std::uint64_t lo = policy.min_bytes, hi = policy.max_bytes;
    if (policy.mode == LeasePolicy::Mode::kAdaptive) {
        if (lo > hi) hi = lo;
        bytes = std::clamp(bytes, lo, hi);
    }

    if (committed + bytes > grantable) {
        spdlog::error("[governor] {} lease REFUSED: requested {} MiB, "
                      "committed {} MiB, grantable {} MiB",
                      budget.from_vram_counter ? "VRAM" : "RAM", bytes >> 20,
                      committed >> 20, grantable >> 20);
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
    // The same view Acquire used: on unified memory the host leases were charged
    // against the device counter, so returning them to the host counter would
    // leak the budget and eventually refuse every acquisition.
    const BudgetView budget = BudgetFor(hw_, opts_, kind);
    std::uint64_t &committed =
        budget.from_vram_counter ? vram_committed_ : ram_committed_;
    // A lease may outlive Shutdown() (which resets the counters) or a relaxed
    // over-commit; releasing more than is booked must clamp to zero instead of
    // wrapping the unsigned counter to ~2^64.
    if (bytes > committed) {
        // Double-release detections are worth surfacing, but a caller stuck in
        // an error loop must not flood the log; warn once, debug afterwards.
        static bool warned = false;
        if (!warned) {
            warned = true;
            spdlog::warn("[governor] release of {} MiB exceeds booked {} MiB; "
                         "clamping to 0 (further occurrences at debug level)",
                         bytes >> 20, committed >> 20);
        } else {
            spdlog::debug("[governor] release of {} MiB exceeds booked {} MiB",
                          bytes >> 20, committed >> 20);
        }
        committed = 0;
        return;
    }
    committed -= bytes;
}

// ── Hardware probing ────────────────────────────────────────────────────────

HardwareProfile BuildHardwareProfile(const HardwareQueries &q)
{
    HardwareProfile hw;
    if (q.cpu_cores) hw.cpu_cores = q.cpu_cores();
    if (q.ram_bytes) q.ram_bytes(&hw.ram_total_bytes, &hw.ram_available_bytes);
    if (q.gpu_info) q.gpu_info(&hw.gpu_name, &hw.gpu_sm_count);
    if (q.vram_bytes) q.vram_bytes(&hw.vram_total_bytes, &hw.vram_free_bytes);
    if (q.unified_shared) {
        bool integrated = false;
        std::uint64_t shared = 0;
        q.unified_shared(&integrated, &shared);
        if (integrated) {
            hw.memory_model = MemoryModel::kUnified;
            // A probe that reports integrated without saying how much is
            // incoherent; the host's total RAM is then the only sane answer, and
            // a lease can at least not exceed it.
            hw.unified_shared_bytes = shared > 0 ? shared : hw.ram_total_bytes;
        }
    }
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

// Unified memory, from the DEVICE: `prop.integrated` is what the driver reports
// for a GPU that shares system RAM (Jetson Orin/Nano). The name is not consulted
// — "Orin" in a marketing string is not a memory model (design §2.8).
//
// The size is the host's physical RAM rather than the carve-out: on unified
// memory the carve-out is fixed at boot and the remainder is still available to
// the GPU through the same physical pages.
static void QueryUnifiedShared(bool *integrated, std::uint64_t *shared_bytes)
{
    *integrated = false;
    *shared_bytes = 0;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) return;
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return;
    if (!p.integrated) return;
    *integrated = true;
    std::uint64_t ram_total = 0, ram_avail = 0;
#if defined(_WIN32)
    QueryRamWin(&ram_total, &ram_avail);
#else
    QueryRamPosix(&ram_total, &ram_avail);
#endif
    *shared_bytes = ram_total;
}

HardwareProfile ProbeHardware()
{
#if defined(_WIN32)
    HardwareQueries q{QueryCoresWin, QueryRamWin, QueryGpuInfo, QueryVram,
                      QueryUnifiedShared};
#else
    HardwareQueries q{QueryCoresPosix, QueryRamPosix, QueryGpuInfo, QueryVram,
                      QueryUnifiedShared};
#endif
    return BuildHardwareProfile(q);
}

} // namespace lsxcommon
