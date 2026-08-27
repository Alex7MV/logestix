#pragma once

#include <cstdint>
#include <mutex>
#include <string>

namespace lsxcommon {

// One-time snapshot of the machine the governor plans for. Filled by
// ProbeHardware() or injected by tests through GovernorOptions.hw_override.
struct HardwareProfile {
    int cpu_cores = 0;
    std::uint64_t ram_total_bytes = 0;
    std::uint64_t ram_available_bytes = 0;
    std::string gpu_name;
    int gpu_sm_count = 0;
    std::uint64_t vram_total_bytes = 0;
    std::uint64_t vram_free_bytes = 0;
};

// Injectable raw probes. Production values are wired in ProbeHardware();
// unit tests substitute fakes so the aggregation logic is GPU-free.
struct HardwareQueries {
    int (*cpu_cores)() = nullptr;
    void (*ram_bytes)(std::uint64_t *total, std::uint64_t *available) = nullptr;
    void (*gpu_info)(std::string *name, int *sm_count) = nullptr;
    void (*vram_bytes)(std::uint64_t *total, std::uint64_t *free_b) = nullptr;
};

// Pure aggregation: null query entries leave the corresponding fields at
// their defaults (0 / empty string). Identical inputs -> identical output.
HardwareProfile BuildHardwareProfile(const HardwareQueries &q);

// Real probes: OS CPU/RAM (Win32 / POSIX) + CUDA device 0 props / free VRAM.
// Never throws; failed probes leave fields zeroed.
HardwareProfile ProbeHardware();

enum class LeaseKind {
    kPermanentWeights,  // VRAM: model weights resident for Load()..Unload()
    kExpertCache,       // VRAM: routed-expert LRU device block
    kInferArena,        // VRAM: transient per-Infer workspaces
    kHostPinned,        // host: page-locked staging
    kHostRam            // host: regular RAM
};

bool IsVramLease(LeaseKind kind);

struct LeasePolicy {
    enum class Mode {
        kRequired,  // shortage fails the acquisition (deterministic refusal)
        kAdaptive   // request is clamped into [min_bytes, max_bytes]; soft
    };
    Mode mode = Mode::kRequired;
    std::uint64_t min_bytes = 0;
    std::uint64_t max_bytes = ~std::uint64_t{0};
};

class ResourceGovernor;

// RAII handle returned by Acquire(). Empty by default; movable, not copyable.
// Destruction returns the bytes to the governor's budget.
class Lease {
public:
    Lease() = default;
    ~Lease();
    Lease(Lease &&other) noexcept;
    Lease &operator=(Lease &&other) noexcept;
    Lease(const Lease &) = delete;
    Lease &operator=(const Lease &) = delete;

    bool Valid() const noexcept { return gov_ != nullptr; }
    std::uint64_t Bytes() const noexcept { return bytes_; }
    LeaseKind Kind() const noexcept { return kind_; }

    // Soft resize (meaningful for kAdaptive leases). Returns false when the
    // governor refuses (budget occupied); the lease keeps its current size.
    bool Resize(std::uint64_t new_bytes);

private:
    friend class ResourceGovernor;
    Lease(ResourceGovernor *gov, LeaseKind kind, std::uint64_t bytes,
          std::uint64_t min_bytes, std::uint64_t max_bytes)
        : gov_(gov), kind_(kind), bytes_(bytes),
          min_bytes_(min_bytes), max_bytes_(max_bytes) {}
    void Release();

    ResourceGovernor *gov_ = nullptr;
    LeaseKind kind_ = LeaseKind::kPermanentWeights;
    std::uint64_t bytes_ = 0;
    std::uint64_t min_bytes_ = 0;
    std::uint64_t max_bytes_ = 0;
};

// Default VRAM margin never granted to any lease (device reserve for the
// desktop / driver). Shared by the governor grants and the strategy plan math.
inline constexpr std::uint64_t kDefaultVramKeepOut = 512ull << 20;

struct GovernorOptions {
    // TEST-ONLY: when true, kRequired shortages log an error but do NOT fail
    // the acquisition (spec "Testing": strict refusal is enabled only in the
    // dedicated refusal tests). Production must leave this false.
    bool relax_memory_refusal = false;
    std::uint64_t vram_keep_out_bytes = kDefaultVramKeepOut;
    // TEST-ONLY: replace the real hardware probe with a synthetic profile.
    bool override_hw = false;
    HardwareProfile hw_override{};
};

// Engine-level singleton owner of all VRAM/RAM budgets (spec approach B).
// Initialized by InferenceEngine::InitializeEngine(); models never query
// hardware directly - they ask the governor for plans and leases.
class ResourceGovernor {
public:
    static ResourceGovernor &Instance();

    ResourceGovernor(const ResourceGovernor &) = delete;
    ResourceGovernor &operator=(const ResourceGovernor &) = delete;
    ResourceGovernor(ResourceGovernor &&) = delete;
    ResourceGovernor &operator=(ResourceGovernor &&) = delete;

    // Idempotent: re-Init on an initialized governor is a no-op that returns
    // true (options of the first call win).
    bool Initialize(const GovernorOptions &opts);
    void Shutdown();
    bool IsInitialized() const noexcept;

    const HardwareProfile &hardware() const;
    const GovernorOptions &options() const;

    Lease Acquire(LeaseKind kind, std::uint64_t bytes, LeasePolicy policy);

    std::uint64_t CommittedVramBytes() const;
    std::uint64_t CommittedRamBytes() const;

    // Bookkeeping sanity (spec: drift reconciliation at Infer boundaries):
    // queried_free + our committed VRAM must not exceed total + slack.
    // Returns false (and logs a warning) on drift. Never throws.
    bool ReconcileVram(std::uint64_t queried_free_vram_bytes,
                       std::uint64_t slack_bytes = 256ull << 20);
    // Convenience wrapper that queries cudaMemGetInfo itself.
    bool ReconcileVram();

private:
    friend class Lease;
    ResourceGovernor() = default;
    ~ResourceGovernor() = default;

    std::uint64_t GrantableVramLocked() const;
    void ReleaseLeaseBytes(LeaseKind kind, std::uint64_t bytes);
    bool ResizeLease(Lease &lease, std::uint64_t new_bytes);

    mutable std::mutex mu_;
    bool initialized_ = false;
    GovernorOptions opts_{};
    HardwareProfile hw_{};
    std::uint64_t vram_committed_ = 0;
    std::uint64_t ram_committed_ = 0;
};

// ── Expert-cache VRAM residency policy (launch strategy, all models) ──
//
// Single source of truth for how aggressively a model's resident-expert
// device cache may claim VRAM at load time. Every model with an expert
// cache (qwen-moe, gemma, deepseek) derives its slot count from here so
// free VRAM is maximally utilized instead of relying on per-model
// conservative guesses. Explicit per-model env overrides take precedence
// at the call sites and bypass this estimate.
namespace lifecycle {

// Headroom kept OUT of the expert cache for KV cache growth, decode
// activations, tensor-core workspaces and desktop/DWM pressure. On WDDM
// (Windows) an overcommitted device silently spills subsequent allocations
// into shared system memory, which throttles the whole decode loop — so
// the reserve must stay comfortably above the steady-state runtime usage.
// The load-time allocators degrade by halves when an allocation fails, so
// an optimistic estimate costs a retry, not a crash.
inline constexpr std::size_t kExpertCacheVramReserveBytes = 4ull << 30;

// Pure, host-only slot estimator: usable = free - reserve, clamped to the
// model's total expert count. per_expert_bytes == 0 (unknown geometry)
// degenerates to "everything resident"; free below reserve yields 0 and
// the caller applies its graceful LRU floor.
inline int ExpertCacheSlots(std::size_t free_bytes,
                            std::size_t per_expert_bytes,
                            int total_experts)
{
    if (total_experts <= 0) return 0;
    if (per_expert_bytes == 0) return total_experts;
    const std::size_t usable =
        free_bytes > kExpertCacheVramReserveBytes
            ? free_bytes - kExpertCacheVramReserveBytes
            : 0;
    const std::size_t affordable = usable / per_expert_bytes;
    return static_cast<int>(affordable < static_cast<std::size_t>(total_experts)
                                ? affordable
                                : static_cast<std::size_t>(total_experts));
}

} // namespace lifecycle

} // namespace lsxcommon
