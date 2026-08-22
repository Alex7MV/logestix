# Resource Governor — Model Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Central ResourceGovernor owning all VRAM/RAM/CPU budgets: layered lifecycle strategies (general → dense/MoE → per-model), load-once GLM-5.2 weights, runtime-tuned expert cache, concurrency groundwork.

**Architecture:** Engine-level singleton `ResourceGovernor` (Meyers, mirrors `InferenceEngine`) probes hardware once, caches per-model `LifecyclePlan`s built by `ILifecycleStrategy` implementations, and accounts RAII `Lease`s. `GlmV5Model` loads pinned weights once in `Load()`, reuses them in every `Infer()`/`Benchmark()`, and reports expert-cache telemetry to a `RuntimeTuner` between requests.

**Tech Stack:** C++20, CUDA 13.3 (cudart runtime API only in this feature), Catch2 v3, spdlog, CMake presets (Windows VS2026 generator; build dir `build\` already configured).

**Spec:** `docs/superpowers/specs/2026-08-22-resource-governor-lifecycle-design.md`

**Build/test commands (this machine, PowerShell):**

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cmake --build build --config Debug --target <target> 2>&1 | Select-Object -Last 25
& "build\bin\Debug\<target>.exe"
```

Adding new sources to `libs/lsxcommon/CMakeLists.txt` or `tests/CMakeLists.txt` triggers automatic re-configure on the next `--build`. Run tests by executing the exe directly (Catch2, no args needed) — output ends with `All tests passed` (or `... assertions` summary). GLM real-model tests SKIP without weights; that is expected here.

**Deviations from spec (documented during planning):**
- `kInferArena` leases exist in the contract and are unit-tested, but `GlmV5Model::Infer` does not acquire one yet: the arena is allocated inside `RunGlmRealGeneration` and its size is unknowable before the run; with single-consumer dispatch the reservation is unobservable. It becomes meaningful in the concurrency iteration.
- MTP epilogue stays permanently resident (spec amendment `75bb2c8`).

---

### Task 1: ResourceGovernor skeleton (singleton, options, init/shutdown)

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/resource_governor.h`
- Create: `libs/lsxcommon/src/resource_governor.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt:16-48` (add one source line)
- Modify: `tests/CMakeLists.txt` (new test target)
- Test: `tests/test_resource_governor.cpp`

- [ ] **Step 1: Create the header (complete — later tasks only add `.cpp` implementations)**

```cpp
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
// hardware directly — they ask the governor for plans and leases.
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

} // namespace lsxcommon
```

- [ ] **Step 2: Create `tests/test_resource_governor.cpp` with the init/shutdown test**

```cpp
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/resource_governor.h"

namespace {

lsxcommon::HardwareProfile FakeHw()
{
    lsxcommon::HardwareProfile hw;
    hw.cpu_cores = 8;
    hw.ram_total_bytes = hw.ram_available_bytes = 16ull << 30;
    hw.gpu_name = "FakeGPU";
    hw.gpu_sm_count = 8;
    hw.vram_total_bytes = hw.vram_free_bytes = 8ull << 30;
    return hw;
}

} // namespace

TEST_CASE("governor init/shutdown lifecycle", "[governor]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();  // deterministic start regardless of other tests

    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;  // tests grant the full budget
    REQUIRE(gov.Initialize(opts));
    REQUIRE(gov.IsInitialized());
    REQUIRE(gov.hardware().gpu_name == "FakeGPU");
    REQUIRE(gov.hardware().vram_total_bytes == (8ull << 30));
    REQUIRE(gov.CommittedVramBytes() == 0);
    REQUIRE(gov.CommittedRamBytes() == 0);

    SECTION("re-init is idempotent: first options win")
    {
        lsxcommon::GovernorOptions same;
        same.override_hw = true;
        same.hw_override = FakeHw();
        same.vram_keep_out_bytes = 12345;
        REQUIRE(gov.Initialize(same));
        REQUIRE(gov.options().vram_keep_out_bytes == 0);
    }

    gov.Shutdown();
    REQUIRE_FALSE(gov.IsInitialized());
}
```

- [ ] **Step 3: Register the test target in `tests/CMakeLists.txt`**

Append (mirroring the plain-test pattern of `test_vram_estimate`, line 1184):

```cmake
add_executable(test_resource_governor test_resource_governor.cpp)
target_link_libraries(test_resource_governor PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_resource_governor POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_resource_governor>")
endif()
add_test(NAME test_resource_governor COMMAND test_resource_governor)
```

- [ ] **Step 4: Build — expect link failure (missing definitions)**

```powershell
& $cmake --build build --config Debug --target test_resource_governor 2>&1 | Select-Object -Last 25
```
Expected: `unresolved external symbol ... Instance/Initialize/Shutdown/...` (header declares, .cpp missing).

- [ ] **Step 5: Implement `src/resource_governor.cpp` (Task-1 subset only)**

```cpp
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
```

Add the source to `libs/lsxcommon/CMakeLists.txt` (alphabetical, after `src/qwen_base_model.cpp`):

```cmake
    src/resource_governor.cpp
```

- [ ] **Step 6: Build + run — expect PASS**

```powershell
& $cmake --build build --config Debug --target test_resource_governor 2>&1 | Select-Object -Last 25
& "build\bin\Debug\test_resource_governor.exe"
```
Expected: `All tests passed` (1 test case, ~8 assertions).

- [ ] **Step 7: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/resource_governor.h libs/lsxcommon/src/resource_governor.cpp libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt tests/test_resource_governor.cpp
git commit -m "feat: ResourceGovernor singleton skeleton with init/shutdown lifecycle"
```

---

### Task 2: HardwareProfile — BuildHardwareProfile + ProbeHardware

**Files:**
- Modify: `libs/lsxcommon/src/resource_governor.cpp` (replace the Task-1 stubs of `BuildHardwareProfile`/`ProbeHardware`)
- Test: `tests/test_hardware_profile.cpp` (new) + CMake target

- [ ] **Step 1: Write the failing test**

`tests/test_hardware_profile.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <string>

#include "lsxcommon/resource_governor.h"

namespace {

int FakeCores() { return 16; }
void FakeRam(std::uint64_t *total, std::uint64_t *avail)
{
    *total = 32ull << 30;
    *avail = 12ull << 30;
}
void FakeGpu(std::string *name, int *sm)
{
    *name = "TestGPU";
    *sm = 148;
}
void FakeVram(std::uint64_t *total, std::uint64_t *free_b)
{
    *total = 32ull << 30;
    *free_b = 31ull << 30;
}

} // namespace

TEST_CASE("hardware profile aggregates raw queries", "[governor][hw]")
{
    lsxcommon::HardwareQueries q{FakeCores, FakeRam, FakeGpu, FakeVram};
    const auto hw = lsxcommon::BuildHardwareProfile(q);
    REQUIRE(hw.cpu_cores == 16);
    REQUIRE(hw.ram_total_bytes == (32ull << 30));
    REQUIRE(hw.ram_available_bytes == (12ull << 30));
    REQUIRE(hw.gpu_name == "TestGPU");
    REQUIRE(hw.gpu_sm_count == 148);
    REQUIRE(hw.vram_total_bytes == (32ull << 30));
    REQUIRE(hw.vram_free_bytes == (31ull << 30));
}

TEST_CASE("null query entries leave fields defaulted", "[governor][hw]")
{
    const auto hw = lsxcommon::BuildHardwareProfile(lsxcommon::HardwareQueries{});
    REQUIRE(hw.cpu_cores == 0);
    REQUIRE(hw.ram_total_bytes == 0);
    REQUIRE(hw.gpu_name.empty());
    REQUIRE(hw.gpu_sm_count == 0);
    REQUIRE(hw.vram_total_bytes == 0);
}

TEST_CASE("real probe yields sane values", "[governor][hw][smoke]")
{
    const auto hw = lsxcommon::ProbeHardware();
    REQUIRE(hw.cpu_cores > 0);
    REQUIRE(hw.ram_total_bytes > 0);
    REQUIRE(hw.ram_available_bytes <= hw.ram_total_bytes);
    // GPU fields are machine-dependent (CI may lack a device): only check
    // consistency when a device was found.
    if (!hw.gpu_name.empty()) {
        REQUIRE(hw.gpu_sm_count > 0);
        REQUIRE(hw.vram_total_bytes > 0);
    }
}
```

Register in `tests/CMakeLists.txt` (same block shape as Task 1 Step 3, target
`test_hardware_profile`, file `test_hardware_profile.cpp`).

- [ ] **Step 2: Build — expect FAIL** (`BuildHardwareProfile` stub returns zeros)

- [ ] **Step 3: Implement in `src/resource_governor.cpp`** (replace the two stubs)

```cpp
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
#include <windows.h>
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
#include <sys/sysinfo.h>
#include <unistd.h>
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

#include <cuda_runtime.h>
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
```

Note: `WIN32_LEAN_AND_MEAN`/`NOMINMAX` are already defined project-wide
(`CMAKE_CXX_FLAGS`), so `<windows.h>` is safe here.

- [ ] **Step 4: Build + run both governor tests — expect PASS**

```powershell
& $cmake --build build --config Debug --target test_hardware_profile test_resource_governor 2>&1 | Select-Object -Last 25
& "build\bin\Debug\test_hardware_profile.exe"
& "build\bin\Debug\test_resource_governor.exe"
```

- [ ] **Step 5: Commit**

```powershell
git add libs/lsxcommon/src/resource_governor.cpp tests/CMakeLists.txt tests/test_hardware_profile.cpp
git commit -m "feat: hardware probing (CPU/RAM/GPU/VRAM) for ResourceGovernor"
```

---

### Task 3: Leases — Acquire/Release, budgets, strict refusal

**Files:**
- Modify: `libs/lsxcommon/src/resource_governor.cpp` (replace `IsVramLease`, `Lease` stubs, `Acquire` stub)
- Test: `tests/test_resource_governor.cpp` (append)

- [ ] **Step 1: Append the failing tests**

```cpp
TEST_CASE("required lease granted within budget", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();  // 8 GiB VRAM
    opts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(opts));

    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             4ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(lease.Valid());
    REQUIRE(lease.Bytes() == (4ull << 30));
    REQUIRE(gov.CommittedVramBytes() == (4ull << 30));
    REQUIRE(gov.CommittedRamBytes() == 0);  // VRAM lease did not touch RAM

    {   // RAM lease lands in the other domain
        auto ram = gov.Acquire(lsxcommon::LeaseKind::kHostPinned,
                               1ull << 30, lsxcommon::LeasePolicy{});
        REQUIRE(ram.Valid());
        REQUIRE(gov.CommittedRamBytes() == (1ull << 30));
    }
    REQUIRE(gov.CommittedRamBytes() == 0);  // RAII release
    gov.Shutdown();
}

TEST_CASE("required lease refused on shortage (strict)", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(opts));

    auto first = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             6ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(first.Valid());
    auto second = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                              4ull << 30, lsxcommon::LeasePolicy{});  // 10 > 8
    REQUIRE_FALSE(second.Valid());
    REQUIRE(gov.CommittedVramBytes() == (6ull << 30));  // unchanged
    gov.Shutdown();
}

TEST_CASE("keep-out margin shrinks the grantable budget", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 2ull << 30;  // grantable = 6 GiB
    REQUIRE(gov.Initialize(opts));
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             6ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(lease.Valid());
    auto over = gov.Acquire(lsxcommon::LeaseKind::kExpertCache,
                            1ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE_FALSE(over.Valid());
    gov.Shutdown();
}

TEST_CASE("relaxed mode grants required leases despite shortage", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;
    opts.relax_memory_refusal = true;  // TEST-ONLY semantics
    REQUIRE(gov.Initialize(opts));
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             10ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(lease.Valid());  // granted anyway, accounting may exceed budget
    gov.Shutdown();
}

TEST_CASE("acquire before initialize fails", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             1ull << 20, lsxcommon::LeasePolicy{});
    REQUIRE_FALSE(lease.Valid());
}
```

- [ ] **Step 2: Build — expect FAIL** (stubs return empty leases / `IsVramLease` always true)

- [ ] **Step 3: Implement** (replace `IsVramLease`, `Lease` methods, `Acquire`)

```cpp
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
```

- [ ] **Step 4: Build + run — expect PASS**

- [ ] **Step 5: Commit**

```powershell
git add libs/lsxcommon/src/resource_governor.cpp tests/test_resource_governor.cpp
git commit -m "feat: RAII leases with budget accounting and strict memory refusal"
```

---

### Task 4: Adaptive Resize + VRAM drift reconciliation

**Files:**
- Modify: `libs/lsxcommon/src/resource_governor.cpp` (replace `ResizeLease`/`ReconcileVram` stubs)
- Test: `tests/test_resource_governor.cpp` (append)

- [ ] **Step 1: Append the failing tests**

```cpp
TEST_CASE("adaptive lease clamps into policy bounds", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(opts));

    lsxcommon::LeasePolicy adaptive{
        lsxcommon::LeasePolicy::Mode::kAdaptive,
        /*min*/ 2ull << 20, /*max*/ 8ull << 20};
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kExpertCache,
                             100ull << 30, adaptive);  // clamped to max
    REQUIRE(lease.Valid());
    REQUIRE(lease.Bytes() == (8ull << 20));
    gov.Shutdown();
}

TEST_CASE("resize grows within budget and refuses beyond it", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();
    opts.vram_keep_out_bytes = 0;
    REQUIRE(gov.Initialize(opts));

    lsxcommon::LeasePolicy adaptive{
        lsxcommon::LeasePolicy::Mode::kAdaptive,
        /*min*/ 1ull << 20, /*max*/ 8ull << 30};
    auto blocker = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                               6ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(blocker.Valid());
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kExpertCache,
                             1ull << 30, adaptive);
    REQUIRE(lease.Valid());

    // Growth that fits: committed 6 GiB + 1 GiB -> +1.5 GiB <= 8 GiB.
    REQUIRE(lease.Resize(2ull << 30));
    REQUIRE(lease.Bytes() == (2ull << 30));
    REQUIRE(gov.CommittedVramBytes() == (8ull << 30));

    // Growth beyond budget: refused, size kept (soft semantics).
    REQUIRE_FALSE(lease.Resize(4ull << 30));
    REQUIRE(lease.Bytes() == (2ull << 30));
    REQUIRE(gov.CommittedVramBytes() == (8ull << 30));

    // Shrink always allowed: returns budget.
    REQUIRE(lease.Resize(1ull << 30));
    REQUIRE(gov.CommittedVramBytes() == (7ull << 30));

    // Resize below the adaptive floor is clamped up to min_bytes.
    REQUIRE(lease.Resize(0));
    REQUIRE(lease.Bytes() == (1ull << 20));

    gov.Shutdown();
}

TEST_CASE("resize on invalid lease fails", "[governor][lease]")
{
    lsxcommon::Lease empty;
    REQUIRE_FALSE(empty.Resize(1ull << 20));
}

TEST_CASE("vram reconciliation detects over-accounting drift", "[governor][lease]")
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    gov.Shutdown();
    lsxcommon::GovernorOptions opts;
    opts.override_hw = true;
    opts.hw_override = FakeHw();  // total 8 GiB
    opts.vram_keep_out_bytes = 0;
    opts.relax_memory_refusal = true;
    REQUIRE(gov.Initialize(opts));
    auto lease = gov.Acquire(lsxcommon::LeaseKind::kPermanentWeights,
                             6ull << 30, lsxcommon::LeasePolicy{});
    REQUIRE(lease.Valid());

    // Consistent world: 2 GiB free + 6 GiB committed == 8 GiB total.
    REQUIRE(gov.ReconcileVram(2ull << 30));
    // Drift: books claim more than the device can have outstanding.
    REQUIRE_FALSE(gov.ReconcileVram(3ull << 30));  // 3+6 > 8+slack(256MiB)
    gov.Shutdown();
}
```

- [ ] **Step 2: Build — expect FAIL** (`ResizeLease`/`ReconcileVram` stubs)

- [ ] **Step 3: Implement**

```cpp
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
```

(`cudaMemGetInfo` is available: Task 2 already includes `<cuda_runtime.h>`.)

- [ ] **Step 4: Build + run — expect PASS**

- [ ] **Step 5: Commit**

```powershell
git add libs/lsxcommon/src/resource_governor.cpp tests/test_resource_governor.cpp
git commit -m "feat: soft adaptive lease resize and VRAM drift reconciliation"
```

---

### Task 5: LifecyclePlan + GeneralStrategy

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/lifecycle_strategy.h`
- Create: `libs/lsxcommon/src/lifecycle_strategy.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (one source line)
- Test: `tests/test_lifecycle_plan.cpp` (new) + CMake target

- [ ] **Step 1: Create the header**

```cpp
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "lsxcommon/resource_governor.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace lifecycle {

// Model execution archetype, resolved from the manifest topology
// (non-empty moe_layers[] -> MoE, else dense).
enum class Archetype { General, Dense, MoE };

// Runtime-tunable parameter ranges (spec: the tuner may only move inside
// these bounds; the plan fixes the start value).
struct TuningBounds {
    std::size_t expert_cache_slots_min = 4;
    std::size_t expert_cache_slots_max = 64;
};

// Deterministic residency plan for one model on one machine (spec 3).
struct LifecyclePlan {
    // Permanent VRAM residency
    bool pin_globals_in_vram = true;
    bool pin_dense_layers = true;
    std::uint64_t permanent_weight_bytes = 0;

    // Expert subsystem (MoE)
    std::size_t expert_cache_slots = 0;  // start value within tuner_bounds
    std::uint64_t expert_cache_min_bytes = 0;
    std::uint64_t expert_cache_max_bytes = 0;
    std::size_t expert_prefetch_batch = 1;

    // Compute pipeline
    int copy_streams = 2;
    std::size_t staging_ring_bytes = 8ull << 20;
    std::size_t host_upload_threads = 1;

    TuningBounds tuner_bounds{};
};

// Strategy inputs: everything except the hardware snapshot itself.
struct LifecycleInput {
    VramProfile vram;
    Archetype archetype = Archetype::General;
    std::string model_type;
    std::uint64_t expert_mean_bytes = 0;  // mean routed-expert payload bytes
};

class ILifecycleStrategy {
public:
    virtual ~ILifecycleStrategy() = default;
    // Deterministic: identical inputs -> identical plan. Never allocates GPU
    // memory and never throws.
    virtual LifecyclePlan BuildPlan(const LifecycleInput &in,
                                    const HardwareProfile &hw,
                                    std::uint64_t vram_keep_out_bytes) const = 0;
};

class GeneralStrategy : public ILifecycleStrategy {
public:
    LifecyclePlan BuildPlan(const LifecycleInput &in, const HardwareProfile &hw,
                            std::uint64_t vram_keep_out_bytes) const override;
};

// Extends the general plan with the expert-cache sizing + prefetch policy.
class MoEStrategy final : public GeneralStrategy {
public:
    LifecyclePlan BuildPlan(const LifecycleInput &in, const HardwareProfile &hw,
                            std::uint64_t vram_keep_out_bytes) const override;
};

// Per-model anchor for GLM-5.2 (registered in ResolveStrategy). Inherits the
// MoE plan; the MTP epilogue stays permanently resident (spec amendment).
class Glm52Strategy final : public MoEStrategy {};

// Registry resolution: per-model override ("glm-5.2") -> archetype -> General.
const ILifecycleStrategy &ResolveStrategy(Archetype archetype,
                                          const std::string &model_type);

// Builds (once) and returns the plan for a model type. The plan is cached per
// model_type: repeated calls return the same object.
const LifecyclePlan &GetOrBuildPlan(const LifecycleInput &in);

void ResetPlanCache();  // tests only

} // namespace lifecycle
} // namespace lsxcommon
```

- [ ] **Step 2: Create `tests/test_lifecycle_plan.cpp` (General part) + CMake target**

```cpp
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "lsxcommon/lifecycle_strategy.h"

using namespace lsxcommon;
using namespace lsxcommon::lifecycle;

namespace {

HardwareProfile FakeHw(std::uint64_t vram_gib)
{
    HardwareProfile hw;
    hw.cpu_cores = 16;
    hw.ram_total_bytes = hw.ram_available_bytes = 64ull << 30;
    hw.gpu_name = "FakeGPU";
    hw.gpu_sm_count = 148;
    hw.vram_total_bytes = vram_gib << 30;
    hw.vram_free_bytes = hw.vram_total_bytes;
    return hw;
}

VramProfile FakeVram(std::uint64_t min_gib, std::uint64_t optimal_gib,
                     std::uint64_t max_gib)
{
    VramProfile v;
    v.min_bytes = min_gib << 30;
    v.optimal_bytes = optimal_gib << 30;
    v.max_bytes = max_gib << 30;
    v.context_tokens = 4096;
    return v;
}

LifecycleInput MakeInput(VramProfile v, Archetype arch,
                         std::uint64_t mean_expert = 0)
{
    LifecycleInput in;
    in.vram = v;
    in.archetype = arch;
    in.model_type = "test-model";
    in.expert_mean_bytes = mean_expert;
    return in;
}

constexpr std::uint64_t kKeepOut = 0;  // tests grant the full budget

} // namespace

TEST_CASE("general plan caps residency to grantable vram", "[lifecycle]")
{
    GeneralStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 6), Archetype::Dense);
    const auto p = s.BuildPlan(in, FakeHw(10), kKeepOut);
    // permanent = min(optimal 4 GiB, total 10 GiB) = 4 GiB
    REQUIRE(p.permanent_weight_bytes == (4ull << 30));
    // remaining 6 GiB - arena_reserve (max-optimal = 2 GiB) = 4 GiB cache,
    // but General archetype carries no experts:
    REQUIRE(p.expert_cache_max_bytes == 0);
    REQUIRE(p.expert_cache_slots == 0);
    REQUIRE(p.pin_globals_in_vram);
    REQUIRE(p.pin_dense_layers);
    // 16 cores / 8 -> 2 upload threads
    REQUIRE(p.host_upload_threads == 2);
}

TEST_CASE("general plan shrinks when vram is smaller than optimal", "[lifecycle]")
{
    GeneralStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 5), Archetype::Dense);
    const auto p = s.BuildPlan(in, FakeHw(2), kKeepOut);
    REQUIRE(p.permanent_weight_bytes == (2ull << 30));  // capped at grantable
    REQUIRE(p.expert_cache_max_bytes == 0);
}

TEST_CASE("general plan handles keep-out margin", "[lifecycle]")
{
    GeneralStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 4), Archetype::Dense);
    const auto p = s.BuildPlan(in, FakeHw(10), 3ull << 30);  // grantable 7 GiB
    REQUIRE(p.permanent_weight_bytes == (4ull << 30));
}
```

CMake block (same shape as Task 1 Step 3; target `test_lifecycle_plan`,
file `test_lifecycle_plan.cpp`).

- [ ] **Step 3: Build — expect FAIL** (undefined `GeneralStrategy::BuildPlan`)

- [ ] **Step 4: Implement `src/lifecycle_strategy.cpp` (General part)**

```cpp
#include "lsxcommon/lifecycle_strategy.h"

#include <algorithm>

#include <absl/container/flat_hash_map.h>
#include <spdlog/spdlog.h>

namespace lsxcommon {
namespace lifecycle {

namespace {
std::mutex g_cache_mu;
absl::flat_hash_map<std::string, LifecyclePlan> g_plan_cache;
} // namespace

LifecyclePlan GeneralStrategy::BuildPlan(const LifecycleInput &in,
                                         const HardwareProfile &hw,
                                         std::uint64_t vram_keep_out_bytes) const
{
    LifecyclePlan p;
    const std::uint64_t grantable =
        hw.vram_total_bytes > vram_keep_out_bytes
            ? hw.vram_total_bytes - vram_keep_out_bytes
            : 0;
    // Permanent residency: the conversion-time optimal footprint, capped at
    // what the machine can grant. Everything above stays host-pinned/mmap.
    p.permanent_weight_bytes = std::min(in.vram.optimal_bytes, grantable);

    const std::uint64_t remaining =
        grantable > p.permanent_weight_bytes
            ? grantable - p.permanent_weight_bytes
            : 0;
    // Reserve headroom for the infer arenas: max - optimal is the
    // conversion-time estimate of the transient workspace.
    const std::uint64_t arena_reserve =
        in.vram.max_bytes > in.vram.optimal_bytes
            ? in.vram.max_bytes - in.vram.optimal_bytes
            : 0;
    const std::uint64_t cache_budget =
        remaining > arena_reserve ? remaining - arena_reserve : 0;
    p.expert_cache_max_bytes =
        in.archetype == Archetype::MoE ? cache_budget : 0;

    p.host_upload_threads = std::clamp(hw.cpu_cores / 8, 1, 4);
    return p;
}

const ILifecycleStrategy &ResolveStrategy(Archetype archetype,
                                          const std::string &model_type)
{
    static GeneralStrategy general;
    if (archetype == Archetype::MoE) return general;  // replaced in Task 6
    (void)model_type;
    return general;
}

const LifecyclePlan &GetOrBuildPlan(const LifecycleInput &in)
{
    const auto &gov = ResourceGovernor::Instance();
    const ILifecycleStrategy &s =
        ResolveStrategy(in.archetype, in.model_type);
    LifecyclePlan p = s.BuildPlan(in, gov.hardware(),
                                  gov.options().vram_keep_out_bytes);
    spdlog::info("[lifecycle] plan for '{}': permanent={} MiB cache slots={} "
                 "[{}..{}] streams={} staging={}",
                 in.model_type, p.permanent_weight_bytes >> 20,
                 p.expert_cache_slots, p.tuner_bounds.expert_cache_slots_min,
                 p.tuner_bounds.expert_cache_slots_max, p.copy_streams,
                 p.staging_ring_bytes >> 20);
    std::lock_guard<std::mutex> lk(g_cache_mu);
    auto [ins, inserted] = g_plan_cache.emplace(in.model_type, std::move(p));
    (void)inserted;
    return ins->second;
}

void ResetPlanCache()
{
    std::lock_guard<std::mutex> lk(g_cache_mu);
    g_plan_cache.clear();
}

} // namespace lifecycle
} // namespace lsxcommon
```

Add `src/lifecycle_strategy.cpp` to `libs/lsxcommon/CMakeLists.txt`.

- [ ] **Step 5: Build + run — expect PASS**

- [ ] **Step 6: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/lifecycle_strategy.h libs/lsxcommon/src/lifecycle_strategy.cpp libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt tests/test_lifecycle_plan.cpp
git commit -m "feat: LifecyclePlan and GeneralStrategy with deterministic plan math"
```

---

### Task 6: MoE + Glm52 strategies and plan cache

**Files:**
- Modify: `libs/lsxcommon/src/lifecycle_strategy.cpp` (implement `MoEStrategy::BuildPlan`, `ResolveStrategy`)
- Test: `tests/test_lifecycle_plan.cpp` (append)

- [ ] **Step 1: Append the failing tests**

```cpp
TEST_CASE("moe plan sizes expert cache from mean expert bytes", "[lifecycle]")
{
    MoEStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 6), Archetype::MoE,
                              /*mean_expert*/ 512ull << 20);
    const auto p = s.BuildPlan(in, FakeHw(10), kKeepOut);
    // cache_budget = 10 GiB - 4 GiB permanent - 2 GiB arena reserve = 4 GiB
    REQUIRE(p.expert_cache_max_bytes == (4ull << 30));
    // fit = 4 GiB / 512 MiB = 8 slots
    REQUIRE(p.expert_cache_slots == 8);
    REQUIRE(p.tuner_bounds.expert_cache_slots_min == 4);
    REQUIRE(p.tuner_bounds.expert_cache_slots_max == 8);
    REQUIRE(p.expert_cache_min_bytes == 4ull * (512ull << 20));
    REQUIRE(p.expert_prefetch_batch == 4);
}

TEST_CASE("moe plan clamps slots into sane bounds", "[lifecycle]")
{
    MoEStrategy s;
    SECTION("tiny experts would explode slot count -> clamped at 4096")
    {
        const auto in = MakeInput(FakeVram(1, 1, 2), Archetype::MoE,
                                  /*mean_expert*/ 1ull << 20);
        const auto p = s.BuildPlan(in, FakeHw(64), kKeepOut);
        REQUIRE(p.expert_cache_slots <= p.tuner_bounds.expert_cache_slots_max);
        REQUIRE(p.tuner_bounds.expert_cache_slots_max == 4096);
    }
    SECTION("huge experts -> floor of 4 slots")
    {
        const auto in = MakeInput(FakeVram(4, 4, 6), Archetype::MoE,
                                  /*mean_expert*/ 8ull << 30);
        const auto p = s.BuildPlan(in, FakeHw(10), kKeepOut);
        REQUIRE(p.expert_cache_slots == 4);
    }
}

TEST_CASE("glm-5.2 resolves the Glm52 strategy via the registry", "[lifecycle]")
{
    ResetPlanCache();
    LifecycleInput in;
    in.model_type = "glm-5.2";
    in.archetype = Archetype::MoE;
    in.vram = FakeVram(4, 4, 6);
    in.expert_mean_bytes = 512ull << 20;
    const auto &a = GetOrBuildPlan(in);
    const auto &b = GetOrBuildPlan(in);
    REQUIRE(&a == &b);  // cached: identical object
    REQUIRE(a.expert_prefetch_batch == 4);
    REQUIRE(a.pin_dense_layers);
    ResetPlanCache();
}

TEST_CASE("plan building is deterministic", "[lifecycle]")
{
    MoEStrategy s;
    const auto in = MakeInput(FakeVram(4, 4, 6), Archetype::MoE,
                              512ull << 20);
    const auto p1 = s.BuildPlan(in, FakeHw(10), kKeepOut);
    const auto p2 = s.BuildPlan(in, FakeHw(10), kKeepOut);
    REQUIRE(p1.permanent_weight_bytes == p2.permanent_weight_bytes);
    REQUIRE(p1.expert_cache_slots == p2.expert_cache_slots);
    REQUIRE(p1.expert_cache_max_bytes == p2.expert_cache_max_bytes);
    REQUIRE(p1.tuner_bounds.expert_cache_slots_max ==
            p2.tuner_bounds.expert_cache_slots_max);
    REQUIRE(p1.host_upload_threads == p2.host_upload_threads);
}
```

- [ ] **Step 2: Build — expect FAIL** (undefined `MoEStrategy::BuildPlan`, cache not shared)

- [ ] **Step 3: Implement**

```cpp
LifecyclePlan MoEStrategy::BuildPlan(const LifecycleInput &in,
                                     const HardwareProfile &hw,
                                     std::uint64_t vram_keep_out_bytes) const
{
    LifecyclePlan p = GeneralStrategy::BuildPlan(in, hw, vram_keep_out_bytes);
    if (p.expert_cache_max_bytes > 0 && in.expert_mean_bytes > 0) {
        const std::size_t fit = static_cast<std::size_t>(
            p.expert_cache_max_bytes / in.expert_mean_bytes);
        TuningBounds b;
        b.expert_cache_slots_min = 4;
        b.expert_cache_slots_max = std::clamp<std::size_t>(fit, 4, 4096);
        p.tuner_bounds = b;
        p.expert_cache_slots = std::clamp(fit, b.expert_cache_slots_min,
                                          b.expert_cache_slots_max);
        p.expert_cache_min_bytes =
            static_cast<std::uint64_t>(b.expert_cache_slots_min) *
            in.expert_mean_bytes;
        p.expert_prefetch_batch = 4;
    }
    return p;
}

const ILifecycleStrategy &ResolveStrategy(Archetype archetype,
                                          const std::string &model_type)
{
    static GeneralStrategy general;
    static MoEStrategy moe;
    static Glm52Strategy glm52;
    // Per-model override first (spec: layered strategies), then archetype.
    if (model_type == "glm-5.2") return glm52;  // GlmV5Model::kType
    if (archetype == Archetype::MoE) return moe;
    return general;
}
```

(Use the file-scope `g_plan_cache`/`g_cache_mu` form from Task 5 Step 4 so
`ResetPlanCache` really clears.)

- [ ] **Step 4: Build + run — expect PASS**

- [ ] **Step 5: Commit**

```powershell
git add libs/lsxcommon/src/lifecycle_strategy.cpp tests/test_lifecycle_plan.cpp
git commit -m "feat: MoE/Glm52 lifecycle strategies with cached plan registry"
```

---

### Task 7: RuntimeTuner

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/runtime_tuner.h`
- Create: `libs/lsxcommon/src/runtime_tuner.cpp`
- Modify: `libs/lsxcommon/CMakeLists.txt` (one source line)
- Test: `tests/test_runtime_tuner.cpp` (new) + CMake target

- [ ] **Step 1: Create the header**

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

#include "lsxcommon/lifecycle_strategy.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon {
namespace lifecycle {

// Sliding-window controller for the expert LRU cache size (the only tunable
// in this iteration, spec: RuntimeTuner). Reports arrive between inference
// requests — never inside the hot loop. Growth goes through the governor
// (soft-refused when the budget is occupied); shrink returns budget.
class RuntimeTuner {
public:
    struct Config {
        std::size_t window_steps = 16;      // reports per decision window
        double grow_miss_ratio = 0.10;      // sustained misses -> grow x2
        double shrink_miss_ratio = 0.02;    // sustained hits -> shrink /2
        std::size_t min_interval_steps = 32;  // hysteresis between resizes
    };

    // Takes ownership of the cache lease. slots_ starts at
    // lease.Bytes() / slot_stride_bytes clamped into bounds.
    RuntimeTuner(ResourceGovernor *gov, Lease lease,
                 std::uint64_t slot_stride_bytes, const TuningBounds &bounds,
                 const Config &cfg = {});

    // Feed one report (hits/misses deltas since the previous call).
    void Report(std::int64_t hits_delta, std::int64_t misses_delta);

    std::size_t Slots() const { return slots_; }
    int Resizes() const { return resizes_; }  // diagnostics/tests
    const Lease &CacheLease() const { return lease_; }

private:
    ResourceGovernor *gov_ = nullptr;
    Lease lease_;
    std::uint64_t stride_ = 1;
    TuningBounds bounds_{};
    Config cfg_{};
    std::size_t slots_ = 0;
    std::size_t window_pos_ = 0;
    std::size_t since_resize_ = 0;
    std::int64_t w_hits_ = 0;
    std::int64_t w_misses_ = 0;
    int resizes_ = 0;
};

} // namespace lifecycle
} // namespace lsxcommon
```

- [ ] **Step 2: Write the failing test**

`tests/test_runtime_tuner.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/runtime_tuner.h"

using namespace lsxcommon;
using namespace lsxcommon::lifecycle;

namespace {

HardwareProfile FakeHw()
{
    HardwareProfile hw;
    hw.cpu_cores = 16;
    hw.ram_total_bytes = hw.ram_available_bytes = 16ull << 30;
    hw.gpu_name = "FakeGPU";
    hw.gpu_sm_count = 8;
    hw.vram_total_bytes = hw.vram_free_bytes = 8ull << 30;
    return hw;
}

struct Harness {
    ResourceGovernor &gov = ResourceGovernor::Instance();
    RuntimeTuner::Config cfg{/*.window_steps*/ 4, /*grow*/ 0.10,
                             /*shrink*/ 0.02, /*min_interval*/ 2};
    Harness()
    {
        gov.Shutdown();
        GovernorOptions o;
        o.override_hw = true;
        o.hw_override = FakeHw();
        o.vram_keep_out_bytes = 0;
        (void)gov.Initialize(o);
    }
    ~Harness() { gov.Shutdown(); }
};

RuntimeTuner MakeTuner(std::size_t start_slots = 4)
{
    GovernorOptions o;
    o.override_hw = true;
    o.hw_override = FakeHw();
    o.vram_keep_out_bytes = 0;
    auto &gov = ResourceGovernor::Instance();
    LeasePolicy adaptive{LeasePolicy::Mode::kAdaptive,
                         2 * 100ull, 16 * 100ull};  // [2,16] slots * 100 B
    Lease lease = gov.Acquire(LeaseKind::kExpertCache, start_slots * 100ull,
                              adaptive);
    return RuntimeTuner(&gov, std::move(lease), 100ull,
                        TuningBounds{2, 16}, cfg_of());
}

} // namespace
```

The helper above needs the config in scope — define it as a lambda-free
struct member instead. Final test file (replace the draft above entirely):

```cpp
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/runtime_tuner.h"

using namespace lsxcommon;
using namespace lsxcommon::lifecycle;

namespace {

HardwareProfile FakeHw()
{
    HardwareProfile hw;
    hw.cpu_cores = 16;
    hw.ram_total_bytes = hw.ram_available_bytes = 16ull << 30;
    hw.gpu_name = "FakeGPU";
    hw.gpu_sm_count = 8;
    hw.vram_total_bytes = hw.vram_free_bytes = 8ull << 30;
    return hw;
}

constexpr std::uint64_t kStride = 100;  // bytes per cache slot

struct Harness {
    ResourceGovernor &gov = ResourceGovernor::Instance();
    // window=4 reports; hysteresis gap=8 steps (> one window, so exactly one
    // full window is skipped between resizes).
    RuntimeTuner::Config cfg{/*.window_steps*/ 4, /*grow*/ 0.10,
                             /*shrink*/ 0.02, /*min_interval*/ 8};
    Harness()
    {
        gov.Shutdown();
        GovernorOptions o;
        o.override_hw = true;
        o.hw_override = FakeHw();
        o.vram_keep_out_bytes = 0;
        (void)gov.Initialize(o);
    }
    ~Harness() { gov.Shutdown(); }

    RuntimeTuner Make(std::size_t start_slots,
                      LeasePolicy::Mode mode = LeasePolicy::Mode::kAdaptive)
    {
        LeasePolicy p{mode, 2 * kStride, 16 * kStride};
        Lease lease = gov.Acquire(LeaseKind::kExpertCache,
                                  start_slots * kStride, p);
        return RuntimeTuner(&gov, std::move(lease), kStride,
                            TuningBounds{2, 16}, cfg);
    }
};

} // namespace

TEST_CASE("tuner starts at the lease-derived slot count", "[tuner]")
{
    Harness h;
    auto t = h.Make(4);
    REQUIRE(t.Slots() == 4);
    REQUIRE(t.Resizes() == 0);
}

TEST_CASE("sustained misses grow the cache x2 within bounds", "[tuner]")
{
    Harness h;
    auto t = h.Make(4);
    // Window 1: heavy misses -> grow 4 -> 8 (first window always acts).
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 8);
    REQUIRE(t.Resizes() == 1);
    // Window 2 ends with since_resize_ = 4 < min_interval (8) -> skipped.
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 8);
    REQUIRE(t.Resizes() == 1);
    // Window 3 ends with since_resize_ = 8 -> gate open -> grow 8 -> 16 (max).
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 16);
    REQUIRE(t.Resizes() == 2);
    // Already at max: further misses do nothing.
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 16);
    REQUIRE(t.Resizes() == 2);
}

TEST_CASE("sustained hits shrink the cache", "[tuner]")
{
    Harness h;
    auto t = h.Make(8);
    for (int i = 0; i < 4; ++i) t.Report(10, 0);  // ratio 0 <= shrink 0.02
    REQUIRE(t.Slots() == 4);
}

TEST_CASE("growth is soft-refused when the budget is occupied", "[tuner]")
{
    Harness h;
    auto t = h.Make(4);
    REQUIRE(t.CacheLease().Valid());
    // Fill the remaining budget exactly: growth (+400 B) would exceed it.
    const std::uint64_t rest =
        (8ull << 30) - t.CacheLease().Bytes();
    auto blocker = h.gov.Acquire(LeaseKind::kPermanentWeights, rest,
                                 LeasePolicy{});
    REQUIRE(blocker.Valid());
    for (int i = 0; i < 4; ++i) t.Report(0, 10);
    REQUIRE(t.Slots() == 4);   // refused: size kept
    REQUIRE(t.Resizes() == 0); // no crash, soft failure
}
```

CMake target `test_runtime_tuner` (Task 1 Step 3 shape).

- [ ] **Step 3: Build — expect FAIL** (undefined RuntimeTuner)

- [ ] **Step 4: Implement `src/runtime_tuner.cpp`**

```cpp
#include "lsxcommon/runtime_tuner.h"

#include <algorithm>

#include <spdlog/spdlog.h>

namespace lsxcommon {
namespace lifecycle {

RuntimeTuner::RuntimeTuner(ResourceGovernor *gov, Lease lease,
                           std::uint64_t slot_stride_bytes,
                           const TuningBounds &bounds, const Config &cfg)
    : gov_(gov), lease_(std::move(lease)),
      stride_(slot_stride_bytes > 0 ? slot_stride_bytes : 1),
      bounds_(bounds), cfg_(cfg)
{
    slots_ = lease_.Valid()
                 ? std::clamp<std::size_t>(
                       static_cast<std::size_t>(lease_.Bytes() / stride_),
                       bounds_.expert_cache_slots_min,
                       bounds_.expert_cache_slots_max)
                 : bounds_.expert_cache_slots_min;
    // Large-but-safe value: the first full window passes the hysteresis gate
    // (a ~SIZE_MAX initializer would wrap to 0 on the first ++).
    since_resize_ = std::size_t{1} << 30;
}

void RuntimeTuner::Report(std::int64_t hits_delta, std::int64_t misses_delta)
{
    w_hits_ += hits_delta;
    w_misses_ += misses_delta;
    ++window_pos_;
    ++since_resize_;
    if (window_pos_ < cfg_.window_steps) return;
    window_pos_ = 0;

    const std::int64_t total = w_hits_ + w_misses_;
    const double miss_ratio =
        total > 0 ? static_cast<double>(w_misses_) / static_cast<double>(total)
                  : 0.0;
    w_hits_ = 0;
    w_misses_ = 0;
    if (total <= 0 || since_resize_ < cfg_.min_interval_steps) return;

    if (miss_ratio >= cfg_.grow_miss_ratio &&
        slots_ < bounds_.expert_cache_slots_max) {
        const std::size_t want =
            std::min(slots_ * 2, bounds_.expert_cache_slots_max);
        if (lease_.Resize(want * stride_)) {
            spdlog::info("[tuner] expert cache {} -> {} slots", slots_, want);
            slots_ = want;
            ++resizes_;
            since_resize_ = 0;
        }
    } else if (miss_ratio <= cfg_.shrink_miss_ratio &&
               slots_ > bounds_.expert_cache_slots_min) {
        const std::size_t want =
            std::max(slots_ / 2, bounds_.expert_cache_slots_min);
        if (lease_.Resize(want * stride_)) {
            spdlog::info("[tuner] expert cache {} -> {} slots", slots_, want);
            slots_ = want;
            ++resizes_;
            since_resize_ = 0;
        }
    }
}

} // namespace lifecycle
} // namespace lsxcommon
```

Add `src/runtime_tuner.cpp` to `libs/lsxcommon/CMakeLists.txt`.

- [ ] **Step 5: Build + run — expect PASS**

- [ ] **Step 6: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/runtime_tuner.h libs/lsxcommon/src/runtime_tuner.cpp libs/lsxcommon/CMakeLists.txt tests/CMakeLists.txt tests/test_runtime_tuner.cpp
git commit -m "feat: RuntimeTuner with hysteresis and soft governor-driven resize"
```

---

### Task 8: Engine wiring + residency mutex + orchestrator locks

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/model.h` (add `ResidencyMutex()` pure virtual)
- Modify: `libs/lsxcommon/include/lsxcommon/model_base.h` (member + override)
- Modify: `libs/lsxcommon/src/model_engine.cpp:15-26` (governor init/shutdown)
- Modify: `libs/lsxcommon/src/models/lsx_orchestrator.cpp:205-246` (shared locks)

- [ ] **Step 1: `model.h`** — add `#include <shared_mutex>` next to the existing
includes and, in the `IModel` public section (after `Tokenizer()`), add:

```cpp
    // Concurrency groundwork (spec: leases and residency locks). Infer()
    // takes a shared lock; unload-during-inference / parallel dispatch take
    // unique. One mutex per model instance, owned by ModelBase.
    virtual std::shared_mutex &ResidencyMutex() = 0;
```

- [ ] **Step 2: `model_base.h`** — add `#include <shared_mutex>`; in the public
section (after `Tokenizer()`) add:

```cpp
    std::shared_mutex &ResidencyMutex() override { return residency_mutex_; }
```

and in the protected section (next to `resident_bytes_`):

```cpp
    std::shared_mutex residency_mutex_;
```

- [ ] **Step 3: `model_engine.cpp`** — add `#include "lsxcommon/resource_governor.h"`
and wire the lifecycle:

```cpp
bool InferenceEngine::InitializeEngine()
{
    ResourceGovernor::Instance().Initialize(GovernorOptions{});
    initialized_.store(true, std::memory_order_release);
    spdlog::debug("[model_engine] initialized");
    return true;
}

void InferenceEngine::ShutdownEngine()
{
    initialized_.store(false, std::memory_order_release);
    ResourceGovernor::Instance().Shutdown();
    spdlog::debug("[model_engine] shutdown");
}
```

- [ ] **Step 4: `lsx_orchestrator.cpp`** — add `#include <shared_mutex>`; wrap both
`Infer` calls (lines ~214 and ~236) with the model's shared residency lock:

```cpp
                std::shared_lock<std::shared_mutex> res_lock(glm_->ResidencyMutex());
                auto res = glm_->Infer(req);
```
```cpp
                std::shared_lock<std::shared_mutex> res_lock(deepseek_->ResidencyMutex());
                auto res = deepseek_->Infer(req);
```

- [ ] **Step 5: Build everything — expect PASS** (any other direct `IModel`
implementer would now fail to compile; fix by inheriting `ModelBase` or
overriding `ResidencyMutex()` — grep first: `rg -l "public IModel|: *IModel" libs apps tests`)

```powershell
& $cmake --build build --config Debug 2>&1 | Select-Object -Last 25
```

- [ ] **Step 6: Run affected suites**

```powershell
& "build\bin\Debug\test_model_engine.exe"
& "build\bin\Debug\test_integrated_pipeline.exe"
& "build\bin\Debug\test_model_factory.exe"
```

- [ ] **Step 7: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/model.h libs/lsxcommon/include/lsxcommon/model_base.h libs/lsxcommon/src/model_engine.cpp libs/lsxcommon/src/models/lsx_orchestrator.cpp
git commit -m "feat: governor-engine wiring and per-model residency locks"
```

---

### Task 9: GLM expert-cache slot override + generation telemetry

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference.h:386-402` (`GlmGenerationResult` telemetry fields)
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference_load.h` (override API)
- Modify: `libs/lsxcommon/src/glm_inference.cu` (atomic override + two sizing sites + counter plumbing)

- [ ] **Step 1: `glm_inference.h`** — extend `GlmGenerationResult` (after
`final_logits`):

```cpp
    // Expert device-cache counters accumulated over the run (0 when no cache
    // was active). Consumed by the governor's RuntimeTuner between requests.
    std::int64_t cache_hits = 0;
    std::int64_t cache_misses = 0;
```

- [ ] **Step 2: `glm_inference_load.h`** — add near `load_glm_expert_index`:

```cpp
// Overrides the expert device-cache slot count used by the generation loop,
// bypassing the LSX_GLM_EXPERT_CACHE_SLOTS env heuristic (governor plan +
// RuntimeTuner drive it). Negative restores the env/default behavior; 0
// disables the cache (same as env 0).
void SetGlmExpertCacheSlotsOverride(int slots);
int GlmExpertCacheSlotsOverride();
```

- [ ] **Step 3: `glm_inference.cu`** — three edits:

1. Define the override (file scope, e.g. near the top after includes):

```cpp
namespace lsxcommon {
namespace glm {
namespace {
std::atomic<int> g_expert_cache_slots_override{-1};
} // namespace

void SetGlmExpertCacheSlotsOverride(int slots)
{
    g_expert_cache_slots_override.store(slots, std::memory_order_relaxed);
}
int GlmExpertCacheSlotsOverride()
{
    return g_expert_cache_slots_override.load(std::memory_order_relaxed);
}
} // namespace glm
} // namespace lsxcommon
```

(add `#include <atomic>` if absent)

2. At both cache-sizing sites (`LSX_GLM_EXPERT_CACHE_SLOTS` reads at
   `glm_inference.cu:565-572` and `:1205-1273`): read the override FIRST and
   skip the env/default/halving logic when it is >= 0:

```cpp
int slots = GlmExpertCacheSlotsOverride();
if (slots < 0) {
    // existing env / default-64 / cudaMemGetInfo halving logic, unchanged
    ...
}
```

Read the surrounding code at both sites and wrap the existing computation —
behavior without the override must be bit-identical to today.

3. Where `RunGlmRealGeneration` fills its `GlmGenerationResult` (search for
   the assignment of `resident_bytes` / `total_ms` into the result struct):
   copy the loop-context cache counters when the cache exists (the loop
   context carries `GlmExpertDeviceCache *xcache`, `glm_inference.cu:269-307`):

```cpp
if (ctx.xcache != nullptr) {
    out.cache_hits = ctx.xcache->hits;
    out.cache_misses = ctx.xcache->misses;
}
```

(match the actual variable names of the result struct and loop context in the
file).

- [ ] **Step 4: Build the GLM object library + run GLM loader tests — expect PASS**

```powershell
& $cmake --build build --config Debug --target lsxcommon_glm_cu test_glm_load_fast test_glm_loader 2>&1 | Select-Object -Last 25
& "build\bin\Debug\test_glm_load_fast.exe"
& "build\bin\Debug\test_glm_loader.exe"
```

- [ ] **Step 5: Commit**

```powershell
git add libs/lsxcommon/include/lsxcommon/glm_inference.h libs/lsxcommon/include/lsxcommon/glm_inference_load.h libs/lsxcommon/src/glm_inference.cu
git commit -m "feat: governor-driven GLM expert-cache slot override and cache telemetry"
```

---

### Task 10: GlmV5Model load-once lifecycle rewrite

**Files:**
- Modify: `libs/lsxcommon/src/glm_v5_model.h` (full rewrite)
- Modify: `libs/lsxcommon/src/glm_v5_model.cpp` (full rewrite)

- [ ] **Step 1: Rewrite `glm_v5_model.h`**

```cpp
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

#include "lsxcommon/glm_inference_load.h"
#include "lsxcommon/lifecycle_strategy.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/runtime_tuner.h"

namespace lsxcommon {

class GlmV5Model final : public ModelBase {
public:
    static const std::string kType;
    explicit GlmV5Model(ModelInitConfig cfg);
    ~GlmV5Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return kType; }

    // Diagnostics/test hook: the persistent pinned weight set. Valid between
    // Load() and Unload(); pointer identity is stable across Infer calls
    // (weights are loaded ONCE, spec: load-once lifecycle).
    const glm::GlmPinnedWeights &PinnedWeights() const { return pw_; }

private:
    bool EnsureExecutionContext();
    std::uint64_t EstimateExpertMeanBytes() const;
    void TeardownWeights();

    glm::GlmPinnedWeights pw_;
    glm::GlmExpertDb experts_{};
    cublasHandle_t cublas_ = nullptr;
    cudaStream_t stream_ = nullptr;
    lifecycle::Lease permanent_lease_;
    lifecycle::Lease cache_lease_;
    std::unique_ptr<lifecycle::RuntimeTuner> tuner_;
    std::int64_t tuner_last_hits_ = 0;
    std::int64_t tuner_last_misses_ = 0;
};

} // namespace lsxcommon
```

- [ ] **Step 2: Rewrite `glm_v5_model.cpp`**

```cpp
#include "glm_v5_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <shared_mutex>

#include "lsxcommon/glm_convert.h"
#include "lsxcommon/glm_inference.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon {

const std::string GlmV5Model::kType = "glm-5.2";

GlmV5Model::GlmV5Model(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)} {}

GlmV5Model::~GlmV5Model() { Unload(); }

bool GlmV5Model::EnsureExecutionContext()
{
    if (!stream_ && cudaStreamCreateWithFlags(&stream_,
                                              cudaStreamNonBlocking) !=
                        cudaSuccess) {
        stream_ = nullptr;
        return false;
    }
    if (!cublas_ && cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
        cublas_ = nullptr;
        return false;
    }
    cublasSetStream(cublas_, stream_);
    return true;
}

std::uint64_t GlmV5Model::EstimateExpertMeanBytes() const
{
    // NVFP4 routed expert: three [kGlmMoeIntermediate x kGlmHidden] matrices,
    // each packed E2M1 nibbles (elems/2 B) + E4M3 per-block scales (elems/16 B).
    constexpr std::int64_t elems =
        static_cast<std::int64_t>(glm::kGlmMoeIntermediate) * glm::kGlmHidden;
    constexpr std::int64_t per_matrix = elems / 2 + elems / 16;
    return static_cast<std::uint64_t>(per_matrix) * 3;
}

void GlmV5Model::TeardownWeights()
{
    tuner_.reset();          // releases the adaptive expert-cache lease
    permanent_lease_ = {};   // RAII release
    cache_lease_ = {};
    glm::free_glm_expert_index(experts_);
    pw_.Free();
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    resident_bytes_ = 0;
}

bool GlmV5Model::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    namespace fs = std::filesystem;
    fs::path idx = fs::path(cfg_.model_path) / "logestix" / "model.index.json";
    if (!fs::exists(idx))
        idx = fs::path(cfg_.model_path) / "model.index.json";
    auto topo = glm::load_topology(idx.string());
    if (!topo.error_msg.empty()) {
        spdlog::error("[glm-model] topology error in {}: {}",
                      idx.string(), topo.error_msg);
        tok_db_.reset();
        return false;
    }
    vram_ = topo.vram;

    auto &gov = ResourceGovernor::Instance();
    if (!gov.IsInitialized()) {
        spdlog::warn("[glm-model] governor not initialized; standalone init");
        GovernorOptions gopts;
        if (!gov.Initialize(gopts)) {
            spdlog::error("[glm-model] governor init failed");
            tok_db_.reset();
            return false;
        }
    }

    if (!EnsureExecutionContext()) {
        spdlog::error("[glm-model] cublas/stream init failed");
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    const cudaError_t e = glm::LoadGlmPinnedWeights(cfg_.model_path, pw_, stream_);
    if (e != cudaSuccess ||
        pw_.hidden != glm::kGlmHidden ||
        pw_.layers.size() != static_cast<size_t>(glm::kGlmNumLayers) ||
        !pw_.d_embed || !pw_.d_head) {
        spdlog::error("[glm-model] pinned-weight load/validation failed: {}",
                      e != cudaSuccess ? cudaGetErrorString(e)
                                       : "geometry mismatch");
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    experts_ = glm::load_glm_expert_index(cfg_.model_path, glm::kGlmNumLayers,
                                          glm::kGlmNumRouted);
    if (experts_.num_layers != glm::kGlmNumLayers ||
        experts_.experts_per_layer != glm::kGlmNumRouted) {
        spdlog::error("[glm-model] expert index incomplete");
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    lifecycle::LifecycleInput in;
    in.model_type = kType;
    in.archetype = lifecycle::Archetype::MoE;
    in.vram = topo.vram;
    in.expert_mean_bytes = EstimateExpertMeanBytes();
    const auto &plan = lifecycle::GetOrBuildPlan(in);

    permanent_lease_ = gov.Acquire(LeaseKind::kPermanentWeights,
                                   pw_.resident_bytes, LeasePolicy{});
    if (!permanent_lease_.Valid() && !gov.options().relax_memory_refusal) {
        spdlog::error("[glm-model] permanent-weight lease refused ({} bytes)",
                      pw_.resident_bytes);
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    LeasePolicy adaptive{LeasePolicy::Mode::kAdaptive,
                         plan.tuner_bounds.expert_cache_slots_min *
                             in.expert_mean_bytes,
                         plan.expert_cache_max_bytes};
    cache_lease_ = gov.Acquire(LeaseKind::kExpertCache,
                               plan.expert_cache_slots * in.expert_mean_bytes,
                               adaptive);
    tuner_ = std::make_unique<lifecycle::RuntimeTuner>(
        &gov, std::move(cache_lease_), in.expert_mean_bytes, plan.tuner_bounds);
    glm::SetGlmExpertCacheSlotsOverride(static_cast<int>(tuner_->Slots()));
    tuner_last_hits_ = tuner_last_misses_ = 0;

    resident_bytes_ = pw_.resident_bytes;
    spdlog::info("[glm-model] loaded once: resident={} bytes, plan: cache "
                 "slots={} streams={} staging={}",
                 pw_.resident_bytes, plan.expert_cache_slots,
                 plan.copy_streams, plan.staging_ring_bytes);
    FinishLoad();
    return true;
}

gpu::InferenceResult GlmV5Model::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_ || !pw_.d_embed) {
        r.error_msg = "model not loaded (call Load(); weights are no longer "
                      "reloaded per Infer call)";
        r.ok = false;
        return r;
    }

    glm::GlmGenerationResult gen;
    {
        std::shared_lock<std::shared_mutex> residency(residency_mutex_);
        const cudaError_t e = glm::RunGlmRealGeneration(
            cublas_, stream_, pw_, experts_, req.input_ids,
            req.max_gen_tokens, gen);
        if (e != cudaSuccess || !gen.ok) {
            r.error_msg = gen.ok ? cudaGetErrorString(e) : gen.error_msg;
            r.ok = false;
            return r;
        }
    }

    if (tuner_) {  // between requests: adapt the expert cache (spec 4)
        tuner_->Report(gen.cache_hits - tuner_last_hits_,
                       gen.cache_misses - tuner_last_misses_);
        tuner_last_hits_ = gen.cache_hits;
        tuner_last_misses_ = gen.cache_misses;
        glm::SetGlmExpertCacheSlotsOverride(static_cast<int>(tuner_->Slots()));
        ResourceGovernor::Instance().ReconcileVram();
    }

    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.num_layers = glm::kGlmNumLayers;
    if (!gen.tokens.empty()) {
        r.first_gen_tok_id = gen.tokens[0];
        r.first_gen_maxv = gen.first_maxv;
    }
    r.gen_tokens = std::move(gen.tokens);
    r.final_logits = std::move(gen.final_logits);
    r.kv_slots_used = gen.kv_slots_used;
    r.ttft_ms = gen.ttft_ms;
    r.total_time_ms = gen.total_ms;
    r.tokens_per_sec = (gen.total_ms > 0.0)
                           ? (r.gen_tokens.size() / (gen.total_ms / 1000.0))
                           : 0.0;
    r.eos_terminated = gen.eos_terminated;
    r.weight_resident_bytes = gen.resident_bytes;

    if (!r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
    return r;
}

void GlmV5Model::Unload()
{
    TeardownWeights();
    tok_db_.reset();
    loaded_ = false;
}

gpu::InferenceResult GlmV5Model::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

gpu::BenchmarkResult GlmV5Model::Benchmark(int num_tokens)
{
    gpu::BenchmarkResult r;
    if (!loaded_ || !pw_.d_embed) { r.error_msg = "model not loaded"; return r; }
    const std::vector<int32_t> prompt = {0, 0, 0, 0};

    glm::GlmGenerationResult gr;
    {
        std::shared_lock<std::shared_mutex> residency(residency_mutex_);
        const cudaError_t e = glm::RunGlmRealGeneration(
            cublas_, stream_, pw_, experts_, prompt, num_tokens, gr);
        if (e != cudaSuccess || !gr.ok) {
            r.error_msg = gr.ok ? cudaGetErrorString(e) : gr.error_msg;
            return r;
        }
    }
    r.ok = true;
    r.num_tokens = static_cast<int>(gr.tokens.size());
    r.num_layers = glm::kGlmNumLayers;
    r.total_time_ms = gr.total_ms;
    r.tokens_per_sec = gr.total_ms > 0.0
        ? (static_cast<double>(r.num_tokens) / gr.total_ms) * 1000.0 : 0.0;
    r.resident_bytes = gr.resident_bytes;
    return r;
}

} // namespace lsxcommon
```

- [ ] **Step 3: Audit every Infer/SelfCheck/Benchmark caller for a preceding Load()**

```powershell
rg -n --glob '!third_party/**' "->(Infer|SelfCheck|Benchmark)\(" apps tests libs
```
Every site must be preceded by `Load()` (directly or via the engine/CLI flow).
Known sites: `apps/lsxfabric` CLI flows, `taskmosaic`, `lsxcli`,
`libs/lsxcommon/src/models/lsx_orchestrator.cpp`, `test_model_engine`,
`test_integrated_pipeline`, `test_glm_*`. If any site calls Infer without
Load, insert the Load call (models previously self-loaded inside Infer).

- [ ] **Step 4: Build + run the full GLM and engine suites — expect PASS
(real-model cases SKIP here)**

```powershell
& $cmake --build build --config Debug --target test_glm_model test_glm_selfcheck test_model_engine test_integrated_pipeline test_glm_loader 2>&1 | Select-Object -Last 25
& "build\bin\Debug\test_glm_model.exe"
& "build\bin\Debug\test_model_engine.exe"
& "build\bin\Debug\test_integrated_pipeline.exe"
```

- [ ] **Step 5: Commit**

```powershell
git add libs/lsxcommon/src/glm_v5_model.h libs/lsxcommon/src/glm_v5_model.cpp
git commit -m "feat: GLM-5.2 load-once lifecycle with governor plan and tuner"
```

---

### Task 11: Lifecycle integration test (synthetic + real-model gated)

**Files:**
- Create: `tests/test_glm_model_lifecycle.cu`
- Modify: `tests/CMakeLists.txt` (OBJECT `.cu` target, mirror `test_glm_load_fast_cu`, lines 1119-1162)

- [ ] **Step 1: Create `tests/test_glm_model_lifecycle.cu`**

```cpp
// GLM-5.2 lifecycle integration tests (ResourceGovernor spec, Testing 5).
// Synthetic part always runs (no weights needed); the real-model part is
// gated by LSX_TEST_GLM_MODEL_PATH and SKIPs otherwise — the operator runs
// it on the GLM-capable machine.

#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/glm_convert.h>
#include <lsxcommon/model_factory.h>

#include "glm_test_fixture.h"
#include "glm_v5_model.h"  // lsxcommon src dir (mirrored include dirs below)

namespace fs = std::filesystem;
using lsxcommon::GlmV5Model;

namespace {

bool RealModelPath(std::string *out)
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    if (!env || !*env) return false;
    *out = env;
    return true;
}

// Builds a converted synthetic GLM model (hidden=16 -> fails the frozen
// geometry validation, which is exactly what the negative test exercises).
std::string MakeSyntheticConvertedModel(const std::string &tag)
{
    auto base = fs::temp_directory_path() / ("logestix_glm_lifecycle_" + tag);
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);
    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    glmtest::WriteTokenizer(in);
    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());
    return out.string();
}

} // namespace

TEST_CASE("glm lifecycle: failed load tears down cleanly and is retryable",
          "[glm][lifecycle]")
{
    const std::string path = MakeSyntheticConvertedModel("neg");

    lsxcommon::ModelInitConfig cfg{path, GlmV5Model::kType, /*max_gen*/ 2};
    GlmV5Model model(std::move(cfg));

    // hidden=16 violates kGlmHidden(6144): Load must fail AFTER uploading
    // weights, release every lease/buffer, and stay retryable.
    REQUIRE_FALSE(model.Load());
    REQUIRE(model.PinnedWeights().d_embed == nullptr);  // torn down
    model.Unload();                                     // idempotent
    REQUIRE_FALSE(model.Load());                        // deterministic retry
    model.Unload();
}

TEST_CASE("glm lifecycle: load-once reuse across Infer calls",
          "[glm][lifecycle][real-model]")
{
    std::string path;
    if (!RealModelPath(&path)) {
        SUCCEED("LSX_TEST_GLM_MODEL_PATH not set - SKIP (no GLM weights)");
        return;
    }

    lsxcommon::ModelInitConfig cfg{path, GlmV5Model::kType, /*max_gen*/ 2};
    GlmV5Model model(std::move(cfg));
    REQUIRE(model.Load());

    const void *embed_before = model.PinnedWeights().d_embed;
    REQUIRE(embed_before != nullptr);

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 0, 0, 0};  // stable prompt, same as Benchmark
    req.max_gen_tokens = 2;

    auto r1 = model.Infer(req);
    REQUIRE(r1.ok);
    // Weights were NOT reloaded: device pointer identity is stable.
    REQUIRE(model.PinnedWeights().d_embed == embed_before);

    auto r2 = model.Infer(req);
    REQUIRE(r2.ok);
    REQUIRE(r2.gen_tokens == r1.gen_tokens);
    REQUIRE(r2.final_logits == r1.final_logits);  // bit-exact determinism

    model.Unload();
    REQUIRE(model.PinnedWeights().d_embed == nullptr);
    REQUIRE(model.Load());  // full re-load after unload works
    REQUIRE(model.PinnedWeights().d_embed != nullptr);
    model.Unload();
}
```

- [ ] **Step 2: Register in `tests/CMakeLists.txt`** — copy the
`test_glm_load_fast_cu` block verbatim (lines 1119-1162) and rename every
`test_glm_load_fast` occurrence to `test_glm_model_lifecycle` (the OBJECT
library, the exe, the include dirs, the compile definitions, the POST_BUILD
DLL copies, and `add_test`).

- [ ] **Step 3: Build + run — synthetic part PASS, real-model part SKIP**

```powershell
& $cmake --build build --config Debug --target test_glm_model_lifecycle 2>&1 | Select-Object -Last 25
& "build\bin\Debug\test_glm_model_lifecycle.exe"
```
Expected: `All tests passed`, with the second test case reporting the SKIP
success note.

- [ ] **Step 4: Commit**

```powershell
git add tests/test_glm_model_lifecycle.cu tests/CMakeLists.txt
git commit -m "test: GLM lifecycle integration suite (synthetic negative + real-model gated)"
```

---

### Task 12: Full verification sweep

**Files:** none (verification only)

- [ ] **Step 1: Full build (all targets, triggers CMake reconfigure)**

```powershell
& $cmake --build build --config Debug 2>&1 | Select-Object -Last 30
```
Expected: zero errors.

- [ ] **Step 2: Full test suite**

```powershell
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $ctest --preset windows-debug 2>&1 | Select-Object -Last 40
```
Expected: 100% tests passed (GLM/DeepSeek real-model cases SKIP without
weights — same baseline as before this feature).

- [ ] **Step 3: Verify the new suites explicitly**

```powershell
& "build\bin\Debug\test_resource_governor.exe"
& "build\bin\Debug\test_hardware_profile.exe"
& "build\bin\Debug\test_lifecycle_plan.exe"
& "build\bin\Debug\test_runtime_tuner.exe"
& "build\bin\Debug\test_glm_model_lifecycle.exe"
```

- [ ] **Step 4: Commit any residual fixes**

```powershell
git status
git add -A
git commit -m "fix: resource-governor lifecycle sweep fixes"
```
(only if Step 1-3 required changes; otherwise skip)

---

## Handoff Notes for the GLM machine (operator)

1. Fetch the branch, build (Release or Debug), then:
   `set LSX_TEST_GLM_MODEL_PATH=<converted glm-5.2 dir>` and run
   `test_glm_model_lifecycle` — the real-model case verifies pointer-stable
   weight reuse and bit-identical `final_logits` across two Infer calls.
2. Run `test_glm_selfcheck` (3-cycle stress): bit-exact `final_logits`
   baselines must stay valid — the loader path is untouched, only its
   lifetime moved from per-Infer to Load().
3. Watch for the `[governor]`/`[tuner]` spdlog lines: plan summary at load,
   expert-cache resizes between requests, any VRAM drift warnings.
