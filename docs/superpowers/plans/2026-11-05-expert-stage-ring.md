# Expert Staging Ring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the shared `ExpertStore` staging path from pageable per-expert `cudaMemcpyAsync` (measured 5.2 GB/s) to a pinned ring fed by a parallel `pread` pool (measured 10.5 GB/s at 4 threads, ceiling 39.8 GB/s), so host-streamed MoE families stop being bound by pageable-copy serialisation.

**Architecture:** The store gains three things it does not have: file-plane descriptors for each resident expert, a pinned ring of staging buffers with per-slot DMA events, and a persistent `pread` thread pool. Families keep their device layout and call `EnqueueStage`/`AwaitStage` instead of their own copy loop; the ring is off by default and only `kimi-k3.1-pro` is wired in this plan.

**Tech Stack:** C++20, Arrow 26.0.1 (`MemoryMappedFile` only — `ReadableFile` is forbidden by guard rule R-A1), CUDA runtime (`cudaHostAlloc`, `cudaMemcpyAsync`, `cudaEventDisableTiming`), Catch2 v3.16.1, CMake via `lsx_add_test`.

**Spec:** `docs/superpowers/specs/2026-11-05-expert-stage-ring-design.md` — the plan argues from the spec; executors read both.

## Global Constraints

- Header reachability: `expert_store.h` is included from `.cu`/`.cuh` TUs. No `<absl/container/*>` in any new header, ever (NVCC fails in `raw_hash_map` with C2794/C2938, machine-checked by `test_absl_container_guard`).
- Logging only via `lsxcommon::log` (spdlog). No `std::cout`, no `printf` in library code (`test_logging_guard`).
- Arrow reads go through `MemoryMappedFile`; `ReadableFile` and raw `mmap` in consumer code are forbidden (`test_arrow_usage_guard`, R-A1).
- Do **not** use `cuMemcpyBatchAsync` in the ring: measured 4.2–5.0 GB/s against 5.2 for plain pageable copies and 10.5 for the pinned arm. It is a loss at every size and attribute combination tested.
- Ring is **off by default** (`LSX_EXPERT_STAGE_RING` unset = legacy path, byte-identical). Task 6 is the only task that wires a family, and it is opt-in.
- Never hand-edit `tests/perf/perf_baselines.json`; the harness owns it. A SKIP or `not run` is a gap, never a pass.
- Kernel/latency/bandwidth budgets go through `kernel_budget::Require` with a `gpu_key`, never a bare constant.
- Commit prefixes: `feat:`/`fix:`/`perf:`/`test:`/`docs:`/`build:`. Commit only on explicit operator request.
- GPU exclusivity: one GPU workload at a time. Tasks 6 and 7 need the card to themselves.

## Review Focus

Conditions the spec implies but no task's ordinary tests would catch, most likely to bite first:

1. **A ring slot refilled while a DMA reading it is still in flight** — expected: the first expert's bytes arrive intact; actual without the event check: silent corruption of one expert per wrap.
2. **A plane descriptor pointing at the wrong plane** — e.g. `w1_scales` landing where `w2_packed` should be. Expected: model output unchanged; actual: a plausible, wrong model with no error. Pinned by byte-identity against the file (Task 2 and Task 3).
3. **An eviction landing between `EnqueueStage` and `AwaitStage`** — expected: fall back to the legacy copy, or keep reading from the immutable file; actual if implemented over the mapping: reading unmapped memory.
4. **`pread` returning short or `EINTR`** — expected: the full plane; actual without a loop: a partially filled staging slot, i.e. garbage weights in one expert.
5. **Fewer ring slots than the layer's top-k** — expected: correct output, just less overlap; actual without the recycle-stream-wait: a host-side stall that makes the change slower than the legacy path.

---

## File Structure

- `libs/lsxcommon/include/lsxcommon/pread_pool.h` — **create**. Persistent N-thread `pread` pool, host-only, plain `std::` + POSIX. One responsibility: fill a caller's buffers from given `{fd, offset, len}` triples with N threads.
- `libs/lsxcommon/include/lsxcommon/expert_store.h` — **modify**. Plane descriptors, ring, `EnqueueStage`/`AwaitStage`, counters, `Free()` teardown. The store stays the single owner of expert residency.
- `libs/lsxcommon/src/kimi_moe_k3_inference_pipeline.cu` — **modify** (Task 6 only). Replace the 6-copy loop with the ring behind the env flag.
- `tests/test_pread_pool.cpp` — **create**. Pool unit tests.
- `tests/test_expert_store.cpp` — **modify**. Descriptors, ring hazards, byte identity, no-op.
- `tests/test_expert_store_stage.cu` — **create**. The bandwidth budget test (needs a stream and pinned memory; the `.cu` shape follows `lsx_add_cuda_test`).
- `tests/CMakeLists.txt` — **modify**. Register the two new targets in the same commits as their sources.

---

### Task 1: Persistent `pread` pool

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/pread_pool.h`
- Create: `tests/test_pread_pool.cpp`
- Modify: `tests/CMakeLists.txt` (add `test_pread_pool`, label `fast`, no GPU)

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```cpp
  namespace lsxcommon::io {
  struct PreadJob { int fd; int64_t offset; void *dst; size_t len; };
  class PreadPool {
   public:
      explicit PreadPool(int threads);          // threads <= 0 => 4
      ~PreadPool();
      // Fills every job's dst from its fd. Blocks until all jobs complete.
      // Returns the number of jobs that read their full len.
      size_t Run(const std::vector<PreadJob> &jobs);
      int Threads() const;
  };
  }
  ```

- [ ] **Step 1: Write the failing test**

`tests/test_pread_pool.cpp`:
```cpp
#include <catch2/catch_test_macros.hpp>
#include "lsxcommon/pread_pool.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace {
// One file per expert, distinct byte pattern so a job that reads the wrong
// file is visible rather than plausible.
struct Fixture {
    std::vector<std::string> paths;
    explicit Fixture(int n, size_t bytes) {
        auto dir = std::filesystem::temp_directory_path() / "lsx_pread_pool_test";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        for (int i = 0; i < n; ++i) {
            auto p = (dir / ("f" + std::to_string(i))).string();
            std::vector<uint8_t> buf(bytes, static_cast<uint8_t>(i + 1));
            int fd = ::open(p.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            REQUIRE(fd >= 0);
            REQUIRE(::write(fd, buf.data(), buf.size()) ==
                    static_cast<ssize_t>(buf.size()));
            ::close(fd);
            paths.push_back(p);
        }
    }
    ~Fixture() { std::filesystem::remove_all(
        std::filesystem::temp_directory_path() / "lsx_pread_pool_test"); }
};
}  // namespace

TEST_CASE("PreadPool fills every job from its own file", "[pread_pool]")
{
    Fixture fx(8, 64 * 1024);
    lsxcommon::io::PreadPool pool(4);
    std::vector<std::vector<uint8_t>> out(8, std::vector<uint8_t>(64 * 1024));
    std::vector<lsxcommon::io::PreadJob> jobs;
    std::vector<int> fds;
    for (int i = 0; i < 8; ++i) {
        const int fd = ::open(fx.paths[i].c_str(), O_RDONLY);
        REQUIRE(fd >= 0);
        fds.push_back(fd);
        jobs.push_back({fd, 0, out[i].data(), out[i].size()});
    }
    REQUIRE(pool.Run(jobs) == 8);
    for (int i = 0; i < 8; ++i)
        REQUIRE(out[i][0] == static_cast<uint8_t>(i + 1));
    for (int fd : fds) ::close(fd);
}

TEST_CASE("PreadPool reads at an offset, not from the start", "[pread_pool]")
{
    Fixture fx(1, 4096);
    // Byte 0 of file 0 is 0x01 everywhere; stage a 1024-byte window at 512 and
    // make sure the destination is the window's content, not the file head.
    lsxcommon::io::PreadPool pool(1);
    std::vector<uint8_t> out(1024, 0xEE);
    const int fd = ::open(fx.paths[0].c_str(), O_RDONLY);
    std::vector<lsxcommon::io::PreadJob> jobs{{fd, 512, out.data(), out.size()}};
    REQUIRE(pool.Run(jobs) == 1);
    REQUIRE(out[0] == 0x01);
    REQUIRE(out[1023] == 0x01);
    ::close(fd);
}

TEST_CASE("PreadPool reports a job it could not complete", "[pread_pool]")
{
    Fixture fx(1, 4096);
    lsxcommon::io::PreadPool pool(2);
    std::vector<uint8_t> out(4096);
    const int fd = ::open(fx.paths[0].c_str(), O_RDONLY);
    // Ask for more than the file holds: pread returns a short read and the job
    // must NOT be counted as complete, or a staging slot is silently half full.
    std::vector<lsxcommon::io::PreadJob> jobs{{fd, 0, out.data(), out.size() * 2}};
    REQUIRE(pool.Run(jobs) == 0);
    ::close(fd);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_pread_pool 2>&1 | tail -5`
Expected: FAIL — `pread_pool.h`: No such file or directory.

- [ ] **Step 3: Implement `PreadPool` in `libs/lsxcommon/include/lsxcommon/pread_pool.h`**

Persistent workers, one job queue per `Run`, `std::atomic<size_t>` cursor over the job vector, a `std::condition_variable` to release a generation and to wait for completion. Each job loops `pread` until `len` is read, retrying `EINTR` and treating a short read as incomplete. No Arrow, no CUDA, no absl.

The short-read rule is the one that is not obvious: `pread` may return fewer bytes than asked even on a regular file (`EINTR`, or a concurrent truncate), so a single call is not enough — the third test exists to pin this.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_pread_pool && ./build/linux-debug/tests/test_pread_pool`
Expected: `All tests passed (3 assertions ... in 3 test cases)` with more than 3 assertions — the point is all three cases pass.

- [ ] **Step 5: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/pread_pool.h tests/test_pread_pool.cpp tests/CMakeLists.txt
git commit -m "feat: persistent pread pool for pinned expert staging"
```

---

### Task 2: File-plane descriptors per resident expert

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/expert_store.h` (after `DecodeEntry`, inside `ParseFile`)
- Modify: `tests/test_expert_store.cpp`
- Modify: `tests/test_expert_store_guard.cpp` (only if the guard's sanctioned-site count needs the new name — check, do not assume)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces, in `ExpertStore<TEntry>`:
  ```cpp
  // Six {file_offset, bytes} pairs per flat index, in ExpertSchema order
  // (w1, w2, w3, w1_scales, w2_scales, w3_scales). Populated by ParseFile;
  // all zeros for an entry that is not resident.
  struct PlaneLocator { int64_t offset = 0; int64_t bytes = 0; };
  std::vector<std::array<PlaneLocator, 6>> planes;
  int64_t plane_bytes[6] = {0, 0, 0, 0, 0, 0};  // per-store sanity record
  int64_t file_bytes = 0;                        // from GetSize()
  ```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expert_store.cpp`. The existing fixture writes every plane as a constant byte (`w1=0x01, w2=0x02, w3=0x03, s1=0x80, s2=0x81, s3=0x82`), which is exactly the pattern needed: a locator pointing at the wrong plane produces the wrong byte.
```cpp
TEST_CASE("ExpertStore records where each plane lives in the file",
          "[expert_store][stage]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    REQUIRE(s.planes.size() == 4);

    // Every resident entry has six locators addressing six distinct, non-empty,
    // ascending spans inside the file.
    for (size_t idx = 0; idx < 4; ++idx) {
        const auto &p = s.planes[idx];
        int64_t prev_end = 0;
        for (int k = 0; k < 6; ++k) {
            INFO("idx " << idx << " plane " << k);
            REQUIRE(p[k].bytes > 0);
            REQUIRE(p[k].offset >= prev_end);      // no overlap, in file order
            prev_end = p[k].offset + p[k].bytes;
            REQUIRE(prev_end <= s.file_bytes);     // stays inside the file
        }
    }
    // Plane sizes match the schema, not some other plane's length.
    CHECK(s.planes[0][0].bytes == s.entries[0].w1_bytes);
    CHECK(s.planes[0][3].bytes == s.entries[0].w1_scales_bytes);

    // The decisive check: reading the file at a plane's locator yields that
    // plane's bytes. A base offset that is wrong by 16 bytes (the Arrow header)
    // would make w1 read 0x00-filled alignment instead of 0x01.
    for (size_t idx = 0; idx < 4; ++idx) {
        const uint8_t want[6] = {0x01, 0x02, 0x03, 0x80, 0x81, 0x82};
        for (int k = 0; k < 6; ++k) {
            std::error_code ec;
            auto path = s.paths[idx];
            std::FILE *f = std::fopen(path.c_str(), "rb");
            REQUIRE(f != nullptr);
            REQUIRE(std::fseek(f, static_cast<long>(s.planes[idx][k].offset),
                               SEEK_SET) == 0);
            std::vector<uint8_t> buf(static_cast<size_t>(s.planes[idx][k].bytes));
            REQUIRE(std::fread(buf.data(), 1, buf.size(), f) == buf.size());
            std::fclose(f);
            INFO("idx " << idx << " plane " << k);
            REQUIRE(buf.front() == want[k]);
            REQUIRE(buf.back() == want[k]);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_expert_store && ./build/linux-debug/tests/test_expert_store "[stage]"`
Expected: FAIL to compile — `planes` is not a member.

- [ ] **Step 3: Implement the locators**

In `Load()`: size `planes` with `entries`, and set `file_bytes` from the first successfully opened file's `GetSize()` (all files in a converted tree are uniform; assert uniformity the first time and `log::error` + fail `Load` on a mismatch — a non-uniform tree would make `max_expert_bytes` a lie).

In `ParseFile(size_t idx)`, after `batch` is read and before `DecodeEntry`:
- Mapping base, once per file: `auto head = mmap->ReadAt(0, 8);` then `base = (*head)->address()`. Arrow's `MemoryMappedFile::ReadAt(position, n)` returns `memory_map_->Slice(position, nbytes)` (verified in `arrow/io/file.cc`), so at `position = 0` the slice address **is** the mapping base. Ask for 8 bytes, not the whole file: `ReadAt` calls `MemoryAdviseWillNeed` on the requested range, and advising 18.6 MB per file would resurrect the page-cache prefetch idea that §R17 P5 measured as a 18% loss.
- For each of the six binary/list columns, take the same buffer the decoder uses (`LargeBinaryArray::GetView(0).data()` / `ListArray` values offset for scales) and record `offset = ptr - base`, `bytes = length`.
- Sanity gate, all three at once, failing `ParseFile` with a `log::error` that names the file:
  1. `0 <= offset && offset + bytes <= file_bytes` for all six;
  2. offsets strictly ascending with no overlap, in schema order;
  3. `plane_bytes[]` is identical across every expert parsed for this store (record on first parse, compare after).
- Populate `plane_bytes` from the first parsed entry.

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build --preset linux-debug --target test_expert_store && ./build/linux-debug/tests/test_expert_store`
Expected: all cases pass, including the three pre-existing ones.

- [ ] **Step 5: Verify the guard still holds**

Run: `./build/linux-debug/tests/test_expert_store_guard && ./build/linux-debug/tests/test_arrow_usage_guard`
Expected: both pass. If `test_arrow_usage_guard` reports the new `ReadAt` site, add it to the sanctioned list in that test — do not weaken the rule.

- [ ] **Step 6: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/expert_store.h tests/test_expert_store.cpp
git commit -m "feat: record per-plane file locators for resident experts"
```

---

### Task 3: The ring — `EnqueueStage` / `AwaitStage`

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/expert_store.h` (new members + methods; `Free()` teardown)
- Modify: `tests/test_expert_store.cpp`

**Interfaces:**
- Consumes: `planes`, `plane_bytes`, `file_bytes` (Task 2); `lsxcommon::io::PreadPool` (Task 1).
- Produces, in `ExpertStore<TEntry>`:
  ```cpp
  struct StageRingConfig {
      size_t slots = 0;          // 0 = disabled
      int pread_threads = 4;     // 0 => 4
      size_t chunk_experts = 1;  // experts staged per pipeline step
  };
  StageRingConfig stage_ring;

  // Allocates `slots` pinned buffers, their events and the pread pool.
  // False + log::error (naming ulimit -l) on allocation failure; safe to call
  // when slots == 0 (returns false, allocates nothing).
  bool InitStageRing();

  // Queues this resident entry's six planes into the ring and DMAs them to
  // `dst_dev + dst_off[k]`. dst_off is indexed in ExpertSchema order, so a
  // family keeps its own device layout. Returns false when the ring is
  // disabled, the entry is not resident, or no slot is free without stalling;
  // the caller then uses its legacy path.
  bool EnqueueStage(size_t idx, uint8_t *dst_dev, const int64_t (&dst_off)[6],
                    cudaStream_t stream);

  // Makes `stream` wait for the DMA queued by the most recent EnqueueStage of
  // `idx`. Never drains the device and never blocks the host.
  void AwaitStage(size_t idx, cudaStream_t stream);

  // Engagement counters (read by tests and by the decode-end log line).
  struct StageStats {
      int64_t staged_experts = 0, staged_bytes = 0;
      int64_t legacy_fallbacks = 0, ring_waits = 0, short_reads = 0;
  } stage_stats;
  ```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expert_store.cpp`:
```cpp
TEST_CASE("ExpertStore staging ring reproduces the file bytes",
          "[expert_store][stage]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    s.stage_ring.slots = 4;
    s.stage_ring.pread_threads = 2;
    REQUIRE(s.InitStageRing());                  // fails loudly on no device

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    const int64_t packed = s.entries[0].w1_bytes;
    const int64_t scaled = s.entries[0].w1_scales_bytes;
    // Deliberately NOT the schema order: a family layout that interleaves
    // weight and scale per plane. The ring must honour dst_off, not assume an
    // order of its own.
    const int64_t off[6] = {0,                    // w1
                            2 * (packed + scaled),// w2
                            (packed + scaled),    // w3
                            packed,               // w1_scales
                            2 * (packed + scaled) + packed,
                            (packed + scaled) + packed};
    const size_t total = static_cast<size_t>(3 * (packed + scaled));
    uint8_t *dev = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&dev), total) == cudaSuccess);

    for (size_t idx = 0; idx < 4; ++idx) {
        REQUIRE(s.EnqueueStage(idx, dev, off, stream));
        s.AwaitStage(idx, stream);
        std::vector<uint8_t> got(total);
        REQUIRE(cudaMemcpy(got.data(), dev, total, cudaMemcpyDeviceToHost) ==
                cudaSuccess);
        // Plane k's destination slice must hold plane k's byte pattern.
        const uint8_t want[6] = {0x01, 0x02, 0x03, 0x80, 0x81, 0x82};
        for (int k = 0; k < 6; ++k)
            CHECK(got[static_cast<size_t>(off[k])] == want[k]);
    }
    CHECK(s.stage_stats.staged_experts == 4);
    CHECK(s.stage_stats.legacy_fallbacks == 0);

    cudaFree(dev);
    cudaStreamDestroy(stream);
}

TEST_CASE("ExpertStore staging ring never refills a slot under an in-flight DMA",
          "[expert_store][stage]")
{
    // slots = 1 means the second EnqueueStage must reuse the only slot, and it
    // may only do so once the first DMA has executed. Without that check the
    // pinned buffer is overwritten while the copy still reads it, and the FIRST
    // expert's bytes arrive as the second expert's — the worst failure this
    // store can have, so it gets its own case.
    //
    // This case needs experts whose payloads DIFFER: the default fixture writes
    // the same pattern into every expert, so a corrupted copy would look
    // identical and the test would pass on broken code. See the fixture change
    // in step 3.
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir(/*distinct=*/true).string(), false));
    s.stage_ring.slots = 1;
    REQUIRE(s.InitStageRing());

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const int64_t packed = s.entries[0].w1_bytes;
    const int64_t scaled = s.entries[0].w1_scales_bytes;
    const int64_t off[6] = {0, 2 * (packed + scaled), (packed + scaled), packed,
                            2 * (packed + scaled) + packed,
                            (packed + scaled) + packed};
    const size_t total = static_cast<size_t>(3 * (packed + scaled));
    // Two destination buffers: the point is which bytes land in the FIRST one.
    uint8_t *dev0 = nullptr, *dev1 = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&dev0), total) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void **>(&dev1), total) == cudaSuccess);

    // Flat 0 = (layer 0, expert 0) with w1 = 0x10; flat 1 = (layer 0, expert 1)
    // with w1 = 0x11 (the `distinct` fixture).
    REQUIRE(s.EnqueueStage(0, dev0, off, stream));
    REQUIRE(s.EnqueueStage(1, dev1, off, stream));  // reuses the only slot
    s.AwaitStage(0, stream);
    s.AwaitStage(1, stream);

    std::vector<uint8_t> got0(total), got1(total);
    REQUIRE(cudaMemcpy(got0.data(), dev0, total, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE(cudaMemcpy(got1.data(), dev1, total, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    CHECK(got0[static_cast<size_t>(off[0])] == 0x10);   // expert 0's own bytes
    CHECK(got1[static_cast<size_t>(off[0])] == 0x11);   // expert 1's
    CHECK(s.stage_stats.ring_waits > 0);                // the slot was recycled

    cudaFree(dev0);
    cudaFree(dev1);
    cudaStreamDestroy(stream);
}

TEST_CASE("ExpertStore staging ring is a no-op when disabled",
          "[expert_store][stage]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 4;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    s.stage_ring.slots = 0;                       // default
    const int64_t off[6] = {0, 0, 0, 0, 0, 0};
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    CHECK_FALSE(s.EnqueueStage(0, nullptr, off, stream));
    CHECK(s.stage_stats.staged_experts == 0);
    CHECK(s.stage_stats.legacy_fallbacks == 1);
    cudaStreamDestroy(stream);
}

TEST_CASE("ExpertStore staging ring refuses a non-resident expert",
          "[expert_store][stage]")
{
    // Hazards 3 in the plan's Review Focus: an evicted expert has no locators.
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 1;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    s.stage_ring.slots = 2;
    REQUIRE(s.InitStageRing());
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    const int64_t off[6] = {0, 0, 0, 0, 0, 0};
    // flat=1 is not resident (max_open 1 keeps flat=0), so the ring must
    // decline rather than read a stale locator.
    CHECK_FALSE(s.EnqueueStage(1, nullptr, off, stream));
    cudaStreamDestroy(stream);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cmake --build --preset linux-debug --target test_expert_store && ./build/linux-debug/tests/test_expert_store "[stage]"`
Expected: FAIL to compile — `InitStageRing`, `stage_ring`, `stage_stats` are not members.

- [ ] **Step 3: Extend the fixture so a corrupt copy is observable**

Change `MakeExpertDir()` in `tests/test_expert_store.cpp` to
`MakeExpertDir(bool distinct = false)`. With `distinct = true` the packed
planes are still per-plane constants, but they are offset by the expert index:
`w1 = 0x10 + exp`, `w2 = 0x20 + exp`, `w3 = 0x30 + exp`, and the scales stay
`0x80/0x81/0x82`. The default (`false`) keeps today's bytes exactly, because
Task 2's locator test asserts `0x01/0x02/0x03` for every expert.

This step exists because the first draft of the hazard test staged two experts
with *identical* payloads — a slot refilled mid-flight produced byte-identical
output, so the test passed against broken code. A hazard test that cannot fail
is worse than no test, since it buys false confidence.

- [ ] **Step 4: Implement the ring**

```cpp
bool InitStageRing();   // allocates slots/pool; false + log::error on failure
```
- `max_expert_bytes` = the largest sum of the six `plane_bytes`. `slots` pinned buffers of that size via `cudaHostAlloc`, each with a `cudaEventDisableTiming` event created once at init (not per use).
- A `lsxcommon::io::PreadPool` sized `pread_threads`.
- If `cudaHostAlloc` fails, `log::error` with the pinning sentence and the `ulimit -l` fix, return false. This failure mode otherwise presents as "every later CUDA call fails".

`EnqueueStage`:
1. `slots == 0 || !entries[idx].host_w1` → `++stage_stats.legacy_fallbacks`, return false.
2. Pick a slot: free, else the oldest-slot whose event has completed (`cudaEventQuery` on that slot only — never scan a list; a full scan cost a measured 6.6% of a 2048-token mistral-small-4 run). If none is ready, `cudaStreamWaitEvent(stream, ev)` on the oldest and `++stage_stats.ring_waits` — a stream wait, never a host sync.
3. Build six `PreadJob`s from the locators into the slot, `pool.Run(jobs)`; on a short read `++stage_stats.short_reads` and return false (caller falls back — never DMA a half-filled slot).
4. Six `cudaMemcpyAsync` pinned→`dst_dev + dst_off[k]`, then `cudaEventRecord(slot.ev, stream)`.
5. Record `slot_idx` per flat index so `AwaitStage` can find it (`std::vector<int> stage_slot` sized with `entries`).
6. `++stage_stats.staged_experts; += staged_bytes`.

`AwaitStage(idx, stream)`: `cudaStreamWaitEvent(stream, slot.ev, 0)`; no-op for an index never staged.

`Free()`: destroy events, `cudaFreeHost` every slot, destroy the pool, clear `stage_slot`. Must be idempotent and safe when the ring was never initialised.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cmake --build --preset linux-debug --target test_expert_store && ./build/linux-debug/tests/test_expert_store`
Expected: all cases pass, 7 in total (3 pre-existing + 4 from this task).

- [ ] **Step 6: Prove the hazard test actually catches the defect**

Temporarily remove the slot-ready check (step 2 of `EnqueueStage` — pick any free-or-oldest slot regardless of its event) and re-run `"[stage]"`.
Expected: the in-flight-DMA case FAILS with a wrong byte. Restore the check and re-run: it passes. **Do not commit the doctored version**; this step exists because a hazard test that passes immediately proves nothing.

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/expert_store.h tests/test_expert_store.cpp
git commit -m "feat: pinned expert staging ring with per-slot DMA lifetime"
```

---

### Task 4: Staging bandwidth budget

**Files:**
- Create: `tests/test_expert_store_stage.cu`
- Modify: `tests/CMakeLists.txt` (`lsx_add_cuda_test(NAME test_expert_store_stage ...)`, label `fast`)
- Modify: `tests/perf/kernel_budgets.json` — **only** through `LSX_KERNEL_BUDGET_WRITE=1`, never by hand

**Interfaces:**
- Consumes: `ExpertStore::EnqueueStage`/`AwaitStage` (Task 3).
- Produces: budget entry `expert_store.stage_ring_gbps`, `gpu_key` resolved by `kernel_budget::CurrentGpuKey()`.

- [ ] **Step 1: Write the failing test**

`tests/test_expert_store_stage.cu`, following `tests/test_kernel_budget.h`'s conventions (best-of-5, `NDEBUG`-gated, `kernel_budget::Require` with `Kind::kMin`). Build a synthetic store large enough to measure (see step 3), stage every expert through the ring, and compute `bytes / elapsed`.

Expected number on this host from the probe: ~54 GB/s pinned DMA. Budget floor: record what is measured (the helper applies the 0.8x floor itself), do not author a number.

Note in the test's comment, in the test file, what this instrument is and is not: it measures the **pinned DMA + pread from page cache** leg. The cold-NVMe dimension is the E2E perf row (Task 7). Saying so prevents the next reader from treating this number as the model's throughput ceiling.

- [ ] **Step 2: Run it to watch it fail**

Run: `cmake --build --preset linux-debug --target test_expert_store_stage && ./build/linux-debug/tests/test_expert_store_stage`
Expected: FAIL/SKIP — no budget record for this `gpu_key`, in Debug it SKIPs by design (`SKIP_RETURN_CODE 4`). A Release build without `LSX_KERNEL_BUDGET_WRITE` must SKIP, not pass.

- [ ] **Step 3: Record the budget**

The fixture must be big enough that the measurement is not launch-bound: ~64 experts × 6 planes, with plane sizes matching the real schema ratio (packed : scales = 8 : 1). Write them once into a temp dir.

```bash
cmake --build --preset linux-release-znver4-aocc --target test_expert_store_stage
LSX_KERNEL_BUDGET_WRITE=1 ./build/linux-release-znver4-aocc/tests/test_expert_store_stage
git diff tests/perf/kernel_budgets.json    # review the entry deliberately
```
Expected: one new budget object with `kind: "min"`, a real `observed_min`/`observed_max`, this host's `gpu_key`, and today's date.

- [ ] **Step 4: Run it to verify it passes**

Run: `./build/linux-release-znver4-aocc/tests/test_expert_store_stage`
Expected: PASS against the recorded floor.

- [ ] **Step 5: Commit**

```bash
git add tests/test_expert_store_stage.cu tests/CMakeLists.txt tests/perf/kernel_budgets.json
git commit -m "test: bandwidth floor for the pinned expert staging ring"
```

---

### Task 5: Env knobs and the decode-end engagement line

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/expert_store.h` (`ResolveStageRingFromEnv()` + the log line)
- Modify: `tests/test_expert_store.cpp`

**Interfaces:**
- Consumes: `stage_ring`, `stage_stats` (Task 3).
- Produces:
  ```cpp
  // Reads LSX_EXPERT_STAGE_RING / _SLOTS / _THREADS / _CHUNK into stage_ring.
  // An unset or "0" ring leaves slots = 0 (legacy path). Called by InitStageRing.
  void ResolveStageRingFromEnv();
  // One log::info line: staged experts/bytes, fallbacks, ring waits, short reads.
  void LogStageStats() const;
  ```

- [ ] **Step 1: Write the failing test**

```cpp
TEST_CASE("ExpertStore stage-ring knobs resolve from the environment",
          "[expert_store][stage]")
{
    Store s;
    ::setenv("LSX_EXPERT_STAGE_RING", "1", 1);
    ::setenv("LSX_EXPERT_STAGE_SLOTS", "3", 1);
    ::setenv("LSX_EXPERT_STAGE_THREADS", "0", 1);   // 0 => default, not 0 threads
    s.ResolveStageRingFromEnv();
    CHECK(s.stage_ring.slots == 3);
    CHECK(s.stage_ring.pread_threads == 4);         // documented default
    ::unsetenv("LSX_EXPERT_STAGE_RING");
    ::unsetenv("LSX_EXPERT_STAGE_SLOTS");
    ::unsetenv("LSX_EXPERT_STAGE_THREADS");
    s.stage_ring = Store::StageRingConfig{};
    s.ResolveStageRingFromEnv();
    CHECK(s.stage_ring.slots == 0);                 // default is OFF
}
```

- [ ] **Step 2: Run to verify it fails** — `ResolveStageRingFromEnv` is not a member.

- [ ] **Step 3: Implement** the env reader (unset / `"0"` ring ⇒ `slots = 0`; `threads <= 0` ⇒ 4; `chunk <= 0` ⇒ 1) and `LogStageStats` at `log::info` with the `[expert_store]` prefix, only when `staged_experts > 0` — a family that never engaged must not print a line that looks like it did.

- [ ] **Step 4: Run to verify it passes**, then `ctest --preset linux-debug -L fast`.
Expected: `100% tests passed`.

- [ ] **Step 5: Commit**
```bash
git commit -am "feat: env knobs and engagement counters for the staging ring"
```

---

### Task 6: Wire `kimi-k3.1-pro` (opt-in, the only family in this plan)

**Files:**
- Modify: `libs/lsxcommon/src/kimi_moe_k3_inference_pipeline.cu:638-668` (the `for (int k = 0; k < top.top_k; ++k)` loop)
- Modify: `libs/lsxcommon/src/kimi_moe_k3_load.cpp` (resolve `stage_ring` at load)
- Modify: `libs/lsxcommon/include/lsxcommon/kimi_moe_k3_load.h` (expose the resolved config, if the pipeline cannot see the store's field)

**Interfaces:**
- Consumes: `EnqueueStage`/`AwaitStage` (Task 3), knobs (Task 5).
- Produces: nothing other families use.

- [ ] **Step 1: Pin the numeric contract before touching the loop**

Run the existing gate first, so a later failure is attributable:
```bash
LSX_DET_STRESS_RUNS=3 ./build/linux-debug/tests/test_kimi_pro_selfcheck
```
Expected: PASS, `logits_hash 98ec1bbb372c9181`, top-1 40 (the recorded values in AGENTS.md). Record the wall time — it is the "before" for Task 7 and it is cheap (Paris is a 1-token request).

- [ ] **Step 2: Replace the copy loop with the ring**

`kimi-k3.1-pro`'s device layout at `ws.expert_stage` is `[w1 | w1s | w3 | w3s | w2 | w2s]` with `one = packed_w + scales_w`, i.e. **not** schema order. Build the `dst_off` array to match it:
```cpp
const int64_t dst_off[6] = {
    0,                      // w1
    2 * one,                // w2
    one,                    // w3
    packed_w,               // w1_scales
    2 * one + packed_w,     // w2_scales
    one + packed_w};        // w3_scales
```
Then, per expert: `if (experts.EnqueueStage(pe_idx, ws.expert_stage, dst_off, stream)) { experts.AwaitStage(pe_idx, stream); } else { <the existing six cudaMemcpyAsync, unchanged> }`.

Keep the existing `if (!pe.host_w1 || ...) return fail("routed expert view is not resident");` **before** the staging call: the ring declines a non-resident entry, and a family that treats "declined" as "staged nothing" would compute on stale bytes.

Do not change the three GEMV calls or their order.

- [ ] **Step 3: Verify the numeric contract is unchanged**

```bash
cmake --build --preset linux-debug --target test_kimi_pro_selfcheck
# ring engaged:
LSX_EXPERT_STAGE_RING=1 LSX_EXPERT_STAGE_SLOTS=64 ./build/linux-debug/tests/test_kimi_pro_selfcheck
# ring off (must equal the pre-change baseline exactly):
./build/linux-debug/tests/test_kimi_pro_selfcheck
```
Expected: both PASS with `logits_hash 98ec1bbb372c9181` and top-1 40; the run with `LSX_EXPERT_STAGE_RING=1` logs `staged_experts > 0` (proving engagement, not a silent fallback).

- [ ] **Step 4: Confirm every other family is untouched**

```bash
ctest --preset linux-debug -L fast
ctest --preset linux-debug -R "selfcheck" --output-on-failure
```
Expected: `fast` 100%; selfchecks for families present in the vault PASS with the flag unset. If a model tree is absent the target SKIPs (exit 4) — that is a gap, record it, do not count it as a pass.

- [ ] **Step 5: Commit**
```bash
git commit -am "perf: stage kimi-k3.1-pro routed experts through the pinned ring"
```

---

### Task 7: Measure at the canonical protocol and sweep the three parameters

**Files:**
- Modify: `docs/superpowers/specs/2026-11-05-expert-stage-ring-design.md` (measured outcome table)
- Modify: `docs/OPEN_TASKS.md` (§R17 P6)
- No code change expected; if a sweep says the design is wrong, that is a finding, not a licence to expand the plan.

**Interfaces:** none.

- [ ] **Step 1: Baseline at 2048 tokens, ring off**

```bash
ulimit -l unlimited
./build/linux-release-znver4-aocc/apps/lsxbenchmark/lsxbenchmark \
  --model /mnt/fast_storage/huggingface/Kimi-K3.1-Pro --bench \
  --max-gen 2048 --prompt "The capital of France is"
```
Expected: ~0.30–0.31 t/s (3.2–3.35 s/token), reproducing the recorded figure. **GPU exclusive**; budget 1.8 h.

- [ ] **Step 2: Ring on, same protocol, twice**

Same command with `LSX_EXPERT_STAGE_RING=1 LSX_EXPERT_STAGE_SLOTS=64`. Two runs; agreement within ~1% is the reproducibility evidence. Expected from the probe: staging drops from 5.05 to ~2.5 s/token of copies, so the model should land around **0.55–0.65 t/s**; a result near 0.31 means the ring did not engage — check `staged_experts` before concluding anything about speed.

- [ ] **Step 3: Sweep `pread_threads` (4 / 8 / 16 / 32) at 512 tokens**

Short protocol for the sweep only; the ranked candidates get the 2048 confirmation in step 4. The array does 39.8 GB/s at 32 cold reads; 4 threads measured 35 GB/s, so the knee is expected early and a flat sweep is a normal outcome for a host-bound term.

- [ ] **Step 4: Confirm the best configuration at 2048, twice**

- [ ] **Step 5: Record**

- Update the design doc with a measured-outcome table (arm → t/s → GB/s), including any arm that was worse.
- Rewrite `docs/OPEN_TASKS.md` §R17 P6: the old sentence ("overlap layer N+1 with layer N") is impossible — expert ids are a function of the previous layer's output. Replace it with what landed and what remains (thread count, `slots`, `O_DIRECT`).
- If the number beats the recorded perf row, seed it through `test_perf_baseline` (**never** by hand):
  ```bash
  LSX_PERF_VAULT_SUBDIR=Kimi-K3.1-Pro ./build/linux-release-znver4-aocc/tests/test_perf_baseline "perf baseline: kimi-k3.1-pro"
  ```
  This writes `tests/perf/perf_baselines.json`; review the diff deliberately.

- [ ] **Step 6: Commit**
```bash
git add docs/ tests/perf/perf_baselines.json
git commit -m "perf: measured outcome of the pinned expert staging ring on kimi-k3.1-pro"
```

---

### Task 8: Documentation and the family-wide rollout record

**Files:**
- Modify: `AGENTS.md` (`expert_store.h` bullet: the ring, the knobs, the fact that P6-as-written was impossible)
- Modify: `.opencode/skills/perf-tuning-a-model/SKILL.md` (the "streamed-expert families" lesson: `pread`-into-pinned beats pageable copies, and batching is a measured loss)
- Modify: `docs/OPEN_TASKS.md` (a follow-up item for the other streaming families, with the measured per-family expectation left open)

**Interfaces:** none.

- [ ] **Step 1: Update the three documents** with the measured numbers from Task 7 — not estimates. Anything not measured says so.
- [ ] **Step 2: Verify no doc claim is unattributed**

```bash
grep -n "EXPLICITLY-UNMEASURED" docs/OPEN_TASKS.md AGENTS.md | head
```
Expected: no number in the new text that Task 7 did not produce. A claim like "the other families will improve similarly" must be written as expected-not-measured, because a family whose disk reads are already cached, or whose entries are much larger, will measure differently.

- [ ] **Step 3: Run the doc-link check and the guards**

```bash
ctest --preset linux-debug
```
Expected: `100% tests passed`.

- [ ] **Step 4: Commit**
```bash
git commit -am "docs: the expert staging ring — measured, knobs, and the remaining rollout"
```

---

## Self-review notes

Checked against the spec section by section: the ring, the plane descriptors, the hazard rules, the knobs, the rollout order, the non-goals (no GPUDirect — absent from this host; no `cuMemcpyBatchAsync` — measured a loss; no next-layer prefetch — impossible) each map to a task. The spec's "what is left to the builder" list (threads, slots/chunk, `O_DIRECT`) is Task 7's sweep plus an explicit decision to leave `O_DIRECT` unshipped until measured.

Two deliberate omissions, stated so a reviewer can reject them: `chunk_experts` is carried in the config but no task gives it a distinct pipeline shape — Task 3 stages one expert per `EnqueueStage`, and the pipeline depth comes from `slots`, so a separate chunk knob would be an unmotivated parameter. If Task 7's sweep shows a real effect, it earns a task then. Likewise the engagement counters have a test but no assertion that a *specific family* engaged; Task 6 step 3's log line is the proof, and pinning it in a unit test would require the vault.
