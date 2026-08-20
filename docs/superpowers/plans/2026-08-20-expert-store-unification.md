# MoE Expert Store Unification — Implementation Plan (§3.2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the two byte-identical MoE mmap-expert LRU loaders (DeepSeek `gpu::ExpertDb`, GLM `glm::GlmExpertDb`) into one shared header-only `ExpertStore<T>` template, preserving exact runtime behavior and every existing call-site name.

**Architecture:** A new header `include/lsxcommon/expert_store.h` owns the storage vectors, the flat-index path builder, deferred-parse + warm-up, LRU eviction (`EnsureResident`) and `Free`. Because `ExpertEntry` and `GlmExpertEntry` expose identical field names in the same order and both read the same 9-column `ExpertSchema`, a single `DecodeEntry()` member serves both. The only per-model differences: three store fields (`log_prefix`, `emit_diag`, `log_batch_fails`) plus the `fallback` path flag. Thin per-model wrappers preserve the existing exported names.

**Tech Stack:** C++20, Arrow (`MemoryMappedFile`, `RecordBatchFileReader`), CUDA `cudaStream_t`/`cudaMemcpyAsync` (host-side only — no kernels, no absl), CMake Presets, Catch2 v3.7.0. Real-model tests SKIP via `SKIP_RETURN_CODE 4`.

---

## File map

- Create: `libs/lsxcommon/include/lsxcommon/expert_store.h`
- Create: `tests/test_expert_store.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cuh`
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cu`
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference_load.h`
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp`
- No CMake source-list change (expert_store.h is header-only under `include/`, already on every consumer's include path).

---

## Design constraints (read first)

1. **No behavior change.** The template reproduces the current warm-up order (first `max_open` flat indices), the touch/evict LRU, and the pre-eviction `cudaStreamSynchronize` barrier exactly. DeepSeek/GLM tests are the regression net.
2. **Same schema.** Both use the identical 9-column `ExpertSchema` (`schema.h:84-105`). `ExpertEntry` (`deepseek_inference_load.cuh:40-56`) and `GlmExpertEntry` (`glm_inference_load.h:234-250`) are field-for-field identical (same 15 members, same order).
3. **Log parity.** Preserve today's exact text:
   - mmap-fail (rate-limited to `idx < 5 || idx % 1000 == 0`): DeepSeek `[load_expert_index] mmap fail <path>: <status>`; GLM `[glm_load] expert mmap fail <path>: <status>`.
   - GLM-only per-batch failure (unconditional): `[glm_load] expert file has no record batches: <path>`; `[glm_load] expert batch read failed for <path>: <status>`. DeepSeek logs nothing on these.
   - DeepSeek-only `[diag-expert]` info block for `idx == 0 || idx == 1 || idx == experts_per_layer`. GLM never emits it.
4. **Standard library only.** No `absl` in `expert_store.h` — it is reachable from NVCC `.cu` TUs (DeepSeek `.cuh`) and MSVC host TUs (GLM public header). `log::debug`/`log::info` come from `lsxcommon/log.h` (host bridge, safe for both).

---

## Task 1: Create `include/lsxcommon/expert_store.h`

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/expert_store.h`

- [ ] **Step 1: Write the template header**

```cpp
// expert_store.h — shared MoE mmap-expert LRU store (kernel-unification §3.2).
//
// Collapses the two byte-identical expert loaders (DeepSeek `gpu::ExpertDb`,
// GLM `gpu::GlmExpertDb`) into one template, keyed on the per-model entry
// type. Both entries share the same 9-column ExpertSchema and identical member
// names in the same order, so a single DecodeEntry() serves both.
//
// Host-side only: no __device__ kernels, no absl. Safe under both the MSVC
// host compiler and NVCC .cu TUs. LRU is NOT internally synchronized (same
// contract as the loaders it replaces: one consumer per instance).
#pragma once

#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/record_batch.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/log.h"
#include "lsxcommon/schema.h"

namespace lsxcommon {
namespace gpu {
namespace expert_store {

template <typename TEntry>
struct ExpertStore {
    std::vector<TEntry> entries;
    std::vector<std::shared_ptr<arrow::io::MemoryMappedFile>> mmaps;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::vector<std::string> paths;      // per flat index (for lazy open)
    std::vector<size_t> lru;             // flat indices, back = least recently used
    size_t max_open = 7000;              // bounded by Windows CRT fd cap (8192)
    int num_layers = 0;
    int experts_per_layer = 0;
    std::string error_msg;

    // Model toggles (see top-of-file).
    std::string log_prefix = "";
    std::string mmap_fail_verb = "mmap fail ";   // DeepSeek; GLM = "expert mmap fail "
    bool emit_diag = false;       // DeepSeek-only [diag-expert] block
    bool log_batch_fails = false; // GLM-only per-batch failure debug lines

    bool Load(const std::string &model_path, bool fallback)
    {
        namespace fs = std::filesystem;
        int total = num_layers * experts_per_layer;
        if (total <= 0) return false;
        entries.assign(static_cast<size_t>(total), TEntry());
        mmaps.assign(static_cast<size_t>(total), nullptr);
        batches.assign(static_cast<size_t>(total), nullptr);
        paths.resize(static_cast<size_t>(total));
        lru.reserve(max_open);

        fs::path exp_dir = fs::path(model_path) / "logestix" / "experts";
        if (fallback && !fs::exists(exp_dir) &&
            fs::exists(fs::path(model_path) / "experts"))
            exp_dir = fs::path(model_path) / "experts";

        for (int layer = 0; layer < num_layers; ++layer) {
            for (int exp = 0; exp < experts_per_layer; ++exp) {
                size_t idx = static_cast<size_t>(layer) *
                                 static_cast<size_t>(experts_per_layer) +
                             static_cast<size_t>(exp);
                paths[idx] = (exp_dir / ("layer_" + std::to_string(layer)) /
                              ("exp_" + std::to_string(exp) + ".arrow")).string();
            }
        }

        // Warm the cache up to max_open (matches the per-model loop order).
        for (size_t idx = 0; idx < paths.size() && lru.size() < max_open; ++idx) {
            if (ParseFile(idx)) lru.push_back(idx);
        }
        return true;
    }

    // Decodes the shared 6-bin + 3-float ExpertSchema fields into @p out.
    static void DecodeEntry(const std::shared_ptr<arrow::RecordBatch> &batch,
                            TEntry &out)
    {
        auto get_bin = [&](int col) -> std::pair<const uint8_t *, int64_t> {
            auto arr = batch->column(col);
            auto bin = std::dynamic_pointer_cast<arrow::LargeBinaryArray>(arr);
            if (!bin) return {nullptr, 0};
            auto view = bin->GetView(0);
            return {reinterpret_cast<const uint8_t *>(view.data()),
                    static_cast<int64_t>(view.size())};
        };
        auto get_scale_bin = [&](int col) -> std::pair<const uint8_t *, int64_t> {
            auto lst = std::dynamic_pointer_cast<arrow::ListArray>(batch->column(col));
            if (!lst) return {nullptr, 0};
            auto u8 = std::dynamic_pointer_cast<arrow::UInt8Array>(lst->values());
            if (!u8) return {nullptr, 0};
            auto offs = std::dynamic_pointer_cast<arrow::Int32Array>(lst->offsets());
            if (!offs) return {nullptr, 0};
            int64_t begin = offs->Value(0);
            int64_t end   = offs->Value(1);
            return {u8->raw_values() + begin, end - begin};
        };
        auto get_gs = [&](int col) -> float {
            auto arr = batch->column(col);
            auto f = std::dynamic_pointer_cast<arrow::FloatArray>(arr);
            return f ? f->Value(0) : 1.0f;
        };

        auto [w1p, w1s] = get_bin(lsxcommon::ExpertSchema::kIdxW1Weight);
        auto [w2p, w2s] = get_bin(lsxcommon::ExpertSchema::kIdxW2Weight);
        auto [w3p, w3s] = get_bin(lsxcommon::ExpertSchema::kIdxW3Weight);
        auto [w1sp, w1ss] = get_scale_bin(lsxcommon::ExpertSchema::kIdxW1Scale);
        auto [w2sp, w2ss] = get_scale_bin(lsxcommon::ExpertSchema::kIdxW2Scale);
        auto [w3sp, w3ss] = get_scale_bin(lsxcommon::ExpertSchema::kIdxW3Scale);

        out = {w1p, w2p, w3p, w1sp, w2sp, w3sp,
               get_gs(lsxcommon::ExpertSchema::kIdxW1GlobalScale),
               get_gs(lsxcommon::ExpertSchema::kIdxW2GlobalScale),
               get_gs(lsxcommon::ExpertSchema::kIdxW3GlobalScale),
               w1s, w2s, w3s, w1ss, w2ss, w3ss};
    }

    bool ParseFile(size_t idx)
    {
        auto &apath = paths[idx];
        auto mmap_res = arrow::io::MemoryMappedFile::Open(
            apath, arrow::io::FileMode::READ);
        if (!mmap_res.ok()) {
            if (idx < 5 || idx % 1000 == 0)
                log::debug(log_prefix + mmap_fail_verb + apath + ": "
                           + mmap_res.status().ToString());
            return false;
        }
        auto mmap = *mmap_res;

        auto reader_res = arrow::ipc::RecordBatchFileReader::Open(mmap.get());
        if (!reader_res.ok() || (*reader_res)->num_record_batches() == 0) {
            if (log_batch_fails)
                log::debug(log_prefix + "expert file has no record batches: " + apath);
            return false;
        }
        auto batch_res = (*reader_res)->ReadRecordBatch(0);
        if (!batch_res.ok()) {
            if (log_batch_fails)
                log::debug(log_prefix + "expert batch read failed for " + apath +
                           ": " + batch_res.status().ToString());
            return false;
        }
        auto batch = *batch_res;

        DecodeEntry(batch, entries[idx]);

        if (emit_diag && (idx == 0 || idx == 1 ||
                          idx == static_cast<size_t>(experts_per_layer))) {
            const TEntry &e = entries[idx];
            auto fmt_scale_bytes = [](const uint8_t *p, int64_t n) -> std::string {
                std::ostringstream os; os << "[";
                for (int64_t i = 0; i < n && i < 8; ++i) {
                    if (i) os << ",";
                    char b[4]; std::snprintf(b, sizeof(b), "%02x", p ? p[i] : 0);
                    os << b;
                }
                return os.str() + "]";
            };
            auto implied_blk = [](int64_t packed, int64_t scl) -> std::string {
                if (scl <= 0) return "n/a (0 scales)";
                return std::to_string((packed * 2) / scl) + " elems/scale (numel=" +
                       std::to_string(packed * 2) + " scales=" + std::to_string(scl) + ")";
            };
            std::ostringstream os;
            os << "[diag-expert] idx=" << idx
               << " w1: packed=" << e.w1_bytes << "B scales=" << e.w1_scales_bytes
               << "B gs=" << e.w1_global_scale
               << " blk=" << implied_blk(e.w1_bytes, e.w1_scales_bytes)
               << " s1[0..7]=" << fmt_scale_bytes(e.host_w1_scales, e.w1_scales_bytes)
               << " | w2: packed=" << e.w2_bytes << "B scales=" << e.w2_scales_bytes
               << "B gs=" << e.w2_global_scale
               << " blk=" << implied_blk(e.w2_bytes, e.w2_scales_bytes)
               << " | w3: packed=" << e.w3_bytes << "B scales=" << e.w3_scales_bytes
               << "B gs=" << e.w3_global_scale
               << " blk=" << implied_blk(e.w3_bytes, e.w3_scales_bytes);
            log::info(os.str());
        }

        mmaps[idx] = std::move(mmap);
        batches[idx] = std::move(batch);
        return true;
    }

    // Touch a resident entry or parse it, evicting the LRU tail under max_open.
    // The caller's stream is synchronized ONCE, only before the first eviction
    // (pre-munmap barrier for in-flight cudaMemcpyAsync reading host mmap).
    void EnsureResident(int layer, int expert, cudaStream_t stream)
    {
        if (expert < 0 || expert >= experts_per_layer) return;
        if (paths.empty()) return;
        size_t idx = static_cast<size_t>(layer) *
                         static_cast<size_t>(experts_per_layer) +
                     static_cast<size_t>(expert);
        if (idx >= entries.size()) return;

        if (entries[idx].host_w1) {
            for (auto it = lru.begin(); it != lru.end(); ++it) {
                if (*it == idx) { lru.erase(it); lru.insert(lru.begin(), idx); break; }
            }
            return;
        }

        if (lru.size() >= max_open)
            cudaStreamSynchronize(stream);
        while (lru.size() >= max_open) {
            size_t evict = lru.back();
            lru.pop_back();
            mmaps[evict].reset();
            batches[evict].reset();
            entries[evict] = TEntry();
        }
        if (ParseFile(idx))
            lru.insert(lru.begin(), idx);
    }

    void Free()
    {
        for (auto &m : mmaps) m.reset();
        for (auto &b : batches) b.reset();
        mmaps.clear(); batches.clear(); lru.clear(); entries.clear(); paths.clear();
        error_msg.clear();
        num_layers = 0; experts_per_layer = 0;
    }
};

} // namespace expert_store
} // namespace gpu
} // namespace lsxcommon
```

- [ ] **Step 2: Reconfigure the unchanged build**

Run: `cmake --preset windows-debug`
Expected: configure succeeds (the header is not yet referenced; no new errors).

- [ ] **Step 3: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/expert_store.h
git commit -m "feat: shared MoE expert LRU store template (expert_store.h)"
```

---

## Task 2: Add the deterministic store unit test

**Files:**
- Create: `tests/test_expert_store.cpp`
- Modify: `tests/CMakeLists.txt`

Host-only Catch2 test. It builds a synthetic `ExpertSchema`-encoded expert dir, loads `ExpertStore<FixtureEntry>`, and asserts warm-up + LRU eviction ordering — no real model needed.

- [ ] **Step 1: Write the failing test**

```cpp
// ExpertStore template behavior on a synthetic expert dir (no checkpoint).
#include <catch2/catch_test_macros.hpp>

#include <arrow/api.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/schema.h"

namespace {

struct FixtureEntry {
    const uint8_t *host_w1 = nullptr, *host_w2 = nullptr, *host_w3 = nullptr;
    const uint8_t *host_w1_scales = nullptr, *host_w2_scales = nullptr, *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f, w2_global_scale = 1.0f, w3_global_scale = 1.0f;
    int64_t w1_bytes = 0, w2_bytes = 0, w3_bytes = 0;
    int64_t w1_scales_bytes = 0, w2_scales_bytes = 0, w3_scales_bytes = 0;
};

using Store = lsxcommon::gpu::expert_store::ExpertStore<FixtureEntry>;

std::filesystem::path MakeExpertDir()
{
    std::filesystem::path container =
        std::filesystem::temp_directory_path() /
        ("lsx_expert_test_" + std::to_string(std::rand()));
    const std::filesystem::path exps = container / "logestix" / "experts";
    std::filesystem::create_directories(exps / "layer_0");
    std::filesystem::create_directories(exps / "layer_1");

    lsxcommon::PackedExpert pe;
    pe.w1_packed.assign(16, 0x01u);
    pe.w2_packed.assign(16, 0x02u);
    pe.w3_packed.assign(16, 0x03u);
    pe.w1_scales.assign(2, 0x80u);
    pe.w2_scales.assign(2, 0x81u);
    pe.w3_scales.assign(2, 0x82u);
    pe.w1_global_scale = 1.0f;
    pe.w2_global_scale = 2.0f;
    pe.w3_global_scale = 3.0f;
    auto batch = lsxcommon::ExpertSchema::MakeBatch(pe);
    REQUIRE(batch.ok());
    for (int layer = 0; layer < 2; ++layer)
        for (int exp = 0; exp < 2; ++exp)
            REQUIRE(lsxcommon::WriteRecordBatchToFile(
                        *batch, exps / ("layer_" + std::to_string(layer)) /
                                    ("exp_" + std::to_string(exp) + ".arrow"))
                        .ok());
    return container;
}

}  // namespace

TEST_CASE("ExpertStore warms up to max_open on Load", "[expert_store]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 2;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    REQUIRE(s.entries.size() == 4);
    REQUIRE(s.lru.size() == 2);                 // flat [0] and [1]
    REQUIRE(s.entries[0].host_w1 != nullptr);       // decoded
}

TEST_CASE("ExpertStore::EnsureResident evicts under max_open", "[expert_store]")
{
    Store s;
    s.num_layers = 2;
    s.experts_per_layer = 2;
    s.max_open = 1;
    REQUIRE(s.Load(MakeExpertDir().string(), false));
    REQUIRE(s.lru.size() == 1);

    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);

    s.EnsureResident(0, 1, stream);   // flat=1 -> evicts flat=0
    CHECK(s.lru.front() == 1);
    s.EnsureResident(1, 0, stream);   // flat=2 -> evicts flat=1
    CHECK(s.lru.front() == 2);
    s.EnsureResident(0, 0, stream);   // re-parse flat=0 -> evicts flat=2
    CHECK(s.lru.front() == 0);

    cudaStreamDestroy(stream);
}
```

Determinism note: use `std::rand()` with `#include <cstdlib>` (per-file seed is fine for the test); each TEST_CASE calls `MakeFixtureDir()` separately so the two stores never collide.

- [ ] **Step 2: Add the test target**

Edit `tests/CMakeLists.txt`, adding (place near `test_sampling`):

```cmake
add_executable(test_expert_store test_expert_store.cpp)
target_link_libraries(test_expert_store PRIVATE lsxcommon Catch2::Catch2WithMain)
if(WIN32)
    add_custom_command(TARGET test_expert_store POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_expert_store>")
    add_custom_command(TARGET test_expert_store POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_expert_store>")
endif()
add_test(NAME test_expert_store COMMAND test_expert_store)
set_tests_properties(test_expert_store PROPERTIES SKIP_RETURN_CODE 4)
```

`lsxcommon/schema.h` and `expert_store.h` already resolve via the `lsxcommon` include dir that `target_link_libraries(... lsxcommon)` propagates — no extra include dir needed.

- [ ] **Step 3: Configure to pick up the target**

Run: `cmake --preset windows-debug`
Expected: configure succeeds; the new test target appears.

- [ ] **Step 4: Verify it fails (RED)**

Run: `cmake --build --preset windows-debug --target test_expert_store 2>&1 | Select -Last 5`
Expected: FAIL — `ExpertSchema::MakeBatch` may need the schema include path, or the LRU assertion catches a store behavior gap (the point of the test).

- [ ] **Step 5: Commit**

```bash
git add tests/test_expert_store.cpp tests/CMakeLists.txt
git commit -m "test: deterministic ExpertStore LRU unit test"
```

---

## Task 3: Migrate DeepSeek (`gpu::ExpertDb`)

**Files:**
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cuh`
- Modify: `libs/lsxcommon/src/deepseek_inference_load.cu`

- [ ] **Step 1: Replace the DeepSeek struct with the alias**

Edit `libs/lsxcommon/src/deepseek_inference_load.cuh`:
- Add `#include "lsxcommon/expert_store.h"` after the existing includes (line ~15).
- Delete `struct ExpertDb { ... }` (`:80-90`).
- After `struct ExpertEntry`, inside `namespace lsxcommon { namespace gpu {`, add:

```cpp
using ExpertDb = lsxcommon::gpu::expert_store::ExpertStore<ExpertEntry>;
```

Keep the three forward declarations (`:92-110`) unchanged. `deepseek_inference_pipeline.cu` compiles unaffected.

- [ ] **Step 2: Reduce `deepseek_inference_load.cu` to wrappers**

Edit `libs/lsxcommon/src/deepseek_inference_load.cu`:
- Delete `parse_expert_file` (`:269-370`).
- Delete the three bodies: `load_expert_index` (`:372-404`), `ensure_expert_resident` (`:406-447`), `free_expert_index` (`:449-458`).
- Keep `load_topology`/`detect_model_type`.
- Add (still in `namespace lsxcommon { namespace gpu {`):

```cpp
ExpertDb load_expert_index(const std::string &model_path,
                           int num_layers, int experts_per_layer)
{
    ExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.max_open = 7000;
    db.log_prefix = "[load_expert_index] ";
    db.mmap_fail_verb = "mmap fail ";
    db.emit_diag = true;             // DeepSeek diag block
    db.log_batch_fails = false;      // DeepSeek logs no batch-failure lines
    db.Load(model_path, /*fallback=*/false);
    return db;
}

void ensure_expert_resident(ExpertDb &db, int layer, int expert,
                            cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void free_expert_index(ExpertDb &db) { db.Free(); }
```

- [ ] **Step 3: Build + run DeepSeek loader test**

Run: `cmake --build --preset windows-debug --target test_deepseek_loader 2>&1 | Select -Last 3`
Then: `tests/test_deepseek_loader.exe` → "All tests passed (64 assertions in 3 test cases)".

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/src/deepseek_inference_load.cuh libs/lsxcommon/src/deepseek_inference_load.cu
git commit -m "refactor: DeepSeek expert store migrates to shared ExpertStore<T>"
```

---

## Task 4: Migrate GLM (`glm::GlmExpertDb`)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/glm_inference_load.h`
- Modify: `libs/lsxcommon/src/glm_inference_load.cpp`

- [ ] **Step 1: Alias + remove struct**

Edit `libs/lsxcommon/include/lsxcommon/glm_inference_load.h`:
- Keep `struct GlmExpertEntry { ... }`.
- Delete `struct GlmExpertDb { ... }` (`:271-281`).
- Add `#include "lsxcommon/expert_store.h"` (GLM header is host-only; compatible with arrow record_batch already there).
- Inside `namespace lsxcommon { namespace glm {`, add:

```cpp
using GlmExpertDb = lsxcommon::gpu::expert_store::ExpertStore<GlmExpertEntry>;
```

Keep the three free-function declarations (`:288-306`) unchanged.

- [ ] **Step 2: Delete `ParseGlmExpertFile` + the three bodies + add wrappers**

Edit `libs/lsxcommon/src/glm_inference_load.cpp`:
- Delete `ParseGlmExpertFile` (`:491-564`).
- Delete the three bodies: `load_glm_expert_index` (`:568-602`), `ensure_glm_expert_resident` (`:604-650`), `free_glm_expert_index` (`:651-663`).
- **Keep `LoadGlmExpertTables` untouched** — GLM-specific device uploader, out of scope.
- Inside `namespace lsxcommon { namespace glm {`, add:

```cpp
GlmExpertDb load_glm_expert_index(const std::string &model_path,
                                  int num_layers, int experts_per_layer)
{
    GlmExpertDb db;
    db.num_layers = num_layers;
    db.experts_per_layer = experts_per_layer;
    db.log_prefix = "[glm_load] ";
    db.mmap_fail_verb = "expert mmap fail ";
    db.emit_diag = false;            // GLM emits no [diag-expert] block
    db.log_batch_fails = true;       // GLM logs batch-failure lines
    db.Load(model_path, /*fallback=*/true);
    return db;
}

void ensure_glm_expert_resident(GlmExpertDb &db, int layer, int expert,
                                cudaStream_t stream)
{
    db.EnsureResident(layer, expert, stream);
}

void free_glm_expert_index(GlmExpertDb &db) { db.Free(); }
```

Existing call sites (`glm_v5_model.cpp:52,56,93,172,175,192`; `glm_inference_load.cu:696,755`) compile unchanged via the alias.

- [ ] **Step 3: Build + run GLM model test**

Run: `cmake --build --preset windows-debug --target test_glm_model 2>&1 | Select -Last 3`
Then `tests/test_glm_model.exe` → "All tests passed (16 assertions in 2 test cases)". If a GLM checkpoint is present, run `tests/test_glm_selfcheck.exe` (bit-exact `final_logits`, `"Paris"`); else SKIP.

- [ ] **Step 4: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/glm_inference_load.h libs/lsxcommon/src/glm_inference_load.cpp
git commit -m "refactor: GLM expert store migrates to shared ExpertStore<T>"
```

---

## Task 5: Full-suite verification

- [ ] **Step 1: Grep for leftover duplicate store definitions**

```bash
rg -n "struct ExpertDb|struct GlmExpertDb|load_expert_index|load_glm_expert_index|ensure_expert_resident|ensure_glm_expert_resident|free_expert_index|free_glm_expert_index" libs/lsxcommon
```
Expected: only the `using` aliases and wrapper bodies; no `struct ExpertDb {...}` / `struct GlmExpertDb {...}` definitions.

- [ ] **Step 2: Full build**

Run: `cmake --preset windows-debug && cmake --build --preset windows-debug`
Expected: clean build (only pre-existing Arrow/GCC warnings).

- [ ] **Step 3: Run the affected + surrounding subsets**

```bash
ctest --preset windows-debug -R "expert_store|loader|model_runtime|model_factory|sampling"
```
plus `tests/test_expert_store.exe` → all cases PASS. If GLM/DeepSeek checkpoints exist, also run `test_glm_selfcheck` / `test_deepseek_selfcheck` / `test_qwen_selfcheck` to confirm zero drift.

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "test: verify expert store unification (LRU parity)"
```
(Only if Step 2/3 surfaced uncommitted changes.)

---

## Self-Review

Crosscheck against `docs/superpowers/specs/2026-08-20-expert-store-unification-design.md`:

- **§3.1 template shape:** Task 1 provides `ExpertStore<T>` with the required vectors, `max_open`, three toggles, `Load`, `EnsureResident`, `Free`. The spec §3.1 sketch's function-pointer `parse` slot is replaced by the generic member `DecodeEntry()` — functionally identical, DRYer, valid because both entry types share field names. ✓
- **§3.2 aliases:** Task 3 `using ExpertDb = ExpertStore<ExpertEntry>`; Task 4 `using GlmExpertDb = ExpertStore<GlmExpertEntry>`. Call sites unchanged. ✓
- **§3.3 ownership:** store owns the vectors; `TEntry` stays model-owned. ✓
- **§4 extensibility:** a new MoE model defines `TEntry` and instantiates the store wholesale. ✓
- **§5 error handling:** `Load` reproduces the rate-limited mmap-fail prefix and GLM batch-failure lines; parse failure leaves `host_w1 == nullptr` (inactive skip). ✓
- **§6 verification:** Task 5 covers the subsets, the store unit test (Task 2), and the duplicate grep. ✓

**Consistency:** Store members — `entries`, `mmaps`, `batches`, `paths`, `lru`, `max_open`, `num_layers`, `experts_per_layer`, `error_msg`, `log_prefix`, `mmap_fail_verb`, `emit_diag`, `log_batch_fails`. Methods — `Load`, `DecodeEntry`, `ParseFile`, `EnsureResident`, `Free`. Wrapper/alias names match existing exported names across every task. ✓