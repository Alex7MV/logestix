# Audit Quick Wins (converter dedup, infra guards, build noise) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the last duplicated converter I/O helpers (39 definitions), lock the 2026-11-03 centralizations behind static guards, and make the build log readable by suppressing third-party / NVCC-pedantic warning noise.

**Architecture:** Three shared helpers land in the existing converter layer (`lsxcommon/converter.{h,cpp}`), the 39 per-family copies are deleted (call sites now resolve to the shared symbols), and a new fast guard test scans the tree for three regression classes (dependency build paths, duplicate helper definitions, per-target CUDA arch literals). The build-noise task changes only the root CMake flags and the `lsx_dep_includes` interface target; no library code changes.

**Tech Stack:** C++20, CUDA 13.4 (Blackwell sm_120/sm_121), CMake presets (`linux-debug`), Catch2 v3.16.1, Apache Arrow IPC, NVFP4.

## Global Constraints

- Branch: `audit/2026-11-03-quick-wins` (created from `main` @ `803412af`; the design is committed as `d0a6d1df`).
- Working directory: `/mnt/fast_storage/projects/cuda/logestix`. Build tree: `build/linux-debug` (Ninja, GCC 15). Test binaries: `build/linux-debug/tests/<name>`.
- No subagents (AGENTS.md autonomous mode). All steps run in the main session.
- TDD: a test must fail before the implementation. For the guard test the "red" is proven by a mutation (Task 3, Steps 4-7).
- Logging only via spdlog; no `std::cout`/`printf` in `libs/` or `apps/` (`test_logging_guard`).
- Headers reachable from `.cu` must not include `<absl/container/*>` (`test_absl_container_guard`).
- Every new ctest gets a label in the same commit (`test_ctest_labels_guard`); new registrations use `lsx_add_test` / `lsx_add_cuda_test`.
- English commit messages with repo prefixes (`test:`, `refactor:`, `build:`, `docs:`).
- Commits are authorized for this branch (the operator asked for the work in a separate branch).
- Device codegen arch comes from `LSX_CUDA_ARCH` only. Dependency include paths come from `lsx_dep_includes` only.
- Warning baseline on `main` (full `linux-debug` build, 2026-11-03): 51 408 `warning:` lines in 159 487 log lines.

---

### Task 1: Shared `WriteBatch` + `WriteExpertArrow` + `ExpertArrowPath`

**Files:**
- Create: `tests/test_converter_helpers.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `libs/lsxcommon/include/lsxcommon/converter.h`
- Modify: `libs/lsxcommon/src/converter.cpp`
- Modify (delete the batch writer): `libs/lsxcommon/src/{step,ling,flint,beam,hy,qwen_moe,qwen_moe_exp,ornith,kimi_moe_k3,alice,nemotron_moe,nemotron_moe_nvfp4,glm_moe_flash}_convert.cpp`, `libs/lsxcommon/src/minimax_moe_convert.cpp`
- Modify (delete the expert writer + update its call site): the same 13 files that define one

**Interfaces:**
- Consumes: `ExpertSchema::MakeBatchView(const PackedExpert&)`, `ExpertSchema::UnpackBatch(batch, int32_t scale_block = 0)`, `DenseNvfp4Schema::{MakeBatch,UnpackBatch}`, `WriteRecordBatchToFile`, `ReadRecordBatchFromFile(path, expected_columns)`, `diag::ValidateBatchBuffers`.
- Produces:
  - `arrow::Status lsxcommon::WriteBatch(const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch, const std::filesystem::path &path);`
  - `std::filesystem::path lsxcommon::ExpertArrowPath(const std::filesystem::path &model_root, int layer, int expert);`
  - `arrow::Status lsxcommon::WriteExpertArrow(const std::filesystem::path &path, const PackedExpert &expert);`

- [ ] **Step 1: Write the failing helper test**

Create `tests/test_converter_helpers.cpp`:

```cpp
// Shared converter write helpers. The per-family copies that used to carry
// these bodies are gone, so their contract is pinned here instead.
#include <catch2/catch_test_macros.hpp>

#include <arrow/api.h>
#include <lsxcommon/converter.h>
#include <lsxcommon/schema.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path FreshDir(const char *tag)
{
    const fs::path dir =
        fs::temp_directory_path() / (std::string("logestix_conv_helpers_") + tag);
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

lsxcommon::DenseNvfp4Record DenseRecord()
{
    lsxcommon::DenseNvfp4Record r;
    r.name = "model.layers.0.self_attn.q_proj.weight";
    r.shape = {2, 16};
    r.dtype = "BF16";
    r.packed.assign(16, 0xA5);
    r.scales = {0x7E, 0x7F};
    r.global_scale = 4.0f;
    return r;
}

lsxcommon::PackedExpert SampleExpert()
{
    lsxcommon::PackedExpert pe;
    pe.w1_shape = pe.w2_shape = pe.w3_shape = {2, 16};
    pe.w1_packed.assign(16, 0x11);
    pe.w2_packed.assign(16, 0x22);
    pe.w3_packed.assign(16, 0x33);
    pe.w1_scales = {0x7E, 0x7F};
    pe.w2_scales = {0x7E, 0x7F};
    pe.w3_scales = {0x7E, 0x7F};
    pe.w1_global_scale = 1.5f;
    pe.w2_global_scale = 2.5f;
    pe.w3_global_scale = 3.5f;
    return pe;
}

} // namespace

TEST_CASE("ExpertArrowPath builds the canonical routed-expert path",
          "[converter][helpers]")
{
    const fs::path root = "/tmp/vault/model";
    CHECK(lsxcommon::ExpertArrowPath(root, 7, 3) ==
          root / "experts" / "layer_7" / "exp_3.arrow");
    CHECK(lsxcommon::ExpertArrowPath(root, 0, 0) ==
          root / "experts" / "layer_0" / "exp_0.arrow");
}

TEST_CASE("WriteBatch propagates a failed batch result", "[converter][helpers]")
{
    const fs::path dir = FreshDir("writebatch_error");
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> bad(
        arrow::Status::Invalid("synthetic batch failure"));

    const auto status = lsxcommon::WriteBatch(bad, dir / "out.arrow");
    REQUIRE_FALSE(status.ok());
    CHECK(status.ToString().find("synthetic batch failure") !=
          std::string::npos);
    CHECK_FALSE(fs::exists(dir / "out.arrow"));
}

TEST_CASE("WriteBatch round-trips a dense NVFP4 batch", "[converter][helpers]")
{
    const fs::path dir = FreshDir("writebatch_ok");
    const auto record = DenseRecord();
    auto batch = lsxcommon::DenseNvfp4Schema::MakeBatch({record});
    REQUIRE(batch.ok());

    const fs::path path = dir / "dense.arrow";
    REQUIRE(lsxcommon::WriteBatch(batch, path).ok());

    auto back = lsxcommon::ReadRecordBatchFromFile(path, 6);
    REQUIRE(back.ok());
    auto records = lsxcommon::DenseNvfp4Schema::UnpackBatch(*back);
    REQUIRE(records.ok());
    REQUIRE(records->size() == 1);
    CHECK((*records)[0].name == record.name);
    CHECK((*records)[0].shape == record.shape);
    CHECK((*records)[0].dtype == record.dtype);
    CHECK((*records)[0].packed == record.packed);
    CHECK((*records)[0].scales == record.scales);
    CHECK((*records)[0].global_scale == record.global_scale);
}

TEST_CASE("WriteExpertArrow creates the layer directory and round-trips",
          "[converter][helpers]")
{
    const fs::path root = FreshDir("expertfile_ok");
    const auto expert = SampleExpert();
    const fs::path path = lsxcommon::ExpertArrowPath(root, 7, 3);

    REQUIRE(lsxcommon::WriteExpertArrow(path, expert).ok());
    REQUIRE(fs::exists(path));

    auto batch = lsxcommon::ReadRecordBatchFromFile(path, 9);
    REQUIRE(batch.ok());
    auto back = lsxcommon::ExpertSchema::UnpackBatch(*batch, 16);
    REQUIRE(back.ok());
    CHECK(back->w1_packed == expert.w1_packed);
    CHECK(back->w2_packed == expert.w2_packed);
    CHECK(back->w3_packed == expert.w3_packed);
    CHECK(back->w1_scales == expert.w1_scales);
    CHECK(back->w2_scales == expert.w2_scales);
    CHECK(back->w3_scales == expert.w3_scales);
    CHECK(back->w1_global_scale == expert.w1_global_scale);
    CHECK(back->w2_global_scale == expert.w2_global_scale);
    CHECK(back->w3_global_scale == expert.w3_global_scale);
}

TEST_CASE("WriteExpertArrow reports an unwritable path", "[converter][helpers]")
{
    const fs::path root = FreshDir("expertfile_fail");
    const fs::path blocker = root / "blocked";
    { std::ofstream(blocker) << "not a directory"; }

    const auto expert = SampleExpert();
    const auto status =
        lsxcommon::WriteExpertArrow(blocker / "layer_0" / "exp_0.arrow", expert);
    CHECK_FALSE(status.ok());
}
```

- [ ] **Step 2: Register the test and its label**

In `tests/CMakeLists.txt`, immediately after the `test_safetensors`
registration block:

```cmake
# ── Shared converter write helpers (host-only, synthetic batches) ──────
lsx_add_test(NAME test_converter_helpers LINK lsxcommon)
```

Add `test_converter_helpers` to the `fast` label group (the
`set_tests_properties(` block that starts with `test_core`), right after
`test_safetensors`:

```cmake
    test_core
    test_fast_load
    test_safetensors
    test_converter_helpers
    test_geometry_guard
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_converter_helpers`
Expected: link failure — `undefined reference to lsxcommon::ExpertArrowPath(...)`
and `... WriteBatch(...)` / `... WriteExpertArrow(...)`.

- [ ] **Step 4: Declare the helpers**

In `libs/lsxcommon/include/lsxcommon/converter.h`, directly below the
`SaveExpertToArrow` declaration:

```cpp
// Unwraps a `MakeBatch()` result and writes it (14 identical per-family
// adapters were deleted in favour of this one).
arrow::Status WriteBatch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path);

// <model_root>/experts/layer_<layer>/exp_<expert>.arrow — the canonical layout
// of the families that do not zero-pad names. glm-mode-flash pads its own path
// and alice passes its layer directory; both call WriteExpertArrow directly.
std::filesystem::path ExpertArrowPath(const std::filesystem::path &model_root,
                                      int layer, int expert);

// Creates the parent directory, wraps `expert` in a zero-copy batch,
// validates the buffers and writes the record. No per-file log: converters
// write tens of thousands of experts (SaveExpertToArrow is the logging
// single-file wrapper).
arrow::Status WriteExpertArrow(const std::filesystem::path &path,
                               const PackedExpert &expert);
```

- [ ] **Step 5: Implement the helpers and delegate `SaveExpertToArrow`**

In `libs/lsxcommon/src/converter.cpp`, replace the body of
`SaveExpertToArrow` with the delegation and add the three implementations above
it (`absl::StrCat`, `diagnostics.h`, `schema.h` are already included):

```cpp
arrow::Status WriteBatch(
    const arrow::Result<std::shared_ptr<arrow::RecordBatch>> &batch,
    const std::filesystem::path &path)
{
    if (!batch.ok()) return batch.status();
    return WriteRecordBatchToFile(*batch, path);
}

std::filesystem::path ExpertArrowPath(const std::filesystem::path &model_root,
                                      int layer, int expert)
{
    return model_root / "experts" / absl::StrCat("layer_", layer) /
           absl::StrCat("exp_", expert, ".arrow");
}

arrow::Status WriteExpertArrow(const std::filesystem::path &path,
                               const PackedExpert &expert)
{
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);

    auto batch = ExpertSchema::MakeBatchView(expert);
    if (!batch.ok()) return batch.status();
    diag::ValidateBatchBuffers(**batch, "WriteExpertArrow pre-write");
    return WriteRecordBatchToFile(*batch, path);
}

arrow::Status SaveExpertToArrow(
    const PackedExpert &expert,
    int layer_idx,
    int expert_idx,
    const std::filesystem::path &output_root)
{
    const auto path = ExpertArrowPath(output_root, layer_idx, expert_idx);
    ARROW_RETURN_NOT_OK(WriteExpertArrow(path, expert));
    lsxcommon::log::info(absl::StrFormat("  wrote layer=%03d expert=%03d  -> %s",
                                         layer_idx, expert_idx, path.string()));
    return arrow::Status::OK();
}
```

Note: the old log line printed the first column's length; the new one prints
the path. No test asserts converter log text.

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_converter_helpers && \
./build/linux-debug/tests/test_converter_helpers
```
Expected: `All tests passed (… assertions in 5 test cases)`.

- [ ] **Step 7: Delete the 14 batch-writer copies**

Delete the whole `arrow::Status WriteBatch(...)` / `write_batch(...)`
definition (body identical to the new one) from:

`step_moe_convert.cpp`, `ling_moe_convert.cpp`, `flint_moe_convert.cpp`,
`beam_moe_convert.cpp`, `hy_moe_convert.cpp`, `qwen_moe_convert.cpp`,
`qwen_moe_exp_convert.cpp`, `ornith_moe_convert.cpp`,
`kimi_moe_k3_convert.cpp`, `alice_moe_convert.cpp`, `nemotron_moe_convert.cpp`,
`nemotron_moe_nvfp4_convert.cpp`, `glm_moe_flash_convert.cpp`,
`minimax_moe_convert.cpp`.

Unqualified `write_batch(...)` / `WriteBatch(...)` calls in those files now
resolve to `lsxcommon::WriteBatch`.

- [ ] **Step 8: Delete the 13 expert-writer copies and repoint their call sites**

For each file, delete the `write_expert_file` / `WriteExpertFile` definition
and replace its single call site.

Standard layout (11 files: `step`, `ling`, `flint`, `beam`, `hy`, `qwen_moe`,
`qwen_moe_exp`, `ornith`, `nemotron_moe`, `nemotron_moe_nvfp4`, `minimax`):
replace `write_expert_file(base, l, e, std::move(pe))` with

```cpp
WriteExpertArrow(ExpertArrowPath(base, l, e), pe)
```

`alice_moe_convert.cpp` (the helper received the layer directory):

```cpp
WriteExpertArrow(dir / absl::StrCat("exp_", e, ".arrow"), pe)
```

`glm_moe_flash_convert.cpp` (zero-padded names, which its loader expects). Add
this lambda next to the other path helpers at the top of the converter function
that writes experts:

```cpp
const auto expert_path = [](const fs::path &root, int l, int e) {
    return root / "experts" /
           absl::StrCat("layer_", absl::Dec(l, absl::kZeroPad3)) /
           absl::StrCat("exp_", absl::Dec(e, absl::kZeroPad3), ".arrow");
};
```

and replace the call with `WriteExpertArrow(expert_path(base, l, e), pe)`.

`minimax_moe_convert.cpp` already uses `MakeBatchView`; the shared implementation
matches it.

- [ ] **Step 9: Compile the library and run the converter suite**

```bash
cmake --build --preset linux-debug --target lsxcommon > /tmp/opencode/m1_lib.log 2>&1; echo "exit=$?"
rg -n "error:" /tmp/opencode/m1_lib.log
```
Expected: `exit=0`, no `error:` lines (the warnings in the log are the known
third-party noise that Task 4 removes).

Then run: `ctest --preset linux-debug-convert`
Expected: `100% tests passed, 0 tests failed out of 37`.

- [ ] **Step 10: Commit**

```bash
git add tests/test_converter_helpers.cpp tests/CMakeLists.txt \
        libs/lsxcommon/include/lsxcommon/converter.h \
        libs/lsxcommon/src/converter.cpp \
        libs/lsxcommon/src/{step,ling,flint,beam,hy,qwen_moe,qwen_moe_exp,ornith,kimi_moe_k3,alice,nemotron_moe,nemotron_moe_nvfp4,glm_moe_flash,minimax_moe}_convert.cpp
git commit -m "refactor(converters): centralize batch and expert writes"
```

---

### Task 2: Shared `QuantizeSlice` (two overloads)

**Files:**
- Modify: `libs/lsxcommon/include/lsxcommon/converter.h`
- Modify: `libs/lsxcommon/src/converter.cpp`
- Modify: `tests/test_nvfp4_pack.cpp`
- Modify (delete the local definition; update call sites): `libs/lsxcommon/src/{step,flint,hy,beam,qwen_moe,ornith,alice,nemotron_moe,qwen_moe_exp,kimi_moe_k3,minimax_moe}_convert.cpp`

**Interfaces:**
- Consumes: `gpu::quantize_tensor_gpu(const uint8_t *raw, int64_t numel, const std::string &dtype, const uint8_t *e8m0_scales, const std::vector<int64_t> &shape, const std::vector<int64_t> &e8m0_shape, std::vector<uint8_t> &packed, std::vector<uint8_t> &scales, float &global_scale)` (`lsxcommon/safetensors_gpu.h`, already included by `converter.cpp`). `out.shape` is an INPUT: the caller sets it.
- Produces:
  - `bool lsxcommon::QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype, const uint8_t *e8m0, const std::vector<int64_t> &shape, const std::vector<int64_t> &e8m0_shape, DenseNvfp4Record &out);`
  - `bool lsxcommon::QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype, DenseNvfp4Record &out);`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_nvfp4_pack.cpp` (it already has `GpuAvailable()`,
`<cmath>`, `<cstdlib>` and the `SKIP_RETURN_CODE 4` CTest property):

```cpp
TEST_CASE("QuantizeSlice produces the dense NVFP4 contract", "[nvfp4][converter]")
{
    if (!GpuAvailable())
        std::exit(4);

    constexpr int64_t kNumel = 32;
    // BF16 1.0 little-endian, 32 elements.
    std::vector<uint8_t> bf16(static_cast<size_t>(kNumel) * 2);
    for (int64_t i = 0; i < kNumel; ++i) {
        bf16[static_cast<size_t>(i) * 2 + 0] = 0x00;
        bf16[static_cast<size_t>(i) * 2 + 1] = 0x3F;
    }

    lsxcommon::DenseNvfp4Record rec;
    rec.shape = {2, 16};
    REQUIRE(lsxcommon::QuantizeSlice(bf16.data(), kNumel, "BF16", rec));
    CHECK(rec.packed.size() == 16);   // 32 elements -> 16 nibble bytes
    CHECK(rec.scales.size() == 2);    // 32 elements / 16 -> 2 E4M3 scales
    CHECK(std::isfinite(rec.global_scale));
    CHECK(rec.global_scale > 0.0f);

    lsxcommon::DenseNvfp4Record again;
    again.shape = {2, 16};
    REQUIRE(lsxcommon::QuantizeSlice(bf16.data(), kNumel, "BF16", again));
    CHECK(again.packed == rec.packed);
    CHECK(again.scales == rec.scales);
    CHECK(again.global_scale == rec.global_scale);
}

TEST_CASE("QuantizeSlice full form without E8M0 equals the short form",
          "[nvfp4][converter]")
{
    if (!GpuAvailable())
        std::exit(4);

    constexpr int64_t kNumel = 32;
    std::vector<uint8_t> bf16(static_cast<size_t>(kNumel) * 2);
    for (int64_t i = 0; i < kNumel; ++i) {
        bf16[static_cast<size_t>(i) * 2 + 0] = 0x00;
        bf16[static_cast<size_t>(i) * 2 + 1] = 0x3F;
    }

    lsxcommon::DenseNvfp4Record short_form;
    short_form.shape = {2, 16};
    REQUIRE(lsxcommon::QuantizeSlice(bf16.data(), kNumel, "BF16", short_form));

    lsxcommon::DenseNvfp4Record full_form;
    full_form.shape = {2, 16};
    REQUIRE(lsxcommon::QuantizeSlice(bf16.data(), kNumel, "BF16", nullptr,
                                     full_form.shape, {}, full_form));
    CHECK(full_form.packed == short_form.packed);
    CHECK(full_form.scales == short_form.scales);
    CHECK(full_form.global_scale == short_form.global_scale);
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cmake --build --preset linux-debug --target test_nvfp4_pack`
Expected: compile error — `'QuantizeSlice' was not declared in this scope`.

- [ ] **Step 3: Implement the overloads**

Declare them in `libs/lsxcommon/include/lsxcommon/converter.h`, next to
`WriteExpertArrow`:

```cpp
// Quantizes a contiguous slice to NVFP4. The full form carries the optional
// E8M0 companion block scales (minimax's already-quantized MXFP8 dense side);
// the short form passes nullptr and uses `out.shape`. Both return false on a
// GPU failure, exactly like the 12 per-family copies they replace.
bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   const uint8_t *e8m0, const std::vector<int64_t> &shape,
                   const std::vector<int64_t> &e8m0_shape,
                   DenseNvfp4Record &out);
bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   DenseNvfp4Record &out);
```

Implement them in `libs/lsxcommon/src/converter.cpp`, next to `WriteBatch`:

```cpp
bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   const uint8_t *e8m0, const std::vector<int64_t> &shape,
                   const std::vector<int64_t> &e8m0_shape,
                   DenseNvfp4Record &out)
{
    return gpu::quantize_tensor_gpu(data, numel, dtype, e8m0, shape, e8m0_shape,
                                    out.packed, out.scales, out.global_scale);
}

bool QuantizeSlice(const uint8_t *data, int64_t numel, const std::string &dtype,
                   DenseNvfp4Record &out)
{
    return QuantizeSlice(data, numel, dtype, nullptr, out.shape, {}, out);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
cmake --build --preset linux-debug --target test_nvfp4_pack && \
./build/linux-debug/tests/test_nvfp4_pack
```
Expected: all cases pass, including the two new ones.

- [ ] **Step 5: Delete the 12 local definitions and fix the call-site shapes**

Delete every local `QuantizeSlice` / `quantize_slice` definition from:
`step_moe_convert.cpp`, `flint_moe_convert.cpp`, `hy_moe_convert.cpp`,
`beam_moe_convert.cpp`, `minimax_moe_convert.cpp` (two overloads),
`qwen_moe_convert.cpp`, `ornith_moe_convert.cpp`, `alice_moe_convert.cpp`,
`nemotron_moe_convert.cpp`, `qwen_moe_exp_convert.cpp`,
`kimi_moe_k3_convert.cpp`.

Call sites that passed a dtype (`QuantizeSlice(data, n, dtype, rec)`) and the
minimax full-form call (`QuantizeSlice(data, n, dtype, e8m0, shape,
e8m0_shape, rec)`) compile unchanged against the shared overloads.

Two shapes do not: add the explicit dtype.

- BF16-only variants (5 files) — `quantize_slice(data, numel, r)` becomes:
  ```cpp
  QuantizeSlice(data, numel, "BF16", r)
  ```
  Files and call counts: `qwen_moe_convert.cpp` (7), `ornith_moe_convert.cpp`
  (7), `alice_moe_convert.cpp` (5), `nemotron_moe_convert.cpp` (2),
  `qwen_moe_exp_convert.cpp` (9).
- `kimi_moe_k3_convert.cpp` (2 calls) — `quantize_slice(data, numel, t, r)`
  becomes (preserving its dtype policy):
  ```cpp
  QuantizeSlice(data, numel, t.dtype == "F32" ? "F32" : "BF16", r)
  ```

- [ ] **Step 6: Run the converter suite**

Run:
```bash
cmake --build --preset linux-debug --target lsxcommon > /tmp/opencode/m2_lib.log 2>&1; echo "exit=$?"
rg -n "error:" /tmp/opencode/m2_lib.log
ctest --preset linux-debug-convert
```
Expected: `exit=0`, no errors, 37/37 green (the GPU pack test skips with code 4
only when no CUDA device is visible).

- [ ] **Step 7: Commit**

```bash
git add libs/lsxcommon/include/lsxcommon/converter.h \
        libs/lsxcommon/src/converter.cpp tests/test_nvfp4_pack.cpp \
        libs/lsxcommon/src/{step,flint,hy,beam,qwen_moe,ornith,alice,nemotron_moe,qwen_moe_exp,kimi_moe_k3,minimax_moe}_convert.cpp
git commit -m "refactor(converters): centralize NVFP4 slice quantization"
```

---

### Task 3: Infrastructure guard test

**Files:**
- Create: `tests/test_infrastructure_guard.cpp`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `LSX_SOURCE_DIR` (compile definition), `lsx_add_test`, the `fast`
  label group.
- Produces: `test_infrastructure_guard` with three pure scanners
  (`FindDependencyPaths`, `FindDuplicateDefinitions`, `FindArchLiterals`) and
  three tree assertions.

- [ ] **Step 1: Create the guard test**

Create `tests/test_infrastructure_guard.cpp`:

```cpp
// Infrastructure guard: the centralizations of 2026-11-03 (dependency include
// paths, the shared tensor reader/writers, the CUDA arch variable) must not be
// undone one call site at a time.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

std::string ReadFile(const fs::path &path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// CMake comments start with '#'.
std::string StripHashComments(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    bool comment = false;
    for (char c : text) {
        if (comment) {
            if (c == '\n') {
                comment = false;
                out += c;
            }
            continue;
        }
        if (c == '#') {
            comment = true;
            continue;
        }
        out += c;
    }
    return out;
}

// Strips // and /* */ comments so a commented-out signature cannot trip the
// definition scan. A state machine, not a regex: line numbers must survive
// multi-line block comments.
std::string StripCppComments(const std::string &text)
{
    std::string out;
    out.reserve(text.size());
    bool line_comment = false;
    bool block_comment = false;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (line_comment) {
            if (c == '\n') {
                line_comment = false;
                out += c;
            }
            continue;
        }
        if (block_comment) {
            if (c == '*' && i + 1 < text.size() && text[i + 1] == '/') {
                block_comment = false;
                ++i;
            } else if (c == '\n') {
                out += c;
            }
            continue;
        }
        if (c == '/' && i + 1 < text.size() && text[i + 1] == '/') {
            line_comment = true;
            ++i;
            continue;
        }
        if (c == '/' && i + 1 < text.size() && text[i + 1] == '*') {
            block_comment = true;
            ++i;
            continue;
        }
        out += c;
    }
    return out;
}

std::vector<std::string> Lines(const std::string &text)
{
    std::vector<std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) out.push_back(line);
    return out;
}

// Rule 1: no CMake call site may spell the FetchContent build directory.
std::vector<int> FindDependencyPaths(const std::string &text)
{
    static const std::regex re(
        R"(\$\{CMAKE_BINARY_DIR\}/_deps|(^|[^A-Za-z0-9_])_deps/)");
    std::vector<int> hits;
    int line_no = 0;
    for (const auto &line : Lines(StripHashComments(text))) {
        ++line_no;
        if (std::regex_search(line, re)) hits.push_back(line_no);
    }
    return hits;
}

// Rule 2: each shared helper is DEFINED once (a call or an '=' initializer is
// not a definition, so the match is anchored on the return type).
std::vector<int> FindDuplicateDefinitions(const std::string &text)
{
    static const std::regex re(
        R"((?:std::vector\s*<\s*uint8_t\s*>|arrow::Status|bool|)"
        R"(std::filesystem::path)\s+)"
        R"((ReadTensorData|read_tensor_data|WriteBatch|write_batch|)"
        R"(WriteExpertArrow|WriteExpertFile|write_expert_file|ExpertArrowPath|)"
        R"(QuantizeSlice|quantize_slice)\s*\()");
    std::vector<int> hits;
    int line_no = 0;
    for (const auto &line : Lines(StripCppComments(text))) {
        ++line_no;
        if (std::regex_search(line, re)) hits.push_back(line_no);
    }
    return hits;
}

// Rule 3: device codegen arch is LSX_CUDA_ARCH or OFF only.
std::vector<int> FindArchLiterals(const std::string &text)
{
    static const std::regex call(R"(CUDA_ARCHITECTURES\s+([^\s\)]+))");
    std::vector<int> hits;
    int line_no = 0;
    for (const auto &line : Lines(StripHashComments(text))) {
        ++line_no;
        std::smatch m;
        auto begin = line.cbegin();
        while (std::regex_search(begin, line.cend(), m, call)) {
            std::string arg = m[1].str();
            if (arg.size() >= 2 && arg.front() == '"' && arg.back() == '"')
                arg = arg.substr(1, arg.size() - 2);
            if (arg != "OFF" &&
                arg.find("${LSX_CUDA_ARCH}") == std::string::npos)
                hits.push_back(line_no);
            begin = m.suffix().first;
        }
    }
    return hits;
}

std::vector<fs::path> CmakeFiles(const fs::path &root)
{
    std::vector<fs::path> out{root / "CMakeLists.txt"};
    std::error_code ec;
    if (fs::exists(root / "cmake", ec)) {
        for (const auto &e : fs::directory_iterator(root / "cmake", ec))
            if (e.is_regular_file() && e.path().extension() == ".cmake")
                out.push_back(e.path());
    }
    for (const char *dir : {"libs", "apps", "tests"}) {
        ec.clear();
        for (fs::recursive_directory_iterator it(root / dir, ec), end;
             it != end; it.increment(ec)) {
            if (ec) break;
            if (it->is_regular_file() &&
                it->path().filename() == "CMakeLists.txt")
                out.push_back(it->path());
        }
    }
    return out;
}

// Production sources only: test fixtures may legitimately name these symbols.
std::vector<fs::path> SourceFiles(const fs::path &root)
{
    std::vector<fs::path> out;
    std::error_code ec;
    for (const char *dir : {"libs", "apps"}) {
        ec.clear();
        for (fs::recursive_directory_iterator it(root / dir, ec), end;
             it != end; it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file()) continue;
            const auto ext = it->path().extension().string();
            if (ext == ".cpp" || ext == ".cu" || ext == ".h" || ext == ".cuh")
                out.push_back(it->path());
        }
    }
    return out;
}

std::string Rel(const fs::path &root, const fs::path &p)
{
    return fs::relative(p, root).generic_string();
}

std::string JoinHits(const std::vector<std::string> &hits)
{
    std::string joined;
    for (const auto &h : hits) joined += h + "\n";
    return joined;
}

} // namespace

TEST_CASE("infra guard: dependency-path scanner", "[guard][infra]")
{
    CHECK(FindDependencyPaths(
              "target_include_directories(t PRIVATE "
              "${CMAKE_BINARY_DIR}/_deps/arrow-src)")
              .size() == 1);
    CHECK(FindDependencyPaths(
              "include_directories(${CMAKE_BINARY_DIR}/_deps/b/include)")
              .size() == 1);
    CHECK(FindDependencyPaths(
              "include_directories(${FETCHCONTENT_BASE_DIR}/arrow-src/cpp/src)")
              .empty());
    CHECK(FindDependencyPaths("# legacy ${CMAKE_BINARY_DIR}/_deps/c").empty());
}

TEST_CASE("infra guard: duplicate-definition scanner", "[guard][infra]")
{
    CHECK(FindDuplicateDefinitions(
              "std::vector<uint8_t> ReadTensorData(const T &a, const T &b)")
              .size() == 1);
    CHECK(FindDuplicateDefinitions(
              "arrow::Status WriteExpertFile(const P &p, int l, int e)")
              .size() == 1);
    CHECK(FindDuplicateDefinitions(
              "bool QuantizeSlice(const uint8_t *d, int64_t n)")
              .size() == 1);
    CHECK(FindDuplicateDefinitions(
              "std::filesystem::path ExpertArrowPath(const P &p, int l, int e)")
              .size() == 1);
    CHECK(FindDuplicateDefinitions(
              "auto d = ReadTensorData(input_dir, shard, t);")
              .empty());
    CHECK(FindDuplicateDefinitions("if (QuantizeSlice(x, 4, dt, r)) {").empty());
    CHECK(FindDuplicateDefinitions(
              "// bool QuantizeSlice(const uint8_t *d, int64_t n)")
              .empty());
}

TEST_CASE("infra guard: arch-literal scanner", "[guard][infra]")
{
    CHECK(FindArchLiterals(
              "set_target_properties(t PROPERTIES CUDA_ARCHITECTURES "
              "\"120-real;121-real\")")
              .size() == 1);
    CHECK(FindArchLiterals("CUDA_ARCHITECTURES \"90\"").size() == 1);
    CHECK(FindArchLiterals("CUDA_ARCHITECTURES ${LSX_CUDA_ARCH}").empty());
    CHECK(FindArchLiterals("CUDA_ARCHITECTURES \"${LSX_CUDA_ARCH}\"").empty());
    CHECK(FindArchLiterals("CUDA_ARCHITECTURES OFF").empty());
}

TEST_CASE("infra guard: no dependency build paths in CMake", "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> bad;
    for (const auto &p : CmakeFiles(root))
        for (int line : FindDependencyPaths(ReadFile(p)))
            bad.push_back(Rel(root, p) + ":" + std::to_string(line));
    INFO("dependency path hits (" << bad.size() << "):\n" << JoinHits(bad));
    REQUIRE(bad.empty());
}

TEST_CASE("infra guard: shared converter helpers have one definition",
          "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    // converter.cpp implements the helpers; converter.h and safetensors.h
    // declare them. Everywhere else a matching signature is a revived copy.
    const fs::path canonical =
        fs::weakly_canonical(root / "libs/lsxcommon/src/converter.cpp");
    const fs::path canonical_header = fs::weakly_canonical(
        root / "libs/lsxcommon/include/lsxcommon/converter.h");
    const fs::path canonical_reader_header = fs::weakly_canonical(
        root / "libs/lsxcommon/include/lsxcommon/safetensors.h");
    std::vector<std::string> bad;
    for (const auto &p : SourceFiles(root)) {
        const fs::path abs = fs::weakly_canonical(p);
        if (abs == canonical || abs == canonical_header ||
            abs == canonical_reader_header)
            continue;
        for (int line : FindDuplicateDefinitions(ReadFile(p)))
            bad.push_back(Rel(root, p) + ":" + std::to_string(line));
    }
    INFO("duplicate definitions (" << bad.size() << "):\n" << JoinHits(bad));
    REQUIRE(bad.empty());
}

TEST_CASE("infra guard: device codegen arch is centralized", "[guard][infra]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> bad;
    for (const auto &p : CmakeFiles(root))
        for (int line : FindArchLiterals(ReadFile(p)))
            bad.push_back(Rel(root, p) + ":" + std::to_string(line));
    INFO("arch literal hits (" << bad.size() << "):\n" << JoinHits(bad));
    REQUIRE(bad.empty());
}
```

- [ ] **Step 2: Register the guard**

In `tests/CMakeLists.txt`, next to the other guards
(`test_model_independence_guard`):

```cmake
# Guard: infrastructure centralization (dependency paths, shared helpers,
# CUDA arch). Host-only static scan, see spec 2026-11-03.
lsx_add_test(NAME test_infrastructure_guard
    DEFINE "LSX_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")
```

Add `test_infrastructure_guard` to the `fast` label group, after
`test_ctest_labels_guard`.

- [ ] **Step 3: Build and run the guard on the clean tree**

Run:
```bash
cmake --build --preset linux-debug --target test_infrastructure_guard && \
./build/linux-debug/tests/test_infrastructure_guard
```
Expected: all cases pass. If `duplicate definitions` reports a hit, that file
still owns a copy from Tasks 1-2 — finish deleting it. If `dependency path
hits` reports the root `CMakeLists.txt`, it is a non-comment occurrence that the
2026-11-03 migration missed — replace it with `lsx_dep_includes`.

- [ ] **Step 4: Mutation check — dependency paths**

Save a backup first (Step 7 must not revert the registration from Step 2):

```bash
cp tests/CMakeLists.txt /tmp/opencode/tests_cmake.bak
```

Temporarily append to `tests/CMakeLists.txt`:

```cmake
target_include_directories(test_core PRIVATE ${CMAKE_BINARY_DIR}/_deps/arrow-src)
```

Run `cmake --build --preset linux-debug --target test_infrastructure_guard && ./build/linux-debug/tests/test_infrastructure_guard`.
Expected: FAIL with `dependency path hits (1)` and the line number.

- [ ] **Step 5: Mutation check — duplicate definition**

Temporarily create `libs/lsxcommon/src/zz_guard_probe.cpp`:

```cpp
#include <cstdint>
#include <vector>

std::vector<uint8_t> ReadTensorData(const void *, const void *, const void *)
{
    return {};
}
```

Run the guard again. Expected: FAIL with `duplicate definitions (1)` naming
`libs/lsxcommon/src/zz_guard_probe.cpp`.

- [ ] **Step 6: Mutation check — arch literal**

Temporarily add to `tests/CMakeLists.txt`:

```cmake
set_target_properties(test_core PROPERTIES CUDA_ARCHITECTURES "90")
```

Run the guard again. Expected: FAIL with `arch literal hits (1)`.

- [ ] **Step 7: Revert every mutation and re-run**

```bash
cp /tmp/opencode/tests_cmake.bak tests/CMakeLists.txt
rm -f libs/lsxcommon/src/zz_guard_probe.cpp
cmake --build --preset linux-debug --target test_infrastructure_guard && \
./build/linux-debug/tests/test_infrastructure_guard
git status --short   # only the intended new/modified files
```
Expected: guard green again, and `git diff tests/CMakeLists.txt` contains only
the test registrations + label lines.

- [ ] **Step 8: Run the fast suite**

Run: `ctest --preset linux-debug-fast`
Expected: `100% tests passed` (134 tests: 132 + `test_converter_helpers` +
`test_infrastructure_guard`).

- [ ] **Step 9: Commit**

```bash
git add tests/test_infrastructure_guard.cpp tests/CMakeLists.txt
git commit -m "test(guard): ban dependency paths, duplicate helpers and arch literals"
```

---

### Task 4: Build-noise removal

**Files:**
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `FETCHCONTENT_BASE_DIR`, `third_party/simdjson`, `LSX_CCCL_INCLUDE_DIR`.
- Produces: `-isystem` for every dependency include dir; `-Wpedantic` only for
  C/CXX; the measured before/after warning counts for the commit body.

- [ ] **Step 1: Record the baseline**

The baseline is already measured on `main` and lives in Global Constraints:
51 408 `warning:` lines in 159 487 log lines (full `linux-debug` build,
2026-11-03). Do not rebuild for it. The after-count comes from Step 4's
`--clean-first` build, which is the only comparable measurement (an incremental
build reports zero warnings because nothing recompiles).

- [ ] **Step 2: Gate `-Wpedantic` to C/CXX only**

In `CMakeLists.txt` replace (line ~117):

```cmake
    add_compile_options(-Wall -Wextra -Wpedantic)
```

with:

```cmake
    # -Wpedantic stays OFF for CUDA: nvcc forwards it to the host compiler,
    # which then flags nvcc's own `#line` markers ("style of line directive is
    # a GCC extension") once per generated line — 2 024 hits in a single test
    # TU. -Wall/-Wextra still apply to the CUDA host pass.
    add_compile_options(-Wall -Wextra)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>")
```

- [ ] **Step 3: Make dependency headers SYSTEM**

In `CMakeLists.txt`, replace the `target_include_directories(lsx_dep_includes INTERFACE ...)`
block with:

```cmake
# Headers of the dependencies built in-tree (add_subdirectory / submodule) are
# plain `-I`, so our TUs inherit their -Wpedantic noise (absl::int128 alone is
# 5 730 warnings). Listing them as SYSTEM INTERFACE turns them into -isystem
# for every consumer; GCC treats a directory listed in -isystem as system even
# when another target also passes it as -I.
set(_lsx_dep_system_includes "")
foreach(_lsx_system_candidate
    "${FETCHCONTENT_BASE_DIR}/spdlog-src/include"
    "${FETCHCONTENT_BASE_DIR}/abseil-cpp-src"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/include"
    "${PROJECT_SOURCE_DIR}/third_party/simdjson/singleheader")
    if(EXISTS "${_lsx_system_candidate}")
        list(APPEND _lsx_dep_system_includes "${_lsx_system_candidate}")
    endif()
endforeach()
unset(_lsx_system_candidate)

target_include_directories(lsx_dep_includes SYSTEM INTERFACE
    ${_lsx_arrow_includes}
    ${_lsx_dep_system_includes}
    ${CUDAToolkit_INCLUDE_DIRS}
    "${LSX_CCCL_INCLUDE_DIR}")
```

- [ ] **Step 4: Rebuild from scratch and measure**

```bash
cmake --build --preset linux-debug --clean-first > /tmp/opencode/build_after.log 2>&1
echo "exit=$?"
rg -c "warning:" /tmp/opencode/build_after.log
rg "warning:" /tmp/opencode/build_after.log | rg -v "_deps|third_party/" | wc -l
```
Expected: `exit=0`; total warnings ≤ 5 % of the Step 1 count.

Measured after execution (2026-11-03): the flag work alone took the build from
51408 to 484 lines, 238 of them ours. Those 238 were then cleared in a follow-up
commit (39 `__restrict__` in-place call sites, fail-closed config parsers, one
dangling test reference, `memcpy` into `__half`, 19 dead functions): the final
clean build is **246 lines, 0 from our sources**, and the trail is archived in
`docs/open-tasks/2026-11-03-warning-backlog-closed.md`. Do not silence.

- [ ] **Step 5: Inspect the remaining warnings**

```bash
rg -o "^[^:]+:[0-9]+:[0-9]+: warning:" /tmp/opencode/build_after.log \
  | sed 's/:[0-9]*:[0-9]*: warning://' | sort | uniq -c | sort -rn | head
```
Expected: any remaining entries are vendored (Arrow/absl) paths. If one of the
four `_lsx_dep_system_includes` entries is absent on this machine (e.g. spdlog
is a system package), record that rather than adding speculative paths.

- [ ] **Step 6: Run the suites**

Run:
```bash
ctest --preset linux-debug-fast
ctest --preset linux-debug-convert
```
Expected: fast 134/134, convert 37/37.

- [ ] **Step 7: Commit with the measured numbers**

Flush any background logging first (the tests do this themselves, but keep the
commit body honest):

```bash
git add CMakeLists.txt
git commit -m "build: suppress third-party and NVCC-pedantic warning noise

lsx_dep_includes exposes the Arrow build-tree, spdlog, abseil and simdjson
headers as SYSTEM, and -Wpedantic is limited to C/CXX so nvcc's host pass no
longer flags its own #line markers.

Full linux-debug build warning lines go from the Task-4 Step-1 count to the
Task-4 Step-4 count printed above, with zero warnings from our own sources;
fast 134/134, convert 37/37."
```

---

### Task 5: Documentation and final verification

**Files:**
- Modify: `AGENTS.md`
- Modify: `.opencode/skills/test-lifecycle/SKILL.md`
- Modify: `.opencode/skills/cuda-cmake-build-gotchas/SKILL.md`
- Modify: `.opencode/skills/logestix-repo-map/SKILL.md`

**Interfaces:**
- Consumes: the guard name from Task 3, the flag rule from Task 4.
- Produces: documentation consistent with the tree; no code.

- [ ] **Step 1: Update `AGENTS.md`**

In "CMake Key Points", find the paragraph starting with
"`lsx_dep_includes` (root `CMakeLists.txt`) is the ONLY place dependency
include paths live" and append after its final sentence:

```markdown
  The target's includes are SYSTEM (dependencies never emit warnings into our
  TUs), and `-Wpedantic` is limited to `COMPILE_LANGUAGE:C,CXX` — nvcc forwards
  it to the host compiler, which then flags nvcc's own `#line` markers. Three
  machine checks keep this state: `test_infrastructure_guard` (no
  `${CMAKE_BINARY_DIR}/_deps` paths, one definition per shared converter helper,
  `CUDA_ARCHITECTURES` only `${LSX_CUDA_ARCH}`/`OFF`).
```

- [ ] **Step 2: Update the guard table in `test-lifecycle`**

Add a row for `test_infrastructure_guard` with the three rules (dependency
paths, single definition of `ReadTensorData`/`WriteBatch`/`WriteExpertArrow`/
`ExpertArrowPath`/`QuantizeSlice`, arch literals), and add
`test_converter_helpers` next to `test_safetensors` in the test inventory
sentence.

- [ ] **Step 3: Update the symptom table in `cuda-cmake-build-gotchas`**

Add two rows:

```markdown
| 50k warnings per full build (absl `__int128`, Arrow `safe_math`) | dependency include dirs are `-I`, so `-Wpedantic` applies; make them SYSTEM via `lsx_dep_includes` |
| "style of line directive is a GCC extension" repeated once per source line in a `.cu` | `-Wpedantic` leaked into the NVCC host pass; gate it to `COMPILE_LANGUAGE:C,CXX` |
```

- [ ] **Step 4: Update `logestix-repo-map`**

Add `tests/test_converter_helpers.cpp` and `tests/test_infrastructure_guard.cpp`
to the tests inventory (one line each, with label and what they pin).

- [ ] **Step 5: Final verification**

```bash
cmake --build --preset linux-debug > /tmp/opencode/build_final.log 2>&1; echo "build=$?"
ctest --preset linux-debug-fast 2>&1 | tail -3
ctest --preset linux-debug-convert 2>&1 | tail -3
rg -c "warning:" /tmp/opencode/build_final.log
git status --short
```
Expected: `build=0`; fast 134/134; convert 37/37; warnings at the Task 4 level;
no stray files.

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md .opencode/skills/test-lifecycle/SKILL.md \
        .opencode/skills/cuda-cmake-build-gotchas/SKILL.md \
        .opencode/skills/logestix-repo-map/SKILL.md
git commit -m "docs: record the infrastructure guards and the SYSTEM-include rule"
```

---

## Spec coverage

| Spec item | Task |
|---|---|
| M1 `WriteBatch` (14 copies) | Task 1 Steps 1-7 |
| M1 `WriteExpertArrow` + `ExpertArrowPath` (13 copies, glm padding, alice dir) | Task 1 Steps 1-9 |
| M1 `QuantizeSlice` (12 copies, two overloads incl. E8M0) | Task 2 |
| M1 RED tests (`test_converter_helpers`, `test_nvfp4_pack` cases) | Tasks 1-2 |
| M1 `SaveExpertToArrow` delegation | Task 1 Step 5 |
| M2 dependency-path rule | Task 3 (Steps 1, 4) |
| M2 single-definition rule | Task 3 (Steps 1, 5) |
| M2 arch-literal rule | Task 3 (Steps 1, 6) |
| M2 scanner unit tests + tree assertions + mutation proof | Task 3 |
| M3 `-Wpedantic` gating | Task 4 Step 2 |
| M3 SYSTEM includes | Task 4 Step 3 |
| M3 measurement + acceptance (0 ours, ≤5 %) | Task 4 Steps 4-5 |
| M3 suites green | Task 4 Step 6 |
| Docs (AGENTS, test-lifecycle, gotchas, repo-map) | Task 5 |
| Acceptance: fast 134/134, convert 37/37, clean tree | Task 5 Step 5 |
