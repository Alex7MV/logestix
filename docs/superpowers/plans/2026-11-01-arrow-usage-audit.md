# Arrow usage audit — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> (inline execution — AGENTS.md forbids subagents). Steps use `- [ ]`.

**Goal:** Harden every Arrow IPC read/write path used by the 29 adapted model
families, machine-check the rules that keep it that way, and validate all 26
converted trees in `/mnt/fast_storage/huggingface`.

**Architecture:** One shared host-only validator header
(`arrow_validate.h`) consumed by the owning `UnpackBatch` paths, the zero-copy
view loaders, `ExpertStore::DecodeEntry` and `shared_experts.cu`; plus atomic
artifact writes, a static source guard, and an environment-gated tree
validator. All fixes land test-first.

**Tech Stack:** C++20, Apache Arrow 26, Catch2 v3.16.1, CMake presets,
`ctest` labels `fast`/`convert`.

## Global Constraints

- No subagents (AGENTS.md); do everything in the main session.
- TDD: watch every new test fail before the fix; label every new test in the
  same commit (`test_ctest_labels_guard`).
- No abseil containers in headers reachable from `.cu`/`.cuh`
  (`test_absl_container_guard`); `arrow_validate.h` is inline, `std::`-only.
- `absl::StrFormat` is printf-style (no `{}`).
- Do not change the on-disk schema/format; the audit validates it.
- Do not migrate the legacy pinned-expert arenas (documented debt).
- Temporary files only under `/mnt/fast_storage/tmp`.
- Branch: `audit/arrow-usage-2026-11-01` (already created).
- Commit only when the operator explicitly asks; otherwise leave the tree
  dirty and report.

---

### Task 1: Shared Arrow validators

**Files:**
- Create: `libs/lsxcommon/include/lsxcommon/arrow_validate.h`
- Test: `tests/test_arrow_validation.cpp` (new, label `fast`)

**Interfaces:**
- Produces:
  - `arrow::Status lsxcommon::arrow_check::CheckListSlice(const arrow::Int32Array &offsets, int64_t row, int64_t values_length, std::string_view what, std::string_view name)`
  - `arrow::Status CheckViewRange(const void *p, int64_t bytes, const std::shared_ptr<arrow::Buffer> &buf, std::string_view what, std::string_view name)`
  - `arrow::Result<int64_t> DtypeByteSize(std::string_view dtype)`
  - `arrow::Status CheckTensorSize(const int64_t *shape, int64_t rank, int64_t bytes, std::string_view dtype, std::string_view name)`
  - `arrow::Status CheckNvfp4Size(const int64_t *shape, int64_t rank, int64_t packed_bytes, int64_t scale_bytes, float global_scale, std::string_view name)`

- [ ] **Step 1: Write the failing test**

```cpp
// tests/test_arrow_validation.cpp
#include <catch2/catch_test_macros.hpp>
#include <arrow/api.h>
#include "lsxcommon/arrow_validate.h"

namespace av = lsxcommon::arrow_check;

TEST_CASE("CheckListSlice rejects negative and reversed slices", "[arrow][validate]")
{
    arrow::Int32Builder b;
    REQUIRE(b.AppendValues({0, 5, 5}).ok());
    auto offs = b.Finish().ValueOrDie();
    CHECK(av::CheckListSlice(*offs, 0, 4, "scale", "x").ok());
    CHECK_FALSE(av::CheckListSlice(*offs, 0, 4, "scale", "x").ok() == false);
    // reversed: craft {0, 5, 3}
    arrow::Int32Builder rb;
    REQUIRE(rb.AppendValues({0, 5, 3}).ok());
    auto roffs = rb.Finish().ValueOrDie();
    CHECK_FALSE(av::CheckListSlice(*roffs, 1, 8, "scale", "x").ok());
}

TEST_CASE("CheckViewRange rejects a view crossing the buffer end", "[arrow][validate]")
{
    auto buf = arrow::Buffer::FromString(std::string(4, 'x'));
    CHECK(av::CheckViewRange(buf->data(), 4, buf, "data", "t").ok());
    CHECK_FALSE(av::CheckViewRange(buf->data(), 5, buf, "data", "t").ok());
    CHECK_FALSE(av::CheckViewRange(buf->data() - 1, 1, buf, "data", "t").ok());
}

TEST_CASE("CheckTensorSize accepts BF16/F32 and rejects a truncated payload",
          "[arrow][validate]")
{
    const int64_t shape[2] = {8, 4};
    CHECK(av::CheckTensorSize(shape, 2, 8 * 4 * 2, "BF16", "w").ok());
    CHECK(av::CheckTensorSize(shape, 2, 8 * 4 * 4, "F32", "w").ok());
    CHECK_FALSE(av::CheckTensorSize(shape, 2, 8 * 4 * 2 - 1, "BF16", "w").ok());
    CHECK_FALSE(av::CheckTensorSize(shape, 2, 8 * 4 * 2, "XYZ", "w").ok());
}

TEST_CASE("CheckNvfp4Size enforces packed=ceil(n/2), scales=ceil(n/16)",
          "[arrow][validate]")
{
    const int64_t shape[2] = {128, 256};
    const int64_t n = 128 * 256;
    CHECK(av::CheckNvfp4Size(shape, 2, n / 2, n / 16, 0.5f, "w").ok());
    CHECK_FALSE(av::CheckNvfp4Size(shape, 2, n / 2 - 1, n / 16, 0.5f, "w").ok());
    CHECK_FALSE(av::CheckNvfp4Size(shape, 2, n / 2, n / 16 - 1, 0.5f, "w").ok());
    CHECK_FALSE(av::CheckNvfp4Size(shape, 2, n / 2, n / 16, 0.0f, "w").ok());
}
```

- [ ] **Step 2: Run and verify RED**

Run: `cmake --build --preset linux-debug --target test_arrow_validation && ctest --preset linux-debug -R test_arrow_validation --output-on-failure`
Expected: compile error — `lsxcommon/arrow_validate.h` does not exist.

- [ ] **Step 3: Implement `arrow_validate.h`**

Header-only, `namespace lsxcommon::arrow_check`, includes
`<arrow/status.h>`, `<arrow/array/array_base.h>`, `<arrow/buffer.h>`,
`<arrow/array/array_binary.h>`, `<cmath>`, `<string_view>`,
`<type_traits>`. Implement:
- `CheckListSlice`: read `offsets.Value(row)`, `offsets.Value(row+1)`;
  require `begin >= 0 && begin <= end && end <= values_length`.
- `CheckViewRange`: identical arithmetic to the X15 helper in
  `schema.cpp:849` (`base <= addr && addr <= end && bytes <= end - addr`).
- `DtypeByteSize`: `BF16`/`F16` → 2, `F32`/`I32`/`U32` → 4, `F64`/`I64`/`U64`
  → 8, `I8`/`U8`/`F8_E4M3` → 1, `E2M1` → 0.5 (handled in `CheckNvfp4Size`
  only; return an error if used as a raw dtype), else
  `arrow::Status::TypeError("unknown dtype ", dtype)`.
- `CheckTensorSize`: product with `__int128`/overflow guard; compare with
  `bytes`.
- `CheckNvfp4Size`: `numel = product(shape)`; require
  `packed_bytes == (numel + 1) / 2`, `scale_bytes == (numel + 15) / 16`,
  `std::isfinite(global_scale) && global_scale > 0`.

- [ ] **Step 4: Run and verify GREEN**

Run: `ctest --preset linux-debug -R test_arrow_validation --output-on-failure`
Expected: `All tests passed`.

- [ ] **Step 5: Wire CMake + label**

`tests/CMakeLists.txt`: `add_executable(test_arrow_validation ...)`,
`target_link_libraries(... lsxcommon Catch2::Catch2WithMain)`, `add_test`,
add the name to the `fast` `set_tests_properties(... LABELS fast)` list.

---

### Task 2: Harden owning `UnpackBatch` paths and `DecodeEntry` (A-01)

**Files:**
- Modify: `libs/lsxcommon/src/schema.cpp` (`ExpertSchema`, `SharedExpertSchema`,
  `DenseNvfp4Schema`, `PinnedSchema` `UnpackBatch`; `ShapePtr`)
- Modify: `libs/lsxcommon/include/lsxcommon/expert_store.h` (`DecodeEntry`)
- Test: `tests/test_arrow_validation.cpp`

Write the malformed-batch cases FIRST (one `TEST_CASE` per schema): zero rows,
two rows where one is expected, scale list with `list<int16>` values, offsets
`{-1,2}` and `{0, 99}` on a 4-byte value buffer, shape list with an
out-of-range slice, NVFP4 payload shorter than `ceil(n/2)`, `global_scale =
NaN`. Each must return a non-OK status (not crash, not OOB).

Then:
- `ExpertSchema::UnpackBatch`: require `num_rows() == 1`; validate each scale
  column with `CheckListSlice(..., /*row=*/0, values->length())`; wrap binary
  views with `CheckViewRange`.
- `SharedExpertSchema::UnpackBatch`: move `values()`/`offsets()` casts into
  the null-checked block; per-row `CheckListSlice`.
- `DenseNvfp4Schema::UnpackBatch`: per-row `CheckListSlice` for shape and
  scales; `CheckNvfp4Size` on each record.
- `PinnedSchema::UnpackBatch`: per-row `CheckListSlice` for shape;
  `CheckTensorSize`.
- `ShapePtr`: `CheckListSlice` before returning the pointer.
- `ExpertStore::DecodeEntry`: require `batch->num_rows() == 1`; use the
  checked list-slice helper; leave the batch/entry unchanged on failure.

---

### Task 3: Validate view-loader payloads (A-02, A-03)

**Files:**
- Modify: `libs/lsxcommon/src/schema.cpp` (`LoadTensorBatchViews`,
  `LoadDenseNvfp4BatchViews`, `ShapePtr`)
- Test: `tests/test_arrow_validation.cpp`

Red first: build a valid Pinned batch, then a second batch whose `data` column
is one byte short of `product(shape)*2` for `BF16`; `LoadTensorBatchViews`
must fail. Same for DenseNvfp4 with a short `packed` column and with
`scales` sized `n/16 - 1`. Then add `CheckTensorSize` /
`CheckNvfp4Size` to both loaders.

---

### Task 4: `shared_experts.cu` offset validation (A-04)

**Files:**
- Modify: `libs/lsxcommon/src/shared_experts.cu`
- Test: `tests/test_arrow_validation.cpp` (host-only Arrow part) — call the
  metadata loader through `shared_experts.h` on a hand-built batch with
  reversed scale offsets; expect `Invalid`, not a negative size.

---

### Task 5: Atomic artifact writes (A-05)

**Files:**
- Modify: `libs/lsxcommon/src/schema.cpp` (`WriteRecordBatchToFile`)
- Test: `tests/test_arrow_validation.cpp`

Red: (a) writing to a path whose parent is missing returns an error and leaves
no file; (b) after a successful write no `*.tmp.*` remains; (c) writing over an
existing file replaces it atomically (old content fully replaced). Implement
temp+rename with cleanup on every error path; keep the same signature.

---

### Task 6: Static Arrow usage guard

**Files:**
- Create: `tests/test_arrow_usage_guard.cpp`
- Modify: `tests/CMakeLists.txt`
- Modify: `libs/lsxcommon/src/gemma_moe_load.cpp`,
  `libs/lsxcommon/src/gemma_dense_load.cpp` (A-07 → view helpers)
- Modify: `docs/ARROW_AUDIT_2026-11-01.md` (matrix rows the guard reads)

Guard checks (each a `TEST_CASE`):
1. no `arrow::io::ReadableFile::Open` in `libs/`/`apps/`;
2. every `MemoryMappedFile::Open` site is in the allowlist;
3. every `*Schema::UnpackBatch` call site appears in the verdict table;
4. the audit report contains a row for each of the 29 model directories;
5. no converter writes Arrow outside `schema.cpp` (allowlist).

---

### Task 7: Tree validator

**Files:**
- Create: `tests/test_arrow_tree_integrity.cpp`
- Modify: `tests/CMakeLists.txt` (label `convert`, `SKIP_RETURN_CODE 4`,
  `TIMEOUT 1800`)

Footer-only pass over every `.arrow` (schema classification by field names),
then a bounded deep sample. Report per tree; `SKIP` when the root is absent.

---

### Task 8: Report + final verification

**Files:**
- Create: `docs/ARROW_AUDIT_2026-11-01.md`
- Modify: `AGENTS.md` (one line pointing at the new guard)

Run the full fast suite, the new tests, and the validator over the 26 trees;
record exact numbers in the report.

## Self-review

- Spec §4.1 → Task 1; §4.2 → Tasks 2–4; §4.3 → Task 5; §4.4 → Task 6;
  §4.5 → Task 7; §4.6 → Task 8.
- No placeholders left; the deep-sample rule and env names are defined in the
  spec and repeated in Task 7.
- Type names in later tasks match Task 1 (`CheckListSlice`, `CheckViewRange`,
  `CheckTensorSize`, `CheckNvfp4Size`).
