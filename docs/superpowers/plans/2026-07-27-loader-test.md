# Loader Round-Trip Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Catch2 tests that verify `lsxcommon::Loader::LoadExpertFromArrow` correctly reads `.arrow` expert files via synthetic data round-trip.

**Architecture:** Generate `PackedExpert` in-memory → `ExpertSchema::MakeBatch` → `WriteRecordBatchToFile` → `Loader::LoadExpertFromArrow` → compare all weight/scale fields. Two test cases: normal data and empty-field edge case.

**Tech Stack:** Catch2 v3.7, Apache Arrow 25 IPC File I/O, C++20, lsxcommon (ExpertSchema + Loader)

---

## File Structure

- Create: `tests/test_loader.cpp` — contains both test cases
- Modify: `tests/CMakeLists.txt` — add `test_loader` target

---

### Task 1: Setup test target in CMakeLists

**Files:**
- Modify: `tests/CMakeLists.txt`

- [ ] **Add test_loader target after existing test_core block**

Edit `tests/CMakeLists.txt`, add after the `add_test(NAME test_core ...)` line (line 19):

```cmake
# ── Тест round-trip загрузчика ──────────────────────────────────────────
add_executable(test_loader test_loader.cpp)
target_link_libraries(test_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_loader>"
    )
    add_custom_command(TARGET test_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_loader>"
    )
endif()

add_test(NAME test_loader COMMAND test_loader)
```

- [ ] **Build to verify CMakeLists parses**

Run:
```bash
cd D:\Projects\GitHub\logestix
cmake -S . -B build -G "Visual Studio 18 2026" -DBUILD_TESTING=ON
```

Expected: success (no CMake errors)

---

### Task 2: Write first failing test — round-trip with synthetic data

**Files:**
- Create: `tests/test_loader.cpp`

- [ ] **Write test file with first test case**

Create `tests/test_loader.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/schema.h>
#include <lsxcommon/loader.h>
#include <arrow/api.h>
#include <filesystem>

namespace fs = std::filesystem;

static lsxcommon::PackedExpert MakeTestExpert()
{
    lsxcommon::PackedExpert e;

    e.w1_packed.resize(1024);
    e.w2_packed.resize(1024);
    e.w3_packed.resize(512);
    for (int i = 0; i < 1024; ++i) {
        e.w1_packed[i] = static_cast<uint8_t>(i);
        e.w2_packed[i] = static_cast<uint8_t>(255 - i);
    }
    for (int i = 0; i < 512; ++i)
        e.w3_packed[i] = (i % 2 == 0) ? uint8_t(0xAA) : uint8_t(0xBB);

    e.w1_scales = {0.5f, 1.0f, 1.5f, 2.0f};
    e.w2_scales = {0.25f, 0.75f};
    e.w3_scales = {1.0f};

    return e;
}

static fs::path TempExpertPath()
{
    return fs::temp_directory_path() / "logestix_test_expert.arrow";
}

TEST_CASE("Loader round-trip with synthetic PackedExpert", "[loader]")
{
    auto expert = MakeTestExpert();

    auto batch_result = lsxcommon::ExpertSchema::MakeBatch(expert);
    REQUIRE(batch_result.ok());
    auto batch = *batch_result;
    REQUIRE(batch->ValidateFull().ok());

    auto path = TempExpertPath();
    auto st = lsxcommon::WriteRecordBatchToFile(batch, path);
    REQUIRE(st.ok());

    lsxcommon::Loader loader;
    auto load_result = loader.LoadExpertFromArrow(path.string());
    REQUIRE(load_result.ok());
    auto loaded = *load_result;

    CHECK(loaded.w1_weight_size == 1024);
    CHECK(loaded.w2_weight_size == 1024);
    CHECK(loaded.w3_weight_size == 512);
    CHECK(loaded.w1_scale_count == 4);
    CHECK(loaded.w2_scale_count == 2);
    CHECK(loaded.w3_scale_count == 1);

    CHECK(memcmp(loaded.w1_weight, expert.w1_packed.data(), 1024) == 0);
    CHECK(memcmp(loaded.w2_weight, expert.w2_packed.data(), 1024) == 0);
    CHECK(memcmp(loaded.w3_weight, expert.w3_packed.data(), 512) == 0);

    for (int64_t i = 0; i < 4; ++i)
        CHECK(loaded.w1_scale[i] == expert.w1_scales[static_cast<size_t>(i)]);
    for (int64_t i = 0; i < 2; ++i)
        CHECK(loaded.w2_scale[i] == expert.w2_scales[static_cast<size_t>(i)]);
    for (int64_t i = 0; i < 1; ++i)
        CHECK(loaded.w3_scale[i] == expert.w3_scales[static_cast<size_t>(i)]);

    fs::remove(path);
}
```

- [ ] **Build and run the test**

```bash
cd D:\Projects\GitHub\logestix
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug -R test_loader
```

Expected: test passes (Loader code already exists, this verifies it against synthetic data)

---

### Task 3: Write second test case — empty fields

**Files:**
- Modify: `tests/test_loader.cpp`

- [ ] **Add second TEST_CASE for empty w1 fields**

Append to `tests/test_loader.cpp`:

```cpp
TEST_CASE("Load expert with empty w1 fields", "[loader][edge]")
{
    auto expert = MakeTestExpert();
    expert.w1_packed.clear();
    expert.w1_scales.clear();

    auto batch_result = lsxcommon::ExpertSchema::MakeBatch(expert);
    REQUIRE(batch_result.ok());
    auto batch = *batch_result;
    REQUIRE(batch->ValidateFull().ok());

    auto path = TempExpertPath();
    REQUIRE(lsxcommon::WriteRecordBatchToFile(batch, path).ok());

    lsxcommon::Loader loader;
    auto load_result = loader.LoadExpertFromArrow(path.string());
    REQUIRE(load_result.ok());
    auto loaded = *load_result;

    CHECK(loaded.w1_weight_size == 0);
    CHECK(loaded.w1_scale_count == 0);

    CHECK(loaded.w2_weight_size == 1024);
    CHECK(loaded.w3_weight_size == 512);
    CHECK(loaded.w2_scale_count == 2);
    CHECK(loaded.w3_scale_count == 1);

    CHECK(memcmp(loaded.w2_weight, expert.w2_packed.data(), 1024) == 0);
    CHECK(memcmp(loaded.w3_weight, expert.w3_packed.data(), 512) == 0);

    fs::remove(path);
}
```

- [ ] **Build and run both tests**

```bash
cd D:\Projects\GitHub\logestix
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
ctest --test-dir build --output-on-failure -C Debug -R test_loader
```

Expected: both test cases pass

---

### Task 4: Run full test suite to confirm no regressions

- [ ] **Run all tests**

```bash
cd D:\Projects\GitHub\logestix
ctest --test-dir build --output-on-failure -C Debug
```

Expected: 6 test targets pass (test_core, test_core, test_fabric, test_forge, test_taskmosaic, test_loader, test_cli)

- [ ] **Commit**

```bash
git add tests/test_loader.cpp tests/CMakeLists.txt docs/superpowers/specs/2026-07-27-loader-test-design.md docs/superpowers/plans/2026-07-27-loader-test.md
git commit -m "test: add loader round-trip tests with synthetic data"
```
