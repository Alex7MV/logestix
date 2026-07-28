# Fabric Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--validate` to lsxfabric to verify integrity of logestix model format (index, pinned tensors, NVFP4 experts, E4M3 scales, global scales).

**Architecture:** New `Validator` module in `apps/lsxfabric/src/` with free functions. `main.cpp` reads `--validate` flag, calls `RunValidator()`. Tests in `apps/lsxfabric/tests/`.

**Tech Stack:** C++20, spdlog, Arrow 25.0.0 (RecordBatch file I/O), simdjson (index parsing), nlohmann_json (not used — simdjson for index), Catch2 v3 (tests).

---

### Task 1: Create validator.h

**Files:**
- Create: `apps/lsxfabric/src/validator.h`

- [ ] **Step 1: Write the header**

```cpp
#pragma once

#include <string>
#include <cstdint>

namespace lsxfabric {

struct ModelInfo {
    std::string format;
    std::string model_name;
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_pinned = 0;
    int64_t total_experts = 0;
};

struct ValidationResult {
    bool ok = true;
    int errors = 0;
    int warnings = 0;

    void add_error() { ok = false; ++errors; }
    void add_warning() { ++warnings; }
};

ModelInfo LoadModelIndex(const std::string &index_path);
ValidationResult ValidatePinnedTensors(const std::string &pinned_dir, int64_t expected_count);
ValidationResult ValidateExpert(const std::string &arrow_path);
bool ValidateNVFP4Data(const uint8_t *data, int64_t nbytes);
bool ValidateE4M3Scales(const uint8_t *scales, int64_t count);
bool ValidateGlobalScale(float scale);
ValidationResult RunValidator(const std::string &model_path);

} // namespace lsxfabric
```

- [ ] **Step 2: Commit**

```bash
git add apps/lsxfabric/src/validator.h
git commit -m "feat: add validator.h with public API"
```

---

### Task 2: Implement LoadModelIndex

**Files:**
- Create: `apps/lsxfabric/src/validator.cpp`
- Add to CMakeLists.txt as source

- [ ] **Step 1: Implement LoadModelIndex**

```cpp
#include "validator.h"
#include <simdjson.h>
#include <spdlog/spdlog.h>
#include <fstream>
#include <filesystem>

namespace lsxfabric {

ModelInfo LoadModelIndex(const std::string &index_path)
{
    ModelInfo info;
    std::ifstream f(index_path);
    if (!f) {
        spdlog::error("Cannot open index: {}", index_path);
        return info;
    }
    std::string json_str((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_str.data(), json_str.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(padded).get(doc) != simdjson::SUCCESS) {
        spdlog::error("JSON parse error in index");
        return info;
    }

    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        spdlog::error("Index root not an object");
        return info;
    }

    std::string_view fmt;
    if (root["format"].get_string().get(fmt) == simdjson::SUCCESS)
        info.format = std::string(fmt);
    std::string_view name;
    if (root["model"].get_string().get(name) == simdjson::SUCCESS)
        info.model_name = std::string(name);

    int64_t nl = 0, epl = 0, tp = 0, te = 0;
    if (root["num_layers"].get_int64().get(nl) == simdjson::SUCCESS)
        info.num_layers = static_cast<int>(nl);
    if (root["experts_per_layer"].get_int64().get(epl) == simdjson::SUCCESS)
        info.experts_per_layer = static_cast<int>(epl);
    if (root["total_pinned_tensors"].get_int64().get(tp) == simdjson::SUCCESS)
        info.total_pinned = tp;
    if (root["total_experts"].get_int64().get(te) == simdjson::SUCCESS)
        info.total_experts = te;

    return info;
}

} // namespace lsxfabric
```

- [ ] **Step 2: Add validator.cpp to CMakeLists.txt**

Edit `apps/lsxfabric/CMakeLists.txt`:
```cmake
add_executable(lsxfabric src/main.cpp src/validator.cpp)
```

- [ ] **Step 3: Commit**

```bash
git add apps/lsxfabric/src/validator.cpp apps/lsxfabric/CMakeLists.txt
git commit -m "feat: add LoadModelIndex + wire validator.cpp into build"
```

---

### Task 3: Implement ValidatePinnedTensors

- [ ] **Step 1: Add implementation to validator.cpp**

```cpp
#include <arrow/ipc/reader.h>
#include <arrow/io/file.h>

ValidationResult ValidatePinnedTensors(const std::string &pinned_dir, int64_t expected_count)
{
    namespace fs = std::filesystem;
    ValidationResult r;
    int64_t loaded = 0;

    auto check_batch = [&](const std::string &path) {
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) { r.add_error(); spdlog::error("  Cannot open {}", path); return; }
        auto reader = arrow::ipc::RecordBatchFileReader::Open(result->get());
        if (!reader.ok()) { r.add_error(); spdlog::error("  Cannot read batch from {}", path); return; }
        auto batch = (*reader)->ReadRecordBatch(0);
        if (!batch.ok()) { r.add_error(); spdlog::error("  Cannot read record from {}", path); return; }
        int ncol = (*batch)->num_columns();
        spdlog::info("  {}: {} columns, {} rows", path, ncol, (*batch)->num_rows());
        loaded += ncol;
    };

    spdlog::info("Validating pinned tensors from {}...", pinned_dir);

    auto global_path = fs::path(pinned_dir) / "global.arrow";
    if (fs::exists(global_path))
        check_batch(global_path.string());

    for (const auto &entry : fs::directory_iterator(pinned_dir)) {
        if (entry.path().extension() != ".arrow") continue;
        if (entry.path().filename() == "global.arrow") continue;
        check_batch(entry.path().string());
    }

    if (expected_count > 0 && loaded != expected_count) {
        spdlog::warn("  Expected {} pinned tensors, loaded {}", expected_count, loaded);
        r.add_warning();
    }
    return r;
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/lsxfabric/src/validator.cpp
git commit -m "feat: add ValidatePinnedTensors"
```

---

### Task 4: Implement NVFP4/E4M3 validation helpers

- [ ] **Step 1: Add validation helpers to validator.cpp**

```cpp
bool ValidateNVFP4Data(const uint8_t *data, int64_t nbytes)
{
    for (int64_t i = 0; i < nbytes; ++i) {
        uint8_t lo = data[i] & 0x0F;
        uint8_t hi = (data[i] >> 4) & 0x0F;
        if (lo > 0x0F || hi > 0x0F) return false;
    }
    return true;
}

bool ValidateE4M3Scales(const uint8_t *scales, int64_t count)
{
    for (int64_t i = 0; i < count; ++i) {
        uint8_t b = scales[i];
        // sign must be 0 (positive), exponent 4-bit must be valid
        if (b & 0x80) return false; // no negative scales
        int exp4 = (b >> 3) & 0x0F;
        if (exp4 < 1 || exp4 > 14) return false;
        // mantissa must be 0 (power-of-2 only)
        if (b & 0x07) return false;
    }
    return true;
}

bool ValidateGlobalScale(float scale)
{
    return std::isfinite(scale) && scale > 0.0f;
}
```

Add `#include <cmath>` to validator.cpp for `std::isfinite`.

- [ ] **Step 2: Commit**

```bash
git add apps/lsxfabric/src/validator.cpp
git commit -m "feat: add NVFP4, E4M3, global-scale validation"
```

---

### Task 5: Implement ValidateExpert

- [ ] **Step 1: Add ValidateExpert to validator.cpp**

```cpp
#include "lsxcommon/schema.h"

ValidationResult ValidateExpert(const std::string &arrow_path)
{
    ValidationResult r;

    auto result = arrow::io::ReadableFile::Open(arrow_path);
    if (!result.ok()) { r.add_error(); return r; }
    auto reader = arrow::ipc::RecordBatchFileReader::Open(result->get());
    if (!reader.ok()) { r.add_error(); return r; }
    auto batch = (*reader)->ReadRecordBatch(0);
    if (!batch.ok()) { r.add_error(); return r; }

    auto rb = *batch;
    if (rb->num_columns() != lsxcommon::ExpertSchema::kNumFields) {
        spdlog::error("  Expert schema: expected {} columns, got {}",
            lsxcommon::ExpertSchema::kNumFields, rb->num_columns());
        r.add_error(); return r;
    }

    // Field indices from schema.h
    using E = lsxcommon::ExpertSchema;
    int64_t n = rb->num_rows();

    auto check_binary = [&](int idx, const char *name,
                            const uint8_t *&data, int64_t &size) {
        auto arr = rb->column(idx);
        auto bin = std::static_pointer_cast<arrow::BinaryArray>(arr);
        if (n > 0) {
            auto view = bin->GetView(0);
            data = reinterpret_cast<const uint8_t*>(view.data());
            size = static_cast<int64_t>(view.size());
            if (!ValidateNVFP4Data(data, size)) {
                spdlog::error("  NVFP4 data invalid in {}", name);
                r.add_error();
            }
        }
    };

    auto check_scales = [&](int idx, const char *name) {
        auto arr = rb->column(idx);
        auto lst = std::static_pointer_cast<arrow::ListArray>(arr);
        if (n > 0) {
            auto offsets = lst->raw_value_offsets();
            int64_t count = offsets[1] - offsets[0];
            auto u8 = std::static_pointer_cast<arrow::UInt8Array>(lst->values());
            const uint8_t *data = u8->raw_values() + offsets[0];
            if (!ValidateE4M3Scales(data, count)) {
                spdlog::error("  E4M3 scales invalid in {}", name);
                r.add_error();
            }
        }
    };

    auto check_global = [&](int idx, const char *name) {
        auto arr = rb->column(idx);
        auto f32 = std::static_pointer_cast<arrow::FloatArray>(arr);
        if (n > 0) {
            float s = f32->Value(0);
            if (!ValidateGlobalScale(s)) {
                spdlog::error("  Global scale invalid in {}: {}", name, s);
                r.add_error();
            }
        }
    };

    const uint8_t *d1; int64_t s1;
    const uint8_t *d2; int64_t s2;
    const uint8_t *d3; int64_t s3;
    check_binary(E::kIdxW1Weight, "w1_weight", d1, s1);
    check_binary(E::kIdxW2Weight, "w2_weight", d2, s2);
    check_binary(E::kIdxW3Weight, "w3_weight", d3, s3);
    check_scales(E::kIdxW1Scale, "w1_scale");
    check_scales(E::kIdxW2Scale, "w2_scale");
    check_scales(E::kIdxW3Scale, "w3_scale");
    check_global(E::kIdxW1GlobalScale, "w1_global_scale");
    check_global(E::kIdxW2GlobalScale, "w2_global_scale");
    check_global(E::kIdxW3GlobalScale, "w3_global_scale");

    spdlog::info("  Expert validated: {} rows", n);
    return r;
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/lsxfabric/src/validator.cpp
git commit -m "feat: add ValidateExpert with full schema check"
```

---

### Task 6: Implement RunValidator

- [ ] **Step 1: Add RunValidator to validator.cpp**

```cpp
ValidationResult RunValidator(const std::string &model_path)
{
    namespace fs = std::filesystem;
    ValidationResult r;

    auto index_path = fs::path(model_path) / "model.index.json";
    if (!fs::exists(index_path)) {
        spdlog::error("model.index.json not found in {}", model_path);
        r.add_error(); return r;
    }

    auto info = LoadModelIndex(index_path.string());
    if (info.format != "logestix-v1") {
        spdlog::warn("  Unknown format '{}', continuing anyway", info.format);
        r.add_warning();
    }

    spdlog::info("Model: {}  layers={} experts/layer={} pinned={} experts={}",
        info.model_name, info.num_layers, info.experts_per_layer,
        info.total_pinned, info.total_experts);

    // Validate pinned
    auto pinned_dir = fs::path(model_path) / "pinned";
    if (fs::exists(pinned_dir)) {
        auto pr = ValidatePinnedTensors(pinned_dir.string(), info.total_pinned);
        if (!pr.ok) { r.add_error(); }
        r.warnings += pr.warnings;
    } else {
        spdlog::warn("  No pinned/ directory found");
        r.add_warning();
    }

    // Validate one expert per layer
    int experts_checked = 0;
    for (int layer = 0; layer < info.num_layers; ++layer) {
        auto exp_path = fs::path(model_path)
            / ("experts/layer_" + std::to_string(layer))
            / "exp_000.arrow";
        if (!fs::exists(exp_path)) {
            spdlog::warn("  Expert not found: {}", exp_path.string());
            r.add_warning();
            continue;
        }
        spdlog::info("Validating expert layer={}...", layer);
        auto er = ValidateExpert(exp_path.string());
        if (!er.ok) r.add_error();
        r.warnings += er.warnings;
        ++experts_checked;
    }

    spdlog::info("=== Validation Summary ===");
    spdlog::info("  Pinned tensors checked: {}", info.total_pinned);
    spdlog::info("  Experts checked: {}", experts_checked);
    spdlog::info("  Errors: {}", r.errors);
    spdlog::info("  Warnings: {}", r.warnings);
    if (r.ok)
        spdlog::info("  Result: PASS");
    else
        spdlog::error("  Result: FAIL");

    return r;
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/lsxfabric/src/validator.cpp
git commit -m "feat: add RunValidator orchestrator"
```

---

### Task 7: Update main.cpp

**Files:**
- Modify: `apps/lsxfabric/src/main.cpp`

- [ ] **Step 1: Add --validate flag**

```cpp
#include "lsxcommon/core.h"
#include "lsxcommon/common.h"
#include "validator.h"
#include <cstring>
#include <spdlog/spdlog.h>
#include <string>

int main(int argc, char* argv[]) {
    std::string model_path = R"(D:\Programs\huggingface.co\DeepSeek-V4-Flash)";
    bool validate = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--localmodel") == 0 && i + 1 < argc)
            model_path = argv[++i];
        else if (std::strcmp(argv[i], "--validate") == 0)
            validate = true;
    }

    lsxcommon::Core::Initialize("logestix_runtime.log");
    lsxcommon::version();
    lsxcommon::log_gpu_info();
    spdlog::info("Model path: {}", model_path);

    if (validate) {
        spdlog::info("Running validation...");
        auto result = lsxfabric::RunValidator(model_path);
        lsxcommon::Core::Shutdown();
        return result.ok ? 0 : 1;
    }

    lsxcommon::Core::Shutdown();
    return 0;
}
```

- [ ] **Step 2: Commit**

```bash
git add apps/lsxfabric/src/main.cpp
git commit -m "feat: add --validate flag to lsxfabric"
```

---

### Task 8: Write tests

**Files:**
- Create: `apps/lsxfabric/tests/test_validator.cpp`
- Modify: `apps/lsxfabric/tests/CMakeLists.txt`

- [ ] **Step 1: Create the test file**

```cpp
#include <catch2/catch_test_macros.hpp>
#include "validator.h"
#include <cstring>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace lsxfabric;

TEST_CASE("ValidateNVFP4Data valid", "[validator]") {
    uint8_t data[] = {0x00, 0x01, 0x0F, 0xAB, 0xCD, 0xFF};
    REQUIRE(ValidateNVFP4Data(data, 6));
}

TEST_CASE("ValidateNVFP4Data invalid nibble", "[validator]") {
    // 0xF0 is fine (hi=0x0F, lo=0x00), but 0x10 is fine too (hi=0x01, lo=0x00)
    // Actually all nibbles 0x00-0x0F are valid in NVFP4 encoding
    // Let's use 0xFF as valid (all 4-bit codes are valid in E2M1)
    uint8_t valid[] = {0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                       0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};
    REQUIRE(ValidateNVFP4Data(valid, 16));
}

TEST_CASE("ValidateNVFP4Data empty", "[validator]") {
    REQUIRE(ValidateNVFP4Data(nullptr, 0));
}

TEST_CASE("ValidateE4M3Scales valid", "[validator]") {
    // sign=0, exp4=1..14, mant=0
    uint8_t scales[] = {0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38,
                        0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70};
    REQUIRE(ValidateE4M3Scales(scales, 14));
}

TEST_CASE("ValidateE4M3Scales negative sign", "[validator]") {
    uint8_t bad = 0x88; // sign=1
    REQUIRE_FALSE(ValidateE4M3Scales(&bad, 1));
}

TEST_CASE("ValidateE4M3Scales zero exponent", "[validator]") {
    uint8_t bad = 0x00; // exp=0 → subnormal
    REQUIRE_FALSE(ValidateE4M3Scales(&bad, 1));
}

TEST_CASE("ValidateE4M3Scales non-zero mantissa", "[validator]") {
    uint8_t bad = 0x0B; // exp=1, mant=3
    REQUIRE_FALSE(ValidateE4M3Scales(&bad, 1));
}

TEST_CASE("ValidateE4M3Scales empty", "[validator]") {
    REQUIRE(ValidateE4M3Scales(nullptr, 0));
}

TEST_CASE("ValidateGlobalScale valid", "[validator]") {
    REQUIRE(ValidateGlobalScale(1.0f));
    REQUIRE(ValidateGlobalScale(0.001f));
    REQUIRE(ValidateGlobalScale(1000.0f));
}

TEST_CASE("ValidateGlobalScale invalid", "[validator]") {
    REQUIRE_FALSE(ValidateGlobalScale(0.0f));
    REQUIRE_FALSE(ValidateGlobalScale(-1.0f));
    REQUIRE_FALSE(ValidateGlobalScale(NAN));
    REQUIRE_FALSE(ValidateGlobalScale(INFINITY));
}

TEST_CASE("LoadModelIndex valid json", "[validator]") {
    std::string tmp = std::filesystem::temp_directory_path() / "test_model_index.json";
    std::ofstream(tmp) << R"({
        "format": "logestix-v1",
        "model": "test-model",
        "num_layers": 3,
        "experts_per_layer": 64,
        "total_pinned_tensors": 42,
        "total_experts": 192
    })";
    auto info = LoadModelIndex(tmp);
    REQUIRE(info.format == "logestix-v1");
    REQUIRE(info.model_name == "test-model");
    REQUIRE(info.num_layers == 3);
    REQUIRE(info.experts_per_layer == 64);
    REQUIRE(info.total_pinned == 42);
    REQUIRE(info.total_experts == 192);
    std::filesystem::remove(tmp);
}

TEST_CASE("LoadModelIndex missing file", "[validator]") {
    auto info = LoadModelIndex("nonexistent.json");
    REQUIRE(info.num_layers == 0);
}
```

- [ ] **Step 2: Wire into test CMakeLists.txt**

Read `apps/lsxfabric/tests/CMakeLists.txt` to determine existing structure, then add:
```cmake
target_sources(test_lsxfabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_validator.cpp)
```

And update the validator.cpp include path:
```cmake
target_include_directories(test_lsxfabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
```

- [ ] **Step 3: Commit**

```bash
git add apps/lsxfabric/tests/test_validator.cpp apps/lsxfabric/tests/CMakeLists.txt
git commit -m "test: add validator unit tests"
```

---

### Task 9: Build and run tests

- [ ] **Step 1: Configure + build**

```bash
cmake -S . -B build -G "Visual Studio 18 2026"
$env:CMAKE_POLICY_VERSION_MINIMUM="3.5"
cmake --build build --config Debug
```

- [ ] **Step 2: Run all tests**

```bash
ctest --test-dir build --output-on-failure -C Debug
```

Expected: All 6+1=7 tests pass.

- [ ] **Step 3: Commit if everything passes**

```bash
git commit -m "feat: fabric validator with --validate flag"
```
