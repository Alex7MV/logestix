# lsxfabric Functional Audit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Do NOT dispatch subagents** (AGENTS.md forbids them).

**Goal:** Audit `lsxfabric` functionality with TDD — add green characterisation tests and hidden red spec tests, run real-vault `--validate` + real GPU generation, and produce `docs/LSXFABRIC_AUDIT.md`.

**Architecture:** Production code is frozen. Tests attach to the two existing CTest targets (`test_lsxfabric`, `test_fabric_cli`) via `target_sources`, so no new labels/targets enter the suite. Divergences are encoded as hidden Catch2 tests (`[.audit-spec]`) that stay out of the default run.

**Tech Stack:** C++20, CMake presets (`linux-debug`), Catch2 v3.16.0, Apache Arrow 25.0.1, simdjson, CUDA 13.4, RTX 5090.

## Global Constraints

- Branch: `audit/lsxfabric-functional-audit` (already checked out).
- Production code untouched: `apps/lsxfabric/src/validator.cpp`, `cli.cpp`, `main.cpp`, `apps/lsxfabric/CMakeLists.txt`.
- New tests only: `apps/lsxfabric/tests/*.cpp` + `target_sources` in `apps/lsxfabric/tests/CMakeLists.txt`.
- Every CTest target keeps its existing label; no new `add_test`.
- Hidden tests carry ONLY the `[.audit-spec]` tag (any hidden tag hides the test).
- GPU-exclusivity: integration Task 8 runs LAST, sequentially, one GPU workload at a time.
- Commit prefixes: `test:`, `docs:`. Commit only these files.
- Logging-state pattern: each test that exercises logging constructs a `CoreScope` locally (mirrors `test_validator.cpp`).
- Exact file paths and commands below; do not improvise names.

---

### Task 1: CLI green characterisation tests (S1)

**Files:**
- Create: `apps/lsxfabric/tests/test_fabric_cli_edges.cpp`
- Modify: `apps/lsxfabric/tests/CMakeLists.txt:22` (after the `add_executable(test_fabric_cli ...)` line)

**Interfaces:**
- Consumes: `lsxcommon::ParseFabricCliArgs(int, char**)` → `lsxfabric::FabricCliArgs{model_path, prompt, max_gen, validate, testprompt}` (`apps/lsxfabric/src/cli.h`).
- Produces: target `test_fabric_cli` gains a translation unit; no new symbols.

- [ ] **Step 1: Write the green tests**

Create `apps/lsxfabric/tests/test_fabric_cli_edges.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "cli.h"

#include <string>
#include <vector>

namespace {

std::vector<char *> MakeArgv(const std::vector<std::string> &storage) {
    std::vector<char *> argv;
    argv.reserve(storage.size());
    for (const std::string &s : storage)
        argv.push_back(const_cast<char *>(s.c_str()));
    return argv;
}

lsxfabric::FabricCliArgs Parse(const std::vector<std::string> &args) {
    std::vector<char *> argv = MakeArgv(args);
    return lsxfabric::ParseFabricCliArgs(static_cast<int>(argv.size()),
                                         argv.data());
}

} // namespace

// Current behaviour (green characterisation). See docs/LSXFABRIC_AUDIT.md.

TEST_CASE("fabric cli: flag without a value is silently ignored",
          "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--localmodel"}).model_path.find(
                "DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE(Parse({"lsxfabric", "--max-gen"}).max_gen == 64);
    REQUIRE(Parse({"lsxfabric", "--prompt"}).prompt ==
            "The capital of France is");
}

TEST_CASE("fabric cli: --prompt consumes the following flag token",
          "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--prompt", "--validate"});
    REQUIRE(a.prompt == "--validate");
    REQUIRE_FALSE(a.validate);
}

TEST_CASE("fabric cli: non-numeric --max-gen becomes 0", "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "abc"}).max_gen == 0);
    REQUIRE(Parse({"lsxfabric", "--max-gen", "12x"}).max_gen == 12);
}

TEST_CASE("fabric cli: negative --max-gen is accepted verbatim",
          "[lsxfabric][cli]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "-5"}).max_gen == -5);
}

TEST_CASE("fabric cli: repeated flags, last one wins", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a =
        Parse({"lsxfabric", "--max-gen", "5", "--max-gen", "9",
               "--localmodel", "/a", "--localmodel", "/b"});
    REQUIRE(a.max_gen == 9);
    REQUIRE(a.model_path == "/b");
}

TEST_CASE("fabric cli: a flag-looking value is taken literally",
          "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a =
        Parse({"lsxfabric", "--localmodel", "--validate", "/tmp/x"});
    REQUIRE(a.model_path == "--validate");
    REQUIRE_FALSE(a.validate);
    REQUIRE(a.max_gen == 64);
}

TEST_CASE("fabric cli: help flags are unknown and ignored", "[lsxfabric][cli]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--help", "-h"});
    REQUIRE(a.model_path.find("DeepSeek-V4-Flash") != std::string::npos);
    REQUIRE_FALSE(a.validate);
    REQUIRE_FALSE(a.testprompt);
}
```

- [ ] **Step 2: Register the file in CMake**

In `apps/lsxfabric/tests/CMakeLists.txt`, immediately after line 22
(`add_executable(test_fabric_cli test_fabric_cli.cpp ../src/cli.cpp)`), insert:

```cmake
target_sources(test_fabric_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_fabric_cli_edges.cpp)
```

- [ ] **Step 3: Build the target**

Run: `cmake --build --preset linux-debug --target test_fabric_cli`
Expected: builds clean.

- [ ] **Step 4: Run the green tests**

Run: `./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli`
Expected: all tests pass (including the pre-existing `test_fabric_cli.cpp` cases).

- [ ] **Step 5: Commit**

```bash
git add apps/lsxfabric/tests/test_fabric_cli_edges.cpp apps/lsxfabric/tests/CMakeLists.txt
git commit -m "test: characterise lsxfabric CLI parsing"
```

---

### Task 2: CLI red spec tests (S1 divergences, F4/F5)

**Files:**
- Modify: `apps/lsxfabric/tests/test_fabric_cli_edges.cpp` (append)

**Interfaces:**
- Consumes: same `Parse` helper defined in Task 1.
- Produces: hidden `[.audit-spec]` tests, run explicitly.

- [ ] **Step 1: Append the hidden spec tests**

Append to `apps/lsxfabric/tests/test_fabric_cli_edges.cpp`:

```cpp
// RED spec tests: the expected contract where current behaviour diverges.
// Hidden (a tag starting with '.'), so the default ctest run stays green.
// Run explicitly: ./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli "[.audit-spec]"

// FINDING F5: --max-gen must not turn a non-numeric value into 0.
TEST_CASE("fabric cli [spec]: --max-gen rejects a non-numeric value",
          "[.audit-spec]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "abc"}).max_gen == 64);
}

// FINDING F5: --max-gen must be positive.
TEST_CASE("fabric cli [spec]: --max-gen is bounded to a positive value",
          "[.audit-spec]") {
    REQUIRE(Parse({"lsxfabric", "--max-gen", "-5"}).max_gen >= 1);
}

// FINDING F4: --prompt must not swallow a following known flag.
TEST_CASE("fabric cli [spec]: --prompt does not swallow a known flag",
          "[.audit-spec]") {
    const lsxfabric::FabricCliArgs a = Parse({"lsxfabric", "--prompt", "--validate"});
    REQUIRE(a.prompt == "The capital of France is");
    REQUIRE(a.validate);
}
```

- [ ] **Step 2: Build and confirm hidden tests are excluded by default**

Run: `cmake --build --preset linux-debug --target test_fabric_cli && ./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli`
Expected: green (the hidden tests do not run).

- [ ] **Step 3: Confirm the red spec tests fail explicitly**

Run: `./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli "[.audit-spec]"`
Expected: 3 tests run, 3 FAIL (this is the intended evidence).

- [ ] **Step 4: Commit**

```bash
git add apps/lsxfabric/tests/test_fabric_cli_edges.cpp
git commit -m "test: add lsxfabric CLI spec tests (hidden red)"
```

---

### Task 3: Validator primitives green tests (S3)

**Files:**
- Create: `apps/lsxfabric/tests/test_validator_contract.cpp`
- Modify: `apps/lsxfabric/tests/CMakeLists.txt` (add `target_sources` for `test_lsxfabric`)

**Interfaces:**
- Consumes: `lsxfabric::ValidateNVFP4Data(const uint8_t*, int64_t)`, `ValidateE4M3Scales(const uint8_t*, int64_t)`, `ValidateGlobalScale(float)` (`validator.h`).
- Produces: file `test_validator_contract.cpp`; later tasks append to it.

- [ ] **Step 1: Write the green primitive tests**

Create `apps/lsxfabric/tests/test_validator_contract.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "validator.h"
#include "lsxcommon/core.h"
#include "lsxcommon/schema.h"

#include <arrow/api.h>
#include <arrow/array/concatenate.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace lsxfabric;
namespace fs = std::filesystem;

namespace {

struct CoreScope {
    CoreScope() { lsxcommon::Core::Initialize("test_validator_contract.log"); }
    ~CoreScope() {
        lsxcommon::Core::Shutdown();
        std::remove("test_validator_contract.log");
    }
};

std::shared_ptr<arrow::RecordBatch> MakeExpertBatch(float w1_global = 1.0f,
                                                     uint8_t w1_scale = 0x08) {
    lsxcommon::PackedExpert e;
    e.w1_packed = {0x01, 0x23, 0x45, 0x67};
    e.w2_packed = {0x89, 0xAB};
    e.w3_packed = {0xCD, 0xEF};
    e.w1_scales = {w1_scale};
    e.w2_scales = {0x10};
    e.w3_scales = {0x18};
    e.w1_global_scale = w1_global;
    e.w2_global_scale = 1.0f;
    e.w3_global_scale = 1.0f;
    return lsxcommon::ExpertSchema::MakeBatch(e).ValueOrDie();
}

std::shared_ptr<arrow::RecordBatch> MakeDenseBatch(float gs = 1.0f) {
    lsxcommon::DenseNvfp4Record r;
    r.name = "w";
    r.shape = {2, 2};
    r.dtype = "NVFP4";
    r.packed = {0x01, 0x23};
    r.scales = {0x08};
    r.global_scale = gs;
    return lsxcommon::DenseNvfp4Schema::MakeBatch({r}).ValueOrDie();
}

std::shared_ptr<arrow::RecordBatch>
ConcatRows(const std::shared_ptr<arrow::RecordBatch> &a,
           const std::shared_ptr<arrow::RecordBatch> &b) {
    std::vector<std::shared_ptr<arrow::Array>> cols;
    for (int i = 0; i < a->num_columns(); ++i)
        cols.push_back(arrow::Concatenate({a->column(i), b->column(i)}).ValueOrDie());
    return arrow::RecordBatch::Make(a->schema(), a->num_rows() + b->num_rows(), cols);
}

fs::path WriteTemp(const std::string &name,
                   const std::shared_ptr<arrow::RecordBatch> &batch) {
    const fs::path p = fs::temp_directory_path() / name;
    fs::remove(p);
    REQUIRE(lsxcommon::WriteRecordBatchToFile(batch, p).ok());
    return p;
}

} // namespace

TEST_CASE("validator: ValidateNVFP4Data accepts every byte value",
          "[validator]") {
    std::vector<uint8_t> all(256);
    for (int i = 0; i < 256; ++i) all[i] = static_cast<uint8_t>(i);
    REQUIRE(ValidateNVFP4Data(all.data(), static_cast<int64_t>(all.size())));
    REQUIRE(ValidateNVFP4Data(nullptr, 0));
}

TEST_CASE("validator: ValidateE4M3Scales accepts E4M3 P2 codes only",
          "[validator]") {
    uint8_t ok[] = {0x08, 0x10, 0x18, 0x20, 0x70};
    REQUIRE(ValidateE4M3Scales(ok, 5));
    REQUIRE(ValidateE4M3Scales(nullptr, 0));
    uint8_t sign = 0x88, zero_exp = 0x00, too_high = 0x78, mantissa = 0x09;
    REQUIRE_FALSE(ValidateE4M3Scales(&sign, 1));
    REQUIRE_FALSE(ValidateE4M3Scales(&zero_exp, 1));
    REQUIRE_FALSE(ValidateE4M3Scales(&too_high, 1));
    REQUIRE_FALSE(ValidateE4M3Scales(&mantissa, 1));
}

TEST_CASE("validator: ValidateGlobalScale requires finite positive",
          "[validator]") {
    REQUIRE(ValidateGlobalScale(1.0f));
    REQUIRE(ValidateGlobalScale(0.001f));
    REQUIRE_FALSE(ValidateGlobalScale(0.0f));
    REQUIRE_FALSE(ValidateGlobalScale(-1.0f));
    REQUIRE_FALSE(ValidateGlobalScale(NAN));
    REQUIRE_FALSE(ValidateGlobalScale(INFINITY));
}
```

- [ ] **Step 2: Register the file in CMake**

In `apps/lsxfabric/tests/CMakeLists.txt`, immediately after line 2
(`target_sources(test_lsxfabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_validator.cpp)`), insert:

```cmake
target_sources(test_lsxfabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test_validator_contract.cpp)
```

- [ ] **Step 3: Build and run**

Run: `cmake --build --preset linux-debug --target test_lsxfabric && ./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add apps/lsxfabric/tests/test_validator_contract.cpp apps/lsxfabric/tests/CMakeLists.txt
git commit -m "test: characterise lsxfabric validator primitives"
```

---

### Task 4: LoadModelIndex green tests (S4)

**Files:**
- Modify: `apps/lsxfabric/tests/test_validator_contract.cpp` (append)

- [ ] **Step 1: Append the tests**

```cpp
TEST_CASE("validator: LoadModelIndex parses a full index", "[validator]") {
    const fs::path p = fs::temp_directory_path() / "lsxfabric_idx_full.json";
    std::ofstream(p) << R"({
        "format": "logestix-v1-nvfp4",
        "model": "ds",
        "num_layers": 43,
        "experts_per_layer": 256,
        "total_pinned_tensors": 2881,
        "total_experts": 11008
    })";
    const ModelInfo info = LoadModelIndex(p.string());
    REQUIRE(info.format == "logestix-v1-nvfp4");
    REQUIRE(info.model_name == "ds");
    REQUIRE(info.num_layers == 43);
    REQUIRE(info.experts_per_layer == 256);
    REQUIRE(info.total_pinned == 2881);
    REQUIRE(info.total_experts == 11008);
    fs::remove(p);
}

TEST_CASE("validator: LoadModelIndex defaults absent fields", "[validator]") {
    CoreScope scope;
    const fs::path p = fs::temp_directory_path() / "lsxfabric_idx_sparse.json";
    std::ofstream(p) << R"({"format": "logestix-granite"})";
    const ModelInfo info = LoadModelIndex(p.string());
    REQUIRE(info.format == "logestix-granite");
    REQUIRE(info.model_name.empty());
    REQUIRE(info.num_layers == 0);
    REQUIRE(info.experts_per_layer == 0);
    REQUIRE(info.total_pinned == 0);
    REQUIRE(info.total_experts == 0);
    fs::remove(p);
}

TEST_CASE("validator: LoadModelIndex defaults on malformed input",
          "[validator]") {
    CoreScope scope;
    const fs::path bad = fs::temp_directory_path() / "lsxfabric_idx_bad.json";
    std::ofstream(bad) << "{ not json";
    REQUIRE(LoadModelIndex(bad.string()).num_layers == 0);

    const fs::path arr = fs::temp_directory_path() / "lsxfabric_idx_arr.json";
    std::ofstream(arr) << R"([1, 2, 3])";
    REQUIRE(LoadModelIndex(arr.string()).num_layers == 0);

    const fs::path typed = fs::temp_directory_path() / "lsxfabric_idx_typed.json";
    std::ofstream(typed) << R"({"num_layers": "43"})";
    REQUIRE(LoadModelIndex(typed.string()).num_layers == 0);

    REQUIRE(LoadModelIndex("lsxfabric_no_such_index.json").num_layers == 0);
    fs::remove(bad);
    fs::remove(arr);
    fs::remove(typed);
}
```

- [ ] **Step 2: Build and run**

Run: `cmake --build --preset linux-debug --target test_lsxfabric && ./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add apps/lsxfabric/tests/test_validator_contract.cpp
git commit -m "test: characterise LoadModelIndex parsing"
```

---

### Task 5: Schema validator green tests (S5)

**Files:**
- Modify: `apps/lsxfabric/tests/test_validator_contract.cpp` (append)

**Interfaces:**
- Consumes: `ValidateExpert(const std::string&)`, `ValidatePinnedTensors(const std::string&, int64_t)`, plus the not-in-header `ValidateDenseNvfp4Schema(const std::string&)` via a local forward declaration.

- [ ] **Step 1: Append the tests**

```cpp
namespace lsxfabric {
// Declared in validator.cpp only; redeclared here to test it directly.
ValidationResult ValidateDenseNvfp4Schema(const std::string &arrow_path);
} // namespace lsxfabric

TEST_CASE("validator: ValidateExpert accepts a valid 9-column batch",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_expert_ok.arrow", MakeExpertBatch());
    const ValidationResult r = ValidateExpert(p.string());
    REQUIRE(r.ok);
    REQUIRE(r.errors == 0);
    fs::remove(p);
}

TEST_CASE("validator: ValidateExpert rejects a wrong column count",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_expert_6col.arrow", MakeDenseBatch());
    const ValidationResult r = ValidateExpert(p.string());
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.errors >= 1);
    fs::remove(p);
}

TEST_CASE("validator: ValidateExpert checks every row", "[validator]") {
    CoreScope scope;
    auto bad = MakeExpertBatch(1.0f, 0x09); // mantissa != 0 -> invalid scale
    auto batch = ConcatRows(MakeExpertBatch(), bad);
    const fs::path p = WriteTemp("lsxfabric_expert_2row.arrow", batch);
    const ValidationResult r = ValidateExpert(p.string());
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.errors >= 1);
    fs::remove(p);
}

TEST_CASE("validator: ValidateExpert rejects an invalid global scale",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_expert_gs.arrow", MakeExpertBatch(0.0f));
    const ValidationResult r = ValidateExpert(p.string());
    REQUIRE_FALSE(r.ok);
    fs::remove(p);
}

TEST_CASE("validator: ValidateExpert errors on an unreadable file",
          "[validator]") {
    CoreScope scope;
    const ValidationResult r = ValidateExpert("lsxfabric_missing_expert.arrow");
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.errors >= 1);
}

TEST_CASE("validator: ValidateDenseNvfp4Schema accepts 6 columns",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_dense_ok.arrow", MakeDenseBatch());
    const ValidationResult r = ValidateDenseNvfp4Schema(p.string());
    REQUIRE(r.ok);
    REQUIRE(r.errors == 0);
    fs::remove(p);
}

TEST_CASE("validator: ValidateDenseNvfp4Schema rejects a wrong column count",
          "[validator]") {
    CoreScope scope;
    const fs::path p = WriteTemp("lsxfabric_dense_9col.arrow", MakeExpertBatch());
    const ValidationResult r = ValidateDenseNvfp4Schema(p.string());
    REQUIRE_FALSE(r.ok);
    fs::remove(p);
}

TEST_CASE("validator: ValidatePinnedTensors skips tokenizer and shared experts",
          "[validator]") {
    CoreScope scope;
    const fs::path dir = fs::temp_directory_path() / "lsxfabric_pinned_skip";
    fs::remove_all(dir);
    fs::create_directories(dir);
    REQUIRE(lsxcommon::WriteRecordBatchToFile(MakeExpertBatch(),
              dir / "global.arrow").ok());
    REQUIRE(lsxcommon::WriteRecordBatchToFile(MakeExpertBatch(),
              dir / "layer_0.arrow").ok());
    REQUIRE(lsxcommon::WriteRecordBatchToFile(MakeExpertBatch(),
              dir / "tokenizer_vocab.arrow").ok());
    REQUIRE(lsxcommon::WriteRecordBatchToFile(MakeExpertBatch(),
              dir / "shared_experts.arrow").ok());
    // Only global.arrow + layer_0.arrow are counted => 2 rows.
    const ValidationResult r = ValidatePinnedTensors(dir.string(), 2);
    REQUIRE(r.ok);
    REQUIRE(r.warnings == 0);
    fs::remove_all(dir);
}
```

- [ ] **Step 2: Build and run**

Run: `cmake --build --preset linux-debug --target test_lsxfabric && ./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add apps/lsxfabric/tests/test_validator_contract.cpp
git commit -m "test: characterise schema validators"
```

---

### Task 6: RunValidator green tests (S6)

**Files:**
- Modify: `apps/lsxfabric/tests/test_validator_contract.cpp` (append)

**Interfaces:**
- Consumes: `RunValidator(const std::string&)`.

- [ ] **Step 1: Append a fixture helper and the green tests**

```cpp
namespace {

struct ModelDir {
    fs::path root;
    explicit ModelDir(const std::string &name)
        : root(fs::temp_directory_path() / name) {
        fs::remove_all(root);
        fs::create_directories(root / "logestix" / "pinned");
    }
    ~ModelDir() { fs::remove_all(root); }
    void index(const std::string &json) const {
        std::ofstream(root / "logestix" / "model.index.json") << json;
    }
    std::string str() const { return root.string(); }
};

} // namespace

TEST_CASE("validator: RunValidator errors when the index is missing",
          "[validator]") {
    CoreScope scope;
    ModelDir md("lsxfabric_rv_noindex");
    fs::remove_all(md.root / "logestix");
    const ValidationResult r = RunValidator(md.str());
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.errors >= 1);
}

TEST_CASE("validator: RunValidator warns on an unknown format and continues",
          "[validator]") {
    CoreScope scope;
    ModelDir md("lsxfabric_rv_unknown");
    md.index(R"({"format":"logestix-granite","model":"g","num_layers":1,
                 "experts_per_layer":0,"total_pinned_tensors":0,"total_experts":0})");
    const ValidationResult r = RunValidator(md.str());
    REQUIRE(r.ok);
    REQUIRE(r.warnings > 0);
}

TEST_CASE("validator: RunValidator warns, not errors, on missing experts",
          "[validator]") {
    CoreScope scope;
    ModelDir md("lsxfabric_rv_moe_noexperts");
    md.index(R"({"format":"logestix-v1","model":"m","num_layers":1,
                 "experts_per_layer":64,"total_pinned_tensors":0,"total_experts":64})");
    const ValidationResult r = RunValidator(md.str());
    REQUIRE(r.ok);
    REQUIRE(r.errors == 0);
    REQUIRE(r.warnings >= 1);
}

TEST_CASE("validator: RunValidator warns when pinned/ is absent", "[validator]") {
    CoreScope scope;
    ModelDir md("lsxfabric_rv_nopinned");
    fs::remove_all(md.root / "logestix" / "pinned");
    md.index(R"({"format":"logestix-qwen","model":"q","num_layers":0,
                 "experts_per_layer":0,"total_pinned_tensors":0,"total_experts":0})");
    const ValidationResult r = RunValidator(md.str());
    REQUIRE(r.ok);
    REQUIRE(r.warnings >= 1);
}
```

- [ ] **Step 2: Build and run**

Run: `cmake --build --preset linux-debug --target test_lsxfabric && ./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add apps/lsxfabric/tests/test_validator_contract.cpp
git commit -m "test: characterise RunValidator accounting"
```

---

### Task 7: Validator red spec tests (S6 divergences, F1/F6/F8)

**Files:**
- Modify: `apps/lsxfabric/tests/test_validator_contract.cpp` (append)

- [ ] **Step 1: Append the hidden spec tests**

```cpp
// RED spec tests. Hidden (tag starts with '.') so the default run stays green.
// Run explicitly: ./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric "[.audit-spec]"

// FINDING F1: the current model format (logestix-v1-nvfp4) must be recognised.
TEST_CASE("validator [spec]: recognises logestix-v1-nvfp4", "[.audit-spec]") {
    CoreScope scope;
    ModelDir md("lsxfabric_spec_v1nvfp4");
    md.index(R"({"format":"logestix-v1-nvfp4","model":"ds","num_layers":1,
                 "experts_per_layer":1,"total_pinned_tensors":0,"total_experts":1})");
    fs::create_directories(md.root / "logestix" / "experts" / "layer_0");
    REQUIRE(lsxcommon::WriteRecordBatchToFile(
                MakeExpertBatch(),
                md.root / "logestix" / "experts" / "layer_0" / "exp_0.arrow").ok());
    const ValidationResult r = RunValidator(md.str());
    REQUIRE(r.ok);
    REQUIRE(r.errors == 0);
    REQUIRE(r.warnings == 0);
}

// FINDING F6: a dense format (granite) must not expect per-layer expert files.
TEST_CASE("validator [spec]: dense formats need no expert files", "[.audit-spec]") {
    CoreScope scope;
    ModelDir md("lsxfabric_spec_granite");
    md.index(R"({"format":"logestix-granite","model":"g","num_layers":1,
                 "experts_per_layer":0,"total_pinned_tensors":0,"total_experts":0})");
    const ValidationResult r = RunValidator(md.str());
    REQUIRE(r.ok);
    REQUIRE(r.errors == 0);
    REQUIRE(r.warnings == 0);
}

// FINDING F8: a declared expert missing on disk must be an error, not a warning.
TEST_CASE("validator [spec]: a missing declared expert is an error",
          "[.audit-spec]") {
    CoreScope scope;
    ModelDir md("lsxfabric_spec_missing_expert");
    md.index(R"({"format":"logestix-v1","model":"m","num_layers":1,
                 "experts_per_layer":64,"total_pinned_tensors":0,"total_experts":64})");
    const ValidationResult r = RunValidator(md.str());
    REQUIRE_FALSE(r.ok);
}
```

- [ ] **Step 2: Build; confirm default run green, explicit run red**

Run: `cmake --build --preset linux-debug --target test_lsxfabric && ./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric`
Expected: green (hidden excluded).

Run: `./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric "[.audit-spec]"`
Expected: 3 tests run, 3 FAIL.

- [ ] **Step 3: Commit**

```bash
git add apps/lsxfabric/tests/test_validator_contract.cpp
git commit -m "test: add validator spec tests (hidden red)"
```

---

### Task 8: Runtime integration evidence (S7) — GPU-exclusive, run LAST

**Files:**
- No source changes; outputs captured to `/tmp/lsxfabric_audit/`.

**Constraints:** `--validate` is host-only. Generation is GPU-exclusive: run the two generation commands strictly one after another, with nothing else on the GPU.

- [ ] **Step 1: Build the app**

Run: `cmake --build --preset linux-debug --target lsxfabric`
Expected: builds clean.

- [ ] **Step 2: No-op mode (S2) is silent and returns 0**

Run: `./build/linux-debug/apps/lsxfabric/lsxfabric --max-gen 1`
Expected: no generation, exit code 0 (verify with `echo $?`).

- [ ] **Step 3: `--validate` over every converted vault model**

Run:

```bash
mkdir -p /tmp/lsxfabric_audit
for d in /mnt/llm_models/models/*/; do
  [ -f "$d/logestix/model.index.json" ] || continue
  n=$(basename "$d")
  ./build/linux-debug/apps/lsxfabric/lsxfabric --localmodel "$d" --validate \
      > "/tmp/lsxfabric_audit/validate_$n.log" 2>&1
  echo "$n rc=$? format=$(grep -o '"format": *"[^"]*"' "$d/logestix/model.index.json" | head -1)"
done
```

Capture the per-model `rc`, format, and the `=== Validation Summary ===` block of each log into the report appendix. Expected (characterised): every model logs `Unknown format ... continuing anyway`; dense `granite-4.2-30b` reports PASS with per-layer "Expert not found" warnings.

- [ ] **Step 4: Real generation — default model path (GPU-exclusive)**

Run: `./build/linux-debug/apps/lsxfabric/lsxfabric --testprompt --max-gen 64`
Expected: loads the default `DeepSeek-V4-Flash`, streams up to 64 tokens; capture output and exit code to `/tmp/lsxfabric_audit/gen_default.log`.

- [ ] **Step 5: Real generation — fast model + dispatch precedence (GPU-exclusive)**

Run: `./build/linux-debug/apps/lsxfabric/lsxfabric --localmodel /mnt/llm_models/models/gemma-4-26B-A4B-it --testprompt --validate --max-gen 64`
Expected: generation runs (proves `--testprompt` wins over `--validate`); capture to `/tmp/lsxfabric_audit/gen_gemma.log`.

- [ ] **Step 6: Commit nothing**

Integration produces evidence only. Do not commit `/tmp/lsxfabric_audit/`; the report quotes it.

---

### Task 9: Findings report `docs/LSXFABRIC_AUDIT.md`

**Files:**
- Create: `docs/LSXFABRIC_AUDIT.md`

- [ ] **Step 1: Write the report**

Create `docs/LSXFABRIC_AUDIT.md` with these sections. The findings table below is derived from static analysis; after Task 8, replace each `Evidence` placeholder `<integration>` with the concrete observed output, and adjust severity only if the runtime evidence contradicts it.

```markdown
# lsxfabric — Functional Audit

Date: 2026-09-24
Branch: `audit/lsxfabric-functional-audit`
Commit: `<HEAD>`
Build: `linux-debug`, CUDA 13.4, RTX 5090 (32.6 GB)
Scope: CLI contract, validator primitives, index parsing, schema validators,
RunValidator accounting, runtime integration. Non-goals: fixes, shared
`RunCliGeneration` internals, performance.

## Method

Contract-first TDD. Green characterisation tests fix current behaviour; hidden
Catch2 tests (`[.audit-spec]`) encode the expected contract where it diverges.
Run the red spec set with:

    ./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli "[.audit-spec]"
    ./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric "[.audit-spec]"

## Findings

| id | severity | surface | expected | actual | evidence | impact |
|---|---|---|---|---|---|---|
| F1 | High | RunValidator | recognise the formats currently produced by lsxforge | only `logestix-v1`/`-qwen`/`-glm`; none of the 10 vault formats matches | `[.audit-spec] recognises logestix-v1-nvfp4`; <integration> | every real model is treated as unknown and validated on the wrong path |
| F2 | Low | ValidateNVFP4Data | detect malformed packed data | `lo>0x0F \|\| hi>0x0F` is never true; always returns true | `validator: ValidateNVFP4Data accepts every byte value` | packed-data validation is a no-op |
| F3 | Info | ValidateE4M3Scales | consistent with the codec | requires zero mantissa (matched to `nvfp4::FloatToE4m3P2`) | `validator: ValidateE4M3Scales accepts E4M3 P2 codes only` | by design, not a defect |
| F4 | Medium | ParseFabricCliArgs | flags must not swallow other flags | `--prompt --validate` sets prompt=`--validate`, disables validate | `fabric cli [spec]: --prompt does not swallow a known flag` | misconfigured invocations change mode silently |
| F5 | Medium | ParseFabricCliArgs | reject/clamp invalid `--max-gen` | `abc`→0, `-5`→-5 via `std::atoi` | `fabric cli [spec]: --max-gen ...` | invalid generation length reaches the runtime |
| F6 | Medium | RunValidator | dense formats need no per-expert files | a dense format falls into the expert loop and emits one warning per layer | `[.audit-spec] dense formats need no expert files`; <integration> | dense models report spurious warnings, hiding real ones |
| F7 | Medium | RunValidator expert coverage | validate the declared expert set | checks only `experts/layer_N/exp_N.arrow` — `num_layers` of `total_experts` | code review; <integration> | corruption outside that set is never seen |
| F8 | High | RunValidator | a declared-but-missing expert is an error | warns, `ok == true` (PASS) | `[.audit-spec] a missing declared expert is an error` | incomplete models report PASS |
| F9 | Low | RunValidator summary | report the loaded pinned count | prints declared `total_pinned` | code review | misleading summary line |
| F10 | Low | RunValidator summary | count dense-schema tensors | the `DenseNvfp4Schema` branch never increments `experts_checked` | code review | "Experts checked: 0" for qwen/glm |
| F11 | Low | main | report an inactive mode | no `--validate`/`--testprompt` runs nothing and returns 0 silently | Task 8 Step 2 | a typo'd flag looks successful |
| F12 | Medium | ValidatePinnedTensors | validate pinned column schema | opens/reads/counts rows only; wrong column types pass | code review | mistyped pinned shards pass validation |
| F13 | Info | LoadModelIndex | strict parsing | absent/mistyped fields silently default to 0/empty | `validator: LoadModelIndex ...` | malformed index degrades rather than fails |

## Test inventory

| file | test | tag | kind | surface |
|---|---|---|---|---|
| test_fabric_cli_edges.cpp | flag without value ignored | [lsxfabric][cli] | green | S1 |
| test_fabric_cli_edges.cpp | --prompt consumes flag token | [lsxfabric][cli] | green | S1 |
| test_fabric_cli_edges.cpp | non-numeric --max-gen → 0 | [lsxfabric][cli] | green | S1 |
| test_fabric_cli_edges.cpp | negative --max-gen verbatim | [lsxfabric][cli] | green | S1 |
| test_fabric_cli_edges.cpp | repeated flags last wins | [lsxfabric][cli] | green | S1 |
| test_fabric_cli_edges.cpp | flag-looking value literal | [lsxfabric][cli] | green | S1 |
| test_fabric_cli_edges.cpp | help flags ignored | [lsxfabric][cli] | green | S1 |
| test_fabric_cli_edges.cpp | --max-gen rejects non-numeric | [.audit-spec] | red | F5 |
| test_fabric_cli_edges.cpp | --max-gen positive | [.audit-spec] | red | F5 |
| test_fabric_cli_edges.cpp | --prompt no swallow | [.audit-spec] | red | F4 |
| test_validator_contract.cpp | ValidateNVFP4Data all bytes | [validator] | green | S3 |
| test_validator_contract.cpp | ValidateE4M3Scales P2 | [validator] | green | S3 |
| test_validator_contract.cpp | ValidateGlobalScale finite>0 | [validator] | green | S3 |
| test_validator_contract.cpp | LoadModelIndex full/sparse/malformed | [validator] | green | S4 |
| test_validator_contract.cpp | ValidateExpert cases | [validator] | green | S5 |
| test_validator_contract.cpp | ValidateDenseNvfp4Schema | [validator] | green | S5 |
| test_validator_contract.cpp | ValidatePinnedTensors skip | [validator] | green | S5 |
| test_validator_contract.cpp | RunValidator index unknown experts pinned | [validator] | green | S6 |
| test_validator_contract.cpp | recognises logestix-v1-nvfp4 | [.audit-spec] | red | F1 |
| test_validator_contract.cpp | dense formats need no experts | [.audit-spec] | red | F6 |
| test_validator_contract.cpp | missing expert is an error | [.audit-spec] | red | F8 |

## Integration evidence

<integration>
```

- [ ] **Step 2: Verify the report renders and the placeholders are resolved**

Run: `grep -n "<integration>" docs/LSXFABRIC_AUDIT.md`
Expected: no output (all placeholders replaced with Task 8 evidence).

- [ ] **Step 3: Commit**

```bash
git add docs/LSXFABRIC_AUDIT.md
git commit -m "docs: lsxfabric functional audit findings"
```

---

### Task 10: Final verification

- [ ] **Step 1: Default fast suite green**

Run: `cmake --build --preset linux-debug && ctest --preset linux-debug -R "lsxfabric"`
Expected: `test_lsxfabric` and `test_fabric_cli` pass.

- [ ] **Step 2: Label/router guards still pass**

Run: `ctest --preset linux-debug -R "labels"`
Expected: `test_ctest_labels_guard` passes (no new targets/labels were added).

- [ ] **Step 3: Red spec set is the documented, expected failure**

Run:

```bash
./build/linux-debug/apps/lsxfabric/tests/test_fabric_cli "[.audit-spec]"
./build/linux-debug/apps/lsxfabric/tests/test_lsxfabric "[.audit-spec]"
```

Expected: 6 red tests total fail; the count matches the report's test inventory.

- [ ] **Step 4: Working tree is clean and review the branch**

Run: `git status --short && git log --oneline main..HEAD`
Expected: clean tree; commits: design doc, CLI characterisation, CLI spec, primitive characterisation, index characterisation, schema characterisation, RunValidator characterisation, validator spec, audit report.

---

## Self-Review

- **Spec coverage:** S1→Tasks 1–2; S2→Task 8 Step 2 + report F11; S3→Task 3; S4→Task 4; S5→Task 5; S6→Tasks 6–7; S7→Task 8; report→Task 9; verification→Task 10. F1–F13 each map to a test and/or integration evidence line in Task 9.
- **Placeholder scan:** the only intentional `<integration>` markers are explicitly resolved in Task 9 Step 2 before commit.
- **Type consistency:** `Parse`, `MakeExpertBatch`, `MakeDenseBatch`, `ConcatRows`, `WriteTemp`, `ModelDir`, `CoreScope` are defined once (Tasks 1/3/6) and reused with identical signatures in later tasks. `ValidateDenseNvfp4Schema` is forward-declared to the exact definition signature.
