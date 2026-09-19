# DeepSeek Tests Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract DeepSeek-specific tests into dedicated `test_deepseek_*.cpp` files so GLM and DeepSeek suites are autonomous and each model can be tested independently via `ctest -R`.

**Architecture:** Two test files are renamed via `git mv` (preserving history): `test_loader.cpp` → `test_deepseek_loader.cpp` and `test_inference_kernels.cu` → `test_deepseek_inference_kernels.cu`. CMake targets, `add_test` names, and `[deepseek]` Catch2 tags are updated to mirror the renames. GLM tests (`test_glm_*.cpp`), shared files (`test_core`, `test_tokenizer`, `test_mosaic_network`, `test_integrated_pipeline`), and `test_model_factory.cpp` are untouched.

**Tech Stack:** C++20, CMake (Ninja, `linux-release-znver4` preset), Catch2 v3.7.0, CUDA (NVCC 13.2, Blackwell sm_120/sm_121).

## Global Constraints

- Target architectures: Blackwell only (`compute_120`/`sm_120`, `compute_121`/`sm_121`). No Hopper/Ada/Ampere.
- `BUILD_TESTING=ON` in every preset; tests build with Catch2.
- MSVC workaround block for `test_inference_kernels` must be preserved **verbatim** (OBJECT library + explicit include dirs + `-Xcompiler=/utf-8` guarded by `$<$<PLATFORM_ID:Windows>:...>`).
- CUDA `.cu` object libraries: `CUDA_SEPARABLE_COMPILATION OFF`, `CUDA_ARCHITECTURES OFF`.
- All logging via spdlog; no `std::cout`/`printf` in source (tests use Catch2 macros).
- Commit messages in English with prefixes: `test:`, `refactor:`, `docs:`.
- Existing env-var model-path pattern: GLM real-model tests use `LSX_TEST_GLM_MODEL_PATH` (default `/mnt/llm_models/models/GLM-5.2`, SKIP if absent); DeepSeek factory test uses `LSX_TEST_MODEL_PATH`. The two renamed files contain only synthetic fixtures — no real model path.

---

### Task 1: Rename loader test and update CMake

**Files:**
- Rename: `tests/test_loader.cpp` → `tests/test_deepseek_loader.cpp` (via `git mv`)
- Modify: `tests/CMakeLists.txt:21-38`

**Interfaces:**
- Consumes: existing `tests/test_loader.cpp` content (3 TEST_CASEs, `lsxcommon::ExpertSchema` / `lsxcommon::Loader` / `lsxcommon::float_to_e4m3_p2`).
- Produces: `tests/test_deepseek_loader.cpp` and CMake target `test_deepseek_loader` (executable). Later `ctest -R deepseek` matches this target.

- [ ] **Step 1: Rename the file with git mv**

Run:
```bash
git mv tests/test_loader.cpp tests/test_deepseek_loader.cpp
```

Expected: file moved, `git status` shows `R  tests/test_loader.cpp -> tests/test_deepseek_loader.cpp`.

- [ ] **Step 2: Rename CMake target block**

In `tests/CMakeLists.txt`, replace the block at lines 21-38 (comment through `add_test`) from:

```cmake
# ── Loader round-trip test ──────────────────────────────────────────
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

to:

```cmake
# ── DeepSeek loader round-trip test ─────────────────────────────────
add_executable(test_deepseek_loader test_deepseek_loader.cpp)
target_link_libraries(test_deepseek_loader PRIVATE lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_loader>"
    )
    add_custom_command(TARGET test_deepseek_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_loader>"
    )
endif()

add_test(NAME test_deepseek_loader COMMAND test_deepseek_loader)
```

- [ ] **Step 3: Add `[deepseek]` tags**

In `tests/test_deepseek_loader.cpp`, edit the three TEST_CASE tag lists:

- Line 48: `"[loader]"` → `"[deepseek][loader]"`
- Line 89: `"[schema][e4m3]"` → `"[deepseek][schema][e4m3]"`
- Line 129: `"[loader][edge]"` → `"[deepseek][loader][edge]"`

Use the `edit` tool with these exact old/new strings (each appears once).

- [ ] **Step 4: Reconfigure and build**

Run:
```bash
cmake --preset linux-release-znver4
cmake --build --preset linux-release-znver4 --target test_deepseek_loader
```

Expected: target builds clean (exit 0).

- [ ] **Step 5: Verify test runs and passes**

Run:
```bash
./build/linux-release-znver4/tests/test_deepseek_loader
```

Expected: 3 test cases pass, exit 0 (e.g. `All tests passed (3 assertions in 3 test cases)` style output; loader cases each assert multiple `REQUIRE`/`CHECK`).

- [ ] **Step 6: Commit**

```bash
git add tests/test_deepseek_loader.cpp tests/CMakeLists.txt
git commit -m "refactor: rename DeepSeek loader test to test_deepseek_loader"
```

---

### Task 2: Rename CUDA kernels test and update CMake

**Files:**
- Rename: `tests/test_inference_kernels.cu` → `tests/test_deepseek_inference_kernels.cu` (via `git mv`)
- Modify: `tests/CMakeLists.txt:335-395`

**Interfaces:**
- Consumes: existing `tests/test_inference_kernels.cu` content (12 TEST_CASEs, `lsxcommon::gpu::run_routed_expert_ffn`, shared_experts/safetensors_gpu helpers). The OBJECT-library compile isolation pattern stays.
- Produces: `tests/test_deepseek_inference_kernels.cu`, OBJECT target `test_deepseek_inference_kernels_cu`, executable `test_deepseek_inference_kernels`. Later `ctest -R deepseek` matches this executable.

- [ ] **Step 1: Rename the file with git mv**

Run:
```bash
git mv tests/test_inference_kernels.cu tests/test_deepseek_inference_kernels.cu
```

Expected: file moved, `git status` shows `R  tests/test_inference_kernels.cu -> tests/test_deepseek_inference_kernels.cu`.

- [ ] **Step 2: Update OBJECT library source reference**

In `tests/CMakeLists.txt:350`, replace:
```cmake
add_library(test_inference_kernels_cu OBJECT test_inference_kernels.cu)
```
with:
```cmake
add_library(test_deepseek_inference_kernels_cu OBJECT test_deepseek_inference_kernels.cu)
```

- [ ] **Step 3: Rename OBJECT target in set_target_properties and compile options**

Lines 351-359: replace every `test_inference_kernels_cu` token with `test_deepseek_inference_kernels_cu`. The block after editing reads:

```cmake
set_target_properties(test_deepseek_inference_kernels_cu PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES OFF
)
target_compile_options(test_deepseek_inference_kernels_cu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-std=c++20;
    --generate-code=arch=compute_120,code=sm_120;
    --generate-code=arch=compute_121,code=sm_121;
    $<$<PLATFORM_ID:Windows>:-Xcompiler=/utf-8>>)
```

- [ ] **Step 4: Rename OBJECT target in include dirs, compile definitions, link**

Lines 360-375: replace every `test_inference_kernels_cu` token with `test_deepseek_inference_kernels_cu`. This covers `target_include_directories(...)`, `target_compile_definitions(...)`, and `target_link_libraries(test_deepseek_inference_kernels_cu PRIVATE Catch2::Catch2WithMain)`.

- [ ] **Step 5: Rename executable target and add_test**

Lines 377-395: replace every `test_inference_kernels` token with `test_deepseek_inference_kernels`. The block after editing reads:

```cmake
add_executable(test_deepseek_inference_kernels)
set_target_properties(test_deepseek_inference_kernels PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(test_deepseek_inference_kernels PRIVATE
    test_deepseek_inference_kernels_cu lsxcommon Catch2::Catch2WithMain)

if(WIN32)
    add_custom_command(TARGET test_deepseek_inference_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_inference_kernels>"
    )
    add_custom_command(TARGET test_deepseek_inference_kernels POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:arrow_cuda_shared>"
            "$<TARGET_FILE_DIR:test_deepseek_inference_kernels>"
    )
endif()

add_test(NAME test_deepseek_inference_kernels COMMAND test_deepseek_inference_kernels)
```

- [ ] **Step 6: Add `[deepseek]` tags**

In `tests/test_deepseek_inference_kernels.cu`, edit the 12 TEST_CASE tag lists, adding `[deepseek]`. Three distinct tag patterns exist:

**Single-line tags** — 8 TEST_CASEs end with `"[inference]")` (lines 53, 90, 151, 192, 241, 287, 338, 416). Use the `edit` tool with `replaceAll: true`: old string `"[inference]")`, new string `"[deepseek][inference]")`. Verify 8 replacements happened.

**Multi-line tag `"[nvfp4][regression]"`** — 3 TEST_CASEs (lines 509, 559, 611). Use the `edit` tool with `replaceAll: true`: old string `"[nvfp4][regression]")`, new string `"[deepseek][nvfp4][regression]")`. Verify 3 replacements happened.

**Multi-line tag `"[inference][regression]"`** — 1 TEST_CASE (line 684). Single `edit`: old string `"[inference][regression]")`, new string `"[deepseek][inference][regression]")`.

After editing, run `rg -n 'TEST_CASE' tests/test_deepseek_inference_kernels.cu` and verify all 12 tag lists now start with `[deepseek]`.

- [ ] **Step 7: Reconfigure and build**

Run:
```bash
cmake --preset linux-release-znver4
cmake --build --preset linux-release-znver4 --target test_deepseek_inference_kernels
```

Expected: OBJECT library compiles with nvcc (12 kernel tests), executable links clean (exit 0). Watch for the MSVC `/utf-8` warning only on Windows — on Linux this must build with zero errors.

- [ ] **Step 8: Verify test runs and passes**

Run:
```bash
./build/linux-release-znver4/tests/test_deepseek_inference_kernels
```

Expected: 12 test cases pass, exit 0 (CUDA kernels run on the RTX 5090; `cudaFree(0)`/cublas init inside tests succeed).

- [ ] **Step 9: Commit**

```bash
git add tests/test_deepseek_inference_kernels.cu tests/CMakeLists.txt
git commit -m "refactor: rename DeepSeek inference kernels test to test_deepseek_inference_kernels"
```

---

### Task 3: Full rebuild + suite verification

**Files:**
- None modified — verification only.

**Interfaces:**
- Consumes: renamed targets from Tasks 1-2 (`test_deepseek_loader`, `test_deepseek_inference_kernels`).
- Produces: verified final state — full 22-test suite green, per-model filtering working.

- [ ] **Step 1: Full clean rebuild**

Run:
```bash
cmake --build --preset linux-release-znver4 --clean-first 2>&1 | tail -5
```

Expected: full build completes; final lines show no errors (arrow/third-party warnings OK).

- [ ] **Step 2: Verify per-model filtering — DeepSeek**

Run:
```bash
ctest --preset linux-release-znver4 -R deepseek
```

Expected: exactly 2 tests listed and passed: `test_deepseek_loader`, `test_deepseek_inference_kernels`.

- [ ] **Step 3: Verify per-model filtering — GLM**

Run:
```bash
ctest --preset linux-release-znver4 -R glm
```

Expected: all 11 GLM executables listed and passed: `test_glm_convert`, `test_glm_loader`, `test_glm_model`, `test_glm_compute`, `test_glm_dense_compute`, `test_glm_moe_router`, `test_glm_experts`, `test_glm_moe_compute`, `test_glm_attention`, `test_glm_inference`, `test_glm_full_compute`.

- [ ] **Step 4: Verify full suite**

Run:
```bash
ctest --preset linux-release-znver4
```

Expected: 100% tests passed — 22 total (same count as before; `test_loader`→`test_deepseek_loader`, `test_inference_kernels`→`test_deepseek_inference_kernels`), 0 failed.

- [ ] **Step 5: Verify rename purity and confirm no other references**

Run:
```bash
git status
rg -n "test_inference_kernels\b|test_loader\b" --glob '!build/**' --glob '!docs/**'
```

Expected: `git status` shows only the two renames + CMake/tag edits (no content drift in renamed files, confirmed by `git diff --stat -M` showing renames with 0 changed lines). The `rg` finds zero remaining references to the old target names outside `build/` and `docs/`.

- [ ] **Step 6: Commit any stragglers (should be none)**

Run:
```bash
git status
```

Expected: working tree clean. If the previous commits from Tasks 1-2 were made, nothing to commit here; otherwise make one commit:
```bash
git add tests/CMakeLists.txt tests/test_deepseek_loader.cpp tests/test_deepseek_inference_kernels.cu
git commit -m "test: separate DeepSeek test suite from GLM"
```
