# Model Code File Naming Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. (NOTE: AGENTS.md forbids Task-subagents — inline execution only.)
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename all model-code files in `libs/lsxcommon` + `tests/` per the scheme
`<family>_<type>[_<variant>]_<role>[_<subsystem>].<ext>` and pin the convention
with a guard test + a section in AGENTS.md.

**Architecture:** TDD: first the guard test (host-only, scans the source tree, red
on legacy names), then a mechanical rename of ~80 files + a scripted replacement of references
(includes, CMakeLists, test target names), verification of zero leftovers, a green guard,
a full rebuild and ctest. Class/namespace names are NOT changed.

**Tech Stack:** CMake 4.3 (VS 2026 bundled), Ninja, MSVC 14.51, Catch2 v3.7.0,
PowerShell 7 (all scripts — pwsh).

**Spec:** `docs/superpowers/specs/2026-08-27-model-code-naming-design.md`

**Tool paths** (AGENTS.md, this machine):

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
```

**Canonical rename map** (the single source for Task 2 and Task 3). Format:
`<old>|<new>` — file names with extensions; the root layout is determined automatically.

```
glm_convert.cpp|glm_moe_convert.cpp
glm_inference_load.cpp|glm_moe_load.cpp
glm_inference_load.cu|glm_moe_load.cu
glm_load_internal.h|glm_moe_load_internal.h
glm_v5_model.cpp|glm_moe_model.cpp
glm_v5_model.h|glm_moe_model.h
glm_inference.cu|glm_moe_inference.cu
glm_device_math.h|glm_moe_device_math.h
glm_expert_kernels.cuh|glm_moe_expert_kernels.cuh
glm_convert.h|glm_moe_convert.h
glm_inference_load.h|glm_moe_load.h
glm_inference.h|glm_moe_inference.h
glm_expert_cache.h|glm_moe_expert_cache.h
qwen_convert.cpp|qwen_dense_convert.cpp
qwen_convert.h|qwen_dense_convert.h
qwen_v5_model.cpp|qwen_dense_model.cpp
qwen_v5_model.h|qwen_dense_model.h
qwen_3_8_27b_model.cpp|qwen_dense_27b_model.cpp
qwen_3_8_27b_model.h|qwen_dense_27b_model.h
qwen_moe_inference_load.cpp|qwen_moe_load.cpp
qwen_moe_inference_load.h|qwen_moe_load.h
qwen_3_5_moe_model.cpp|qwen_moe_model.cpp
qwen_3_5_moe_model.h|qwen_moe_model.h
gemma4_convert.cpp|gemma_moe_convert.cpp
gemma4_convert.h|gemma_moe_convert.h
gemma4_inference_load.cpp|gemma_moe_load.cpp
gemma4_inference_load.h|gemma_moe_load.h
gemma4_inference.h|gemma_moe_inference.h
gemma_4_26b_a4b_model.cpp|gemma_moe_model.cpp
gemma_4_26b_a4b_model.h|gemma_moe_model.h
gemma_inference.cu|gemma_moe_inference.cu
kimi_convert.cpp|kimi_moe_convert.cpp
kimi_convert.h|kimi_moe_convert.h
kimi_inference_load.cpp|kimi_moe_load.cpp
kimi_inference_load.h|kimi_moe_load.h
kimi_v5_model.cpp|kimi_moe_model.cpp
kimi_v5_model.h|kimi_moe_model.h
kimi_inference.cu|kimi_moe_inference.cu
kimi_inference_kernels.h|kimi_moe_kernels.h
deepseek_v4_flash_model.cpp|deepseek_moe_model.cpp
deepseek_v4_flash_model.h|deepseek_moe_model.h
deepseek_inference_load.cu|deepseek_moe_load.cu
deepseek_inference_load.cuh|deepseek_moe_load.cuh
deepseek_inference_attn.cu|deepseek_moe_inference_attn.cu
deepseek_inference_csa.cu|deepseek_moe_inference_csa.cu
deepseek_inference_head.cu|deepseek_moe_inference_head.cu
deepseek_inference_moe.cu|deepseek_moe_inference_moe.cu
deepseek_inference_pipeline.cu|deepseek_moe_inference_pipeline.cu
deepseek_inference_csa.h|deepseek_moe_inference_csa.h
deepseek_norm_kernels.cu|deepseek_moe_norm_kernels.cu
deepseek_norm_kernels.h|deepseek_moe_norm_kernels.h
glm_test_fixture.h|glm_moe_test_fixture.h
deepseek_test_fixture.h|deepseek_moe_test_fixture.h
kimi_test_fixture.h|kimi_moe_test_fixture.h
gemma4_test_fixture.h|gemma_moe_test_fixture.h
qwen_test_fixture.h|qwen_dense_test_fixture.h
test_glm_attention.cpp|test_glm_moe_attention.cpp
test_glm_compute.cpp|test_glm_moe_layer_compute.cpp
test_glm_convert.cpp|test_glm_moe_convert.cpp
test_glm_dense_compute.cpp|test_glm_moe_dense_compute.cpp
test_glm_expert_cache.cpp|test_glm_moe_expert_cache.cpp
test_glm_experts.cpp|test_glm_moe_experts.cpp
test_glm_full_compute.cpp|test_glm_moe_full_compute.cpp
test_glm_inference.cpp|test_glm_moe_inference.cpp
test_glm_load_fast.cu|test_glm_moe_load_fast.cu
test_glm_loader.cpp|test_glm_moe_loader.cpp
test_glm_model_lifecycle.cu|test_glm_moe_model_lifecycle.cu
test_glm_model.cpp|test_glm_moe_model.cpp
test_glm_moe_compute.cpp|test_glm_moe_moe_compute.cpp
test_glm_moe_router.cpp|test_glm_moe_moe_router.cpp
test_glm_prefill_seqp.cpp|test_glm_moe_prefill_seqp.cpp
test_glm_prompt_framing.cpp|test_glm_moe_prompt_framing.cpp
test_glm_prompt_output.cpp|test_glm_moe_prompt_output.cpp
test_glm_selfcheck.cpp|test_glm_moe_selfcheck.cpp
test_deepseek_arena.cu|test_deepseek_moe_arena.cu
test_deepseek_inference_kernels.cu|test_deepseek_moe_inference_kernels.cu
test_deepseek_load_fast.cu|test_deepseek_moe_load_fast.cu
test_deepseek_loader.cpp|test_deepseek_moe_loader.cpp
test_deepseek_moe.cu|test_deepseek_moe_moe.cu
test_deepseek_prompt_output.cpp|test_deepseek_moe_prompt_output.cpp
test_deepseek_selfcheck.cpp|test_deepseek_moe_selfcheck.cpp
test_kimi_attention.cu|test_kimi_moe_attention.cu
test_kimi_convert.cpp|test_kimi_moe_convert.cpp
test_kimi_experts.cu|test_kimi_moe_experts.cu
test_kimi_inference.cpp|test_kimi_moe_inference.cpp
test_kimi_int4_dequant.cu|test_kimi_moe_int4_dequant.cu
test_kimi_int4_real.cu|test_kimi_moe_int4_real.cu
test_kimi_loader.cpp|test_kimi_moe_loader.cpp
test_kimi_model.cpp|test_kimi_moe_model.cpp
test_kimi_router.cu|test_kimi_moe_router.cu
test_kimi_selfcheck.cpp|test_kimi_moe_selfcheck.cpp
test_kimi_tokenizer.cpp|test_kimi_moe_tokenizer.cpp
test_gemma4_attn.cpp|test_gemma_moe_attn.cpp
test_gemma4_convert.cpp|test_gemma_moe_convert.cpp
test_gemma4_loader.cpp|test_gemma_moe_loader.cpp
test_gemma4_model.cpp|test_gemma_moe_model.cpp
test_gemma4_moe.cpp|test_gemma_moe_moe.cpp
test_gemma4_norm.cpp|test_gemma_moe_norm.cpp
test_gemma4_rotary.cpp|test_gemma_moe_rotary.cpp
test_gemma4_selfcheck.cpp|test_gemma_moe_selfcheck.cpp
test_qwen_convert.cpp|test_qwen_dense_convert.cpp
test_qwen_converter_integrity.cpp|test_qwen_dense_converter_integrity.cpp
test_qwen_loader.cpp|test_qwen_dense_loader.cpp
test_qwen_model.cpp|test_qwen_dense_model.cpp
test_qwen_kv_cache.cpp|test_qwen_dense_kv_cache.cpp
test_qwen_kv_parity.cpp|test_qwen_dense_kv_parity.cpp
test_qwen_linear_attn.cu|test_qwen_dense_linear_attn.cu
test_qwen_full_attn.cu|test_qwen_dense_full_attn.cu
test_qwen_full_attn_parallel.cu|test_qwen_dense_full_attn_parallel.cu
test_qwen_fused_linear_attn.cu|test_qwen_dense_fused_linear_attn.cu
test_qwen_fused_decode_attn.cu|test_qwen_dense_fused_decode_attn.cu
test_qwen_fused_mlp.cu|test_qwen_dense_fused_mlp.cu
test_qwen_mlp.cu|test_qwen_dense_mlp.cu
test_qwen_norm.cpp|test_qwen_dense_norm.cpp
test_qwen_rotary.cpp|test_qwen_dense_rotary.cpp
test_qwen_shuffle_reduce.cu|test_qwen_dense_shuffle_reduce.cu
test_qwen_nvfp4_dense.cu|test_qwen_dense_nvfp4.cu
test_qwen_selfcheck.cpp|test_qwen_dense_selfcheck.cpp
test_qwen_3_8_27b_model.cpp|test_qwen_dense_27b_model.cpp
test_qwen_3_8_27b_selfcheck.cpp|test_qwen_dense_27b_selfcheck.cpp
```

NOT renamed (allowlist, cross-type/shared): `qwen_inference.{h,cu}`,
`qwen_base_model.{cpp,h}`, `qwen_inference_load.{cpp,h}`, `qwen_device_math.h`,
`qwen_reduce.cuh`, `qwen_tc_gemm.{cu,h}`, `gemma_base_model.{cpp,h}`, `qwen_moe_*`
(already per the scheme), all shared files outside family prefixes.

---

### Task 1: Guard test (red)

**Files:**
- Create: `tests/test_code_naming_guard.cpp`
- Modify: `tests/CMakeLists.txt` (a block at the end of the file)

- [ ] **Step 1.1: Create the test**

```cpp
// tests/test_code_naming_guard.cpp
// Model code file naming convention guard (host-only, zero CUDA deps).
// Scheme: <family>_<type>[_<variant>]_<role>[_<subsystem>].<ext>
// Spec: docs/superpowers/specs/2026-08-27-model-code-naming-design.md
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

bool StartsWith(const std::string& s, const std::string& p)
{
    return s.rfind(p, 0) == 0;
}

std::regex SchemeRegex()
{
    const std::string fam    = "(?:glm|qwen|gemma|kimi|deepseek)";
    const std::string type   = "(?:moe|dense)";
    const std::string var    = "(?:27b)";
    const std::string subsys = "(?:attn|csa|head|moe|pipeline)";
    const std::string tail   =
        "(?:convert"
        "|load"
        "|load_internal"
        "|model"
        "|inference(?:_" + subsys + ")?"
        "|kernels"
        "|expert_kernels"
        "|expert_cache"
        "|device_math"
        "|norm_kernels"
        "|constants)";
    return std::regex("^" + fam + "_" + type + "(?:_" + var + ")?" + "_" + tail + "$");
}

const std::vector<std::string>& FamilyStems()
{
    static const std::vector<std::string> v =
        {"glm", "qwen", "gemma", "kimi", "deepseek"};
    return v;
}

// Family-prefixed files that intentionally carry NO type marker:
// shared between model types or base infrastructure of the family.
bool AllowedCommon(const std::string& stem)
{
    static const std::vector<std::string> kAllowed = {
        "qwen_inference",     // kernels/GEMM API+impl shared by dense+moe
        "qwen_base_model",
        "qwen_inference_load",
        "qwen_device_math",
        "qwen_reduce",
        "qwen_tc_gemm",
        "gemma_base_model",
    };
    return std::find(kAllowed.begin(), kAllowed.end(), stem) != kAllowed.end();
}

bool IsFamilyPrefixed(const std::string& stem)
{
    if (StartsWith(stem, "gemma4")) // legacy glued version prefix
        return true;
    for (const auto& f : FamilyStems())
        if (stem == f || StartsWith(stem, f + "_"))
            return true;
    return false;
}

void ScanDir(const fs::path& dir, const std::regex& scheme,
             std::vector<std::string>& out)
{
    std::error_code ec;
    if (!fs::exists(dir, ec))
    {
        out.push_back("MISSING DIR: " + dir.string());
        return;
    }
    for (auto& e : fs::recursive_directory_iterator(dir))
    {
        if (!e.is_regular_file())
            continue;
        const std::string ext = e.path().extension().string();
        if (ext != ".h" && ext != ".cuh" && ext != ".cpp" && ext != ".cu")
            continue;
        const std::string stem = e.path().stem().string();
        if (!IsFamilyPrefixed(stem) || AllowedCommon(stem))
            continue;
        if (!std::regex_match(stem, scheme))
        {
            out.push_back(
                e.path().lexically_proximate(fs::path(LSX_SOURCE_DIR)).string() +
                "  -> expected <family>_<type>[_<variant>]_<role>[_<subsystem>]>");
        }
    }
}

} // namespace

TEST_CASE("naming guard: family files match family_type_[variant_]role[_subsystem]",
          "[naming][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const auto scheme = SchemeRegex();

    std::vector<std::string> violations;
    ScanDir(root / "libs" / "lsxcommon" / "src", scheme, violations);
    ScanDir(root / "libs" / "lsxcommon" / "include" / "lsxcommon", scheme, violations);

    std::sort(violations.begin(), violations.end());
    std::string joined;
    for (const auto& v : violations)
    {
        joined += v;
        joined += '\n';
    }
    INFO("naming violations (" << violations.size() << "):\n" << joined);
    REQUIRE(violations.empty());
}
```

- [ ] **Step 1.2: Register the target in `tests/CMakeLists.txt` (at the end of the file)**

```cmake
# Guard: model code file naming convention (host-only, no lsxcommon deps).
add_executable(test_code_naming_guard test_code_naming_guard.cpp)
target_link_libraries(test_code_naming_guard PRIVATE Catch2::Catch2WithMain)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD)
target_compile_definitions(test_code_naming_guard
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD}\"")
add_test(NAME test_code_naming_guard COMMAND test_code_naming_guard)
```

- [ ] **Step 1.3: Build and run — the test MUST be red**

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $cmake --build build --config Debug --target test_code_naming_guard
& $ctest --preset windows-debug -R code_naming --output-on-failure
```

Expected: FAIL, ~49 violations (glm 13, qwen 9, gemma 8, kimi 8, deepseek 11), the list
matches the rename map (every row of the Task 2 map except the allowlist). Do not
commit anything — an atomic commit in Task 4.

### Task 2: Mechanical file rename (git mv)

**Files:** all pairs from the rename map (see the top of the plan).

- [ ] **Step 2.0: Write the map to an artifact file** (the full content — the
  "Canonical rename map" block from the plan header, verbatim; the file is then read by Task 3 and Task 5)

```powershell
$mapText = @'
glm_convert.cpp|glm_moe_convert.cpp
glm_inference_load.cpp|glm_moe_load.cpp
glm_inference_load.cu|glm_moe_load.cu
glm_load_internal.h|glm_moe_load_internal.h
glm_v5_model.cpp|glm_moe_model.cpp
glm_v5_model.h|glm_moe_model.h
glm_inference.cu|glm_moe_inference.cu
glm_device_math.h|glm_moe_device_math.h
glm_expert_kernels.cuh|glm_moe_expert_kernels.cuh
glm_convert.h|glm_moe_convert.h
glm_inference_load.h|glm_moe_load.h
glm_inference.h|glm_moe_inference.h
glm_expert_cache.h|glm_moe_expert_cache.h
qwen_convert.cpp|qwen_dense_convert.cpp
qwen_convert.h|qwen_dense_convert.h
qwen_v5_model.cpp|qwen_dense_model.cpp
qwen_v5_model.h|qwen_dense_model.h
qwen_3_8_27b_model.cpp|qwen_dense_27b_model.cpp
qwen_3_8_27b_model.h|qwen_dense_27b_model.h
qwen_moe_inference_load.cpp|qwen_moe_load.cpp
qwen_moe_inference_load.h|qwen_moe_load.h
qwen_3_5_moe_model.cpp|qwen_moe_model.cpp
qwen_3_5_moe_model.h|qwen_moe_model.h
gemma4_convert.cpp|gemma_moe_convert.cpp
gemma4_convert.h|gemma_moe_convert.h
gemma4_inference_load.cpp|gemma_moe_load.cpp
gemma4_inference_load.h|gemma_moe_load.h
gemma4_inference.h|gemma_moe_inference.h
gemma_4_26b_a4b_model.cpp|gemma_moe_model.cpp
gemma_4_26b_a4b_model.h|gemma_moe_model.h
gemma_inference.cu|gemma_moe_inference.cu
kimi_convert.cpp|kimi_moe_convert.cpp
kimi_convert.h|kimi_moe_convert.h
kimi_inference_load.cpp|kimi_moe_load.cpp
kimi_inference_load.h|kimi_moe_load.h
kimi_v5_model.cpp|kimi_moe_model.cpp
kimi_v5_model.h|kimi_moe_model.h
kimi_inference.cu|kimi_moe_inference.cu
kimi_inference_kernels.h|kimi_moe_kernels.h
deepseek_v4_flash_model.cpp|deepseek_moe_model.cpp
deepseek_v4_flash_model.h|deepseek_moe_model.h
deepseek_inference_load.cu|deepseek_moe_load.cu
deepseek_inference_load.cuh|deepseek_moe_load.cuh
deepseek_inference_attn.cu|deepseek_moe_inference_attn.cu
deepseek_inference_csa.cu|deepseek_moe_inference_csa.cu
deepseek_inference_head.cu|deepseek_moe_inference_head.cu
deepseek_inference_moe.cu|deepseek_moe_inference_moe.cu
deepseek_inference_pipeline.cu|deepseek_moe_inference_pipeline.cu
deepseek_inference_csa.h|deepseek_moe_inference_csa.h
deepseek_norm_kernels.cu|deepseek_moe_norm_kernels.cu
deepseek_norm_kernels.h|deepseek_moe_norm_kernels.h
glm_test_fixture.h|glm_moe_test_fixture.h
deepseek_test_fixture.h|deepseek_moe_test_fixture.h
kimi_test_fixture.h|kimi_moe_test_fixture.h
gemma4_test_fixture.h|gemma_moe_test_fixture.h
qwen_test_fixture.h|qwen_dense_test_fixture.h
test_glm_attention.cpp|test_glm_moe_attention.cpp
test_glm_compute.cpp|test_glm_moe_layer_compute.cpp
test_glm_convert.cpp|test_glm_moe_convert.cpp
test_glm_dense_compute.cpp|test_glm_moe_dense_compute.cpp
test_glm_expert_cache.cpp|test_glm_moe_expert_cache.cpp
test_glm_experts.cpp|test_glm_moe_experts.cpp
test_glm_full_compute.cpp|test_glm_moe_full_compute.cpp
test_glm_inference.cpp|test_glm_moe_inference.cpp
test_glm_load_fast.cu|test_glm_moe_load_fast.cu
test_glm_loader.cpp|test_glm_moe_loader.cpp
test_glm_model_lifecycle.cu|test_glm_moe_model_lifecycle.cu
test_glm_model.cpp|test_glm_moe_model.cpp
test_glm_moe_compute.cpp|test_glm_moe_moe_compute.cpp
test_glm_moe_router.cpp|test_glm_moe_moe_router.cpp
test_glm_prefill_seqp.cpp|test_glm_moe_prefill_seqp.cpp
test_glm_prompt_framing.cpp|test_glm_moe_prompt_framing.cpp
test_glm_prompt_output.cpp|test_glm_moe_prompt_output.cpp
test_glm_selfcheck.cpp|test_glm_moe_selfcheck.cpp
test_deepseek_arena.cu|test_deepseek_moe_arena.cu
test_deepseek_inference_kernels.cu|test_deepseek_moe_inference_kernels.cu
test_deepseek_load_fast.cu|test_deepseek_moe_load_fast.cu
test_deepseek_loader.cpp|test_deepseek_moe_loader.cpp
test_deepseek_moe.cu|test_deepseek_moe_moe.cu
test_deepseek_prompt_output.cpp|test_deepseek_moe_prompt_output.cpp
test_deepseek_selfcheck.cpp|test_deepseek_moe_selfcheck.cpp
test_kimi_attention.cu|test_kimi_moe_attention.cu
test_kimi_convert.cpp|test_kimi_moe_convert.cpp
test_kimi_experts.cu|test_kimi_moe_experts.cu
test_kimi_inference.cpp|test_kimi_moe_inference.cpp
test_kimi_int4_dequant.cu|test_kimi_moe_int4_dequant.cu
test_kimi_int4_real.cu|test_kimi_moe_int4_real.cu
test_kimi_loader.cpp|test_kimi_moe_loader.cpp
test_kimi_model.cpp|test_kimi_moe_model.cpp
test_kimi_router.cu|test_kimi_moe_router.cu
test_kimi_selfcheck.cpp|test_kimi_moe_selfcheck.cpp
test_kimi_tokenizer.cpp|test_kimi_moe_tokenizer.cpp
test_gemma4_attn.cpp|test_gemma_moe_attn.cpp
test_gemma4_convert.cpp|test_gemma_moe_convert.cpp
test_gemma4_loader.cpp|test_gemma_moe_loader.cpp
test_gemma4_model.cpp|test_gemma_moe_model.cpp
test_gemma4_moe.cpp|test_gemma_moe_moe.cpp
test_gemma4_norm.cpp|test_gemma_moe_norm.cpp
test_gemma4_rotary.cpp|test_gemma_moe_rotary.cpp
test_gemma4_selfcheck.cpp|test_gemma_moe_selfcheck.cpp
test_qwen_convert.cpp|test_qwen_dense_convert.cpp
test_qwen_converter_integrity.cpp|test_qwen_dense_converter_integrity.cpp
test_qwen_loader.cpp|test_qwen_dense_loader.cpp
test_qwen_model.cpp|test_qwen_dense_model.cpp
test_qwen_kv_cache.cpp|test_qwen_dense_kv_cache.cpp
test_qwen_kv_parity.cpp|test_qwen_dense_kv_parity.cpp
test_qwen_linear_attn.cu|test_qwen_dense_linear_attn.cu
test_qwen_full_attn.cu|test_qwen_dense_full_attn.cu
test_qwen_full_attn_parallel.cu|test_qwen_dense_full_attn_parallel.cu
test_qwen_fused_linear_attn.cu|test_qwen_dense_fused_linear_attn.cu
test_qwen_fused_decode_attn.cu|test_qwen_dense_fused_decode_attn.cu
test_qwen_fused_mlp.cu|test_qwen_dense_fused_mlp.cu
test_qwen_mlp.cu|test_qwen_dense_mlp.cu
test_qwen_norm.cpp|test_qwen_dense_norm.cpp
test_qwen_rotary.cpp|test_qwen_dense_rotary.cpp
test_qwen_shuffle_reduce.cu|test_qwen_dense_shuffle_reduce.cu
test_qwen_nvfp4_dense.cu|test_qwen_dense_nvfp4.cu
test_qwen_selfcheck.cpp|test_qwen_dense_selfcheck.cpp
test_qwen_3_8_27b_model.cpp|test_qwen_dense_27b_model.cpp
test_qwen_3_8_27b_selfcheck.cpp|test_qwen_dense_27b_selfcheck.cpp
'@
New-Item -ItemType Directory -Force "$env:TEMP\opencode" | Out-Null
Set-Content -LiteralPath "$env:TEMP\opencode\lsx_rename_map.txt" -Value $mapText -Encoding utf8NoBOM
```

- [ ] **Step 2.1: Execute git mv per the map**

```powershell
$moves = [ordered]@{}
Get-Content "$env:TEMP\opencode\lsx_rename_map.txt" | ForEach-Object {
  if ($_ -match '^([^|]+)\|([^|]+)$') { $moves[$matches[1]] = $matches[2] }
}
"pairs loaded: $($moves.Count)"
$roots = @('libs\lsxcommon\src', 'libs\lsxcommon\include\lsxcommon', 'tests')
$moved = 0
foreach ($d in $roots) {
  foreach ($o in $moves.Keys) {
    $src = Join-Path $d $o
    if (Test-Path -LiteralPath $src) {
      git mv $src (Join-Path $d $moves[$o]); $moved++
    }
  }
}
"moved: $moved"
```

Expected: `pairs loaded: 120`, `moved: 120` (38 lib src + 13 lib include + 5 fixtures +
64 test files).

- [ ] **Step 2.2: Residual check**

```powershell
git status --short | Measure-Object -Line | Select-Object -ExpandProperty Lines
Get-ChildItem libs\lsxcommon\src -Filter 'glm_*','qwen_*','gemma*','kimi_*','deepseek_*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
```

Expected: `git status` contains only `R` records (renames) from the map + 2 new files
from Task 1 (`A tests/test_code_naming_guard.cpp`, `M tests/CMakeLists.txt`). In src/ not a single
name from the old column should remain.

### Task 3: Scripted reference replacement (includes, CMake, target names)

**Files (modify):** `libs/**/*.{cpp,h,cu,cuh}`, `apps/**/*.{cpp,h,cu,cuh}`,
`tests/**/*.{cpp,cu,h,txt}`, `libs/lsxcommon/CMakeLists.txt`, the root
`CMakeLists.txt`. NOT touched: `third_party/**`, `build/**`, `docs/**`, `*.md`
(AGENTS.md — a separate Task 5).

- [ ] **Step 3.1: Run the replacement**

```powershell
$moves = [ordered]@{}
Get-Content "$env:TEMP\opencode\lsx_rename_map.txt" | ForEach-Object {
  if ($_ -match '^([^|]+)\|([^|]+)$') { $moves[$matches[1]] = $matches[2] }
}
# extensionless pairs for the test targets (add_executable/add_test NAME)
$stemMap = [ordered]@{}
foreach ($k in $moves.Keys) {
  if ($k -like 'test_*') {
    $stemMap[[IO.Path]::GetFileNameWithoutExtension($k)] =
      [IO.Path]::GetFileNameWithoutExtension($moves[$k])
  }
}
# longest-first — protection against prefix overlaps
$all = @()
foreach ($k in $moves.Keys)    { $all += ,@($k, $moves[$k]) }
foreach ($k in $stemMap.Keys)  { $all += ,@($k, $stemMap[$k]) }
$all = $all | Sort-Object { $_[0].Length } -Descending

$targets = @(Get-ChildItem libs, apps, tests -Recurse -File -Include *.cpp,*.cc,*.h,*.cu,*.cuh,*.txt) +
           @(Get-Item CMakeLists.txt -ErrorAction SilentlyContinue)
$changed = 0
foreach ($f in $targets) {
  $text = [IO.File]::ReadAllText($f.FullName)
  $orig = $text
  foreach ($p in $all) {
    $text = [regex]::Replace($text, ('(?<![A-Za-z0-9_])' + [regex]::Escape($p[0]) + '(?![A-Za-z0-9_])'), $p[1])
  }
  if ($text -ne $orig) { [IO.File]::WriteAllText($f.FullName, $text); $changed++ }
}
"changed files: $changed"
```

- [ ] **Step 3.2: Zero-residual verification**

```powershell
$leftovers = 0
foreach ($k in ($moves.Keys + $stemMap.Keys)) {
  $c = (rg -F -l -- $k libs apps tests CMakeLists.txt | Measure-Object).Count
  if ($c -gt 0) { "LEFTOVER: $k ($c)"; $leftovers++ }
}
"leftover tokens: $leftovers"
```

Expected: `leftover tokens: 0`. If >0 — fix individually by hand and repeat.

### Task 4: Green guard + rebuild + tests

- [ ] **Step 4.1: Guard green**

```powershell
& $cmake --build build --config Debug --target test_code_naming_guard
& $ctest --preset windows-debug -R code_naming --output-on-failure
```

Expected: `100% tests passed, 1 test` (`test_code_naming_guard` PASSED).

- [ ] **Step 4.2: Full rebuild**

```powershell
& $cmake --build build --config Debug
```

Expected: BUILD SUCCEEDED, 0 errors (all includes already replaced; `#pragma once` — without
guard macros, no further header edits are required).

- [ ] **Step 4.3: ctest (Windows restriction: only qwen|gemma models + the host-only suite)**

```powershell
& $ctest --preset windows-debug -R 'code_naming|geometry_guard|core|fast_load|schema|factory|runtime|integrity|hardware|resource_governor|runtime_tuner|lifecycle|expert_store|tokenizer|unicode|pretokenizer|path_registry|vram|sampling|pack|translator|weld|mosaic|frontend|generation_cli|prompt|nvfp4|kv_|qwen|gemma' --output-on-failure
```

Expected: all selected tests PASSED (the `qwen|gemma` filter catches the renamed
`test_qwen_dense_*` / `test_qwen_moe_*` / `test_gemma_moe_*`; GLM/DeepSeek/Kimi model
tests do not run on Windows — an AGENTS.md restriction).

- [ ] **Step 4.4: Atomic commit**

```powershell
git add -A
git commit -m "refactor: unify model code file naming to family_type_role scheme"
```

The same commit includes the spec edit (the `qwen_inference.cu` row moved to the allowlist —
already made in `docs/superpowers/specs/2026-08-27-model-code-naming-design.md` during
planning).

### Task 5: AGENTS.md — paths + convention

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 5.1: Replace the old file names in AGENTS.md (the same map, longest-first)**

```powershell
$moves = [ordered]@{}
Get-Content "$env:TEMP\opencode\lsx_rename_map.txt" | ForEach-Object {
  if ($_ -match '^([^|]+)\|([^|]+)$') { $moves[$matches[1]] = $matches[2] }
}
$all = @($moves.Keys | Sort-Object Length -Descending | ForEach-Object { ,@($_, $moves[$_]) })
$t = [IO.File]::ReadAllText('AGENTS.md')
foreach ($p in $all) {
  $t = [regex]::Replace($t, ('(?<![A-Za-z0-9_])' + [regex]::Escape($p[0]) + '(?![A-Za-z0-9_])'), $p[1])
}
[IO.File]::WriteAllText('AGENTS.md', $t)
```

- [ ] **Step 5.2: Add the convention section (after `## Conventions`)**

```markdown
## Model Code File Naming Convention

The model-code file scheme in `libs/lsxcommon/{src,include}` and `tests/`:

`<family>_<type>[_<variant>]_<role>[_<subsystem>].<ext>`

- `family` ∈ {glm, qwen, gemma, kimi, deepseek}
- `type` ∈ {moe, dense} — always required
- `variant` — only on a collision of two models of one type (example: `qwen_dense_27b_model`)
- `role` — short vocabulary: convert | load | load_internal | model | inference | kernels |
  expert_kernels | expert_cache | device_math | norm_kernels | constants
- `subsystem` — split of a large kernel: attn | csa | head | moe | pipeline
- Model versions are removed from file names; they live in class names and `kType`.
- Cross-type/base files of a family do NOT get a type marker
  (`qwen_inference.*`, `qwen_base_model.*`, `qwen_device_math.h`, …) and are in the allowlist.
- Control: `tests/test_code_naming_guard.cpp` — scans `libs/lsxcommon/{src,include}`,
  must be green in CI; the family/role/subsystem registries are extended in the same commit
  as the new model's files (Model Ingestion Protocol, step 3).
```

- [ ] **Step 5.3: Documentation commit**

```powershell
git add AGENTS.md
git commit -m "docs: record model code naming convention in AGENTS.md"
```

---

## Self-Review (done during planning)

1. **Spec coverage:** scheme (item 2) — Task 1/2; map (item 3) — Task 2; allowlist (item 4) — guard
   allowlist + spec refinement; tests (item 5) — Task 2 map; guard (item 6) — Task 1; accompanying
   edits (item 7) — Task 3/5; TDD order and commits (item 8) — Task 4/5; criteria (item 9) — Task 4.
2. **Placeholders:** the rename map is inlined in full (Task 2 Step 2.0, 120 pairs), the guard code is
   complete, the commands are exact; Task 3/5 read the map artifact from a temp directory (the path and the
   recovery method — re-running Step 2.0 — are stated explicitly).
3. **Type consistency:** the names in the map, the guard regex (`tail`), and the AGENTS section use
   the same role/subsystem vocabulary (checked line by line against the final names).
