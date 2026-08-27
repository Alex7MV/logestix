# Model Code File Naming Unification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. (NOTE: AGENTS.md forbids Task-subagents — inline execution only.)
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Переименовать все файлы кода моделей в `libs/lsxcommon` + `tests/` по схеме
`<family>_<type>[_<variant>]_<role>[_<subsystem>].<ext>` и зафиксировать конвенцию
guard-тестом + разделом в AGENTS.md.

**Architecture:** TDD: сначала guard-тест (host-only, сканирует дерево исходников, красный
на legacy-именах), затем механический ренейм ~80 файлов + скриптовая замена ссылок
(include, CMakeLists, имена тестовых целей), верификация нулевых остатков, зелёный guard,
полная пересборка и ctest. Имена классов/неймспейсов НЕ меняются.

**Tech Stack:** CMake 4.3 (VS 2026 bundled), Ninja, MSVC 14.51, Catch2 v3.7.0,
PowerShell 7 (все скрипты — pwsh).

**Spec:** `docs/superpowers/specs/2026-08-27-model-code-naming-design.md`

**Инструментальные пути** (AGENTS.md,这台 машина):

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
```

**Каноническая карта ренеймов** (единый источник для Task 2 и Task 3). Формат:
`<old>|<new>` — имена файлов с расширением; раскладка по корню определяется автоматически.

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

НЕ переименовываются (allowlist, кросс-типовые/общие): `qwen_inference.{h,cu}`,
`qwen_base_model.{cpp,h}`, `qwen_inference_load.{cpp,h}`, `qwen_device_math.h`,
`qwen_reduce.cuh`, `qwen_tc_gemm.{cu,h}`, `gemma_base_model.{cpp,h}`, `qwen_moe_*`
(уже по схеме), все общие файлы вне family-префиксов.

---

### Task 1: Guard-тест (красный)

**Files:**
- Create: `tests/test_code_naming_guard.cpp`
- Modify: `tests/CMakeLists.txt` (блок в конец файла)

- [ ] **Step 1.1: Создать тест**

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

- [ ] **Step 1.2: Зарегистрировать цель в `tests/CMakeLists.txt` (в конец файла)**

```cmake
# Guard: model code file naming convention (host-only, no lsxcommon deps).
add_executable(test_code_naming_guard test_code_naming_guard.cpp)
target_link_libraries(test_code_naming_guard PRIVATE Catch2::Catch2WithMain)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}" LSX_SRC_DIR_FWD)
target_compile_definitions(test_code_naming_guard
    PRIVATE "LSX_SOURCE_DIR=\"${LSX_SRC_DIR_FWD}\"")
add_test(NAME test_code_naming_guard COMMAND test_code_naming_guard)
```

- [ ] **Step 1.3: Собрать и запустить — тест ОБЯЗАН быть красным**

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ctest = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"
& $cmake --build build --config Debug --target test_code_naming_guard
& $ctest --preset windows-debug -R code_naming --output-on-failure
```

Expected: FAIL, ~49 нарушений (glm 13, qwen 9, gemma 8, kimi 8, deepseek 11), список
совпадает с картой ренейма (каждая строка карты из Task 2, кроме allowlist). Ничего не
коммитим — атомарный коммит в Task 4.

### Task 2: Механический ренейм файлов (git mv)

**Files:** все пары из карты ренейма (см. в начале плана).

- [ ] **Step 2.0: Записать карту в файл-артефакт** (полное содержимое — блок
  «Каноническая карта ренеймов» из шапки плана, дословно; файл потом читают Task 3 и Task 5)

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

- [ ] **Step 2.1: Выполнить git mv по карте**

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
64 test файлов).

- [ ] **Step 2.2: Контрольная сверка остатков**

```powershell
git status --short | Measure-Object -Line | Select-Object -ExpandProperty Lines
Get-ChildItem libs\lsxcommon\src -Filter 'glm_*','qwen_*','gemma*','kimi_*','deepseek_*' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name
```

Expected: `git status` содержит только `R`-записи (renames) из карты + 2 новых файла
Task 1 (`A tests/test_code_naming_guard.cpp`, `M tests/CMakeLists.txt`). В src/ не должно
остаться ни одного имени из колонки old.

### Task 3: Скриптовая замена ссылок (include, CMake, имена целей)

**Files (modify):** `libs/**/*.{cpp,h,cu,cuh}`, `apps/**/*.{cpp,h,cu,cuh}`,
`tests/**/*.{cpp,cu,h,txt}`, `libs/lsxcommon/CMakeLists.txt`, корневой
`CMakeLists.txt`. НЕ трогаем: `third_party/**`, `build/**`, `docs/**`, `*.md`
(AGENTS.md — отдельный Task 5).

- [ ] **Step 3.1: Запустить замену**

```powershell
$moves = [ordered]@{}
Get-Content "$env:TEMP\opencode\lsx_rename_map.txt" | ForEach-Object {
  if ($_ -match '^([^|]+)\|([^|]+)$') { $moves[$matches[1]] = $matches[2] }
}
# пары без расширения для тестовых целей (add_executable/add_test NAME)
$stemMap = [ordered]@{}
foreach ($k in $moves.Keys) {
  if ($k -like 'test_*') {
    $stemMap[[IO.Path]::GetFileNameWithoutExtension($k)] =
      [IO.Path]::GetFileNameWithoutExtension($moves[$k])
  }
}
# longest-first — защита от префиксных наложений
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

- [ ] **Step 3.2: Верификация нулевых остатков**

```powershell
$leftovers = 0
foreach ($k in ($moves.Keys + $stemMap.Keys)) {
  $c = (rg -F -l -- $k libs apps tests CMakeLists.txt | Measure-Object).Count
  if ($c -gt 0) { "LEFTOVER: $k ($c)"; $leftovers++ }
}
"leftover tokens: $leftovers"
```

Expected: `leftover tokens: 0`. Если >0 — точечно исправить вручную и повторить.

### Task 4: Зелёный guard + пересборка + тесты

- [ ] **Step 4.1: Guard зелёный**

```powershell
& $cmake --build build --config Debug --target test_code_naming_guard
& $ctest --preset windows-debug -R code_naming --output-on-failure
```

Expected: `100% tests passed, 1 test` (`test_code_naming_guard` PASSED).

- [ ] **Step 4.2: Полная пересборка**

```powershell
& $cmake --build build --config Debug
```

Expected: BUILD SUCCEEDED, 0 errors (все include уже заменены; `#pragma once` — без
guard-макросов, дополнительных правок заголовков не требуется).

- [ ] **Step 4.3: ctest (Windows-ограничение: только qwen|gemma модели + host-only suite)**

```powershell
& $ctest --preset windows-debug -R 'code_naming|geometry_guard|core|fast_load|schema|factory|runtime|integrity|hardware|resource_governor|runtime_tuner|lifecycle|expert_store|tokenizer|unicode|pretokenizer|path_registry|vram|sampling|pack|translator|weld|mosaic|frontend|generation_cli|prompt|nvfp4|kv_|qwen|gemma' --output-on-failure
```

Expected: все отобранные тесты PASSED (фильтр `qwen|gemma` ловит переименованные
`test_qwen_dense_*` / `test_qwen_moe_*` / `test_gemma_moe_*`; GLM/DeepSeek/Kimi модельные
тесты на Windows не запускаются — ограничение AGENTS.md).

- [ ] **Step 4.4: Атомарный коммит**

```powershell
git add -A
git commit -m "refactor: unify model code file naming to family_type_role scheme"
```

В этот же коммит входит правка спеки (строка `qwen_inference.cu` перенесена в allowlist —
уже внесена в `docs/superpowers/specs/2026-08-27-model-code-naming-design.md` при
планировании).

### Task 5: AGENTS.md — пути + конвенция

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 5.1: Заменить старые имена файлов в AGENTS.md (та же карта, longest-first)**

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

- [ ] **Step 5.2: Добавить раздел конвенции (после `## Conventions`)**

```markdown
## Model Code File Naming Convention

Схема файлов кода моделей в `libs/lsxcommon/{src,include}` и `tests/`:

`<family>_<type>[_<variant>]_<role>[_<subsystem>].<ext>`

- `family` ∈ {glm, qwen, gemma, kimi, deepseek}
- `type` ∈ {moe, dense} — обязателен всегда
- `variant` — только при коллизии двух моделей одного type (пример: `qwen_dense_27b_model`)
- `role` — короткий словарь: convert | load | load_internal | model | inference | kernels |
  expert_kernels | expert_cache | device_math | norm_kernels | constants
- `subsystem` — разбиение крупного ядра: attn | csa | head | moe | pipeline
- Версии моделей из имён файлов убраны; живут в именах классов и `kType`.
- Кросс-типовые/базовые файлы семейства НЕ получают маркер типа
  (`qwen_inference.*`, `qwen_base_model.*`, `qwen_device_math.h`, …) и внесены в allowlist.
- Контроль: `tests/test_code_naming_guard.cpp` — сканирует `libs/lsxcommon/{src,include}`,
  обязан быть зелёным в CI; реестры family/role/subsystem расширяются в том же коммите,
  что и файлы новой модели (Model Ingestion Protocol, шаг 3).
```

- [ ] **Step 5.3: Коммит документации**

```powershell
git add AGENTS.md
git commit -m "docs: record model code naming convention in AGENTS.md"
```

---

## Self-Review (выполнено при планировании)

1. **Spec coverage:** схема (п.2) — Task 1/2; карта (п.3) — Task 2; allowlist (п.4) — guard
   allowlist + уточнение спеки; тесты (п.5) — Task 2 карта; guard (п.6) — Task 1; сопутствующие
   правки (п.7) — Task 3/5; порядок TDD и коммиты (п.8) — Task 4/5; критерии (п.9) — Task 4.
2. **Placeholders:** карта ренейма инлайнена полностью (Task 2 Step 2.0, 120 пар), guard-код
   полный, команды точные; Task 3/5 читают артефакт карты из temp-каталога (путь и способ
   восстановления — перезапуск Step 2.0 — указаны явно).
3. **Type consistency:** имена в карте, guard-regex (`tail`), и AGENTS-раздел используют
   один и тот же словарь ролей/подсистем (проверено построчно против финальных имён).
