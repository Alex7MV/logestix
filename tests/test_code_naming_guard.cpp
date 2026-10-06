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
    const std::string fam    = "(?:glm|qwen|gemma|kimi|deepseek|nemotron|gigachat|ornith|granite|mimo|phi|alice|xing)";
    const std::string type   = "(?:moe|dense)";
    const std::string var    = "(?:27b|flash|36|v41|nvfp4)";
    const std::string subsys = "(?:attn|csa|head|moe|pipeline|exp)";
    const std::string tail   =
        "(?:convert"
        "|load"
        "|repack"
        "|load_internal"
        "|model"
        "|inference(?:_" + subsys + ")?"
        "|kernels"
        "|expert_kernels"
        "|expert_cache"
        "|device_math"
        "|dequant"
        "|norm_kernels"
        "|fp8"
        "|constants)";
    return std::regex("^" + fam + "_" + type + "(?:_" + var + ")?" + "(?:_exp)?" + "_" + tail + "$");
}

const std::vector<std::string>& FamilyStems()
{
    static const std::vector<std::string> v =
        {"glm", "qwen", "gemma", "kimi", "deepseek", "nemotron", "gigachat",
         "ornith", "granite", "mimo", "phi", "alice", "xing"};
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
        "gemma_base_model",
        "deepseek_moe_tool_format",  // V4/V4.1 DSML codec shared by both
        "qwen_tool_format",          // ChatML codec shared by dense + MoE
        "gemma_tool_format",         // Gemma-4 tool codec
        "gigachat_tool_format",      // GigaChat chatml-v5 GCML codec
        "mimo_tool_format",          // MiMo ChatML + <tool_call> codec
        "alice_tool_format",         // AliceAI native chat/tool codec
        "xing_moe_tool_format",       // Xing4.0 <_user>/<_bot> + ZWSP tool codec
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

TEST_CASE("naming guard accepts qwen_moe_exp_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("qwen_moe_exp_convert"));
    CHECK(m("qwen_moe_exp_inference"));
    CHECK(m("qwen_moe_exp_model"));
    CHECK(m("qwen_moe_exp_load"));
    CHECK(m("qwen_moe_exp_constants"));
}

TEST_CASE("naming guard accepts deepseek_moe_v41_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("deepseek_moe_v41_constants"));
    CHECK(m("deepseek_moe_v41_load"));
    CHECK(m("deepseek_moe_v41_convert"));
    CHECK(m("deepseek_moe_v41_model"));
    CHECK(m("deepseek_moe_v41_inference"));
    CHECK(m("deepseek_moe_v41_inference_attn"));
    CHECK(m("deepseek_moe_v41_device_math"));
    CHECK(m("deepseek_moe_v41_dequant"));
}

TEST_CASE("naming guard accepts nemotron_moe_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("nemotron_moe_constants"));
    CHECK(m("nemotron_moe_convert"));
    CHECK(m("nemotron_moe_load"));
    CHECK(m("nemotron_moe_expert_cache"));
    CHECK(m("nemotron_moe_model"));
    CHECK(m("nemotron_moe_inference"));
    CHECK(m("nemotron_moe_inference_pipeline"));
    CHECK(m("nemotron_moe_inference_attn"));
    CHECK(m("nemotron_moe_kernels"));
    CHECK(m("nemotron_moe_device_math"));
}

TEST_CASE("naming guard accepts nemotron_moe_nvfp4_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("nemotron_moe_nvfp4_constants"));
    CHECK(m("nemotron_moe_nvfp4_convert"));
    CHECK(m("nemotron_moe_nvfp4_load"));
    CHECK(m("nemotron_moe_nvfp4_dequant"));
    CHECK(m("nemotron_moe_nvfp4_model"));
}

TEST_CASE("naming guard accepts granite_dense_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("granite_dense_constants"));
    CHECK(m("granite_dense_convert"));
    CHECK(m("granite_dense_load"));
    CHECK(m("granite_dense_model"));
    CHECK(m("granite_dense_inference"));
}

TEST_CASE("naming guard accepts ornith_moe_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("ornith_moe_constants"));
    CHECK(m("ornith_moe_convert"));
    CHECK(m("ornith_moe_load"));
    CHECK(m("ornith_moe_model"));
    CHECK(m("ornith_moe_inference"));
    CHECK(m("ornith_moe_expert_cache"));
}

TEST_CASE("naming guard accepts phi_dense_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("phi_dense_constants"));
    CHECK(m("phi_dense_convert"));
    CHECK(m("phi_dense_load"));
    CHECK(m("phi_dense_model"));
    CHECK(m("phi_dense_inference"));
}
