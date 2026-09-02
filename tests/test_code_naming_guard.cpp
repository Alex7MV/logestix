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
    const std::string subsys = "(?:attn|csa|head|moe|pipeline|exp)";
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
    return std::regex("^" + fam + "_" + type + "(?:_" + var + ")?" + "(?:_exp)?" + "_" + tail + "$");
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
