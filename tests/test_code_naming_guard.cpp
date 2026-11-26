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
    const std::string fam    = "(?:glm|qwen|gemma|kimi|deepseek|nemotron|gigachat|ornith|granite|mimo|phi|dory|alice|xing|mistral|minimax|hy|flint|ling|beam|step|inkling|muse)";
    const std::string type   = "(?:moe|dense)";
    const std::string var    = "(?:27b|37b|8b|flash|36|v41|pro|nvfp4|switch|simulmt|large4|k3|ultra|base|loop|varloop|1017)";
    const std::string subsys = "(?:attn|csa|head|moe|pipeline|exp|indexer)";
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
    // Two optional variant slots, not one: granite-switch needs an architecture
    // marker (`switch`) AND a size marker (`8b`) to distinguish it from the
    // plain dense `granite_dense_*` files, e.g.
    // granite_dense_switch_8b_constants.h. Both slots draw from the same
    // vocabulary, so this widens the grammar without widening the vocabulary.
    return std::regex("^" + fam + "_" + type + "(?:_" + var + ")?(?:_" + var + ")?" +
                      "(?:_exp)?" + "_" + tail + "$");
}

const std::vector<std::string>& FamilyStems()
{
    static const std::vector<std::string> v =
        {"glm", "qwen", "gemma", "kimi", "deepseek", "nemotron", "gigachat",
         "ornith", "granite", "mimo", "phi", "dory", "alice", "xing",
         "mistral", "minimax", "hy", "flint", "ling", "beam", "step", "muse"};
    return v;
}

// Family-prefixed files that intentionally carry NO type marker:
// shared between model types or base infrastructure of the family.
bool AllowedCommon(const std::string& stem)
{
    static const std::vector<std::string> kAllowed = {
        "lsx_attn_inference",  // neutral attention/MLP kernel layer
        "qwen_base_model",
        "qwen_inference_load",
        "qwen_device_math",
        "qwen_reduce",
        "gemma_base_model",
        "deepseek_moe_tool_format", "kimi_moe_tool_format",  // V4/V4.1 DSML codec shared by both
        "nemotron_moe_shared_kernels",  // lightning/ultra shared kernels (dedup Ф7)
        "nemotron_moe_expert_cache_impl",  // lightning/ultra shared cache engine (dedup Ф7)
        "qwen_tool_format",          // ChatML codec shared by dense + MoE
        "dory_tool_format",          // Nemotron-v3 ChatML + XML tool codec
        "gemma_tool_format",         // Gemma-4 tool codec
        "granite_tool_format",       // Granite-4.2 ChatML tool codec
        "gigachat_tool_format",      // GigaChat chatml-v5 GCML codec
        "mimo_tool_format",          // MiMo ChatML + <tool_call> codec
        "alice_tool_format",         // AliceAI native chat/tool codec
        "xing_moe_tool_format",       // Xing4.0 <_user>/<_bot> + ZWSP tool codec
        "mistral_moe_tool_format",    // Mistral [INST]/[TOOL_CALLS] codec
        "minimax_moe_tool_format",    // MiniMax-M3 ]~b] / <invoke> XML codec
        "hy_moe_tool_format",         // Hy-Memory <|User|>/<arg_key> XML codec
        "flint_moe_tool_format",      // Flint inline ChatML + <arg_key> XML codec
        "ling_moe_tool_format",       // Ling <role> framing + <tool_call> XML
        "beam_moe_tool_format",       // Beam Harmony-XML codec
        "step_moe_tool_format",       // Step-5 ChatML + <tool_call> XML codec
        "kimi_moe_k3_tool_format",    // Kimi-K3 XTML chat/tool codec
        "nemotron_ultra_tool_format", // Nemotron-Ultra ChatML + <function=> codec
        "granite_switch_tool_format", // Granite-4.1 role markers + aLoRA codec
        "muse_tool_format",           // Muse-Glimmer ATEM channel + XML codec
        "hy_moe_flash_tool_format",    // Hy4-flash Hunyuan <｜hy_*｜> + <arg_key> codec
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

TEST_CASE("naming guard accepts muse_dense_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) { return std::regex_match(s, SchemeRegex()); };
    CHECK(m("muse_dense_constants"));
    CHECK(m("muse_dense_convert"));
    CHECK(m("muse_dense_load"));
    CHECK(m("muse_dense_model"));
    CHECK(m("muse_dense_inference"));
    CHECK(m("muse_dense_inference_pipeline"));
    CHECK(AllowedCommon("muse_tool_format"));
    CHECK(IsFamilyPrefixed("muse_dense_model"));
}

TEST_CASE("naming guard accepts granite_dense_switch_* stems", "[naming][guard]")
{
    auto m = [](const std::string &s) { return std::regex_match(s, SchemeRegex()); };
    CHECK(m("granite_dense_switch_constants"));
    CHECK(m("granite_dense_switch_convert"));
    CHECK(m("granite_dense_switch_load"));
    CHECK(m("granite_dense_switch_model"));
    CHECK(m("granite_dense_switch_inference"));
    CHECK(m("granite_dense_switch_kernels"));
}

TEST_CASE("naming guard accepts inkling_moe_loop_* stems", "[naming][guard]")
{
    const auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("inkling_moe_loop_constants"));
    CHECK(m("inkling_moe_loop_convert"));
    CHECK(m("inkling_moe_loop_load"));
    CHECK(m("inkling_moe_loop_model"));
    CHECK(m("inkling_moe_loop_inference"));
    CHECK(m("inkling_moe_loop_inference_pipeline"));
    CHECK(m("inkling_moe_loop_kernels"));
}

TEST_CASE("naming guard accepts inkling_moe_varloop_* stems", "[naming][guard]")
{
    const auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("inkling_moe_varloop_constants"));
    CHECK(m("inkling_moe_varloop_convert"));
    CHECK(m("inkling_moe_varloop_load"));
    CHECK(m("inkling_moe_varloop_model"));
    CHECK(m("inkling_moe_varloop_inference"));
    CHECK(m("inkling_moe_varloop_inference_pipeline"));
    CHECK(m("inkling_moe_varloop_kernels"));
    // `varloop` is a VARIANT slot, not a new family: the looped sibling must
    // keep parsing as inkling_<type>_<variant>_<role>.
    CHECK(m("inkling_moe_loop_model"));
    CHECK_FALSE(m("inkling_moe_varloop"));
    CHECK_FALSE(m("inkling_varloop_model"));
}

TEST_CASE("naming guard accepts deepseek_moe_pro_* stems", "[naming][guard]")
{
    const auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("deepseek_moe_pro_constants"));
    CHECK(m("deepseek_moe_pro_convert"));
    CHECK(m("deepseek_moe_pro_load"));
    CHECK(m("deepseek_moe_pro_load_internal"));
    CHECK(m("deepseek_moe_pro_model"));
    CHECK(m("deepseek_moe_pro_inference"));
    CHECK(m("deepseek_moe_pro_inference_attn"));
    CHECK(m("deepseek_moe_pro_inference_csa"));
    CHECK(m("deepseek_moe_pro_inference_moe"));
    CHECK(m("deepseek_moe_pro_inference_pipeline"));
    CHECK(m("deepseek_moe_pro_norm_kernels"));
}

TEST_CASE("naming guard accepts kimi_moe_k3_* stems", "[naming][guard]")
{
    const auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("kimi_moe_k3_constants"));
    CHECK(m("kimi_moe_k3_convert"));
    CHECK(m("kimi_moe_k3_dequant"));
    CHECK(m("kimi_moe_k3_load"));
    CHECK(m("kimi_moe_k3_model"));
    CHECK(m("kimi_moe_k3_inference"));
    CHECK(m("kimi_moe_k3_inference_attn"));
    CHECK(m("kimi_moe_k3_inference_pipeline"));
    CHECK(m("kimi_moe_k3_kernels"));
    CHECK(m("kimi_moe_k3_expert_cache"));
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

TEST_CASE("naming guard accepts gigachat_moe_37b_* stems", "[naming][guard]")
{
    // `37b` is a VARIANT slot (the GigaChat4 sibling of GigaChat3.5-432B) and
    // `indexer` is a SUBSYSTEM slot: the DSA lightning indexer is an
    // attention-adjacent decode unit of its own, not part of the MLA kernel.
    const auto m = [](const std::string &s) {
        return std::regex_match(s, SchemeRegex());
    };
    CHECK(m("gigachat_moe_37b_constants"));
    CHECK(m("gigachat_moe_37b_convert"));
    CHECK(m("gigachat_moe_37b_load"));
    CHECK(m("gigachat_moe_37b_model"));
    CHECK(m("gigachat_moe_37b_kernels"));
    CHECK(m("gigachat_moe_37b_norm_kernels"));
    CHECK(m("gigachat_moe_37b_inference_indexer"));
    // The type marker is mandatory: `gigachat_37b_*` is not a legal stem.
    CHECK_FALSE(m("gigachat_37b_model"));
    CHECK_FALSE(m("gigachat_moe_37b"));
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
