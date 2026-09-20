// TDD — Model test-suite guard (host-only static scan of tests/).
//
// INVARIANT (operator directive): model test suites must be AUTONOMOUS
// relative to each other — mirroring the model-independence invariant of the
// production code (test_model_independence_guard.cpp) — and every suite must
// cover the FULL model lifecycle (Model Ingestion Protocol):
//
//     convert -> load -> model -> kernels -> selfcheck
//
// This guard enforces the structural side of that directive:
//   1. Every model suite declares every lifecycle stage in its registry row
//      and each stage target is actually registered in tests/CMakeLists.txt.
//      Stages shared between sibling variants (e.g. the dense converter reused
//      by qwen3.8-27b) point at the SAME target on purpose — the sharing is
//      explicit, never silent.
//   2. Every selfcheck-stage target declares ctest SKIP_RETURN_CODE 4 (real
//      weights are optional; absence must skip, not fail).
//   3. No test file includes another model FAMILY's fixture/header. Family is
//      derived from the first path segment after the test_ prefix (glm, qwen,
//      deepseek, gemma, kimi); family-neutral headers (no family stem, e.g.
//      test_synth_checkpoint.h) are allowed everywhere. This is the test-side
//      mirror of "cross-family geometry borrowing is forbidden".

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

const std::set<std::string> &FamilyStems()
{
    static const std::set<std::string> v = {"glm", "qwen", "deepseek", "gemma",
                                            "kimi", "nemotron", "gigachat"};
    return v;
}

// "test_glm_moe_fixture.h" -> "glm"; "test_synth_checkpoint.h" -> nullopt
// (family-neutral infrastructure, allowed for every suite).
std::optional<std::string> FamilyStem(const std::string &file_name)
{
    const std::string kPrefix = "test_";
    if (file_name.rfind(kPrefix, 0) != 0) return std::nullopt;
    const std::string rest = file_name.substr(kPrefix.size());
    const auto pos = rest.find('_');
    if (pos == std::string::npos) return std::nullopt;
    const std::string stem = rest.substr(0, pos);
    if (!FamilyStems().count(stem)) return std::nullopt;
    return stem;
}

struct StageSpec
{
    const char *stage;
    std::vector<const char *> targets;
};

struct SuiteSpec
{
    const char *model;
    std::vector<StageSpec> stages;
};

// One row per ingested model suite; extend in the SAME commit as the new
// model's tests (Model Ingestion Protocol step 8). Stage order is the
// lifecycle order. A stage target may be shared with a sibling variant —
// that sharing is the registry row, in plain sight.
const std::vector<SuiteSpec> &Suites()
{
    static const std::vector<SuiteSpec> v = {
        {"glm-5.2",
         {{"convert", {"test_glm_moe_convert"}},
          {"load", {"test_glm_moe_loader", "test_glm_moe_load_fast"}},
          {"model", {"test_glm_moe_model"}},
          {"kernels",
           {"test_glm_moe_moe_compute", "test_glm_moe_attention",
            "test_glm_moe_experts"}},
          {"selfcheck", {"test_glm_moe_selfcheck"}}}},
        {"glm-5.3-flash",
         {{"convert", {"test_glm_moe_flash_convert"}},
          {"load", {"test_glm_moe_flash_loader"}},
          {"model", {"test_glm_moe_flash_model"}},
          {"kernels", {"test_glm_moe_flash_kernels"}},
          {"selfcheck", {"test_glm_moe_flash_selfcheck"}}}},
        // No convert stage by design: DeepSeek-V4-Flash ingests a
        // pre-converted logestix layout (no converter exists in the repo);
        // classification/topology of that layout is the model stage.
        {"deepseek-v4-flash",
         {{"convert", {"test_deepseek_moe_convert"}},
          {"load", {"test_deepseek_moe_loader", "test_deepseek_moe_load_fast"}},
          {"model", {"test_deepseek_moe_model"}},
          {"kernels", {"test_deepseek_moe_inference_kernels"}},
          {"selfcheck", {"test_deepseek_moe_selfcheck"}}}},
        {"deepseek-v4.1-flash",
         {{"convert", {"test_deepseek_moe_v41_convert"}},
          {"load",
           {"test_deepseek_moe_v41_loader", "test_deepseek_moe_v41_load_fast"}},
          {"model", {"test_deepseek_moe_v41_model"}},
          {"kernels",
           {"test_deepseek_moe_v41_dequant", "test_deepseek_moe_v41_kernels",
            "test_deepseek_moe_v41_actquant",
            "test_deepseek_moe_v41_pipeline"}},
          {"selfcheck", {"test_deepseek_moe_v41_selfcheck"}}}},
        {"qwen3.5-9b",
         {{"convert",
           {"test_qwen_dense_convert", "test_qwen_dense_converter_integrity"}},
          {"load", {"test_qwen_dense_loader"}},
          {"model", {"test_qwen_dense_model"}},
          {"kernels",
           {"test_qwen_dense_mlp", "test_qwen_dense_full_attn",
            "test_qwen_dense_nvfp4"}},
          {"selfcheck", {"test_qwen_dense_selfcheck"}}}},
        // qwen3.8-27b shares the dense convert/load/kernels pipeline with
        // qwen3.5-9b (QwenBaseModel); its own stages are model+selfcheck.
        {"qwen3.8-27b",
         {{"convert", {"test_qwen_dense_convert"}},
          {"load", {"test_qwen_dense_loader"}},
          {"model", {"test_qwen_dense_27b_model"}},
          {"kernels", {"test_qwen_dense_nvfp4"}},
          {"selfcheck", {"test_qwen_dense_27b_selfcheck"}}}},
        {"qwen3.5-35b-a3b",
         {{"convert", {"test_qwen_moe_convert"}},
          {"load", {"test_qwen_moe_loader"}},
          {"model", {"test_qwen_moe_model"}},
          {"kernels", {"test_qwen_moe_kernels"}},
          {"selfcheck", {"test_qwen_moe_selfcheck"}}}},
        // qwen3.6 shares loader + kernels with qwen3.5-35b (QwenMoeModelBase).
        {"qwen3.6-35b-a3b",
         {{"convert", {"test_qwen_moe_36_convert"}},
          {"load", {"test_qwen_moe_loader"}},
          {"model", {"test_qwen_moe_36_model"}},
          {"kernels", {"test_qwen_moe_kernels"}},
          {"selfcheck", {"test_qwen_moe_36_selfcheck"}}}},
        {"qwen3.8-flash-next",
         {{"convert", {"test_qwen_moe_exp_convert"}},
          {"load", {"test_qwen_moe_exp_loader"}},
          {"model", {"test_qwen_moe_exp_model"}},
          {"kernels", {"test_qwen_moe_exp_kernels"}},
          {"selfcheck", {"test_qwen_moe_exp_selfcheck"}}}},
        {"gemma-4-26b-a4b-it",
         {{"convert", {"test_gemma_moe_convert"}},
          {"load", {"test_gemma_moe_loader"}},
          {"model", {"test_gemma_moe_model"}},
          {"kernels",
           {"test_gemma_moe_norm", "test_gemma_moe_rotary",
            "test_gemma_moe_attn", "test_gemma_moe_moe"}},
          {"selfcheck", {"test_gemma_moe_selfcheck"}}}},
        {"kimi-k2.7-code",
         {{"convert", {"test_kimi_moe_convert"}},
          {"load", {"test_kimi_moe_loader"}},
          {"model", {"test_kimi_moe_model"}},
          {"kernels",
           {"test_kimi_moe_router", "test_kimi_moe_attention",
            "test_kimi_moe_experts", "test_kimi_moe_int4_dequant"}},
          {"selfcheck", {"test_kimi_moe_selfcheck"}}}},
        {"nemotron-3.5-lightning-30b-a3b",
         {{"convert", {"test_nemotron_moe_convert"}},
          {"load", {"test_nemotron_moe_loader"}},
          {"model", {"test_nemotron_moe_model"}},
          {"kernels", {"test_nemotron_moe_kernels"}},
          {"selfcheck", {"test_nemotron_moe_selfcheck"}}}},
    };
    return v;
}

std::string ReadFileOrThrow(const fs::path &p)
{
    std::ifstream in(p);
    if (!in.is_open())
        throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<std::string> RegisteredTestNames(const std::string &text)
{
    std::vector<std::string> names;
    static const std::regex add_re(
        R"(add_test\s*\(\s*NAME\s+([A-Za-z0-9_]+))");
    static const std::regex fe_re(
        R"(foreach\s*\(\s*\w+((?:\s+(?!IN\b|LISTS\b|RANGE\b)[A-Za-z0-9_]+)+)\s*\))");
    std::smatch m;
    auto begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, add_re))
    {
        names.push_back(m[1]);
        begin = m.suffix().first;
    }
    begin = text.cbegin();
    while (std::regex_search(begin, text.cend(), m, fe_re))
    {
        std::istringstream is(m[1].str());
        std::string tok;
        while (is >> tok) names.push_back(tok);
        begin = m.suffix().first;
    }
    return names;
}

std::vector<std::string> CheckLifecycle(const std::vector<SuiteSpec> &suites,
                                        const std::set<std::string> &registered)
{
    std::vector<std::string> v;
    for (const auto &s : suites)
        for (const auto &st : s.stages)
            for (const auto *t : st.targets)
                if (!registered.count(t))
                    v.push_back(std::string("suite ") + s.model + " stage " +
                                st.stage + ": target not registered: " + t);
    return v;
}

bool HasSkipReturnCode4(const std::string &cmake_text, const std::string &name)
{
    static const std::regex re(
        R"(set_tests_properties\s*\(([^)]*?)SKIP_RETURN_CODE\s+4)");
    std::smatch m;
    auto begin = cmake_text.cbegin();
    while (std::regex_search(begin, cmake_text.cend(), m, re))
    {
        std::istringstream is(m[1].str());
        std::string tok;
        while (is >> tok)
            if (tok == name) return true;
        begin = m.suffix().first;
    }
    return false;
}

std::vector<std::string> CheckSelfcheckSkip(const std::string &cmake_text)
{
    std::vector<std::string> v;
    for (const auto &s : Suites())
        for (const auto &st : s.stages)
        {
            if (std::string(st.stage) != "selfcheck") continue;
            for (const auto *t : st.targets)
                if (!HasSkipReturnCode4(cmake_text, t))
                    v.push_back(std::string("selfcheck target without "
                                            "SKIP_RETURN_CODE 4: ") +
                                t);
        }
    return v;
}

std::vector<std::string> CheckCrossFamilyIncludes(const fs::path &tests_dir)
{
    std::vector<std::string> v;
    static const std::regex inc_re(
        R"inc(^\s*#\s*include\s+"(test_[A-Za-z0-9_.]+\.h)")inc");
    static const std::regex ext_re(R"(\.(cpp|cu|h)$)");
    std::error_code ec;
    for (auto &e : fs::directory_iterator(tests_dir, ec))
    {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        if (!std::regex_search(name, ext_re)) continue;
        const auto self = FamilyStem(name);
        if (!self) continue;  // family-neutral file: nothing to check
        std::istringstream in(ReadFileOrThrow(e.path()));
        std::string line;
        while (std::getline(in, line))
        {
            std::smatch m;
            if (!std::regex_match(line, m, inc_re)) continue;
            const auto other = FamilyStem(m[1].str());
            if (!other || *other == *self) continue;
            v.push_back(name + " borrows another family's header " +
                        m[1].str() + " (cross-suite coupling)");
        }
    }
    return v;
}

// Loader-stage tests run on SYNTHETIC converted fixtures only. A loader test
// that opens the model vault turns the whole suite into a multi-GB download
// (exactly the regression the convert-suite discipline killed). Rules:
//   A. every test_*_loader.cpp must build its checkpoint from a fixture
//      header (its own family fixture or the family-neutral
//      test_synth_checkpoint.h);
//   B. no loader-named test (test_*_loader.*, test_*_load_fast.*) may touch
//      the model vault: no vault-root literals, no std::getenv model-root
//      lookups (real-weight coverage lives in selfcheck/perf suites).
//   C. family model-stage tests (test_<family>_model.cpp) must not hardcode
//      vault roots either — env-var opt-in + SKIP is the allowed real-weight
//      pattern (factory/selfcheck/perf stages own vault integration).
bool IsLoaderTest(const std::string &name)
{
    static const std::regex re(
        R"(test_[A-Za-z0-9_]+_(loader|load_fast)\.(cpp|cu)$)");
    return std::regex_match(name, re);
}

bool IsLoaderStageTest(const std::string &name)
{
    static const std::regex re(R"(test_[A-Za-z0-9_]+_loader\.cpp$)");
    return std::regex_match(name, re);
}

std::vector<std::string> CheckLoaderSyntheticDiscipline(
    const fs::path &tests_dir)
{
    std::vector<std::string> v;
    static const std::regex ext_re(R"(\.(cpp|cu)$)");
    static const std::regex fixture_inc_re(
        R"inc(^\s*#\s*include\s+"(test_[A-Za-z0-9_]+_fixture\.h|test_synth_checkpoint\.h)")inc");
    static const std::regex model_stage_re(
        R"(test_(glm|qwen|deepseek|gemma|kimi|nemotron)[A-Za-z0-9_]*_model\.cpp$)");
    const std::vector<std::string> vault_markers = {
        "mnt/llm_models/models", "huggingface.co"};
    std::error_code ec;
    for (auto &e : fs::directory_iterator(tests_dir, ec))
    {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        if (!std::regex_search(name, ext_re)) continue;
        const std::string text = ReadFileOrThrow(e.path());
        const bool loader = IsLoaderTest(name);
        const bool model_stage = std::regex_match(name, model_stage_re);
        if (!loader && !model_stage) continue;
        // Strip line comments so prose mentions of vault paths in docs
        // don't trip the literal scan.
        std::string stripped;
        {
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line))
            {
                const auto pos = line.find("//");
                stripped += line.substr(0, pos) + "\n";
            }
        }
        bool has_fixture_inc = false;
        {
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line))
                if (std::regex_match(line, fixture_inc_re))
                {
                    has_fixture_inc = true;
                    break;
                }
        }
        if (IsLoaderStageTest(name) && !has_fixture_inc)
            v.push_back(name +
                        ": loader-stage test without a synthetic fixture "
                        "include (must build its checkpoint, not load a "
                        "vault)");
        for (const auto &marker : vault_markers)
            if (stripped.find(marker) != std::string::npos)
                v.push_back(name + ": hardcodes vault path (" + marker +
                            ")");
        if (loader && stripped.find("getenv") != std::string::npos)
            v.push_back(name +
                        ": loader test resolves the model root via getenv "
                        "(synthetic fixtures only)");
    }
    return v;
}

} // namespace

TEST_CASE("suite guard: family stem classification", "[suite][guard][unit]")
{
    CHECK(FamilyStem("test_glm_moe_fixture.h") == std::optional<std::string>{"glm"});
    CHECK(FamilyStem("test_qwen_dense_fixture.h") == std::optional<std::string>{"qwen"});
    CHECK(FamilyStem("test_deepseek_moe_model.cpp") ==
          std::optional<std::string>{"deepseek"});
    CHECK(FamilyStem("test_kimi_moe_fixture.h") == std::optional<std::string>{"kimi"});
    CHECK(FamilyStem("test_gemma_moe_loader.cpp") == std::optional<std::string>{"gemma"});
    // Family-neutral infrastructure has no family stem.
    CHECK(FamilyStem("test_synth_checkpoint.h") == std::nullopt);
    CHECK(FamilyStem("test_core.cpp") == std::nullopt);
    CHECK(FamilyStem("CMakeLists.txt") == std::nullopt);
}

TEST_CASE("suite guard: lifecycle checker flags missing targets",
          "[suite][guard][unit]")
{
    const std::vector<SuiteSpec> suites = {
        {"m-x",
         {{"convert", {"t_convert"}}, {"model", {"t_model"}}}}};
    CHECK(CheckLifecycle(suites, {"t_convert"}).size() == 1);
    CHECK(CheckLifecycle(suites, {"t_convert", "t_model"}).empty());
}

TEST_CASE("suite guard: selfcheck skip checker", "[suite][guard][unit]")
{
    const std::string cm = R"(add_test(NAME t_self COMMAND t_self)
set_tests_properties(t_self PROPERTIES SKIP_RETURN_CODE 4)
add_test(NAME t_bare COMMAND t_bare))";
    CHECK(HasSkipReturnCode4(cm, "t_self"));
    CHECK_FALSE(HasSkipReturnCode4(cm, "t_bare"));
    const std::vector<SuiteSpec> suites = {
        {"m-x", {{"selfcheck", {"t_self", "t_bare"}}}}};
    // Re-run via the public checker by mirroring its loop.
    std::vector<std::string> v;
    for (const auto &s : suites)
        for (const auto &st : s.stages)
            for (const auto *t : st.targets)
                if (!HasSkipReturnCode4(cm, t))
                    v.push_back(t);
    REQUIRE(v.size() == 1);
    CHECK(v[0] == "t_bare");
}

TEST_CASE("suite guard: cross-family include checker",
          "[suite][guard][unit]")
{
    SECTION("cross-family borrow is a violation (stems differ)")
    {
        CHECK(FamilyStem("test_glm_moe_fixture.h") ==
              std::optional<std::string>{"glm"});
        CHECK(FamilyStem("test_kimi_moe_fixture.h") ==
              std::optional<std::string>{"kimi"});
        CHECK(*FamilyStem("test_glm_moe_fixture.h") !=
              *FamilyStem("test_kimi_moe_fixture.h"));
    }
    SECTION("neutral header has no family stem")
    {
        CHECK(FamilyStem("test_synth_checkpoint.h") == std::nullopt);
    }
}

TEST_CASE("suite guard: every model suite covers the full lifecycle",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const auto registered = [&] {
        std::set<std::string> s;
        for (const auto &n : RegisteredTestNames(cmake_text)) s.insert(n);
        return s;
    }();
    INFO("registered tests: " << registered.size());
    auto v = CheckLifecycle(Suites(), registered);
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("lifecycle violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("suite guard: selfcheck targets declare SKIP_RETURN_CODE 4",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    auto v = CheckSelfcheckSkip(cmake_text);
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("selfcheck skip violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("suite guard: loader tests stay on synthetic fixtures",
          "[suite][guard][unit]")
{
    SECTION("loader-stage classification")
    {
        CHECK(IsLoaderStageTest("test_glm_moe_flash_loader.cpp"));
        CHECK(IsLoaderStageTest("test_deepseek_moe_loader.cpp"));
        CHECK_FALSE(IsLoaderStageTest("test_glm_moe_load_fast.cu"));
        CHECK(IsLoaderTest("test_glm_moe_load_fast.cu"));
        CHECK(IsLoaderTest("test_qwen_dense_loader.cpp"));
        CHECK_FALSE(IsLoaderTest("test_blackwell_perf_glm.cpp"));
    }
}

TEST_CASE("suite guard: no cross-family fixture borrowing", "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    auto v = CheckCrossFamilyIncludes(root / "tests");
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("cross-family violations:\n" << joined);
    REQUIRE(v.empty());
}

TEST_CASE("suite guard: loader tests never load the model vault",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    auto v = CheckLoaderSyntheticDiscipline(root / "tests");
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("loader discipline violations:\n" << joined);
    REQUIRE(v.empty());
}
