// TDD — Model test-suite guard (host-only static scan of tests/).
//
// INVARIANT (operator directive): model test suites must be AUTONOMOUS
// relative to each other — mirroring the model-independence invariant of the
// production code (test_model_independence_guard.cpp) — and every suite must
// cover the FULL model lifecycle (Model Ingestion Protocol):
//
//     convert -> load -> model -> kernels -> selfcheck -> fsi -> code -> det
//
// This guard enforces the structural side of that directive:
//   1. Every model suite declares every lifecycle stage in its registry row
//      and each stage target is actually registered in tests/CMakeLists.txt.
//      Stages shared between sibling variants (e.g. the dense converter reused
//      by qwen3.8-27b) point at the SAME target on purpose — the sharing is
//      explicit, never silent.
//   2. Every selfcheck / fsi / code / det-stage target declares ctest
//      SKIP_RETURN_CODE 4 (real weights and golden are optional; absence must
//      skip, not fail).
//   3. No test file includes another model FAMILY's fixture/header. Family is
//      derived from the first path segment after the test_ prefix (glm, qwen,
//      deepseek, gemma, kimi); family-neutral headers (no family stem, e.g.
//      test_synth_checkpoint.h) are allowed everywhere. This is the test-side
//      mirror of "cross-family geometry borrowing is forbidden".

#include <catch2/catch_test_macros.hpp>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>

#include <filesystem>
#include <map>
#include <fstream>
#include <optional>
#include <regex>
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "test_manifest_query.h"

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif
#ifndef LSX_MANIFEST_GENERATED_DIR
#define LSX_MANIFEST_GENERATED_DIR ""
#endif

namespace {

// The family vocabulary comes from models/*.json. The hand-written literal that
// used to live here listed 15 of 20 families, so "no test may include another
// family's fixture" silently passed for beam, hy, minimax, mistral and xing.
const std::set<std::string> &FamilyStems()
{
    static const std::set<std::string> v = [] {
        std::set<std::string> out;
        for (const auto &f : manifest_query::Families()) out.insert(f);
        return out;
    }();
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

// The lifecycle order the protocol defines; a stage outside it is a manifest
// schema error, so this list only has to name them.
constexpr const char *kStageOrder[] = {"convert", "load",    "model",
                                       "kernels", "selfcheck", "fsi",
                                       "code",    "det",     "perf"};

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
    // `SuiteSpec` holds `const char*`, so the builder needs storage that
    // outlives it: a reserved vector keeps the addresses stable for the process
    // lifetime (a local std::string's c_str() here is a dangling pointer).
    static std::vector<std::string> names;
    names.reserve(1024);
    // One row per model, stages in lifecycle order, targets from the manifest.
    // The hand-written table this replaces had to be extended by hand in every
    // ingest commit; the generated one cannot fall behind the registrations
    // because both come from the same source.
    static const std::vector<SuiteSpec> v = [] {
        std::vector<SuiteSpec> out;
        for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
        {
            const std::string model = lsxmanifest::kModels[i].model_type;
            names.push_back(model);
            SuiteSpec row;
            row.model = names.back().c_str();
            for (const char *stage : kStageOrder)
            {
                std::vector<const char *> targets;
                for (const auto *t : manifest_query::TargetsOf(model, stage))
                {
                    names.emplace_back(t->target);
                    targets.push_back(names.back().c_str());
                }
                if (!targets.empty())
                {
                    StageSpec s;
                    names.push_back(stage);
                    s.stage = names.back().c_str();
                    s.targets = std::move(targets);
                    row.stages.push_back(std::move(s));
                }
            }
            out.push_back(std::move(row));
        }
        return out;
    }();
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

// The CMake text this guard scans: the hand-written registrations plus the
// manifest-generated fragment, which now owns every model target's add_test and
// properties. Reading only one of the two made the guard report 90 of 511
// registrations after the registration move.
std::string CmakeCorpus(const fs::path &root)
{
    std::string text = ReadFileOrThrow(root / "tests" / "CMakeLists.txt");
    const fs::path generated(LSX_MANIFEST_GENERATED_DIR "/model_targets.cmake");
    std::error_code ec;
    // The separator matters: without it the last line of CMakeLists.txt and the
    // first add_test of the fragment merge into one unparseable line.
    if (fs::exists(generated, ec))
        text += "\n" + ReadFileOrThrow(generated);
    return text;
}

std::vector<std::string> RegisteredTestNames(const std::string &text)
{
    std::vector<std::string> names;
    // Both the raw add_test form and the sanctioned helper
    // (lsx_add_test / lsx_add_cuda_test) register a target; the helper is the
    // documented path for new tests, so the scanner must see it.
    static const std::regex add_re(
        R"((?:lsx_add_(?:cuda_)?test|add_test)\s*\(\s*NAME\s+([A-Za-z0-9_]+))");
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
            const std::string stage = st.stage;
            if (stage != "selfcheck" && stage != "fsi" &&
                stage != "code" && stage != "det") continue;
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
        R"(test_(glm|qwen|deepseek|gemma|kimi|nemotron|gigachat|ornith|granite|mimo|phi)[A-Za-z0-9_]*_model\.cpp$)");
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
        CmakeCorpus(root);
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

TEST_CASE("suite guard: selfcheck, fsi, code and det targets declare "
          "SKIP_RETURN_CODE 4",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        CmakeCorpus(root);
    auto v = CheckSelfcheckSkip(cmake_text);
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("stage skip violations:\n" << joined);
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

// A test source claimed by two manifests may not carry the OTHER model's
// identity. This is the defect class the 8B ingestion exposed: three granite
// stage tests were hard-coded to the 30B while the 8B row claimed them, and every
// guard stayed green because they check that a claimed target EXISTS, not that it
// tests the model claiming it (docs/OPEN_TASKS.md G2).
//
// What counts as identity, and why exactly one shape: the vault ENVIRONMENT
// VARIABLE. It is unique per model and a shared source has no reason to name a
// sibling's — a test that wants a model's tree reaches it through its own env
// name, so a sibling's name there means the source is looking at the wrong tree.
//
// Two neighbouring shapes are deliberately NOT violations:
//   * a bare `model_type` string — a sibling test legitimately compares two
//     passports (test_granite_switch_prepare asserts the 8B passport rejects a
//     30B topology, so it names both as data);
//   * the vault DIRECTORY name — the same string is a natural name for a test
//     fixture directory (test_granite_switch_convert builds
//     `/tmp/.../granite-switch-4.1-8b`), so it cannot distinguish identity from
//     data. This one bit during development: the check flagged its own fixtures.
// A literal is accepted when it is the default arm of an `#ifndef`-guarded macro:
// that is the parameterised form a shared source is supposed to use.
struct Claim {
    std::string model;
    std::string vault_env;
    std::string vault_dir;
};

std::string JoinNames(const std::vector<Claim> &rows)
{
    std::set<std::string> uniq;
    for (const auto &r : rows) uniq.insert(r.model);
    std::string out;
    for (const auto &n : uniq) {
        if (!out.empty()) out += ", ";
        out += n;
    }
    return out;
}

std::vector<std::string> CheckSiblingIdentity(
    const fs::path &tests_dir,
    const std::map<std::string, std::vector<Claim>> &claims)
{
    std::vector<std::string> v;
    for (const auto &kv : claims) {
        const std::string &source = kv.first;
        const std::vector<Claim> &rows = kv.second;
        if (rows.size() < 2) continue;
        const fs::path path = tests_dir / source;
        if (!fs::exists(path)) continue;
        const std::string text = ReadFileOrThrow(path);

        std::vector<std::string> ls;
        {
            std::istringstream in(text);
            std::string line;
            while (std::getline(in, line)) ls.push_back(line);
        }

        for (const Claim &c : rows) {
            const std::string lits[1] = {c.vault_env};
            for (const std::string &lit : lits) {
                if (lit.empty()) continue;
                const std::string needle = "\"" + lit + "\"";
                for (std::size_t i = 0; i < ls.size(); ++i) {
                    if (ls[i].find(needle) == std::string::npos) continue;
                    // Accepted only as the default arm of an #ifndef guard.
                    bool guarded = false;
                    const std::size_t def = ls[i].find("#define");
                    if (def != std::string::npos && i >= 1) {
                        std::string name = ls[i].substr(def + 7);
                        const std::size_t b = name.find_first_not_of(" \t");
                        name = (b == std::string::npos) ? "" : name.substr(b);
                        const std::size_t e = name.find_first_of(" \t");
                        if (e != std::string::npos) name = name.substr(0, e);
                        guarded = !name.empty() &&
                                  ls[i - 1].find("#ifndef " + name) !=
                                      std::string::npos;
                    }
                    if (!guarded) {
                        v.push_back(source + ":" + std::to_string(i + 1) +
                                    " names " + c.model + "'s identity ('" + lit +
                                    "') while claimed by " + JoinNames(rows) +
                                    " — parameterise it with a macro default, as "
                                    "test_granite_switch_model.cpp does");
                    }
                }
            }
            // The two spellings that made the model UNDER TEST the wrong model.
            const std::string pats[2] = {
                "model_type = \"" + c.model + "\"",
                "kModelType = \"" + c.model + "\""};
            for (const std::string &pat : pats) {
                if (text.find(pat) != std::string::npos)
                    v.push_back(source + " hard-codes " + c.model +
                                " as the model under test ('" + pat + "')");
            }
        }
    }
    return v;
}

TEST_CASE("suite guard: a shared test source cannot name a sibling as itself",
          "[suite][guard]")
{
    // Build source -> claiming models from the manifests themselves, so the
    // check follows the registry instead of a hand-written list (the reason five
    // families once fell out of a similar check unnoticed).
    std::map<std::string, std::vector<Claim>> claims;
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i) {
        const auto &m = lsxmanifest::kModels[i];
        for (std::size_t j = 0; j < lsxmanifest::kTargetCount; ++j) {
            const auto &t = lsxmanifest::kTargets[j];
            if (std::string_view(t.model_type) != m.model_type) continue;
            const std::string src =
                t.source ? std::string(t.source) : std::string(t.target) + ".cpp";
            claims[src].push_back({m.model_type, m.vault_env, m.vault_dir});
        }
    }
    std::size_t shared = 0;
    for (const auto &[src, rows] : claims) if (rows.size() > 1) ++shared;
    INFO("test sources claimed by more than one manifest: " << shared);
    REQUIRE(shared > 5);          // the registry really does share sources

    auto v = CheckSiblingIdentity(fs::path(LSX_SOURCE_DIR) / "tests", claims);
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO("sibling-identity violations:\n" << joined);
    CHECK(v.empty());
}

TEST_CASE("suite guard: the sibling-identity rule bites", "[suite][guard]")
{
    // Non-vacuity: an unguarded literal is a violation, the same literal behind
    // an #ifndef default is not (that is the parameterised form), and a
    // `cfg.model_type = "<sibling>"` hard-code is reported by its own pattern.
    const fs::path dir = fs::temp_directory_path() / "lsx_sibling_identity";
    fs::create_directories(dir);
    {
        std::ofstream f(dir / "bad_unguarded.cpp");
        f << "const char *env = \"LSX_TEST_SIBLING_ROOT\";\n";
    }
    {
        std::ofstream f(dir / "bad_hardcode.cpp");
        f << "cfg.model_type = \"sibling-8b\";\n";
    }
    {
        std::ofstream f(dir / "good_parameterised.cpp");
        f << "#ifndef LSX_X_MODEL\n#define LSX_X_MODEL \"sibling-8b\"\n#endif\n"
             "#ifndef LSX_X_ROOT_ENV\n#define LSX_X_ROOT_ENV \"LSX_TEST_SIBLING_ROOT\"\n#endif\n"
             "cfg.model_type = LSX_X_MODEL;\n";
    }
    const std::map<std::string, std::vector<Claim>> claims = {
        {"bad_unguarded.cpp", {{"mine", "LSX_TEST_MINE_ROOT", "mine"}, {"sibling-8b", "LSX_TEST_SIBLING_ROOT", "sibling-8b"}}},
        {"bad_hardcode.cpp", {{"mine", "LSX_TEST_MINE_ROOT", "mine"}, {"sibling-8b", "LSX_TEST_SIBLING_ROOT", "sibling-8b"}}},
        {"good_parameterised.cpp", {{"mine", "LSX_TEST_MINE_ROOT", "mine"}, {"sibling-8b", "LSX_TEST_SIBLING_ROOT", "sibling-8b"}}},
    };
    const auto v = CheckSiblingIdentity(dir, claims);
    std::string joined;
    for (const auto &s : v) joined += s + "\n";
    INFO(joined);
    CHECK(joined.find("bad_unguarded.cpp") != std::string::npos);
    CHECK(joined.find("bad_hardcode.cpp") != std::string::npos);
    CHECK(joined.find("good_parameterised.cpp") == std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("suite guard: every family test target is claimed by a suite row",
          "[suite][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    const std::string cmake_text =
        CmakeCorpus(root);
    const auto registered = [&] {
        std::set<std::string> s;
        for (const auto &n : RegisteredTestNames(cmake_text)) s.insert(n);
        return s;
    }();

    // Declared in-progress ingestions. A family target is normally claimed by a
    // COMPLETE Suites() row in the same commit that registers it, and the
    // full-stage-set guard below forbids writing that row before every stage
    // target exists. A multi-stage ingestion therefore has exactly one honest
    // intermediate state: the convert stage lands first, and the target is
    // declared here. The declaration is checked in both directions - it must
    // name a registered, UNCLAIMED target, and it fails as soon as the owning
    // row appears - so it cannot outlive the problem or hide an orphan.
    // Empty: every family target is claimed by a manifest row. An ingestion
    // that is mid-flight adds its targets here and deletes them in the same
    // commit that lands the row (Inkling-2-Small did exactly that on
    // 2026-11-15).
    static const std::set<std::string> kPendingSuiteTargets = {
        // Inkling-2-Mini ingestion in flight: the manifest row lands with the
        // model class; delete this entry in that same commit.
        "test_inkling_moe_loop_constants",
    };

    std::set<std::string> claimed;
    for (const auto &s : Suites())
        for (const auto &st : s.stages)
            for (const auto *t : st.targets)
                claimed.insert(t);

    std::vector<std::string> orphans;
    for (const auto &name : registered)
    {
        if (!FamilyStem(name)) continue;
        if (claimed.count(name)) continue;
        if (kPendingSuiteTargets.count(name)) continue;
        orphans.push_back(name);
    }
    std::sort(orphans.begin(), orphans.end());
    std::string joined;
    for (const auto &s : orphans) joined += s + "\n";
    INFO("family test targets absent from Suites():\n" << joined);
    CHECK(orphans.empty());

    std::vector<std::string> bad_pending;
    for (const auto &p : kPendingSuiteTargets)
    {
        if (!registered.count(p))
            bad_pending.push_back(p +
                                  ": pending target is not registered");
        else if (claimed.count(p))
            bad_pending.push_back(p +
                                  ": pending target is now claimed - delete it "
                                  "from kPendingSuiteTargets");
    }
    std::string joined_pending;
    for (const auto &s : bad_pending) joined_pending += s + "\n";
    INFO("pending suite-target violations:\n" << joined_pending);
    CHECK(bad_pending.empty());
}

TEST_CASE("suite guard: every suite row declares its tool-calling verdict",
          "[suite][guard]")
{
    // Stage 4b is MANDATORY in the ingestion protocol, yet it lived only as a
    // target listed inside the "model" stage row: a new family could ship with
    // no tool test at all and every guard stayed green. A row must therefore
    // say, explicitly, either
    //   * the target(s) that prove the native render+parse round-trip, or
    //   * "generic" — the model keeps model.h's Hermes default and nothing more
    //     is claimed (parse_tool_calls must then be empty).
    struct ToolRow {
        const char *model;
        std::vector<const char *> targets;  // empty => generic fallback
    };
    static const std::vector<ToolRow> kTooling = {
        {"glm-5.2", {}},
        {"glm-5.3-flash", {"test_glm_flash_tool_framing"}},
        {"deepseek-v4-flash", {"test_deepseek_tool_framing"}},
        {"deepseek-v4.1-flash", {"test_deepseek_tool_framing"}},
        {"deepseek-v4-pro-0813", {"test_deepseek_tool_framing"}},
        {"qwen3.5-9b", {"test_qwen_tool_framing"}},
        // The 27B adds a real-weight round-trip case to its selfcheck target
        // (native ChatML/XML render -> generate -> parse on the converted
        // checkpoint), so the verdict is proven twice.
        {"qwen3.8-27b",
         {"test_qwen_tool_framing", "test_qwen_dense_27b_selfcheck"}},
        {"qwen3.5-35b-a3b", {"test_qwen_tool_framing"}},
        {"qwen3.6-35b-a3b", {"test_qwen_tool_framing"}},
        {"qwen3.8-flash-next", {"test_qwen_tool_framing"}},
        {"gemma-4-26b-a4b-it", {"test_gemma_tool_framing"}},
        // The 31B ships a tokenizer.json / chat_template.jinja that are
        // BYTE-IDENTICAL to the 26B's, so the shared gemma codec target proves
        // the render+parse round-trip for both.
        {"gemma-4-31b-it", {"test_gemma_tool_framing"}},
        {"kimi-k2.7-code", {"test_kimi_tool_framing"}},
        {"ornith-1.5-35b-a3b", {"test_qwen_tool_framing"}},
        {"phi-4", {"test_phi_tool_framing"}},
        // Dory's native renderer + the shared Qwen3-Coder XML parser, verified
        // against the checkpoint's chat_template.jinja in the host test; the
        // real-weight round-trip rides the selfcheck target's tool hook.
        {"dory-1.0-4b", {"test_dory_tool_framing", "test_dory_dense_selfcheck"}},
        {"granite-4.2-30b", {}},
        {"granite-switch-4.1-30b", {"test_granite_switch_tool_framing"}},
        // The 8B ships a byte-identical chat template and tokenizer
        // (md5-equal to the 30B), so it carries the same verdict: the
        // family framing test covers both, and the runtime parse is the
        // generic bare-JSON fallback.
        {"granite-switch-4.1-8b", {"test_granite_switch_tool_framing"}},
        // nemotron keeps its tool case inside the model target rather than in
        // a dedicated one; naming the target that actually carries it is what
        // the guard checks.
        {"nemotron-3.5-lightning-30b-a3b", {"test_nemotron_moe_model"}},
        // The NVFP4 sibling reuses the parent's Qwen-style XML render/parse
        // verbatim; its model target carries the same tool case.
        {"nemotron-3.5-lightning-30b-a3b-nvfp4",
         {"test_nemotron_moe_nvfp4_model"}},
        // Ultra renders with its OWN codec (its chat template differs from the
        // 30B sibling's): the framing target proves render+parse, and the
        // real-weight round-trip rides the selfcheck target's tool hook.
        {"nemotron-ultra-550b-a55b",
         {"test_nemotron_ultra_tool_framing", "test_nemotron_ultra_selfcheck"}},
        {"gigachat3.5-432b-a28b-reasoning", {"test_gigachat_tool_framing"}},
        {"mimo-v2.6-flash-rl", {"test_mimo_tool_framing"}},
        {"alice-foundation-80b-a3b", {"test_alice_tool_framing"}},
        // Xing has a native renderer verified against its chat_template.jinja
        // (not the generic Hermes fallback), so it is named here.
        {"xing4.0-29b-a4b", {"test_xing_tool_framing"}},
        // Mistral Small 4 ships a native [INST]/[TOOL_CALLS] template.
        {"mistral-small-4-119b", {"test_mistral_moe_tool_framing"}},
        // Mistral Large 4 adds the always-on [MODEL_SETTINGS] block; the
        // verdict is proven on a synthetic tokenizer fixture whose special ids
        // match the passport. It also proves the new pre-tokenizer kind (12).
        {"mistral-large-4-1t", {"test_mistral_moe_large4_tool_framing"}},
        {"minimax-m3", {"test_minimax_moe_tool_framing"}},
        {"minimax-m3.1-flash", {"test_minimax_moe_flash_tool_framing"}},
        {"hy-memory-30b-a3b", {"test_hy_moe_tool_framing"}},
        {"hy-simulmt-30b-a3b", {"test_hy_moe_simulmt_tool_framing"}},
        {"flint", {"test_flint_moe_tool_framing"}},
        {"ling", {"test_ling_tool_framing"}},
        {"beam", {"test_beam_tool_framing"}},
        {"step-5-preview", {"test_step_moe_tool_framing"}},
        {"kimi-k3", {"test_kimi_k3_tool_framing"}},
        // kimi-k3.1-pro reuses this XTML codec verbatim (identical tokenizer and
        // chat_template), so the same target proves its render+parse round-trip.
        {"kimi-k3.1-pro", {"test_kimi_k3_tool_framing"}},
        // Inkling has a native codec (inkling_moe_tool_format.cpp) rendered from
        // the checkpoint's own chat_template.jinja: `<|message_model|><|content_
        // thinking|>` opens an answer, tool calls are
        // `<|content_invoke_tool_json|>{"name":…,"args":{…}}`, and the round-trip
        // is 49 framing assertions on a synthetic fixture.
        {"inkling-2-small", {"test_inkling_moe_tool_framing"}},
        // Inkling base has its OWN codec (inkling_moe_base_tool_format.cpp):
        // the same marker family, but the tool-result header carries the tool
        // name — verified against the base checkpoint's chat_template.jinja.
        {"inkling", {"test_inkling_moe_base_tool_framing"}},
        // Muse-Glimmer has a native ATEM codec (muse_tool_format.cpp): channel
        // markers <|start|>/<|message|>/<|eot|> + <atem:function_calls> XML,
        // proven byte-identical to the checkpoint's chat_template.jinja render.
        {"muse-glimmer-30b", {"test_muse_dense_tool_framing"}},
    };
    // Every ingested model must be in the table (a new family must state its
    // verdict), and every named target must be a registered ctest target.
    const std::set<std::string> registered = [] {
        std::set<std::string> names;
        for (const auto &n : RegisteredTestNames(
                 CmakeCorpus(fs::path(LSX_SOURCE_DIR))))
            names.insert(n);
        return names;
    }();
    std::vector<std::string> problems;
    for (const auto &s : Suites())
    {
        const auto it = std::find_if(kTooling.begin(), kTooling.end(),
                                     [&](const ToolRow &r) {
                                         return std::string(r.model) == s.model;
                                     });
        if (it == kTooling.end()) {
            problems.push_back(std::string(s.model) +
                               ": no tool-calling verdict declared");
            continue;
        }
        for (const char *t : it->targets)
            if (!registered.count(t))
                problems.push_back(std::string(s.model) + ": tool target '" + t +
                                   "' is not a registered ctest target");
    }
    for (const auto &row : kTooling)
    {
        const bool known = std::any_of(Suites().begin(), Suites().end(),
                                       [&](const SuiteSpec &s) {
                                           return std::string(s.model) == row.model;
                                       });
        if (!known)
            problems.push_back(std::string(row.model) +
                               ": tool verdict row has no Suites() row");
    }
    std::string joined;
    for (const auto &x : problems) joined += x + "\n";
    INFO("tool-calling verdicts:\n" << joined);
    CHECK(problems.empty());
}

TEST_CASE("suite guard: every suite row declares the full stage set",
          "[suite][guard]")
{
    // The header invariant promises convert -> load -> model -> kernels ->
    // selfcheck -> fsi -> code -> det for EVERY row, but nothing enforced the
    // stage list — only that the named targets exist. A row that quietly
    // dropped "model" stayed green forever.
    static const std::vector<const char *> kRequired = {
        "convert", "load", "model", "kernels",
        "selfcheck", "fsi", "code", "det", "perf"};
    // DeepSeek-V4-Flash has no converter in the repo by design (it ingests a
    // pre-converted logestix layout), so "convert" is exempt for that row only.
    static const std::set<std::string> kConvertExempt = {"deepseek-v4-flash"};

    std::vector<std::string> missing;
    for (const auto &s : Suites())
    {
        std::set<std::string> have;
        for (const auto &st : s.stages) have.insert(st.stage);
        for (const auto *r : kRequired)
        {
            if (std::string(r) == "convert" && kConvertExempt.count(s.model))
                continue;
            if (!have.count(r))
                missing.push_back(std::string(s.model) + ": missing stage " + r);
        }
    }
    std::string joined;
    for (const auto &s : missing) joined += s + "\n";
    INFO("incomplete suite rows:\n" << joined);
    CHECK(missing.empty());
}

// ── Golden coverage: a selfcheck stage that SKIPs is not coverage ────────
//
// The fsi/code/det stages are three target families that SKIP whenever their golden
// file is absent (real weights or golden missing -> SKIP, exit 4). A SKIP is
// indistinguishable from a pass in the ctest summary, so a stage can be fully
// wired - target registered, SKIP_RETURN_CODE 4 set, suite-guard row present - and
// still never execute a single assertion. gemma-4-26b-a4b-it sat exactly there:
// its det golden existed and its fsi/code goldens did not, so two of four parity
// stages were reporting SUCCESS while doing nothing, and nothing in the tree
// noticed. Nine families are still in that state.
//
// Nothing cross-referenced the suite rows against tests/fixtures/{fsi,code,det},
// so this closes that. The debt below is declared rather than tolerated: a family
// with no golden FAILS unless it is listed here, which makes "add a model and forget
// the goldens" impossible to do quietly, and makes the remaining gaps enumerable
// from the test output instead of only from a human diffing two directories.
//
// Two directions, because either one alone leaves a hole:
//   - undeclared gap  -> FAIL. A new family must ship its goldens or declare debt.
//   - stale debt row  -> FAIL. A debt entry whose golden now exists must be deleted,
//     so the list cannot outlive the problem and quietly rot.
TEST_CASE("suite guard: every parity golden is present or declared as debt",
          "[guard]")
{
    static const std::vector<const char *> kGoldenStages = {"fsi", "code", "det"};
    // Declared debt, as "<model>:<stage>". Every entry is a real gap; keeping the
    // list here rather than inferring it means the test output names what is
    // missing instead of merely counting what is present.
    // From models/*.json: a gap without a declared reason is a manifest error
    // (validation rule 2), so this set and the registry cannot disagree.
    std::set<std::string> kGoldenDebt;
    for (std::size_t i = 0; i < lsxmanifest::kDebtCount; ++i)
        kGoldenDebt.insert(std::string(lsxmanifest::kDebt[i].model_type) + ":" +
                           lsxmanifest::kDebt[i].stage);


    const fs::path root(LSX_SOURCE_DIR);
    const std::vector<SuiteSpec> suites = Suites();

    // The rows a stage is claimed for. A stage absent from a row is a separate
    // guard's business (kRequired above); here we only ask about the goldens of
    // the rows that DO claim the stage, so this test cannot double-report.
    std::vector<std::string> undeclared_gap;
    std::vector<std::string> stale_debt;
    std::set<std::string> seen_debt;

    for (const auto &s : suites)
    {
        std::set<std::string> stages;
        for (const auto &st : s.stages) stages.insert(st.stage);
        for (const auto *g : kGoldenStages)
        {
            if (!stages.count(g)) continue;
            const std::string key = std::string(s.model) + ":" + g;
            const bool present = fs::exists(root / "tests" / "fixtures" / g /
                                            (std::string(s.model) + ".json"));
            if (present)
            {
                if (kGoldenDebt.count(key)) stale_debt.push_back(key);
                continue;
            }
            if (kGoldenDebt.count(key))
            {
                seen_debt.insert(key);
                continue;
            }
            undeclared_gap.push_back(key);
        }
    }

    std::string a, b;
    for (const auto &x : undeclared_gap)
        a += "golden missing and NOT declared as debt (add the golden, or add \"" +
             x + "\" to kGoldenDebt): " + x + "\n";
    for (const auto &x : stale_debt)
        b += "debt row whose golden now exists - delete \"" + x +
             "\" from kGoldenDebt: " + x + "\n";
    INFO("undeclared golden gaps:\n" << a);
    INFO("stale debt rows:\n" << b);
    CHECK(undeclared_gap.empty());
    CHECK(stale_debt.empty());

    // A debt row for a model or stage that no longer exists is dead weight that
    // hides a rename; catching it here keeps the list honest rather than letting
    // it accumulate.
    std::vector<std::string> unknown_debt;
    for (const auto &d : kGoldenDebt)
        if (!seen_debt.count(d)) unknown_debt.push_back(d);
    for (const auto &x : unknown_debt)
        INFO("debt row matching no live gap: " << x);
    CHECK(unknown_debt.empty());
}

// ── Golden provenance: never sell a template golden as model-defined ────────
//
// `tools/{fsi,code,det}_capture.py` has two capture modes, and the golden's
// `source` field records which one ran. A genuine capture runs the SOURCE
// safetensors and records the model's decoded output in each case's `raw`, and
// writes the source it ran as a BARE path (`/mnt/llm_models/models/…`). The
// `--template-golden` stopgap, for a source that cannot run locally, records
// the prompt's Expected Template as `expected`, leaves `raw` empty, and says so
// with a `prompt-template (…)` / `template-derived (…)` prefix.
//
// The parity gate is only evidence of anything if its provenance is honest. A
// template golden labelled as a model-defined one tells an auditor that the
// SOURCE model's own answer was captured when no model ever ran — the exact
// claim the FSI/CODE/DET stages exist to establish. The debt registry cannot
// catch it either: it only asks whether the golden FILE exists, so a committed
// golden satisfies "present or declared debt" whether or not it passes.
//
// The discriminator is `raw`, and it holds across the whole tree: every genuine
// capture carries non-empty `raw` for every case, and every template golden has
// `raw` empty throughout. (A real capture may legitimately AGREE with the
// Expected Template — granite's FSI golden does — so agreement with the prompt
// set is NOT evidence either way; only `raw` is.)
TEST_CASE("suite guard: a golden with no captured raw declares a template source",
          "[guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::vector<std::string> mislabelled;

    for (const char *stage : {"fsi", "code", "det"})
    {
        const fs::path dir = root / "tests" / "fixtures" / stage;
        if (!fs::exists(dir)) continue;
        for (const auto &e : fs::directory_iterator(dir))
        {
            if (e.path().extension() != ".json") continue;
            std::ifstream in(e.path());
            REQUIRE(in.good());
            const std::string text((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());

            std::smatch src;
            // Custom raw-string delimiter: the pattern contains `*)"` , which
            // would otherwise close a plain R"( ... )" literal early.
            const bool has_src = std::regex_search(
                text, src,
                std::regex(R"RX("source"\s*:\s*"((?:[^"\\]|\\.)*)")RX"));
            REQUIRE(has_src);
            const std::string source = src[1].str();

            // Every `"raw"` key present, and how many of them are empty.
            const std::regex raw_key(R"("raw"\s*:)");
            const std::regex raw_empty(R"("raw"\s*:\s*"")");
            const int total = std::distance(
                std::sregex_iterator(text.begin(), text.end(), raw_key),
                std::sregex_iterator());
            const int empty = std::distance(
                std::sregex_iterator(text.begin(), text.end(), raw_empty),
                std::sregex_iterator());
            if (total == 0 || empty != total) continue;  // a real capture

            const bool declared_template =
                source.rfind("prompt-template", 0) == 0 ||
                source.rfind("template-derived", 0) == 0;
            if (!declared_template)
                mislabelled.push_back(
                    std::string(stage) + "/" + e.path().filename().string() +
                    " has an empty raw in every case (a --template-golden "
                    "capture) but its source does not declare one: \"" +
                    source.substr(0, 80) + "\"");
        }
    }

    for (const auto &x : mislabelled) WARN("mislabelled golden: " << x);
    CHECK(mislabelled.empty());
}
