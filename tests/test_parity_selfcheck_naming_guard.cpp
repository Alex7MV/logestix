// Guard: a parity selfcheck must name ITS OWN model (label fast).
//
// The three parity stages (FSI / CODE / DET) share one body, so the files are
// produced by copying a sibling. Two leftovers survive such a copy and both make
// the gate LIE while still reporting a clean SKIP:
//
//   1. `TEST_CASE("xing fsi selfcheck: ...")` inside, say, the hy or mistral
//      file — ctest then prints "xing" for a model that is not xing.
//   2. `SKIP("FSI golden not captured for xing4.0-29b-a4b")` in another
//      family's file, and `[fsi]` tags on a CODE/DET target so a label filter
//      selects the wrong stage.
//
// This is not cosmetic: while running the mistral-small-4 gates, a SKIP reading
// "not captured for xing4.0-29b-a4b" from `test_mistral_moe_code_selfcheck`
// reads as "the wrong test ran", and a report that quotes it is wrong.
//
// The registry below lists the files that predate this guard. It is two-way: an
// entry must still be broken (once a file is fixed, its entry FAILS and must be
// deleted), so the debt cannot silently outlive the fix.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

namespace fs = std::filesystem;

// Family stem -> the model_type strings that belong to it. Used both to decide
// whether a file names a FOREIGN family and to resolve a SKIP's model name.
const std::vector<std::string> kFamilies = {
    "alice", "beam",   "deepseek", "flint", "gemma", "gigachat", "glm",
    "granite", "hy",   "kimi",     "ling",  "mimo",  "minimax", "mistral",
    "nemotron", "ornith", "phi",   "qwen",  "step",  "xing"};

std::string ReadFile(const fs::path &p)
{
    std::ifstream f(p);
    return std::string(std::istreambuf_iterator<char>(f),
                       std::istreambuf_iterator<char>());
}

std::string Lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

// "test_<family>_..._<stage>_selfcheck.cpp" -> (family, stage)
struct FileInfo {
    std::string name;
    std::string family;
    std::string stage;
};

// Collects every kModelType declared across the parity files, so a SKIP message
// can be checked against the OTHER models' identifiers rather than a guess.
std::map<std::string, std::string> ModelTypesByFile(const fs::path &tests)
{
    std::map<std::string, std::string> out;
    const std::regex re("kModelType\\s*=\\s*\"([^\"]+)\"");
    for (const auto &e : fs::directory_iterator(tests)) {
        if (!e.is_regular_file() || e.path().extension() != ".cpp") continue;
        const std::string text = ReadFile(e.path());
        std::smatch m;
        if (std::regex_search(text, m, re)) out[e.path().filename().string()] = m[1];
    }
    return out;
}

std::vector<FileInfo> ParityFiles(const fs::path &tests)
{
    std::vector<FileInfo> out;
    const std::regex name_re(R"(^test_([a-z0-9]+)_.*_(fsi|code|det)_selfcheck\.cpp$)");
    for (const auto &e : fs::directory_iterator(tests)) {
        if (!e.is_regular_file()) continue;
        const std::string n = e.path().filename().string();
        std::smatch m;
        if (!std::regex_match(n, m, name_re)) continue;
        out.push_back({n, m[1], m[2]});
    }
    std::sort(out.begin(), out.end(),
              [](const FileInfo &a, const FileInfo &b) { return a.name < b.name; });
    return out;
}

// A file that predates this guard: "<file>:<rule>" -> why it is still unfixed.
const std::map<std::string, std::string> kKnownOffenders = {
    // Copied from the xing file: TEST_CASE and the SKIP message still say xing.
    {"test_hy_moe_code_selfcheck.cpp:testcase", "xing leftover"},
    {"test_hy_moe_det_selfcheck.cpp:testcase", "xing leftover"},
    {"test_hy_moe_code_selfcheck.cpp:skip", "xing leftover"},
    {"test_hy_moe_det_selfcheck.cpp:skip", "xing leftover"},
    {"test_hy_moe_simulmt_code_selfcheck.cpp:testcase", "xing leftover"},
    {"test_hy_moe_simulmt_det_selfcheck.cpp:testcase", "xing leftover"},
    {"test_hy_moe_simulmt_code_selfcheck.cpp:skip", "xing leftover"},
    {"test_hy_moe_simulmt_det_selfcheck.cpp:skip", "xing leftover"},
    {"test_minimax_moe_code_selfcheck.cpp:testcase", "xing leftover"},
    {"test_minimax_moe_det_selfcheck.cpp:testcase", "xing leftover"},
    {"test_minimax_moe_code_selfcheck.cpp:skip", "xing leftover"},
    {"test_minimax_moe_det_selfcheck.cpp:skip", "xing leftover"},
    {"test_minimax_moe_flash_code_selfcheck.cpp:testcase", "xing leftover"},
    {"test_minimax_moe_flash_det_selfcheck.cpp:testcase", "xing leftover"},
    {"test_minimax_moe_flash_code_selfcheck.cpp:skip", "xing leftover"},
    {"test_minimax_moe_flash_det_selfcheck.cpp:skip", "xing leftover"},
    {"test_mistral_moe_large4_code_selfcheck.cpp:testcase", "xing leftover"},
    {"test_mistral_moe_large4_det_selfcheck.cpp:testcase", "xing leftover"},
    {"test_mistral_moe_large4_code_selfcheck.cpp:skip", "xing leftover"},
    {"test_mistral_moe_large4_det_selfcheck.cpp:skip", "xing leftover"},
    // SKIP names a sibling family's model_type.
    {"test_beam_moe_fsi_selfcheck.cpp:skip", "names hy-memory-30b-a3b"},
    {"test_beam_moe_code_selfcheck.cpp:skip", "names hy-memory-30b-a3b"},
    {"test_beam_moe_det_selfcheck.cpp:skip", "names hy-memory-30b-a3b"},
    {"test_flint_moe_fsi_selfcheck.cpp:skip", "names hy-memory-30b-a3b"},
    {"test_flint_moe_code_selfcheck.cpp:skip", "names hy-memory-30b-a3b"},
    {"test_flint_moe_det_selfcheck.cpp:skip", "names hy-memory-30b-a3b"},
    {"test_minimax_moe_fsi_selfcheck.cpp:skip", "names mistral-small-4-119b"},
    // CODE/DET targets carrying the FSI tag.
    {"test_beam_moe_code_selfcheck.cpp:tag", "tag says [fsi]"},
    {"test_beam_moe_det_selfcheck.cpp:tag", "tag says [fsi]"},
    {"test_flint_moe_code_selfcheck.cpp:tag", "tag says [fsi]"},
    {"test_flint_moe_det_selfcheck.cpp:tag", "tag says [fsi]"},
    {"test_ling_moe_code_selfcheck.cpp:tag", "tag says [fsi]"},
    {"test_ling_moe_det_selfcheck.cpp:tag", "tag says [fsi]"},
};

} // namespace

TEST_CASE("parity selfchecks name their own model", "[guard][fast]")
{
    const fs::path tests = fs::path(LSX_SOURCE_DIR) / "tests";
    REQUIRE(fs::exists(tests));

    const auto files = ParityFiles(tests);
    INFO("parity selfcheck files found: " << files.size());
    REQUIRE(files.size() > 20);
    const auto model_types = ModelTypesByFile(tests);

    std::vector<std::string> broken;
    std::set<std::string> seen;

    for (const auto &f : files) {
        const std::string text = ReadFile(tests / f.name);

        // Rule 1: the TEST_CASE name must not start with another family's stem.
        static const std::regex tc_re("TEST_CASE\\(\"([^\"]+)\"");
        std::smatch tc;
        if (std::regex_search(text, tc, tc_re)) {
            const std::string first =
                Lower(std::string(tc[1]).substr(0, std::string(tc[1]).find(' ')));
            const bool own = first.rfind(f.family, 0) == 0;
            bool foreign = false;
            for (const auto &fam : kFamilies)
                if (fam != f.family && first.rfind(fam, 0) == 0) foreign = true;
            if (!own && foreign) broken.push_back(f.name + ":testcase");
            seen.insert(f.name + ":testcase");
        }

        // Rule 2: the target's own Catch2 tag must carry its stage.
        static const std::regex tag_re(
            "TEST_CASE\\(\"[^\"]*\",\\s*\"(\\[[^\"]+\\])\"");
        std::smatch tag;
        if (std::regex_search(text, tag, tag_re)) {
            if (tag[1].str().find("[" + f.stage + "]") == std::string::npos)
                broken.push_back(f.name + ":tag");
            seen.insert(f.name + ":tag");
        }

        // Rule 3: a "not captured for X" SKIP must name THIS file's model.
        // X is compared against the other files' kModelType values, so a
        // human-readable product name (alice, Ling-3.1-flash) is fine while a
        // sibling family's identifier is not.
        static const std::regex skip_re(
            "not captured for ([A-Za-z0-9._-]+)");
        const std::string own_model =
            model_types.count(f.name) ? model_types.at(f.name) : std::string();
        for (auto it = std::sregex_iterator(text.begin(), text.end(), skip_re);
             it != std::sregex_iterator(); ++it) {
            const std::string named = (*it)[1];
            if (named == own_model) continue;
            bool foreign = false;
            for (const auto &[other_file, other_model] : model_types)
                if (other_file != f.name && other_model == named) foreign = true;
            if (foreign) broken.push_back(f.name + ":skip");
            seen.insert(f.name + ":skip");
        }
    }

    // Two-way registry: every entry must still be broken, and every breakage
    // must be declared.
    std::sort(broken.begin(), broken.end());
    broken.erase(std::unique(broken.begin(), broken.end()), broken.end());

    std::vector<std::string> undeclared, stale;
    for (const auto &b : broken)
        if (!kKnownOffenders.count(b)) undeclared.push_back(b);
    for (const auto &[key, why] : kKnownOffenders)
        if (!std::binary_search(broken.begin(), broken.end(), key))
            stale.push_back(key + " (" + why + ")");

    if (!undeclared.empty()) {
        std::string msg = "parity selfchecks naming a foreign model:";
        for (const auto &u : undeclared) msg += "\n  " + u;
        FAIL(msg);
    }
    if (!stale.empty()) {
        std::string msg =
            "kKnownOffenders entries are fixed - delete them:";
        for (const auto &s : stale) msg += "\n  " + s;
        FAIL(msg);
    }
}
