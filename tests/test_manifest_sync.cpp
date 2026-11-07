// Model manifest sync guard (host-only, label fast).
//
// `models/<model_type>.json` is the single source of truth for the model
// registry, and every other representation is generated from it. This test is
// the machine that keeps that true:
//
//   1. the COMPILED table (model_manifest.h) equals the generator's own
//      projection of the manifests - one JSON implementation, two views. A
//      difference means a stale build tree or a wrong emitter.
//   2. the COMMITTED docs (docs/MODELS.md) are byte-identical to a fresh
//      regeneration (`--check`), so nobody can hand-edit the catalogue.
//   3. every tests/test_*.{cpp,cu} file is claimed by a manifest (or is a
//      declared infrastructure test). This is the check that a model's tests
//      cannot silently stop running: Review Focus #1 of the migration plan.
//
// A mutated manifest MUST fail here - see the plan's Task 10 negative test.
#include <catch2/catch_test_macros.hpp>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>

#include "model_manifest.h"

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif
#ifndef LSX_PYTHON3
#define LSX_PYTHON3 "python3"
#endif
#ifndef LSX_GENERATED_DIR
#define LSX_GENERATED_DIR ""
#endif

namespace {

namespace fs = std::filesystem;

const fs::path kRoot(LSX_SOURCE_DIR);

// Runs the generator with `args` and returns (exit_code, stdout_text).
// Output goes through a temp file: std::system cannot capture a pipe portably
// (this test also has to build under MSVC).
std::pair<int, std::string> RunGenerator(const std::string &args)
{
    const fs::path out =
        fs::temp_directory_path() / "lsx_manifest_projection.txt";
    std::error_code ec;
    fs::remove(out, ec);
    const std::string cmd = absl::StrCat("\"", LSX_PYTHON3, "\" \"",
                                         (kRoot / "tools" / "gen_model_manifest.py").string(),
                                         "\" --root \"", kRoot.string(), "\" ",
                                         args, " > \"", out.string(), "\"");
    const int rc = std::system(cmd.c_str());
    std::ifstream in(out);
    std::string text((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    return {rc, text};
}

std::vector<std::string> Lines(const std::string &text)
{
    std::vector<std::string> out;
    for (absl::string_view line : absl::StrSplit(text, '\n'))
        if (!line.empty()) out.emplace_back(line);
    return out;
}

std::string RowKey(const lsxmanifest::ModelRow &r)
{
    return absl::StrCat(r.model_type, "\t", r.family, "\t",
                        r.variant ? r.variant : "", "\t", r.model_class, "\t",
                        r.model_header, "\t", r.passport, "\t", r.ns, "\t",
                        r.format_tag, "\t", r.vault_dir, "\t", r.vault_env,
                        "\t", r.tokenizer_kind, "\t",
                        r.tool_roundtrip ? 1 : 0, "\t",
                        r.auto_path_determinism ? 1 : 0, "\t",
                        r.cooperative_cancel ? 1 : 0);
}

std::string Escaped(const std::string &s)
{
    std::string out;
    for (char c : s)
    {
        if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

} // namespace

TEST_CASE("manifest sync: generated table matches the manifests",
          "[manifest][fast]")
{
    REQUIRE(lsxmanifest::kModelCount > 0);   // a tree with no manifests is a bug
    const auto [rc, text] = RunGenerator("--print-model-rows");
    INFO("generator exit " << rc);
    REQUIRE(rc == 0);

    std::vector<std::string> generated = Lines(text);
    REQUIRE(generated.size() == lsxmanifest::kModelCount);

    std::vector<std::string> compiled;
    compiled.reserve(lsxmanifest::kModelCount);
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        compiled.push_back(RowKey(row));
        // The compiled table must be sorted by model_type so that a diff of the
        // two lists points at the model, not at an arbitrary permutation.
        if (i > 0)
            CHECK(std::string(lsxmanifest::kModels[i - 1].model_type) <
                  std::string(row.model_type));
    }

    for (std::size_t i = 0; i < compiled.size(); ++i)
    {
        INFO("model row " << i << ": compiled='" << Escaped(compiled[i])
                          << "' generated='" << Escaped(generated[i]) << "'");
        REQUIRE(compiled[i] == generated[i]);
    }

    // Every target row must belong to a declared model and name its stage.
    for (std::size_t i = 0; i < lsxmanifest::kTargetCount; ++i)
    {
        const auto &t = lsxmanifest::kTargets[i];
        INFO("target " << t.target << " of " << t.model_type);
        bool known = false;
        for (std::size_t m = 0; m < lsxmanifest::kModelCount; ++m)
            if (std::string(lsxmanifest::kModels[m].model_type) == t.model_type)
                known = true;
        REQUIRE(known);
        REQUIRE(t.stage != nullptr);
        REQUIRE(t.label != nullptr);
    }
}

TEST_CASE("manifest sync: --check reports no drift", "[manifest][fast]")
{
    // --check: the committed catalogue is current. --strict: every test file is
    // owned. Both together are the state a release must be in.
    const auto [rc, _unused] = RunGenerator("--check --strict");
    INFO("regenerate docs/MODELS.md with: tools/gen_model_manifest.py --docs "
         "docs/MODELS.md --root .");
    REQUIRE(rc == 0);
}

TEST_CASE("manifest sync: the factory registers every manifest row",
          "[manifest][fast]")
{
    // The registry TU used to be a hand-maintained list of 35 Register<T>() calls
    // plus 3 vault-normalized aliases. A model added to the manifests but not to
    // that list "does not exist" at runtime while every guard stays green, so the
    // list is generated and this case proves it reaches every row.
    lsxcommon::EnsureModelRegistry();
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        lsxcommon::ModelInitConfig cfg;
        cfg.model_type = row.model_type;
        cfg.model_path = "/nonexistent";   // construction must not need weights
        INFO("model_type " << row.model_type);
        REQUIRE(lsxcommon::LsxModelFactory::Create(cfg) != nullptr);
    }

    // Aliases resolve to the same class (that is their whole purpose).
    const char *alias = std::getenv("LSX_TEST_ALIAS_PROBE");
    if (alias)
    {
        lsxcommon::ModelInitConfig cfg;
        cfg.model_type = alias;
        cfg.model_path = "/nonexistent";
        REQUIRE(lsxcommon::LsxModelFactory::Create(cfg) != nullptr);
    }
}

TEST_CASE("manifest sync: no unclaimed model test file", "[manifest][fast]")
{
    // Every tests/test_*.{cpp,cu} is either claimed by a manifest stage (or is
    // the second TU of a claimed target) or is a frozen infrastructure test.
    // A model test that no manifest names would silently stop running when the
    // ctest registration moves to the manifest: Review Focus #1.
    const auto [rc, text] = RunGenerator("--print-non-model-tests");
    INFO("generator exit " << rc);
    REQUIRE(rc == 0);
    const auto lines = Lines(text);
    const std::set<std::string> allowed(lines.begin(), lines.end());
    REQUIRE(allowed.size() > 100);

    std::vector<std::string> offenders;
    std::size_t seen = 0;
    for (const auto &entry : fs::directory_iterator(kRoot / "tests"))
    {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind("test_", 0) != 0) continue;
        const auto ext = entry.path().extension().string();
        if (ext != ".cpp" && ext != ".cu") continue;
        ++seen;
        if (!allowed.count(entry.path().stem().string())) offenders.push_back(name);
    }
    INFO("scanned " << seen << " test files");
    REQUIRE(seen > 100);   // a directory scan that finds nothing proves nothing

    std::string report;
    for (const auto &n : offenders)
        report += "unclaimed model test file (add it to a manifest stage, or to "
                  "the generator's NON_MODEL_TESTS): " + n + "\n";
    INFO(report);
    REQUIRE(offenders.empty());
}
