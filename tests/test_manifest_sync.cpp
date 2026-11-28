// Model manifest sync guard (host-only, label fast).
//
// `models/<org>/<model>.json` is the single source of truth for the model
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
#include "lsxstorage/roots.h"

#include <catch2/catch_test_macros.hpp>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <lsxstorage/roots.h>

#include "test_manifest_query_root.h"

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
                        r.format_tag, "\t", r.vault_org, "\t", r.vault_dir,
                        "\t", r.vault_profile, "\t", r.vault_env,
                        "\t", r.tokenizer_kind, "\t",
                        r.tool_roundtrip ? 1 : 0, "\t",
                        r.auto_path_determinism ? 1 : 0, "\t",
                        r.cooperative_cancel ? 1 : 0, "\t",
                        r.selfcheck_frame_prompt ? 1 : 0, "\t",
                        r.selfcheck_max_gen, "\t",
                        r.selfcheck_bit_exact ? 1 : 0);
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

// ── Store layout: a row must name a place that exists ─────────────────────
//
// The store is ORG-NESTED (`<source root>/<org>/<model>`), the way Hugging Face
// lays a download out, and the artifact store mirrors it
// (`<artifact root>/<org>/<model>/<profile>`). A row that carried only a bare
// directory name resolved to `<root>/phi-4`, a path that has never existed:
// every real-weight gate then SKIPped for a reason ("model.index.json is not
// there") that named the symptom and hid the cause.
//
// This asserts the two halves the registry owes the gates: the SOURCE path is
// real, and `ModelRoot()` names the ARTIFACT (which is what every gate driver
// can actually load), not the checkpoint.
namespace {

bool HoldsCheckpoint(const std::filesystem::path &dir)
{
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return false;
    for (const auto &e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file()) continue;
        const std::string n = e.path().filename().string();
        if (n == "config.json" || n == "params.json") return true;
        if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0)
            return true;
    }
    return false;
}

}  // namespace

TEST_CASE("manifest sync: every row names its org and its artifact profile",
          "[manifest][fast]")
{
    // Non-vacuity first: the table must be big enough for the loop to mean
    // something, and every row must carry a usable identity.
    REQUIRE(lsxmanifest::kModelCount > 40);
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        INFO("model " << row.model_type);
        REQUIRE(row.vault_org != nullptr);
        REQUIRE(std::string(row.vault_org).find('/') == std::string::npos);
        REQUIRE(std::string(row.vault_dir) != "");
        REQUIRE(row.vault_profile != nullptr);
        REQUIRE(std::string(row.vault_profile) != "");
    }
}

TEST_CASE("manifest sync: the source path a row names holds its checkpoint",
          "[manifest][fast]")
{
    // Only rows whose weights are on this box can be checked; a row whose
    // directory is an empty placeholder is a download gap, not a manifest bug.
    const std::optional<std::filesystem::path> configured =
        lsx::storage::SourceRoot();
    if (!configured)
        SKIP("LSX_SOURCE_ROOT is not set — there is no checkpoint store to "
             "resolve against, and it has no default");
    const std::filesystem::path source_root = *configured;
    std::error_code ec;
    if (!std::filesystem::is_directory(source_root, ec))
        SKIP("source store not present: " << source_root.string());

    int checked = 0;
    int placeholders = 0;
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        const std::filesystem::path dir =
            source_root / row.vault_org / row.vault_dir;
        INFO("model " << row.model_type << " -> " << dir.string());
        // The directory must EXIST: a manifest that names an org the store does
        // not have points the operator at a path no `ls` shows.
        const bool exists = std::filesystem::is_directory(dir, ec);
        REQUIRE(exists);
        if (!HoldsCheckpoint(dir)) { ++placeholders; continue; }
        ++checked;
    }
    INFO("rows with weights on disk: " << checked
                                       << ", empty placeholders: " << placeholders);
    // A green run that checked nothing would prove nothing.
    REQUIRE(checked > 0);
}

TEST_CASE("manifest sync: ModelRoot resolves the ARTIFACT, not the checkpoint",
          "[manifest][fast]")
{
    // The gates load converted trees; `model.index.json` and `artifact.fb` live
    // in the artifact store and never in a Hugging Face download. A ModelRoot
    // that answers with the source directory makes every real-weight target
    // SKIP by default and work only when an operator exports
    // LSX_TEST_<MODEL>_ROOT by hand — which is how every measured verdict in
    // docs/SELFCHECK_MATRIX.md was obtained, i.e. the registry described a path
    // nobody used.
    const std::optional<std::filesystem::path> root = lsx::storage::ArtifactRoot();
    if (!root) SKIP("LSX_ARTIFACT_ROOT is unset — there is no artifact store to "
                    "resolve against, and inventing a default is the bug");
    const std::filesystem::path artifact_root = *root;
    int checked = 0;
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        const std::filesystem::path want =
            artifact_root / row.vault_org / row.vault_dir / row.vault_profile;
        std::error_code ec;
        if (!std::filesystem::exists(want / "artifact.fb", ec)) continue;

        // The env override still wins (an operator may point at a tree
        // anywhere), so the default path is what is under test here.
        const std::filesystem::path got = manifest_query::ModelRoot(row);
        INFO("model " << row.model_type << ": ModelRoot='" << got.string()
                      << "', artifact='" << want.string() << "'");
        CHECK(got == want);
        CHECK(std::filesystem::exists(got / "artifact.fb", ec));
        ++checked;
    }
    INFO("rows with an artifact on disk: " << checked);
    // The two bootstrap artifacts must be covered, or this case is vacuous.
    REQUIRE(checked >= 2);
}

TEST_CASE("manifest sync: ModelRoot without an artifact still names the "
          "artifact path",
          "[manifest][fast]")
{
    // For a row with no artifact, answering with the SOURCE directory makes the
    // gate's SKIP message name a file the checkpoint never holds ("no
    // model.index.json"), which reads as a broken conversion instead of "not
    // converted yet". The path must point where the artifact WOULD be.
    const std::optional<std::filesystem::path> root = lsx::storage::ArtifactRoot();
    if (!root) SKIP("LSX_ARTIFACT_ROOT is unset — there is no artifact store to "
                    "resolve against, and inventing a default is the bug");
    const std::filesystem::path artifact_root = *root;
    int checked = 0;
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        const std::filesystem::path want =
            artifact_root / row.vault_org / row.vault_dir / row.vault_profile;
        std::error_code ec;
        if (std::filesystem::exists(want / "artifact.fb", ec)) continue;
        const std::filesystem::path got = manifest_query::ModelRoot(row);
        INFO("model " << row.model_type << ": ModelRoot='" << got.string() << "'");
        CHECK(got == want);
        ++checked;
        if (checked >= 5) break;   // one honest sample is enough
    }
    INFO("rows without an artifact checked: " << checked);
    REQUIRE(checked > 0);
}

// ── The catalogue must describe the STORE, not a bare directory name ──────
//
// `docs/MODELS.md` is the only per-model page a reader gets, and it said
// "- vault `phi-4`". Nobody can find that: the checkpoint is
// `<source root>/microsoft/phi-4` and the artifact is
// `<artifact root>/microsoft/phi-4/nvfp4`. A catalogue that omits the org and
// the profile sends the reader to a path no `ls` shows, which is how the
// registry came to describe a layout the store had already stopped using.
TEST_CASE("manifest sync: the catalogue names each row's store path",
          "[manifest][fast]")
{
    const std::filesystem::path docs =
        std::filesystem::path(LSX_SOURCE_DIR) / "docs" / "MODELS.md";
    std::ifstream in(docs);
    REQUIRE(in.good());
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());

    // Every row must appear with its qualified store path: org/model, and the
    // artifact profile. Checked for all rows rather than a sample, because the
    // one row a sample skips is the row that regresses.
    std::vector<std::string> missing;
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        const std::string want =
            std::string(row.vault_org) + "/" + row.vault_dir;
        if (text.find(want) == std::string::npos)
            missing.push_back(std::string(row.model_type) + ": " + want);
    }
    std::string joined;
    for (const auto &m : missing) joined += "  " + m + "\n";
    INFO("rows whose qualified store path is absent from docs/MODELS.md:\n"
         << joined);
    CHECK(missing.empty());

    // And the profile ON THE ROW, so "is this the nvfp4 tree or a bf16 one?"
    // is answerable from the model's own section rather than by guessing. A
    // document that mentioned the profile once for all 47 rows would pass a
    // weaker check and help nobody.
    std::vector<std::string> no_profile;
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        const std::string prefix = "- store `" + std::string(row.vault_org) +
                                   "/" + row.vault_dir + "`";
        const std::size_t at = text.find(prefix);
        if (at == std::string::npos) continue;   // reported by the check above
        const std::size_t eol = text.find('\n', at);
        const std::string line = text.substr(at, eol - at);
        if (line.find(std::string("`") + row.vault_profile + "`") ==
            std::string::npos)
            no_profile.push_back(std::string(row.model_type) + ": " + line);
    }
    std::string joined_profile;
    for (const auto &m : no_profile) joined_profile += "  " + m + "\n";
    INFO("rows whose store line omits the profile:\n" << joined_profile);
    CHECK(no_profile.empty());
}

// ── The compiled table must carry the profile the registry declares ───────
//
// `profiles` is the registry's answer to "what does this model hold, and how was
// each variant measured". If the block never reaches the compiled table, every
// guard and gate driver reads a registry that knows one profile per model while
// the manifests describe several — and the manifest becomes documentation.
TEST_CASE("manifest sync: the compiled row carries the preferred profile",
          "[manifest][fast]")
{
    // Non-vacuity: the table must be the real registry.
    REQUIRE(lsxmanifest::kModelCount > 40);
    for (std::size_t i = 0; i < lsxmanifest::kModelCount; ++i)
    {
        const auto &row = lsxmanifest::kModels[i];
        INFO("model " << row.model_type);
        REQUIRE(row.vault_profile != nullptr);
        REQUIRE(std::string(row.vault_profile) != "");
        // The profile a gate looks for must be a profile the store can hold,
        // i.e. it names a directory level, never a path or an empty string.
        const std::string profile = row.vault_profile;
        CHECK(profile.find('/') == std::string::npos);
    }

    // And the two bootstrap artifacts must agree with their manifests: the
    // profile directory on disk IS the row's preferred profile.
    const auto *phi = manifest_query::FindModel("phi-4");
    REQUIRE(phi != nullptr);
    CHECK(std::string(phi->vault_profile) == "nvfp4");
    CHECK(std::string(phi->vault_org) == "microsoft");

    const auto *qwen = manifest_query::FindModel("qwen3.5-9b");
    REQUIRE(qwen != nullptr);
    CHECK(std::string(qwen->vault_profile) == "nvfp4");
    CHECK(std::string(qwen->vault_org) == "Qwen");
}
