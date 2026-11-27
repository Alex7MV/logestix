// The parity gates — FSI, CODE and DET — as ONE driver.
//
// Each model ran three near-identical files (28 lines of body, differing only in
// `kModelType`, a vault path and a SKIP string). Copying them shipped three
// files whose SKIP named a NEIGHBOUR model, so a clean skip read as "the wrong
// test ran"; test_parity_selfcheck_naming_guard existed only to police that.
// This driver removes the copy: the model and the gate kind are compile-time
// constants per target, and everything else comes from the model's manifest.
//
// SKIP (exit 4) is not a pass: it means the converted tree or the golden is
// absent, and the message names THIS model and the directory it looked in.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>

#include <lsxcommon/code_selfcheck.h>
#include <lsxcommon/det_selfcheck.h>
#include <lsxcommon/fsi_selfcheck.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>

#include "test_manifest_query.h"
#include "test_manifest_query_root.h"

#ifndef LSX_GATE_MODEL
#error "LSX_GATE_MODEL must be defined (the model_type of one manifest row)"
#endif
#ifndef LSX_GATE_KIND
#error "LSX_GATE_KIND must be defined (fsi, code or det)"
#endif
#include <lsxcommon/path_registry.h>
// The golden directory MUST come from CMake, and a missing define is a BUILD
// ERROR rather than a relative fallback. A fallback like "tests/fixtures/fsi" is
// resolved against the process CWD, and ctest runs from the build dir, so the gate
// found no golden, SKIPped, and reported a skip that reads as "not applicable"
// while the golden sat in the source tree. The same mistake made both
// granite-switch model targets skip their real-weight tool round-trip
// (docs/OPEN_TASKS.md G7).
//
// ONE macro, not one per kind: the kind is a string and cannot be compared in
// `#if`, so three per-kind macros forced three "define it or fall back" branches
// here — which is what made the fallback tempting in the first place. CMake knows
// the directories and passes the right one.
#ifndef LSX_GOLDEN_DIR
#error "LSX_GOLDEN_DIR must be defined by CMake (absolute path to fixtures/<kind>)"
#endif

namespace {

constexpr std::string_view kModel = LSX_GATE_MODEL;
constexpr std::string_view kKind = LSX_GATE_KIND;
static_assert(kKind == "fsi" || kKind == "code" || kKind == "det",
              "LSX_GATE_KIND must be fsi, code or det");

std::filesystem::path GoldenDir()
{
    // The env override keeps its per-kind name: it is a documented ad-hoc knob
    // (see the fsi/code/det skills) and a caller setting it expects that spelling.
    if (const char *env = std::getenv(
            kKind == "fsi" ? "LSX_FSI_GOLDEN_DIR"
                           : kKind == "code" ? "LSX_CODE_GOLDEN_DIR"
                                             : "LSX_DET_GOLDEN_DIR"))
        return std::filesystem::path(env);
    return std::filesystem::path(LSX_GOLDEN_DIR);
}

std::filesystem::path GoldenPath()
{
    return GoldenDir() / (std::string(kModel) + ".json");
}

// Loads the model the manifest names and asserts it opens. Included in the
// driver so each gate cannot drift into its own config.
std::unique_ptr<lsxcommon::IModel> LoadModel(const lsxmanifest::ModelRow &row,
                                            const std::filesystem::path &root)
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = row.model_type;
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;   // the goldens pin kOff
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    return model;
}

// A failing gate must say WHAT failed. `INFO` inside the case loop is out of
// scope by the time the aggregate `REQUIRE` runs, so Catch2 printed nothing and a
// wrong answer, an empty generation and an inference error all looked the same
// ("all_passed == false"). Collect the per-case verdicts and attach them at the
// assertion instead.
template <typename Cases>
std::string ExplainCases(const Cases &cases)
{
    // The RAW text is included, tail-first and truncated: without it, "the model
    // reasoned correctly but the extraction latched onto a reasoning line", "the
    // budget cut the answer mid-thought" and "the answer is wrong" all print
    // identically as extracted != expected. The tail matters because a reasoning
    // model states its answer LAST.
    std::string why;
    for (const auto &c : cases)
    {
        std::string tail = c.raw;
        if (tail.size() > 400) tail = "..." + tail.substr(tail.size() - 400);
        why += "\n  " + c.id + " expected='" + c.expected + "' extracted='" +
               c.extracted + "' error='" + c.error + "'\n    raw=" + tail;
    }
    return why;
}

} // namespace

TEST_CASE("parity gate: golden answers match", "[parity][real-model]")
{
    const lsxmanifest::ModelRow *row = manifest_query::FindModel(kModel);
    // A gate target with no manifest row is a bug in the registration, not a
    // skip: the whole point is that the manifest owns this model.
    REQUIRE(row != nullptr);

    const std::filesystem::path root = manifest_query::ModelRoot(*row);
    if (!std::filesystem::exists(root / "model.index.json"))
        SKIP(manifest_query::SkipMessage(*row, "converted model"));

    const std::filesystem::path golden_path = GoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP(std::string(row->model_type) + " " + std::string(kKind) +
             " golden not captured (declared debt: " +
             (manifest_query::IsDeclaredDebt(kModel, kKind) ? "yes" : "NO") + ")");

    lsxcommon::EnsureModelRegistry();
    auto model = LoadModel(*row, root);

    if constexpr (std::string_view(LSX_GATE_KIND) == "fsi")
    {
        lsxcommon::fsi::FsiGolden golden;
        std::string err;
        REQUIRE(lsxcommon::fsi::LoadFsiGoldenFile(golden_path.string(), golden, &err));
        REQUIRE(golden.model_type == kModel);
        const auto result = lsxcommon::fsi::RunFsiSelfCheck(*model, golden);
        INFO(ExplainCases(result.cases));
        REQUIRE(result.all_passed);
    }
    else if constexpr (std::string_view(LSX_GATE_KIND) == "code")
    {
        lsxcommon::code::CodeGolden golden;
        std::string err;
        REQUIRE(lsxcommon::code::LoadCodeGoldenFile(golden_path.string(), golden, &err));
        REQUIRE(golden.model_type == kModel);
        const auto result = lsxcommon::code::RunCodeSelfCheck(*model, golden);
        INFO(ExplainCases(result.cases));
        REQUIRE(result.all_passed);
    }
    else
    {
        lsxcommon::det::DetGolden golden;
        std::string err;
        REQUIRE(lsxcommon::det::LoadDetGoldenFile(golden_path.string(), golden, &err));
        REQUIRE(golden.model_type == kModel);
        const auto result = lsxcommon::det::RunDetSelfCheck(*model, golden);
        INFO(ExplainCases(result.cases));
        REQUIRE(result.all_passed);
    }

    model->Unload();
}
