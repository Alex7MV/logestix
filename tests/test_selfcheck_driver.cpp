// SelfCheck — the "Paris" gate — as ONE driver with declared hooks.
//
// Each model used to carry its own file: the same "The capital of France is" ->
// "Paris" case, >= 3 bit-exact cycles, an optional tool round-trip and an
// optional kAuto determinism case, differing only in a vault path and a cycle
// count env var. Thirty-five copies of that is how the family-name drift in the
// parity gates started, so the shapes are DECLARED here instead:
//
//   paris_cycles           always: Paris + >= 3 bit-exact final_logits cycles
//   tool_roundtrip         manifest `tool_roundtrip: true`
//   auto_path_determinism  manifest `auto_path_determinism: true`
//   cooperative_cancel     manifest `cooperative_cancel: true`
//
// The PARIS CASE ITSELF is parameterised, because "the same case" was never the
// same probe: `selfcheck_frame_prompt` (raw BPE vs BuildPromptInputIds),
// `selfcheck_max_gen` (2 for most families, up to 256) and `selfcheck_bit_exact`
// (kOff vs kAuto) were three different constants in three different files, and all
// three move the recorded logits_hash. See docs/OPEN_TASKS.md §G5 — a uniform
// driver re-baselined every family's hash and turned two families' gates red.
//
// `cooperative_cancel` exists because THIS REFACTOR DROPPED IT: eight per-family
// selfchecks called ExpectCooperativeCancelStops (and, for v4.1, the streaming
// order assert) and the single driver reproduced only the Paris case, so the
// cancel gate silently stopped running for eight models while every target stayed
// green. Found 2026-11-07 by auditing docs/OPEN_TASKS.md §A2, whose fix had been
// blocked on a failure that could no longer even be observed.
//
// A family case that is not one of those (qwen_moe_exp's EOS/KV-growth gate)
// keeps its own file and its own target - opting out needs no manifest change.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/tokenizer.h>

#include "test_manifest_query.h"
#include "test_manifest_query_root.h"
#include "test_streaming_assert.h"
#include "test_tool_roundtrip.h"

#ifndef LSX_SELFCHECK_MODEL
#error "LSX_SELFCHECK_MODEL must be defined (the model_type of one manifest row)"
#endif

namespace {

constexpr std::string_view kModel = LSX_SELFCHECK_MODEL;

// >= 3 cycles, overridable per model (the old files each read their own env).
int Cycles(const lsxmanifest::ModelRow &row)
{
    const std::string env = std::string("LSX_TEST_") +
                            [&] {
                                std::string s(row.model_type);
                                for (char &c : s)
                                    if (c == '.' || c == '-') c = '_';
                                for (char &c : s)
                                    c = static_cast<char>(::toupper(c));
                                return s;
                            }() +
                            "_SELFCHECK_CYCLES";
    const char *v = std::getenv(env.c_str());
    return std::max(3, v ? std::atoi(v) : 3);
}

std::unique_ptr<lsxcommon::IModel> Load(const lsxmanifest::ModelRow &row,
                                        const std::filesystem::path &root,
                                        lsxcommon::FastMathMode mode)
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = row.model_type;
    cfg.fast_math_mode = mode;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load())
        FAIL(std::string(row.model_type) + " Load failed");
    return model;
}

void RequireVault(const lsxmanifest::ModelRow &row,
                  const std::filesystem::path &root)
{
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP(manifest_query::SkipMessage(row, "converted model"));
}

// The math mode the family's own selfcheck ran in.
//
// Fourteen per-family files pinned `FastMathMode::kOff`; the other twenty-one
// left `ModelInitConfig` at its kAuto default. That is not a detail either: the
// mode is what makes `final_logits` reproducible, and the recorded logits_hash in
// docs/SELFCHECK_MATRIX.md was captured under the family's own setting. Forcing
// kOff everywhere kept the top-1 token identical and changed the hash for every
// kAuto family, which would have re-baselined the whole matrix. Transcribed as
// `selfcheck_bit_exact`, defaulting to true (the gate's preference) for models
// with no predecessor file.
lsxcommon::FastMathMode SelfCheckMathMode(const lsxmanifest::ModelRow &row)
{
    return row.selfcheck_bit_exact ? lsxcommon::FastMathMode::kOff
                                   : lsxcommon::FastMathMode::kAuto;
}

// The canonical SelfCheck probe's ids.
//
// `arrow_tokenizer_encode` is PURE BPE — the raw sentence, no chat framing — and
// that is what eighteen per-family selfchecks used before the shared driver
// existed. The driver first framed with BuildPromptInputIds for all models, which
// silently changed the probe for every family WITH a chat template: phi-4 then
// answered "The capital of France is Paris" (it continues the fragment) instead of
// "Paris", and its gate went red for a reason nothing recorded. Framing is
// therefore DECLARED per model (`selfcheck_frame_prompt`) — seventeen families
// genuinely need it, and the declaration is transcribed from the deleted files,
// not chosen.
std::vector<int32_t> BuildSelfCheckIds(const lsxmanifest::ModelRow &row,
                                       lsxcommon::IModel &model)
{
    const char *kPrompt = "The capital of France is";
    std::vector<int32_t> ids;
    if (row.selfcheck_frame_prompt) {
        REQUIRE(model.BuildPromptInputIds(model.Tokenizer(), kPrompt, ids));
        return ids;
    }
    ids.assign(std::strlen(kPrompt) + 64, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        model.Tokenizer(), kPrompt, std::strlen(kPrompt), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("selfcheck answers Paris with bit-exact cycles", "[selfcheck][inference]")
{
    const lsxmanifest::ModelRow *row = manifest_query::FindModel(kModel);
    REQUIRE(row != nullptr);
    const std::filesystem::path root = manifest_query::ModelRoot(*row);
    RequireVault(*row, root);

    lsxcommon::EnsureModelRegistry();
    auto model = Load(*row, root, SelfCheckMathMode(*row));

    std::vector<int32_t> ids = BuildSelfCheckIds(*row, *model);
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    // The family's own budget, transcribed from the per-family file this driver
    // replaced (2 for most; 16/24/32/64/192/256 for a handful). It is not
    // cosmetic: final_logits is the LAST decode step's logits, so the recorded
    // logits_hash — the bit-exactness evidence in docs/SELFCHECK_MATRIX.md — is a
    // function of this number. A uniform 256 silently re-baselined every family's
    // hash and made the matrix unreproducible.
    req.max_gen_tokens = row->selfcheck_max_gen;
    // No raw `Infer` here. Twelve per-family files additionally asserted the
    // greedy continuation ("The capital of France is" -> "Paris") and twelve did
    // not; enforcing it everywhere was a requirement the gate never had, and for a
    // FRAMED family it is not even the same probe — ornith, given the framed ids
    // and a 2-token budget, answers "The capital" and is correct to do so.
    // The canonical contract is `SelfCheck`: for the non-framed families it is
    // exactly that strict first-token comparison, and for the framed ones it is
    // the declared answer-window variant. `selfcheck_passed` is the assertion.
    const int cycles = Cycles(*row);
    const auto sc0 = model->SelfCheck(req);
    INFO("selfcheck error: '" << sc0.error_msg << "'");
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    const std::vector<float> &ref_logits = sc0.final_logits;
    for (int c = 1; c < cycles; ++c)
    {
        const auto sc = model->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        REQUIRE(sc.final_logits.size() == ref_logits.size());
        REQUIRE(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                            ref_logits.size() * sizeof(float)) == 0);
        REQUIRE(sc.first_gen_tok_id == sc0.first_gen_tok_id);
    }
    model->Unload();
}

TEST_CASE("selfcheck tool round-trip on the converted weights",
          "[selfcheck][tool][real-model]")
{
    const lsxmanifest::ModelRow *row = manifest_query::FindModel(kModel);
    REQUIRE(row != nullptr);
    if (!row->tool_roundtrip)
    {
        SUCCEED("tool_roundtrip hook not declared for " << row->model_type);
        return;
    }
    const std::filesystem::path root = manifest_query::ModelRoot(*row);
    RequireVault(*row, root);

    lsxcommon::EnsureModelRegistry();
    auto model = Load(*row, root, SelfCheckMathMode(*row));

    std::vector<int32_t> gen_tokens;
    const auto calls = RunToolRoundTrip(*model, 512, &gen_tokens);
    REQUIRE(!calls.empty());
    CHECK_FALSE(calls[0].name.empty());
    model->Unload();
}

TEST_CASE("selfcheck is deterministic on the auto (ids-streaming) path",
          "[selfcheck][inference]")
{
    const lsxmanifest::ModelRow *row = manifest_query::FindModel(kModel);
    REQUIRE(row != nullptr);
    if (!row->auto_path_determinism)
    {
        SUCCEED("auto_path_determinism hook not declared for " << row->model_type);
        return;
    }
    const std::filesystem::path root = manifest_query::ModelRoot(*row);
    RequireVault(*row, root);

    lsxcommon::EnsureModelRegistry();
    auto model = Load(*row, root, lsxcommon::FastMathMode::kAuto);

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "The capital of France is", ids));
    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 8;
    const auto a = model->Infer(req);
    INFO(a.error_msg);
    REQUIRE(a.ok);
    REQUIRE_FALSE(a.gen_tokens.empty());
    model->Unload();
    REQUIRE(model->Load());
    const auto b = model->Infer(req);
    INFO(b.error_msg);
    REQUIRE(b.ok);
    CHECK(b.gen_tokens == a.gen_tokens);
    REQUIRE(b.final_logits.size() == a.final_logits.size());
    CHECK(std::memcmp(b.final_logits.data(), a.final_logits.data(),
                      a.final_logits.size() * sizeof(float)) == 0);
    model->Unload();
}

TEST_CASE("selfcheck streams and stops on a cooperative cancel",
          "[selfcheck][streaming][real-model]")
{
    // OPEN_TASKS §E1a: a cancelled request must not run to budget, and the token
    // callback must observe exactly the produced tokens in order. Both are
    // declared per model because a family that polls the flag only before the
    // first token passes every other test.
    const lsxmanifest::ModelRow *row = manifest_query::FindModel(kModel);
    REQUIRE(row != nullptr);
    if (!row->cooperative_cancel)
    {
        SUCCEED("cooperative_cancel hook not declared for " << row->model_type);
        return;
    }
    const std::filesystem::path root = manifest_query::ModelRoot(*row);
    RequireVault(*row, root);

    lsxcommon::EnsureModelRegistry();
    auto model = Load(*row, root, SelfCheckMathMode(*row));

    std::vector<int32_t> ids = BuildSelfCheckIds(*row, *model);
    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = row->selfcheck_max_gen;
    ExpectStreamIsTokensWithoutTrailingEos(*model, req);
    ExpectCooperativeCancelStops(*model, req);
    model->Unload();
}

