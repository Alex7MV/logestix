// TDD — GigaChat3.5 model contour. Synthetic converted fixture only; the
// frozen geometry is bypassed via LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 (set by
// the ctest environment). Exercises the full 2-layer hybrid decode path.
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_moe_fixture.h"
#include "test_streaming_assert.h"

#include "lsxcommon/gigachat_moe_convert.h"
#include "lsxcommon/gigachat_moe_model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"

#include <arrow/io/file.h>

#include <cmath>

using namespace lsxcommon;

TEST_CASE("gigachat model loads a synthetic converted fixture",
          "[gigachat][model]")
{
    const auto root = gigachattest::MakeTempDir("gc_model");
    gigachattest::WriteGigaChatCheckpoint(root);
    auto st = gigachat::ConvertGigaChatModel(root.string(), root.string());
    REQUIRE(st.ok());

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat35Model m(cfg);
    REQUIRE(m.Load());
    CHECK(m.IndexForTest().num_layers == 2);
    CHECK(m.Inspect().shards.size() == 1);

    ModelRequest req;
    req.input_ids = {1};
    req.max_gen_tokens = 1;
    auto r = m.Infer(req);
    INFO("infer error: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(r.gen_tokens.size() == 1);
    REQUIRE(r.final_logits.size() == 128);
    for (float f : r.final_logits) CHECK(std::isfinite(f));

    // Deterministic re-run (same weights, same caches reset).
    auto r2 = m.Infer(req);
    REQUIRE(r2.ok);
    CHECK(r2.gen_tokens == r.gen_tokens);
    ExpectStreamMatchesTokens(m, req);
    m.Unload();
}

// Cooperative cancel (OPEN_TASKS §E1a): the decode loop runs host-side (one
// Infer per step), so it must poll the abort flag there and stop a cancelled
// request instead of publishing the full budget.
TEST_CASE("gigachat honours cooperative cancel", "[gigachat][model][lifecycle]")
{
    const auto root = gigachattest::MakeTempDir("gc_cancel");
    gigachattest::WriteGigaChatCheckpoint(root);
    REQUIRE(gigachat::ConvertGigaChatModel(root.string(), root.string()).ok());

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat35Model m(cfg);
    REQUIRE(m.Load());

    ModelRequest req;
    req.input_ids = {1};
    ExpectCooperativeCancelStops(m, req);

    m.Unload();
}

TEST_CASE("gigachat chatml-v5 framing uses the right sentinels",
          "[gigachat][model]")
{
    const auto pinned = gigachattest::MakeTempDir("gc_tok");
    gigachattest::WriteTokenizerArrows(pinned);

    ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(m.ok());
    auto st = load_tokenizer_metadata(*v, *m, db);
    INFO("load_tokenizer_metadata: " << st.ToString());
    REQUIRE(st.ok());
    CHECK(db.bos_id == 1);
    CHECK(db.eos_id == 2);
    CHECK(arrow_tokenizer_lookup(db, "<think>", 7) == 128014);
    CHECK(arrow_tokenizer_lookup(db, "</think>", 8) == 128015);

    ModelInitConfig cfg;
    cfg.model_type = GigaChat35Model::kType;
    GigaChat35Model model(cfg);
    std::vector<int32_t> ids;
    REQUIRE(model.BuildPromptInputIds(db, "The capital of France is", ids));
    REQUIRE(ids.size() >= 2);
    CHECK(ids.front() == db.bos_id);
    CHECK(ids[ids.size() - 2] == 128014);  // <think>
    CHECK(ids.back() == 128015);           // </think>
}

TEST_CASE("gigachat factory resolves the kType", "[gigachat][model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = GigaChat35Model::kType;
    cfg.model_path = "/nonexistent";
    auto m = LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "gigachat3.5-432b-a28b-reasoning");
}
