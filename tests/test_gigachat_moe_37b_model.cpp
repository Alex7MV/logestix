// TDD — GigaChat4-37B-A4B model contour. Synthetic converted fixture only; the
// frozen geometry is bypassed via LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 (set by
// the ctest environment). Exercises the full 2-layer hybrid decode path with
// the DSA indexer, the non-zero-centred norms and the logit soft cap.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat_37b_fixture.h"
#include "test_streaming_assert.h"

#include "lsxcommon/gigachat_moe_37b_convert.h"
#include "lsxcommon/gigachat_moe_37b_model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"

#include <arrow/io/file.h>

#include <cmath>
#include <filesystem>

using namespace lsxcommon;

namespace {

std::filesystem::path BuildConverted(const std::string &tag)
{
    const auto root = gigachat4test::MakeTempDir(tag);
    gigachat4test::WriteGigaChat4Checkpoint(root, /*verified=*/true);
    auto st = gigachat4::ConvertGigaChat4Model(root.string(), root.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());
    return root;
}

} // namespace

TEST_CASE("gigachat4 model decodes a synthetic converted fixture",
          "[gigachat4][model]")
{
    const auto root = BuildConverted("gc4_model");

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat4MoeModel m(cfg);
    REQUIRE(m.Load());
    CHECK(m.IndexForTest().num_layers == 2);
    CHECK(m.IndexForTest().index_topk == gigachat4test::kTIndexTopk);

    ModelRequest req;
    req.input_ids = {1, 2, 3};
    req.max_gen_tokens = 3;
    auto r = m.Infer(req);
    INFO("infer error: " << r.error_msg);
    REQUIRE(r.ok);
    REQUIRE(r.gen_tokens.size() == 3);
    REQUIRE(r.final_logits.size() == static_cast<size_t>(gigachat4test::kTVocab));
    for (float f : r.final_logits) CHECK(std::isfinite(f));
    // final_logit_softcapping = 30: the recorded logits are post-cap.
    for (float f : r.final_logits) CHECK(std::abs(f) <= 30.0f + 1e-3f);
    CHECK(r.weight_resident_bytes > 0);

    // Deterministic re-run (same weights, caches reset between calls).
    auto r2 = m.Infer(req);
    REQUIRE(r2.ok);
    CHECK(r2.gen_tokens == r.gen_tokens);
    CHECK(r2.final_logits == r.final_logits);

    // One Emit per decoded token, in order.
    ExpectStreamMatchesTokens(m, req);
    m.Unload();
}

TEST_CASE("gigachat4 honours cooperative cancel", "[gigachat4][model][lifecycle]")
{
    const auto root = BuildConverted("gc4_cancel");

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat4MoeModel m(cfg);
    REQUIRE(m.Load());

    ModelRequest req;
    req.input_ids = {1, 2};
    ExpectCooperativeCancelStops(m, req);

    m.Unload();
}

TEST_CASE("gigachat4 rejects a prompt longer than the session capacity",
          "[gigachat4][model]")
{
    const auto root = BuildConverted("gc4_ctx");

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat4MoeModel m(cfg);
    REQUIRE(m.Load());

    ModelRequest req;
    req.input_ids.assign(4200, 1);
    req.max_gen_tokens = 1;
    auto r = m.Infer(req);
    CHECK_FALSE(r.ok);
    CHECK(r.error_msg.find("context length exceeded") != std::string::npos);
    m.Unload();
}

TEST_CASE("gigachat4 chatml-v5 framing uses the right sentinels",
          "[gigachat4][model]")
{
    const auto pinned = gigachat4test::MakeTempDir("gc4_tok");
    gigachattest::WriteTokenizerArrows(pinned);

    ArrowTokenizerDb db;
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto mg = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    REQUIRE(mg.ok());
    auto st = load_tokenizer_metadata(*v, *mg, db);
    INFO("load_tokenizer_metadata: " << st.ToString());
    REQUIRE(st.ok());

    ModelInitConfig cfg;
    cfg.model_type = GigaChat4MoeModel::kType;
    GigaChat4MoeModel model(cfg);
    std::vector<int32_t> ids;
    REQUIRE(model.BuildPromptInputIds(db, "The capital of France is", ids));
    REQUIRE(ids.size() >= 2);
    CHECK(ids.front() == db.bos_id);
    CHECK(ids[ids.size() - 2] == 128014);  // <think>
    CHECK(ids.back() == 128015);           // </think>
}

TEST_CASE("gigachat4 factory resolves the kType", "[gigachat4][model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = GigaChat4MoeModel::kType;
    cfg.model_path = "/nonexistent";
    auto m = LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    CHECK(m->Type() == "gigachat4-37b-a4b");
}
