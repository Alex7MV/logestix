// TDD — GigaChat4-708B-A41B model contour. Synthetic converted fixture only;
// the frozen geometry is bypassed via LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY=1 (set
// by the ctest environment). Exercises the full hybrid decode path through the
// member's own entry point: GDN + DSA MLA + MoE, the non-zero-centred norms, the
// logit soft cap, streaming and cooperative cancel.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "test_gigachat4_fixture.h"
#include "test_streaming_assert.h"

#include "lsxcommon/gigachat_moe_708b_convert.h"
#include "lsxcommon/gigachat_moe_708b_model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"

#include <arrow/io/file.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>

using namespace lsxcommon;
namespace gc708 = lsxcommon::gigachat4_708b;

namespace {

std::filesystem::path BuildConverted(const std::string &tag)
{
    const auto root = gigachat4test::MakeTempDir(tag);
    gigachat4test::WriteGigaChat4Checkpoint(root, /*verified=*/true);
    auto st = gc708::ConvertGigaChat4_708BModel(root.string(), root.string());
    INFO("convert status: " << st.ToString());
    REQUIRE(st.ok());
    return root;
}

void SetEnv(const char *k, const char *v)
{
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

} // namespace

TEST_CASE("gigachat4-708b model decodes a synthetic converted fixture",
          "[gigachat708b][model]")
{
    const auto root = BuildConverted("gc708_model");

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat4Moe708BModel m(cfg);
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

TEST_CASE("gigachat4-708b honours cooperative cancel",
          "[gigachat708b][model][lifecycle]")
{
    const auto root = BuildConverted("gc708_cancel");

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat4Moe708BModel m(cfg);
    REQUIRE(m.Load());

    ModelRequest req;
    req.input_ids = {1, 2};
    ExpectCooperativeCancelStops(m, req);

    m.Unload();
}

TEST_CASE("gigachat4-708b rejects a prompt longer than the session capacity",
          "[gigachat708b][model]")
{
    const auto root = BuildConverted("gc708_ctx");

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    GigaChat4Moe708BModel m(cfg);
    REQUIRE(m.Load());

    ModelRequest req;
    // The session cap is 4096: prompt + generation must fit inside it, or the
    // KV allocation and the prompt disagree about the sequence length.
    req.input_ids.assign(4200, 1);
    req.max_gen_tokens = 1;
    auto r = m.Infer(req);
    CHECK_FALSE(r.ok);
    CHECK(r.error_msg.find("context length exceeded") != std::string::npos);

    // An exactly-fitting request must NOT be rejected by the same check.
    ModelRequest ok_req;
    ok_req.input_ids.assign(4095, 1);
    ok_req.max_gen_tokens = 1;
    auto r2 = m.Infer(ok_req);
    INFO("fit error: " << r2.error_msg);
    CHECK(r2.error_msg.find("context length exceeded") == std::string::npos);
    m.Unload();
}

TEST_CASE("gigachat4-708b is registered under its own type",
          "[gigachat708b][model]")
{
    CHECK(GigaChat4Moe708BModel::kType == "gigachat4-708b-a41b");
    // The factory resolves the member and rejects the sibling's type as an
    // alias of this row (they are different models with different geometry).
    ModelInitConfig cfg;
    cfg.model_type = GigaChat4Moe708BModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == GigaChat4Moe708BModel::kType);
}

TEST_CASE("gigachat4-708b chatml-v5 framing uses the right sentinels",
          "[gigachat708b][model]")
{
    const auto pinned = gigachat4test::MakeTempDir("gc708_tok");
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
    cfg.model_type = GigaChat4Moe708BModel::kType;
    GigaChat4Moe708BModel model(cfg);
    std::vector<int32_t> ids;
    REQUIRE(model.BuildPromptInputIds(db, "The capital of France is", ids));
    REQUIRE(ids.size() >= 2);
    CHECK(ids.front() == db.bos_id);
    CHECK(ids[ids.size() - 2] == 128014);  // <think>
    CHECK(ids.back() == 128015);           // </think>
}
