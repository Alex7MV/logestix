#include <catch2/catch_all.hpp>

#include <lsxcommon/granite_dense_convert.h>
#include <lsxcommon/model_factory.h>

#include "test_granite_dense_fixture.h"
#include "test_streaming_assert.h"

#include <filesystem>
#include <memory>

using namespace lsxcommon;

namespace {

std::filesystem::path BuildConvertedFixture()
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_model_test";
    remove_all(tmp);
    create_directories(tmp);
    grantest::WriteSafetensors(tmp, "model.safetensors",
                               grantest::BuildSyntheticGraniteModel());
    grantest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    auto st = granite::ConvertGraniteModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());
    return tmp;
}

} // namespace

TEST_CASE("granite model type autodetects from the converted index",
          "[granite][model]")
{
    auto tmp = BuildConvertedFixture();
    CHECK(LsxModelFactory::DetectType(tmp.string()) == "granite-4.2-30b");
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite model loads, infers and is deterministic",
          "[granite][model]")
{
    auto tmp = BuildConvertedFixture();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "granite-4.2-30b";

    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "granite-4.2-30b");
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    req.max_gen_tokens = 3;

    auto r1 = model->Infer(req);
    REQUIRE(r1.ok);
    CHECK(!r1.gen_tokens.empty());
    CHECK(r1.num_layers == grantest::TestCfg().num_layers);

    // Cross-instance determinism: a fresh instance must produce bit-identical
    // tokens and final logits for the same request.
    auto model2 = LsxModelFactory::Create(cfg);
    REQUIRE(model2 != nullptr);
    REQUIRE(model2->Load());
    auto r2 = model2->Infer(req);
    REQUIRE(r2.ok);
    CHECK(r1.gen_tokens == r2.gen_tokens);
    REQUIRE(r1.final_logits.size() == r2.final_logits.size());
    for (size_t i = 0; i < r1.final_logits.size(); ++i)
        REQUIRE(r1.final_logits[i] == r2.final_logits[i]);

    ExpectStreamMatchesTokens(*model, req);

    model2->Unload();
    model->Unload();
    std::filesystem::remove_all(tmp);
}

// Cooperative cancel (OPEN_TASKS §E1a): a raised abort_flag must stop the
// granite decode loop, which is what `lsxfabric --serve` cancels against.
TEST_CASE("granite honours cooperative cancel", "[granite][model][lifecycle]")
{
    auto tmp = BuildConvertedFixture();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "granite-4.2-30b";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    ExpectCooperativeCancelStops(*model, req);

    model->Unload();
    std::filesystem::remove_all(tmp);
}

// Audit X2: a granite continuation must append to the existing cache, not
// prefill the tail at position 0 and reset kv_rows. Compare against a fresh
// prefill of prompt + generated tokens.
TEST_CASE("granite continuation matches a fresh prefill", "[granite][model]")
{
    auto tmp = BuildConvertedFixture();
    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    // Fixture: this tree was built by the family converter into a temporary
    // directory, so it predates artifact.fb. The storage resolve and the codec
    // gate are skipped explicitly; the gate itself is covered by
    // tests/test_load_codec_gate.cpp.
    cfg.synthetic = true;
    cfg.model_type = "granite-4.2-30b";

    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    req.max_gen_tokens = 3;
    req.ignore_eos = true;
    const auto r1 = model->Infer(req);
    REQUIRE(r1.ok);
    REQUIRE(r1.gen_tokens.size() == 3);

    // Granite's session_tail_ is the generated tokens.
    ModelRequest cont;
    cont.input_ids = r1.gen_tokens;
    cont.max_gen_tokens = 3;
    cont.ignore_eos = true;
    const auto r2 = model->Infer(cont);
    REQUIRE(r2.ok);
    model->Unload();

    auto fresh = LsxModelFactory::Create(cfg);
    REQUIRE(fresh != nullptr);
    REQUIRE(fresh->Load());
    ModelRequest same;
    same.input_ids = req.input_ids;
    same.input_ids.insert(same.input_ids.end(), r1.gen_tokens.begin(),
                          r1.gen_tokens.end());
    same.max_gen_tokens = 3;
    same.ignore_eos = true;
    const auto r3 = fresh->Infer(same);
    REQUIRE(r3.ok);
    fresh->Unload();

    CHECK(r2.gen_tokens == r3.gen_tokens);
    REQUIRE(r2.final_logits.size() == r3.final_logits.size());
    double max_abs = 0.0;
    for (size_t i = 0; i < r2.final_logits.size(); ++i)
        max_abs = std::max(max_abs,
                           (double)std::abs(r2.final_logits[i] -
                                            r3.final_logits[i]));
    INFO("max logit delta = " << max_abs);
    CHECK(max_abs < 1e-3);
    // Structural oracle: a continuation must APPEND to the cache, so the
    // slot count matches a fresh prefill of the same context. The old path
    // prefilled the tail at pos 0 and reset kv_rows to 1.
    CHECK(r2.kv_slots_used == r3.kv_slots_used);

    std::filesystem::remove_all(tmp);
}
