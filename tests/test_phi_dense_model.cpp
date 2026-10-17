#include <catch2/catch_all.hpp>

#include <lsxcommon/phi_dense_convert.h>
#include <lsxcommon/model_factory.h>

#include "test_phi_dense_fixture.h"
#include "test_streaming_assert.h"

#include <filesystem>
#include <memory>

using namespace lsxcommon;

namespace {

std::filesystem::path BuildConvertedFixture()
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "phi_model_test";
    remove_all(tmp);
    create_directories(tmp);
    phitest::WriteSafetensors(tmp, "model.safetensors",
                              phitest::BuildSyntheticPhiModel());
    phitest::WriteConfig(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    auto st = phi::ConvertPhiModel(tmp.string(), tmp.string());
    REQUIRE(st.ok());
    return tmp;
}

} // namespace

TEST_CASE("phi model type autodetects from the converted index",
          "[phi][model]")
{
    auto tmp = BuildConvertedFixture();
    CHECK(LsxModelFactory::DetectType(tmp.string()) == "phi-4");
    std::filesystem::remove_all(tmp);
}

TEST_CASE("phi model loads, infers and is deterministic",
          "[phi][model]")
{
    auto tmp = BuildConvertedFixture();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    cfg.model_type = "phi-4";

    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "phi-4");
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    req.max_gen_tokens = 3;

    auto r1 = model->Infer(req);
    REQUIRE(r1.ok);
    CHECK(!r1.gen_tokens.empty());
    CHECK(r1.num_layers == phitest::TestCfg().num_layers);

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

// Cooperative cancel (OPEN_TASKS §E1a): a raised abort_flag must stop the phi
// decode loop, not run the full budget and publish it.
TEST_CASE("phi honours cooperative cancel", "[phi][model][lifecycle]")
{
    auto tmp = BuildConvertedFixture();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    cfg.model_type = "phi-4";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    ExpectCooperativeCancelStops(*model, req);

    model->Unload();
    std::filesystem::remove_all(tmp);
}
