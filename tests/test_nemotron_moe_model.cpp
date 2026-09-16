// TDD — Nemotron-3.5 model/factory stage (Model Ingestion Protocol step 3).
#include <catch2/catch_test_macros.hpp>

#include "test_nemotron_moe_fixture.h"

#include "nemotron_moe_model.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/nemotron_moe_convert.h"

#include <cstdlib>
#include <filesystem>

using namespace lsxcommon;

namespace {

void SetEnv(const char *k, const char *v)
{
#ifdef _WIN32
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

} // namespace

TEST_CASE("factory resolves nemotron-3.5-lightning-30b-a3b", "[nemotron][model]")
{
    EnsureModelRegistry();
    ModelInitConfig cfg;
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == NemotronHModel::kType);
}

TEST_CASE("nemotron Load accepts a synthetic converted model", "[nemotron][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_model_src");
    const auto dst = nemotrontest::MakeTempDir("nm_model_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    auto st = nemotron_moe::ConvertNemotronModel(src.string(), dst.string());
    REQUIRE(st.ok());

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    CHECK(model->Type() == NemotronHModel::kType);
    model->Unload();
}

TEST_CASE("nemotron Infer runs the hybrid pipeline end-to-end on a tiny model",
          "[nemotron][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "1");
    const auto src = nemotrontest::MakeTempDir("nm_infer_src");
    const auto dst = nemotrontest::MakeTempDir("nm_infer_dst");
    nemotrontest::WriteNemotronCheckpoint(src);
    REQUIRE(nemotron_moe::ConvertNemotronModel(src.string(), dst.string()).ok());

    ModelInitConfig cfg;
    cfg.model_path = dst.string();
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {1, 2, 3};
    req.max_gen_tokens = 2;
    req.ignore_eos = true;
    auto r = model->Infer(req);
    REQUIRE(r.error_msg.empty());
    REQUIRE(r.ok);
    CHECK(r.num_tokens == 2);
    CHECK(r.gen_tokens.size() == 2);
    CHECK(r.final_logits.size() == 32);
    model->Unload();
}

TEST_CASE("nemotron Load rejects a frozen-geometry mismatch", "[nemotron][model]")
{
    SetEnv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY", "0");
    const auto root = nemotrontest::MakeTempDir("nm_model_geo");
    nemotrontest::WriteNemotronIndex(root, /*hidden_override=*/4096);

    ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = NemotronHModel::kType;
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK_FALSE(model->Load());
}
