#include <catch2/catch_all.hpp>

#include <lsxcommon/granite_dense_switch_convert.h>
#include <lsxcommon/granite_dense_switch_inference.h>
#include <lsxcommon/granite_dense_switch_load.h>
#include <lsxcommon/model_factory.h>

#include "test_granite_switch_fixture.h"
#include "test_streaming_assert.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <memory>

using namespace lsxcommon;

namespace {

std::filesystem::path BuildConvertedFixture()
{
    using namespace std::filesystem;
    auto tmp = temp_directory_path() / "granite_switch_model_test";
    remove_all(tmp);
    create_directories(tmp);
    gswtest::BuildConverted(tmp);
    copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
              tmp / "tokenizer.json");
    REQUIRE(lsxcommon::granite_switch::ConvertGraniteSwitchModel(
                tmp.string(), tmp.string())
                .ok());
    return tmp;
}

} // namespace

TEST_CASE("granite-switch model autodetects from the converted index",
          "[granite_switch][model]")
{
    auto tmp = BuildConvertedFixture();
    CHECK(LsxModelFactory::DetectType(tmp.string()) ==
          "granite-switch-4.1-30b");
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite-switch generation tracks the active adapter",
          "[granite_switch][model][inference]")
{
    auto tmp = BuildConvertedFixture();
    granite_switch::GraniteSwitchPinnedWeights pw;
    REQUIRE(granite_switch::LoadGraniteSwitchPinnedWeights(tmp.string(), pw,
                                                           0));
    granite_switch::GraniteSwitchKvCache kv;
    granite_switch::GraniteSwitchGenerationResult r;
    auto &sampler = sample::DefaultSampler();
    REQUIRE(granite_switch::RunGraniteSwitchGeneration(
                0, pw, {5, 64, 6, 7}, 3, -1, r, sampler, &kv) == cudaSuccess);
    REQUIRE(r.ok);
    CHECK(r.last_adapter == 1);
    CHECK(kv.last_adapter == 1);
    CHECK(r.tokens.size() == 3);

    INFO("kv after prefill rows=" << kv.kv_rows << " adapter=" << kv.last_adapter << " last=" << kv.last_tok);
    granite_switch::GraniteSwitchGenerationResult c;
    REQUIRE(granite_switch::RunGraniteSwitchGeneration(
                0, pw, {kv.last_tok}, 2, -1, c, sampler, &kv) == cudaSuccess);
    REQUIRE(c.ok);
    INFO("kv after cont rows=" << kv.kv_rows << " adapter=" << kv.last_adapter << " c.last=" << c.last_adapter);
    CHECK(c.last_adapter == 1);

    // A decision control token switches the route mid-session.
    granite_switch::GraniteSwitchGenerationResult d;
    REQUIRE(granite_switch::RunGraniteSwitchGeneration(
                0, pw, {65, 8}, 2, -1, d, sampler, nullptr) == cudaSuccess);
    REQUIRE(d.ok);
    CHECK(d.last_adapter == 2);

    pw.Free();
    kv.Free();
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite-switch model loads, infers and is deterministic",
          "[granite_switch][model]")
{
    auto tmp = BuildConvertedFixture();

    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    cfg.model_type = "granite-switch-4.1-30b";

    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "granite-switch-4.1-30b");
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    req.max_gen_tokens = 3;

    auto r1 = model->Infer(req);
    REQUIRE(r1.ok);
    CHECK(!r1.gen_tokens.empty());
    CHECK(r1.num_layers == gswtest::Cfg().num_layers);

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

TEST_CASE("granite-switch honours cooperative cancel",
          "[granite_switch][model][lifecycle]")
{
    auto tmp = BuildConvertedFixture();
    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    cfg.model_type = "granite-switch-4.1-30b";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 11, 12, 13, 14};
    ExpectCooperativeCancelStops(*model, req);

    model->Unload();
    std::filesystem::remove_all(tmp);
}

TEST_CASE("granite-switch continuation matches a fresh prefill",
          "[granite_switch][model]")
{
    auto tmp = BuildConvertedFixture();
    ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    cfg.model_type = "granite-switch-4.1-30b";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.input_ids = {10, 64, 11, 12, 13};
    req.max_gen_tokens = 3;
    auto first = model->Infer(req);
    REQUIRE(first.ok);

    ModelRequest cont;
    cont.input_ids = first.gen_tokens;
    cont.max_gen_tokens = 3;
    auto a = model->Infer(cont);
    REQUIRE(a.ok);

    // Fresh prefill of the concatenated prompt must reproduce the same tokens.
    ModelRequest fresh;
    fresh.input_ids = req.input_ids;
    fresh.input_ids.insert(fresh.input_ids.end(), first.gen_tokens.begin(),
                           first.gen_tokens.end());
    fresh.max_gen_tokens = 3;
    auto model2 = LsxModelFactory::Create(cfg);
    REQUIRE(model2 != nullptr);
    REQUIRE(model2->Load());
    auto b = model2->Infer(fresh);
    REQUIRE(b.ok);
    CHECK(a.gen_tokens == b.gen_tokens);

    model2->Unload();
    model->Unload();
    std::filesystem::remove_all(tmp);
}
