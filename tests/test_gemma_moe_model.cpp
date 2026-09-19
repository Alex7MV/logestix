// Gemma-4 model test: factory + Load + Infer on synthetic geometry.
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>
#include <cstring>

#include "test_gemma_moe_fixture.h"

#include <filesystem>

#include "lsxcommon/gemma_moe_convert.h"
#include "lsxcommon/gemma_moe_load.h"
#include "lsxcommon/model_factory.h"

#include "gemma_base_model.h"

namespace fs = std::filesystem;

namespace {

fs::path MakeConvertedSynthetic()
{
    auto tmp = fs::temp_directory_path() / "gemma4_model_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto tensors = gemmatest::BuildSyntheticGemmaModel();
    gemmatest::WriteSafetensors(tmp, "model.safetensors", tensors);
    gemmatest::WriteConfig(tmp);
    std::filesystem::copy_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
                               tmp / "tokenizer.json");
    auto out = tmp / "out";
    REQUIRE(lsxcommon::gemma::ConvertGemmaModel(tmp.string(), out.string())
                .ok());
    return out;
}

} // namespace

TEST_CASE("factory creates registered gemma model on synthetic fixture",
          "[gemma][model]") {
    auto out = MakeConvertedSynthetic();

    // Direct loader probe: isolate weight-load failure from factory plumbing.
    {
        lsxcommon::gemma::GemmaPinnedWeights probe;
        auto rc = lsxcommon::gemma::LoadGemmaPinnedWeights(out.string(), probe,
                                                           nullptr);
        if (rc != cudaSuccess) {
            INFO("direct LoadGemmaPinnedWeights rc="
                 << cudaGetErrorString(rc));
            REQUIRE(rc == cudaSuccess);
        }
        probe.Free();
    }

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-26b-a4b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Type() == "gemma-4-26b-a4b-it");
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 3, 5, 7}; // synthetic vocab ids
    req.max_gen_tokens = 2;
    auto r = model->Infer(req);
    if (!r.ok) FAIL("Infer failed: " << r.error_msg);
    CHECK(r.gen_tokens.size() >= 1);
    CHECK(r.gen_tokens.size() <= 2);

    // Determinism: second identical run must be bit-exact.
    auto r2 = model->Infer(req);
    if (!r2.ok) FAIL("second Infer failed: " << r2.error_msg);
    CHECK(r2.gen_tokens == r.gen_tokens);
    REQUIRE(r.final_logits.size() == r2.final_logits.size());
    CHECK(std::memcmp(r.final_logits.data(), r2.final_logits.data(),
                      r.final_logits.size() * sizeof(float)) == 0);

    model->Unload();
    fs::remove_all(out.parent_path());
}


TEST_CASE("gemma expert index parsed once across Infers", "[gemma][model]") {
    auto out = MakeConvertedSynthetic();

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = out.string();
    cfg.model_type = "gemma-4-26b-a4b-it";
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 3, 5, 7};
    req.max_gen_tokens = 2;

    auto r1 = model->Infer(req);
    if (!r1.ok) FAIL("first Infer failed: " << r1.error_msg);
    auto r2 = model->Infer(req);
    if (!r2.ok) FAIL("second Infer failed: " << r2.error_msg);

    auto *gm = dynamic_cast<lsxcommon::GemmaBaseModel *>(model.get());
    REQUIRE(gm != nullptr);
    const auto &idx = gm->ExpertIndexForTest();
    CHECK(idx.parse_calls == idx.num_layers * idx.experts_per_layer);
    CHECK(idx.parse_ok == idx.num_layers * idx.experts_per_layer);

    model->Unload();
    fs::remove_all(out.parent_path());
}
