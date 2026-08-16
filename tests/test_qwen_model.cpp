#include <catch2/catch_all.hpp>

#include <lsxcommon/model_factory.h>
#include <lsxcommon/qwen_convert.h>
#include <lsxcommon/qwen_inference_load.h>

#include <spdlog/spdlog.h>

#include "qwen_test_fixture.h"

#include <filesystem>
#include <thread>

using namespace std::filesystem;

TEST_CASE("Qwen V5 model factory registration and generation")
{
    auto tmp = temp_directory_path() / "qwen_model_test";
    remove_all(tmp);
    create_directories(tmp);

    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(tmp, "model.safetensors", tensors);
    qwentest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        tmp / "tokenizer.json");
    REQUIRE(lsxcommon::qwen::ConvertQwenModel(tmp.string(), tmp.string()).ok());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == qwentest::TestCfg().model_name);

    bool loaded = model->Load();
    CHECK(loaded);

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 1, 2, 3};
    req.max_gen_tokens = 5;

    auto result = model->Infer(req);
    spdlog::info("[qwen_test] Infer error_msg='{}'", result.error_msg);
    REQUIRE(result.ok);
    CHECK(result.num_tokens > 0);
    CHECK(result.first_gen_tok_id >= 0);
    CHECK(result.num_layers == 4);
    CHECK(result.total_time_ms > 0.0);
    CHECK(result.tokens_per_sec > 0.0);
    // Regression: Infer must populate first-gen decode text through the same
    // tokenizer round-trip the self-check uses (synthetic special ids may
    // decode empty, so only assert visibility when a plain token is produced).
    if (result.first_gen_tok_id >= 0) {
        char tb[64];
        size_t tc = sizeof(tb);
        if (lsxcommon::arrow_tokenizer_decode(model->Tokenizer(),
                                              &result.first_gen_tok_id, 1,
                                              tb, &tc) == 0) {
            std::string d(tb, tc);
            CHECK(result.first_gen_text == d);
        }
    }

    auto sc = model->SelfCheck(req);
    CHECK(sc.ok);
    CHECK(sc.error_msg.empty());

    model->Unload();
    remove_all(tmp);
}