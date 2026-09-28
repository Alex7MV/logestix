#include <catch2/catch_all.hpp>

#include <lsxcommon/model_factory.h>
#include <lsxcommon/qwen_dense_convert.h>
#include <lsxcommon/qwen_inference_load.h>
#include <lsxcommon/tokenizer.h>

#include <spdlog/spdlog.h>

#include "test_streaming_assert.h"
#include "test_qwen_dense_fixture.h"

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

    // Unified runtime contract (Task 7): token parity, logits, decode.
    REQUIRE(result.num_tokens == static_cast<int>(result.gen_tokens.size()));
    REQUIRE_FALSE(result.final_logits.empty());
    if (!result.gen_tokens.empty()) {
        std::vector<char> buf(256, '\0');
        size_t cap = buf.size();
        int dr = arrow_tokenizer_decode(model->Tokenizer(),
                                        result.gen_tokens.data(),
                                        result.gen_tokens.size(),
                                        buf.data(), &cap);
        REQUIRE(dr == 0);
        REQUIRE(result.output_text == std::string(buf.data(), cap));
    }

    auto sc = model->SelfCheck(req);
    CHECK(sc.ok);
    CHECK(sc.error_msg.empty());

    ExpectStreamMatchesTokens(*model, req);

    model->Unload();
    remove_all(tmp);
}