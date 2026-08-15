#include <catch2/catch_all.hpp>

#include <lsxcommon/model_factory.h>
#include <lsxcommon/qwen_convert.h>
#include <lsxcommon/qwen_inference_load.h>

#include <spdlog/spdlog.h>

#include "qwen_test_fixture.h"

#include <filesystem>

using namespace std::filesystem;

TEST_CASE("Qwen3.8-27B model factory registration and generation")
{
    auto tmp = temp_directory_path() / "qwen_27b_model_test";
    remove_all(tmp);
    create_directories(tmp);

    auto &cfg = const_cast<qwentest::QwenTestConfig&>(qwentest::TestCfg());
    cfg.model_name = "qwen3.8-27b";
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.vocab_size = 64;
    cfg.num_layers = 4;
    cfg.num_attention_heads = 2;
    cfg.num_key_value_heads = 1;
    cfg.head_dim = 32;
    cfg.linear_num_key_heads = 2;
    cfg.linear_num_value_heads = 4;
    cfg.linear_key_head_dim = 8;
    cfg.linear_value_head_dim = 8;
    cfg.linear_conv_kernel_dim = 4;
    cfg.layer_types = {
        "linear_attention", "linear_attention", "linear_attention", "full_attention"
    };

    auto tensors = qwentest::BuildSyntheticQwenModel();
    qwentest::WriteSafetensors(tmp, "model.safetensors", tensors);
    qwentest::WriteConfig(tmp);
    std::filesystem::copy_file(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json",
        tmp / "tokenizer.json");
    REQUIRE(lsxcommon::qwen::ConvertQwenModel(tmp.string(), tmp.string()).ok());

    lsxcommon::ModelInitConfig mcfg;
    mcfg.model_path = tmp.string();
    mcfg.model_type = "qwen3.8-27b";
    auto model = lsxcommon::LsxModelFactory::Create(mcfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "qwen3.8-27b");

    bool loaded = model->Load();
    CHECK(loaded);

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 1, 2, 3};
    req.max_gen_tokens = 5;

    auto result = model->Infer(req);
    spdlog::info("[qwen_27b_test] Infer error_msg='{}'", result.error_msg);
    REQUIRE(result.ok);
    CHECK(result.num_tokens > 0);
    CHECK(result.first_gen_tok_id >= 0);

    auto sc = model->SelfCheck(req);
    CHECK(sc.ok);
    CHECK(sc.error_msg.empty());

    model->Unload();
    remove_all(tmp);
}