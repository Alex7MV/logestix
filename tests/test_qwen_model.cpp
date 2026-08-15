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
    REQUIRE(lsxcommon::qwen::ConvertQwenModel(tmp.string(), tmp.string()).ok());

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = tmp.string();
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    CHECK(model->Type() == "qwen3.5-9b");

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

    // SelfCheck runs real inference, so it must report the same first token
    // that Infer produced and populate the selfcheck result consistently.
    auto sc = model->SelfCheck(req);
    CHECK(sc.ok);
    CHECK(sc.first_gen_tok_id == result.first_gen_tok_id);
    // On the synthetic model the decoded first token is not "Paris", so the
    // selfcheck is exercised but not expected to pass; it must still be a valid
    // inference run (no error) rather than a structural stub.
    CHECK(sc.error_msg.empty());

    model->Unload();
    remove_all(tmp);
}