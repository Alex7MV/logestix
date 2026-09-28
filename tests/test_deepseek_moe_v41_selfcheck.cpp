// DeepSeek-V4.1-Flash real-model selfcheck: "The capital of France is" ->
// "Paris", incremental 1->2 tokens, >=3 stress cycles with bit-exact
// final_logits. Skips (RETURN_CODE 4) without the converted model.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"
#include "test_streaming_assert.h"
#include "test_tool_roundtrip.h"

using namespace lsxcommon;

namespace fs = std::filesystem;

namespace {
std::string ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_DEEPSEEK_V41_ROOT")) return env;
    return "/mnt/llm_models/models/DeepSeek-V4.1-Flash";
}
} // namespace

TEST_CASE("deepseek-v41 flash selfcheck: Paris, 3 bit-exact cycles")
{
    const std::string root = ModelRoot();
    if (!fs::exists(root + "/logestix/model.index.json")) {
        SKIP("converted DeepSeek-V4.1-Flash not present");
    }
    ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "deepseek-v4.1-flash";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.max_gen_tokens = 2;
    int cycles = 3;
    if (const char *env = std::getenv("LSX_TEST_DEEPSEEK_V41_SELFCHECK_CYCLES"))
        cycles = std::atoi(env);
    std::vector<float> first_logits;
    for (int c = 0; c < cycles; ++c) {
        auto r = runtime::RunParisSelfCheck(
            *model, req,
            runtime::ParisSelfCheckOptions{.frame_prompt = true});
        INFO("cycle " << c << ": '" << r.error_msg << "' tokens="
                      << r.gen_tokens.size());
        REQUIRE(r.ok);
        REQUIRE(r.selfcheck_passed);
        if (c == 0) first_logits = r.final_logits;
        else REQUIRE(r.final_logits == first_logits);
    }

    // Plan 2 Task 12: the streamed callback must reproduce gen_tokens exactly.
    std::vector<int32_t> sids;
    REQUIRE(model->BuildPromptInputIds(model->Tokenizer(),
                                       "The capital of France is", sids));
    ModelRequest sreq;
    sreq.input_ids = sids;
    sreq.max_gen_tokens = 2;
    ExpectStreamMatchesTokens(*model, sreq);

    model->Unload();
}

TEST_CASE("DeepSeek-V4.1-Flash end-to-end tool round-trip (real weights)",
          "[deepseek-v41][model][tool][real-model]")
{
    const std::string root = ModelRoot();
    if (!fs::exists(root + "/logestix/model.index.json"))
        SKIP("converted DeepSeek-V4.1-Flash not present");

    ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "deepseek-v4.1-flash";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    const std::vector<ToolCall> calls = RunToolRoundTrip(*model);
    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    model->Unload();
}
