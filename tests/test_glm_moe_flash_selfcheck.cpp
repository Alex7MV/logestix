// GLM-5.3-Flash real-model selfcheck: "The capital of France is" -> "Paris",
// incremental 1->2 tokens, 3 stress cycles with bit-exact final_logits.
// Skips (RETURN_CODE 4) without the converted model.
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
    if (const char *env = std::getenv("LSX_TEST_GLM_FLASH_ROOT")) return env;
    return "/mnt/llm_models/models/GLM-5.3-Flash";
}

// Sets an env var for the enclosing scope and restores it on destruction (so a
// failing REQUIRE cannot leak LSX_TRACE_FLASH into sibling test cases).
struct ScopedEnv {
    std::string name;
    ScopedEnv(const char *n, const char *v) : name(n)
    {
#ifdef _WIN32
        _putenv_s(n, v);
#else
        setenv(n, v, 1);
#endif
    }
    ~ScopedEnv()
    {
#ifdef _WIN32
        _putenv_s(name.c_str(), "");
#else
        unsetenv(name.c_str());
#endif
    }
};
} // namespace

TEST_CASE("glm5next flash selfcheck: Paris, 3 bit-exact cycles")
{
    const std::string root = ModelRoot();
    if (!fs::exists(root + "/logestix/model.index.json")) {
        SKIP("converted GLM-5.3-Flash not present");
    }
    ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "glm-5.3-flash";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    ModelRequest req;
    req.max_gen_tokens = 2;
    int cycles = 3;
    if (const char *env = std::getenv("LSX_TEST_GLM_SELFCHECK_CYCLES"))
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
        // TTFT must be measured over the first forward, not a placeholder.
        REQUIRE(r.ttft_ms > 0.0);
        REQUIRE(r.total_time_ms >= r.ttft_ms);
        // Bit-exact determinism across cycles.
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

    // Cooperative cancel (OPEN_TASKS §E1a): the flash driver polls per step, so
    // both the pre-set and mid-generation abort paths must stop the decode.
    ExpectCooperativeCancelStops(*model, sreq);

    model->Unload();
}

TEST_CASE("GLM-5.3-Flash end-to-end tool round-trip (real weights)",
          "[glm-flash][model][tool][real-model]")
{
    const std::string root = ModelRoot();
    if (!fs::exists(root + "/logestix/model.index.json"))
        SKIP("converted GLM-5.3-Flash not present");

    ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "glm-5.3-flash";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    const std::vector<ToolCall> calls = RunToolRoundTrip(*model);
    REQUIRE(!calls.empty());
    CHECK(calls[0].name == "get_weather");
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    model->Unload();
}

// D1 (docs/open-tasks/2026-09-30-...): no test drove RunGlmFlashGeneration, so the
// LSX_TRACE_FLASH host-decision branch fixed in 6108ecf was never executed.
// The Paris selfcheck above reaches generation but with the debug flag unset;
// this case sets it so RunGlmFlashGeneration must launch the probe kernels via
// the HOST mirror (`flash_dbg_on_host`) instead of evaluating the __device__
// symbol from host code (undefined behaviour: reads a host address).
TEST_CASE("GLM-5.3-Flash generation with LSX_TRACE_FLASH set "
          "(RunGlmFlashGeneration debug branch)",
          "[glm-flash][model][generation][real-model]")
{
    const std::string root = ModelRoot();
    if (!fs::exists(root + "/logestix/model.index.json"))
        SKIP("converted GLM-5.3-Flash not present");

    const ScopedEnv trace("LSX_TRACE_FLASH", "1");

    ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = "glm-5.3-flash";
    auto model = LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(model->Tokenizer(),
                                       "The capital of France is", ids));

    ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 4;
    std::vector<int32_t> streamed;
    req.token_callback = [&](const TokenChunk &c) {
        streamed.push_back(static_cast<int32_t>(c.token_id));
    };

    const auto r = model->Infer(req);
    INFO("error: " << r.error_msg << " tokens=" << r.gen_tokens.size());
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());
    REQUIRE(streamed == r.gen_tokens);
    REQUIRE(r.output_text.find("Paris") != std::string::npos);

    model->Unload();
}
