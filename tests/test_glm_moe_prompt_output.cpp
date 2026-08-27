// TDD — GLM-5.2 meaningful prompt output (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no GLM-5.2 checkpoint is present (CTest
// SKIP_RETURN_CODE 4, same as the other GLM real-weight tests).
//
// Contract under test (GlmV5Model::Infer):
//   - the GLM chat-template framed "The capital of France is" must generate a
//     non-empty tokenizer-decoded output_text,
//   - output_text must contain "Paris" (bit-exact determinism of the greedy
//     decode is covered by test_glm_moe_selfcheck),
//   - r.ok == true && !r.gen_tokens.empty().
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/prompt_framing.h>

namespace {

std::filesystem::path GlmModelPath()
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "GLM-5.2";
}

} // namespace

TEST_CASE("GLM-5.2 Infer: meaningful output_text on the test prompt",
          "[glm][model][prompt][real-model]")
{
    const std::filesystem::path model = GlmModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("GLM real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "glm-5.2");
    REQUIRE(m->Load());

    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids;
    REQUIRE(m->BuildPromptInputIds(m->Tokenizer(), prompt, ids));
    REQUIRE(ids.size() > 6);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 8;

    auto r = m->Infer(req);
    REQUIRE(r.ok);
    REQUIRE(!r.gen_tokens.empty());
    REQUIRE(!r.output_text.empty());
    REQUIRE(r.output_text.find("Paris") != std::string::npos);

    // Unified runtime contract (Task 5): timings, residency, occupancy.
    REQUIRE(r.ttft_ms >= 0.0);
    REQUIRE(r.total_time_ms >= r.ttft_ms);
    REQUIRE(r.tokens_per_sec >= 0.0);
    REQUIRE(r.kv_slots_used >= static_cast<int>(ids.size()));
    REQUIRE(r.weight_resident_bytes > 0);
    REQUIRE(r.num_tokens == static_cast<int>(r.gen_tokens.size()));
    REQUIRE(!r.final_logits.empty());

    m->Unload();
}