// Real-model test: GLM-5.2 seq-P prefill parity harness.
//
// Harden/verify the existing seq-P prefill: the MoE layers (3..77) run the
// router/slice kernels with block_rows = P (the full prompt length), so every
// prompt token occupies one IndexShare KV slot. This gate FAILS if the prefill
// ever collapses rows (wrong kv_slots_used), loses bit-exactness across two
// identical runs, or if the fused argmax of the final-logits row disagrees with
// the first generated token. SKIPs when the GLM-5.2 checkpoint is absent
// (CTest SKIP_RETURN_CODE 4, same as the other GLM real-weight tests).

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/prompt_framing.h>
#include <lsxcommon/tokenizer.h>

namespace {

std::filesystem::path GlmModelPath()
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "GLM-5.2";
}

} // namespace

TEST_CASE("GLM seq-P prefill stores P KV rows and stays bit-exact", "[glm][model][prefill][real-model]")
{
    const std::filesystem::path model = GlmModelPath();
    if (!std::filesystem::exists(model / "pinned"))
        SKIP("GLM real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.max_gen_tokens = 4;
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "glm-5.2");
    REQUIRE(m->Load());

    // Tokenizer-encoded prompt of length P >= 8 so the MoE prefill sees seq > 1.
    const std::string text = "The capital of France is the city of";
    std::vector<int32_t> input_ids;
    REQUIRE(m->BuildPromptInputIds(m->Tokenizer(), text, input_ids));
    const int P = static_cast<int>(input_ids.size());
    REQUIRE(P >= 8);

    lsxcommon::ModelRequest req;
    req.input_ids = input_ids;
    req.max_gen_tokens = 4;

    auto r1 = m->Infer(req);
    REQUIRE(r1.ok);
    REQUIRE(r1.num_tokens >= 1);
    // seq-P prefill: every prompt token occupies one IndexShare KV slot.
    // A terminating EOS is part of gen_tokens (final-logits argmax parity
    // below relies on it) but the engine breaks before appending it, so an
    // EOS-terminated turn owns one slot less than P + num_tokens.
    const bool ended_on_eos = r1.num_tokens < req.max_gen_tokens;
    REQUIRE(r1.kv_slots_used ==
            P + r1.num_tokens - (ended_on_eos ? 1 : 0));
    REQUIRE_FALSE(r1.final_logits.empty());
    REQUIRE(r1.gen_tokens.size() == static_cast<size_t>(r1.num_tokens));

    // argmax of the final-logits row == the last generated token (the engine
    // publishes the raw head row of the final decode step, not the prefill row).
    int best = 0;
    for (int i = 1; i < static_cast<int>(r1.final_logits.size()); ++i)
        if (r1.final_logits[i] > r1.final_logits[best]) best = i;
    REQUIRE(best == r1.gen_tokens.back());

    // Bit-exact determinism across a second identical run.
    auto r2 = m->Infer(req);
    REQUIRE(r2.ok);
    REQUIRE(r1.final_logits.size() == r2.final_logits.size());
    REQUIRE(std::memcmp(r1.final_logits.data(), r2.final_logits.data(),
                        r1.final_logits.size() * sizeof(float)) == 0);

    m->Unload();
}