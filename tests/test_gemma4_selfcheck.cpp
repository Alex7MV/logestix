// TDD — Gemma-4-26B-A4B SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no Gemma checkpoint is present (CTest
// SKIP_RETURN_CODE 4). Prompt framing (Gemma chat template) happens inside
// GemmaBaseModel::SelfCheck via the prompt_framing registry.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_GEMMA_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path GemmaModelPath()
{
    const char *env = std::getenv("LSX_TEST_GEMMA_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "gemma-4-26B-A4B-it";
}

} // namespace

TEST_CASE("Gemma-4-26B-A4B SelfCheck: framed 'The capital of France is' -> "
          "'Paris', stress cycles with bit-exact deterministic logits",
          "[gemma][model][selfcheck][real-model]")
{
    const std::filesystem::path model = GemmaModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Gemma real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "gemma-4-26b-a4b-it");
    REQUIRE(m->Load());

    // input_ids are unused: GemmaBaseModel::SelfCheck frames the canonical
    // prompt through the chat-template registry itself.
    lsxcommon::ModelRequest req;
    req.input_ids = {2}; // <bos> placeholder
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    if (!sc0.ok) FAIL("SelfCheck failed: " << sc0.error_msg);
    REQUIRE(sc0.selfcheck_passed);
    REQUIRE(sc0.first_gen_text == "Paris");
    REQUIRE(!sc0.final_logits.empty());

    const std::vector<float> &ref_logits = sc0.final_logits;
    const int32_t ref_first = sc0.first_gen_tok_id;
    const int cycles = SelfCheckCycles();

    for (int i = 1; i < cycles; ++i) {
        auto sc = m->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        CHECK(sc.first_gen_text == "Paris");
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                          ref_logits.size() * sizeof(float)) == 0);
    }

    m->Unload();
}
