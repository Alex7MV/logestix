// TDD — phi-4 SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no converted phi checkpoint is present
// (CTest SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_PHI_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path PhiModelPath()
{
    const char *env = std::getenv("LSX_TEST_PHI_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "phi-4";
}

std::vector<int32_t> EncodeSelfCheckPrompt(lsxcommon::ArrowTokenizerDb &tok)
{
    const char *prompt = "The capital of France is";
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data(), &cap);
    if (n <= 0) return {};
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("Phi-4.2-30B SelfCheck: 'The capital of France is' -> 'Paris', "
          "3 stress cycles with bit-exact deterministic logits",
          "[phi][model][selfcheck][real-model]")
{
    const std::filesystem::path model = PhiModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Phi converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "phi-4");
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    REQUIRE(sc0.ok);
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

// NOTE: no real-weight tool round-trip case. phi-4 ships no native tool
// syntax and, unlike Granite, is not instruction-tuned to emit a bare Hermes
// JSON call, so the generic parser floor (Model Ingestion Protocol step 4b)
// does not recover a call from the real checkpoint. The bridge itself is
// implemented (Phi4Model::BuildConversationInputIds / inherited
// ParseToolCalls) and verified at the host level in test_phi_tool_framing.
