// TDD — NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 SelfCheck harness
// (Model Ingestion Protocol step 8). Real-model test: SKIPs when no converted
// model is present (CTest SKIP_RETURN_CODE 4).
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

#include "nemotron_moe_nvfp4_model.h"

namespace {

constexpr const char *kModelType = "nemotron-3.5-lightning-30b-a3b-nvfp4";
constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p =
            std::getenv("LSX_TEST_NEMOTRON_NVFP4_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path ConvertedPath()
{
    if (const char *env = std::getenv("LSX_TEST_NEMOTRON_NVFP4_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4";
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

TEST_CASE("Nemotron-3.5-NVFP4 SelfCheck: 'The capital of France is' -> 'Paris', "
          "stress cycles with bit-exact deterministic logits",
          "[nemotron][nvfp4][model][selfcheck][real-model]")
{
    const std::filesystem::path model_dir = ConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "model.index.json"))
        SKIP("Nemotron-3.5-NVFP4 converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == kModelType);
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
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                          ref_logits.size() * sizeof(float)) == 0);
    }

    m->Unload();
}
