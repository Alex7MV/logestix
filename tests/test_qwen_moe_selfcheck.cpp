// TDD — Qwen3.5-35B-A3B SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no converted Qwen3.5-35B-A3B is present (CTest
// SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
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
    if (const char *p = std::getenv("LSX_TEST_QWEN_MOE_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

// The converted root (contains logestix-qwen35moe/). Default: sibling of the
// source checkpoint in the models vault.
std::filesystem::path QwenMoeConvertedPath()
{
    if (const char *env = std::getenv("LSX_TEST_QWEN_MOE_MODEL_PATH"))
        return std::filesystem::path(env);
    const char *src = std::getenv("LSX_TEST_QWEN_MOE_SOURCE_PATH");
    const std::filesystem::path base =
        src ? std::filesystem::path(src)
            : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                  "Qwen3.5-35B-A3B";
    return base.parent_path() / (base.filename().string() + "-logestix");
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

TEST_CASE("Qwen3.5-35B-A3B SelfCheck: 'The capital of France is' -> 'Paris', "
          "stress cycles with bit-exact deterministic logits",
          "[qwen-moe][model][selfcheck][real-model]")
{
    const std::filesystem::path model_dir = QwenMoeConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix-qwen35moe" / "pinned"))
        SKIP("Qwen3.5-35B-A3B converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "qwen3.5-35b-a3b");
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
