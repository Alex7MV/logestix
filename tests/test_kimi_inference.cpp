// TDD — Kimi-K2.7-Code full Inference parity (Model Ingestion Protocol).
// Real-model test: SKIPs when no converted Kimi checkpoint is present.
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

namespace {

std::filesystem::path KimiModelPath()
{
    const char *env = std::getenv("LSX_TEST_KIMI_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Kimi-K2.7-Code";
}

std::vector<int32_t> EncodePrompt(lsxcommon::ArrowTokenizerDb &tok,
                                  const char *prompt)
{
    std::vector<int32_t> ids(512, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        tok, prompt, std::strlen(prompt), ids.data(), &cap);
    if (n <= 0) return {};
    ids.resize(static_cast<size_t>(n));
    return ids;
}

} // namespace

TEST_CASE("Kimi-K2.7-Code full inference is deterministic across calls",
          "[kimi][model][inference][real-model]")
{
    const std::filesystem::path model = KimiModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("converted Kimi model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<int32_t> ids =
        EncodePrompt(m->Tokenizer(), "The capital of France is");
    REQUIRE(ids.size() > 0);

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 4;

    auto r1 = m->Infer(req);
    REQUIRE(r1.ok);
    REQUIRE(r1.gen_tokens.size() == 4);
    REQUIRE(!r1.final_logits.empty());

    auto r2 = m->Infer(req);
    REQUIRE(r2.ok);
    CHECK(r2.gen_tokens == r1.gen_tokens);
    CHECK(std::memcmp(r2.final_logits.data(), r1.final_logits.data(),
                      r1.final_logits.size() * sizeof(float)) == 0);

    // Inspect reports the safetensors inventory of the SOURCE checkpoint.
    (void)m->Inspect();

    m->Unload();
}
