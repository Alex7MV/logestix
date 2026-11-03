// Kimi-K3 selfcheck: "The capital of France is" must answer "Paris" inside the
// thinking channel, deterministically. Loads the real converted model from
// LSX_TEST_KIMI_K3_ROOT, else from the vault path; SKIPs when neither exists.
#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::filesystem::path ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_KIMI_K3_ROOT"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "Kimi-K3";
}

int Cycles()
{
    if (const char *env = std::getenv("LSX_TEST_KIMI_K3_SELFCHECK_CYCLES"))
        if (const int n = std::atoi(env); n > 0) return n;
    return 3;
}

}  // namespace

TEST_CASE("kimi-k3 selfcheck answers Paris", "[kimi-k3][inference]")
{
    const std::filesystem::path root = ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("kimi-k3 converted model directory not present");

    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = "kimi-k3";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("kimi-k3 Load failed");

    const auto &tok = model->Tokenizer();
    std::vector<int32_t> ids;
    REQUIRE(model->BuildPromptInputIds(tok, "The capital of France is", ids));
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 64;

    const auto sc0 = model->SelfCheck(req);
    INFO("selfcheck error: '" << sc0.error_msg << "'");
    INFO("output: '" << sc0.output_text << "'");
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);

    const std::vector<float> &ref = sc0.final_logits;
    const int cycles = Cycles();
    for (int c = 1; c < cycles; ++c) {
        const auto sc = model->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        REQUIRE(sc.final_logits.size() == ref.size());
        REQUIRE(std::memcmp(sc.final_logits.data(), ref.data(),
                            ref.size() * sizeof(float)) == 0);
        REQUIRE(sc.first_gen_tok_id == sc0.first_gen_tok_id);
    }
    model->Unload();
}
