// GLM-5.2 lifecycle integration tests (ResourceGovernor spec, Testing 5).
// Synthetic part always runs (no weights needed); the real-model part is
// gated by LSX_TEST_GLM_MODEL_PATH and SKIPs otherwise - the operator runs
// it on the GLM-capable machine.

#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/glm_convert.h>
#include <lsxcommon/model_factory.h>

#include "glm_test_fixture.h"
#include "glm_v5_model.h"  // lsxcommon src dir (mirrored include dirs below)

namespace fs = std::filesystem;
using lsxcommon::GlmV5Model;

namespace {

bool RealModelPath(std::string *out)
{
    const char *env = std::getenv("LSX_TEST_GLM_MODEL_PATH");
    if (!env || !*env) return false;
    *out = env;
    return true;
}

// Builds a converted synthetic GLM model (hidden=16 -> fails the frozen
// geometry validation, which is exactly what the negative test exercises).
std::string MakeSyntheticConvertedModel(const std::string &tag)
{
    auto base = fs::temp_directory_path() / ("logestix_glm_lifecycle_" + tag);
    fs::remove_all(base);
    auto in = base / "model";
    auto out = base / "converted";
    fs::create_directories(in);
    glmtest::WriteSafetensors(in, "model-00001-of-00001.safetensors",
                              glmtest::BuildSyntheticModel());
    glmtest::WriteConfig(in);
    glmtest::WriteTokenizer(in);
    auto st = lsxcommon::glm::ConvertGlmModel(in.string(), out);
    REQUIRE(st.ok());
    return out.string();
}

} // namespace

TEST_CASE("glm lifecycle: failed load tears down cleanly and is retryable",
          "[glm][lifecycle]")
{
    const std::string path = MakeSyntheticConvertedModel("neg");

    lsxcommon::ModelInitConfig cfg{path, GlmV5Model::kType, /*max_gen*/ 2};
    GlmV5Model model(std::move(cfg));

    // hidden=16 violates kGlmHidden(6144): Load must fail AFTER uploading
    // weights, release every lease/buffer, and stay retryable.
    REQUIRE_FALSE(model.Load());
    REQUIRE(model.PinnedWeights().d_embed == nullptr);  // torn down
    model.Unload();                                     // idempotent
    REQUIRE_FALSE(model.Load());                        // deterministic retry
    model.Unload();
}

TEST_CASE("glm lifecycle: load-once reuse across Infer calls",
          "[glm][lifecycle][real-model]")
{
    std::string path;
    if (!RealModelPath(&path)) {
        SUCCEED("LSX_TEST_GLM_MODEL_PATH not set - SKIP (no GLM weights)");
        return;
    }

    lsxcommon::ModelInitConfig cfg{path, GlmV5Model::kType, /*max_gen*/ 2};
    GlmV5Model model(std::move(cfg));
    REQUIRE(model.Load());

    const void *embed_before = model.PinnedWeights().d_embed;
    REQUIRE(embed_before != nullptr);

    lsxcommon::ModelRequest req;
    req.input_ids = {0, 0, 0, 0};  // stable prompt, same as Benchmark
    req.max_gen_tokens = 2;

    auto r1 = model.Infer(req);
    REQUIRE(r1.ok);
    // Weights were NOT reloaded: device pointer identity is stable.
    REQUIRE(model.PinnedWeights().d_embed == embed_before);

    auto r2 = model.Infer(req);
    REQUIRE(r2.ok);
    REQUIRE(r2.gen_tokens == r1.gen_tokens);
    REQUIRE(r2.final_logits == r1.final_logits);  // bit-exact determinism

    model.Unload();
    REQUIRE(model.PinnedWeights().d_embed == nullptr);
    REQUIRE(model.Load());  // full re-load after unload works
    REQUIRE(model.PinnedWeights().d_embed != nullptr);
    model.Unload();
}
