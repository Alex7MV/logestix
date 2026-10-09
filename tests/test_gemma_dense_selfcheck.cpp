// TDD — Gemma-4-31B-it (dense) SelfCheck harness (Model Ingestion Protocol
// step 8). Real-model test: SKIPs when no converted checkpoint is present
// (CTest SKIP_RETURN_CODE 4). Framing happens inside the model's own
// SelfCheck via the shared Gemma-4 chat-template registry.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/model_runtime.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/prompt_framing.h>
#include "test_tool_roundtrip.h"

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_GEMMA_DENSE_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path GemmaDenseModelPath()
{
    const char *env = std::getenv("LSX_TEST_GEMMA_DENSE_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "gemma-4-31B-it";
}

} // namespace

TEST_CASE("Gemma-4-31B-it SelfCheck: framed 'The capital of France is' -> "
          "'Paris', stress cycles with bit-exact deterministic logits",
          "[gemma_dense][model][selfcheck][real-model]")
{
    const std::filesystem::path model = GemmaDenseModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Gemma-4-31B-it real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.model_type = "gemma-4-31b-it";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "gemma-4-31b-it");
    REQUIRE(m->Load());

    // input_ids are unused: the model frames the canonical prompt itself.
    lsxcommon::ModelRequest req;
    req.input_ids = {2};  // <bos> placeholder
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    if (!sc0.ok) FAIL("SelfCheck failed: " << sc0.error_msg);

    REQUIRE(sc0.selfcheck_passed);
    REQUIRE(sc0.first_gen_text.find("Paris") != std::string::npos);
    REQUIRE(!sc0.final_logits.empty());

    const std::vector<float> &ref_logits = sc0.final_logits;
    const int32_t ref_first = sc0.first_gen_tok_id;
    const int cycles = SelfCheckCycles();

    for (int i = 1; i < cycles; ++i) {
        auto sc = m->SelfCheck(req);
        REQUIRE(sc.ok);
        REQUIRE(sc.selfcheck_passed);
        CHECK(sc.first_gen_text.find("Paris") != std::string::npos);
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                          ref_logits.size() * sizeof(float)) == 0);
    }

    m->Unload();
}

TEST_CASE("Gemma-4-31B-it end-to-end tool round-trip (real weights)",
          "[gemma_dense][model][tool][real-model]")
{
    const std::filesystem::path model = GemmaDenseModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("Gemma-4-31B-it real model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    cfg.model_type = "gemma-4-31b-it";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<lsxcommon::ToolCall> calls = RunToolRoundTrip(*m);
    REQUIRE(!calls.empty());
    // The model may namespace the function (Gemma emits e.g. "google:weather");
    // the codec contract is that a native call is rendered AND parsed back with
    // the right function token and argument, not the exact spelling.
    CHECK(calls[0].name.find("weather") != std::string::npos);
    CHECK(calls[0].arguments_json.find("Paris") != std::string::npos);

    m->Unload();
}
