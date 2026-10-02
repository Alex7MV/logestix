// TDD — MiMo-V2.6-Flash-RL SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no converted MiMo checkpoint is present
// (CTest SKIP_RETURN_CODE 4).
//
// KNOWN PERMANENT FAILURE if a MiMo checkpoint ever appears in the vault: the
// checkpoint is bad, not this code. A 48-layer forward built from the vendor's
// own modeling_mimo_v2.py produces the same top-5 token ids and values as this
// implementation, and the vendor oracle fails on a plain question with the
// same repetitive garbage — see docs/open-tasks/2026-09-30-..., item A2. The
// two intermediate hypotheses ("the routed expert weights are wrong", then
// "decode attention diverges at pos >= 2") were both disproved by the oracle
// and by clean single-run dumps respectively. So a red selfcheck here means the
// checkpoint, not logestix: do not start debugging the kernels.
//
// The synthetic MiMo suites (convert, loader, kernels, model, dequant,
// geometry, expert cache, tool framing) DO cover our code and are green — they
// stay the real signal for this family.
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
    if (const char *p = std::getenv("LSX_TEST_MIMO_SELFCHECK_CYCLES"))
    {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

std::filesystem::path MiMoModelPath()
{
    const char *env = std::getenv("LSX_TEST_MIMO_MODEL_PATH");
    return env ? std::filesystem::path(env)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "MiMo-V2.6-Flash-RL";
}

} // namespace

TEST_CASE("MiMo-V2.6-Flash-RL SelfCheck: 'The capital of France is' -> "
          "'Paris', 3 stress cycles with bit-exact deterministic logits",
          "[mimo][model][selfcheck][real-model]")
{
    const std::filesystem::path model = MiMoModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("MiMo converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Type() == "mimo-v2.6-flash-rl");
    REQUIRE(m->Load());

    lsxcommon::ModelRequest req;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    REQUIRE(sc0.ok);
    REQUIRE(sc0.selfcheck_passed);
    REQUIRE(sc0.first_gen_text == "Paris");
    REQUIRE(!sc0.final_logits.empty());

    const std::vector<float> ref_logits = sc0.final_logits;
    const int32_t ref_first = sc0.first_gen_tok_id;
    const int cycles = SelfCheckCycles();
    for (int i = 1; i < cycles; ++i)
    {
        auto sc = m->SelfCheck(req);
        REQUIRE(sc.ok);
        CHECK(sc.selfcheck_passed);
        CHECK(sc.first_gen_text == "Paris");
        CHECK(sc.first_gen_tok_id == ref_first);
        CHECK(sc.final_logits.size() == ref_logits.size());
        if (sc.final_logits.size() == ref_logits.size())
            CHECK(std::memcmp(sc.final_logits.data(), ref_logits.data(),
                              ref_logits.size() * sizeof(float)) == 0);
    }

    m->Unload();
}
