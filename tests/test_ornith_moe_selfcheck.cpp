// TDD — Ornith-1.5-35B-A3B SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no converted Ornith-1.5-35B-A3B is present
// (CTest SKIP_RETURN_CODE 4).
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

#include "ornith_moe_model.h"

namespace {

constexpr int kStressCycles = 3;

int SelfCheckCycles()
{
    if (const char *p = std::getenv("LSX_TEST_ORNITH_SELFCHECK_CYCLES")) {
        const int n = std::atoi(p);
        if (n >= 1) return n;
    }
    return kStressCycles;
}

// Converted root: unified vault convention <model>/logestix.
std::filesystem::path OrnithConvertedPath()
{
    if (const char *env = std::getenv("LSX_TEST_ORNITH_MODEL_PATH"))
        return std::filesystem::path(env);
    const char *src = std::getenv("LSX_TEST_ORNITH_SOURCE_PATH");
    return src ? std::filesystem::path(src)
               : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                     "Ornith-1.5-35B-A3B-MLX";
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

TEST_CASE("Ornith-1.5-35B-A3B SelfCheck: 'The capital of France is' -> "
          "'Paris', stress cycles with bit-exact deterministic logits",
          "[ornith][model][selfcheck][real-model]")
{
    const std::filesystem::path model_dir = OrnithConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "pinned"))
        SKIP("Ornith-1.5-35B-A3B converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    // Auto-detection path: index "model" -> "ornith-1.5-35b-a3b".
    REQUIRE(m->Type() == "ornith-1.5-35b-a3b");
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

TEST_CASE("Ornith-1.5-35B-A3B generation scratch persists across SelfCheck "
          "cycles",
          "[ornith][model]")
{
    const std::filesystem::path model_dir = OrnithConvertedPath();
    if (!std::filesystem::exists(model_dir / "logestix" / "pinned"))
        SKIP("Ornith-1.5-35B-A3B converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    const std::vector<int32_t> ids = EncodeSelfCheckPrompt(m->Tokenizer());
    REQUIRE(!ids.empty());

    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 2;

    auto sc0 = m->SelfCheck(req);
    REQUIRE(sc0.ok);

    auto *qm = dynamic_cast<lsxcommon::OrnithMoeModel *>(m.get());
    REQUIRE(qm != nullptr);
    // Load() runs a 1-token warmup generation, so the arena is first sized
    // for a tiny sequence and the first SelfCheck may grow it exactly once.
    // The invariant under test is that subsequent cycles never reallocate.
    const int allocs0 = qm->GenScratchForTest().arena_allocs;
    CHECK(qm->GenScratchForTest().dequant_runs == 1);
    CHECK(allocs0 <= 2);

    auto sc1 = m->SelfCheck(req);
    REQUIRE(sc1.ok);
    CHECK(qm->GenScratchForTest().dequant_runs == 1);
    CHECK(qm->GenScratchForTest().arena_allocs == allocs0);

    REQUIRE(sc1.final_logits.size() == sc0.final_logits.size());
    CHECK(std::memcmp(sc1.final_logits.data(), sc0.final_logits.data(),
                      sc0.final_logits.size() * sizeof(float)) == 0);

    m->Unload();
}
