// TDD — MiMo-V2.6-Flash-RL SelfCheck harness (Model Ingestion Protocol step 8).
// Real-model test: SKIPs when no converted MiMo checkpoint is present
// (CTest SKIP_RETURN_CODE 4).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/log.h>
#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/tokenizer.h>

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

TEST_CASE("MiMo diagnostic: raw vs ChatML-framed prompt output",
          "[mimo][model][selfcheck][real-model][diagnostic]")
{
    const std::filesystem::path model = MiMoModelPath();
    if (!std::filesystem::exists(model / "logestix" / "pinned"))
        SKIP("MiMo converted model directory not present");

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model.string();
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(m != nullptr);
    REQUIRE(m->Load());

    auto run = [&](const std::vector<int32_t> &ids) {
        lsxcommon::ModelRequest req;
        req.input_ids = ids;
        req.max_gen_tokens = 200;
        auto r = m->Infer(req);
        REQUIRE(r.ok);
        return r.output_text;
    };

    // Raw BPE (no chat frame).
    {
        std::vector<int32_t> ids(256, 0);
        size_t cap = ids.size();
        const char *p = "The capital of France is";
        const int32_t n = lsxcommon::arrow_tokenizer_encode(
            m->Tokenizer(), p, std::strlen(p), ids.data(), &cap);
        REQUIRE(n > 0);
        ids.resize(static_cast<size_t>(n));
        const std::string out = run(ids);
        lsxcommon::log::info("[diag] raw prompt output: " + out);
        INFO("raw: " << out);
    }
    // ChatML frame.
    {
        std::vector<int32_t> ids;
        REQUIRE(m->BuildPromptInputIds(m->Tokenizer(),
                                       "The capital of France is", ids));
        if (const char *dd = std::getenv("LSX_MIMO_DUMP_DIR"))
        {
            FILE *f = std::fopen((std::string(dd) + "/ids.bin").c_str(), "wb");
            if (f) { std::fwrite(ids.data(), 4, ids.size(), f); std::fclose(f); }
            std::string txt;
            for (int32_t id : ids) txt += std::to_string(id) + " ";
            lsxcommon::log::info("[diag] framed ids: " + txt);
        }
        const std::string out = run(ids);
        lsxcommon::log::info("[diag] framed prompt output: " + out);
        INFO("framed: " << out);
    }
    // Single-token full-model logits diagnostic (pos 0; rope identity).
    {
        lsxcommon::ModelRequest rq;
        rq.input_ids = {785};
        rq.max_gen_tokens = 1;
        auto r = m->Infer(rq);
        REQUIRE(r.ok);
        std::vector<int> idx(r.final_logits.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
        if (idx.size() >= 5) {
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int a, int b) {
                                  return r.final_logits[a] > r.final_logits[b];
                              });
            std::string s;
            for (int i = 0; i < 5; ++i)
                s += std::to_string(idx[i]) + "(" +
                     std::to_string(r.final_logits[idx[i]]) + ") ";
            lsxcommon::log::info("[diag] single-token top5: " + s);
        }
    }
    m->Unload();
}
