// TML-d24-varloop-research SelfCheck (Model Ingestion Protocol stage 8).
//
// Runs the engine on the REAL converted tree (24 physical layers, 12 of them
// looped three times by default -> 48 applications per token; 18 GiB artifact)
// through the checkpoint's own chat template, and requires the control prompt to
// answer "Paris", three cycles with BIT-IDENTICAL final logits.
//
// Why this row has its own gate rather than reusing the sibling's: the weights
// are a different fine-tune, so the answers (and the logits hash) are this
// model's own. The pass count is the one the index declares (3) unless
// LSX_INKLING_VARLOOP_PASSES overrides it — an override changes the stream, so
// the bit-exactness below is asserted at whatever count the instance runs.
//
// Real weights are optional: an absent vault SKIPs (exit 4), never fails.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "test_manifest_query_root.h"
#include "inkling_moe_varloop_constants.h"
#include "inkling_moe_varloop_model.h"
#include "lsxcommon/inkling_moe_loop_inference_pipeline.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"

namespace fs = std::filesystem;

namespace {

namespace vf = lsxcommon::inkling::varloop::frozen;

int EnvIntOr(const char *name, int dflt)
{
    if (const char *v = std::getenv(name))
        if (const int parsed = std::atoi(v); parsed > 0) return parsed;
    return dflt;
}

// The CONVERTED tree (artifact) — not the checkpoint. Named for what it
// returns: in this tree 'vault' is the read-only SOURCE store, so a helper
// called VaultRoot() that answered an artifact path read backwards.
std::string ConvertedRoot()
{
    if (const char *env = std::getenv("LSX_TEST_TML_D24_VARLOOP_RESEARCH_ROOT"))
        return env;
    return manifest_query::ConvertedDirFor("tml-d24-varloop-research");
}

lsxcommon::ModelInitConfig Cfg(const std::string &root)
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = lsxcommon::InklingMoeVarloopModel::kType;
    return cfg;
}

uint64_t Fnv1a(const std::vector<float> &v)
{
    uint64_t h = 1469598103934665603ULL;
    const auto *p = reinterpret_cast<const unsigned char *>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(float); ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

bool VaultPresent(const fs::path &root)
{
    return fs::exists(root / "model.index.json");
}

}  // namespace

TEST_CASE("varloop real-weight sanity: the model keeps its knowledge",
          "[inkling][varloop][sanity]")
{
    const fs::path root = ConvertedRoot();
    if (!VaultPresent(root))
        SKIP("converted TML-d24-varloop-research tree absent: " + root.string());

    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root.string()));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    // Teacher-forced continuation of the answer sentence, encoded by this
    // checkpoint's own tokenizer (byte-identical to the sibling's, so the ids
    // coincide — the WEIGHTS do not).
    const std::vector<int32_t> sentence = {371, 151451, 149879, 151194, 149884,
                                          152717};
    double nll_sum = 0.0;
    for (std::size_t i = 1; i < sentence.size(); ++i) {
        lsxcommon::ModelRequest req;
        req.input_ids.assign(sentence.begin(),
                             sentence.begin() + static_cast<long>(i));
        req.max_gen_tokens = 1;
        req.ignore_eos = true;
        const auto res = model->Infer(req);
        INFO("prefix length " << i << ": " << res.error_msg);
        REQUIRE(res.ok);
        REQUIRE_FALSE(res.final_logits.empty());
        const auto &lg = res.final_logits;
        const float mx = *std::max_element(lg.begin(), lg.end());
        double sum = 0.0;
        for (float v : lg) sum += std::exp(static_cast<double>(v - mx));
        const double lse = mx + std::log(sum);
        REQUIRE(static_cast<std::size_t>(sentence[i]) < lg.size());
        const double nll = lse - static_cast<double>(lg[sentence[i]]);
        INFO("token " << sentence[i] << " nll " << nll);
        nll_sum += nll;
        if (i + 1 == sentence.size()) {
            std::size_t top = 0;
            for (std::size_t k = 1; k < lg.size(); ++k)
                if (lg[k] > lg[top]) top = k;
            INFO("top-1 after \"The capital of France is\": " << top);
            REQUIRE(top == 152717);
        }
    }
    const double mean_nll = nll_sum / static_cast<double>(sentence.size() - 1);
    WARN("varloop sanity: mean nll " << mean_nll << " -> perplexity "
                                     << std::exp(mean_nll));
    REQUIRE(std::exp(mean_nll) < 100.0);
    model->Unload();
}

TEST_CASE("varloop SelfCheck: the capital of France is Paris",
          "[inkling][varloop][selfcheck]")
{
    const fs::path root = ConvertedRoot();
    if (!VaultPresent(root))
        SKIP("converted TML-d24-varloop-research tree absent: " + root.string());

    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root.string()));
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("TML-d24-varloop-research failed to load");
    const auto *var = dynamic_cast<const lsxcommon::InklingMoeVarloopModel *>(
        model.get());
    REQUIRE(var != nullptr);

    const int cycles =
        EnvIntOr("LSX_TEST_TML_D24_VARLOOP_RESEARCH_SELFCHECK_CYCLES", 3);
    const int max_gen =
        EnvIntOr("LSX_TEST_TML_D24_VARLOOP_RESEARCH_SELFCHECK_MAX_GEN", 32);

    lsxcommon::ModelRequest req;
    REQUIRE(model->BuildPromptInputIds(model->Tokenizer(),
                                       "The capital of France is",
                                       req.input_ids));
    REQUIRE_FALSE(req.input_ids.empty());
    req.max_gen_tokens = max_gen;
    req.ignore_eos = false;

    std::vector<float> first_logits;
    std::vector<int32_t> first_tokens;
    uint64_t first_hash = 0;
    for (int cycle = 0; cycle < cycles; ++cycle) {
        const auto r = model->Infer(req);
        INFO("cycle " << cycle << " error: " << r.error_msg);
        INFO("cycle " << cycle << " text: " << r.output_text);
        REQUIRE(r.ok);
        REQUIRE_FALSE(r.gen_tokens.empty());

        if (cycle == 0) {
            std::string ids;
            for (std::size_t i = 0; i < r.gen_tokens.size() && i < 12; ++i)
                ids += std::to_string(r.gen_tokens[i]) + " ";
            WARN("varloop selfcheck: first ids: "
                 << ids << "| eos=" << r.eos_terminated
                 << "| prompt_len=" << req.input_ids.size()
                 << "| passes=" << var->ActiveLoopPasses());
            first_logits = r.final_logits;
            first_tokens = r.gen_tokens;
            first_hash = Fnv1a(first_logits);
            REQUIRE(r.output_text.find("Paris") != std::string::npos);
        } else {
            REQUIRE(r.gen_tokens == first_tokens);
            REQUIRE(Fnv1a(r.final_logits) == first_hash);
        }
    }
    WARN("varloop selfcheck: "
         << first_tokens.size() << " tokens, logits_hash=" << std::hex
         << first_hash << std::dec << ", passes=" << var->ActiveLoopPasses());
    // The slab is the trained ceiling's whatever count this run used, so the
    // three cycles above also prove the count did not drift between requests.
    CHECK(var->CacheSlotCount() ==
          vf::kLayers + (vf::kLoopPassesMax - 1) * (vf::kLoopEnd - vf::kLoopStart));
    model->Unload();
}

// The IModel-level SelfCheck (the shared Paris driver) must agree with the
// family test above: same prompt, same answer, no SKIP when the vault is there.
TEST_CASE("varloop SelfCheck driver reaches Paris", "[inkling][varloop][selfcheck]")
{
    const fs::path root = ConvertedRoot();
    if (!VaultPresent(root))
        SKIP("converted TML-d24-varloop-research tree absent: " + root.string());

    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root.string()));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    lsxcommon::ModelRequest req;
    req.max_gen_tokens = 32;
    req.ignore_eos = false;
    const auto r = model->SelfCheck(req);
    INFO("error: " << r.error_msg);
    INFO("text: " << r.output_text);
    REQUIRE(r.ok);
    REQUIRE(r.selfcheck_passed);
    model->Unload();
}
