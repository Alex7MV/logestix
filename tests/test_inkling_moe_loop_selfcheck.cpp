// Inkling-2-Mini SelfCheck (Model Ingestion Protocol stage 8).
//
// Runs the engine on the REAL converted tree (24 physical layers, 12 of them
// looped three times -> 48 applications per token; 18 GiB artifact) through the
// checkpoint's own chat template, and requires the control prompt to answer
// "Paris", three cycles with BIT-IDENTICAL final logits.
//
// The loop is the reason this gate exists as its own TU: the extra passes share
// weights but must restart from the pre-loop snapshots, and a pass that reads
// another pass's KV/conv slot, or composes on the previous pass's output,
// still produces fluent text — it just stops being the release's model. Bit
// equality across cycles plus the measurement below are what separate the two.
//
// MEASURED from the source (transformers 5.24.0, release code, real weights,
// see `.superpowers/…/probes/paris.json`): the framed prompt emits
// `<|message_model|><|content_thinking|>` first and reaches " Paris" at token
// 27 of a 32-token window, without EOS. Hence min_tokens/answer_window below
// are 32 and the match is a substring one: a 2-token window would fail on a
// correct model.
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
#include "inkling_moe_loop_model.h"
#include "lsxcommon/inkling_moe_loop_inference_pipeline.h"
#include "lsxcommon/inkling_moe_loop_tool_format.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model_runtime.h"

namespace fs = std::filesystem;

namespace {

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
    if (const char *env = std::getenv("LSX_TEST_INKLING_2_MINI_ROOT")) return env;
    return manifest_query::ConvertedDirFor("inkling-2-mini");
}

lsxcommon::ModelInitConfig Cfg(const std::string &root)
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = lsxcommon::InklingMoeLoopModel::kType;
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

// No manifest row yet (it lands in the closing commit), so the generated
// registration list does not carry this class.
void RegisterOnce()
{
    static bool once = [] {
        lsxcommon::LsxModelFactory::Register<lsxcommon::InklingMoeLoopModel>();
        return true;
    }();
    (void)once;
}

}  // namespace

TEST_CASE("inkling loop real-weight sanity: the model keeps its knowledge",
          "[inkling][sanity]")
{
    const fs::path root = ConvertedRoot();
    if (!fs::exists(root / "model.index.json"))
        SKIP("converted Inkling-2-Mini tree absent: " + root.string());

    RegisterOnce();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root.string()));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

    // Teacher-forced continuation of the answer sentence. The ids are the
    // checkpoint's tokenizer's own encoding ("The capital of France is Paris"),
    // identical to Small's because the tokenizers are byte-identical.
    //
    // What this catches is the defect class that is symmetric in both the
    // artifact and the kernels: Mini's routed experts are a fused [2*I, H]
    // NVFP4 slab whose halves must be DE-INTERLEAVED (verified byte-verbatim
    // and numerically in Tasks 3/4). An unnoted interleave keeps every byte,
    // every plane size and every magnitude, and only destroys knowledge.
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
            // The whole point of the model: the answer, not a filler token.
            REQUIRE(top == 152717);
        }
    }
    const double mean_nll = nll_sum / static_cast<double>(sentence.size() - 1);
    WARN("inkling loop sanity: mean nll " << mean_nll << " -> perplexity "
                                          << std::exp(mean_nll));
    REQUIRE(std::exp(mean_nll) < 100.0);
    model->Unload();
}

TEST_CASE("inkling loop SelfCheck: the capital of France is Paris",
          "[inkling][selfcheck]")
{
    const fs::path root = ConvertedRoot();
    if (!fs::exists(root / "model.index.json"))
        SKIP("converted Inkling-2-Mini tree absent: " + root.string());

    RegisterOnce();
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root.string()));
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("Inkling-2-Mini failed to load");

    const int cycles = EnvIntOr("LSX_TEST_INKLING_2_MINI_SELFCHECK_CYCLES", 3);
    const int max_gen = EnvIntOr("LSX_TEST_INKLING_2_MINI_SELFCHECK_MAX_GEN", 32);

    lsxcommon::ModelRequest req;
    // The prompt is framed by the model's own BuildPromptInputIds inside Infer,
    // so the control string is the raw one the protocol names.
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
            WARN("inkling loop selfcheck: first ids: "
                 << ids << "| eos=" << r.eos_terminated
                 << "| prompt_len=" << req.input_ids.size());
            first_logits = r.final_logits;
            first_tokens = r.gen_tokens;
            first_hash = Fnv1a(first_logits);
            // The measured source stream opens with the assistant/thinking
            // markers and reaches the answer at token 27; a shorter window
            // would be a fixture error, not a model failure.
            REQUIRE(r.output_text.find("Paris") != std::string::npos);
        } else {
            REQUIRE(r.gen_tokens == first_tokens);
            REQUIRE(Fnv1a(r.final_logits) == first_hash);
        }
    }
    WARN("inkling loop selfcheck: "
         << first_tokens.size() << " tokens, logits_hash=" << std::hex
         << first_hash << std::dec << ", text=" << model->Type());
    model->Unload();
}

// The IModel-level SelfCheck (the shared Paris driver) must agree with the
// family test above: same prompt, same answer, no SKIP when the vault is there.
TEST_CASE("inkling loop SelfCheck driver reaches Paris", "[inkling][selfcheck]")
{
    const fs::path root = ConvertedRoot();
    if (!fs::exists(root / "model.index.json"))
        SKIP("converted Inkling-2-Mini tree absent: " + root.string());

    RegisterOnce();
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
