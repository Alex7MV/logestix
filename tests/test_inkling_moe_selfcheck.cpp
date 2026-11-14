// Inkling-2-Small SelfCheck (Model Ingestion Protocol stage 8).
//
// Runs the engine on the REAL converted vault (594B total / 17B active, 312 GB
// artifact) through the checkpoint's own chat template and requires the control
// prompt to answer "Paris", three cycles with BIT-IDENTICAL final logits.
//
// This is a family-owned selfcheck rather than the shared driver because the
// model has no manifest row yet (the decode engine is the last stage) and the
// driver is parameterised by one. When the row lands this file can be replaced
// by the driver + a `selfcheck_frame_prompt` hook.
//
// Real weights are optional: an absent vault SKIPs (exit 4), never fails.
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "inkling_moe_model.h"
#include "lsxcommon/model_factory.h"

namespace fs = std::filesystem;

namespace {

int EnvIntOr(const char *name, int dflt)
{
    if (const char *v = std::getenv(name))
        if (const int parsed = std::atoi(v); parsed > 0) return parsed;
    return dflt;
}

std::string VaultRoot()
{
    if (const char *env = std::getenv("LSX_TEST_INKLING_2_SMALL_ROOT")) return env;
    return "/mnt/fast_storage/huggingface/Inkling-2-Small";
}

lsxcommon::ModelInitConfig Cfg(const std::string &root)
{
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root;
    cfg.model_type = lsxcommon::InklingMoeModel::kType;
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

}  // namespace

TEST_CASE("inkling SelfCheck: the capital of France is Paris", "[inkling][selfcheck]")
{
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "logestix" / "model.index.json"))
        SKIP("converted Inkling-2-Small tree absent: " + root.string());

    // No manifest row yet, so the generated registration list does not carry
    // this class; register it here (the row arrives in the closing commit).
    static bool once = [] {
        lsxcommon::LsxModelFactory::Register<lsxcommon::InklingMoeModel>();
        return true;
    }();
    (void)once;

    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root.string()));
    REQUIRE(model != nullptr);
    if (!model->Load()) FAIL("Inkling-2-Small failed to load");
    REQUIRE(model->Load());

    const int cycles = EnvIntOr("LSX_TEST_INKLING_SELFCHECK_CYCLES", 3);
    const int max_gen = EnvIntOr("LSX_TEST_INKLING_SELFCHECK_MAX_GEN", 32);

    lsxcommon::ModelRequest req;
    // The prompt is framed by the model's own BuildPromptInputIds inside Infer,
    // so the control string is the raw one the protocol names.
    REQUIRE(model->BuildPromptInputIds(model->Tokenizer(),
                                       "The capital of France is", req.input_ids));
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
            first_logits = r.final_logits;
            first_tokens = r.gen_tokens;
            first_hash = Fnv1a(first_logits);
            // The answer must appear in the decoded stream.
            REQUIRE(r.output_text.find("Paris") != std::string::npos);
        } else {
            REQUIRE(r.gen_tokens == first_tokens);
            REQUIRE(Fnv1a(r.final_logits) == first_hash);
        }
    }
    WARN("inkling selfcheck: " << first_tokens.size()
                              << " tokens, logits_hash=" << std::hex << first_hash
                              << ", text=" << model->Type());
    model->Unload();
}
