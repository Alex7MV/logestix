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
#include "lsxcommon/inkling_moe_inference_pipeline.h"
#include "lsxcommon/inkling_moe_tool_format.h"
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

// Parity diagnostic (see the ingest record): dumps the residual stream after
// every layer for ONE fixed token so a host reference can name the layer where
// the engine diverges. Enabled by LSX_INKLING_TRACE=<path>; the token id comes
// from LSX_INKLING_TRACE_TOK (default 1000).
TEST_CASE("inkling parity trace: per-layer hidden states", "[inkling][trace]")
{
    const char *path = std::getenv("LSX_INKLING_TRACE");
    if (path == nullptr) SKIP("LSX_INKLING_TRACE not set");
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "logestix" / "model.index.json"))
        SKIP("converted Inkling-2-Small tree absent: " + root.string());

    static bool once = [] {
        lsxcommon::LsxModelFactory::Register<lsxcommon::InklingMoeModel>();
        return true;
    }();
    (void)once;
    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root.string()));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    auto *m = dynamic_cast<lsxcommon::InklingMoeModel *>(model.get());
    REQUIRE(m != nullptr);
    // A SECOND token can be appended so the trace exercises the multi-position
    // path (conv rings after the first push, two cached keys).
    std::vector<int32_t> toks{EnvIntOr("LSX_INKLING_TRACE_TOK", 1000)};
    for (const char *name : {"LSX_INKLING_TRACE_TOK2", "LSX_INKLING_TRACE_TOK3",
                             "LSX_INKLING_TRACE_TOK4"})
        if (const char *v = std::getenv(name)) toks.push_back(std::atoi(v));
    // What does the head actually predict for the REAL framed prompt? Printed
    // here because "the model emits a space forever" is only actionable once we
    // know whether the top-1 is the assistant role marker.
    {
        lsxcommon::ModelRequest r;
        REQUIRE(model->BuildPromptInputIds(model->Tokenizer(),
                                           "The capital of France is",
                                           r.input_ids));
        r.max_gen_tokens = 1;
        r.ignore_eos = true;
        const auto res = model->Infer(r);
        REQUIRE(res.ok);
        std::vector<std::size_t> top(5, 0);
        std::vector<float> tv(5, -1e30f);
        for (std::size_t i = 0; i < res.final_logits.size(); ++i)
            for (int s2 = 0; s2 < 5; ++s2)
                if (res.final_logits[i] > tv[s2]) {
                    for (int m2 = 4; m2 > s2; --m2) { tv[m2] = tv[m2-1]; top[m2] = top[m2-1]; }
                    tv[s2] = res.final_logits[i]; top[s2] = i;
                    break;
                }
        std::string dump;
        for (int s2 = 0; s2 < 5; ++s2) {
            std::string txt(64, '\0');
            std::size_t cap = txt.size();
            arrow_tokenizer_decode(model->Tokenizer(),
                                   reinterpret_cast<const int32_t *>(&top[s2]), 0,
                                   txt.data(), &cap);
            dump += " " + std::to_string(top[s2]) + "(" +
                    std::to_string(tv[s2]) + ")";
        }
        WARN("inkling head top5 at the framed prompt:" << dump);
        {
            std::string t(4096, '\0');
            std::size_t cap = t.size();
            if (arrow_tokenizer_decode(model->Tokenizer(), r.input_ids.data(),
                                       r.input_ids.size(), t.data(), &cap) == 0) {
                t.resize(cap);
                WARN("inkling prompt decodes to: [" << t << "]");
            }
            // Prefix probes: the model must predict the NEXT token of its own
            // template. If the first few prefixes are right and later ones are
            // not, the defect is positional; if none are right, it is not.
            const std::vector<int32_t> full = {
                200002, 200004, 34182, 279, 154554, 150628, 25, 220, 15, 13,
                24, 200010, 200000, 200004, 371, 151451, 149879, 151194,
                149884, 200010};
            for (std::size_t cut = 1; cut <= full.size(); ++cut) {
                lsxcommon::ModelRequest pr;
                pr.input_ids.assign(full.begin(), full.begin() + cut);
                pr.max_gen_tokens = 1;
                pr.ignore_eos = true;
                const auto pc = model->Infer(pr);
                if (!pc.ok) { WARN("prefix " << cut << " failed: " << pc.error_msg); continue; }
                std::size_t best = 0;
                for (std::size_t i = 1; i < pc.final_logits.size(); ++i)
                    if (pc.final_logits[i] > pc.final_logits[best]) best = i;
                const std::string want =
                    cut < full.size() ? std::to_string(full[cut]) : "-";
                WARN("prefix " << cut << ": top1=" << best << " ("
                               << pc.final_logits[best] << ") next=" << want
                               << (cut < full.size() && static_cast<int32_t>(best) == full[cut] ? "  MATCH" : ""));
            }
            std::string g(512, '\0');
            cap = g.size();
            if (arrow_tokenizer_decode(model->Tokenizer(), res.gen_tokens.data(),
                                       res.gen_tokens.size(), g.data(), &cap) == 0) {
                g.resize(cap);
                WARN("inkling first token decodes to: [" << g << "]");
            }
        }
    }
    // TraceForward LAST: its stage dump (when enabled) must not be overwritten
    // by the diagnostic Infer above, which runs a DIFFERENT sequence.
    REQUIRE(m->TraceForward(toks, path));
    model->Unload();
}

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
            std::string ids;
            for (std::size_t i = 0; i < r.gen_tokens.size() && i < 12; ++i)
                ids += std::to_string(r.gen_tokens[i]) + " ";
            WARN("inkling selfcheck: first ids: " << ids
                 << "| eos=" << r.eos_terminated
                 << "| prompt_len=" << req.input_ids.size());
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
