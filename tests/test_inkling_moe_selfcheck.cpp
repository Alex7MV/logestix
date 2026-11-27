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

#include <algorithm>
#include <cmath>

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
    return "/mnt/fast_storage/lsx_model/thinkingmachines/Inkling-2-Small";
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
    fs::path root = VaultRoot();
    if (const char *alt = std::getenv("LSX_INKLING_TRACE_ROOT")) root = alt;
    if (!fs::exists(root / "model.index.json"))
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
    // A comma-separated list overrides the individual variables (used by
    // tools/inkling_transformers_crosscheck.py, which must feed the engine the
    // exact ids it fed transformers).
    if (const char *list = std::getenv("LSX_INKLING_TRACE_TOKS")) {
        toks.clear();
        std::string s2(list);
        std::size_t p2 = 0;
        while (p2 <= s2.size()) {
            const std::size_t c2 = s2.find(',', p2);
            const std::string piece = s2.substr(p2, c2 == std::string::npos
                                                       ? std::string::npos
                                                       : c2 - p2);
            if (!piece.empty()) toks.push_back(std::atoi(piece.c_str()));
            if (c2 == std::string::npos) break;
            p2 = c2 + 1;
        }
        REQUIRE_FALSE(toks.empty());
    }
    // What does the head actually predict for the REAL framed prompt? Printed
    // here because "the model emits a space forever" is only actionable once we
    // know whether the top-1 is the assistant role marker.
    // The diagnostic block needs the checkpoint's tokenizer; a converted
    // fixture without one (the transformers cross-check tree) skips it.
    if (std::getenv("LSX_INKLING_SKIP_DIAG") == nullptr &&
        InklingMarkerId(model->Tokenizer(), "<|message_user|>") >= 0) {
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
            std::string ids_txt;
            for (std::size_t i2 = 0; i2 < r.input_ids.size(); ++i2)
                ids_txt += (i2 ? "," : "") + std::to_string(r.input_ids[i2]);
            WARN("inkling prompt ids (" << r.input_ids.size() << "): " << ids_txt);
        }
        {
            std::string t(4096, '\0');
            std::size_t cap = t.size();
            if (arrow_tokenizer_decode(model->Tokenizer(), r.input_ids.data(),
                                       r.input_ids.size(), t.data(), &cap) == 0) {
                t.resize(cap);
                WARN("inkling prompt decodes to: [" << t << "]");
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

// The KV session grows on demand. Growing it must copy the rows already written
// WITHOUT reading past the old allocation: `LayerStride()` includes the capacity,
// so `kv_rows * LayerStride()` over-reads by capacity/kv_rows -- past the layer,
// and past the whole buffer once the new stride is bigger. The runtime rejects
// that D2D copy with cudaErrorInvalidValue, and the error then surfaces at the
// next request's first unchecked launch as if an unrelated kernel had failed
// (measured: "inkling prompt: [embed] embed row (cuda 1)" on the second request).
// This pins the layout too: every layer's rows must land at the same
// (layer, head, pos, dim) offsets in the new buffer.
TEST_CASE("inkling KV session growth keeps every layer's rows intact",
          "[inkling][session]")
{
    if (cudaFree(nullptr) != cudaSuccess) SKIP("no CUDA device");
    lsxcommon::inkling::InklingTopology top;
    top.num_layers = 3;
    top.num_kv_heads = 2;
    top.head_dim = 4;

    lsxcommon::inkling::InklingSession s;
    REQUIRE(s.EnsureCapacity(top, 2, cudaStream_t{}) == cudaSuccess);
    const lsxcommon::KvCacheGeometry old_geom = s.geom;
    REQUIRE(s.capacity == 2);

    auto value = [](int l, int h, int p, int d) {
        return static_cast<__half>(100.0f * l + 10.0f * h + p + 0.25f * d);
    };
    std::vector<__half> seed(static_cast<std::size_t>(old_geom.Elements()), __half{});
    for (int l = 0; l < 3; ++l)
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < 2; ++p)
                for (int d = 0; d < 4; ++d)
                    seed[static_cast<std::size_t>(
                        old_geom.Offset(l, h, p, d))] = value(l, h, p, d);
    REQUIRE(cudaMemcpy(s.d_k, seed.data(), seed.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(s.d_v, seed.data(), seed.size() * sizeof(__half),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    s.kv_rows = 2;

    // Grow: this is the call that used to over-read.
    REQUIRE(s.EnsureCapacity(top, 5, cudaStream_t{}) == cudaSuccess);
    REQUIRE(s.capacity >= 5);
    // The copy is asynchronous on the stream we passed; the same stream's
    // synchronize is what the pipeline relies on.
    REQUIRE(cudaStreamSynchronize(cudaStream_t{}) == cudaSuccess);

    std::vector<__half> got(static_cast<std::size_t>(s.geom.Elements()), __half{});
    REQUIRE(cudaMemcpy(got.data(), s.d_k, got.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int l = 0; l < 3; ++l)
        for (int h = 0; h < 2; ++h)
            for (int p = 0; p < 2; ++p)
                for (int d = 0; d < 4; ++d) {
                    INFO("l" << l << " h" << h << " p" << p << " d" << d);
                    REQUIRE(got[static_cast<std::size_t>(s.geom.Offset(l, h, p, d))] ==
                            value(l, h, p, d));
                }
    // A second, no-op growth must not disturb anything.
    REQUIRE(s.EnsureCapacity(top, 4, cudaStream_t{}) == cudaSuccess);
    s.Free();
}

// Quantitative sanity gate on the REAL weights.
//
// The byte-verbatim checks prove the artifact carries the release's bytes, and
// the parity traces prove the kernels implement the reference maths. Neither can
// see a defect that is symmetric in both: the fused expert slab taken as two
// halves instead of de-interleaved keeps every byte, every plane size and every
// magnitude, and both my float64 reference and the engine reproduced it exactly.
// What it destroys is the model's KNOWLEDGE -- measured 12,881 mean perplexity
// over "The capital of France is Paris" and the answer " the" instead of
// " Paris", against 30.4 and " Paris" once the slab is de-interleaved.
//
// So this gate scores the model's own continuation. It is deterministic, needs
// one prefill per position and no sampling, and its bound sits ~100x above a
// healthy value: a defect that only makes the model vague still passes, a defect
// that costs it its knowledge fails loudly. The ids below are the checkpoint's
// own tokenizer's encoding of "The capital of France is".
TEST_CASE("inkling real-weight sanity: the model keeps its knowledge",
          "[inkling][sanity]")
{
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "model.index.json"))
        SKIP("converted Inkling-2-Small tree absent: " + root.string());

    static bool once = [] {
        lsxcommon::LsxModelFactory::Register<lsxcommon::InklingMoeModel>();
        return true;
    }();
    (void)once;

    auto model = lsxcommon::LsxModelFactory::Create(Cfg(root.string()));
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());

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
        // The target id must be inside the unpadded vocabulary the engine scores.
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
    WARN("inkling sanity: mean nll " << mean_nll << " -> perplexity "
                                     << std::exp(mean_nll));
    REQUIRE(std::exp(mean_nll) < 100.0);
    model->Unload();
}

TEST_CASE("inkling SelfCheck: the capital of France is Paris", "[inkling][selfcheck]")
{
    const fs::path root = VaultRoot();
    if (!fs::exists(root / "model.index.json"))
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
