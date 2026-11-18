// Inkling base real-tree numeric probe (NOT a ctest — a manual diagnostic
// tool, like the inkling-2 trace harness). Loads the converted vault, runs
// TraceForward on a short prompt and dumps the per-layer residual stream for
// the independent float64 reference (/mnt/fast_storage/tmp/inkling_base/
// inkling_base_ref.py). Also prints the framed-Paris top-5 and a fixed-
// paragraph mean NLL (the knowledge sanity gate).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "inkling_moe_base_model.h"
#include "lsxcommon/core.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/tokenizer.h"

namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: inkling_base_probe <vault> <trace-out> "
                     "[--nll]\n");
        return 2;
    }
    const std::string vault = argv[1];
    const std::string trace_out = argv[2];
    const bool nll = argc > 3 && std::string(argv[3]) == "--nll";

    lsxcommon::Core::Initialize();
    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = vault;
    cfg.model_type = "inkling";
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!model) { std::fprintf(stderr, "factory failed\n"); return 1; }
    auto *m = dynamic_cast<lsxcommon::InklingMoeBaseModel *>(model.get());
    if (!m) { std::fprintf(stderr, "wrong class\n"); return 1; }
    if (!m->Load()) { std::fprintf(stderr, "load failed\n"); return 1; }
    std::fprintf(stderr, "loaded, resident %llu MiB\n",
                 static_cast<unsigned long long>(m->ResidentWeightBytes() >> 20));

    auto &tok = m->Tokenizer();
    // The framed Paris probe (the model's own template).
    std::vector<int32_t> ids;
    if (!m->BuildPromptInputIds(tok, "The capital of France is", ids)) {
        std::fprintf(stderr, "framing failed\n");
        return 1;
    }
    std::fprintf(stderr, "framed prompt: %zu ids\n", ids.size());
    if (!m->TraceForward(ids, trace_out)) {
        std::fprintf(stderr, "trace failed\n");
        return 1;
    }
    std::fprintf(stderr, "trace written: %s\n", trace_out.c_str());

    // One-token decode at the framed prompt: top-5 of the final logits.
    lsxcommon::ModelRequest req;
    req.input_ids = ids;
    req.max_gen_tokens = 1;
    req.ignore_eos = true;
    const auto r = m->Infer(req);
    if (!r.ok) {
        std::fprintf(stderr, "infer failed: %s\n", r.error_msg.c_str());
        return 1;
    }
    std::vector<std::pair<float, int>> top;
    for (size_t i = 0; i < r.final_logits.size(); ++i)
        top.emplace_back(r.final_logits[i], static_cast<int>(i));
    std::partial_sort(top.begin(), top.begin() + 5, top.end(),
                      [](auto a, auto b) { return a.first > b.first; });
    std::printf("framed-Paris top5:");
    for (int i = 0; i < 5; ++i) {
        char buf[128];
        size_t cap = sizeof(buf);
        int32_t id = top[i].second;
        lsxcommon::arrow_tokenizer_decode(tok, &id, 1, buf, &cap);
        buf[cap] = 0;
        std::printf(" %d(%.3f,'%s')", id, top[i].first, buf);
    }
    std::printf("\n");

    // Full generation (the model is inline-reasoning: it emits a thinking
    // marker first, then the answer). Decode the whole span to verify Paris.
    {
        lsxcommon::ModelRequest gr;
        gr.input_ids = ids;
        gr.max_gen_tokens = 48;
        gr.ignore_eos = false;
        const auto g = m->Infer(gr);
        if (g.ok && !g.gen_tokens.empty()) {
            std::vector<char> txt(g.gen_tokens.size() * 8 + 64, 0);
            size_t cap = txt.size();
            lsxcommon::arrow_tokenizer_decode(
                tok, g.gen_tokens.data(), g.gen_tokens.size(), txt.data(), &cap);
            txt[cap] = 0;
            std::printf("framed-Paris generation (%zu tok): %s\n",
                        g.gen_tokens.size(), txt.data());
        } else {
            std::printf("generation failed: %s\n", g.error_msg.c_str());
        }
    }

    if (nll) {
        // Teacher-forced mean NLL over a fixed paragraph (raw BPE ids, the
        // knowledge sanity gate of the ingestion skill).
        const char *text =
            "The capital of France is Paris. The capital of Germany is Berlin. "
            "The capital of Italy is Rome. The capital of Spain is Madrid. "
            "Water boils at one hundred degrees Celsius at sea level. "
            "The Earth orbits the Sun once every year. "
            "Two plus two equals four. Three times three equals nine.";
        std::vector<int32_t> sent(strlen(text) * 4 + 64, 0);
        size_t cap = sent.size();
        int32_t n = lsxcommon::arrow_tokenizer_encode(tok, text, strlen(text),
                                                      sent.data(), &cap);
        if (n <= 0) { std::fprintf(stderr, "encode failed\n"); return 1; }
        sent.resize(static_cast<size_t>(n));
        double nll_sum = 0;
        int counted = 0;
        for (size_t i = 1; i < sent.size(); ++i) {
            lsxcommon::ModelRequest tr;
            tr.input_ids.assign(sent.begin(), sent.begin() + i);
            tr.max_gen_tokens = 1;
            tr.ignore_eos = true;
            const auto rr = m->Infer(tr);
            if (!rr.ok) {
                std::fprintf(stderr, "nll infer failed at %zu: %s\n", i,
                             rr.error_msg.c_str());
                return 1;
            }
            const int32_t want = sent[i];
            if (want < 0 ||
                static_cast<size_t>(want) >= rr.final_logits.size())
                continue;
            float mx = -1e30f;
            for (float v : rr.final_logits) mx = std::max(mx, v);
            double sum = 0;
            for (float v : rr.final_logits) sum += std::exp(double(v) - mx);
            const double lp =
                double(rr.final_logits[want]) - mx - std::log(sum);
            nll_sum += -lp;
            ++counted;
        }
        const double mean = nll_sum / counted;
        std::printf("knowledge sanity: mean nll %.4f -> perplexity %.2f "
                    "(%d positions)\n", mean, std::exp(mean), counted);
    }
    m->Unload();
    lsxcommon::Core::Shutdown();
    return 0;
}