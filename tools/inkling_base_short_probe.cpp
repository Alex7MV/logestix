// Temporary short-prompt NaN diagnostic: Infer at n=1..20 tokens, report NaN
// count + max logit. Loads the real vault; not part of the test suite.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include "inkling_moe_base_model.h"
#include "lsxcommon/core.h"
#include "lsxcommon/model_factory.h"
int main(int argc, char **argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: %s <vault>\n", argv[0]); return 1; }
    lsxcommon::Core::Initialize();
    lsxcommon::EnsureModelRegistry();
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = argv[1];
    cfg.model_type = "inkling";
    auto m = lsxcommon::LsxModelFactory::Create(cfg);
    if (!m->Load()) { std::fprintf(stderr, "load failed\n"); return 1; }
    auto &tok = m->Tokenizer();
    auto *bm = dynamic_cast<lsxcommon::InklingMoeBaseModel *>(m.get());
    // RAW trace at n=5 to localize the NaN.
    {
        std::vector<int32_t> ids = {976, 9029, 328, 10128, 382};
        if (bm) bm->TraceForward(ids, "/mnt/fast_storage/tmp/inkling_base/raw_trace.bin");
    }
    if (std::getenv("LSX_SHORT_TRACE_ONLY")) { m->Unload(); lsxcommon::Core::Shutdown(); return 0; }
    // RAW prompts (no template markers) at several lengths.
    for (int n : {2, 5, 8, 20}) {
        std::vector<int32_t> ids = {976, 9029, 328, 10128, 382, 12650, 220, 15,
                                    13, 24, 976, 9029, 328, 10128, 382, 12650,
                                    220, 15, 13, 24};
        ids.resize(static_cast<size_t>(n));
        lsxcommon::ModelRequest req;
        req.input_ids = ids;
        req.max_gen_tokens = 1;
        req.ignore_eos = true;
        auto r = m->Infer(req);
        int nan = 0;
        float mx = -1e30f;
        for (float v : r.final_logits) { if (std::isnan(v)) nan++; mx = std::max(mx, v); }
        std::printf("RAW n=%d ok=%d err=%s nan=%d maxlogit=%.3f nlogits=%zu\n",
                    n, r.ok, r.error_msg.c_str(), nan, mx, r.final_logits.size());
    }
    // FRAMED prompt truncated.
    for (int n : {2, 5, 20}) {
        std::vector<int32_t> full;
        if (!m->BuildPromptInputIds(tok, "The capital of France is", full)) {
            std::fprintf(stderr, "framing failed\n"); return 1;
        }
        std::vector<int32_t> ids(full.begin(), full.begin() + std::min<size_t>(n, full.size()));
        lsxcommon::ModelRequest req;
        req.input_ids = ids;
        req.max_gen_tokens = 1;
        req.ignore_eos = true;
        auto r = m->Infer(req);
        int nan = 0;
        float mx = -1e30f;
        for (float v : r.final_logits) { if (std::isnan(v)) nan++; mx = std::max(mx, v); }
        std::printf("FRAMED n=%d ok=%d err=%s nan=%d maxlogit=%.3f nlogits=%zu\n",
                    n, r.ok, r.error_msg.c_str(), nan, mx, r.final_logits.size());
    }
    m->Unload();
    lsxcommon::Core::Shutdown();
    return 0;
}