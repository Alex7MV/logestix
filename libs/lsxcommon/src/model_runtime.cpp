#include "lsxcommon/model_runtime.h"

#include <absl/strings/ascii.h>
#include <algorithm>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <vector>

namespace lsxcommon {
namespace runtime {

namespace {
constexpr size_t kMaxDecodeBytes = 1u << 20;
constexpr size_t kInitialDecodeCap = 4096;
constexpr int kMaxDecodeAttempts = 8;
} // namespace

std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids)
{
    if (ids.empty()) return {};
    size_t cap = kInitialDecodeCap;
    std::string out(cap, '\0');
    for (int attempt = 0; attempt < kMaxDecodeAttempts; ++attempt) {
        const int dr = arrow_tokenizer_decode(tok_db, ids.data(), ids.size(),
                                              out.data(), &cap);
        if (dr == 0) {
            out.resize(cap);
            return out;
        }
        if (dr == -2 && cap > out.size() && cap <= kMaxDecodeBytes) {
            out.resize(cap);
            continue;
        }
        break;
    }
    spdlog::warn("[model_runtime] tokenizer decode failed ({} ids)", ids.size());
    return {};
}

gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req)
{
    ModelRequest sc_req = req;
    sc_req.max_gen_tokens = std::max(2, req.max_gen_tokens);
    gpu::InferenceResult r = model.Infer(sc_req);
    if (!r.ok) {
        r.selfcheck_passed = false;
        return r;
    }
    if (!r.gen_tokens.empty()) {
        char buf[512];
        for (int ntok : {1, 2}) {
            if (ntok > static_cast<int>(r.gen_tokens.size())) break;
            size_t cap = sizeof(buf);
            if (arrow_tokenizer_decode(model.Tokenizer(), r.gen_tokens.data(),
                                       static_cast<size_t>(ntok), buf, &cap) != 0)
                continue;
            std::string t(buf, cap);
            absl::StripAsciiWhitespace(&t);
            r.first_gen_text = t;
            if (r.first_gen_text == "Paris") {
                r.selfcheck_passed = true;
                break;
            }
        }
        if (!r.selfcheck_passed && r.first_gen_text.empty()) {
            size_t cap = sizeof(buf);
            if (arrow_tokenizer_decode(model.Tokenizer(), r.gen_tokens.data(),
                                       std::min(static_cast<int>(r.gen_tokens.size()), 8),
                                       buf, &cap) == 0)
                r.first_gen_text = std::string(buf, cap);
        }
    }
    spdlog::info("[selfcheck] top1_tok={} text='{}' passed={}",
                 r.first_gen_tok_id, r.first_gen_text,
                 r.selfcheck_passed ? "true" : "false");
    return r;
}

} // namespace runtime
} // namespace lsxcommon