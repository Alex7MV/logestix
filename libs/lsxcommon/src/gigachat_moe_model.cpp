#include "lsxcommon/gigachat_moe_model.h"

#include "gigachat_moe_constants.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_format.h>

#include <cstdint>

namespace lsxcommon {
namespace {

// Encodes `s` and appends the ids, dropping the encoder's auto-prepended BOS
// (the framing places exactly one BOS at position 0).
bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view s,
                   std::vector<int32_t> &out)
{
    for (size_t cap = 64; cap <= 8192; cap *= 2)
    {
        std::vector<int32_t> ids(cap, 0);
        size_t n = cap;
        const int32_t r =
            arrow_tokenizer_encode(tok, s.data(), s.size(), ids.data(), &n);
        if (r == -2) continue;
        if (r <= 0) return false;
        size_t off = 0;
        if (tok.bos_id >= 0 && !ids.empty() && ids[0] == tok.bos_id) off = 1;
        out.insert(out.end(), ids.begin() + static_cast<ptrdiff_t>(off),
                   ids.begin() + r);
        return true;
    }
    return false;
}

} // namespace

const std::string GigaChat35Model::kType = gigachat::frozen::kType;

GigaChat35Model::GigaChat35Model(ModelInitConfig cfg)
    : ModelBase(std::move(cfg))
{
    // Synthetic fixtures may omit the tokenizer arrows.
    tokenizer_optional_ = true;
}

GigaChat35Model::~GigaChat35Model() { Unload(); }

void GigaChat35Model::TeardownWeights() {}

bool GigaChat35Model::Load()
{
    if (loaded_) return true;
    if (!BeginLoad(cfg_.model_path)) return AbortLoad();

    std::string err;
    if (!gigachat::LoadGigaChatIndex(cfg_.model_path, idx_, err))
    {
        log::error(absl::StrFormat("[gigachat] index error: %s", err));
        return AbortLoad();
    }
    if (!gigachat::ValidateGigaChatGeometry(idx_, err))
    {
        log::error(absl::StrFormat("[gigachat] frozen geometry mismatch: %s",
                                   err));
        return AbortLoad();
    }
    vram_ = idx_.vram;

    // Device weights, expert streaming and the decode pipeline land in Plan 2.
    FinishLoad();
    return true;
}

gpu::InferenceResult GigaChat35Model::Infer(const ModelRequest &req)
{
    // Model-independence invariant: each model asserts its OWN fast-math
    // policy per call (the routing toggles are process-global).
    lsx_nvfp4::LsxFastMathScope scope(EffectiveFastMathMode(cfg_));
    (void)req;
    gpu::InferenceResult r;
    r.ok = false;
    r.error_msg =
        "gigachat inference pipeline not implemented yet (Plan 2)";
    return r;
}

gpu::InferenceResult GigaChat35Model::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

bool GigaChat35Model::BuildPromptInputIds(const ArrowTokenizerDb &tok,
                                          std::string_view raw_prompt,
                                          std::vector<int32_t> &out_ids) const
{
    const int32_t bos = tok.bos_id >= 0
                            ? tok.bos_id
                            : arrow_tokenizer_lookup(tok, "<s>", 3);
    const int32_t role_sep =
        arrow_tokenizer_lookup(tok, "<|role_sep|>\n", 13);
    const int32_t msg_sep =
        arrow_tokenizer_lookup(tok, "<|message_sep|>\n\n", 17);
    const int32_t think = arrow_tokenizer_lookup(tok, "<think>", 7);
    const int32_t think_end = arrow_tokenizer_lookup(tok, "</think>", 8);
    if (bos < 0 || role_sep < 0 || msg_sep < 0 || think < 0 || think_end < 0)
        return prompt::EncodeRawPrompt(tok, raw_prompt, out_ids);

    // chatml v5 single turn + prefilled empty think (so the control prompt
    // answers directly; the reasoning-only template forces `<think>` on).
    static constexpr std::string_view kSystem =
        "\xD0\xA2\xD1\x8B - GigaChat, \xD0\x98\xD0\x98-\xD0\xB0\xD1\x81\xD1\x81"
        "\xD0\xB8\xD1\x81\xD1\x82\xD0\xB5\xD0\xBD\xD1\x82 \xD0\xA1\xD0\xB1"
        "\xD0\xB5\xD1\x80\xD0\xB0. \xD0\x9E\xD1\x82\xD0\xB2\xD0\xB5\xD1\x87"
        "\xD0\xB0\xD0\xB5\xD1\x88\xD1\x8C \xD0\xBA\xD0\xBE\xD1\x80\xD1\x80"
        "\xD0\xB5\xD0\xBA\xD1\x82\xD0\xBD\xD0\xBE, \xD0\xBF\xD0\xBE\xD0\xBB"
        "\xD0\xB5\xD0\xB7\xD0\xBD\xD0\xBE \xD0\xB8 \xD0\xBF\xD0\xBE \xD0\xB4"
        "\xD0\xB5\xD0\xBB\xD1\x83.";

    out_ids.clear();
    out_ids.push_back(bos);
    auto text = [&](std::string_view s) {
        return AppendEncoded(tok, s, out_ids);
    };
    if (!text("system")) return false;
    out_ids.push_back(role_sep);
    if (!text(kSystem)) return false;
    out_ids.push_back(msg_sep);
    if (!text("user")) return false;
    out_ids.push_back(role_sep);
    if (!text(raw_prompt)) return false;
    out_ids.push_back(msg_sep);
    if (!text("assistant")) return false;
    out_ids.push_back(role_sep);
    out_ids.push_back(think);
    out_ids.push_back(think_end);
    return true;
}

} // namespace lsxcommon
