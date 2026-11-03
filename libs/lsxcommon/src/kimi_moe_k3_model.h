#pragma once

#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/kimi_moe_k3_inference_pipeline.h"
#include "lsxcommon/kimi_moe_k3_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Kimi-K3 (kType "kimi-k3"): KimiLinear — 69 KDA + 24 gated NoPE MLA layers,
// an AttnRes depth mixer (block 12), Stable LatentMoE (896 experts top-16 with
// a 3584 latent + 2 shared experts) and SiTU-GLU. Frozen geometry lives in
// src/kimi_moe_k3_constants.h (the passport); Load cross-checks the converted
// index against it.
class KimiK3Model final : public ModelBase {
public:
    static const std::string kType;  // "kimi-k3"

    explicit KimiK3Model(ModelInitConfig cfg);
    ~KimiK3Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // A chat checkpoint must never see an unframed prompt: a raw string is
    // out of distribution, so the single-prompt path renders the same XTML
    // user turn + thinking generation prompt the conversation bridge does.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Model Ingestion Protocol step 4b: native XTML conversation/tool codec.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    const std::string &Type() const override { return kType; }

protected:
    void TeardownWeights() override;

    kimi_k3::KimiK3Topology top_;
    kimi_k3::KimiK3PinnedWeights pw_;
    std::unique_ptr<kimi_k3::KimiK3ExpertDb> experts_;
    bool weights_loaded_ = false;
};

}  // namespace lsxcommon
