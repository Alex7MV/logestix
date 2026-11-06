#pragma once

// Mistral-Small-4-119B (kType "mistral-small-4-119b") — Mistral AI. MLA
// (q/kv latent) + 128-expert top-4 MoE with one ungated shared expert, a
// Llama-4 logit temperature over a YaRN (factor 128) original window.
//
// Frozen geometry is compile-time locked in src/mistral_moe_constants.h and
// cross-checked against the converted index at Load. LSX_TEST_ALLOW_SYNTHETIC_
// GEOMETRY relaxes that gate for synthetic fixtures only.

#include <memory>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/mistral_moe_convert.h"
#include "lsxcommon/mistral_moe_inference.h"
#include "lsxcommon/mistral_moe_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

class MistralSmall4MoeModel final : public ModelBase {
public:
    static const std::string kType;  // "mistral-small-4-119b"

    explicit MistralSmall4MoeModel(ModelInitConfig cfg);

    const std::string &Type() const override { return kType; }
    KvCacheLayout KvLayout() const override { return kv_cache_layout_; }

    // Conversion-time placement profile ("vram" block of model.index.json).
    // Empty on a tree converted before the block became mandatory.
    const VramProfile &VramRequirements() const override { return vram_; }

    // Tool/function-calling bridge (step 4b): the checkpoint's own
    // [INST]/[TOOL_CALLS]name[ARGS]{json} framing.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

protected:
    void TeardownWeights() override;
    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

private:
    bool ValidateGeometry(const mistral_moe::MistralTopology &top) const;

    mistral_moe::MistralPinnedWeights pw_;
    mistral_moe::MistralDims dims_;
    VramProfile vram_;
    KvCacheLayout kv_cache_layout_ = KvCacheLayout::kPosMajor;
    std::unique_ptr<mistral_moe::MistralSession> session_;
    int expert_slots_ = 0;
    std::vector<float> rc_, rs_;
    float *d_rc_ = nullptr;
    float *d_rs_ = nullptr;
    std::vector<float> host_logits_;
    int32_t *d_next_token_ = nullptr;
    float *d_next_maxv_ = nullptr;
    std::vector<int32_t> session_tail_;
    int yarn_capacity_ = 0;
};

} // namespace lsxcommon
