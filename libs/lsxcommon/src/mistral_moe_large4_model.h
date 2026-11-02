#pragma once

// Mistral-Large-4-1T-A52B (kType "mistral-large-4-1t") — Mistral AI.
// MLA (latent cache + absorbed decode) with hybrid RoPE/NoPE, LeRoPE, per-head
// position-dependent query scaling, a DSA Lightning Indexer with cross-layer
// KV/mask sharing, 128 routed experts top-4 + one gated shared expert, a dense
// layer 0 and final logit softcapping.
//
// Frozen geometry is compile-time locked in src/mistral_moe_large4_constants.h
// and cross-checked against the converted index at Load.
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY relaxes that gate for synthetic fixtures.

#include <memory>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/mistral_moe_large4_inference_pipeline.h"
#include "lsxcommon/mistral_moe_large4_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

class MistralLarge4MoeModel final : public ModelBase {
public:
    static const std::string kType;  // "mistral-large-4-1t"

    explicit MistralLarge4MoeModel(ModelInitConfig cfg);

    const std::string &Type() const override { return kType; }
    KvCacheLayout KvLayout() const override { return kv_cache_layout_; }

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
    bool PrepareRopeTables(int capacity);
    bool ValidateGeometry() const;

    mistral_large4::Large4PinnedWeights pw_;
    std::unique_ptr<mistral_large4::Large4Session> session_;
    KvCacheLayout kv_cache_layout_ = KvCacheLayout::kPosMajor;
    int expert_slots_ = 0;
    std::vector<float> rc_, rs_;
    std::vector<float> host_logits_;
    int32_t *d_next_token_ = nullptr;
    float *d_next_maxv_ = nullptr;
    std::vector<int32_t> session_tail_;
    int rope_capacity_ = 0;
};

} // namespace lsxcommon
