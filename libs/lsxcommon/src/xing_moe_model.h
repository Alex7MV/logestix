#pragma once

// Xing4.0-29B-A4B (kType "xing4.0-29b-a4b") — China Telecom AI, TeleChat
// lineage. First family in the repo with a FOUR-WIDE residual stream: each
// layer begins by collapsing hc_mult parallel hidden states through a learned,
// Sinkhorn-normalised mixing matrix (Xing4_0HyperConnection) before any norm
// sees them, and mixes the sub-layer output back with the same matrix.
//
// Frozen geometry is compile-time locked in src/xing_moe_constants.h and
// cross-checked against the converted index at Load. LSX_TEST_ALLOW_SYNTHETIC_
// GEOMETRY relaxes that gate for synthetic fixtures only.

#include <memory>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/xing_moe_inference.h"
#include "lsxcommon/xing_moe_convert.h"
#include "lsxcommon/xing_moe_load.h"

namespace lsxcommon {

class XingMoeModel final : public ModelBase {
public:
    static const std::string kType;  // "xing4.0-29b-a4b"

    explicit XingMoeModel(ModelInitConfig cfg);

    const std::string &Type() const override { return kType; }

    // Reports what the session was actually built with, not what was requested:
    // the allocation builds its geometry from cfg.kv_cache_layout.
    KvCacheLayout KvLayout() const override { return kv_cache_layout_; }

    // Tool/function-calling bridge (step 4b): the checkpoint's own
    // <_system>/<_user>/<_bot> framing with <tool_call><param_key>…
    // (the zero-width-space marker is what upstream ships).
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
    bool ValidateGeometry(const xing::XingTopology &top) const;

    xing::XingPinnedWeights pw_;
    xing::XingDims dims_;
    // Set once at construction; the decode session's KV geometry is built from
    // it and never changes for the lifetime of this model.
    KvCacheLayout kv_cache_layout_ = KvCacheLayout::kPosMajor;
    // Per-request decode state (KV cache + four-stream scratch). Created and
    // destroyed INSIDE RunUnderResidencyLock, like every other family.
    std::unique_ptr<xing::XingSession> session_;
    std::vector<float> rc_, rs_;
    float *d_rc_ = nullptr;
    float *d_rs_ = nullptr;
    std::vector<float> host_logits_;
    // Device-side sampling scratch (token id + max logit), allocated once per
    // model instead of once per token: the old loop cudaMalloc'd and freed three
    // buffers for every generated token.
    int32_t *d_next_token_ = nullptr;
    float *d_next_maxv_ = nullptr;
    std::vector<int32_t> session_tail_;
    int yarn_capacity_ = 0;
};

}  // namespace lsxcommon