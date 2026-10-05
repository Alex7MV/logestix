#pragma once

// Nemotron-3.5-Lightning-30B-A3B model (kType "nemotron-3.5-lightning-30b-a3b").
// Hybrid Mamba-2 + MoE + full-attention, standalone ModelBase subclass
// (GlmV5Model / QwenMoeModelBase pattern). The frozen geometry passport is
// compile-time locked (src/nemotron_moe_constants.h); Load cross-checks the
// converted index against it and fails on mismatch.
//
// Milestone: loader/model contour. Device kernels (Mamba-2 / attention / MoE)
// and the layer pipeline land in the same protocol sequence; Infer opens its
// own LsxFastMathScope (model-independence invariant) and is wired to the
// pipeline driver once it exists.

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/nemotron_moe_expert_cache.h"
#include "lsxcommon/nemotron_moe_inference.h"
#include "lsxcommon/nemotron_moe_load.h"

namespace lsxcommon {

class NemotronHModel final : public ModelBase {
public:
    static const std::string kType;  // "nemotron-3.5-lightning-30b-a3b"

    explicit NemotronHModel(ModelInitConfig cfg);
    ~NemotronHModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    const std::string &Type() const override { return kType; }

    // Already head-major by construction: its attention indexes
    // (hk * cache_stride + t) * head_dim, so cfg.kv_cache_layout has nothing to
    // switch here. Reported rather than assumed, so a test asking "what layout is
    // this model using" gets the truth instead of the request echoed back.
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kHeadMajor; }

    // Tool/function-calling bridge (step 4b): Nemotron-3.5 uses the Qwen-style
    // ChatML XML call syntax (`<tool_call><function=…>`), but its own tool
    // declaration block; parse is shared with the Qwen codec.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    static bool RenderNemotronConversation(const ChatConversation &conv,
                                           std::string &out);

    // Test/diagnostics access to the parsed passport.
    const nemotron_moe::NemotronIndex &IndexForTest() const { return idx_; }

    // Test/diagnostics access to the routed-expert slot LRU.
    const nemotron_moe::NmExpertCache &ExpertCacheForTest() const
    {
        return expert_cache_;
    }

private:
    void TeardownWeights() override;
    bool ValidateGeometry(const nemotron_moe::NemotronIndex &idx) const;

    nemotron_moe::NemotronIndex idx_;
    nemotron_moe::NmWeights weights_;
    nemotron_moe::NmSession session_;
    nemotron_moe::NmExpertCache expert_cache_;
};

} // namespace lsxcommon
