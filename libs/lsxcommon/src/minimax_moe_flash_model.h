#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/minimax_moe_inference_pipeline.h"
#include "lsxcommon/minimax_moe_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// MiniMax-M3.1-Flash text backbone (registry "minimax-m3.1-flash"), a SIBLING
// of MiniMax-M3: the same GQA + per-head Gemma QK-norm + partial RoPE +
// sigmoid-router MoE contour, with MiniMax Sparse Attention on EVERY layer and
// RoPE theta 1e7. The generation pipeline, kernels and the neutral routed-
// expert store are the parent's (`lsxcommon::minimax`); only the frozen
// passport and the chat-template policy are this model's own.
class MiniMaxM31FlashModel final : public ModelBase {
public:
    static const std::string kType;
    explicit MiniMaxM31FlashModel(ModelInitConfig cfg);
    ~MiniMaxM31FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kPosMajor; }

    // Single-prompt framing through the checkpoint's own template (adaptive
    // thinking: the generation prompt carries no forced think-END prefix).
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Native tool bridge (Model Ingestion Protocol step 4b): the M3.1 template
    // (`]~b]`/`[e~[` framing, `<model_id>8bb8149fe</model_id>`, adaptive
    // thinking line, developer blocks for mid-conversation system turns, one
    // `]<]minimax[>[<tool_call>` block of `<invoke>` XML calls).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;

    minimax::MinimaxPinnedWeights pw_;
    minimax::MinimaxExpertStore experts_;
    gpu::ExpertDeviceCache cache_;
    minimax::MinimaxSession session_;
    minimax::MinimaxScratch scratch_;
    bool expert_cache_ready_ = false;
};

} // namespace lsxcommon
