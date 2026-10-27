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

// MiniMax-M3 text backbone (registry "minimax-m3").
// GQA + per-head Gemma QK-norm + partial RoPE + MiniMax Sparse Attention
// (Lightning Indexer block selection) + sigmoid-router MoE with an ungated
// shared expert and GPT-OSS-style SwiGLU-OAI. Owns its pinned weights, the
// mmap routed-expert store and the device slot LRU; no other family's headers.
class MinimaxM3Model final : public ModelBase {
public:
    static const std::string kType;
    explicit MinimaxM3Model(ModelInitConfig cfg);
    ~MinimaxM3Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    // The cache this engine actually builds (pos-major, see MinimaxSession).
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kPosMajor; }

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b):
    // MiniMax-M3 native template (chat_template.jinja: `]~b]`/`[e~[` framing,
    // `]<]minimax[>[<tool_call>` + `<invoke name=...>` XML parameters).
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
