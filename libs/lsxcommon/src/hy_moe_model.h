#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/hy_moe_inference_pipeline.h"
#include "lsxcommon/hy_moe_load.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Hy-Memory-30B-A3B (registry "hy-memory-30b-a3b").
// Hybrid dense-layer-0 + routed-MoE decoder: GQA + plain per-head QK-norm +
// full RoPE + sigmoid router with an expert bias + one gated shared expert.
// Every routed expert is resident in a per-layer device slab (no expert cache).
class HyMemoryModel final : public ModelBase {
public:
    static const std::string kType;
    explicit HyMemoryModel(ModelInitConfig cfg);
    ~HyMemoryModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    // The cache this engine actually builds (pos-major, see HySession).
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kPosMajor; }

    // Single-prompt framing: a chat model must never see an unframed prompt, so
    // the raw control prompt ("The capital of France is") goes through the
    // checkpoint's own template as one user turn with thinking disabled.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b):
    // native `<|User|>`/`<|Assistant|>` framing plus the `<tool_calls>` /
    // `<arg_key>` XML codec of chat_template.jinja.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;

    hy::HyPinnedWeights pw_;
    hy::HySession session_;
    hy::HyScratch scratch_;
};

} // namespace lsxcommon
