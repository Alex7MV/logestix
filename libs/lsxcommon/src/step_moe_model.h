#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/step_moe_inference_pipeline.h"
#include "lsxcommon/step_moe_load.h"

namespace lsxcommon {

// Step-5-Preview (registry "step-5-preview"). 600B-total / 27B-active sparse
// MoE: GQA + per-head QK RMSNorm + partial NEOX RoPE + per-head sigmoid output
// gate, hybrid sliding/full attention, 352 routed experts (top-8) plus a gated
// shared expert, FP32 residual stream. The routed expert set is mmap-streamed
// through the neutral ExpertStore with a device slot LRU.
class Step5PreviewModel final : public ModelBase {
public:
    static const std::string kType;
    explicit Step5PreviewModel(ModelInitConfig cfg);
    ~Step5PreviewModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    // The cache this engine actually builds (pos-major, see StepSession).
    KvCacheLayout KvLayout() const override { return KvCacheLayout::kPosMajor; }

    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;

    step::StepPinnedWeights pw_;
    step::StepSession session_;
    step::StepScratch scratch_;
};

} // namespace lsxcommon
