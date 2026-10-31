#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/beam_moe_inference.h"
#include "lsxcommon/beam_moe_load.h"
#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Beam-1.0-501B-A23B (registry "beam").
// Gated GQA with sliding-window layers (RoPE) and full-attention layers (NoPE
// + SSMax), sandwich norms + depth-scaled residual, a group-limited sigmoid
// router and 256 routed experts streamed from the host store.
class BeamMoeModel final : public ModelBase {
public:
    static const std::string kType;
    explicit BeamMoeModel(ModelInitConfig cfg);
    ~BeamMoeModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
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

    beam::BeamPinnedWeights pw_;
    beam::BeamSession session_;
    beam::BeamScratch scratch_;
    std::vector<int32_t> eos_ids_;
};

} // namespace lsxcommon
