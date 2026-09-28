#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"

#include "deepseek_moe_v41_load_internal.h"

namespace lsxcommon {

// DeepSeek-V4.1-Flash text backbone (registry "deepseek-v4.1-flash").
// Autonomous contour: owns its own pinned weights and expert store; no
// DeepSeek-V4-Flash headers.
class DeepSeekV41FlashModel final : public ModelBase {
public:
    static const std::string kType;
    explicit DeepSeekV41FlashModel(ModelInitConfig cfg);
    ~DeepSeekV41FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b).
    // DeepSeek-V4.1 native DSML template (encoding.py, leading-space tags).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;
    deepseek_v41::gpu::PinnedWeights pw_;
    deepseek_v41::gpu::ExpertDb expert_db_;
    bool expert_db_ready_ = false;
};

} // namespace lsxcommon
