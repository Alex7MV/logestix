#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"
#include "deepseek_moe_pro_load.cuh"

namespace lsxcommon {

// DeepSeek-V4-Pro-0813 text backbone (registry "deepseek-v4-pro-0813").
// Autonomous contour: owns its own pinned weights and expert store; no
// DeepSeek-V4-Flash / V4.1 headers.
class DeepSeekV4ProModel final : public ModelBase {
public:
    static const std::string kType;
    explicit DeepSeekV4ProModel(ModelInitConfig cfg);
    ~DeepSeekV4ProModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b).
    // DeepSeek-V4-Pro native DSML template (encoding/encoding_dsv4.py).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;
    deepseek_pro::gpu::PinnedWeights pw_;
};

} // namespace lsxcommon
