#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"
#include "deepseek_moe_load.cuh"

namespace lsxcommon {

class DeepSeekV4FlashModel final : public ModelBase {
public:
    static const std::string kType;
    explicit DeepSeekV4FlashModel(ModelInitConfig cfg);
    ~DeepSeekV4FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b).
    // DeepSeek-V4 native DSML template (encoding_dsv4.py, no leading-space tags).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

private:
    void TeardownWeights() override;
    gpu::PinnedWeights pw_;
};

} // namespace lsxcommon
