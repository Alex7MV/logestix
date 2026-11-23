#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"

#include "hy_moe_flash_load.h"
#include "lsxcommon/hy_moe_flash_inference_pipeline.h"

namespace lsxcommon {

// Hy4-flash text backbone (tencent/Hy4-flash): 42 layers, hybrid sparse/sliding
// MLA, LatentMoE, iHC residual streams. One registry id, one frozen passport.
class Hy4FlashModel final : public ModelBase {
public:
    static const std::string kType;

    explicit Hy4FlashModel(ModelInitConfig cfg);
    ~Hy4FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Model Ingestion Protocol step 4b: the checkpoint's native Hunyuan
    // conversation and tool format.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

protected:
    void TeardownWeights() override;

private:
    hy4::gpu::PinnedWeights pw_;
    hy4::gpu::ExpertDb expert_db_;
    bool expert_db_ready_ = false;
    hy4::gpu::Session session_;
    hy4::gpu::Scratch scratch_;
    std::vector<int32_t> eos_ids_;
};

} // namespace lsxcommon
