#pragma once

// GLM-5.3-Flash model (kType "glm-5.3-flash"). Standalone ModelBase
// subclass (Kimi/Qwen38 pattern). Frozen geometry passport in
// src/glm_moe_flash_constants.h; loader validates at load.
#include <memory>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/glm_moe_flash_inference.h"
#include "lsxcommon/glm_moe_flash_load.h"

namespace lsxcommon {

class Glm5NextFlashModel final : public ModelBase {
public:
    static const std::string kType;  // "glm-5.3-flash"

    explicit Glm5NextFlashModel(ModelInitConfig cfg);
    ~Glm5NextFlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b). GLM-5.3-
    // Flash native template (verified chat_template.jinja): a `<|system|>` tools
    // block and `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>
    // </tool_call>` calls; tool results ride a `<|observation|>` turn.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    static bool RenderGlmFlashConversation(const ChatConversation &conv,
                                           std::string &out);
    static std::vector<ToolCall> ParseGlmFlashToolCallText(std::string_view text);

private:
    void TeardownWeights() override;

    glm_moe_flash::FlashPinnedWeights pw_;
    moe::MoEExpertStore experts_;
    gpu::ExpertDeviceCache cache_{};
};

} // namespace lsxcommon
