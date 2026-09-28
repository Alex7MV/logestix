#pragma once

// GigaChat3.5-432B-A28B-Reasoning model (kType
// "gigachat3.5-432b-a28b-reasoning"). Hybrid MLA + GatedDeltaNet + MoE,
// standalone ModelBase subclass. The frozen geometry passport is compile-time
// locked (src/gigachat_moe_constants.h); Load cross-checks the converted index
// against it and fails on mismatch.
//
// Plan 1 milestone: loader/model contour. CUDA inference kernels and the
// decode pipeline land in Plan 2; Infer opens its own LsxFastMathScope
// (model-independence invariant) and returns a not-implemented sentinel until
// then.

#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/gigachat_moe_expert_cache.h"
#include "lsxcommon/gigachat_moe_inference_pipeline.h"
#include "lsxcommon/gigachat_moe_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

class GigaChat35Model final : public ModelBase {
public:
    static const std::string kType;  // "gigachat3.5-432b-a28b-reasoning"

    explicit GigaChat35Model(ModelInitConfig cfg);
    ~GigaChat35Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    const std::string &Type() const override { return kType; }

    // Tool/function-calling bridge (step 4b): GigaChat chatml-v5 GCML calls
    // (gigachat_tool_format.h).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    // Test/diagnostics access to the parsed passport.
    const gigachat::GigaChatIndex &IndexForTest() const { return idx_; }

private:
    void TeardownWeights() override;
    bool BuildPipelineDims(gigachat::GcPipelineDims &dims) const;

    gigachat::GigaChatIndex idx_;
    gigachat::GcWeights weights_;
    gigachat::GcExpertCache expert_cache_;
    gigachat::GcPipeline pipeline_;
    float *d_logits_ = nullptr;
    int max_seq_ = 0;  // pipeline KV/state capacity (prompt + generated)
};

} // namespace lsxcommon
