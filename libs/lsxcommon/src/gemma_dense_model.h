#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/gemma_dense_constants.h"
#include "lsxcommon/gemma_dense_inference.h"
#include "lsxcommon/gemma_dense_load.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/prompt_framing.h"

namespace lsxcommon {

// Gemma-4-31B-it (dense): the distilled/dense member of the Gemma-4 family.
// 60 layers / 5376 hidden / 32 Q / 16 sliding KV + 4 global KV / 21504 MLP,
// GeGLU, 5:1 sliding/global attention with per-head QK-norm and V tied to K on
// the global layers, tied embed/lm_head, softcapped logits. Frozen geometry
// lives in gemma_dense_constants.h (passport, model-independence invariant).
class Gemma4_31BDenseModel final : public ModelBase {
public:
    static const std::string kType;

    explicit Gemma4_31BDenseModel(ModelInitConfig cfg);
    ~Gemma4_31BDenseModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Gemma-4 framing — the 31B checkpoint ships a tokenizer.json and
    // chat_template.jinja BYTE-IDENTICAL to the 26B's, so the family framing
    // and the <|tool_call> codec are reused verbatim (stages 4/4b).
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override
    {
        return gemma::BuildPromptInputIds(tok, raw_prompt, out_ids);
    }
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

protected:
    // Hard geometry gate (protocol step 3): false when the loaded topology does
    // not match the frozen compile-time constants (synthetic tests bypass via
    // SyntheticGeometryAllowed()).
    bool ValidateGeometry(const gemma_dense::GemmaDenseTopology &top) const;

    void TeardownWeights() override;

    gemma_dense::GemmaDensePinnedWeights pw_;
    bool weights_loaded_ = false;

    // Incremental-session state. The generation loop always prefills its input
    // (capacity-grown workspace); the tail is tracked for orchestrator
    // introspection. Both are created/reset/published INSIDE the residency lock.
    std::unique_ptr<gemma_dense::GemmaDenseKvCache> kv_;
    std::vector<int32_t> session_tail_;

    gemma_dense::GemmaDenseGenContext *gen_ctx_ = nullptr;
};

} // namespace lsxcommon
