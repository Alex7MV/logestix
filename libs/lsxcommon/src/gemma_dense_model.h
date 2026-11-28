#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/gemma_dense_12b_constants.h"
#include "lsxcommon/gemma_dense_31b_constants.h"
#include "lsxcommon/gemma_dense_inference.h"
#include "lsxcommon/gemma_dense_load.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/prompt_framing.h"

namespace lsxcommon {

// ── The dense Gemma-4 decoder (one body, one passport per checkpoint) ──────
//
// Gemma-4 ships the same decoder at several sizes: 5:1 sliding/global
// attention with per-head QK-norm and V tied to K on the global layers, four
// RMSNorms + a per-layer scalar per layer, GeGLU, tied embed/lm_head and
// softcapped logits. ONE converter, ONE loader, ONE kernel set and THIS
// lifecycle body serve every dense member; the numbers travel as
// `GemmaDenseFrozen` so a checkpoint is only ever decoded by the passport it
// describes (the `granite-switch-4.1` shape, see GemmaDenseGeometryMismatch):
//
//   Gemma4_31BDenseModel -> gemma_dense::frozen        (60 layers, 5376 wide)
//   Gemma4_12BDenseModel -> gemma_dense_12b::frozen    (48 layers, 3840 wide)
//
// A shared copy of the geometry is what would make one sibling decode a
// plausible-but-wrong model, so `ValidateGeometry` compares the converted tree
// against the instance's passport instead of against a namespace.
class GemmaDenseModel : public ModelBase {
public:
    GemmaDenseModel(ModelInitConfig cfg, gemma_dense::GemmaDenseFrozen frozen);
    ~GemmaDenseModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // Gemma-4 framing — every member of the family ships a tokenizer.json and
    // a chat_template.jinja BYTE-IDENTICAL to the others, so the family framing
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
    // not match THIS checkpoint's frozen expectations (synthetic tests bypass
    // via SyntheticGeometryAllowed()).
    bool ValidateGeometry(const gemma_dense::GemmaDenseTopology &top) const;

    void TeardownWeights() override;

    gemma_dense::GemmaDenseFrozen frozen_;
    gemma_dense::GemmaDensePinnedWeights pw_;
    bool weights_loaded_ = false;

    // Incremental-session state. The generation loop always prefills its input
    // (capacity-grown workspace); the tail is tracked for orchestrator
    // introspection. Both are created/reset/published INSIDE the residency lock.
    std::unique_ptr<gemma_dense::GemmaDenseKvCache> kv_;
    std::vector<int32_t> session_tail_;

    gemma_dense::GemmaDenseGenContext *gen_ctx_ = nullptr;
};

// Gemma-4-31B-it (dense): 60 layers / 5376 hidden / 32 Q / 16 sliding KV +
// 4 global KV / 21504 MLP. Frozen geometry: gemma_dense_31b_constants.h.
class Gemma4_31BDenseModel final : public GemmaDenseModel {
public:
    static const std::string kType;

    explicit Gemma4_31BDenseModel(ModelInitConfig cfg);
    ~Gemma4_31BDenseModel() override;

    const std::string &Type() const override { return kType; }
};

} // namespace lsxcommon
