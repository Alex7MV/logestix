#pragma once

#include <string>
#include <vector>

#include "lsxcommon/inkling_moe_loop_inference_pipeline.h"
#include "lsxcommon/inkling_moe_loop_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// TML-d24-varloop-research (kType "tml-d24-varloop-research"): the VARLOOP
// sibling of Inkling-2-Mini. Same architecture and same artifact layout — 24
// physical MoE layers of which [6,18) run `passes` times, 20 sliding-window
// layers (1024) plus 4 full-attention layers, an AttnRes depth mixer (block 6),
// a learned relative-position bank instead of RoPE, per-head L2 clipping, gated
// attention with learned sinks, 128 routed experts (top-8) with a softcapped
// SwiGLU — and ONE new capability: `loop_config.is_varloop` means the pass count
// is chosen at inference time instead of being fixed by the checkpoint.
//
// What the row owns (independence: its own passport, own format tag, own
// class):
//   * the default count is the one the checkpoint DECLARES (3 here), and the
//     operator override (`LSX_INKLING_VARLOOP_PASSES`) clamps into the trained
//     range 1..5 — the model card's range, which the config does not carry;
//   * the KV/conv slab is sized for the ceiling (72 slots), so changing the
//     count never re-allocates and never moves a slot.
// What it reuses deliberately: the parent's loader, kernels, generation
// pipeline, and the family's verified tool codec (the template is byte-identical
// to Inkling-2-Mini's). Frozen geometry: src/inkling_moe_varloop_constants.h.
class InklingMoeVarloopModel final : public ModelBase {
public:
    static const std::string kType;  // "tml-d24-varloop-research"

    explicit InklingMoeVarloopModel(ModelInitConfig cfg);
    ~InklingMoeVarloopModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // A chat checkpoint must never see an unframed prompt.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Model Ingestion Protocol step 4b: the checkpoint's own chat_template.jinja
    // through the family's shared codec (role markers, tool_declare block,
    // thinking-effort line, native JSON tool calls).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    const std::string &Type() const override { return kType; }

    // ── observability of the pass knob (what the engine actually runs) ─────
    // The ACTIVE pass count of this loaded instance: the declared default, or
    // the operator's override clamped into the trained range.
    int ActiveLoopPasses() const { return top_.loop_passes; }
    // KV/conv slots in the slab: the trained ceiling's, not the active count's.
    int CacheSlotCount() const { return top_.CacheLayers(); }
    // Layer applications a token performs at the active count.
    int LayerApplications() const { return top_.LayerApplications(); }

protected:
    void TeardownWeights() override;

    inkling::InklingLoopTopology top_;
    inkling::InklingLoopPinnedWeights pw_;
    std::unique_ptr<inkling::InklingLoopExpertDb> experts_;
    inkling::InklingLoopSession session_;
    inkling::InklingLoopScratch scratch_;
    bool weights_loaded_ = false;
};

// The varloop row's frozen cross-check: the geometry its own passport pins, the
// capability flag, the trained pass range and the slab ceiling. Relaxed under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY. Exposed so the model test can prove that a
// fixed-pass tree (varloop false) is refused.
bool VarloopTopologyMatches(const inkling::InklingLoopTopology &top);

}  // namespace lsxcommon
