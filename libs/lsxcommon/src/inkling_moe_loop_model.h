#pragma once

#include <string>
#include <vector>

#include "lsxcommon/inkling_moe_loop_inference_pipeline.h"
#include "lsxcommon/inkling_moe_loop_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Inkling-2-Mini (kType "inkling-2-mini"): the LOOPED inkling2 architecture —
// 24 physical MoE layers of which [6,18) run three passes per token (48 layer
// applications from 24 layers of weights), 20 sliding-window layers (1024) plus
// 4 full-attention layers, an AttnRes depth mixer (block 6), a learned
// relative-position bank instead of RoPE, per-head L2 clipping, gated attention
// with learned sinks, and 128 routed experts (top-8) with a softcapped SwiGLU.
// A separate family from Inkling-2-Small: different geometry (hidden 2304 vs
// 5120), GQA 3 vs 6, log-scale floor 8192 vs 128000, one third of the layers
// with the loop, and per-pass KV/conv cache slots. Frozen geometry:
// src/inkling_moe_loop_constants.h.
class InklingMoeLoopModel final : public ModelBase {
public:
    static const std::string kType;  // "inkling-2-mini"

    explicit InklingMoeLoopModel(ModelInitConfig cfg);
    ~InklingMoeLoopModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // A chat checkpoint must never see an unframed prompt: a raw string is out
    // of distribution, so the single-prompt path renders the same native user
    // turn the conversation bridge does.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Model Ingestion Protocol step 4b: the checkpoint's own chat_template.jinja
    // (role markers, tool_declare block, thinking-effort line, native JSON tool
    // calls, tool-result header carrying the resolved tool name).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    // Parity diagnostic: runs ONE token from position 0 and writes the residual
    // stream after every layer APPLICATION to `path` as raw little-endian FP32
    // ([CacheLayers()+1][hidden], row 0 = post-embed_norm). The logits row rides
    // along as `<path>.logits`. See RunInklingLoopForwardTrace.
    bool TraceForward(const std::vector<int32_t> &tokens,
                      const std::string &path);

    const std::string &Type() const override { return kType; }

protected:
    void TeardownWeights() override;

    inkling::InklingLoopTopology top_;
    inkling::InklingLoopPinnedWeights pw_;
    std::unique_ptr<inkling::InklingLoopExpertDb> experts_;
    inkling::InklingLoopSession session_;
    inkling::InklingLoopScratch scratch_;
    bool weights_loaded_ = false;
};

}  // namespace lsxcommon
