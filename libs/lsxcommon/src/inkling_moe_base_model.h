#pragma once

#include <string>
#include <vector>

#include "lsxcommon/inkling_moe_base_inference_pipeline.h"
#include "lsxcommon/inkling_moe_base_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Inkling base (kType "inkling"): 66-layer hybrid MoE, hidden 6144, 64 query
// heads; 55 sliding-window layers (512, 16 kv heads) + 11 full-attention layers
// (8 kv heads). No RoPE: a learned relative-position bank (d_rel 16) replaces
// it; q/k are per-head RMS-normed and the attention scale is 1/head_dim. Four
// static short convolutions per layer (`conv(x)+x`, no normalizer). Layers 0..1
// are dense SwiGLU; layers 2..65 route top-6 of 256 experts plus 2 shared-sink
// experts (joint 8-way renorm). muP head: hidden/24 before lm_head, no softcap.
// Frozen geometry: src/inkling_moe_base_constants.h. NOT inkling2 — see the
// passport header for the differences that break a copied decoder.
class InklingMoeBaseModel final : public ModelBase {
public:
    static const std::string kType;  // "inkling"

    explicit InklingMoeBaseModel(ModelInitConfig cfg);
    ~InklingMoeBaseModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // A chat checkpoint must never see an unframed prompt: the single-prompt
    // path renders the same native user turn the conversation bridge does.
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Model Ingestion Protocol step 4b: the checkpoint's own chat_template.jinja
    // (role markers, tool_declare block, thinking-effort line, native JSON tool
    // calls; the tool-result header carries the tool name).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    // Parity diagnostic: see RunForwardTrace in the pipeline header.
    bool TraceForward(const std::vector<int32_t> &tokens,
                      const std::string &path);

    const std::string &Type() const override { return kType; }

protected:
    void TeardownWeights() override;

    inkling::base::Topology top_;
    inkling::base::PinnedWeights pw_;
    std::unique_ptr<inkling::base::ExpertDb> experts_;
    inkling::base::Session session_;
    inkling::base::Scratch scratch_;
    bool weights_loaded_ = false;
};

}  // namespace lsxcommon