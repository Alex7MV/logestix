#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/muse_dense_constants.h"
#include "lsxcommon/muse_dense_inference.h"
#include "lsxcommon/muse_dense_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Muse-Glimmer-30B: dense gated-attention decoder (52 layers / 6656 hidden /
// 32 Q / 2 KV x 128 / 19968 SwiGLU MLP), [sliding x3, full-NoPE] x 13 pattern,
// centered norms, weightless embed norm, untied FP16 head with softcapped
// logits. Frozen geometry lives in muse_dense_constants.h (passport,
// model-independence invariant).
class MuseGlimmer30BModel final : public ModelBase {
public:
    static const std::string kType;

    explicit MuseGlimmer30BModel(ModelInitConfig cfg);
    ~MuseGlimmer30BModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // ATEM framing (stages 4/4b): native channel markers + tool codec.
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
    // Hard geometry gate (protocol step 3): false when the loaded topology
    // does not match the frozen compile-time constants (synthetic tests
    // bypass via SyntheticGeometryAllowed()).
    bool ValidateGeometry(const muse_dense::MuseDenseTopology &top) const;

    void TeardownWeights() override;

    muse_dense::MuseDensePinnedWeights pw_;
    bool weights_loaded_ = false;

    // Incremental-session state. The generation loop always prefills its
    // input (capacity-grown workspace); the tail is tracked for orchestrator
    // introspection. Both are created/reset/published INSIDE the residency
    // lock.
    std::unique_ptr<muse_dense::MuseDenseKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon