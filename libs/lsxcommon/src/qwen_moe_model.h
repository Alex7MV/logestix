#pragma once

// Qwen3.5-35B-A3B MoE model (kType "qwen3.5-35b-a3b") and the shared qwen_moe
// runtime graph (QwenMoeModelBase) reused by the geometry-identical
// Qwen3.6-35B-A3B sibling (qwen_moe_36_model.h). Both are standalone
// ModelBase subclasses (GlmV5Model / KimiK27CodeModel pattern) — NOT the
// dense QwenBaseModel: the FFN call site is a routed-expert stack, not an
// MLP.
//
// Frozen geometry passports are compile-time locked per model
// (src/qwen_moe_constants.h, src/qwen_moe_36_constants.h);
// ValidateGeometry() cross-checks the converted index at load.
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY relaxes both gates for tiny synthetic
// fixtures.

#include <memory>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_expert_cache.h"
#include "lsxcommon/qwen_moe_inference_pipeline.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/qwen_moe_load.h"

namespace lsxcommon {

// Shared MoE-model lifecycle + generation plumbing (3.5 / 3.6). Owns the
// weight graphs and delegates the FFN call site to the qwen_moe family
// drivers. Leaf models provide kType, their own frozen-geometry gate and an
// Infer that opens LsxFastMathScope with the instance's own config before
// RunInferShared (model independence invariant).
class QwenMoeModelBase : public ModelBase {
public:
    // Test/diagnostics access to persistent generation scratch.
    const qwen_moe::QwenMoeGenScratch &GenScratchForTest() const
    {
        return scratch_;
    }

    bool Load() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    // Tool/function-calling bridge (step 4b): Qwen3.5/3.6 ChatML XML, via the
    // codec shared with the dense QwenBaseModel (qwen_tool_format.h).
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

protected:
    explicit QwenMoeModelBase(ModelInitConfig cfg);

    void TeardownWeights() override;

    // Runtime geometry gate: parsed index vs the leaf model's frozen
    // passport.
    virtual bool ValidateGeometry(
        const qwen_moe::QwenMoeTopology &top) const = 0;

    // Shared Infer body. The caller MUST have opened its own
    // LsxFastMathScope(cfg_.fast_math, ...) first (per-model policy).
    gpu::InferenceResult RunInferShared(const ModelRequest &req);

    qwen_moe::QwenMoePinnedWeights pw_;
    moe::MoEExpertStore experts_;
    qwen_moe::QwenMoeExpertDeviceCache cache_{};
    qwen_moe::QwenMoeGenScratch scratch_{};
    std::unique_ptr<qwen::KvCache> kv_;
    std::vector<int32_t> session_tail_;
};

class Qwen35MoeModel final : public QwenMoeModelBase {
public:
    static const std::string kType;  // "qwen3.5-35b-a3b"

    // Frozen geometry lives once in src/qwen_moe_constants.h
    // (namespace qwen_moe::frozen); ValidateGeometry cross-checks against it.

    explicit Qwen35MoeModel(ModelInitConfig cfg);
    ~Qwen35MoeModel() override;

    gpu::InferenceResult Infer(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

private:
    bool ValidateGeometry(const qwen_moe::QwenMoeTopology &top) const override;
};

} // namespace lsxcommon
