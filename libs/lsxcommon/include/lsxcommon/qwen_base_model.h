#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/qwen_dense_inference_pipeline.h"
#include "lsxcommon/lsx_attn_inference.h"
#include "lsxcommon/qwen_inference_load.h"

namespace lsxcommon {

class QwenBaseModel : public ModelBase {
public:
    explicit QwenBaseModel(ModelInitConfig cfg);
    ~QwenBaseModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return type_; }
    void SetType(std::string t) { type_ = std::move(t); }

    // Tool/function-calling bridge (Model Ingestion Protocol step 4b). Qwen3.5/
    // 3.6 use the ChatML template with a native XML tool syntax (verified
    // against the checkpoint chat_template.jinja); the codec lives in
    // qwen_tool_format.h and is shared with the MoE QwenMoeModelBase.
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

protected:
    // Hard geometry gate (Model Ingestion Protocol step 3): returns false when
    // the loaded topology does not match the model's frozen compile-time
    // constants; Load() fails in that case. Test-only opt-out: when
    // LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY is set, the gate is bypassed
    // (ModelBase::SyntheticGeometryAllowed) so synthetic tiny fixtures can
    // exercise converter/loader/kernel paths. Production binaries never set
    // it and keep the hard gate.
    virtual bool ValidateGeometry(const qwen::QwenTopology &top) const;

    // Canonical ModelBase teardown: pinned weights, session state, context.
    void TeardownWeights() override;

    std::string type_;
    qwen::QwenPinnedWeights pw_;
    bool weights_loaded_ = false;

    // Incremental session state (spec item 15): a fresh request (input_ids not
    // equal to the previous call's generated tail) resets the cache and runs a
    // full prefill+decode; a continuation request reuses it with prompt =
    // {kv_->last_tok}.
    std::unique_ptr<qwen::QwenKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon
