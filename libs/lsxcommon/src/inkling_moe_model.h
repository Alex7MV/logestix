#pragma once

#include <string>

#include "lsxcommon/inkling_moe_inference_pipeline.h"
#include "lsxcommon/inkling_moe_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

// Inkling-2-Small (kType "inkling-2-small"): 48-layer hybrid MoE with a 1024
// sliding window, 8 full-attention layers, an AttnRes depth mixer (block 6),
// learned relative-position bias instead of RoPE, per-head L2 clipping, gated
// attention with learned sinks, and 384 routed experts (top-8) with a
// softcapped SwiGLU. Frozen geometry: src/inkling_moe_constants.h.
class InklingMoeModel final : public ModelBase {
public:
    static const std::string kType;  // "inkling-2-small"

    explicit InklingMoeModel(ModelInitConfig cfg);
    ~InklingMoeModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

protected:
    void TeardownWeights() override;

    inkling::InklingTopology top_;
    inkling::InklingPinnedWeights pw_;
    std::unique_ptr<inkling::InklingExpertDb> experts_;
    inkling::InklingSession session_;
    inkling::InklingScratch scratch_;
    bool weights_loaded_ = false;
};

}  // namespace lsxcommon
