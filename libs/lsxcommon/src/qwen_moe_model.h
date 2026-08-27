#pragma once

// Qwen3.5-35B-A3B MoE model (kType "qwen3.5-35b-a3b"). Standalone ModelBase
// subclass (GlmV5Model / KimiK27CodeModel pattern) — NOT the dense
// QwenBaseModel: the FFN call site is a routed-expert stack, not an MLP.
//
// Frozen geometry passport is compile-time locked below; ValidateGeometry()
// cross-checks the converted index at load. LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY
// relaxes both gates for tiny synthetic fixtures.

#include <memory>

#include "lsxcommon/model_base.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_expert_cache.h"
#include "lsxcommon/qwen_moe_inference.h"
#include "lsxcommon/qwen_moe_load.h"

namespace lsxcommon {

class Qwen35MoeModel final : public ModelBase {
public:
    static const std::string kType;  // "qwen3.5-35b-a3b"

    // Frozen geometry lives once in src/qwen_moe_constants.h
    // (namespace qwen_moe::frozen); ValidateGeometry cross-checks against it.

    explicit Qwen35MoeModel(ModelInitConfig cfg);
    ~Qwen35MoeModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

    // Test/diagnostics access to persistent generation scratch.
    const qwen_moe::QwenMoeGenScratch &GenScratchForTest() const
    {
        return scratch_;
    }

private:
    bool EnsureExecutionContext();
    void TeardownWeights();

    // Runtime geometry gate: parsed index vs frozen constants.
    bool ValidateGeometry(const qwen_moe::QwenMoeTopology &top) const;

    qwen_moe::QwenMoePinnedWeights pw_;
    qwen_moe::QwenMoeExpertStore experts_;
    qwen_moe::QwenMoeExpertDeviceCache cache_{};
    qwen_moe::QwenMoeGenScratch scratch_{};
    std::unique_ptr<qwen::QwenKvCache> kv_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon
