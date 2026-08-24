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
#include "lsxcommon/qwen_moe_inference_load.h"

namespace lsxcommon {

class Qwen35MoeModel final : public ModelBase {
public:
    static const std::string kType;  // "qwen3.5-35b-a3b"

    static constexpr int kHidden = 2048;
    static constexpr int kLayers = 40;
    static constexpr int kFullAttnInterval = 4;
    static constexpr int kHeads = 16;
    static constexpr int kKVHeads = 2;
    static constexpr int kHeadDim = 256;
    static constexpr int kVocab = 248320;
    static constexpr int kNumExperts = 256;
    static constexpr int kTopK = 8;
    static constexpr int kMoeIntermediate = 512;
    static constexpr int kSharedIntermediate = 512;

    static_assert(kTopK <= kNumExperts, "top-k exceeds expert count");
    static_assert(kLayers % kFullAttnInterval == 0,
                  "hybrid interval pattern incomplete");
    static_assert(kMoeIntermediate > 0 && kSharedIntermediate > 0,
                  "bad MoE geometry");

    explicit Qwen35MoeModel(ModelInitConfig cfg);
    ~Qwen35MoeModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }

private:
    bool EnsureExecutionContext();
    void TeardownWeights();

    // Runtime geometry gate: parsed index vs frozen constants.
    bool ValidateGeometry(const qwen_moe::QwenMoeTopology &top) const;

    qwen_moe::QwenMoePinnedWeights pw_;
    qwen_moe::QwenMoeExpertStore experts_;
    qwen_moe::QwenMoeExpertDeviceCache cache_{};
    std::unique_ptr<qwen::QwenKvCache> kv_;
    std::vector<int32_t> session_tail_;
    cudaStream_t stream_ = nullptr;
};

} // namespace lsxcommon
