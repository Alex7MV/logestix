#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

#include "lsxcommon/glm_moe_load.h"
#include "lsxcommon/lifecycle_strategy.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/runtime_tuner.h"

namespace lsxcommon {

class GlmV5Model final : public ModelBase {
public:
    static const std::string kType;
    explicit GlmV5Model(ModelInitConfig cfg);
    ~GlmV5Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return kType; }

    // GLM-5.2 chat framing (model-owned token bridge).
    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;

    // Diagnostics/test hook: the persistent pinned weight set. Valid between
    // Load() and Unload(); pointer identity is stable across Infer calls
    // (weights are loaded ONCE, spec: load-once lifecycle).
    const glm::GlmPinnedWeights &PinnedWeights() const { return pw_; }

private:
    std::uint64_t EstimateExpertMeanBytes() const;
    void TeardownWeights() override;

    glm::GlmPinnedWeights pw_;
    glm::GlmExpertDb experts_{};
    std::unique_ptr<lifecycle::RuntimeTuner> tuner_;
    std::int64_t tuner_last_hits_ = 0;
    std::int64_t tuner_last_misses_ = 0;
};

} // namespace lsxcommon
