#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

#include "lsxcommon/kimi_moe_expert_cache.h"
#include "lsxcommon/kimi_moe_load.h"
#include "lsxcommon/lifecycle_strategy.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

class KimiK27CodeModel final : public ModelBase {
public:
    static const std::string kType;   // "kimi-k2.7-code"
    explicit KimiK27CodeModel(ModelInitConfig cfg);
    ~KimiK27CodeModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return kType; }

    // Diagnostics/test hook: persistent pinned weights (load-once lifecycle).
    const kimi::KimiPinnedWeights &PinnedWeights() const { return pw_; }

private:
    void TeardownWeights() override;

    kimi::KimiPinnedWeights pw_;
    kimi::KimiExpertDb experts_{};
    kimi::KimiExpertCacheState expert_cache_{};
    Lease permanent_lease_;
};

} // namespace lsxcommon
