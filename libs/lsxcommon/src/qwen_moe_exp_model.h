#pragma once

// Qwen3.8-Flash-Next model (kType "qwen3.8-flash-next"). Standalone ModelBase
// subclass (Qwen35MoeModel pattern). Frozen geometry passport in
// src/qwen_moe_exp_constants.h; ValidateGeometry cross-checks at load.

#include <memory>
#include <vector>

#include "lsxcommon/model_base.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_moe_exp_inference.h"
#include "lsxcommon/qwen_moe_exp_load.h"

namespace lsxcommon {

class Qwen38FlashNextModel final : public ModelBase {
public:
    static const std::string kType;  // "qwen3.8-flash-next"

    explicit Qwen38FlashNextModel(ModelInitConfig cfg);
    ~Qwen38FlashNextModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return kType; }

private:
    void TeardownWeights() override;
    bool ValidateGeometry(const qwen_moe_exp::Qwen4ExpTopology &top) const;

    qwen_moe_exp::ExpPinnedWeights pw_;
    qwen_moe_exp::ExpExpertStore experts_;
    std::vector<qwen_moe_exp::ExpPleShardView> ple_shards_;
    gpu::ExpertDeviceCache cache_{};
    qwen_moe_exp::ExpPleState ple_state_{};
    std::unique_ptr<qwen::QwenKvCache> kv_;
    std::vector<qwen_moe_exp::ExpQsaIndexerState> qsa_;
    std::vector<int32_t> session_tail_;
};

} // namespace lsxcommon
