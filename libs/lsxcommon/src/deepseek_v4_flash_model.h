#pragma once

#include <cuda_runtime.h>

#include <string>

#include "lsxcommon/model_base.h"
#include "deepseek_inference_load.cuh"

namespace lsxcommon {

class DeepSeekV4FlashModel final : public ModelBase {
public:
    static const std::string kType;
    explicit DeepSeekV4FlashModel(ModelInitConfig cfg);
    ~DeepSeekV4FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return kType; }

private:
    gpu::PinnedWeights pw_;
    cudaStream_t stream_ = nullptr;
};

} // namespace lsxcommon
