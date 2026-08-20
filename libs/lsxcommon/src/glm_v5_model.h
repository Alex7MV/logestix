#pragma once

#include <string>

#include "lsxcommon/glm_inference_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

class GlmV5Model final : public ModelBase {
public:
    static const std::string kType;
    explicit GlmV5Model(ModelInitConfig cfg);
    ~GlmV5Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return kType; }

private:
    glm::GlmPinnedWeights pw_;
};

} // namespace lsxcommon