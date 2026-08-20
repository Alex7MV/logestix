#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lsxcommon/model_base.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"

namespace lsxcommon {

class QwenBaseModel : public ModelBase {
public:
    explicit QwenBaseModel(ModelInitConfig cfg);
    ~QwenBaseModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    SafetensorsModel Inspect() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    gpu::BenchmarkResult Benchmark(int num_tokens = 2048) override;

    const std::string &Type() const override { return type_; }
    void SetType(std::string t) { type_ = std::move(t); }
    std::uint64_t ResidentWeightBytes() const override { return pw_.resident_bytes; }

protected:
    virtual void ValidateGeometry(const qwen::QwenTopology &top) const;

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
