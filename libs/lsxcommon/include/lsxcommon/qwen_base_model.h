#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "lsxcommon/model_base.h"
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

    const std::string &Type() const override { return type_; }
    void SetType(std::string t) { type_ = std::move(t); }
    std::uint64_t ResidentWeightBytes() const override { return pw_.resident_bytes; }

protected:
    virtual void ValidateGeometry(const qwen::QwenTopology &top) const;

    std::string type_;
    qwen::QwenPinnedWeights pw_;
    bool weights_loaded_ = false;
};

} // namespace lsxcommon
