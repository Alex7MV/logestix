#pragma once

#include <string>

#include "lsxcommon/model_base.h"

namespace lsxcommon {

class DeepSeekV4FlashModel final : public ModelBase {
public:
    static const std::string kType;
    explicit DeepSeekV4FlashModel(ModelInitConfig cfg);
    ~DeepSeekV4FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    SafetensorsModel Inspect() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
};

} // namespace lsxcommon