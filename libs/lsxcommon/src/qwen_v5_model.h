#pragma once

#include <memory>
#include <string>

#include "lsxcommon/model.h"
#include "lsxcommon/qwen_inference_load.h"

namespace lsxcommon {

class QwenV5Model final : public IModel {
public:
    explicit QwenV5Model(ModelInitConfig cfg);
    ~QwenV5Model() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    SafetensorsModel Inspect() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    ArrowTokenizerDb &Tokenizer() override;

    static const std::string kType;

private:
    ModelInitConfig cfg_;
    std::unique_ptr<ArrowTokenizerDb> tok_db_;
    qwen::QwenPinnedWeights pw_;
    bool weights_loaded_ = false;
    bool loaded_ = false;
};

} // namespace lsxcommon