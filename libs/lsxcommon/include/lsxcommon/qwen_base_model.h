#pragma once

#include <memory>
#include <string>

#include "lsxcommon/model.h"
#include "lsxcommon/qwen_inference_load.h"

namespace lsxcommon {

class QwenBaseModel : public IModel {
public:
    explicit QwenBaseModel(ModelInitConfig cfg);
    ~QwenBaseModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    SafetensorsModel Inspect() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;
    ArrowTokenizerDb &Tokenizer() override;

    const std::string &Type() const override { return type_; }
    void SetType(std::string t) { type_ = std::move(t); }

protected:
    virtual void ValidateGeometry(const qwen::QwenTopology &top) const;

    ModelInitConfig cfg_;
    std::string type_;
    std::unique_ptr<ArrowTokenizerDb> tok_db_;
    qwen::QwenPinnedWeights pw_;
    bool weights_loaded_ = false;
    bool loaded_ = false;
};

} // namespace lsxcommon