#pragma once

#include <memory>
#include <string>

#include "lsxcommon/model.h"

namespace lsxcommon {

class GlmV5Model final : public IModel {
public:
    explicit GlmV5Model(ModelInitConfig cfg);
    ~GlmV5Model() override;

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
    bool loaded_ = false;
};

} // namespace lsxcommon