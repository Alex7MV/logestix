#pragma once

#include <memory>
#include <string>

#include "lsxcommon/model.h"

namespace lsxcommon {

class DeepSeekV4FlashModel final : public IModel {
public:
    explicit DeepSeekV4FlashModel(ModelInitConfig cfg);
    ~DeepSeekV4FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    void Unload() override;
    SafetensorsModel Inspect() override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    const std::string &Type() const override { return kType; }
    ArrowTokenizerDb &Tokenizer() override;

    static constexpr const char *kType = "deepseek-v4-flash";

private:
    ModelInitConfig cfg_;
    std::unique_ptr<ArrowTokenizerDb> tok_db_;
    bool loaded_ = false;
};

} // namespace lsxcommon