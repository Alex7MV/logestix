#pragma once

#include <memory>
#include <string>
#include <utility>

#include "lsxcommon/model.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

// Thin shared base for every lsxcommon model. Owns the tokenizer Arrow db
// (Item 1: tokenizer-db loading) and the canonical Infer/SelfCheck request
// guards (Item 5: infer-preamble guards). All three models migrate onto it.
// Non-copyable / non-movable: model instances own GPU residency (pinned
// tensors, KV caches), so cloning the base would duplicate weights illegally.
class ModelBase : public IModel {
public:
    ModelBase(const ModelBase &) = delete;
    ModelBase &operator=(const ModelBase &) = delete;
    ModelBase(ModelBase &&) = delete;
    ModelBase &operator=(ModelBase &&) = delete;

    explicit ModelBase(ModelInitConfig cfg) : cfg_(std::move(cfg)) {}

    ModelInitConfig cfg_;
    std::unique_ptr<ArrowTokenizerDb> tok_db_;
    bool loaded_ = false;
    bool tokenizer_optional_ = false;  // Qwen: warn+continue when arrows absent

    bool LoadTokenizerDb(const std::string &model_path);
    bool ValidateInferRequest(const ModelRequest &req, gpu::InferenceResult &r);

    ArrowTokenizerDb &Tokenizer() override { return *tok_db_; }

    SafetensorsModel Inspect() override;
    std::uint64_t ResidentWeightBytes() const override { return resident_bytes_; }

protected:
    // Shared Load-preamble (triplicated in DeepSeek/GLM/Qwen): loaded_ early-out,
    // empty model_path guard, LoadTokenizerDb. Callers keep any model-specific
    // weight-pin between BeginLoad and FinishLoad.
    bool BeginLoad(const std::string &model_path);
    void FinishLoad() { loaded_ = true; }

    // Persistent device-resident weight set (pinned for MoE GLM/DeepSeek; full
    // for dense Qwen). Assign in subclass Load/Unload. See spec 3.3.
    std::uint64_t resident_bytes_ = 0;

    ~ModelBase() override = default;
};

} // namespace lsxcommon