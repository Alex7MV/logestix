#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <memory>
#include <shared_mutex>
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

    std::shared_mutex &ResidencyMutex() override { return residency_mutex_; }

    SafetensorsModel Inspect() override;
    std::uint64_t ResidentWeightBytes() const override { return resident_bytes_; }
    const VramProfile &VramRequirements() const override { return vram_; }

protected:
    // Shared Load-preamble (triplicated in DeepSeek/GLM/Qwen): loaded_ early-out,
    // empty model_path guard, LoadTokenizerDb. Callers keep any model-specific
    // weight-pin between BeginLoad and FinishLoad.
    bool BeginLoad(const std::string &model_path);
    void FinishLoad() { loaded_ = true; }

    // ── Persistent CUDA execution context (model-unification 2026-08) ──────
    // Lazy-creates a non-blocking stream (and optionally a cuBLAS handle bound
    // to it). Idempotent: repeated calls keep the existing resources; passing
    // with_cublas=true upgrades a stream-only context in place. Every model
    // owns exactly one context for its whole lifetime (no temp-per-call
    // churn); Unload paths must call TeardownContext().
    bool EnsureContext(bool with_cublas);
    void TeardownContext();
    cudaStream_t stream() const { return ctx_stream_; }
    cublasHandle_t cublas() const { return ctx_cublas_; }

    // Persistent device-resident weight set (pinned for MoE GLM/DeepSeek; full
    // for dense Qwen). Assign in subclass Load/Unload. See spec 3.3.
    std::uint64_t resident_bytes_ = 0;

    // Conversion-time VRAM requirements; populated by subclass Load() from
    // the model's model.index.json "vram" block.
    VramProfile vram_;

    // Persistent CUDA execution context (see EnsureContext above).
    cudaStream_t ctx_stream_ = nullptr;
    cublasHandle_t ctx_cublas_ = nullptr;

    // Residency lock (see IModel::ResidencyMutex). Held shared during Infer.
    std::shared_mutex residency_mutex_;

    ~ModelBase() override = default;
};

} // namespace lsxcommon