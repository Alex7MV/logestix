#include "lsxcommon/model_base.h"

#include <absl/strings/str_format.h>
#include <arrow/io/file.h>
#include <filesystem>

#include "lsxcommon/log.h"

namespace lsxcommon {

bool ModelBase::LoadTokenizerDb(const std::string &model_path)
{
    if (tok_db_) return true;
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    auto pinned = std::filesystem::path(model_path) / "logestix" / "pinned";
    // MoE layouts keep their own root (e.g. logestix-qwen35moe/pinned).
    if (!std::filesystem::exists(pinned)) {
        for (const char *alt : {"logestix-qwen35moe", "logestix-glm",
                                "logestix-kimi", "logestix-gemma"}) {
            auto cand = std::filesystem::path(model_path) / alt / "pinned";
            if (std::filesystem::exists(cand)) {
                pinned = cand;
                break;
            }
        }
    }
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (!v.ok() || !m.ok()) {
        if (tokenizer_optional_) {
            log::warn("[model] tokenizer arrows not found (OK for synthetic test)");
            return true;
        }
        log::error(
            absl::StrFormat("[model] cannot open tokenizer arrows in {}",
                            pinned.string()));
        tok_db_.reset();
        return false;
    }
    auto st = load_tokenizer_metadata(*v, *m, *tok_db_);
    if (!st.ok()) {
        if (tokenizer_optional_) {
            log::warn(absl::StrFormat("[model] tokenizer load failed: {}",
                                      st.ToString()));
            return true;
        }
        log::error(
            absl::StrFormat("[model] load_tokenizer_metadata: {}",
                            st.ToString()));
        tok_db_.reset();
        return false;
    }
    return true;
}

bool ModelBase::BeginLoad(const std::string &model_path)
{
    if (loaded_) return true;
    if (model_path.empty()) {
        log::error("[model] empty model_path");
        return false;
    }
    return LoadTokenizerDb(model_path);
}

SafetensorsModel ModelBase::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

bool ModelBase::ValidateInferRequest(const ModelRequest &req, gpu::InferenceResult &r)
{
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return false;
    }
    if (req.input_ids.empty()) {
        r.error_msg = "empty input_ids";
        return false;
    }
    if (req.max_gen_tokens <= 0) {
        r.error_msg = "max_gen_tokens must be positive";
        return false;
    }
    return true;
}

// ── Persistent CUDA execution context (model-unification 2026-08) ───────────

bool ModelBase::EnsureContext(bool with_cublas)
{
    if (!ctx_stream_ &&
        cudaStreamCreateWithFlags(&ctx_stream_, cudaStreamNonBlocking) !=
            cudaSuccess) {
        ctx_stream_ = nullptr;
        return false;
    }
    if (with_cublas && !ctx_cublas_ &&
        cublasCreate(&ctx_cublas_) != CUBLAS_STATUS_SUCCESS) {
        ctx_cublas_ = nullptr;
        return false;
    }
    if (ctx_cublas_) cublasSetStream(ctx_cublas_, ctx_stream_);
    return true;
}

void ModelBase::TeardownContext()
{
    if (ctx_cublas_) {
        cublasDestroy(ctx_cublas_);
        ctx_cublas_ = nullptr;
    }
    if (ctx_stream_) {
        cudaStreamDestroy(ctx_stream_);
        ctx_stream_ = nullptr;
    }
}

} // namespace lsxcommon
