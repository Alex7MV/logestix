#include "lsxcommon/qwen_base_model.h"

#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <absl/strings/ascii.h>
#include <cctype>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <filesystem>
#include <spdlog/spdlog.h>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

QwenBaseModel::QwenBaseModel(ModelInitConfig cfg)
    : cfg_(std::move(cfg)) {}

QwenBaseModel::~QwenBaseModel() { Unload(); }

bool QwenBaseModel::Load()
{
    if (loaded_) return true;
    if (cfg_.model_path.empty()) {
        spdlog::error("[qwen-model] empty model_path");
        return false;
    }
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    auto pinned = std::filesystem::path(cfg_.model_path) / "logestix" / "pinned";
    auto v = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(), arrow::io::FileMode::READ);
    auto m = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(), arrow::io::FileMode::READ);
    if (v.ok() && m.ok()) {
        auto st = load_tokenizer_metadata(*v, *m, *tok_db_);
        if (!st.ok()) {
            spdlog::warn("[qwen-model] tokenizer load failed: {}", st.ToString());
        }
    } else {
        spdlog::warn("[qwen-model] tokenizer arrows not found (OK for synthetic test)");
    }
    if (!weights_loaded_) {
        cudaStream_t st = nullptr;
        if (cudaStreamCreate(&st) == cudaSuccess) {
            bool ok = qwen::LoadQwenPinnedWeights(cfg_.model_path, pw_, st);
            cudaStreamSynchronize(st);
            cudaStreamDestroy(st);
            weights_loaded_ = ok;
            if (!ok) {
                spdlog::error("[qwen-model] LoadQwenPinnedWeights failed");
                return false;
            }
            spdlog::info("[qwen-model] pinned weights loaded: {} layers",
                         pw_.layers.size());
        } else {
            spdlog::error("[qwen-model] stream create failed");
            return false;
        }
    }
    loaded_ = true;
    ValidateGeometry(pw_.top);
    return true;
}

gpu::InferenceResult QwenBaseModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    if (req.input_ids.empty()) {
        r.error_msg = "empty input_ids";
        return r;
    }
    if (req.max_gen_tokens <= 0) {
        r.error_msg = "max_gen_tokens must be positive";
        return r;
    }

    cublasHandle_t ch = nullptr;
    cudaStream_t   st = nullptr;
    const bool ch_ok = (cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    const bool st_ok = (cudaStreamCreate(&st) == cudaSuccess);
    if (ch_ok && st_ok) cublasSetStream(ch, st);

    std::vector<int32_t> out_ids;
    std::vector<float> final_logits;
    auto run = [&]() -> cudaError_t {
        qwen::QwenGenerationResult gen;
        cudaError_t e = qwen::RunQwenGeneration(st, pw_, req.input_ids,
                                                req.max_gen_tokens, gen);
        if (e == cudaSuccess) {
            out_ids = std::move(gen.tokens);
            final_logits = std::move(gen.final_logits);
        }
        r.error_msg = std::move(gen.error_msg);
        return e;
    };

    cudaError_t e = run();
    if (e != cudaSuccess) {
        r.error_msg = std::string(cudaGetErrorString(e)) + " / " + r.error_msg;
        r.ok = false;
        if (st) cudaStreamDestroy(st);
        if (ch) cublasDestroy(ch);
        return r;
    }

    r.ok = true;
    r.num_tokens = static_cast<int>(out_ids.size());
    if (!out_ids.empty())
        r.first_gen_tok_id = out_ids[0];
    if (st) cudaStreamDestroy(st);
    if (ch) cublasDestroy(ch);
    return r;
}

void QwenBaseModel::Unload()
{
    pw_.Free();
    weights_loaded_ = false;
    tok_db_.reset();
    loaded_ = false;
}

SafetensorsModel QwenBaseModel::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

gpu::InferenceResult QwenBaseModel::SelfCheck(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    cudaStream_t st = nullptr;
    if (cudaStreamCreate(&st) != cudaSuccess) { r.error_msg = "stream"; return r; }
    qwen::QwenGenerationResult gen;
    int max_gen = std::max(2, req.max_gen_tokens);
    cudaError_t e = qwen::RunQwenGeneration(st, pw_, req.input_ids,
                                            max_gen, gen);
    cudaStreamSynchronize(st);
    cudaStreamDestroy(st);
    if (e != cudaSuccess) {
        r.error_msg = std::string(cudaGetErrorString(e)) + " / " + gen.error_msg;
        r.ok = false;
        r.selfcheck_passed = false;
        return r;
    }
    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    if (!gen.tokens.empty())
        r.first_gen_tok_id = gen.tokens[0];
    if (!gen.tokens.empty()) {
        char buf[512];
        for (int ntok : {1, 2}) {
            if (ntok > (int)gen.tokens.size()) break;
            size_t cap = sizeof(buf);
            if (arrow_tokenizer_decode(*tok_db_, gen.tokens.data(), ntok,
                                       buf, &cap) != 0)
                continue;
            std::string t(buf, cap);
            absl::StripAsciiWhitespace(&t);
            r.first_gen_text = t;
            if (r.first_gen_text == "Paris") {
                r.selfcheck_passed = true;
                break;
            }
        }
        if (!r.selfcheck_passed && r.first_gen_text.empty()) {
            size_t cap = sizeof(buf);
            if (arrow_tokenizer_decode(*tok_db_, gen.tokens.data(),
                                       std::min((int)gen.tokens.size(), 8),
                                       buf, &cap) == 0)
                r.first_gen_text = std::string(buf, cap);
        }
    }
    spdlog::info("[qwen-selfcheck] top1_tok={} text='{}' passed={}",
                 r.first_gen_tok_id, r.first_gen_text,
                 r.selfcheck_passed ? "true" : "false");
    return r;
}

ArrowTokenizerDb &QwenBaseModel::Tokenizer()
{
    return *tok_db_;
}

void QwenBaseModel::ValidateGeometry(const qwen::QwenTopology &) const {}

} // namespace lsxcommon