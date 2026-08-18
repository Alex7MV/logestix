#include "lsxcommon/qwen_base_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <spdlog/spdlog.h>

#include "lsxcommon/model_runtime.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

QwenBaseModel::QwenBaseModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

QwenBaseModel::~QwenBaseModel() { Unload(); }

bool QwenBaseModel::Load()
{
    if (loaded_) return true;
    if (cfg_.model_path.empty()) {
        spdlog::error("[qwen-model] empty model_path");
        return false;
    }
    tokenizer_optional_ = true;
    if (!LoadTokenizerDb(cfg_.model_path)) return false;
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
            spdlog::info("[qwen-model] pinned weights loaded: {} layers "
                         "({} bytes resident)",
                         pw_.layers.size(), pw_.resident_bytes);
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
    if (!ValidateInferRequest(req, r)) return r;

    cublasHandle_t ch = nullptr;
    cudaStream_t   st = nullptr;
    const bool ch_ok = (cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    const bool st_ok = (cudaStreamCreate(&st) == cudaSuccess);
    if (ch_ok && st_ok) cublasSetStream(ch, st);

    qwen::QwenGenerationResult gen;
    auto run = [&]() -> cudaError_t {
        const int eos_id = tok_db_ ? tok_db_->eos_id : -1;
        return qwen::RunQwenGeneration(st, pw_, req.input_ids,
                                       req.max_gen_tokens, eos_id, gen);
    };

    cudaError_t e = run();
    if (e != cudaSuccess) {
        r.error_msg = std::string(cudaGetErrorString(e)) + " / " + gen.error_msg;
        r.ok = false;
        if (st) cudaStreamDestroy(st);
        if (ch) cublasDestroy(ch);
        return r;
    }

    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    if (!gen.tokens.empty())
        r.first_gen_tok_id = gen.tokens[0];
    r.gen_tokens = std::move(gen.tokens);
    r.final_logits = std::move(gen.final_logits);
    r.ttft_ms = gen.ttft_ms;
    r.total_time_ms = gen.total_ms;
    r.tokens_per_sec = (gen.total_ms > 0.0)
                           ? (r.num_tokens / (gen.total_ms / 1000.0))
                           : 0.0;
    r.eos_terminated = gen.eos_terminated;
    r.kv_slots_used = r.num_tokens + static_cast<int>(req.input_ids.size());
    r.weight_resident_bytes = gen.resident_bytes;
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);

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
    return runtime::RunParisSelfCheck(*this, req);
}

void QwenBaseModel::ValidateGeometry(const qwen::QwenTopology &) const {}

} // namespace lsxcommon
