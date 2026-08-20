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

        // Incremental session policy (spec item 15): a request whose input_ids
    // exactly match the previous call's generated tail continues the session
    // KV cache from {kv_->last_tok}; anything else starts a fresh session.
    const bool is_continuation =
        kv_ && kv_->allocated() && !session_tail_.empty() &&
        req.input_ids == session_tail_;
    if (!is_continuation) {
        kv_.reset();
        kv_ = std::make_unique<qwen::QwenKvCache>();
        session_tail_.clear();
    }
    const std::vector<int32_t> gen_input =
        is_continuation ? std::vector<int32_t>{kv_->last_tok} : req.input_ids;

    qwen::QwenGenerationResult gen;
    auto run = [&]() -> cudaError_t {
        const int eos_id = tok_db_ ? tok_db_->eos_id : -1;
        return qwen::RunQwenGeneration(st, pw_, gen_input,
                                       req.max_gen_tokens, eos_id, gen,
                                       lsxcommon::sample::DefaultSampler(),
                                       kv_.get());
    };

    cudaError_t e = run();
    if (e != cudaSuccess) {
        r.error_msg = std::string(cudaGetErrorString(e)) + " / " + gen.error_msg;
        r.ok = false;
        // A failed continuation may have advanced kv_rows mid-stream; drop the
        // session so the next call starts fresh.
        kv_.reset();
        session_tail_.clear();
        if (st) cudaStreamDestroy(st);
        if (ch) cublasDestroy(ch);
        return r;
    }

    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    if (!gen.tokens.empty())
        r.first_gen_tok_id = gen.tokens[0];
    r.gen_tokens = std::move(gen.tokens);
    session_tail_ = r.gen_tokens;
    r.final_logits = std::move(gen.final_logits);
    r.ttft_ms = gen.ttft_ms;
    r.total_time_ms = gen.total_ms;
    r.tokens_per_sec = (gen.total_ms > 0.0)
                           ? (r.num_tokens / (gen.total_ms / 1000.0))
                           : 0.0;
    r.eos_terminated = gen.eos_terminated;
    // Occupancy (spec item 15): the incremental cache holds prompt + generated
    // rows once the pending last-token row is counted (kv_rows + 1). The
    // fallback (non-incremental) mirrors the same prompt + gen semantic.
    r.kv_slots_used =
        (kv_ && kv_->allocated())
            ? kv_->kv_rows + 1
            : r.num_tokens + static_cast<int>(req.input_ids.size());
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
    kv_.reset();
    session_tail_.clear();
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

gpu::BenchmarkResult QwenBaseModel::Benchmark(int num_tokens)
{
    gpu::BenchmarkResult r;
    if (!loaded_) { r.error_msg = "model not loaded"; return r; }
    const std::vector<int32_t> prompt = {0};
    // Mirror Infer: open a stream, run RunQwenGeneration, map the result.
    qwen::QwenGenerationResult gr;
    cudaStream_t st = nullptr;
    cublasHandle_t ch = nullptr;
    const bool ch_ok = (cublasCreate(&ch) == CUBLAS_STATUS_SUCCESS);
    const bool st_ok = (cudaStreamCreate(&st) == cudaSuccess);
    if (ch_ok && st_ok) cublasSetStream(ch, st);
    if (!ch_ok || !st_ok) {
        r.error_msg = "cuda/cublas init failed";
        if (st) cudaStreamDestroy(st);
        if (ch) cublasDestroy(ch);
        return r;
    }
    const int eos_id = tok_db_ ? tok_db_->eos_id : -1;
    cudaError_t e = qwen::RunQwenGeneration(st, pw_, prompt, num_tokens,
                                            eos_id, gr);
    if (st) cudaStreamDestroy(st);
    if (ch) cublasDestroy(ch);
    if (e != cudaSuccess) {
        r.error_msg = std::string(cudaGetErrorString(e)) + " / " + gr.error_msg;
        return r;
    }
    r.ok = true;
    r.num_tokens = static_cast<int>(gr.tokens.size());
    r.total_time_ms = gr.total_ms;
    r.tokens_per_sec = gr.total_ms > 0.0
        ? (static_cast<double>(r.num_tokens) / gr.total_ms) * 1000.0 : 0.0;
    r.resident_bytes = gr.resident_bytes;
    return r;
}

void QwenBaseModel::ValidateGeometry(const qwen::QwenTopology &) const {}

} // namespace lsxcommon
