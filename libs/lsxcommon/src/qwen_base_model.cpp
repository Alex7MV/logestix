#include "lsxcommon/qwen_base_model.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/tokenizer.h"

#include <absl/strings/str_cat.h>
#include <cstdlib>

namespace lsxcommon {

bool QwenBaseModel::SyntheticGeometryAllowed()
{
    static const bool allowed =
        std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr;
    return allowed;
}

QwenBaseModel::QwenBaseModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

QwenBaseModel::~QwenBaseModel() { Unload(); }

bool QwenBaseModel::Load()
{
    tokenizer_optional_ = true;
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!weights_loaded_) {
        cudaStream_t st = nullptr;
        if (cudaStreamCreate(&st) == cudaSuccess) {
            bool ok = qwen::LoadQwenPinnedWeights(cfg_.model_path, pw_, st);
            cudaStreamSynchronize(st);
            cudaStreamDestroy(st);
            weights_loaded_ = ok;
            if (!ok) {
                lsxcommon::log::error(
                    "[qwen-model] LoadQwenPinnedWeights failed");
                return false;
            }
            // Dense residency (spec 3.3): the entire model is resident after
            // Load(), so ResidentWeightBytes() reports the full persistent set.
            resident_bytes_ = pw_.resident_bytes;
            vram_ = pw_.top.vram;
            lsxcommon::log::info(absl::StrCat(
                "[qwen-model] pinned weights loaded: ", pw_.layers.size(),
                " layers (", pw_.resident_bytes, " bytes resident)"));
        } else {
            lsxcommon::log::error("[qwen-model] stream create failed");
            return false;
        }
    }
    FinishLoad();
    if (!ValidateGeometry(pw_.top)) {
        lsxcommon::log::error(absl::StrCat(
            "[qwen-model] geometry validation failed for ", cfg_.model_path));
        return false;
    }
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
    const qwen::SessionDecision session = qwen::DecideSession(
        kv_ && kv_->allocated(),
        kv_ ? kv_->last_tok : -1, session_tail_, req.input_ids);
    if (!session.is_continuation) {
        kv_.reset();
        kv_ = std::make_unique<qwen::QwenKvCache>();
        session_tail_.clear();
    }
    const std::vector<int32_t> &gen_input = session.gen_input;

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

    // Occupancy (spec item 15): the incremental cache holds prompt + generated
    // rows once the pending last-token row is counted (kv_rows + 1). The
    // fallback (non-incremental) mirrors the same prompt + gen semantic.
    const int kv_slots =
        (kv_ && kv_->allocated())
            ? kv_->kv_rows + 1
            : static_cast<int>(gen.tokens.size()) +
                  static_cast<int>(req.input_ids.size());
    runtime::FillInferenceResult(
        r, {.tokens = &gen.tokens,
            .final_logits = &gen.final_logits,
            .ttft_ms = gen.ttft_ms,
            .total_ms = gen.total_ms,
            .error_msg = &gen.error_msg,
            .eos_terminated = gen.eos_terminated,
            .kv_slots_used = kv_slots,
            .first_maxv = 0.f,
            .resident_bytes = gen.resident_bytes},
        pw_.top.num_layers);
    session_tail_ = r.gen_tokens;
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
    resident_bytes_ = 0;
    tok_db_.reset();
    kv_.reset();
    session_tail_.clear();
    loaded_ = false;
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
    runtime::FillBenchmarkResult(r, gr.total_ms,
                                 static_cast<int>(gr.tokens.size()),
                                 /*num_layers=*/pw_.top.num_layers,
                                 gr.resident_bytes);
    return r;
}

bool QwenBaseModel::ValidateGeometry(const qwen::QwenTopology &) const { return true; }

} // namespace lsxcommon
