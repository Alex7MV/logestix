#include "gemma_base_model.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cstdlib>

#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"

namespace lsxcommon {

bool GemmaBaseModel::SyntheticGeometryAllowed()
{
    static const bool allowed =
        std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr;
    return allowed;
}

// Base default: no frozen geometry. Concrete models override with the
// compile-time constants gate.
bool GemmaBaseModel::ValidateGeometry(const gemma::GemmaTopology &) const
{
    return true;
}

GemmaBaseModel::GemmaBaseModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

GemmaBaseModel::~GemmaBaseModel() { Unload(); }

bool GemmaBaseModel::Load()
{
    tokenizer_optional_ = true;
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!weights_loaded_) {
        cudaStream_t st = nullptr;
        if (cudaStreamCreate(&st) == cudaSuccess) {
            cudaError_t lrc =
                gemma::LoadGemmaPinnedWeights(cfg_.model_path, pw_, st);
            bool ok = (lrc == cudaSuccess);
            cudaStreamSynchronize(st);
            cudaStreamDestroy(st);
            weights_loaded_ = ok;
            if (!ok) {
                spdlog::error(
                    "[gemma-model] LoadGemmaPinnedWeights failed: rc={} ({})",
                    static_cast<int>(lrc), cudaGetErrorString(lrc));
                return false;
            }
            resident_bytes_ = pw_.resident_bytes;
            vram_ = pw_.top.vram;
            spdlog::info("[gemma-model] pinned weights loaded: {} layers "
                         "({} bytes resident)",
                         pw_.layers.size(), pw_.resident_bytes);
        } else {
            spdlog::error("[gemma-model] stream create failed");
            return false;
        }
    }
    FinishLoad();
    if (!ValidateGeometry(pw_.top)) {
        spdlog::error("[gemma-model] geometry validation failed for {}",
                      cfg_.model_path);
        return false;
    }
    return true;
}

gpu::InferenceResult GemmaBaseModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    cudaStream_t st = nullptr;
    const bool st_ok = (cudaStreamCreate(&st) == cudaSuccess);
    if (!st_ok) {
        r.error_msg = "cuda stream create failed";
        return r;
    }

    // Session policy: every request runs a fresh prefill+decode (the
    // generation loop always prefills its input); session_tail_ tracks the
    // last generated tail for orchestrator introspection.
    kv_.reset();
    kv_ = std::make_unique<gemma::GemmaKvCache>();
    session_tail_.clear();
    const std::vector<int32_t> &gen_input = req.input_ids;

    auto expert_db = gemma::load_gemma_expert_index(
        cfg_.model_path,
        static_cast<int>(pw_.layers.size()),
        pw_.top.error_msg.empty() ? pw_.top.num_experts : 1);

    // Capacity: prompt + generation + slack.
    const int need = static_cast<int>(gen_input.size()) +
                     std::max(1, req.max_gen_tokens) + 8;
    cudaError_t e = kv_->Allocate(pw_, need);
    if (e != cudaSuccess) {
        r.error_msg = "kv alloc failed";
        free_gemma_expert_index(expert_db);
        cudaStreamDestroy(st);
        return r;
    }

    gemma::GemmaGenerationResult gen;
    e = gemma::RunGemmaGeneration(
        st, pw_, expert_db, gen_input, req.max_gen_tokens,
        tok_db_ ? tok_db_->eos_id : -1,
        pw_.top.error_msg.empty() ? pw_.top.final_logit_softcapping : 0.f, gen,
        *kv_);
    if (e != cudaSuccess)
        spdlog::error("[gemma-model] RunGemmaGeneration rc={} ({}) gen_err='{}'",
                      static_cast<int>(e), cudaGetErrorString(e), gen.error_msg);
    free_gemma_expert_index(expert_db);

    if (e != cudaSuccess || !gen.ok) {
        r.error_msg = gen.error_msg.empty()
                          ? std::string(cudaGetErrorString(e))
                          : gen.error_msg;
        r.ok = false;
        kv_.reset();
        session_tail_.clear();
        cudaStreamDestroy(st);
        return r;
    }

    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.num_layers = static_cast<int>(pw_.layers.size());
    if (!gen.tokens.empty()) r.first_gen_tok_id = gen.tokens[0];
    r.gen_tokens = std::move(gen.tokens);
    session_tail_ = r.gen_tokens;
    r.final_logits = std::move(gen.final_logits);
    r.ttft_ms = gen.ttft_ms;
    r.total_time_ms = gen.total_ms;
    r.tokens_per_sec =
        (gen.total_ms > 0.0)
            ? (r.num_tokens / (gen.total_ms / 1000.0))
            : 0.0;
    r.eos_terminated = gen.eos_terminated;
    r.kv_slots_used = static_cast<int>(gen_input.size()) + r.num_tokens;
    r.weight_resident_bytes = resident_bytes_;
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);

    cudaStreamDestroy(st);
    return r;
}

void GemmaBaseModel::Unload()
{
    pw_.Free();
    weights_loaded_ = false;
    resident_bytes_ = 0;
    tok_db_.reset();
    kv_.reset();
    session_tail_.clear();
    loaded_ = false;
}

gpu::InferenceResult GemmaBaseModel::SelfCheck(const ModelRequest &req)
{
    // Gemma chat framing: <|turn>user\n{prompt}<turn|>\n<|turn>model\n.
    ModelRequest framed = req;
    if (tok_db_ && !prompt::BuildPromptInputIds(type_, *tok_db_,
                                                "The capital of France is",
                                                framed.input_ids)) {
        gpu::InferenceResult r;
        r.error_msg = "selfcheck prompt framing failed";
        return r;
    }
    return runtime::RunParisSelfCheck(*this, framed);
}

gpu::BenchmarkResult GemmaBaseModel::Benchmark(int num_tokens)
{
    gpu::BenchmarkResult r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    ModelRequest req;
    req.input_ids = {2}; // <bos>
    req.max_gen_tokens = num_tokens;
    auto res = Infer(req);
    r.ok = res.ok;
    r.total_time_ms = res.total_time_ms;
    r.tokens_per_sec = res.tokens_per_sec;
    r.num_tokens = res.num_tokens;
    r.num_layers = res.num_layers;
    r.resident_bytes = res.weight_resident_bytes;
    if (!res.ok) r.error_msg = res.error_msg;
    return r;
}

} // namespace lsxcommon
