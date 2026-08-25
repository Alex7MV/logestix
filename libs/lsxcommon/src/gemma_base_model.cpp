#include "gemma_base_model.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <string>

#include <absl/strings/str_format.h>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/prompt_framing.h"
#include "fast_load_internal.h"

namespace lsxcommon {

// Gemma-4 canonical chat template (chat_template.jinja, generation prompt;
// enable_thinking defaults to false -> the EMPTY thinking channel is part of
// the generation prompt):
//   {{ bos_token }}<|turn>user\n{prompt}<turn|>\n<|turn>model\n
//   <|channel>thought\n<channel|>
// Sentinel strings verified against the real checkpoint tokenizer.json:
// <|turn>=105, <turn|>=106, "\n"=107, <|channel>=100, thought=45518,
// <channel|>=101; "user"/"model" are single vocab keys.
namespace gemma {
const prompt::FramingDescriptor kFraming = {
    {"<|turn>", "user", "\n"},
    {"<turn|>", "\n", "<|turn>", "model", "\n",
     "<|channel>", "thought", "\n", "<channel|>"},
    3, 9,
    /*leading_bos=*/true,
};
bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                         std::string_view raw_prompt,
                         std::vector<int32_t> &out_ids)
{
    return prompt::ApplyFraming(kFraming, tok, raw_prompt, out_ids);
}
} // namespace gemma

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
        if (EnsureContext(/*with_cublas=*/false)) {
            cudaError_t lrc = gemma::LoadGemmaPinnedWeights(
                cfg_.model_path, pw_, stream());
            fastload::StreamDrain{stream()};
            bool ok = (lrc == cudaSuccess);
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
    if (!expert_db_ready_) {
        expert_db_ = gemma::load_gemma_expert_index(
            cfg_.model_path, static_cast<int>(pw_.layers.size()),
            pw_.top.error_msg.empty() ? pw_.top.num_experts : 1);
        expert_db_ready_ = true;
        spdlog::info("[gemma-model] expert index ready: {}/{} files parsed",
                     expert_db_.parse_ok,
                     expert_db_.num_layers * expert_db_.experts_per_layer);
    }
    return true;
}

gpu::InferenceResult GemmaBaseModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    if (!EnsureContext(/*with_cublas=*/false)) {
        r.error_msg = "cuda stream create failed";
        return r;
    }
    cudaStream_t st = stream();

    // Session policy: every request runs a fresh prefill+decode (the
    // generation loop always prefills its input); session_tail_ tracks the
    // last generated tail for orchestrator introspection.
    kv_.reset();
    kv_ = std::make_unique<gemma::GemmaKvCache>();
    session_tail_.clear();
    const std::vector<int32_t> &gen_input = req.input_ids;

    gemma::GemmaExpertDb &expert_db = expert_db_;

    // Capacity: prompt + generation + slack.
    const int need = static_cast<int>(gen_input.size()) +
                     std::max(1, req.max_gen_tokens) + 8;
    cudaError_t e = kv_->Allocate(pw_, need);
    if (e != cudaSuccess) {
        r.error_msg = "kv alloc failed";
        return r;
    }

    gemma::GemmaGenerationResult gen;
    if (!gen_ctx_) gen_ctx_ = gemma::CreateGemmaGenContext();
    e = RunUnderResidencyLock([&] {
        return gemma::RunGemmaGeneration(
            st, pw_, expert_db, gen_input, req.max_gen_tokens,
            tok_db_ ? tok_db_->eos_id : -1,
            pw_.top.error_msg.empty() ? pw_.top.final_logit_softcapping : 0.f,
            gen, *kv_, gen_ctx_);
    });
    if (e != cudaSuccess)
        spdlog::error("[gemma-model] RunGemmaGeneration rc={} ({}) gen_err='{}'",
                      static_cast<int>(e), cudaGetErrorString(e), gen.error_msg);

    if (e != cudaSuccess || !gen.ok) {
        r.error_msg = gen.error_msg.empty()
                          ? std::string(cudaGetErrorString(e))
                          : gen.error_msg;
        r.ok = false;
        kv_.reset();
        session_tail_.clear();
        return r;
    }

    runtime::FillInferenceResult(
        r, {.tokens = &gen.tokens,
            .final_logits = &gen.final_logits,
            .ttft_ms = gen.ttft_ms,
            .total_ms = gen.total_ms,
            .error_msg = &gen.error_msg,
            .eos_terminated = gen.eos_terminated,
            .kv_slots_used = static_cast<int>(gen_input.size()) +
                             static_cast<int>(gen.tokens.size()),
            .first_maxv = 0.f,
            .resident_bytes = resident_bytes_},
        static_cast<int>(pw_.layers.size()));
    session_tail_ = r.gen_tokens;
    if (tok_db_ && !r.gen_tokens.empty())
        r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);

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
    free_gemma_expert_index(expert_db_);
    expert_db_ready_ = false;
    gemma::DestroyGemmaGenContext(gen_ctx_);
    gen_ctx_ = nullptr;
    TeardownContext();
    loaded_ = false;
}

gpu::InferenceResult GemmaBaseModel::SelfCheck(const ModelRequest &req)
{
    // Gemma semantics expressed through the unified harness options:
    // chat framing + leading <bos>, forced >= 8 generated tokens, and a
    // 12-token window in which "Paris" must appear (this checkpoint greedily
    // begins its answer with "The capital of France is **Paris**", verified
    // against HF transformers greedy decode with identical framing).
    runtime::ParisSelfCheckOptions opts;
    opts.frame_prompt = true;
    opts.min_tokens = 8;
    opts.answer_window = 12;
    opts.substring_match = true;
    return runtime::RunParisSelfCheck(*this, req, opts);
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
    lsxcommon::runtime::FillBenchmarkResult(r, res.total_time_ms,
                                            res.num_tokens, res.num_layers,
                                            res.weight_resident_bytes);
    if (!res.ok) {
        r.ok = false;
        r.error_msg = res.error_msg;
    }
    return r;
}

} // namespace lsxcommon
