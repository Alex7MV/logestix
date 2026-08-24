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
        return r;
    }

    gemma::GemmaGenerationResult gen;
    e = RunUnderResidencyLock([&] {
        return gemma::RunGemmaGeneration(
            st, pw_, expert_db, gen_input, req.max_gen_tokens,
            tok_db_ ? tok_db_->eos_id : -1,
            pw_.top.error_msg.empty() ? pw_.top.final_logit_softcapping : 0.f,
            gen, *kv_);
    });
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
    TeardownContext();
    loaded_ = false;
}

gpu::InferenceResult GemmaBaseModel::SelfCheck(const ModelRequest &req)
{
    // Gemma chat framing + leading <bos> (model-owned, leading_bos policy):
    // <|turn>user\n{prompt}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
    ModelRequest framed = req;
    framed.max_gen_tokens = std::max(req.max_gen_tokens, 8);
    if (tok_db_ && !BuildPromptInputIds(*tok_db_,
                                        "The capital of France is",
                                        framed.input_ids)) {
        gpu::InferenceResult r;
        r.error_msg = "selfcheck prompt framing failed";
        return r;
    }

    gpu::InferenceResult r = Infer(framed);
    if (!r.ok) {
        r.selfcheck_passed = false;
        return r;
    }

    // Pass criterion: this checkpoint greedily begins its answer with
    // "The capital of France is **Paris**" (verified against HF transformers
    // greedy decode with the identical framing), so the canonical answer
    // string must appear within the first generated tokens rather than be
    // the literal first token.
    if (!r.gen_tokens.empty() && tok_db_) {
        constexpr size_t kAnswerWindow = 12;
        const size_t n = std::min(r.gen_tokens.size(), kAnswerWindow);
        r.first_gen_text = runtime::DecodeGeneratedText(*tok_db_,
            std::vector<int32_t>(r.gen_tokens.begin(),
                                 r.gen_tokens.begin() +
                                     static_cast<long>(n)));
        r.selfcheck_passed =
            r.first_gen_text.find("Paris") != std::string::npos;
    } else {
        r.selfcheck_passed = false;
    }

    log::info(absl::StrFormat(
        "[selfcheck] top1_tok=%d text='%s' passed=%s",
        static_cast<int>(r.first_gen_tok_id), r.first_gen_text.c_str(),
        r.selfcheck_passed ? "true" : "false"));
    return r;
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
