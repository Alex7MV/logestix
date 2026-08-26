#include "qwen_3_5_moe_model.h"
#include "qwen_moe_constants.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {

namespace {
double MsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}
} // namespace

const std::string Qwen35MoeModel::kType = "qwen3.5-35b-a3b";

Qwen35MoeModel::Qwen35MoeModel(ModelInitConfig cfg)
    : ModelBase{cfg}
{
    // The TC W4A4 toggle is process-wide; the dense Qwen models flip it in
    // QwenBaseModel, but this MoE model derives from ModelBase directly and
    // silently ran every GEMM on the scalar path without this.
    qwen::SetQwenTcFastMath(cfg_.fast_math);
}

Qwen35MoeModel::~Qwen35MoeModel() { Unload(); }

void Qwen35MoeModel::TeardownWeights()
{
    qwen_moe::FreeQwenMoeGenScratch(scratch_);
    qwen_moe::FreeQwenMoeExpertCache(cache_);
    experts_.Free();
    qwen_moe::FreeQwenMoePinnedWeights(pw_);
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
    resident_bytes_ = 0;
}

bool Qwen35MoeModel::ValidateGeometry(
    const qwen_moe::QwenMoeTopology &top) const
{
    using namespace qwen_moe::frozen;
    if (SyntheticGeometryAllowed()) return true;

    const GeoFieldCheck fields[] = {
        {"hidden_size", kHidden, top.hidden_size},
        {"num_layers", kLayers, top.num_layers},
        {"num_attention_heads", kHeads, top.num_attention_heads},
        {"num_key_value_heads", kKVHeads, top.num_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"vocab_size", kVocab, top.vocab_size},
        {"num_experts", kNumExperts, top.num_experts},
        {"num_experts_per_tok", kTopK, top.num_experts_per_tok},
        {"moe_intermediate_size", kMoeIntermediate,
         top.moe_intermediate_size},
        {"shared_expert_intermediate_size", kSharedIntermediate,
         top.shared_expert_intermediate_size},
    };
    if (!CheckGeometryFields("qwen-moe-model", fields, 10)) return false;
    for (int l = 0; l < top.num_layers; ++l) {
        const std::string want =
            (l % kFullAttnInterval == kFullAttnInterval - 1) ? "full_attention"
                                                             : "linear_attention";
        if (top.layer_types[static_cast<size_t>(l)] != want) {
            log::error(absl::StrCat("[qwen-moe-model] layer_types[", l,
                                    "] != ", want));
            return false;
        }
    }
    return true;
}

bool Qwen35MoeModel::Load()
{
    Unload();  // idempotent re-load support
    if (!BeginLoad(cfg_.model_path)) return false;

    if (!EnsureContext(/*with_cublas=*/false)) {
        log::error("[qwen-moe-model] stream init failed");
        tok_db_.reset();
        return false;
    }

    std::string err;
    const auto t_dense = std::chrono::steady_clock::now();
    if (!qwen_moe::LoadQwenMoeDenseWeights(cfg_.model_path, pw_, stream(), err)) {
        log::error("[qwen-moe-model] dense load failed: " + err);
        TeardownWeights();
        tok_db_.reset();
        return false;
    }
    log::info(absl::StrCat("[qwen-moe-model] dense load ", MsSince(t_dense),
                           " ms"));
    if (!ValidateGeometry(pw_.moe_top)) {
        TeardownWeights();
        tok_db_.reset();
        return false;
    }
    const auto t_experts = std::chrono::steady_clock::now();
    if (!qwen_moe::LoadQwenMoeExpertStore(cfg_.model_path, experts_, err)) {
        log::error("[qwen-moe-model] expert store load failed: " + err);
        TeardownWeights();
        tok_db_.reset();
        return false;
    }
    log::info(absl::StrCat("[qwen-moe-model] expert store ",
                           MsSince(t_experts), " ms"));
    // Cache slots: lifecycle-driven start value; bounded by expert count.
    // LSX_QWEN_MOE_EXPERT_CACHE_SLOTS overrides for VRAM sweeps (parity with
    // LSX_DEEPSEEK_EXPERT_CACHE_SLOTS); values < 1 keep the default bound.
    int slots = static_cast<int>(experts_.num_layers * experts_.num_experts);
    slots = std::min(slots, 512);  // VRAM-bounded default (~0.9 GB at 1.73 MB/expert)
    if (const char *env_slots = std::getenv("LSX_QWEN_MOE_EXPERT_CACHE_SLOTS")) {
        const int v = std::atoi(env_slots);
        if (v > 0) slots = std::min(v, static_cast<int>(experts_.num_layers *
                                                        experts_.num_experts));
    }
    if (slots <= 0) slots = 4;
    if (!qwen_moe::AllocateQwenMoeExpertCache(experts_, slots, cache_, err)) {
        log::error("[qwen-moe-model] expert cache alloc failed: " + err);
        TeardownWeights();
        tok_db_.reset();
        return false;
    }

    resident_bytes_ = pw_.resident_bytes;
    FinishLoad();
    log::info(absl::StrCat("[qwen-moe-model] loaded: resident=",
                           resident_bytes_, " bytes"));
    return true;
}

gpu::InferenceResult Qwen35MoeModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return r;
    }
    if (!ContextReady()) {
        r.error_msg = "execution context missing";
        return r;
    }

    // Incremental session policy (mirrors QwenBaseModel).
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
    const int eos_id = tok_db_ ? tok_db_->eos_id : -1;
    cudaError_t e = RunUnderResidencyLock([&] {
        return qwen_moe::RunQwenMoeGeneration(
            stream(), pw_, experts_, cache_, scratch_, gen_input,
            req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::DefaultSampler(), kv_.get());
    });
    if (e != cudaSuccess) {
        r.error_msg = std::string(cudaGetErrorString(e)) + " / " + gen.error_msg;
        r.ok = false;
        kv_.reset();
        session_tail_.clear();
        return r;
    }

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
    return r;
}

void Qwen35MoeModel::Unload()
{
    TeardownWeights();
    loaded_ = false;
}

gpu::InferenceResult Qwen35MoeModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

} // namespace lsxcommon
