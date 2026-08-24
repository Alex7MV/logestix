#include "qwen_3_5_moe_model.h"
#include "qwen_moe_constants.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>

#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {

const std::string Qwen35MoeModel::kType = "qwen3.5-35b-a3b";

namespace {
bool SyntheticGeometryAllowed()
{
    return std::getenv("LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY") != nullptr;
}
} // namespace

Qwen35MoeModel::Qwen35MoeModel(ModelInitConfig cfg)
    : ModelBase{std::move(cfg)} {}

Qwen35MoeModel::~Qwen35MoeModel() { Unload(); }

void Qwen35MoeModel::TeardownWeights()
{
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

    auto check = [&](bool ok, const char *what) {
        if (!ok)
            log::error(absl::StrCat("[qwen-moe-model] geometry mismatch: ",
                                    what));
        return ok;
    };
    if (!check(top.hidden_size == kHidden, "hidden_size")) return false;
    if (!check(top.num_layers == kLayers, "num_layers")) return false;
    if (!check(top.num_attention_heads == kHeads, "num_attention_heads")) return false;
    if (!check(top.num_key_value_heads == kKVHeads, "num_key_value_heads")) return false;
    if (!check(top.head_dim == kHeadDim, "head_dim")) return false;
    if (!check(top.vocab_size == kVocab, "vocab_size")) return false;
    if (!check(top.num_experts == kNumExperts, "num_experts")) return false;
    if (!check(top.num_experts_per_tok == kTopK, "num_experts_per_tok")) return false;
    if (!check(top.moe_intermediate_size == kMoeIntermediate,
               "moe_intermediate_size")) return false;
    if (!check(top.shared_expert_intermediate_size == kSharedIntermediate,
               "shared_expert_intermediate_size")) return false;
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
    if (!qwen_moe::LoadQwenMoeDenseWeights(cfg_.model_path, pw_, stream(), err)) {
        log::error("[qwen-moe-model] dense load failed: " + err);
        TeardownWeights();
        tok_db_.reset();
        return false;
    }
    if (!ValidateGeometry(pw_.moe_top)) {
        TeardownWeights();
        tok_db_.reset();
        return false;
    }
    if (!qwen_moe::LoadQwenMoeExpertStore(cfg_.model_path, experts_, err)) {
        log::error("[qwen-moe-model] expert store load failed: " + err);
        TeardownWeights();
        tok_db_.reset();
        return false;
    }
    // Cache slots: lifecycle-driven start value; bounded by expert count.
    int slots = static_cast<int>(experts_.num_layers * experts_.num_experts);
    slots = std::min(slots, 64);  // TuningBounds default upper bound
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
            stream(), pw_, experts_, cache_, gen_input, req.max_gen_tokens,
            eos_id, gen, lsxcommon::sample::DefaultSampler(), kv_.get());
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
