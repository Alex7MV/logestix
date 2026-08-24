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

bool Qwen35MoeModel::EnsureExecutionContext()
{
    if (!stream_ &&
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
            cudaSuccess) {
        stream_ = nullptr;
        return false;
    }
    return true;
}

void Qwen35MoeModel::TeardownWeights()
{
    qwen_moe::FreeQwenMoeExpertCache(cache_);
    experts_.Free();
    qwen_moe::FreeQwenMoePinnedWeights(pw_);
    kv_.reset();
    session_tail_.clear();
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
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

    if (!EnsureExecutionContext()) {
        log::error("[qwen-moe-model] stream init failed");
        tok_db_.reset();
        return false;
    }

    std::string err;
    if (!qwen_moe::LoadQwenMoeDenseWeights(cfg_.model_path, pw_, stream_, err)) {
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
    if (!stream_) {
        r.error_msg = "execution context missing";
        return r;
    }

    // Incremental session policy (mirrors QwenBaseModel).
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
    const int eos_id = tok_db_ ? tok_db_->eos_id : -1;
    cudaError_t e = qwen_moe::RunQwenMoeGeneration(
        stream_, pw_, experts_, cache_, gen_input, req.max_gen_tokens,
        eos_id, gen, lsxcommon::sample::DefaultSampler(), kv_.get());
    if (e != cudaSuccess) {
        r.error_msg = std::string(cudaGetErrorString(e)) + " / " + gen.error_msg;
        r.ok = false;
        kv_.reset();
        session_tail_.clear();
        return r;
    }

    r.ok = true;
    r.num_tokens = static_cast<int>(gen.tokens.size());
    r.num_layers = pw_.top.num_layers;
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
    r.kv_slots_used = (kv_ && kv_->allocated()) ? kv_->kv_rows + 1
                                                : r.num_tokens + static_cast<int>(req.input_ids.size());
    r.weight_resident_bytes = gen.resident_bytes;
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
