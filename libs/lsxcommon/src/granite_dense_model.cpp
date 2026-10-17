#include "granite_dense_model.h"

#include <cuda_runtime.h>

#include "fast_load_internal.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/sampling.h"

#include <absl/strings/str_cat.h>

namespace lsxcommon {

const std::string Granite42_30BModel::kType = "granite-4.2-30b";

Granite42_30BModel::Granite42_30BModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

Granite42_30BModel::~Granite42_30BModel() { Unload(); }

void Granite42_30BModel::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool Granite42_30BModel::Load()
{
    Unload();
    tokenizer_optional_ = false;
    const FastMathMode fm = EffectiveFastMathMode(cfg_);
    if (fm != FastMathMode::kAuto) {
        lsx_nvfp4::SetLsxTcFastMath(fm == FastMathMode::kOn);
        lsx_nvfp4::SetLsxSeq1Gemv(fm == FastMathMode::kOn);
    }
    if (!BeginLoad(cfg_.model_path)) return false;
    if (!EnsureContext()) {
        lsxcommon::log::error("[granite-model] stream create failed");
        return AbortLoad();
    }
    const bool ok =
        granite::LoadGranitePinnedWeights(cfg_.model_path, pw_, stream());
    fastload::StreamDrain{stream()};
    weights_loaded_ = ok;
    if (!ok) {
        lsxcommon::log::error("[granite-model] LoadGranitePinnedWeights failed");
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes;
    vram_ = pw_.top.vram;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    lsxcommon::log::info(absl::StrCat(
        "[granite-model] pinned weights loaded: ", pw_.layers.size(),
        " layers (", pw_.resident_bytes, " bytes resident)"));
    if (!ValidateGeometry(pw_.top)) {
        lsxcommon::log::error(absl::StrCat(
            "[granite-model] geometry validation failed for ",
            cfg_.model_path));
        return AbortLoad();
    }
    FinishLoad();
    return true;
}

bool Granite42_30BModel::ValidateGeometry(
    const granite::GraniteTopology &top) const
{
    if (SyntheticGeometryAllowed()) return true;
    const GeoFieldCheck fields[] = {
        {"hidden_size", kHiddenSize, top.hidden_size},
        {"num_layers", kNumLayers, top.num_layers},
        {"num_attention_heads", kNumHeads, top.num_attention_heads},
        {"num_key_value_heads", kNumKVHeads, top.num_key_value_heads},
        {"head_dim", kHeadDim, top.head_dim},
        {"intermediate_size", kIntermediateSize, top.intermediate_size},
        {"vocab_size", kVocabSize, top.vocab_size},
    };
    return CheckGeometryFields("granite-4.2-30b", fields, 7);
}

gpu::InferenceResult Granite42_30BModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // Model independence: this model's own kernel-routing policy for the
    // duration of the call.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    if (!EnsureContext()) {
        r.error_msg = "cuda init failed";
        return r;
    }
    cudaStream_t st = stream();

    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    // The whole request runs inside ONE shared-lock scope. Incremental
    // session policy: a request matching the previous call's generated tail
    // continues from {kv_->last_tok}; anything else restarts. Creating,
    // decoding through and publishing the KV state under the lock. The lock is shared, so it orders this request against Unload()'s unique
    // lock and does NOT serialise Infer against Infer - two concurrent
    // Infer calls on one instance both hold it. One Infer per instance at
    // a time is a precondition (OPEN_TASKS E1b), and that is what keeps
    // them from freeing each other's cache.
    return RunUnderResidencyLock([&]() -> gpu::InferenceResult {
        const bool continuation =
            kv_ && kv_->allocated() && !session_tail_.empty() &&
            req.input_ids == session_tail_;
        std::vector<int32_t> gen_input;
        if (continuation) {
            gen_input = std::vector<int32_t>{kv_->last_tok};
        } else {
            kv_.reset();
            kv_.reset(new granite::GraniteKvCache());
            session_tail_.clear();
            gen_input = req.input_ids;
        }

        granite::GraniteGenerationResult gen;
        const int eos_id = (req.ignore_eos || !tok_db_)
                               ? lsxcommon::kIgnoreEosTokenId
                               : tok_db_->eos_id;
        const cudaError_t e = granite::RunGraniteGeneration(
            st, pw_, gen_input, req.max_gen_tokens, eos_id, gen,
            lsxcommon::sample::DefaultSampler(), kv_.get(), &gs);
        if (e != cudaSuccess) {
            r.error_msg = runtime::CombineGenError(e, gen.error_msg);
            r.ok = false;
            kv_.reset();
            session_tail_.clear();
            return r;
        }

        const int kv_slots =
            (kv_ && kv_->allocated())
                ? kv_->kv_rows
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
    });
}

gpu::InferenceResult Granite42_30BModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

} // namespace lsxcommon
