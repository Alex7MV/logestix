#include "muse_dense_model.h"

#include <cuda_runtime.h>

#include <chrono>
#include <string>

#include <absl/strings/str_format.h>

#include "fast_load_internal.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {

const std::string MuseGlimmer30BModel::kType = "muse-glimmer-30b";

MuseGlimmer30BModel::MuseGlimmer30BModel(ModelInitConfig cfg)
    : ModelBase(std::move(cfg)) {}

MuseGlimmer30BModel::~MuseGlimmer30BModel() { Unload(); }

void MuseGlimmer30BModel::TeardownWeights()
{
    pw_.Free();
    weights_loaded_ = false;
    kv_.reset();
    session_tail_.clear();
    TeardownContext();
}

bool MuseGlimmer30BModel::ValidateGeometry(
    const muse_dense::MuseDenseTopology &top) const
{
    if (!top.error_msg.empty()) return false;
    namespace fz = muse_dense::frozen;
    if (!SyntheticGeometryAllowed()) {
        const GeoFieldCheck fields[] = {
            {"hidden_size", fz::kHiddenSize, top.hidden_size},
            {"num_layers", fz::kNumLayers, top.num_layers},
            {"num_attention_heads", fz::kNumHeads, top.num_attention_heads},
            {"num_key_value_heads", fz::kNumKVHeads, top.num_key_value_heads},
            {"head_dim", fz::kHeadDim, top.head_dim},
            {"intermediate_size", fz::kIntermediateSize, top.intermediate_size},
            {"vocab_size", fz::kVocabSize, top.vocab_size},
            {"sliding_window", fz::kSlidingWindow, top.sliding_window},
        };
        if (!CheckGeometryFields("muse-glimmer-30b", fields, 8)) return false;
    }
    // The 3:1 sliding/full pattern is checked EVEN under the synthetic
    // bypass: it is the structural invariant the kernels rely on (window +
    // NoPE set), and the synthetic fixture preserves it at its own depth.
    if (static_cast<int>(top.layer_types.size()) != top.num_layers) {
        lsxcommon::log::error(
            "[muse-model] layer_types size does not cover num_layers");
        return false;
    }
    for (int l = 0; l < top.num_layers; ++l) {
        if (top.IsFull(l) != fz::IsFullLayer(l)) {
            lsxcommon::log::error(absl::StrFormat(
                "[muse-model] layer_types[%d] contradicts the frozen 3:1 "
                "sliding/full pattern", l));
            return false;
        }
    }
    return true;
}

bool MuseGlimmer30BModel::Load()
{
    Unload();  // idempotent re-load support
    tokenizer_optional_ = true;
    if (!BeginLoad(cfg_.model_path)) return false;
    const auto t_pinned = std::chrono::steady_clock::now();
    if (!EnsureContext()) {
        lsxcommon::log::error("[muse-model] stream create failed");
        return AbortLoad();
    }
    const cudaError_t lrc =
        muse_dense::LoadMuseDensePinnedWeights(cfg_.model_path, pw_, stream());
    fastload::StreamDrain{stream()};
    const bool ok = (lrc == cudaSuccess);
    weights_loaded_ = ok;
    if (!ok) {
        lsxcommon::log::error(
            "[muse-model] LoadMuseDensePinnedWeights failed: rc=" +
            std::to_string(static_cast<int>(lrc)) + " (" +
            cudaGetErrorString(lrc) + ")");
        return AbortLoad();
    }
    resident_bytes_ = pw_.resident_bytes;
    vram_ = pw_.top.vram;
    if (!AcquirePermanentWeights(pw_.resident_bytes)) return AbortLoad();
    lsxcommon::log::info(
        "[muse-model] pinned weights loaded: " +
        std::to_string(pw_.layers.size()) + " layers (" +
        std::to_string(pw_.resident_bytes) + " bytes resident)");
    // Geometry gate BEFORE FinishLoad: a failed Load must leave loaded_ false.
    if (!ValidateGeometry(pw_.top)) {
        lsxcommon::log::error("[muse-model] geometry validation failed for " +
                              cfg_.model_path);
        return AbortLoad();
    }
    FinishLoad();
    lsxcommon::log::info(
        "[muse-model] pinned load " +
        absl::StrFormat("%.1f", std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t_pinned)
                                    .count()) +
        " ms");
    return true;
}

gpu::InferenceResult MuseGlimmer30BModel::Infer(const ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!ValidateInferRequest(req, r)) return r;

    // Model-independence invariant: this model's OWN kernel-routing policy
    // for the duration of the call.
    const lsx_nvfp4::LsxFastMathScope fm_scope(EffectiveFastMathMode(cfg_));

    if (!EnsureContext()) {
        r.error_msg = "cuda stream create failed";
        return r;
    }
    cudaStream_t st = stream();

    // Streaming + cancellation handle (one Emit per decoded token; the abort
    // flag is honoured between decode steps).
    lsxcommon::GenerationStream gs;
    gs.on_token = req.token_callback ? &req.token_callback : nullptr;
    gs.abort_flag = req.abort_flag;

    // The whole request runs inside ONE shared-lock scope: creating,
    // allocating, decoding through and publishing the KV state under the
    // lock. One Infer per instance at a time is a precondition (OPEN_TASKS
    // E1b); the lock orders this request against Unload()'s unique lock.
    return RunUnderResidencyLock([&]() -> gpu::InferenceResult {
        kv_.reset();
        kv_ = std::make_unique<muse_dense::MuseDenseKvCache>();
        session_tail_.clear();

        const int need = static_cast<int>(req.input_ids.size()) +
                         std::max(1, req.max_gen_tokens) + 8;
        const cudaError_t ae = kv_->Allocate(pw_, need);
        if (ae != cudaSuccess) {
            r.error_msg = "kv alloc failed";
            return r;
        }

        std::vector<int> eos_ids;
        if (req.ignore_eos || !tok_db_)
            eos_ids = {lsxcommon::kIgnoreEosTokenId};
        else if (!pw_.top.eos_ids.empty())
            eos_ids = pw_.top.eos_ids;
        else
            eos_ids = {tok_db_->eos_id};

        muse_dense::MuseDenseGenerationResult gen;
        const cudaError_t ge = muse_dense::RunMuseDenseGeneration(
            st, pw_, req.input_ids, req.max_gen_tokens, eos_ids, gen, *kv_,
            sample::SamplerFor(req.sampling), &gs);
        if (ge != cudaSuccess || !gen.ok) {
            lsxcommon::log::error(
                "[muse-model] RunMuseDenseGeneration rc=" +
                std::to_string(static_cast<int>(ge)) + " (" +
                cudaGetErrorString(ge) + ") gen_err='" + gen.error_msg + "'");
            r.error_msg = runtime::CombineGenError(ge, gen.error_msg);
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
                .kv_slots_used = static_cast<int>(req.input_ids.size()) +
                                 static_cast<int>(gen.tokens.size()),
                .first_maxv = 0.f,
                .resident_bytes = gen.resident_bytes},
            static_cast<int>(pw_.layers.size()));
        session_tail_ = r.gen_tokens;
        if (tok_db_ && !r.gen_tokens.empty())
            r.output_text = runtime::DecodeGeneratedText(*tok_db_, r.gen_tokens);
        return r;
    });
}

gpu::InferenceResult MuseGlimmer30BModel::SelfCheck(const ModelRequest &req)
{
    return runtime::RunParisSelfCheck(*this, req);
}

} // namespace lsxcommon