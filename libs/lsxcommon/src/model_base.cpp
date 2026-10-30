#include "lsxcommon/model_base.h"

#include <absl/strings/str_cat.h>

#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/lsx_tc_gemm.h"
#include "lsxcommon/perf_calibration.h"

#include <absl/strings/str_format.h>
#include <cstdlib>
#include <mutex>

#include "lsxcommon/log.h"
#include "lsxcommon/tokenizer_arrows.h"

namespace lsxcommon {

// ── Canonical lifecycle sequence (lifecycle-unification 2026-08) ────────────

void ModelBase::Unload()
{
    // Serialize teardown against in-flight inference. Infer runs under a shared
    // residency lock (RunUnderResidencyLock); taking the unique lock here is
    // what makes "unload during inference" safe instead of a teardown racing
    // the decode (freed weights / destroyed stream).
    std::unique_lock<std::shared_mutex> lock(residency_mutex_);
    // Weights are about to be freed: invalidate every process-lifetime cache
    // keyed by a weight's device address, so a later model that lands on a
    // recycled address cannot inherit this one's entries (OPEN_TASKS E1d).
    lsx_tc::BumpLsxTcWeightEpoch();
    TeardownWeights();
    permanent_lease_ = {};
    expert_cache_lease_ = {};
    tok_db_.reset();
    loaded_ = false;
    resident_bytes_ = 0;
}

bool ModelBase::AbortLoad()
{
    // The canonical Load-failure tail tears down subclass state; serialize it
    // with any residency reader exactly like Unload().
    std::unique_lock<std::shared_mutex> lock(residency_mutex_);
    lsx_tc::BumpLsxTcWeightEpoch();  // partial weights are freed: see Unload
    TeardownWeights();
    permanent_lease_ = {};
    expert_cache_lease_ = {};
    tok_db_.reset();
    loaded_ = false;
    resident_bytes_ = 0;
    return false;
}

bool ModelBase::AcquirePermanentWeights(std::uint64_t bytes)
{
    auto &gov = ResourceGovernor::Instance();
    if (!gov.IsInitialized()) {
        log::warn("[model] governor not initialized; standalone init");
        GovernorOptions gopts;
        if (!gov.Initialize(gopts)) {
            log::error("[model] governor init failed");
            return false;
        }
    }
    permanent_lease_ = gov.Acquire(LeaseKind::kPermanentWeights, bytes,
                                   LeasePolicy{});
    if (!permanent_lease_.Valid() && !gov.options().relax_memory_refusal) {
        log::error("[model] permanent-weight lease refused ("
                   + std::to_string(bytes) + " bytes)");
        return false;
    }
    return true;
}

bool ModelBase::AcquireExpertCacheLease(std::uint64_t bytes,
                                        std::uint64_t min_bytes,
                                        std::uint64_t max_bytes)
{
    LeasePolicy adaptive{LeasePolicy::Mode::kAdaptive, min_bytes, max_bytes};
    expert_cache_lease_ = ResourceGovernor::Instance().Acquire(
        LeaseKind::kExpertCache, bytes, adaptive);
    return expert_cache_lease_.Valid();
}

bool ModelBase::LoadTokenizerDb(const std::string &model_path)
{
    if (tok_db_) return true;
    tok_db_ = std::make_unique<ArrowTokenizerDb>();
    if (!LoadTokenizerArrows(model_path, *tok_db_)) {
        if (tokenizer_optional_) {
            log::warn(
                "[model] tokenizer arrows unavailable (OK for synthetic test)");
            return true;
        }
        log::error(absl::StrFormat(
            "[model] cannot load tokenizer arrows for %s: %s", model_path,
            tok_db_->error_msg));
        tok_db_.reset();
        return false;
    }
    return true;
}

bool ModelBase::BeginLoad(const std::string &model_path)
{
    if (loaded_) return true;
    if (model_path.empty()) {
        log::error("[model] empty model_path");
        return false;
    }
    return LoadTokenizerDb(model_path);
}

SafetensorsModel ModelBase::Inspect()
{
    return lsxcommon::scan_directory(cfg_.model_path);
}

bool ModelBase::ValidateInferRequest(const ModelRequest &req, gpu::InferenceResult &r)
{
    if (!loaded_) {
        r.error_msg = "model not loaded";
        return false;
    }
    if (req.input_ids.empty()) {
        r.error_msg = "empty input_ids";
        return false;
    }
    if (req.max_gen_tokens <= 0) {
        r.error_msg = "max_gen_tokens must be positive";
        return false;
    }
    return true;
}

// ── Persistent CUDA execution context (model-unification 2026-08) ───────────

bool ModelBase::EnsureContext()
{
    if (!ctx_stream_) {
        const cudaError_t e =
            cudaStreamCreateWithFlags(&ctx_stream_, cudaStreamNonBlocking);
        if (e != cudaSuccess) {
            ctx_stream_ = nullptr;
            log::error(absl::StrCat("[model] CUDA stream create failed: ",
                                    cudaGetErrorString(e)));
            return false;
        }
    }
    return true;
}

void ModelBase::TeardownContext()
{
    if (ctx_stream_) {
        cudaStreamDestroy(ctx_stream_);
        ctx_stream_ = nullptr;
    }
}

// ── Shared geometry validation (model-unification 2026-08) ──────────────────

bool ModelBase::CheckGeometryFields(const char *tag,
                                    const GeoFieldCheck *fields,
                                    size_t n) const
{
    for (size_t i = 0; i < n; ++i) {
        const auto &f = fields[i];
        if (f.expected != f.actual) {
            log::error(std::string("[") + tag +
                       "] geometry mismatch: " + f.name + ": expected " +
                       std::to_string(f.expected) + ", got " +
                       std::to_string(f.actual));
            return false;
        }
    }
    return true;
}


void ModelBase::FinishLoad()
{
    // Weights are now resident at addresses CUDA will later recycle: start a
    // fresh cache generation so no entry can outlive the weights it was built
    // from (OPEN_TASKS E1d).
    lsx_tc::BumpLsxTcWeightEpoch();
    // Perf policy: one-time local calibration on unknown GPUs (no committed
    // base entry, no vault cache). Runs after weights are resident so the
    // device is warm; result lands in perf-cache/<gpu-key>.json.
    perf::EnsureCalibrated(perf::ProbeGpuPerfProfile(),
                           perf::CalibrateGemvLimitMib);
    loaded_ = true;
}

} // namespace lsxcommon

