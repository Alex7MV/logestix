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
#include "lsxstorage/resolve.h"

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
    artifact_.reset();   // a stale per-layer map must not outlive the weights
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
    artifact_.reset();
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

bool ModelBase::CodecGateAllows(const lsx::storage::ArtifactInfo &info,
                                const hw::HardwareTarget &target,
                                std::string *why) const
{
    const auto decision = hw::CheckCodecs(info.used_codecs, target);
    if (!decision.allowed && why != nullptr) *why = decision.message;
    return decision.allowed;
}

bool ModelBase::LayerUsesCodec(int layer, hw::PlaneCodec codec) const
{
    if (!artifact_) return false;
    if (layer < 0 ||
        static_cast<size_t>(layer) >= artifact_->layer_codec.size())
        return false;
    return (artifact_->layer_codec[static_cast<size_t>(layer)] &
            hw::Mask(codec)) != 0;
}

bool ModelBase::BeginLoad(const std::string &model_path)
{
    if (loaded_) return true;
    if (model_path.empty()) {
        log::error("[model] empty model_path");
        return false;
    }

    // Synthetic fixtures (unit tests) build their own Arrow files and have no
    // artifact.fb. This is an explicit opt-in field, not a fallback: nothing a
    // production binary can set reaches the gate.
    if (cfg_.synthetic) {
        log::warn("[load] synthetic model: storage resolve and codec gate "
                  "skipped by ModelInitConfig::synthetic");
        return LoadTokenizerDb(model_path);
    }

    // ── (1) RESOLVE: source dir / bare name / artifact dir -> artifact dir ──
    const auto resolved = lsx::storage::Resolve(model_path);
    if (resolved.status != lsx::storage::ResolveStatus::kArtifact) {
        log::error(absl::StrFormat(
            "[load] %s is not a LogestiX artifact (%s). %s", model_path,
            lsx::storage::ResolveStatusName(resolved.status),
            resolved.detail.empty() ? "convert it with lsxforge" : resolved.detail));
        return false;
    }

    // ── (2) DECLARE: artifact.fb is mandatory; no inference, no fallback ────
    auto read = lsx::storage::ReadArtifact(resolved.dir);
    if (read.status != lsx::storage::ReadStatus::kOk) {
        log::error(absl::StrFormat(
            "[load] %s has no usable artifact.fb (%s): %s",
            resolved.dir.string(), lsx::storage::ReadStatusName(read.status),
            read.detail));
        return false;
    }

    // ── (3) GATE: invariant I3, a pure function, no hardware call ───────────
    const auto &target = hw::ActiveTarget();
    std::string why;
    if (!CodecGateAllows(read.info, target, &why)) {
        log::error(absl::StrFormat(
            "[load] %s (profile %s) is refused on %s: %s",
            read.info.model_type.c_str(), read.info.profile.c_str(),
            target.Id().c_str(), why.c_str()));
        // Name the alternative so the operator does not have to guess: another
        // profile of the SAME model may be exactly what this device can run.
        // Siblings are found by the artifact's STORE coordinates, which it
        // records itself, so this works even when the caller passed a bare path.
        const auto alternatives =
            lsx::storage::ListProfiles(read.info.org, read.info.model_dir);
        std::string usable;
        for (const auto &alt : alternatives) {
            const auto alt_read = lsx::storage::ReadArtifact(alt.dir);
            if (alt_read.status != lsx::storage::ReadStatus::kOk) continue;
            if (!CodecGateAllows(alt_read.info, target, nullptr)) continue;
            if (!usable.empty()) usable += ", ";
            usable += alt.profile;
        }
        if (!usable.empty())
            log::error("[load] profiles of this model this device CAN run: " +
                       usable);
        else
            log::error("[load] no profile of '" + read.info.model_type +
                       "' is runnable on '" + target.Id() +
                       "'; reconvert with a compatible --codec-profile");
        return false;
    }

    log::info(absl::StrFormat("[load] %s profile=%s codecs=%s target=%s",
                              read.info.model_type.c_str(),
                              read.info.profile.c_str(),
                              hw::FormatMask(read.info.used_codecs).c_str(),
                              target.Id().c_str()));

    // ── (4) REBIND: only after the gate passes, so a refused model never has
    // a mutated config. Every loader joins content subtrees off cfg_.model_path.
    artifact_ = std::make_unique<lsx::storage::ArtifactInfo>(std::move(read.info));
    cfg_.model_path = resolved.dir.string();

    return LoadTokenizerDb(cfg_.model_path);
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

