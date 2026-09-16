#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <shared_mutex>
#include <string>
#include <utility>

#include "lsxcommon/model.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/tokenizer.h"

namespace lsxcommon {

// Thin shared base for every lsxcommon model. Owns the tokenizer Arrow db
// (Item 1: tokenizer-db loading) and the canonical Infer/SelfCheck request
// guards (Item 5: infer-preamble guards). All three models migrate onto it.
// Non-copyable / non-movable: model instances own GPU residency (pinned
// tensors, KV caches), so cloning the base would duplicate weights illegally.
class ModelBase : public IModel {
public:
    ModelBase(const ModelBase &) = delete;
    ModelBase &operator=(const ModelBase &) = delete;
    ModelBase(ModelBase &&) = delete;
    ModelBase &operator=(ModelBase &&) = delete;

    explicit ModelBase(ModelInitConfig cfg) : cfg_(std::move(cfg)) {}

    ModelInitConfig cfg_;
    std::unique_ptr<ArrowTokenizerDb> tok_db_;
    bool loaded_ = false;
    bool tokenizer_optional_ = false;  // Qwen: warn+continue when arrows absent

    bool LoadTokenizerDb(const std::string &model_path);
    bool ValidateInferRequest(const ModelRequest &req, gpu::InferenceResult &r);

    ArrowTokenizerDb &Tokenizer() override { return *tok_db_; }

    std::shared_mutex &ResidencyMutex() override { return residency_mutex_; }

    SafetensorsModel Inspect() override;
    std::uint64_t ResidentWeightBytes() const override { return resident_bytes_; }
    const VramProfile &VramRequirements() const override { return vram_; }

    // Canonical lifecycle template method (lifecycle-unification 2026-08):
    // every model tears down through the same sequence — subclass weight
    // teardown (virtual), tokenizer db release, residency flags. Subclasses
    // MUST NOT override Unload; they put their state into TeardownWeights().
    void Unload() final;

    // One frozen-geometry expectation row: field name, expected value from
    // the compile-time passport, actual value parsed from model.index.json.
    struct GeoFieldCheck {
        const char *name;
        long long expected;
        long long actual;
    };

protected:
    // Shared Load-preamble (triplicated in DeepSeek/GLM/Qwen): loaded_ early-out,
    // empty model_path guard, LoadTokenizerDb. Callers keep any model-specific
    // weight-pin between BeginLoad and FinishLoad.
    bool BeginLoad(const std::string &model_path);
    // Marks the model loaded and runs the one-time perf calibration on
    // unknown GPUs (no committed base entry, no vault cache). Hundreds of
    // ms once per GPU; subsequent Loads hit the cache.
    void FinishLoad();

    // Canonical Load-failure tail: full teardown (subclass weights, tokenizer
    // db, residency flags) and `return false` semantics. Every Load() failure
    // path MUST end with `return AbortLoad();` so a failed Load leaves the
    // instance exactly as Unload() would (loaded_==false invariant).
    bool AbortLoad();

    // Governor lease ownership (lifecycle-unification II). Every model
    // accounts its VRAM through these instead of touching the governor
    // directly:
    //  - AcquirePermanentWeights: standalone governor init + kRequired
    //    kPermanentWeights lease. Returns false ONLY on a hard refusal
    //    (relax_memory_refusal keeps the old continue-anyway semantics);
    //    callers do `if (!AcquirePermanentWeights(bytes)) return AbortLoad();`
    //  - AcquireExpertCacheLease: soft kAdaptive kExpertCache lease clamped
    //    into [min_bytes, max_bytes]; false is non-fatal (callers log + keep
    //    their shrink-retry allocation path).
    // Both leases are released by the canonical Unload/AbortLoad.
    bool AcquirePermanentWeights(std::uint64_t bytes);
    bool AcquireExpertCacheLease(std::uint64_t bytes, std::uint64_t min_bytes,
                                 std::uint64_t max_bytes);

    // Subclass-owned GPU/pinned state teardown. Called by Unload(), by
    // AbortLoad() and therefore by every destructor via `~Model() { Unload(); }`.
    // Must be safe on a never-loaded (empty) instance and idempotent.
    virtual void TeardownWeights() {}

    // ── Persistent CUDA execution context (model-unification 2026-08) ──────
    // Lazy-creates a non-blocking stream. Idempotent: repeated calls keep the
    // existing stream. Every model owns exactly one context for its whole
    // lifetime (no temp-per-call churn); Unload paths must call
    // TeardownContext().
    bool EnsureContext();
    void TeardownContext();
    cudaStream_t stream() const { return ctx_stream_; }
    bool ContextReady() const { return ctx_stream_ != nullptr; }

    // Persistent device-resident weight set (pinned for MoE GLM/DeepSeek; full
    // for dense Qwen). Assign in subclass Load/Unload. See spec 3.3.
    std::uint64_t resident_bytes_ = 0;

    // Conversion-time VRAM requirements; populated by subclass Load() from
    // the model's model.index.json "vram" block.
    VramProfile vram_;

    // ── Shared geometry validation (model-unification 2026-08) ─────────────
    // LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY env gate (was triplicated per family).
    static bool SyntheticGeometryAllowed();

    // Logs one uniform error line per mismatch and returns false at the first
    // mismatch; true when every field matches. Synthetic bypass remains the
    // CALLER's decision (some validators gate extra loops, e.g. layer_types).
    bool CheckGeometryFields(const char *tag, const GeoFieldCheck *fields,
                             size_t n) const;

    // Runs `fn` under the shared residency lock: unload-during-inference and
    // parallel dispatch serialize behind a unique_lock, while generation
    // itself stays shared-safe. Every Infer implementation MUST
    // route its generation call through this helper (uniform preamble).
    template <typename F>
    auto RunUnderResidencyLock(F &&fn) -> decltype(fn())
    {
        std::shared_lock<std::shared_mutex> lock(residency_mutex_);
        return fn();
    }

    // Persistent CUDA execution context (see EnsureContext above).
    cudaStream_t ctx_stream_ = nullptr;

    // Residency lock (see IModel::ResidencyMutex). Held shared during Infer.
    std::shared_mutex residency_mutex_;

    // Governor leases owned by the base; released by Unload/AbortLoad.
    // GLM moves expert_cache_lease_ into its RuntimeTuner (adaptive cache);
    // the moved-from member stays empty and releasing it again is a no-op.
    Lease permanent_lease_;
    Lease expert_cache_lease_;

    ~ModelBase() override = default;
};

} // namespace lsxcommon