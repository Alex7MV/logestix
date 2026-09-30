#include "lsxcommon/model_engine.h"

#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/resource_governor.h"

namespace lsxcommon {

InferenceEngine &InferenceEngine::Instance()
{
    static InferenceEngine instance;  // Meyers singleton
    return instance;
}

void InferenceEngine::EnsureEngineUp()
{
    ResourceGovernor::Instance().Initialize(GovernorOptions{});
    // Perf auto-tuning: call-once resolve+apply of the per-GPU knob policy —
    // the same guarantee direct-factory consumers get via LsxModelFactory.
    perf::EnsurePerfPolicyApplied();
    initialized_.store(true, std::memory_order_release);
    log::debug("[model_engine] initialized");
}

bool InferenceEngine::InitializeEngine()
{
    EnsureEngineUp();
    // Every direct init is an explicit ownership claim, even when a Session
    // already holds the engine up: a later Session release must not tear the
    // engine down under the direct caller. ShutdownEngine() clears it.
    direct_owned_.store(true, std::memory_order_release);
    return true;
}

void InferenceEngine::AcquireSession()
{
    session_depth_.fetch_add(1, std::memory_order_acq_rel);
    // The Session counts itself in session_depth_, so it must not take a
    // direct claim; only bring the engine up when nobody else holds it.
    if (!IsInitialized()) EnsureEngineUp();
}

void InferenceEngine::ReleaseSession()
{
    const int prev = session_depth_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 0) {
        // Unbalanced release: clamp instead of wrapping the counter below
        // zero, which would make a later 1->0 shutdown unreachable.
        session_depth_.store(0, std::memory_order_release);
        return;
    }
    if (prev == 1 && !direct_owned_.load(std::memory_order_acquire))
        ShutdownEngine();
}

void InferenceEngine::ShutdownEngine()
{
    initialized_.store(false, std::memory_order_release);
    direct_owned_.store(false, std::memory_order_release);
    ResourceGovernor::Instance().Shutdown();
    log::debug("[model_engine] shutdown");
}

std::unique_ptr<IModel> InferenceEngine::CreateModel(const ModelInitConfig &cfg)
{
    if (!IsInitialized()) return nullptr;
    return LsxModelFactory::Create(cfg);
}

gpu::InferenceResult InferenceEngine::Run(IModel &model, ModelRequest &req)
{
    gpu::InferenceResult r;
    if (!IsInitialized()) {
        r.error_msg = "engine not initialized";
        return r;
    }
    if (req.abort_flag && req.abort_flag->load(std::memory_order_acquire)) {
        r.error_msg = "aborted by caller (abort_flag)";
        return r;
    }

    // IModel::Infer owns the streaming contract: every family threads a
    // GenerationStream through its decode loop and invokes req.token_callback
    // once per decoded token (and honors req.abort_flag between steps). The
    // facade must forward the request untouched and must NOT replay the
    // returned gen_tokens, or each token would be published twice.
    return model.Infer(req);
}

} // namespace lsxcommon