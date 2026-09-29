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

bool InferenceEngine::InitializeEngine()
{
    ResourceGovernor::Instance().Initialize(GovernorOptions{});
    // Perf auto-tuning: call-once resolve+apply of the per-GPU knob policy —
    // the same guarantee direct-factory consumers get via LsxModelFactory.
    perf::EnsurePerfPolicyApplied();
    initialized_.store(true, std::memory_order_release);
    log::debug("[model_engine] initialized");
    return true;
}

void InferenceEngine::AcquireSession()
{
    session_depth_.fetch_add(1, std::memory_order_acq_rel);
    InitializeEngine();
}

void InferenceEngine::ReleaseSession()
{
    const int prev = session_depth_.fetch_sub(1, std::memory_order_acq_rel);
    if (prev <= 0) {
        // Unbalanced release (e.g. a direct InitializeEngine() paired with a
        // Session): clamp instead of wrapping the counter below zero, which
        // would make a later 1->0 shutdown unreachable.
        session_depth_.store(0, std::memory_order_release);
        return;
    }
    if (prev == 1) ShutdownEngine();
}

void InferenceEngine::ShutdownEngine()
{
    initialized_.store(false, std::memory_order_release);
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