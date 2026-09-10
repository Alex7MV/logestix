#include "lsxcommon/model_engine.h"

#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/resource_governor.h"

#include <absl/strings/str_format.h>

namespace lsxcommon {

InferenceEngine &InferenceEngine::Instance()
{
    static InferenceEngine instance;  // Meyers singleton
    return instance;
}

bool InferenceEngine::InitializeEngine()
{
    ResourceGovernor::Instance().Initialize(GovernorOptions{});
    // Perf auto-tuning: resolve the per-GPU knob policy (committed base /
    // vault calibration cache / heuristic) and apply it to the process-global
    // kernel-routing state. Precedence inside ResolvePerfKnobs: env > CLI >
    // cache > heuristic > base; LSX_PERF_AUTO=0 restores legacy defaults.
    const perf::GpuPerfProfile gpu = perf::ProbeGpuPerfProfile();
    if (!gpu.name.empty()) {
        perf::LsxPerfKnobs knobs;
        perf::PerfSource source = perf::PerfSource::kHeuristic;
        if (perf::LoadPerfCache(gpu, knobs)) {
            source = perf::PerfSource::kCalibrated;
        } else {
            const perf::LsxPerfKnobSet resolved =
                perf::ResolvePerfKnobs(gpu);
            knobs = resolved.knobs;
            source = resolved.source;
        }
        perf::SetCurrentKnobs(knobs, source);
        perf::ApplyKnobs(knobs);
        log::info(absl::StrFormat(
            "[model_engine] perf policy: gpu='%s' sm=%d bus=%dbit "
            "tc=%d gemv_mib=%d splitk_min=%d slots_pct=%d dual=%d",
            gpu.name.c_str(), gpu.sm_count, gpu.bus_width_bits,
            static_cast<int>(knobs.tc_fast_math_default), knobs.gemv_max_mib,
            knobs.splitk_grid_min, knobs.expert_slots_pct,
            static_cast<int>(knobs.dual_stream_staging)));
    } else {
        log::info("[model_engine] no CUDA device: perf policy left at legacy "
                  "defaults");
    }
    initialized_.store(true, std::memory_order_release);
    log::debug("[model_engine] initialized");
    return true;
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

    r = model.Infer(req);
    if (!r.ok) return r;

    // Replay the generated token stream into the caller's callback as
    // TokenChunk events (step-fanout is synchronous, on the calling thread).
    // The cooperative interruption intercept is honored between chunks so an
    // abort mid-decode truncates the published stream cleanly.
    TokenCallback &cb = req.token_callback;
    if (cb && !r.gen_tokens.empty()) {
        TokenChunk chunk;
        for (chunk.step_t = 0; chunk.step_t < r.gen_tokens.size(); ++chunk.step_t) {
            if (req.abort_flag &&
                req.abort_flag->load(std::memory_order_acquire))
                break;
            chunk.token_id = static_cast<uint32_t>(r.gen_tokens[chunk.step_t]);
            chunk.logit_maxv = 0.0f;  // per-step argmax is published by the
                                      // model's own streaming hooks when present
            chunk.is_special = false;
            cb(chunk);
        }
    }
    return r;
}

} // namespace lsxcommon