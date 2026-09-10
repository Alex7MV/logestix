#include "lsxcommon/model_runtime.h"

#include "lsxcommon/benchmark_gpu.h"
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/perf_calibration.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/log.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/safetensors.h"

namespace lsxcommon {
namespace runtime {

namespace {
constexpr size_t kMaxDecodeBytes = 1u << 20;
constexpr size_t kInitialDecodeCap = 4096;
constexpr int kMaxDecodeAttempts = 8;
} // namespace

std::string CombineGenError(cudaError_t err, const std::string &gen_error)
{
    const char *cuda_str =
        (err != cudaSuccess) ? cudaGetErrorString(err) : nullptr;
    const bool has_cuda = cuda_str != nullptr && *cuda_str != '\0';
    const bool has_gen = !gen_error.empty();
    if (has_cuda && has_gen)
        return std::string(cuda_str) + " / " + gen_error;
    if (has_gen) return gen_error;
    if (has_cuda) return std::string(cuda_str);
    return {};
}

std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids)
{
    if (ids.empty()) return {};
    size_t cap = kInitialDecodeCap;
    std::string out(cap, '\0');
    for (int attempt = 0; attempt < kMaxDecodeAttempts; ++attempt) {
        const int dr = arrow_tokenizer_decode(tok_db, ids.data(), ids.size(),
                                              out.data(), &cap);
        if (dr == 0) {
            out.resize(cap);
            return out;
        }
        if (dr == -2 && cap > out.size() && cap <= kMaxDecodeBytes) {
            out.resize(cap);
            continue;
        }
        break;
    }
    lsxcommon::log::warn(
        absl::StrFormat("[model_runtime] tokenizer decode failed (%u ids)",
                        ids.size()));
    return {};
}

gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req,
                                       const ParisSelfCheckOptions &opts)
{
    ModelRequest sc_req = req;
    if (opts.frame_prompt) {
        if (!model.BuildPromptInputIds(model.Tokenizer(),
                                       "The capital of France is",
                                       sc_req.input_ids)) {
            gpu::InferenceResult r;
            r.error_msg = "selfcheck prompt framing failed";
            return r;
        }
    }
    sc_req.max_gen_tokens = std::max(opts.min_tokens, req.max_gen_tokens);
    gpu::InferenceResult r = model.Infer(sc_req);
    if (!r.ok) {
        r.selfcheck_passed = false;
        return r;
    }
    constexpr char kNeedle[] = "Paris";
    if (!r.gen_tokens.empty()) {
        if (opts.substring_match) {
            // Window semantics (Gemma parity): the canonical answer must
            // appear within the first `answer_window` generated tokens.
            const size_t n = std::min(r.gen_tokens.size(),
                                      static_cast<size_t>(opts.answer_window));
            size_t cap = 4096;
            std::string out(cap, '\0');
            if (arrow_tokenizer_decode(model.Tokenizer(),
                                       r.gen_tokens.data(), n, out.data(),
                                       &cap) == 0) {
                out.resize(cap);
                r.first_gen_text = out;
                r.selfcheck_passed =
                    r.first_gen_text.find(kNeedle) != std::string::npos;
            }
        } else {
            // Legacy strict contract: stripped equality on short prefixes.
            char buf[512];
            const int window =
                std::min(opts.answer_window, 2);
            for (int ntok = 1; ntok <= window; ++ntok) {
                if (ntok > static_cast<int>(r.gen_tokens.size())) break;
                size_t cap = sizeof(buf);
                if (arrow_tokenizer_decode(model.Tokenizer(),
                                           r.gen_tokens.data(),
                                           static_cast<size_t>(ntok), buf,
                                           &cap) != 0)
                    continue;
                std::string t(buf, cap);
                absl::StripAsciiWhitespace(&t);
                r.first_gen_text = t;
                if (r.first_gen_text == kNeedle) {
                    r.selfcheck_passed = true;
                    break;
                }
            }
            if (!r.selfcheck_passed && r.first_gen_text.empty()) {
                size_t cap = sizeof(buf);
                if (arrow_tokenizer_decode(model.Tokenizer(),
                                           r.gen_tokens.data(),
                                           std::min(static_cast<int>(
                                                        r.gen_tokens.size()),
                                                    8),
                                           buf, &cap) == 0)
                    r.first_gen_text = std::string(buf, cap);
            }
        }
    }
    lsxcommon::log::info(absl::StrFormat(
        "[selfcheck] top1_tok=%d text='%s' passed=%s",
        static_cast<int>(r.first_gen_tok_id), r.first_gen_text.c_str(),
        r.selfcheck_passed ? "true" : "false"));
    return r;
}

void FillInferenceResult(gpu::InferenceResult &r, const GenResultView &g,
                         int num_layers)
{
    r.ok = true;
    r.num_tokens = static_cast<int>(g.tokens->size());
    r.num_layers = num_layers;
    if (!g.tokens->empty()) r.first_gen_tok_id = (*g.tokens)[0];
    if (g.first_maxv != 0.0f) r.first_gen_maxv = g.first_maxv;
    r.gen_tokens = *g.tokens;
    if (g.final_logits) r.final_logits = *g.final_logits;
    r.ttft_ms = g.ttft_ms;
    r.total_time_ms = g.total_ms;
    r.tokens_per_sec =
        (g.total_ms > 0.0)
            ? (static_cast<double>(r.num_tokens) / (g.total_ms / 1000.0))
            : 0.0;
    r.eos_terminated = g.eos_terminated;
    r.kv_slots_used = g.kv_slots_used;  // -1 => caller computes its own fallback
    r.weight_resident_bytes = g.resident_bytes;
}

void FillBenchmarkResult(gpu::BenchmarkResult &r, double total_ms,
                         int num_tokens, int num_layers,
                         std::uint64_t resident_bytes)
{
    r.ok = true;
    r.num_tokens = num_tokens;
    r.num_layers = num_layers;
    r.total_time_ms = total_ms;
    r.tokens_per_sec =
        total_ms > 0.0 ? (static_cast<double>(num_tokens) / total_ms) * 1000.0
                       : 0.0;
    r.resident_bytes = resident_bytes;
}

namespace {

struct CliModel {
    std::unique_ptr<IModel> model;
    bool ok = false;
};

CliModel CreateCliModel(const std::string &model_path, FastMathMode mode)
{
    CliModel out;
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_path;
    if (mode == FastMathMode::kOn) {
        cfg.fast_math = true;   // legacy explicit fast-math opt-in
    } else {
        cfg.fast_math = false;
        cfg.fast_math_mode = mode;
    }
    out.model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!out.model || !out.model->Load()) {
        lsxcommon::log::error(
            absl::StrFormat("Failed to create/load model: %s", model_path));
        return out;
    }
    out.ok = true;
    return out;
}

} // namespace

void PrintGenerationResult(const std::string &prompt, const gpu::InferenceResult &r)
{
    lsxcommon::log::info("=== --run Results ===");
    // StrCat (not StrFormat): the abseil compile-time format checker under
    // AOCC rejects fmt-style {} placeholders, leaving literal "{}" in the
    // output (see AGENTS.md AOCC notes).
    lsxcommon::log::info(absl::StrCat("prompt   : ", prompt));
    lsxcommon::log::info(absl::StrCat("output   : ", r.output_text));
    lsxcommon::log::info(absl::StrCat(
        "tokens   : ", r.num_tokens, " at ",
        absl::StrFormat("%.1f", r.tokens_per_sec), " t/s (",
        absl::StrFormat("%.1f", r.total_time_ms), " ms)"));
}

int RunCliGeneration(const GenerationCliArgs &args)
{
    lsxcommon::log::info(
        absl::StrFormat("Running generation path on prompt: %s", args.prompt));
    CliModel m = CreateCliModel(args.model_path, args.fast_math
                                           ? FastMathMode::kOn
                                           : args.fast_math_mode);
    if (!m.ok) return 1;

    std::vector<int32_t> input_ids;
    if (!m.model->BuildPromptInputIds(m.model->Tokenizer(), args.prompt,
                                      input_ids)) {
        lsxcommon::log::error(
            absl::StrFormat("Tokenizer encode failed for prompt: %s",
                            args.prompt));
        m.model->Unload();
        return 1;
    }
    lsxcommon::ModelRequest req{std::move(input_ids), args.max_gen_tokens};
    gpu::InferenceResult result = m.model->Infer(req);
    m.model->Unload();
    if (!result.ok) {
        lsxcommon::log::error(
            absl::StrFormat("Inference failed: %s", result.error_msg));
        return 1;
    }
    PrintGenerationResult(args.prompt, result);
    return 0;
}

int RunCliSelfCheck(const GenerationCliArgs &args)
{
    lsxcommon::log::info("Running single-token self-check...");
    CliModel m = CreateCliModel(args.model_path, args.fast_math
                                           ? FastMathMode::kOn
                                           : args.fast_math_mode);
    if (!m.ok) return 1;

    std::vector<int32_t> input_ids;
    if (!m.model->BuildPromptInputIds(m.model->Tokenizer(), args.prompt,
                                      input_ids)) {
        lsxcommon::log::error("Tokenizer encode failed");
        m.model->Unload();
        return 1;
    }
    lsxcommon::ModelRequest req{std::move(input_ids), args.max_gen_tokens};
    gpu::InferenceResult result = m.model->SelfCheck(req);
    m.model->Unload();
    if (!result.ok) {
            lsxcommon::log::error(
            absl::StrFormat("SelfCheck failed: %s", result.error_msg));
        return 1;
    }
    lsxcommon::log::info("=== --selfcheck Results ===");
    lsxcommon::log::info(absl::StrFormat("first token : %d '%s'",
                                         result.first_gen_tok_id,
                                         result.first_gen_text));
    lsxcommon::log::info(
        absl::StrFormat("maxv        : %.4f", result.first_gen_maxv));
    lsxcommon::log::info(absl::StrFormat(
        "passed      : %s", result.selfcheck_passed ? "yes" : "no"));
    return result.selfcheck_passed ? 0 : 1;
}

int RunCliInspect(const std::string &model_path, bool full)
{
    CliModel m = CreateCliModel(model_path, FastMathMode::kAuto);
    if (!m.ok) return 1;
    SafetensorsModel mi = m.model->Inspect();
    if (full) {
        for (auto &shard : mi.shards) lsxcommon::print_shard(shard);
    }
    lsxcommon::print_model(mi);
    const auto &vp = m.model->VramRequirements();
    if (vp.min_bytes > 0) {
        lsxcommon::log::info(absl::StrFormat(
            "=== VRAM Profile (conversion-time, ctx=%d) ===",
            vp.context_tokens));
        lsxcommon::log::info(absl::StrFormat(
            "min     (weights only)         : %s",
            FormatBytesHuman(vp.min_bytes)));
        lsxcommon::log::info(absl::StrFormat(
            "optimal (+KV cache+workspace)  : %s",
            FormatBytesHuman(vp.optimal_bytes)));
        lsxcommon::log::info(absl::StrFormat(
            "max     (full expert placement): %s",
            FormatBytesHuman(vp.max_bytes)));
    }
    m.model->Unload();
    return 0;
}


// ── Developer calibration run (perf base-authoring) ─────────────────────────

int RunCliCalibrate(const std::string &model_path)
{
    using lsxcommon::perf::GpuPerfProfile;
    using lsxcommon::perf::LsxPerfKnobs;
    using lsxcommon::perf::ProbeGpuPerfProfile;
    using lsxcommon::perf::ResolvePerfKnobs;
    using lsxcommon::perf::SavePerfCache;
    GpuPerfProfile gpu = ProbeGpuPerfProfile();
    if (gpu.name.empty()) {
        lsxcommon::log::error("[calibrate] no CUDA device");
        return 1;
    }
    // A/B the GEMM paths on the real model: t/s with (a) TC+streaming GEMV,
    // (b) TC only, (c) bit-exact scalar. Each pass restores the toggles.
    auto run = [&](bool tc, bool gemv) -> double {
        lsxcommon::lsx_nvfp4::SetLsxTcFastMath(tc);
        lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(gemv);
        const gpu::BenchmarkResult r =
            lsxcommon::gpu::RunBenchmark(model_path, 512, tc);
        lsxcommon::lsx_nvfp4::SetLsxTcFastMath(false);
        lsxcommon::lsx_nvfp4::SetLsxSeq1Gemv(false);
        if (!r.ok) return 0.0;
        return r.tokens_per_sec;
    };
    const double t_gemv = run(true, true);
    const double t_tc = run(true, false);
    const double t_scalar = run(false, false);

    LsxPerfKnobs knobs;
    if (!perf::LoadPerfCache(gpu, knobs))
        knobs = ResolvePerfKnobs(gpu).knobs;
    knobs.tc_fast_math_default = t_tc >= t_scalar && t_scalar > 0.0;
    knobs.seq1_gemv_default = t_gemv >= t_tc && t_gemv > 0.0;
    lsxcommon::log::info(absl::StrFormat(
        "[calibrate] t/s: gemv=%.2f tc=%.2f scalar=%.2f -> tc=%d gemv=%d",
        t_gemv, t_tc, t_scalar,
        static_cast<int>(knobs.tc_fast_math_default),
        static_cast<int>(knobs.seq1_gemv_default)));
    if (!perf::SavePerfCache(gpu, knobs))
        lsxcommon::log::warn("[calibrate] perf cache save failed (vault "
                             "read-only?); snippet below still valid");
    lsxcommon::log::info(absl::StrFormat(
        "[calibrate] paste into BaseKnobs(): "
        "m.emplace(\"%s\", LsxPerfKnobs{%d, %d, %d, %d, %d, %d});",
        perf::GpuProfileKey(gpu).c_str(),
        static_cast<int>(knobs.tc_fast_math_default),
        static_cast<int>(knobs.seq1_gemv_default), knobs.gemv_max_mib,
        knobs.splitk_grid_min, knobs.expert_slots_pct,
        static_cast<int>(knobs.dual_stream_staging)));
    return 0;
}

} // namespace runtime
} // namespace lsxcommon