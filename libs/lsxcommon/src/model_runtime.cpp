#include "lsxcommon/model_runtime.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "lsxcommon/gpu_power.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/safetensors.h"
#include "lsxcommon/fsi_selfcheck.h"
#include "lsxcommon/code_selfcheck.h"
#include "lsxcommon/det_selfcheck.h"

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
                                const int32_t *ids, size_t n_ids)
{
    if (ids == nullptr || n_ids == 0) return {};
    size_t cap = kInitialDecodeCap;
    std::string out(cap, '\0');
    for (int attempt = 0; attempt < kMaxDecodeAttempts; ++attempt) {
        const int dr = arrow_tokenizer_decode(tok_db, ids, n_ids, out.data(),
                                              &cap);
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
                        n_ids));
    return {};
}

std::string DecodeGeneratedText(const ArrowTokenizerDb &tok_db,
                                const std::vector<int32_t> &ids)
{
    return DecodeGeneratedText(tok_db, ids.data(), ids.size());
}

gpu::InferenceResult RunParisSelfCheck(IModel &model, const ModelRequest &req,
                                       const ParisSelfCheckOptions &opts)
{
    ModelRequest sc_req = req;
    if (opts.frame_prompt) {
        if (!model.BuildPromptInputIds(
                model.Tokenizer(),
                opts.prompt ? opts.prompt : "The capital of France is",
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
            // appear within the first `answer_window` generated tokens. Decode
            // with capacity retry (a long window can exceed any fixed buffer).
            const size_t n = std::min(r.gen_tokens.size(),
                                      static_cast<size_t>(opts.answer_window));
            r.first_gen_text =
                DecodeGeneratedText(model.Tokenizer(), r.gen_tokens.data(), n);
            r.selfcheck_passed =
                r.first_gen_text.find(kNeedle) != std::string::npos;
        } else {
            // Legacy strict contract: stripped equality on short prefixes.
            const int window = std::min(opts.answer_window, 2);
            for (int ntok = 1; ntok <= window; ++ntok) {
                if (ntok > static_cast<int>(r.gen_tokens.size())) break;
                std::string t =
                    DecodeGeneratedText(model.Tokenizer(), r.gen_tokens.data(),
                                        static_cast<size_t>(ntok));
                if (t.empty()) continue;
                absl::StripAsciiWhitespace(&t);
                r.first_gen_text = t;
                if (r.first_gen_text == kNeedle) {
                    r.selfcheck_passed = true;
                    break;
                }
            }
            if (!r.selfcheck_passed && r.first_gen_text.empty()) {
                const size_t m = std::min<size_t>(r.gen_tokens.size(), 8);
                r.first_gen_text = DecodeGeneratedText(
                    model.Tokenizer(), r.gen_tokens.data(), m);
            }
        }
    }
    lsxcommon::log::info(absl::StrFormat(
        "[selfcheck] top1_tok=%d text='%s' passed=%s",
        static_cast<int>(r.first_gen_tok_id), r.first_gen_text.c_str(),
        r.selfcheck_passed ? "true" : "false"));
    // Cross-model R3 evidence (numeric-contract §1): emit a stable hash of the
    // selfcheck logits so the SAME build can be compared across the supported
    // GPU-model set (RTX 5090 / RTX 5060 Ti). Set LSX_SELFCHECK_EXPECT_HASH to
    // the golden machine's hash to make a cross-machine mismatch a hard failure.
    if (!r.final_logits.empty()) {
        uint64_t h = 1469598103934665603ull;   // FNV-1a 64
        const uint8_t *bytes =
            reinterpret_cast<const uint8_t *>(r.final_logits.data());
        const size_t nbytes = r.final_logits.size() * sizeof(float);
        for (size_t i = 0; i < nbytes; ++i) {
            h ^= bytes[i];
            h *= 1099511628211ull;
        }
        lsxcommon::log::info(absl::StrFormat(
            "[selfcheck] logits_hash=%016llx n=%u tok=%d",
            static_cast<unsigned long long>(h),
            static_cast<unsigned>(r.final_logits.size()),
            static_cast<int>(r.first_gen_tok_id)));
        const char *expect = std::getenv("LSX_SELFCHECK_EXPECT_HASH");
        if (expect && *expect) {
            const uint64_t want = std::strtoull(expect, nullptr, 16);
            if (want != h) {
                lsxcommon::log::error(absl::StrFormat(
                    "[selfcheck] logits hash mismatch: got %016llx want %016llx",
                    static_cast<unsigned long long>(h),
                    static_cast<unsigned long long>(want)));
                r.selfcheck_passed = false;
            }
        }
    }
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
    cfg.fast_math = false;
    cfg.fast_math_mode = mode;
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
    CliModel m = CreateCliModel(args.model_path, args.fast_math_mode);
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
    CliModel m = CreateCliModel(args.model_path, args.fast_math_mode);
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

int RunCliFsiSelfCheck(const GenerationCliArgs &args)
{
    lsxcommon::log::info("Running FSI self-check (golden parity)...");
    const char *dir = std::getenv("LSX_FSI_GOLDEN_DIR");
    if (!dir || !*dir) {
        lsxcommon::log::error(
            "LSX_FSI_GOLDEN_DIR is not set (directory with <model_type>.json)");
        return 1;
    }
    CliModel m = CreateCliModel(args.model_path, FastMathMode::kOff);
    if (!m.ok) return 1;
    const std::string golden_path =
        std::string(dir) + "/" + m.model->Type() + ".json";
    fsi::FsiGolden golden;
    std::string err;
    if (!fsi::LoadFsiGoldenFile(golden_path, golden, &err)) {
        lsxcommon::log::error(
            absl::StrFormat("FSI golden load failed: %s", err.c_str()));
        m.model->Unload();
        return 1;
    }
    const fsi::FsiResult result = fsi::RunFsiSelfCheck(*m.model, golden);
    lsxcommon::log::info("=== lsxbenchmark: fsi-selfcheck ===");
    for (const fsi::FsiCaseResult &c : result.cases)
        lsxcommon::log::info(absl::StrFormat(
            "%s  expected='%s'  extracted='%s'  %s", c.id.c_str(),
            c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "PASS" : "FAIL"));
    lsxcommon::log::info(
        absl::StrFormat("result   : %s", result.all_passed ? "PASS" : "FAIL"));
    m.model->Unload();
    return result.all_passed ? 0 : 1;
}

int RunCliCodeSelfCheck(const GenerationCliArgs &args)
{
    lsxcommon::log::info("Running CODE self-check (syntactic golden parity)...");
    const char *dir = std::getenv("LSX_CODE_GOLDEN_DIR");
    if (!dir || !*dir) {
        lsxcommon::log::error(
            "LSX_CODE_GOLDEN_DIR is not set (directory with <model_type>.json)");
        return 1;
    }
    CliModel m = CreateCliModel(args.model_path, FastMathMode::kOff);
    if (!m.ok) return 1;
    const std::string golden_path =
        std::string(dir) + "/" + m.model->Type() + ".json";
    code::CodeGolden golden;
    std::string err;
    if (!code::LoadCodeGoldenFile(golden_path, golden, &err)) {
        lsxcommon::log::error(
            absl::StrFormat("CODE golden load failed: %s", err.c_str()));
        m.model->Unload();
        return 1;
    }
    const code::CodeResult result = code::RunCodeSelfCheck(*m.model, golden);
    lsxcommon::log::info("=== lsxbenchmark: code-selfcheck ===");
    for (const code::CodeCaseResult &c : result.cases)
        lsxcommon::log::info(absl::StrFormat(
            "%s  expected='%s'  extracted='%s'  %s", c.id.c_str(),
            c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "PASS" : "FAIL"));
    lsxcommon::log::info(
        absl::StrFormat("result   : %s", result.all_passed ? "PASS" : "FAIL"));
    m.model->Unload();
    return result.all_passed ? 0 : 1;
}

int RunCliDetSelfCheck(const GenerationCliArgs &args)
{
    lsxcommon::log::info(
        "Running DET self-check (determinism + near-tie golden parity)...");
    const char *dir = std::getenv("LSX_DET_GOLDEN_DIR");
    if (!dir || !*dir) {
        lsxcommon::log::error(
            "LSX_DET_GOLDEN_DIR is not set (directory with <model_type>.json)");
        return 1;
    }
    CliModel m = CreateCliModel(args.model_path, FastMathMode::kOff);
    if (!m.ok) return 1;
    const std::string golden_path =
        std::string(dir) + "/" + m.model->Type() + ".json";
    det::DetGolden golden;
    std::string err;
    if (!det::LoadDetGoldenFile(golden_path, golden, &err)) {
        lsxcommon::log::error(
            absl::StrFormat("DET golden load failed: %s", err.c_str()));
        m.model->Unload();
        return 1;
    }
    int override = 0;
    if (const char *sr = std::getenv("LSX_DET_STRESS_RUNS")) {
        const int v = std::atoi(sr);
        if (v > 0) override = v;
    }
    const det::DetResult result = det::RunDetSelfCheck(
        *m.model, golden,
        [&m](const std::vector<int32_t> &ids) {
            return DecodeGeneratedText(m.model->Tokenizer(), ids);
        },
        override);
    lsxcommon::log::info("=== lsxbenchmark: det-selfcheck ===");
    for (const det::DetCaseResult &c : result.cases)
        lsxcommon::log::info(absl::StrFormat(
            "%s  runs=%d  deterministic=%s  expected='%s'  extracted='%s'  %s",
            c.id.c_str(), c.runs, c.deterministic ? "yes" : "no",
            c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "PASS" : "FAIL"));
    lsxcommon::log::info(
        absl::StrFormat("result   : %s", result.all_passed ? "PASS" : "FAIL"));
    m.model->Unload();
    return result.all_passed ? 0 : 1;
}

int RunCliBenchmark(const GenerationCliArgs &args)
{
    lsxcommon::log::info(absl::StrFormat(
        "Running generation benchmark on prompt: %s", args.prompt));
    CliModel m = CreateCliModel(args.model_path, args.fast_math_mode);
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
    lsxcommon::ModelRequest req;
    req.input_ids = std::move(input_ids);
    req.max_gen_tokens = args.max_gen_tokens;
    // Fixed-length benchmark: ignore EOS so t/s is comparable across numeric
    // modes and unaffected by where EOS lands.
    req.ignore_eos = true;
    // Board power is sampled on a background host thread for the Infer
    // duration only (NVML; gracefully absent on a driverless host).
    perf::GpuPowerSampler power;
    power.Start();
    gpu::InferenceResult result = m.model->Infer(req);
    const perf::PowerStats power_stats = power.Stop();
    m.model->Unload();
    if (!result.ok) {
        lsxcommon::log::error(
            absl::StrFormat("Benchmark inference failed: %s",
                            result.error_msg));
        return 1;
    }
    lsxcommon::log::info("=== lsxbenchmark: bench ===");
    // StrCat for concatenations (abseil format guard: printf-style specs
    // only inside StrFormat).
    lsxcommon::log::info(absl::StrCat("prompt   : ", args.prompt));
    lsxcommon::log::info(absl::StrCat(
        "tokens   : ", result.num_tokens, " at ",
        absl::StrFormat("%.1f", result.tokens_per_sec), " t/s (",
        absl::StrFormat("%.1f", result.total_time_ms), " ms)"));
    lsxcommon::log::info(
        absl::StrFormat("ttft     : %.2f ms", result.ttft_ms));
    if (power_stats.valid) {
        const double limit_w = perf::GpuPowerLimitW();
        lsxcommon::log::info(absl::StrFormat(
            "power    : %.1f W avg, %.1f W peak (%d samples, limit %.0f W)",
            power_stats.avg_w, power_stats.peak_w, power_stats.samples,
            limit_w));
    } else {
        lsxcommon::log::info("power    : unavailable (NVML)");
    }
    lsxcommon::log::info(absl::StrFormat(
        "resident : %.0f MiB",
        static_cast<double>(result.weight_resident_bytes) /
            (1024.0 * 1024.0)));
    return 0;
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

} // namespace runtime
} // namespace lsxcommon