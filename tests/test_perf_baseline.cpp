// Perf baseline guard (label "perf", manual command): runs every registry
// model present in the vault and checks the measured t/s against the
// committed reference tests/perf/perf_baselines.json.
//
// Semantics (docs/superpowers/specs/2026-09-11-perf-baseline-design.md):
//   - entry absent or actual > baseline  -> update the reference (INFO);
//   - actual within -3% of baseline      -> pass (run noise);
//   - actual below -3%                   -> FAIL: root-cause the regression,
//     the reference is never rewritten on degradation.
// Baselines are keyed (model_type, gpu_key, os); the measurement protocol
// (prompt, max_gen) is stored inside the entry — a protocol change re-seeds.
#include <catch2/catch_test_macros.hpp>

#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "lsxcommon/core.h"
#include "lsxcommon/gpu_perf.h"
#include "lsxcommon/gpu_power.h"
#include "lsxcommon/model_engine.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/model.h"
#include "lsxcommon/path_registry.h"
#include "lsxcommon/perf_baseline.h"

#ifdef _WIN32
constexpr const char *kThisOs = "windows";
#else
constexpr const char *kThisOs = "linux";
#endif

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

namespace fs = std::filesystem;
namespace pb = lsxcommon::perf_baseline;

struct PerfModel {
    const char *model_type;
    const char *vault_subdir;
    const char *prompt;
    int max_gen;
};

// All vault models that can run on this machine. The qwen3.8-flash-next
// exclusion (conversion OOM) is obsolete since the GenContextBytes cache
// sizing fix (2026-09-12) — the vault loads and benchmarks.
const PerfModel kGemma = {
    "gemma-4-26b-a4b-it", "gemma-4-26B-A4B-it",
    "Write a detailed long essay about the history of computing, from "
    "Babbage to modern GPUs.",
    2048};
// Gemma-4-31B-it (dense): the same 2048-token essay protocol.
const PerfModel kGemmaDense = {
    "gemma-4-31b-it", "gemma-4-31B-it",
    "Write a detailed long essay about the history of computing, from "
    "Babbage to modern GPUs.",
    2048};
const PerfModel kGlmFlash = {
    "glm-5.3-flash", "GLM-5.3-Flash", "The capital of France is", 2048};
// glm-5.2 and qwen3.5-9b are registered but not present in the local vault:
// their targets SKIP until the model is converted, then seed the reference.
const PerfModel kGlm52 = {
    "glm-5.2", "GLM-5.2", "The capital of France is", 2048};
const PerfModel kDeepseek = {
    "deepseek-v4-flash", "DeepSeek-V4-Flash", "The capital of France is",
    2048};
const PerfModel kDeepseekV41 = {
    "deepseek-v4.1-flash", "DeepSeek-V4.1-Flash", "The capital of France is",
    2048};
const PerfModel kQwen35 = {
    "qwen3.5-35b-a3b", "Qwen3.5-35B-A3B", "The capital of France is", 2048};
const PerfModel kQwen36 = {
    "qwen3.6-35b-a3b", "Qwen3.6-35B-A3B", "The capital of France is", 2048};
const PerfModel kQwen38 = {
    "qwen3.8-flash-next", "Qwen3.8-Flash-Next", "The capital of France is",
    2048};
const PerfModel kKimi = {
    "kimi-k2.7-code", "Kimi-K2.7-Code", "The capital of France is", 2048};
const PerfModel kQwen27 = {
    "qwen3.8-27b", "Qwen3.8-27B", "The capital of France is", 2048};
const PerfModel kQwen35_9B = {
    "qwen3.5-9b", "Qwen3.5-9B", "The capital of France is", 2048};
const PerfModel kNemotron = {
    "nemotron-3.5-lightning-30b-a3b",
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "The capital of France is",
    2048};
const PerfModel kNemotronNvfp4 = {
    "nemotron-3.5-lightning-30b-a3b-nvfp4",
    "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "The capital of France is",
    2048};
const PerfModel kGigachat = {
    "gigachat3.5-432b-a28b-reasoning", "GigaChat3.5-432B-A28B-Reasoning",
    "The capital of France is", 2048};
const PerfModel kOrnith = {
    "ornith-1.5-35b-a3b", "Ornith-1.5-35B-A3B", "The capital of France is",
    2048};
const PerfModel kGranite = {
    "granite-4.2-30b", "granite-4.2-30b", "The capital of France is", 2048};
const PerfModel kMiMo = {
    "mimo-v2.6-flash-rl", "MiMo-V2.6-Flash-RL", "The capital of France is",
    2048};
const PerfModel kPhi = {
    "phi-4", "phi-4", "The capital of France is", 2048};
const PerfModel kXing = {
    "xing4.0-29b-a4b", "Xing4.0-29B-A4B", "The capital of France is", 2048};
const PerfModel kAlice = {
    "alice-foundation-80b-a3b", "AliceAI-Foundation-80B-A3B-Base",
    "The capital of France is", 2048};
const PerfModel kMistral = {
    "mistral-small-4-119b", "Mistral-Small-4-119B-2603-NVFP4",
    "The capital of France is", 2048};

std::string BaselinePath()
{
    return std::string(LSX_SOURCE_DIR) + "/tests/perf/perf_baselines.json";
}

pb::PerfBaselineTable LoadTable()
{
    std::ifstream f(BaselinePath(), std::ios::binary);
    if (!f.is_open()) return {};
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    auto t = pb::ParsePerfBaselines(json);
    return t.value_or(pb::PerfBaselineTable{});
}

void SaveTable(const pb::PerfBaselineTable &t)
{
    const fs::path p(BaselinePath());
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    REQUIRE(f.is_open());
    f << pb::SerializePerfBaselines(t);
}

std::string TodayIso()
{
    const std::time_t now = std::time(nullptr);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", std::localtime(&now));
    return buf;
}

// The environment oracle lives in the same table as a synthetic entry, so the
// committed JSON schema is unchanged: model_type "__gpu_probe__", mode
// "probe", tokens_per_sec = GB/s. Like a model reference it only ratchets up.
constexpr const char *kProbeModel = "__gpu_probe__";
constexpr const char *kProbeMode = "probe";
// The probe size is part of its protocol: a 64 MiB copy fits in the 96 MB L2
// and measured 2165 GB/s (L2 speed), so only the DRAM-bound 512 MiB protocol
// is comparable and an entry from another size is ignored.
constexpr const char *kProbePrompt = "d2d-copy-512MiB";

std::optional<double> ProbeReference(const pb::PerfBaselineTable &t,
                                     const std::string &gpu_key)
{
    const pb::PerfBaselineEntry *e =
        pb::FindEntry(t, kProbeModel, gpu_key, kThisOs, kProbeMode);
    if (e == nullptr || e->prompt != kProbePrompt ||
        !(e->tokens_per_sec > 0.0))
        return std::nullopt;
    return e->tokens_per_sec;
}

// Records an improved probe reading; a lower one is kept out of the reference
// so the oracle stays an upper bound on what this card can do.
void UpdateProbeReference(double gbs, const std::string &gpu_key)
{
    pb::PerfBaselineTable table = LoadTable();
    const std::optional<double> ref = ProbeReference(table, gpu_key);
    if (ref.has_value() && gbs <= *ref) return;
    pb::PerfBaselineEntry upsert;
    upsert.model_type = kProbeModel;
    upsert.gpu_key = gpu_key;
    upsert.os = kThisOs;
    upsert.mode = kProbeMode;
    upsert.tokens_per_sec = gbs;
    upsert.max_gen = 0;
    upsert.prompt = kProbePrompt;
    upsert.date = TodayIso();
    upsert.commit = "-";
    bool replaced = false;
    for (auto &e : table.entries) {
        if (e.model_type == kProbeModel && e.gpu_key == gpu_key &&
            e.os == kThisOs && e.mode == kProbeMode) {
            e = upsert;
            replaced = true;
            break;
        }
    }
    if (!replaced) table.entries.push_back(upsert);
    table.format_version = pb::kPerfBaselineFormatVersion;
    SaveTable(table);
    INFO("gpu probe reference " << (replaced ? "updated" : "recorded") << " -> "
                                << gbs << " GB/s");
}

// Measures one model; returns the lsxbenchmark-style t/s (the value is
// computed by the model itself from its generation wall time) plus the
// ACTUAL generated token count (EOS timing is part of the measurement
// protocol: t/s is only comparable at equal fact length).
struct MeasureResult {
    double tokens_per_sec = 0.0;
    int gen_tokens = 0;
    double ttft_ms = 0.0;
    double avg_power_w = 0.0;
    double peak_power_w = 0.0;
    int power_samples = 0;
};

// ctest pins the vault subdir via env (one model per process); a direct binary
// run falls back to the struct value.
fs::path ModelDir(const PerfModel &m)
{
    const char *subdir_env = std::getenv("LSX_PERF_VAULT_SUBDIR");
    return fs::path(lsxcommon::lsx_model_models_dir()) /
           (subdir_env != nullptr ? subdir_env : m.vault_subdir);
}

std::optional<MeasureResult> MeasureModel(const PerfModel &m,
                                          const std::string &mode)
{
#ifndef NDEBUG
    // A Debug build measures an unoptimized binary: the t/s is far below the
    // Release reference and would be judged DEGRADED. There is no useful Debug
    // perf baseline, so SKIP (ctest SKIP_RETURN_CODE 4) instead of failing.
    SKIP("perf baseline requires a Release build (NDEBUG)");
#endif
    const fs::path model_dir = ModelDir(m);
    std::error_code ec;
    if (!fs::exists(model_dir / "logestix" / "model.index.json", ec)) {
        SKIP(("vault model missing: " + model_dir.string()).c_str());
    }

    // Same logging core as every app (lsxbenchmark/lsxfabric). Without it the
    // default spdlog logger is the SYNCHRONOUS single-thread one, and the measured
    // decode differed from the canonical lsxbenchmark run by 9% on the same
    // binary and prompt (Xing-4.0: 134.4 vs 146.6 t/s at 2048, reproduced both
    // ways). A baseline guard that measures a path no user runs cannot see a
    // regression in the path they do run, so initialize once per process.
    static const bool core_ready = [] {
        lsxcommon::Core::Initialize("test_perf_baseline.log");
        return true;
    }();
    (void)core_ready;
    lsxcommon::InferenceEngine::Session engine;
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    cfg.model_type = m.model_type;
    // R3 = explicit bit-exact (kOff); perf = per-GPU max-speed policy (kAuto).
    cfg.fast_math_mode = (mode == "r3") ? lsxcommon::FastMathMode::kOff
                                        : lsxcommon::FastMathMode::kAuto;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!model) FAIL("factory returned nullptr for " << m.model_type);
    if (!model->Load()) FAIL("model Load failed for " << m.model_type);

    std::vector<int32_t> input_ids;
    if (!model->BuildPromptInputIds(model->Tokenizer(), m.prompt, input_ids))
        FAIL("prompt tokenization failed for " << m.model_type);
    lsxcommon::ModelRequest req;
    req.input_ids = std::move(input_ids);
    req.max_gen_tokens = m.max_gen;
    // Fixed-length measurement: ignore EOS so exactly max_gen steps decode and
    // t/s is comparable across numeric modes (perf/r3) and across bit-drift.
    req.ignore_eos = true;

    lsxcommon::perf::GpuPowerSampler power;
    power.Start();
    const auto r = model->Infer(req);
    const lsxcommon::perf::PowerStats ps = power.Stop();
    model->Unload();
    if (!r.ok) FAIL("Infer failed for " << m.model_type << ": "
                                        << r.error_msg);
    return MeasureResult{r.tokens_per_sec,
                         static_cast<int>(r.gen_tokens.size()),
                         r.ttft_ms,
                         ps.valid ? ps.avg_w : 0.0,
                         ps.valid ? ps.peak_w : 0.0,
                         ps.samples};
}

void CheckAgainstBaseline(const PerfModel &m, const MeasureResult &mr,
                          const std::string &mode, double probe_gbs,
                          std::optional<double> probe_ref)
{
    pb::PerfBaselineTable table = LoadTable();
    const std::string gpu_key = lsxcommon::perf::GpuProfileKey(
        lsxcommon::perf::ProbeGpuPerfProfile());
    REQUIRE_FALSE(gpu_key.empty());
    const double actual = mr.tokens_per_sec;

    const pb::PerfBaselineEntry *entry =
        pb::FindEntry(table, m.model_type, gpu_key, kThisOs, mode);
    std::optional<double> baseline;
    int baseline_len = 0;
    if (entry != nullptr && entry->max_gen == m.max_gen &&
        entry->prompt == m.prompt) {
        baseline = entry->tokens_per_sec;
        baseline_len = entry->gen_tokens;
    }
    const pb::PerfVerdict verdict =
        pb::JudgePerfLen(baseline, actual, mr.gen_tokens, baseline_len);

    switch (verdict) {
    case pb::PerfVerdict::kDegrade: {
        // The model cannot certify its own environment. Ask the independent
        // copy-bandwidth probe first: if it is down at least as much as the
        // model, this is a thermal/power state, not a code regression.
        const double model_ratio = actual / *baseline;
        const double probe_ratio =
            (probe_ref.has_value() && *probe_ref > 0.0 && probe_gbs > 0.0)
                ? probe_gbs / *probe_ref
                : 0.0;
        if (pb::AttributeDegrade(model_ratio, probe_ratio) ==
            pb::PerfCause::kEnvironment) {
            WARN(m.model_type
                 << " [" << gpu_key << "/" << kThisOs << "/" << mode
                 << "]: ENVIRONMENT (not a regression) baseline=" << *baseline
                 << " t/s, actual=" << actual << " t/s ("
                 << (100.0 * model_ratio) << "%), probe " << probe_gbs
                 << " vs " << *probe_ref << " GB/s ("
                 << (100.0 * probe_ratio) << "%). The card is in a lower "
                    "power/thermal state; re-run when the probe is nominal.");
            break;
        }
        FAIL(m.model_type << " [" << gpu_key << "/" << kThisOs
                          << "]: DEGRADED baseline=" << *baseline
                          << " t/s, actual=" << actual
                          << " t/s (" << (100.0 * actual / *baseline)
                          << "%) at equal fact length " << mr.gen_tokens
                          << " tok. Root-cause the regression; the "
                             "reference is not updated."
                          << " (probe " << probe_gbs << " vs "
                          << (probe_ref.has_value() ? *probe_ref : 0.0)
                          << " GB/s: the environment is not the cause)");
        break;
    }
    case pb::PerfVerdict::kUpdate: {
        pb::PerfBaselineEntry upsert;
        upsert.model_type = m.model_type;
        upsert.gpu_key = gpu_key;
        upsert.os = kThisOs;
        upsert.mode = mode;
        upsert.tokens_per_sec = actual;
        upsert.max_gen = m.max_gen;
        upsert.prompt = m.prompt;
        upsert.date = TodayIso();
        upsert.commit = "-";
        upsert.gen_tokens = mr.gen_tokens;
        upsert.ttft_ms = mr.ttft_ms;
        upsert.avg_power_w = mr.avg_power_w;
        upsert.peak_power_w = mr.peak_power_w;
        upsert.power_samples = mr.power_samples;
        bool replaced = false;
        for (auto &e : table.entries) {
            if (e.model_type == upsert.model_type && e.gpu_key == gpu_key &&
                e.os == kThisOs && e.mode == mode) {
                e = upsert;
                replaced = true;
                break;
            }
        }
        if (!replaced) table.entries.push_back(upsert);
        // A rewritten file always carries the current schema (new fields).
        table.format_version = pb::kPerfBaselineFormatVersion;
        SaveTable(table);
        INFO(m.model_type << ": reference updated -> " << actual << " t/s"
                          << " at " << mr.gen_tokens << " tok"
                          << (replaced ? (baseline.has_value() &&
                                          baseline_len != 0 &&
                                          mr.gen_tokens != baseline_len)
                                           ? " (length re-seed)"
                                           : " (improvement)"
                                       : " (new entry)"));
        break;
    }
    case pb::PerfVerdict::kPass: {
        // Backfill TTFT/power evidence missing from legacy entries so a
        // normal (non-improving) run still persists the new metrics. The
        // baseline t/s and measurement protocol are never touched.
        if (entry != nullptr) {
            pb::PerfBaselineEntry updated = *entry;
            if (pb::BackfillEvidence(updated, mr.ttft_ms, mr.avg_power_w,
                                     mr.peak_power_w, mr.power_samples)) {
                for (auto &e : table.entries) {
                    if (e.model_type == m.model_type && e.gpu_key == gpu_key &&
                        e.os == kThisOs && e.mode == mode) {
                        e = updated;
                        break;
                    }
                }
                table.format_version = pb::kPerfBaselineFormatVersion;
                SaveTable(table);
                INFO(m.model_type << ": evidence backfilled (ttft=" << mr.ttft_ms
                                  << " ms, power=" << mr.avg_power_w << "/"
                                  << mr.peak_power_w << " W)");
            }
        }
        INFO(m.model_type << ": within tolerance, baseline=" << *baseline
                          << " t/s, actual=" << actual << " t/s at "
                          << mr.gen_tokens << " tok");
        break;
    }
    }
}

// The GPU heat-soaks within one 35-45 s decode and then settles ~11% below its
// max SM clock (measured 2812/3150 MHz), costing ~4% of t/s. Back-to-back perf
// targets therefore read low and look DEGRADED against a reference recorded on
// a cool card. Cool between targets: CoolDownGpu waits until the temperature
// plateaus (as cool as the ambient allows; it takes ~2-3 min to return to the
// ~40 C idle floor here). LSX_PERF_COOLDOWN_TIMEOUT_S caps the wait (0
// disables). Called once per target, before the perf->r3 pair, so the
// within-target protocol the baselines were recorded with is unchanged.
void CooldownBetweenTargets()
{
    int timeout_s = 300;
    if (const char *e = std::getenv("LSX_PERF_COOLDOWN_TIMEOUT_S"))
        timeout_s = std::atoi(e);
    lsxcommon::perf::CoolDownGpu(timeout_s);
}

void RunPerfModel(const PerfModel &m)
{
    // Cool before the perf->r3 pair, but only when the model is present (an
    // absent vault SKIPs immediately — no point holding the GPU cool first).
    std::error_code ec;
    if (!fs::exists(ModelDir(m) / "logestix" / "model.index.json", ec))
        return;
    CooldownBetweenTargets();
    // Environment oracle, measured once per target on the same cool card the
    // model is about to run on.
    const std::string gpu_key = lsxcommon::perf::GpuProfileKey(
        lsxcommon::perf::ProbeGpuPerfProfile());
    const double probe_gbs = lsxcommon::perf::MeasureMemoryBandwidthGbs();
    const std::optional<double> probe_ref = ProbeReference(LoadTable(), gpu_key);
    if (probe_gbs > 0.0) UpdateProbeReference(probe_gbs, gpu_key);
    // Two numeric modes, one mechanism: "perf" (kAuto) and "r3" (kOff,
    // bit-exact). Both are recorded in perf_baselines.json, keyed by mode.
    for (const char *mode : {"perf", "r3"}) {
        const auto actual = MeasureModel(m, mode);
        if (actual.has_value())
            CheckAgainstBaseline(m, *actual, mode, probe_gbs, probe_ref);
    }
}

} // namespace

TEST_CASE("perf baseline: xing4.0-29b-a4b", "[perf]")
{
    RunPerfModel(kXing);
}

TEST_CASE("perf baseline: gemma-4-26b-a4b-it", "[perf]")
{
    RunPerfModel(kGemma);
}

TEST_CASE("perf baseline: gemma-4-31b-it", "[perf]")
{
    RunPerfModel(kGemmaDense);
}

TEST_CASE("perf baseline: glm-5.3-flash", "[perf]")
{
    RunPerfModel(kGlmFlash);
}

TEST_CASE("perf baseline: deepseek-v4-flash", "[perf]")
{
    RunPerfModel(kDeepseek);
}

TEST_CASE("perf baseline: deepseek-v4.1-flash", "[perf]")
{
    RunPerfModel(kDeepseekV41);
}

TEST_CASE("perf baseline: qwen3.5-35b-a3b", "[perf]")
{
    RunPerfModel(kQwen35);
}

TEST_CASE("perf baseline: qwen3.6-35b-a3b", "[perf]")
{
    RunPerfModel(kQwen36);
}

TEST_CASE("perf baseline: qwen3.8-flash-next", "[perf]")
{
    RunPerfModel(kQwen38);
}

TEST_CASE("perf baseline: kimi-k2.7-code", "[perf]")
{
    RunPerfModel(kKimi);
}

TEST_CASE("perf baseline: nemotron-3.5-lightning-30b-a3b", "[perf]")
{
    RunPerfModel(kNemotron);
}

TEST_CASE("perf baseline: nemotron-3.5-lightning-30b-a3b-nvfp4", "[perf]")
{
    RunPerfModel(kNemotronNvfp4);
}

TEST_CASE("perf baseline: qwen3.8-27b", "[perf]")
{
    RunPerfModel(kQwen27);
}

TEST_CASE("perf baseline: glm-5.2", "[perf]")
{
    RunPerfModel(kGlm52);
}

TEST_CASE("perf baseline: qwen3.5-9b", "[perf]")
{
    RunPerfModel(kQwen35_9B);
}

TEST_CASE("perf baseline: gigachat3.5-432b-a28b-reasoning", "[perf]")
{
    RunPerfModel(kGigachat);
}

TEST_CASE("perf baseline: ornith-1.5-35b-a3b", "[perf]")
{
    RunPerfModel(kOrnith);
}

TEST_CASE("perf baseline: granite-4.2-30b", "[perf]")
{
    RunPerfModel(kGranite);
}

TEST_CASE("perf baseline: mimo-v2.6-flash-rl", "[perf]")
{
    RunPerfModel(kMiMo);
}

TEST_CASE("perf baseline: phi-4", "[perf]")
{
    RunPerfModel(kPhi);
}

TEST_CASE("perf baseline: alice-foundation-80b-a3b", "[perf]")
{
    RunPerfModel(kAlice);
}

TEST_CASE("perf baseline: mistral-small-4-119b", "[perf]")
{
    RunPerfModel(kMistral);
}
