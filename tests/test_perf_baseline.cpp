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

#include "lsxcommon/gpu_perf.h"
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
const PerfModel kGlmFlash = {
    "glm-5.3-flash", "GLM-5.3-Flash", "The capital of France is", 2048};
const PerfModel kDeepseek = {
    "deepseek-v4-flash", "DeepSeek-V4-Flash", "The capital of France is",
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

// Measures one model; returns the lsxbenchmark-style t/s (the value is
// computed by the model itself from its generation wall time).
std::optional<double> MeasureModel(const PerfModel &m)
{
    // The ctest entry pins the vault subdir via env (per-model process);
    // direct binary runs fall back to the struct value.
    const char *subdir_env = std::getenv("LSX_PERF_VAULT_SUBDIR");
    const fs::path model_dir =
        fs::path(lsxcommon::lsx_model_models_dir()) /
        (subdir_env != nullptr ? subdir_env : m.vault_subdir);
    std::error_code ec;
    if (!fs::exists(model_dir / "logestix" / "model.index.json", ec)) {
        SKIP(("vault model missing: " + model_dir.string()).c_str());
    }

    lsxcommon::InferenceEngine::Session engine;
    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = model_dir.string();
    cfg.model_type = m.model_type;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    if (!model) FAIL("factory returned nullptr for " << m.model_type);
    if (!model->Load()) FAIL("model Load failed for " << m.model_type);

    std::vector<int32_t> input_ids;
    if (!model->BuildPromptInputIds(model->Tokenizer(), m.prompt, input_ids))
        FAIL("prompt tokenization failed for " << m.model_type);
    lsxcommon::ModelRequest req{std::move(input_ids), m.max_gen};

    const auto r = model->Infer(req);
    model->Unload();
    if (!r.ok) FAIL("Infer failed for " << m.model_type << ": "
                                        << r.error_msg);
    return r.tokens_per_sec;
}

void CheckAgainstBaseline(const PerfModel &m, double actual)
{
    pb::PerfBaselineTable table = LoadTable();
    const std::string gpu_key = lsxcommon::perf::GpuProfileKey(
        lsxcommon::perf::ProbeGpuPerfProfile());
    REQUIRE_FALSE(gpu_key.empty());

    const pb::PerfBaselineEntry *entry =
        pb::FindEntry(table, m.model_type, gpu_key, kThisOs);
    std::optional<double> baseline;
    if (entry != nullptr && entry->max_gen == m.max_gen &&
        entry->prompt == m.prompt) {
        baseline = entry->tokens_per_sec;
    }
    const pb::PerfVerdict verdict = pb::JudgePerf(baseline, actual);

    switch (verdict) {
    case pb::PerfVerdict::kDegrade:
        FAIL(m.model_type << " [" << gpu_key << "/" << kThisOs
                          << "]: DEGRADED baseline=" << *baseline
                          << " t/s, actual=" << actual
                          << " t/s (" << (100.0 * actual / *baseline)
                          << "%). Root-cause the regression; the reference "
                             "is not updated.");
        break;
    case pb::PerfVerdict::kUpdate: {
        pb::PerfBaselineEntry upsert;
        upsert.model_type = m.model_type;
        upsert.gpu_key = gpu_key;
        upsert.os = kThisOs;
        upsert.tokens_per_sec = actual;
        upsert.max_gen = m.max_gen;
        upsert.prompt = m.prompt;
        upsert.date = TodayIso();
        upsert.commit = "-";
        bool replaced = false;
        for (auto &e : table.entries) {
            if (e.model_type == upsert.model_type && e.gpu_key == gpu_key &&
                e.os == kThisOs) {
                e = upsert;
                replaced = true;
                break;
            }
        }
        if (!replaced) table.entries.push_back(upsert);
        SaveTable(table);
        INFO(m.model_type << ": reference updated -> " << actual << " t/s"
                          << (replaced ? " (improvement)" : " (new entry)"));
        break;
    }
    case pb::PerfVerdict::kPass:
        INFO(m.model_type << ": within tolerance, baseline=" << *baseline
                          << " t/s, actual=" << actual << " t/s");
        break;
    }
}

void RunPerfModel(const PerfModel &m)
{
    const auto actual = MeasureModel(m);
    if (actual.has_value()) CheckAgainstBaseline(m, *actual);
}

} // namespace

TEST_CASE("perf baseline: gemma-4-26b-a4b-it", "[perf]")
{
    RunPerfModel(kGemma);
}

TEST_CASE("perf baseline: glm-5.3-flash", "[perf]")
{
    RunPerfModel(kGlmFlash);
}

TEST_CASE("perf baseline: deepseek-v4-flash", "[perf]")
{
    RunPerfModel(kDeepseek);
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
