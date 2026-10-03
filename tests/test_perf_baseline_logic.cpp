// Perf baseline reference: pure decision logic + JSON round-trip.
// Host-only, fast label — no CUDA, no model weights involved.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/perf_baseline.h"

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

#include <algorithm>
#include <fstream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using lsxcommon::perf_baseline::JudgePerf;
using lsxcommon::perf_baseline::JudgePerfLen;
using lsxcommon::perf_baseline::ParsePerfBaselines;
using lsxcommon::perf_baseline::PerfBaselineEntry;
using lsxcommon::perf_baseline::PerfBaselineTable;
using lsxcommon::perf_baseline::PerfVerdict;
using lsxcommon::perf_baseline::SerializePerfBaselines;
using lsxcommon::perf_baseline::kPerfBaselineFormatVersion;

TEST_CASE("JudgePerf: absent baseline updates", "[perf_baseline]")
{
    std::optional<double> absent;
    REQUIRE(JudgePerf(absent, 98.1) == PerfVerdict::kUpdate);
    REQUIRE(JudgePerf(absent, 0.1) == PerfVerdict::kUpdate);
}

TEST_CASE("JudgePerf: improvement updates", "[perf_baseline]")
{
    const std::optional<double> base{98.1};
    REQUIRE(JudgePerf(base, 99.0) == PerfVerdict::kUpdate);
    REQUIRE(JudgePerf(base, 120.0) == PerfVerdict::kUpdate);
    // Strictly greater: an equal value is not an improvement.
    REQUIRE(JudgePerf(base, 98.1) == PerfVerdict::kPass);
}

TEST_CASE("JudgePerf: within tolerance passes", "[perf_baseline]")
{
    const std::optional<double> base{100.0};
    // 3% tolerance: >= 97.0 passes.
    REQUIRE(JudgePerf(base, 99.9) == PerfVerdict::kPass);
    REQUIRE(JudgePerf(base, 97.0) == PerfVerdict::kPass);
    // Just below the boundary degrades.
    REQUIRE(JudgePerf(base, 96.99) == PerfVerdict::kDegrade);
    REQUIRE(JudgePerf(base, 50.0) == PerfVerdict::kDegrade);
}

TEST_CASE("JudgePerf: custom tolerance", "[perf_baseline]")
{
    const std::optional<double> base{100.0};
    REQUIRE(JudgePerf(base, 91.0, 10.0) == PerfVerdict::kPass);
    REQUIRE(JudgePerf(base, 89.0, 10.0) == PerfVerdict::kDegrade);
}

TEST_CASE("JudgePerf: generation-length mismatch re-seeds, not degrades",
          "[perf_baseline]")
{
    // t/s depends on the actual generated length (EOS timing shifts the
    // KV-attention cost curve). A run whose fact length differs from the
    // baseline protocol's fact length is NOT comparable: it re-seeds the
    // reference (kUpdate) regardless of the direction of the change.
    const std::optional<double> base{134.84};
    REQUIRE(JudgePerfLen(base, 125.0, 1484, 2048) == PerfVerdict::kUpdate);
    REQUIRE(JudgePerfLen(base, 160.0, 300, 2048) == PerfVerdict::kUpdate);
    // Same fact length: normal verdicts apply.
    REQUIRE(JudgePerfLen(base, 133.0, 2048, 2048) == PerfVerdict::kPass);
    REQUIRE(JudgePerfLen(base, 125.0, 2048, 2048) == PerfVerdict::kDegrade);
    // Unknown baseline length (legacy v1 entries) falls back to the
    // max_gen protocol check: equal protocol = comparable.
    REQUIRE(JudgePerfLen(base, 133.0, 2048, 0) == PerfVerdict::kPass);
    REQUIRE(JudgePerfLen(base, 125.0, 2048, 0) == PerfVerdict::kDegrade);
}

TEST_CASE("Perf baselines: JSON round-trip", "[perf_baseline]")
{
    PerfBaselineTable t;
    t.format_version = kPerfBaselineFormatVersion;
    PerfBaselineEntry e;
    e.model_type = "gemma-4-26b-a4b-it";
    e.gpu_key = "rtx 5090";
    e.os = "linux";
    e.tokens_per_sec = 98.12;
    e.max_gen = 2048;
    e.prompt = "Write a detailed long essay";
    e.date = "2026-09-11";
    e.commit = "8c3a20e";
    e.gen_tokens = 2048;
    e.ttft_ms = 123.45;
    e.avg_power_w = 412.5;
    e.peak_power_w = 601.0;
    e.power_samples = 137;
    t.entries.push_back(e);
    e.model_type = "glm-5.3-flash";
    e.os = "windows";
    e.tokens_per_sec = 32.1;
    e.gen_tokens = 0;   // legacy-style entry
    e.ttft_ms = 0.0;
    e.avg_power_w = 0.0;
    e.peak_power_w = 0.0;
    e.power_samples = 0;
    t.entries.push_back(e);

    const auto parsed = ParsePerfBaselines(SerializePerfBaselines(t));
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->format_version == kPerfBaselineFormatVersion);
    REQUIRE(parsed->entries.size() == 2);
    REQUIRE(parsed->entries[0].model_type == "gemma-4-26b-a4b-it");
    REQUIRE(parsed->entries[0].gpu_key == "rtx 5090");
    REQUIRE(parsed->entries[0].os == "linux");
    REQUIRE(parsed->entries[0].tokens_per_sec == 98.12);
    REQUIRE(parsed->entries[0].max_gen == 2048);
    REQUIRE(parsed->entries[0].prompt == "Write a detailed long essay");
    REQUIRE(parsed->entries[0].date == "2026-09-11");
    REQUIRE(parsed->entries[0].commit == "8c3a20e");
    REQUIRE(parsed->entries[0].gen_tokens == 2048);
    REQUIRE(parsed->entries[0].ttft_ms == 123.45);
    REQUIRE(parsed->entries[0].avg_power_w == 412.5);
    REQUIRE(parsed->entries[0].peak_power_w == 601.0);
    REQUIRE(parsed->entries[0].power_samples == 137);
    REQUIRE(parsed->entries[1].os == "windows");
    REQUIRE(parsed->entries[1].tokens_per_sec == 32.1);
    REQUIRE(parsed->entries[1].gen_tokens == 0);
    // Absent power/TTFT fields degrade to the "unknown" sentinels.
    REQUIRE(parsed->entries[1].ttft_ms == 0.0);
    REQUIRE(parsed->entries[1].avg_power_w == 0.0);
    REQUIRE(parsed->entries[1].peak_power_w == 0.0);
    REQUIRE(parsed->entries[1].power_samples == 0);
}

TEST_CASE("Perf baselines: serializer escapes strings so output stays valid "
          "JSON",
          "[perf_baseline]")
{
    // The writer is hand-rolled; quotes/backslashes/control chars in any string
    // field (e.g. a user --prompt) must not corrupt the document.
    PerfBaselineTable t;
    PerfBaselineEntry e;
    e.model_type = "model\"x";
    e.gpu_key = "rtx\\5090";
    e.os = "linux";
    e.mode = "perf";
    e.tokens_per_sec = 1.0;
    e.max_gen = 16;
    e.prompt = "say \"hi\"\nand\\back\ttab";
    e.date = "2026-01-01";
    e.commit = "a\"b\\c";
    t.entries.push_back(e);

    const auto parsed = ParsePerfBaselines(SerializePerfBaselines(t));
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->entries.size() == 1);
    REQUIRE(parsed->entries[0].model_type == e.model_type);
    REQUIRE(parsed->entries[0].gpu_key == e.gpu_key);
    REQUIRE(parsed->entries[0].prompt == e.prompt);
    REQUIRE(parsed->entries[0].commit == e.commit);
}

TEST_CASE("Perf baselines: v2 file without TTFT/power still parses",
          "[perf_baseline]")
{
    const std::string v2 = R"({
  "format_version": 2,
  "entries": [
    {
      "model_type": "qwen3.8-27b",
      "gpu_key": "rtx 5090",
      "os": "linux",
      "mode": "perf",
      "tokens_per_sec": 30.6,
      "max_gen": 2048,
      "prompt": "The capital of France is",
      "date": "2026-09-13",
      "commit": "-",
      "gen_tokens": 2048
    }
  ]
})";
    const auto parsed = ParsePerfBaselines(v2);
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->entries.size() == 1);
    REQUIRE(parsed->entries[0].gen_tokens == 2048);
    REQUIRE(parsed->entries[0].ttft_ms == 0.0);
    REQUIRE(parsed->entries[0].power_samples == 0);
}

TEST_CASE("Perf baselines: v1 file without gen_tokens still parses",
          "[perf_baseline]")
{
    const std::string v1 = R"({
  "format_version": 1,
  "entries": [
    {
      "model_type": "qwen3.5-35b-a3b",
      "gpu_key": "rtx 5090",
      "os": "linux",
      "tokens_per_sec": 134.84,
      "max_gen": 2048,
      "prompt": "The capital of France is",
      "date": "2026-09-11",
      "commit": "-"
    }
  ]
})";
    const auto parsed = ParsePerfBaselines(v1);
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->entries.size() == 1);
    REQUIRE(parsed->entries[0].tokens_per_sec == 134.84);
    REQUIRE(parsed->entries[0].gen_tokens == 0);
}

TEST_CASE("Perf baselines: evidence backfill fills legacy entries only",
          "[perf_baseline]")
{
    using lsxcommon::perf_baseline::BackfillEvidence;
    PerfBaselineEntry legacy;
    legacy.tokens_per_sec = 100.0;   // must never change
    legacy.gen_tokens = 2048;
    const bool changed = BackfillEvidence(legacy, 123.45, 400.0, 600.0, 500);
    REQUIRE(changed);
    REQUIRE(legacy.ttft_ms == 123.45);
    REQUIRE(legacy.avg_power_w == 400.0);
    REQUIRE(legacy.peak_power_w == 600.0);
    REQUIRE(legacy.power_samples == 500);
    REQUIRE(legacy.tokens_per_sec == 100.0);
    REQUIRE(legacy.gen_tokens == 2048);

    // Already-recorded evidence is not overwritten.
    PerfBaselineEntry recorded = legacy;
    REQUIRE_FALSE(BackfillEvidence(recorded, 999.0, 1.0, 2.0, 3));
    REQUIRE(recorded.ttft_ms == 123.45);
    REQUIRE(recorded.avg_power_w == 400.0);
    REQUIRE(recorded.power_samples == 500);

    // Missing NVML on the fresh run: TTFT is still backfilled, power is not.
    PerfBaselineEntry no_power;
    no_power.ttft_ms = 0.0;
    no_power.power_samples = 0;
    REQUIRE(BackfillEvidence(no_power, 50.0, 0.0, 0.0, 0));
    REQUIRE(no_power.ttft_ms == 50.0);
    REQUIRE(no_power.power_samples == 0);
}

TEST_CASE("Perf baselines: empty table and garbage rejected", "[perf_baseline]")
{
    REQUIRE(ParsePerfBaselines("not json").has_value() == false);
    REQUIRE(ParsePerfBaselines("").has_value() == false);

    PerfBaselineTable t;
    t.format_version = kPerfBaselineFormatVersion;
    auto json = SerializePerfBaselines(t);
    auto empty = ParsePerfBaselines(json);
    REQUIRE(empty.has_value());
    REQUIRE(empty->entries.empty());

    // Foreign format version invalidates (schema evolution).
    const std::string poisoned =
        SerializePerfBaselines(t).replace(
            json.find("\"format_version\": ") + 17, 1, "9");
    REQUIRE(ParsePerfBaselines(poisoned).has_value() == false);
}

TEST_CASE("Perf baselines: entry lookup by (model_type, gpu_key)",
          "[perf_baseline]")
{
    PerfBaselineTable t;
    t.format_version = kPerfBaselineFormatVersion;
    PerfBaselineEntry e;
    e.model_type = "gemma-4-26b-a4b-it";
    e.gpu_key = "rtx 5090";
    e.os = "linux";
    e.tokens_per_sec = 98.1;
    t.entries.push_back(e);

    using lsxcommon::perf_baseline::FindEntry;
    const auto *found = FindEntry(t, "gemma-4-26b-a4b-it", "rtx 5090", "linux");
    REQUIRE(found != nullptr);
    REQUIRE(found->tokens_per_sec == 98.1);
    // Same GPU, other OS: miss (per-OS baselines).
    REQUIRE(FindEntry(t, "gemma-4-26b-a4b-it", "rtx 5090", "windows") == nullptr);
    REQUIRE(FindEntry(t, "glm-5.3-flash", "rtx 5090", "linux") == nullptr);
    REQUIRE(FindEntry(t, "gemma-4-26b-a4b-it", "rtx 4090", "linux") == nullptr);
}


// ── Registry pairing (host-only) ────────────────────────────────────────────
//
// The protocol requires BOTH halves of the perf stage: a
// tests/perf/perf_baselines.json entry and a perf-labelled TEST_CASE that
// measures the model. A JSON entry without a target is an unguarded baseline
// (nothing ever fails when the model regresses), and a target without an entry
// measures against nothing. Nothing bound the two, so either half could be
// added or deleted silently.

namespace {

std::string ReadAll(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// model_type values that carry at least one entry in perf_baselines.json.
std::set<std::string> BaselineModels(const std::string &json)
{
    auto parsed = ParsePerfBaselines(json);
    REQUIRE(parsed.has_value());
    const PerfBaselineTable &table = *parsed;
    std::set<std::string> out;
    for (const auto &e : table.entries) out.insert(e.model_type);
    return out;
}

// model_type values named by a TEST_CASE in tests/test_perf_baseline.cpp.
std::set<std::string> PerfCaseModels(const std::string &test_src)
{
    std::set<std::string> out;
    static const std::regex case_re(R"(TEST_CASE\(\s*"([^"]+))");
    std::smatch m;
    auto begin = test_src.cbegin();
    while (std::regex_search(begin, test_src.cend(), m, case_re)) {
        for (const auto &mid :
             {std::string("glm-5.2"), std::string("glm-5.3-flash"),
              std::string("deepseek-v4-flash"),
              std::string("deepseek-v4.1-flash"), std::string("gemma-4"),
              std::string("kimi-k2.7"), std::string("qwen3.5-9b"),
              std::string("qwen3.8-27b"), std::string("qwen3.5-35b"),
              std::string("qwen3.6-35b"), std::string("qwen3.8-flash-next"),
              std::string("nemotron"), std::string("mimo"), std::string("gigachat"),
              std::string("granite"), std::string("phi-4"), std::string("ornith"),
              std::string("alice")}) {
            if (m[1].str().find(mid) != std::string::npos) out.insert(mid);
        }
        begin = m.suffix().first;
    }
    return out;
}

} // namespace

TEST_CASE("perf registry: every baseline entry has a measuring TEST_CASE",
          "[perf_baseline][guard]")
{
    const std::string root = LSX_SOURCE_DIR;
    const auto baselines =
        BaselineModels(ReadAll(root + "/tests/perf/perf_baselines.json"));
    const auto cases =
        PerfCaseModels(ReadAll(root + "/tests/test_perf_baseline.cpp"));
    std::vector<std::string> orphans;
    for (const auto &m : baselines) {
        // The case names a display stem (e.g. "glm-5.3-flash"), which is a
        // prefix-stable match: every model_type in the table appears verbatim
        // inside some case name.
        bool covered = false;
        for (const auto &c : cases) {
            const auto &[it, unused] = std::minmax(m, c);
            (void)unused;
            if (c.find(m) != std::string::npos || m.find(c) != std::string::npos)
                covered = true;
        }
        if (!covered) orphans.push_back("baseline entry with no perf TEST_CASE: " + m);
    }
    std::string joined;
    for (const auto &x : orphans) joined += x + "\n";
    INFO("perf registry drift:\n" << joined);
    REQUIRE(orphans.empty());
}
