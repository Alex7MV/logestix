// Perf baseline reference: pure decision logic + JSON round-trip.
// Host-only, fast label — no CUDA, no model weights involved.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/perf_baseline.h"

#include <optional>
#include <string>

using lsxcommon::perf_baseline::JudgePerf;
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
    t.entries.push_back(e);
    e.model_type = "glm-5.3-flash";
    e.os = "windows";
    e.tokens_per_sec = 32.1;
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
    REQUIRE(parsed->entries[1].os == "windows");
    REQUIRE(parsed->entries[1].tokens_per_sec == 32.1);
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
