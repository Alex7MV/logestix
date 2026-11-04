// Perf baseline reference: pure decision logic + JSON round-trip.
// Host-only, fast label — no CUDA, no model weights involved.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/perf_baseline.h"
#include "test_kernel_budget.h"

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

TEST_CASE("expert-cache capacity is recorded with the measurement",
          "[perf_baseline]")
{
    using lsxcommon::perf_baseline::AttributeDegrade;
    using lsxcommon::perf_baseline::PerfCause;
    // A streamed-expert family's decode t/s is a measured function of the
    // resident expert-cache capacity: on qwen3.8-flash-next the same binary
    // decodes 41.40 / 43.02 / 44.15 t/s at 4962 / 5350 / 5750 slots. The
    // capacity itself comes from the free VRAM at load time
    // (lifecycle::ExpertCacheSlots), so an ambient 2.4 GiB of desktop VRAM is
    // worth ~4% of throughput with no code change at all. A run at a smaller
    // capacity is therefore not comparable to a reference recorded with more,
    // and must not be judged as a regression.
    //
    // The verdict is capacity-only when the reference recorded a capacity and
    // this run had less. Everything else keeps the probe rule.
    CHECK(AttributeDegrade(0.96, 0.995, 5350, 5960) == PerfCause::kCapacity);
    CHECK(AttributeDegrade(0.96, 0.995, 5350, 5350) == PerfCause::kCode);
    // More capacity than the reference is an improvement, not an excuse.
    CHECK(AttributeDegrade(0.96, 0.995, 6000, 5350) == PerfCause::kCode);
    // A family without a streamed cache (or a legacy entry) is unaffected.
    CHECK(AttributeDegrade(0.96, 0.995, 0, 0) == PerfCause::kCode);
    CHECK(AttributeDegrade(0.96, 0.995, 5350, 0) == PerfCause::kCode);
    // The probe still decides when the capacity explains nothing.
    CHECK(AttributeDegrade(0.96, 0.95, 5350, 5350) == PerfCause::kEnvironment);
    CHECK(AttributeDegrade(0.96, 0.95, 5350, 5960) == PerfCause::kCapacity);
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
    e.expert_cache_slots = 5350;
    t.entries.push_back(e);
    e.model_type = "glm-5.3-flash";
    e.os = "windows";
    e.tokens_per_sec = 32.1;
    e.gen_tokens = 0;   // legacy-style entry
    e.ttft_ms = 0.0;
    e.avg_power_w = 0.0;
    e.peak_power_w = 0.0;
    e.power_samples = 0;
    e.expert_cache_slots = 0;  // no streamed cache
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
    REQUIRE(parsed->entries[0].expert_cache_slots == 5350);
    REQUIRE(parsed->entries[1].os == "windows");
    REQUIRE(parsed->entries[1].tokens_per_sec == 32.1);
    REQUIRE(parsed->entries[1].gen_tokens == 0);
    // Absent power/TTFT fields degrade to the "unknown" sentinels.
    REQUIRE(parsed->entries[1].ttft_ms == 0.0);
    REQUIRE(parsed->entries[1].avg_power_w == 0.0);
    REQUIRE(parsed->entries[1].peak_power_w == 0.0);
    REQUIRE(parsed->entries[1].power_samples == 0);
    // An entry without the field (every pre-existing row, and every family
    // with no streamed expert cache) reads back as "no capacity recorded".
    REQUIRE(parsed->entries[1].expert_cache_slots == 0);
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
              std::string("deepseek-v4.1-flash"),
              std::string("deepseek-v4-pro-0813"), std::string("gemma-4"),
              std::string("kimi-k2.7"), std::string("qwen3.5-9b"),
              std::string("qwen3.8-27b"), std::string("qwen3.5-35b"),
              std::string("qwen3.6-35b"), std::string("qwen3.8-flash-next"),
              std::string("nemotron"), std::string("mimo"), std::string("gigachat"),
              std::string("granite"), std::string("phi-4"), std::string("ornith"),
              std::string("alice"), std::string("xing"),
              std::string("mistral"), std::string("minimax"),
              std::string("hy-memory"), std::string("hy-simulmt"),
              std::string("ling"),
              std::string("beam"), std::string("step-5-preview")}) {
            if (m[1].str().find(mid) != std::string::npos) out.insert(mid);
        }
        begin = m.suffix().first;
    }
    return out;
}

} // namespace

TEST_CASE("perf baseline file is exactly what the harness writes",
          "[perf_baseline][guard]")
{
    // The perf-labelled test rewrites this tracked file whenever it measures an
    // improvement, so its formatting IS the serializer's. A hand edit (or a
    // serializer change without a regeneration) makes the next measurement
    // rewrite unrelated rows, which buries the actual measurement in a diff --
    // and a row that was never measured by the instrument cannot honestly carry
    // ttft/power evidence. Pin the two together.
    const std::string path = std::string(LSX_SOURCE_DIR) +
                             "/tests/perf/perf_baselines.json";
    const std::string tracked = ReadAll(path);
    REQUIRE_FALSE(tracked.empty());
    const auto parsed = ParsePerfBaselines(tracked);
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->format_version == kPerfBaselineFormatVersion);
    const std::string rewritten = SerializePerfBaselines(*parsed);
    if (rewritten != tracked) {
        // Point at the first divergence: a whole-file mismatch is useless on its
        // own when the file is ~1000 lines.
        std::size_t i = 0;
        while (i < rewritten.size() && i < tracked.size() &&
               rewritten[i] == tracked[i])
            ++i;
        const std::size_t from = i > 120 ? i - 120 : 0;
        FAIL("tests/perf/perf_baselines.json is not in the harness's own format "
             "(first difference at byte "
             << i << ").\n--- tracked ---\n"
             << tracked.substr(from, 240) << "\n--- rewritten ---\n"
             << rewritten.substr(from, 240));
    }
}

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
        // The environment probe is a synthetic row (no model, no TEST_CASE);
        // it is maintained by test_perf_baseline's probe protocol instead.
        if (m == "__gpu_probe__") continue;
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

TEST_CASE("perf registry: every perf TEST_CASE has a registered ctest target",
          "[perf_baseline][guard]")
{
    // The case above is one-directional: a JSON entry with no TEST_CASE fails, but
    // a TEST_CASE with no add_test passed silently. That is how xing, mimo and
    // alice ended up with TEST_CASEs but no ctest wiring - `ctest -L perf` ran 16
    // of 19 families and nobody noticed, while xing's and alice's baselines in the
    // JSON kept looking maintained. Xing and alice had been recorded by invoking
    // the binary directly (Catch2 runs every TEST_CASE in the binary); mimo had
    // never been measured at all.
    //
    // A TEST_CASE without a target is worse than no test: it reads as coverage in
    // the source and as absence in ctest, and neither view contradicts the other.
    const std::string root = LSX_SOURCE_DIR;
    const std::string test_src =
        ReadAll(root + "/tests/test_perf_baseline.cpp");
    const std::string cmake_src =
        ReadAll(root + "/tests/CMakeLists.txt");

    // Every `add_test(NAME test_perf_baseline_<name> ... COMMAND test_perf_baseline
    // "<case name>")` in the CMake file. The COMMAND argument is the TEST_CASE
    // filter string, which is what actually ties the two together - matching on
    // the target NAME alone would only re-derive the model list.
    static const std::regex wired_re(R"(add_test\(NAME\s+test_perf_baseline_\S+\s*\n?[^)]*?COMMAND\s+test_perf_baseline\s+"([^"]+))");
    std::set<std::string> wired;
    std::smatch m;
    auto begin = cmake_src.cbegin();
    while (std::regex_search(begin, cmake_src.cend(), m, wired_re)) {
        wired.insert(m[1].str());
        begin = m.suffix().first;
    }

    static const std::regex case_re(R"(TEST_CASE\(\s*"([^"]+))");
    std::vector<std::string> unwired;
    begin = test_src.cbegin();
    while (std::regex_search(begin, test_src.cend(), m, case_re)) {
        const std::string name = m[1].str();
        if (name.rfind("perf baseline:", 0) == 0 && wired.count(name) == 0)
            unwired.push_back(name);
        begin = m.suffix().first;
    }

    std::string joined;
    for (const auto &x : unwired) joined += "TEST_CASE with no ctest target: " + x + "\n";
    INFO("perf target drift:\n" << joined);
    REQUIRE(unwired.empty());
}

TEST_CASE("kernel budgets: a hand-formatted multi-line entry is parsed",
          "[perf_baseline][guard]")
{
    // kernel_budgets.json is hand-editable and the long-note rows were written
    // across several lines. The loader used to be strictly one-object-per-line
    // and read only the fields on the FIRST line of such an entry, so
    // `observed_min/observed_max/samples/date/note` came back empty -- and any
    // LSX_KERNEL_BUDGET_WRITE=1 run then REWROTE the file from that truncated
    // view, destroying the note and the recorded range. Measured 2026-10-24:
    // recording the phi GEMV row that way wiped the notes of
    // `xing.kv_read.decode_gbps` and `mem.stream_read_ceiling_gbps`.
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "kernel_budgets_parse.json";
    {
        std::ofstream f(tmp);
        f << "{\n  \"budgets\": [\n"
          << "    {\"name\": \"one.line\", \"gpu_key\": \"rtx 5090\", "
             "\"kind\": \"min\", \"budget\": 10.5, "
             "\"observed_min\": 12.5, \"observed_max\": 13.5, "
             "\"samples\": 3, \"date\": \"2026-10-01\", "
             "\"note\": \"single\"},\n"
          << "    {\"name\": \"multi.line\", \"gpu_key\": \"rtx 5090\", "
             "\"kind\": \"max\", \"budget\": 47,\n"
          << "     \"observed_min\": 42.82, \"observed_max\": 58.1456,\n"
          << "     \"samples\": 5, \"date\": \"2026-10-04\",\n"
          << "     \"note\": \"spread across lines, with commas, and "
             "kAttnSplits=4\"}\n"
          << "  ]\n}\n";
    }
    const auto entries = kernel_budget::LoadAll(tmp.string());
    REQUIRE(entries.size() == 2);
    const auto &one = entries[0];
    CHECK(one.name == "one.line");
    CHECK(one.kind == kernel_budget::Kind::kMin);
    CHECK(one.budget == 10.5);
    CHECK(one.observed_min == 12.5);
    CHECK(one.observed_max == 13.5);
    CHECK(one.samples == 3);
    CHECK(one.date == "2026-10-01");
    CHECK(one.note == "single");
    const auto &multi = entries[1];
    CHECK(multi.name == "multi.line");
    CHECK(multi.kind == kernel_budget::Kind::kMax);
    CHECK(multi.budget == 47.0);
    CHECK(multi.observed_min == 42.82);
    CHECK(multi.observed_max == 58.1456);
    CHECK(multi.samples == 5);
    CHECK(multi.date == "2026-10-04");
    CHECK(multi.note ==
          "spread across lines, with commas, and kAttnSplits=4");

    // A malformed entry must fail loudly instead of being dropped into a
    // false "no record for this gpu" SKIP. Two shapes: an object that never
    // closes, and a closed object with no name/gpu_key.
    {
        std::ofstream f(tmp);
        f << "{\n  \"budgets\": [\n"
          << "    {\"name\": \"broken\", \"gpu_key\": \"rtx 5090\", "
             "\"budget\": 1,\n";   // truncated: the object never closes
    }
    REQUIRE_THROWS(kernel_budget::LoadAll(tmp.string()));
    {
        std::ofstream f(tmp);
        f << "{\n  \"budgets\": [\n"
          << "    {\"budget\": 1, \"gpu_key\": \"rtx 5090\"}\n"
          << "  ]\n}\n";
    }
    REQUIRE_THROWS(kernel_budget::LoadAll(tmp.string()));
    std::filesystem::remove(tmp);

    // The committed file itself: every entry must come through complete. The
    // truncating read this test exists for showed up as empty date/samples on
    // the entries whose notes span lines.
    const auto real = kernel_budget::LoadAll(
        std::string(LSX_SOURCE_DIR) + "/tests/perf/kernel_budgets.json");
    REQUIRE(real.size() >= 8);
    for (const auto &e : real) {
        INFO("entry " << e.name);
        CHECK_FALSE(e.name.empty());
        CHECK_FALSE(e.gpu_key.empty());
        CHECK(e.budget > 0.0);
        if (e.samples > 0) CHECK_FALSE(e.date.empty());
    }
}

TEST_CASE("perf harness: a Debug run must not write the tracked baseline file",
          "[perf_baseline][guard]")
{
    // The Debug gate used to live only in MeasureModel(), i.e. AFTER
    // RunPerfModel() had already run the cooldown and the 512 MiB DRAM probe.
    // The probe is recorded into tests/perf/perf_baselines.json by
    // UpdateProbeReference(), so a Debug tree (which ctest runs by default,
    // and which measures a debug-compiled probe nobody compares against the
    // Release reference) mutated a tracked file before SKIPping. Measured
    // 2026-10-24: two `ctest --preset linux-debug -R test_perf_baseline_phi_4`
    // runs rewrote the __gpu_probe__ row. The gate must come first.
    const std::string src =
        ReadAll(std::string(LSX_SOURCE_DIR) + "/tests/test_perf_baseline.cpp");
    const size_t fn = src.find("void RunPerfModel(");
    REQUIRE(fn != std::string::npos);
    const size_t body_end = src.find("\n}\n", fn);
    REQUIRE(body_end != std::string::npos);
    const std::string body = src.substr(fn, body_end - fn);
    const size_t gate = body.find("#ifndef NDEBUG");
    const size_t probe = body.find("MeasureMemoryBandwidthGbs");
    const size_t cooldown = body.find("CooldownBetweenTargets");
    INFO("RunPerfModel body:\n" << body);
    REQUIRE(gate != std::string::npos);
    REQUIRE(probe != std::string::npos);
    REQUIRE(cooldown != std::string::npos);
    // The skip must precede both writers (the cooldown is harmless but means
    // the GPU was already parked; the probe is what rewrites the JSON).
    REQUIRE(gate < probe);
    REQUIRE(gate < cooldown);
}

TEST_CASE("environment attribution needs the probe to explain the loss",
          "[perf-baseline][logic]")
{
    using lsxcommon::perf_baseline::AttributeDegrade;
    using lsxcommon::perf_baseline::PerfCause;
    // No probe -> no oracle -> a code regression.
    CHECK(AttributeDegrade(0.90, 0.0) == PerfCause::kCode);
    // Probe nominal -> the model's loss is code.
    CHECK(AttributeDegrade(0.90, 1.00) == PerfCause::kCode);
    // Probe down as much as the model -> environment.
    CHECK(AttributeDegrade(0.90, 0.89) == PerfCause::kEnvironment);
    CHECK(AttributeDegrade(0.96, 0.95) == PerfCause::kEnvironment);
    // Probe down but NOT enough to explain the model -> code (the oracle must
    // never mask a regression).
    CHECK(AttributeDegrade(0.85, 0.96) == PerfCause::kCode);
    CHECK(AttributeDegrade(0.80, 0.90) == PerfCause::kCode);
    // A probe inside noise (>= 97%) is not an environment signal on its own.
    CHECK(AttributeDegrade(0.94, 0.975) == PerfCause::kCode);
    // A large probe loss always explains a smaller model loss.
    CHECK(AttributeDegrade(0.97, 0.80) == PerfCause::kEnvironment);
}

