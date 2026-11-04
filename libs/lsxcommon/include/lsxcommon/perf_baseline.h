#pragma once

// ── Perf baseline reference (model-neutral, host-only) ──────────────────────
// Committed speed reference: (model_type, gpu_key, os) -> tokens_per_sec,
// measured by the perf-labeled test (tests/test_perf_baseline.cpp). Pure
// decision logic + JSON codec live here; file I/O stays in the test so the
// library is fully unit-testable. NEVER include from .cu TUs.

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lsxcommon {
namespace perf_baseline {

inline constexpr int kPerfBaselineFormatVersion = 3;

struct PerfBaselineEntry {
    std::string model_type;
    std::string gpu_key;
    std::string os;             // "windows" | "linux" (compile-time)
    std::string mode = "perf";  // numeric mode: "perf" (kAuto) | "r3" (kOff)
    double tokens_per_sec = 0.0;
    int max_gen = 0;            // measurement protocol (honest comparisons)
    std::string prompt;
    std::string date;           // ISO date of the measurement
    std::string commit;         // repo commit the measurement was taken at
    int gen_tokens = 0;         // ACTUAL generated length (0 = legacy entry);
                                // t/s is only comparable at equal length
    double ttft_ms = 0.0;       // stream start -> first generated token
    double avg_power_w = 0.0;   // mean board power over the run (0 = unknown)
    double peak_power_w = 0.0;  // peak board power over the run (0 = unknown)
    int power_samples = 0;      // 0 => NVML unavailable / not sampled
    int expert_cache_slots = 0; // resident routed-expert cache capacity the
                                // measurement decoded with. For a
                                // streamed-expert family t/s is a measured
                                // function of this capacity, so the number is
                                // part of the measurement protocol. Absent (0)
                                // means the family has no streamed cache, or the
                                // entry predates the field; the verdict then
                                // falls back to the environment probe.
};

struct PerfBaselineTable {
    int format_version = kPerfBaselineFormatVersion;
    std::vector<PerfBaselineEntry> entries;
};

// simdjson-backed parser; rejects foreign format versions and garbage.
std::optional<PerfBaselineTable> ParsePerfBaselines(std::string_view json);

// Hand-rolled stable-order writer (mirrors perf-cache serializer).
std::string SerializePerfBaselines(const PerfBaselineTable &t);

// Entry lookup by the full (model_type, gpu_key, os, mode) key; nullptr if
// absent. mode defaults to "perf" so older single-mode callers keep working.
const PerfBaselineEntry *FindEntry(const PerfBaselineTable &t,
                                   const std::string &model_type,
                                   const std::string &gpu_key,
                                   const std::string &os,
                                   const std::string &mode = "perf");

enum class PerfVerdict {
    kUpdate,   // absent entry or actual > baseline: record it
    kPass,     // within run-noise tolerance below the baseline
    kDegrade   // below tolerance: FAIL, root-cause the regression
};

// Pure verdict: absent -> update; actual > baseline -> update (strictly
// greater: an equal value is a pass, not an update); within tolerance_pct
// below baseline -> pass; else degrade.
PerfVerdict JudgePerf(const std::optional<double> &baseline, double actual,
                      double tolerance_pct = 3.0);

// Backfills the recorded evidence (TTFT/power) of an existing entry from a
// fresh measurement when the entry predates those fields. Never touches
// tokens_per_sec or the measurement protocol. Returns true when the entry
// changed. A fresh run without NVML leaves the power fields untouched; TTFT
// is still filled when absent.
bool BackfillEvidence(PerfBaselineEntry &entry, double ttft_ms,
                      double avg_power_w, double peak_power_w, int samples);

// Length-aware verdict: t/s depends on the actual generated length (EOS
// timing shifts the KV-attention cost curve), so a run is only comparable
// to a baseline recorded at the same fact length. actual_len is the run's
// generated-token count; baseline_len is the entry's recorded fact length
// (0 = legacy entry, falls back to comparing the max_gen protocol).
// Length mismatch -> kUpdate (re-seed) regardless of the t/s direction;
// match -> JudgePerf semantics.
PerfVerdict JudgePerfLen(const std::optional<double> &baseline,
                         double actual, int actual_len, int baseline_len,
                         double tolerance_pct = 3.0);

// Why a measurement read below its reference. The model does not certify its
// own environment, so two model-independent facts decide. (1) Capacity: for a
// streamed-expert family the decode rate is a measured function of the resident
// expert-cache slot count, and that count is derived from the free VRAM at load
// time -- so ambient desktop VRAM silently changes t/s by several percent with
// no code change. A run made with less capacity than the reference is not
// comparable and is attributed to capacity, never to code. (2) A fixed
// device-to-device copy probe tracks the card's thermal/power state; a degraded
// model reading is attributed to the environment only when the probe is ALSO
// down (below kProbeDegradeRatio) by at least the model's own loss. Anything
// else stays a code regression, so neither oracle can mask one.
enum class PerfCause { kCode, kEnvironment, kCapacity };

inline constexpr double kProbeDegradeRatio = 0.97;

// actual_slots / baseline_slots are the expert-cache capacities of this run and
// of the reference entry (0 = not applicable/not recorded). Capacity is only
// consulted when both sides recorded one and the reference had more.
PerfCause AttributeDegrade(double model_ratio, double probe_ratio,
                           int actual_slots = 0, int baseline_slots = 0);

} // namespace perf_baseline
} // namespace lsxcommon
