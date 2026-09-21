#include "lsxcommon/perf_baseline.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <simdjson.h>

namespace lsxcommon {
namespace perf_baseline {

std::optional<PerfBaselineTable> ParsePerfBaselines(std::string_view json_view)
{
    // simdjson's DOM parser borrows the input buffer: copy into stable
    // storage so temporaries stay valid (perf-cache pattern).
    const std::string json(json_view);
    simdjson::dom::parser parser;
    simdjson::dom::element root;
    if (parser.parse(json).get(root) != simdjson::SUCCESS)
        return std::nullopt;
    int64_t ver = 0;
    if (root["format_version"].get(ver) != simdjson::SUCCESS ||
        (ver != kPerfBaselineFormatVersion && ver != 2 && ver != 1))
        return std::nullopt;
    simdjson::dom::array entries;
    if (root["entries"].get_array().get(entries) != simdjson::SUCCESS)
        return std::nullopt;
    PerfBaselineTable t;
    t.format_version = static_cast<int>(ver);
    for (simdjson::dom::element el : entries) {
        simdjson::dom::object o;
        if (el.get_object().get(o) != simdjson::SUCCESS) return std::nullopt;
        PerfBaselineEntry e;
        std::string_view sv;
        double d = 0.0;
        int64_t v = 0;
        if (o["model_type"].get_string().get(sv) != simdjson::SUCCESS)
            return std::nullopt;
        e.model_type = sv;
        if (o["gpu_key"].get_string().get(sv) != simdjson::SUCCESS)
            return std::nullopt;
        e.gpu_key = sv;
        if (o["os"].get_string().get(sv) != simdjson::SUCCESS)
            return std::nullopt;
        e.os = sv;
        // Numeric mode is optional (pre-mode entries default to "perf").
        if (o["mode"].get_string().get(sv) == simdjson::SUCCESS)
            e.mode = sv;
        if (o["tokens_per_sec"].get_double().get(d) != simdjson::SUCCESS)
            return std::nullopt;
        e.tokens_per_sec = d;
        if (o["max_gen"].get_int64().get(v) != simdjson::SUCCESS)
            return std::nullopt;
        e.max_gen = static_cast<int>(v);
        // Provenance fields are optional (older entries keep parsing).
        if (o["prompt"].get_string().get(sv) == simdjson::SUCCESS)
            e.prompt = sv;
        if (o["date"].get_string().get(sv) == simdjson::SUCCESS)
            e.date = sv;
        if (o["commit"].get_string().get(sv) == simdjson::SUCCESS)
            e.commit = sv;
        if (o["gen_tokens"].get_int64().get(v) == simdjson::SUCCESS)
            e.gen_tokens = static_cast<int>(v);
        // TTFT/power evidence is optional (v1/v2 entries keep parsing).
        if (o["ttft_ms"].get_double().get(d) == simdjson::SUCCESS)
            e.ttft_ms = d;
        if (o["avg_power_w"].get_double().get(d) == simdjson::SUCCESS)
            e.avg_power_w = d;
        if (o["peak_power_w"].get_double().get(d) == simdjson::SUCCESS)
            e.peak_power_w = d;
        if (o["power_samples"].get_int64().get(v) == simdjson::SUCCESS)
            e.power_samples = static_cast<int>(v);
        t.entries.push_back(std::move(e));
    }
    return t;
}

std::string SerializePerfBaselines(const PerfBaselineTable &t)
{
    std::string s;
    absl::StrAppend(&s, "{\n  \"format_version\": ", t.format_version,
                    ",\n  \"entries\": [");
    for (size_t i = 0; i < t.entries.size(); ++i) {
        const PerfBaselineEntry &e = t.entries[i];
        absl::StrAppend(&s, i == 0 ? "\n" : ",\n", "    {\n");
        absl::StrAppend(&s, "      \"model_type\": \"", e.model_type,
                        "\",\n");
        absl::StrAppend(&s, "      \"gpu_key\": \"", e.gpu_key, "\",\n");
        absl::StrAppend(&s, "      \"os\": \"", e.os, "\",\n");
        absl::StrAppend(&s, "      \"mode\": \"", e.mode, "\",\n");
        absl::StrAppendFormat(&s, "      \"tokens_per_sec\": %.2f,\n",
                              e.tokens_per_sec);
        absl::StrAppend(&s, "      \"max_gen\": ", e.max_gen, ",\n");
        absl::StrAppend(&s, "      \"prompt\": \"", e.prompt, "\",\n");
        absl::StrAppend(&s, "      \"date\": \"", e.date, "\",\n");
        absl::StrAppend(&s, "      \"commit\": \"", e.commit, "\",\n");
        absl::StrAppend(&s, "      \"gen_tokens\": ", e.gen_tokens, ",\n");
        absl::StrAppendFormat(&s, "      \"ttft_ms\": %.2f,\n", e.ttft_ms);
        absl::StrAppendFormat(&s, "      \"avg_power_w\": %.2f,\n",
                              e.avg_power_w);
        absl::StrAppendFormat(&s, "      \"peak_power_w\": %.2f,\n",
                              e.peak_power_w);
        absl::StrAppend(&s, "      \"power_samples\": ", e.power_samples, "\n");
        absl::StrAppend(&s, "    }");
    }
    absl::StrAppend(&s, t.entries.empty() ? "]\n" : "\n  ]\n");
    absl::StrAppend(&s, "}\n");
    return s;
}

const PerfBaselineEntry *FindEntry(const PerfBaselineTable &t,
                                   const std::string &model_type,
                                   const std::string &gpu_key,
                                   const std::string &os,
                                   const std::string &mode)
{
    for (const auto &e : t.entries) {
        if (e.model_type == model_type && e.gpu_key == gpu_key &&
            e.os == os && e.mode == mode)
            return &e;
    }
    return nullptr;
}

PerfVerdict JudgePerf(const std::optional<double> &baseline, double actual,
                      double tolerance_pct)
{
    if (!baseline.has_value()) return PerfVerdict::kUpdate;
    if (actual > *baseline) return PerfVerdict::kUpdate;
    if (actual >= *baseline * (1.0 - tolerance_pct / 100.0))
        return PerfVerdict::kPass;
    return PerfVerdict::kDegrade;
}

PerfVerdict JudgePerfLen(const std::optional<double> &baseline,
                         double actual, int actual_len, int baseline_len,
                         double tolerance_pct)
{
    // Legacy entry (no recorded fact length): comparable iff the run used
    // the same max_gen protocol.
    if (baseline_len <= 0)
        return JudgePerf(baseline, actual, tolerance_pct);
    if (!baseline.has_value() || actual_len != baseline_len)
        return PerfVerdict::kUpdate;
    return JudgePerf(baseline, actual, tolerance_pct);
}

} // namespace perf_baseline
} // namespace lsxcommon
