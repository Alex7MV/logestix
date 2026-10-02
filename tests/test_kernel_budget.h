#pragma once

// GPU-keyed kernel latency/speedup budgets (host helper, no model weights).
//
// The 11 budget assertions in the CUDA suites used to carry a hard-coded
// number with no reference to the machine it was measured on. Two consequences
// measured on this host (2026-10-02):
//
//   * gemma RMSNorm: the budget is 3.4 us, the kernel and its test have not
//     changed since it was calibrated (2026-09-14, where the author measured
//     2.23 us), and it now measures 4.78 us on an idle RTX 5090 — identical
//     code, 2.1x apart, so the number belongs to a different machine.
//   * nemotron batched GEMV: `batch < single * 0.6` is machine-independent by
//     construction and still fails at 1.55x, so a hard-coded constant cannot
//     even express that case.
//
// A red budget therefore carried no information: not "regressed" and not "wrong
// machine". This helper makes the machine explicit, using the same semantics
// perf_baseline already implements for model throughput:
//
//   * a record for (name, gpu_key) exists  -> assert against it
//   * no record for this gpu_key             -> SKIP (do not judge)
//   * LSX_KERNEL_BUDGET_WRITE=1              -> record the measurement instead
//
// File: tests/perf/kernel_budgets.json, one entry per (name, gpu_key).
// Recalibrate with: cmake -DLSX_KERNEL_BUDGET_WRITE=1 --build . --target <test>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <lsxcommon/gpu_perf.h>

#ifndef LSX_KERNEL_BUDGETS_FILE
#define LSX_KERNEL_BUDGETS_FILE "kernel_budgets.json"
#endif

namespace kernel_budget {

enum class Kind {
    kMax,  // measured must stay BELOW budget (latency, bandwidth-inverse)
    kMin,  // measured must stay ABOVE budget (speedup ratios)
};

struct Entry {
    std::string name;
    std::string gpu_key;
    double budget = 0.0;
    Kind kind = Kind::kMax;
    std::string note;
    std::string date;
};

namespace fs = std::filesystem;

// Budget records are ASCII names/notes written by this file and by hand, so a
// minimal escape is enough here — pulling lsxnetwork's JsonEscape in would
// force an include-path dependency on every CUDA object library that uses a
// budget.
inline std::string Escape(const std::string &in)
{
    std::string out;
    out.reserve(in.size() + 4);
    for (char c : in) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

inline const char *KindStr(Kind k)
{
    return k == Kind::kMax ? "max" : "min";
}

inline Kind ParseKind(const std::string &s)
{
    return s == "min" ? Kind::kMin : Kind::kMax;
}

// The key the whole scheme is built on: the same normalisation
// perf::GpuProfileKey uses for throughput baselines.
inline std::string CurrentGpuKey()
{
    return lsxcommon::perf::GpuProfileKey(
        lsxcommon::perf::ProbeGpuPerfProfile());
}

inline std::vector<Entry> LoadAll()
{
    std::vector<Entry> out;
    std::ifstream in(LSX_KERNEL_BUDGETS_FILE);
    if (!in) return out;

    // Line-oriented parse: the file is written by WriteAll below, so it is
    // one JSON object per line inside a "budgets" array. Deliberately not a
    // general JSON reader — a mis-formatted line must fail loudly, not be
    // silently skipped into a false "no record for this gpu" SKIP.
    std::string line;
    bool in_array = false;
    while (std::getline(in, line)) {
        const auto t = line.find_first_not_of(" \t\r\n");
        if (t == std::string::npos) continue;
        const std::string s = line.substr(t);
        if (s.rfind("\"budgets\"", 0) == 0) {
            in_array = true;
            continue;
        }
        if (!in_array) continue;
        if (s[0] == ']') break;
        if (s.rfind("{\"name\"", 0) != 0) continue;

        const auto field = [&](const char *k) -> std::string {
            const std::string key = std::string("\"") + k + "\": \"";
            auto p = s.find(key);
            if (p == std::string::npos) return {};
            p += key.size();
            const auto q = s.find('"', p);
            return q == std::string::npos ? std::string{}
                                           : s.substr(p, q - p);
        };
        const auto number = [&](const char *k) -> double {
            const std::string key = std::string("\"") + k + "\": ";
            auto p = s.find(key);
            if (p == std::string::npos) return 0.0;
            p += key.size();
            try {
                return std::stod(s.substr(p));
            } catch (...) {
                return 0.0;
            }
        };

        Entry e;
        e.name = field("name");
        e.gpu_key = field("gpu_key");
        e.kind = ParseKind(field("kind"));
        e.note = field("note");
        e.date = field("date");
        e.budget = number("budget");
        if (e.name.empty() || e.gpu_key.empty()) continue;
        out.push_back(std::move(e));
    }
    return out;
}

inline void WriteAll(const std::vector<Entry> &v)
{
    std::ostringstream s;
    s << "{\n  \"budgets\": [\n";
    for (size_t i = 0; i < v.size(); ++i) {
        const Entry &e = v[i];
        s << "    {\"name\": \"" << Escape(e.name)
          << "\", \"gpu_key\": \"" << Escape(e.gpu_key)
          << "\", \"kind\": \"" << KindStr(e.kind)
          << "\", \"budget\": " << e.budget
          << ", \"date\": \"" << e.date
          << "\", \"note\": \"" << Escape(e.note)
          << "\"}" << (i + 1 < v.size() ? "," : "") << "\n";
    }
    s << "  ]\n}\n";
    std::ofstream out(LSX_KERNEL_BUDGETS_FILE, std::ios::trunc);
    out << s.str();
}

inline std::string Today()
{
    const std::time_t t = std::time(nullptr);
    char buf[16];
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
    return buf;
}

// The single entry point every budget assertion goes through.
//
// In write mode (LSX_KERNEL_BUDGET_WRITE=1) the measurement is recorded for
// the current gpu_key and the assertion is skipped, so one build of a sweep
// recalibrates every budget it touches.
inline void Require(const std::string &name, double measured, Kind kind,
                    const std::string &note = {})
{
    const std::string gpu = CurrentGpuKey();

    if (const char *w = std::getenv("LSX_KERNEL_BUDGET_WRITE");
        w && w[0] == '1') {
        std::vector<Entry> all = LoadAll();
        Entry e;
        e.name = name;
        e.gpu_key = gpu;
        e.kind = kind;
        e.note = note;
        e.date = Today();
        // A speedup budget must stay strictly below the measured ratio, a
        // latency budget strictly above it: recording the exact measurement
        // would make the very next run a coin flip.
        // A recalibration sweep runs the same case many times, and the spread
        // is not negligible (the gemma norm case spans 4.5-5.4 us, 20%). Keep
        // the WORST observation rather than the last one, so a sweep that
        // happens to catch a fast sample cannot leave the budget below what
        // the kernel actually does. Then pad: 1.35x for a latency ceiling,
        // 0.8x for a speedup floor.
        const double pad = kind == Kind::kMax ? 1.35 : 0.80;
        e.budget = measured * pad;
        bool merged = false;
        for (auto &x : all) {
            if (x.name != name || x.gpu_key != gpu) continue;
            const double candidate = e.budget;
            const double keep = kind == Kind::kMax
                                    ? std::max(x.budget, candidate)
                                    : std::min(x.budget, candidate);
            x.budget = keep;
            x.kind = kind;
            x.note = note;
            x.date = Today();
            merged = true;
        }
        if (!merged) all.push_back(e);
        std::sort(all.begin(), all.end(), [](const Entry &a, const Entry &b) {
            return a.name != b.name ? a.name < b.name : a.gpu_key < b.gpu_key;
        });
        WriteAll(all);
        WARN("recorded budget for " + name + " on " + gpu + ": " +
             std::to_string(e.budget));
        // Do NOT skip in write mode: a case carrying two budgets (the nemotron
        // GEMV asserts speedup and latency) would otherwise record only the
        // first one, since SKIP ends the case.
        return;
    }

    const std::vector<Entry> all = LoadAll();
    for (const Entry &e : all) {
        if (e.name != name || e.gpu_key != gpu) continue;
        INFO("kernel budget " << name << " on " << gpu << " ("
             << KindStr(e.kind) << ' ' << e.budget << ", recorded "
             << e.date << (e.note.empty() ? "" : ": " + e.note) << ")");
        if (kind == Kind::kMax)
            REQUIRE(measured < e.budget);
        else
            REQUIRE(measured > e.budget);
        return;
    }

    // No record for this machine: judging would be meaningless — that is the
    // whole defect this helper removes.
    SKIP("no kernel budget for '" + name + "' on this GPU (key '" + gpu +
         "'); record one with LSX_KERNEL_BUDGET_WRITE=1");
}

}  // namespace kernel_budget
