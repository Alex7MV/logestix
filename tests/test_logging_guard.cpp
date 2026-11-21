// logging guard (host-only static scan; zero deps beyond Catch2).
//
// Rule (AGENTS.md -> coding-conventions): all host logging goes through spdlog
// / the lsxcommon::log facade. No std::cout / std::cerr / std::clog and no
// stdio logging (printf / fprintf / vprintf / vfprintf / puts / putchar /
// perror) in production code: the async logger owns levels, rotation and
// thread-safe interleaving, and a stray fprintf bypasses all three.
//
// Scanned: libs/ and apps/ (production code). Excluded: tests/ subtrees (test
// diagnostics are out of scope) and third_party.
//
// Declared exceptions live in Allowlist() and are enforced in BOTH directions:
//   * a logging call that is not declared fails (with file:line), and
//   * a declared line that no longer occurs fails (stale entry),
// so the list cannot outlive the problem. The key is the exact trimmed source
// line: moving a line is free, adding/editing a call requires touching the
// allowlist in the same commit. Categories: device-printf (kernels cannot call
// the host logger), env-gated-diagnostic (perf/debug probes behind LSX_*/KIMI_*
// env vars), dump-file (file dumps, not logging), teardown (logger already
// destroyed), header-only (must not link lsxcommon).
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

struct Hit
{
    std::string file;
    std::size_t line = 0;
    std::string text;
};

struct AllowedLine
{
    const char *text;
    int count;
};

struct AllowedFile
{
    const char *path;
    const char *category;
    const char *reason;
    std::vector<AllowedLine> lines;
};

bool IsIdentChar(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

bool IsLoggingFunction(const std::string &id)
{
    static const std::vector<std::string> v = {
        "printf", "fprintf", "vprintf", "vfprintf",
        "puts", "putchar", "perror",
    };
    return std::find(v.begin(), v.end(), id) != v.end();
}

bool IsLoggingStream(const std::string &id)
{
    static const std::vector<std::string> v = {
        "cout", "cerr", "clog", "wcout", "wcerr", "wclog",
    };
    return std::find(v.begin(), v.end(), id) != v.end();
}

// Blank out comments and string/char literals while preserving every offset
// and newline, so line numbers and the original (unstripped) line text stay
// valid. Raw strings R"delim(...)delim" are handled; a quote preceded by an
// identifier character is a C++14 digit separator, not a literal.
void StripForScan(const std::string &src, std::string &out)
{
    out = src;
    const std::size_t n = src.size();
    const auto blank = [&out](std::size_t from, std::size_t to) {
        for (std::size_t k = from; k < to && k < out.size(); ++k)
            if (out[k] != '\n') out[k] = ' ';
    };
    std::size_t i = 0;
    while (i < n)
    {
        const char c = src[i];
        if (c == '/' && i + 1 < n && src[i + 1] == '/')
        {
            std::size_t j = src.find('\n', i);
            if (j == std::string::npos) j = n;
            blank(i, j);
            i = j;
        }
        else if (c == '/' && i + 1 < n && src[i + 1] == '*')
        {
            std::size_t j = src.find("*/", i + 2);
            j = (j == std::string::npos) ? n : std::min(n, j + 2);
            blank(i, j);
            i = j;
        }
        else if (c == 'R' && i + 1 < n && src[i + 1] == '"' &&
                 (i == 0 || !IsIdentChar(src[i - 1])))
        {
            std::size_t k = i + 2;
            std::string delim;
            while (k < n && src[k] != '(') delim += src[k++];
            if (k >= n)
            {
                ++i;
                continue;
            }
            const std::string close = ")" + delim + "\"";
            std::size_t j = src.find(close, k + 1);
            j = (j == std::string::npos) ? n : std::min(n, j + close.size());
            blank(i, j);
            i = j;
        }
        else if (c == '"' || (c == '\'' && (i == 0 || !IsIdentChar(src[i - 1]))))
        {
            const char q = c;
            std::size_t k = i + 1;
            while (k < n)
            {
                if (src[k] == '\\' && k + 1 < n)
                {
                    k += 2;
                    continue;
                }
                if (src[k] == q)
                {
                    ++k;
                    break;
                }
                ++k;
            }
            blank(i, k);
            i = k;
        }
        else
        {
            ++i;
        }
    }
}

std::vector<std::string> SourceLines(const std::string &src)
{
    std::vector<std::string> out;
    std::size_t begin = 0;
    while (begin <= src.size())
    {
        const std::size_t end = src.find('\n', begin);
        out.push_back(src.substr(begin, end == std::string::npos
                                            ? std::string::npos
                                            : end - begin));
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return out;
}

std::string Trim(const std::string &s)
{
    std::size_t begin = 0;
    std::size_t end = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin])))
        ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1])))
        --end;
    return s.substr(begin, end - begin);
}

// One hit per source line: the allowlist key is the line, not the call, so a
// line with two logging calls still counts once (matching how a human edits
// the list).
std::vector<Hit> ScanText(const std::string &src, const std::string &name)
{
    std::string clean;
    StripForScan(src, clean);
    const std::vector<std::string> lines = SourceLines(src);

    std::vector<Hit> hits;
    std::size_t line = 1;
    std::size_t i = 0;
    while (i < clean.size())
    {
        const char c = clean[i];
        if (c == '\n')
        {
            ++line;
            ++i;
            continue;
        }
        if (!(std::isalpha(static_cast<unsigned char>(c)) || c == '_'))
        {
            ++i;
            continue;
        }
        std::size_t j = i;
        while (j < clean.size() && IsIdentChar(clean[j])) ++j;
        const std::string id = clean.substr(i, j - i);

        bool hit = false;
        if (IsLoggingFunction(id))
        {
            std::size_t k = j;
            while (k < clean.size() &&
                   std::isspace(static_cast<unsigned char>(clean[k])))
                ++k;
            hit = k < clean.size() && clean[k] == '(';
        }
        else if (id == "std")
        {
            std::size_t k = j;
            while (k < clean.size() &&
                   std::isspace(static_cast<unsigned char>(clean[k])))
                ++k;
            if (k + 1 < clean.size() && clean[k] == ':' && clean[k + 1] == ':')
            {
                k += 2;
                while (k < clean.size() &&
                       std::isspace(static_cast<unsigned char>(clean[k])))
                    ++k;
                const std::size_t begin = k;
                while (k < clean.size() && IsIdentChar(clean[k])) ++k;
                hit = IsLoggingStream(clean.substr(begin, k - begin));
            }
        }

        if (hit && (hits.empty() || hits.back().line != line))
            hits.push_back({name, line,
                            line <= lines.size() ? Trim(lines[line - 1])
                                                 : std::string()});
        i = j;
    }
    return hits;
}

bool IsScannedFile(const fs::path &p)
{
    const std::string ext = p.extension().string();
    return ext == ".cpp" || ext == ".cc" || ext == ".cxx" || ext == ".h" ||
           ext == ".hpp" || ext == ".cuh" || ext == ".cu";
}

bool IsProductionPath(const fs::path &rel)
{
    for (const auto &part : rel)
    {
        const std::string s = part.string();
        if (s == "tests" || s == "third_party" || s == "node_modules")
            return false;
    }
    return true;
}

std::vector<Hit> ScanTree(const fs::path &root)
{
    std::vector<Hit> hits;
    for (const char *dir : {"libs", "apps"})
    {
        std::error_code ec;
        const fs::path base = root / dir;
        if (!fs::exists(base, ec)) continue;
        for (auto &entry : fs::recursive_directory_iterator(base, ec))
        {
            if (!entry.is_regular_file()) continue;
            const fs::path rel = fs::relative(entry.path(), root);
            if (!IsScannedFile(entry.path()) || !IsProductionPath(rel))
                continue;
            std::ifstream in(entry.path(), std::ios::binary);
            if (!in) continue;
            std::ostringstream ss;
            ss << in.rdbuf();
            const auto file_hits = ScanText(ss.str(), rel.generic_string());
            hits.insert(hits.end(), file_hits.begin(), file_hits.end());
        }
    }
    return hits;
}

// Two-way coverage: every hit must be declared exactly as often as the
// allowlist says, and every declared line must still occur.
std::vector<std::string> CoverageViolations(
    const std::vector<Hit> &hits, const std::vector<AllowedFile> &allow)
{
    std::map<std::string, std::map<std::string, int>> expected;
    for (const auto &f : allow)
        for (const auto &l : f.lines)
            expected[f.path][l.text] += l.count;

    std::vector<std::string> violations;
    std::set<std::string> hit_files;
    for (const auto &h : hits)
    {
        hit_files.insert(h.file);
        auto fit = expected.find(h.file);
        if (fit == expected.end())
        {
            violations.push_back(h.file + ":" + std::to_string(h.line) + ": " +
                                 h.text + "  [file is not in the allowlist]");
            continue;
        }
        auto lit = fit->second.find(h.text);
        if (lit == fit->second.end() || lit->second <= 0)
        {
            violations.push_back(h.file + ":" + std::to_string(h.line) + ": " +
                                 h.text + "  [line is not in the allowlist]");
            continue;
        }
        --lit->second;
    }

    for (const auto &f : allow)
    {
        if (!hit_files.count(f.path))
            violations.push_back(std::string(f.path) +
                                 "  [allowlisted file has no logging hit; "
                                 "stale entry]");
        for (const auto &[text, left] : expected[f.path])
            if (left > 0)
                violations.push_back(
                    std::string(f.path) + ": expected " +
                    std::to_string(left) + " more occurrence(s) of \"" + text +
                    "\"  [stale allowlist entry]");
    }
    return violations;
}

// Declared exceptions. Adding an entry is a deliberate act: prefer converting
// the site to log::info/warn/error; only a real constraint (device code, an
// env-gated perf probe, logger teardown, a header that must not link
// lsxcommon) justifies an entry, and the category must be one of the five
// documented in the file header.
const std::vector<AllowedFile> &Allowlist()
{
    static const std::vector<AllowedFile> v = {
        {"libs/lsxcommon/src/glm_moe_flash_load.cpp", "env-gated-diagnostic",
         "LSX_FLASH_LOAD_DEBUG loader trace",
         {
             {"std::fprintf(stderr, \"[load] l=%d kda=%d q=%p o=%p A=%p\\n\", l,", 1},
         }},
        {"libs/lsxcommon/src/flint_moe_inference_pipeline.cu", "env-gated-diagnostic",
         "LSX_FLINT_DEBUG prompt/top-5 trace",
         {
             {"fprintf(stderr, \"[flint-debug] prompt_ids=%s\\n\", pids.c_str());", 1},
             {"fprintf(stderr, \"[flint-debug] step=%d top5=%s\\n\", generated,", 1},
         }},
        {"libs/lsxcommon/src/lsx_moe_inference_pipeline.cu", "env-gated-diagnostic",
         "LSX_AUDIT_STAGES stage attribution (shared qwen/ornith driver, dedup Ф6)",
         {
             {"fprintf(stderr,", 1},
             {"fprintf(stderr, \"[AUDIT] top%d %s L=%d %.3f ms (%.1f%%)\\n\",", 1},
         }},
        {"libs/lsxcommon/src/lsx_moe_inference.cu", "env-gated-diagnostic",
         "stage replay / FFN probe diagnostics",
         {
             {"std::fprintf(stderr,", 1},
             {"std::fprintf(stderr, \"[stage] %s: %s\\n\", name,               \\", 1},
             {"std::fprintf(stderr, \"[stage] shared_swiglu: %s\\n\",", 1},
             {"std::fprintf(stderr, \"[stage] shared_gate: %s\\n\",", 1},
             {"std::fprintf(stderr, \"[stage] combine: %s\\n\",", 1},
         }},
        {"libs/lsxcommon/src/qwen_moe_exp_inference.cu", "env-gated-diagnostic",
         "stage diagnostic",
         {
             {"std::fprintf(stderr,", 1},
         }},
        {"libs/lsxcommon/src/minimax_moe_inference_pipeline.cu", "env-gated-diagnostic",
         "LSX_MINIMAX_DEBUG trace and LSX_MINIMAX_DUMP_* file dumps",
         {
             {"std::fprintf(df, \"embed\");", 1},
             {"std::fprintf(df, \" %.6f\",", 2},
             {"std::fprintf(df, \"\\n\");", 2},
             {"std::fprintf(df, \"layer %d\", l);", 1},
             {"if (fp) std::fprintf(fp, \"%.6f\\n\", __half2float(hpre[static_cast<size_t>(i)]));", 1},
             {"if (fq) std::fprintf(fq, \"%.6f\\n\", __half2float(hpost[static_cast<size_t>(i)]));", 1},
             {"fprintf(stderr, \"[minimax-debug] prompt_ids=%s\\n\", pids.c_str());", 1},
             {"fprintf(stderr, \"[minimax-debug] top5=%s\\n\", t5.c_str());", 1},
         }},
        {"libs/lsxcommon/src/core.cpp", "teardown",
         "logger already torn down; spdlog must not be re-entered from destructors",
         {
             {"std::fprintf(stderr, \"[core] cudaFreeIgnoreShutdown: %s\\n\",", 1},
         }},
        {"libs/lsxcommon/src/xing_moe_inference.cu", "env-gated-diagnostic",
         "LSX_TEST_XING_LAYER_DEBUG stage probe",
         {
             {"std::fprintf(stderr, \"[xing-stage] %-22s L=%d tok=%d rms=%.6g nonfinite=%ld\\n\",", 1},
             {"std::fprintf(stderr, \"[xing-embed] tok=%d rms=%.6g max=%.6g\\n\", token_id,", 1},
             {"std::fprintf(stderr, \"[xing-kv] tok=%d pos=%d layer=%d v=%p cap=%d\\n\",", 1},
             {"std::fprintf(stderr,", 1},
             {"std::fprintf(stderr, \"[xing-layer] %d %d pos=%d hash=%08x\\n\", l,", 1},
             {"std::fprintf(stderr, \"[xing-layer] tok=%d %d streams\", token_id, l);", 1},
             {"std::fprintf(stderr, \" %.6g\", std::sqrt(per[d] / H));", 1},
             {"std::fprintf(stderr, \"\\n\");", 1},
         }},
        {"libs/lsxcommon/src/step_moe_inference_attn.cu", "env-gated-diagnostic",
         "LSX_STEP_DUMP_FINE attention dump",
         {
             {"std::fprintf(stderr, \"[fine] %s l=%d pos=%d hash=%016llx v=%g\", tag, layer,", 1},
             {"std::fprintf(stderr, \" %.6g\", static_cast<double>(__half2float(h[static_cast<size_t>(i)])));", 1},
             {"std::fprintf(stderr, \"\\n\");", 1},
         }},
        {"libs/lsxcommon/src/lsx_inference_nvfp4.cu", "device-printf",
         "device-side out-of-range guard; kernels cannot call the host logger",
         {
             {"printf(\"[lsx-nvfp4] expert id %d out of range (experts=%d)\\n\", expert,", 1},
         }},
        {"libs/lsxcommon/src/glm_moe_flash_inference.cu", "device-printf",
         "LSX_TRACE_FLASH device probe",
         {
             {"printf(\"[probe] kind=%d v0=%.6f v1=%.6f nan=%lld/%lld inf=%lld\\n\",", 1},
             {"printf(\"[step-kernel] acc=%.6f col0=%.6f state0=%.6f gd0=%.6f q0=%.6f\\n\",", 1},
             {"printf(\"[gated] v0=%.8f inv=%.4f gate0=%.6f w0=%.6f out0=%.8f\\n\",", 1},
             {"printf(\"[norm] inv=%.6f out0=%.6f x0=%.6f\\n\", inv,", 1},
             {"printf(\"[mla2] h=%d mx=%.4f psum=%.4f score0=%.4f q0=%.4f k0=%.4f\\n\",", 1},
             {"printf(\"[emb] id=%d\\n\", flash_dbg_id);", 1},
         }},
        {"libs/lsxcommon/src/deepseek_moe_pro_inference_moe.cu", "env-gated-diagnostic",
         "LSX_DS_FFN_PROBE / LSX_DS_VERIFY_COPY",
         {
             {"fprintf(stderr, \"P2 y2: %f %f %f\\n\", __half2float(hq[0]),", 1},
             {"fprintf(stderr, \"VREAD flat=%d slot=%d pk=%d sc=%d\\n\",", 1},
             {"fprintf(stderr, \"VCOPY flat=%d slot=%d match=%d\\n\",", 1},
         }},
        {"libs/lsxcommon/src/kimi_moe_inference.cu", "env-gated-diagnostic",
         "KIMI_DIAG_LAYER / KIMI_TRACE_FILE diagnostics",
         {
             {"fprintf(stderr, \"[DIAG] l=%d scores_sum=%.6f\\n\", l, sum);", 1},
             {"fprintf(stderr,", 1},
             {"fprintf(stderr, \"[DIAG-IDS] l=%d ids=\", l);", 1},
             {"fprintf(stderr, \"%d%s\", h_ids[k],", 1},
             {"fprintf(stderr, \" w=\");", 1},
             {"fprintf(stderr, \"%.5f%s\", h_w[k],", 1},
             {"fprintf(stderr, \"\\n\");", 1},
             {"fprintf(stderr, \"[DIAG] dumped layer %d (%d rows)\\n\", l,", 1},
         }},
        {"libs/lsxcommon/src/xing_moe_model.cpp", "env-gated-diagnostic",
         "LSX_TEST_XING_LAYER_DEBUG logits hash",
         {
             {"std::fprintf(stderr, \"[xing-gen] pos=%d tok=%d hash=%08x\\n\", pos,", 1},
         }},
        {"libs/lsxcommon/src/ling_moe_inference.cu", "env-gated-diagnostic",
         "LSX_LING_DEBUG / LSX_LING_DUMP",
         {
             {"std::fprintf(stderr,", 6},
             {"std::fprintf(stderr, \"[ling-prompt]\");", 1},
             {"for (int32_t id : prefill) std::fprintf(stderr, \" %d\", id);", 1},
             {"std::fprintf(stderr, \"\\n\");", 2},
             {"std::fprintf(stderr, \"[ling-debug] step fail: '%s' token=%d\\n\",", 1},
             {"std::fprintf(stderr, \"[ling-top5 step=%d]\", step);", 1},
             {"std::fprintf(stderr, \" %d:%.4f\", top[static_cast<size_t>(k)],", 1},
         }},
        {"libs/lsxcommon/src/step_moe_inference_pipeline.cu", "env-gated-diagnostic",
         "LSX_STEP_DEBUG and layer dump",
         {
             {"std::fprintf(stderr,", 1},
             {"std::fprintf(stderr, \"[step-debug] prompt_ids=%s\\n\", pids.c_str());", 1},
             {"std::fprintf(stderr, \"[step-debug] top5=%s\\n\", t5.c_str());", 1},
         }},
        {"libs/lsxcommon/src/deepseek_moe_inference_moe.cu", "env-gated-diagnostic",
         "LSX_DS_FFN_PROBE / LSX_DS_VERIFY_COPY",
         {
             {"fprintf(stderr, \"P2 y2: %f %f %f\\n\", __half2float(hq[0]),", 1},
             {"fprintf(stderr, \"VREAD flat=%d slot=%d pk=%d sc=%d\\n\",", 1},
             {"fprintf(stderr, \"VCOPY flat=%d slot=%d match=%d\\n\",", 1},
         }},
        {"libs/lsxcommon/src/hy_moe_inference_pipeline.cu", "env-gated-diagnostic",
         "LSX_HY_DEBUG prompt/top-5 trace",
         {
             {"fprintf(stderr, \"[hy-debug] prompt_ids=%s\\n\", pids.c_str());", 1},
             {"fprintf(stderr, \"[hy-debug] top5=%s\\n\", t5.c_str());", 1},
         }},
        {"libs/lsxcommon/include/lsxcommon/lsx_inference_nvfp4.h", "header-only",
         "E1 sequential-dispatch invariant; header must not link lsxcommon",
         {
             {"std::fprintf(stderr,", 1},
         }},
    };
    return v;
}

} // namespace

TEST_CASE("logging guard: scanner flags stdout/stdio logging",
          "[logging][guard]")
{
    SECTION("std::cout / cerr / clog family")
    {
        CHECK(ScanText("std::cout << x;\n", "s").size() == 1);
        CHECK(ScanText("std::cerr << x;\n", "s").size() == 1);
        CHECK(ScanText("std::clog << x;\n", "s").size() == 1);
        CHECK(ScanText("std :: cout << x;\n", "s").size() == 1);
        CHECK(ScanText("std::wcout << x;\n", "s").size() == 1);
    }
    SECTION("stdio logging functions")
    {
        CHECK(ScanText("printf(\"x\\n\");\n", "s").size() == 1);
        CHECK(ScanText("std::printf(\"x\\n\");\n", "s").size() == 1);
        CHECK(ScanText("fprintf(stderr, \"x\\n\");\n", "s").size() == 1);
        CHECK(ScanText("vprintf(fmt, ap);\n", "s").size() == 1);
        CHECK(ScanText("vfprintf(stderr, fmt, ap);\n", "s").size() == 1);
        CHECK(ScanText("puts(\"x\");\n", "s").size() == 1);
        CHECK(ScanText("putchar('x');\n", "s").size() == 1);
        CHECK(ScanText("perror(\"x\");\n", "s").size() == 1);
    }
    SECTION("multi-line calls are attributed to the opening line")
    {
        const std::string code =
            "std::fprintf(\n"
            "    stderr, \"x %d\\n\", 1);\n";
        const auto hits = ScanText(code, "s");
        REQUIRE(hits.size() == 1);
        CHECK(hits[0].line == 1);
    }
    SECTION("one hit per line")
    {
        CHECK(ScanText("printf(\"a\"); printf(\"b\");\n", "s").size() == 1);
    }
    SECTION("similar identifiers are not logging")
    {
        CHECK(ScanText("snprintf(buf, n, \"x\");\n", "s").empty());
        CHECK(ScanText("std::snprintf(buf, n, \"x\");\n", "s").empty());
        CHECK(ScanText("my_printf(\"x\");\n", "s").empty());
        CHECK(ScanText("printfx(\"x\");\n", "s").empty());
        CHECK(ScanText("std::couts << x;\n", "s").empty());
        CHECK(ScanText("const char *s = \"printf(\";\n", "s").empty());
    }
    SECTION("comments and raw strings are ignored")
    {
        CHECK(ScanText("// printf(\"debug\");\n", "s").empty());
        CHECK(ScanText("/* fprintf(stderr, \"x\"); */\n", "s").empty());
        CHECK(ScanText("auto s = R\"(printf(\"x\"))\";\n", "s").empty());
        CHECK(ScanText("char c = '{';\n", "s").empty());
    }
    SECTION("the sanctioned logger is not a hit")
    {
        CHECK(ScanText("log::error(\"printf failed\");\n", "s").empty());
        CHECK(ScanText("spdlog::info(\"x {}\", v);\n", "s").empty());
        CHECK(ScanText("using namespace std;\n", "s").empty());
    }
}

TEST_CASE("logging guard: no stdio logging outside the allowlist",
          "[logging][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    REQUIRE(fs::exists(root / "libs"));
    REQUIRE(fs::exists(root / "apps"));

    const auto hits = ScanTree(root);
    // The scanner must see the declared exception sites; an empty result means
    // the walk or the parser is broken and the case would pass vacuously.
    REQUIRE_FALSE(hits.empty());
    REQUIRE_FALSE(Allowlist().empty());

    // CoverageViolations reports both directions: an undeclared call (with its
    // file:line) and a declared line that no longer exists.
    const auto violations = CoverageViolations(hits, Allowlist());
    std::string joined;
    for (const auto &v : violations) joined += v + "\n";
    INFO("logging guard violations (fix the site or declare it in Allowlist):\n"
         << joined);
    REQUIRE(violations.empty());
}

TEST_CASE("logging guard: allowlist coverage is two-way",
          "[logging][guard]")
{
    const std::vector<AllowedFile> allow = {
        {"a.cpp", "test", "test", {{"printf(\"x\");", 1}}},
    };

    SECTION("exact match is clean")
    {
        CHECK(CoverageViolations({{"a.cpp", 1, "printf(\"x\");"}}, allow)
                  .empty());
    }
    SECTION("extra occurrence fails")
    {
        const std::vector<Hit> hits = {
            {"a.cpp", 1, "printf(\"x\");"},
            {"a.cpp", 2, "printf(\"x\");"},
        };
        CHECK_FALSE(CoverageViolations(hits, allow).empty());
    }
    SECTION("unknown file fails")
    {
        CHECK_FALSE(CoverageViolations({{"b.cpp", 1, "printf(\"x\");"}}, allow)
                        .empty());
    }
    SECTION("stale entry fails")
    {
        CHECK_FALSE(CoverageViolations({}, allow).empty());
    }
    SECTION("undeclared line in a declared file fails")
    {
        CHECK_FALSE(CoverageViolations({{"a.cpp", 1, "fprintf(stderr, \"y\");"}},
                                       allow)
                        .empty());
    }
}
