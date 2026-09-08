// absl str_format guard (host-only, zero deps beyond Catch2).
//
// Bug class: absl::StrFormat / StrAppendFormat / StreamFormat / Format are
// printf-style ("%" conversions). fmt-style "{}" placeholders are printed
// literally ("GPU {}: ..." logs "GPU {}:"), because the abseil compile-time
// format checker is inert under GCC and is patched to "#if 0" for Clang/AOCC
// in the root CMakeLists.txt (checker.h guard).
//
// This guard statically scans libs/ and apps/ for absl format calls whose
// format string contains fmt-style placeholders and fails, listing every
// violation. Conversions cheat-sheet: {} on string-ish -> %s (path -> %s with
// .string()), signed ints (incl. int64_t) -> %d, unsigned ints (incl. size_t)
// -> %u, floats/doubles -> %f/%.Nf, widths {:<N}/{:>N} -> %-Ns/%Ns.
#include <catch2/catch_test_macros.hpp>

#include <absl/strings/str_format.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace {

// absl format entry points (printf-style conversion charset only).
const std::vector<std::string>& FormatEntryPoints()
{
    static const std::vector<std::string> v = {
        "absl::StrFormat",       // (fmt, args...)
        "absl::Format",          // (fmt, args...)
        "absl::StrAppendFormat", // (&sink, fmt, args...)
        "absl::StreamFormat",    // (&stream, fmt, args...)
    };
    return v;
}

// Entry points whose format string is the SECOND argument (first is the
// sink/stream reference).
bool HasSinkArgument(const std::string& entry)
{
    return entry == "absl::StrAppendFormat" || entry == "absl::StreamFormat";
}

// fmt-style placeholder: "{}", "{:.4f}", "{:03d}", "{:<22}", ...
const std::regex& FmtPlaceholder()
{
    static const std::regex r("\\{[^{}]*\\}");
    return r;
}

void SkipWhitespaceAndComments(const std::string& src, size_t& i)
{
    const size_t n = src.size();
    for (;;)
    {
        if (i < n && (src[i] == ' ' || src[i] == '\t' || src[i] == '\r' ||
                      src[i] == '\n'))
        {
            ++i;
            continue;
        }
        if (i + 1 < n && src[i] == '/' && src[i + 1] == '/')
        {
            const size_t e = src.find('\n', i);
            if (e == std::string::npos) { i = n; return; }
            i = e + 1;
            continue;
        }
        if (i + 1 < n && src[i] == '/' && src[i + 1] == '*')
        {
            const size_t e = src.find("*/", i + 2);
            i = (e == std::string::npos) ? n : e + 2;
            continue;
        }
        return;
    }
}

// Skip one C++ token sequence: a string/char literal (escapes kept raw) or a
// single non-delimiter character. Used when walking over a sink argument.
void SkipToken(const std::string& src, size_t& i)
{
    const size_t n = src.size();
    SkipWhitespaceAndComments(src, i);
    if (i >= n) return;
    const char c = src[i];
    if (c == '"' || c == '\'')
    {
        ++i;
        while (i < n)
        {
            if (src[i] == '\\' && i + 1 < n) { i += 2; continue; }
            if (src[i] == c) { ++i; break; }
            ++i;
        }
        return;
    }
    ++i;
}

// Skip a full argument (balanced parens/brackets/braces, literals, comments)
// and the following top-level comma. Returns false at end of input or before
// the argument list's closing ')'.
bool SkipArgumentAndComma(const std::string& src, size_t& i)
{
    const size_t n = src.size();
    int depth = 0;
    for (;;)
    {
        SkipWhitespaceAndComments(src, i);
        if (i >= n) return false;
        const char c = src[i];
        if (depth == 0 && (c == ',' || c == ')'))
        {
            if (c == ')') return false;
            ++i; // consume ','
            return true;
        }
        if (c == '(' || c == '[' || c == '{')
        {
            ++depth;
            ++i;
            continue;
        }
        if (c == ')' || c == ']' || c == '}')
        {
            --depth;
            ++i;
            continue;
        }
        SkipToken(src, i);
    }
}

// Extract the format string (concatenated adjacent string literals, possibly
// spanning lines, with comments skipped) at the current argument position.
// Returns false when the first token is not a string literal (variable format
// string) or a literal is unterminated.
bool ExtractFormatString(const std::string& src, size_t i, std::string& out)
{
    const size_t n = src.size();
    bool got_any = false;
    for (;;)
    {
        SkipWhitespaceAndComments(src, i);
        if (i >= n || src[i] != '"')
            return got_any; // stop at ',', ')' or a non-literal arg
        // Scan one string literal, keeping escapes raw (braces never need
        // unescaping; \" and \\ must not terminate the literal).
        ++i;
        bool closed = false;
        while (i < n)
        {
            const char c = src[i];
            if (c == '\\' && i + 1 < n)
            {
                out += c;
                out += src[i + 1];
                i += 2;
                continue;
            }
            if (c == '"') { ++i; closed = true; break; }
            out += c;
            ++i;
        }
        if (!closed) return false;
        got_any = true;
    }
}

struct Violation
{
    std::string file;
    size_t line = 0;
    std::string entry;
    std::string format;
};

// Scan a source text for absl format calls with fmt-style placeholders.
void ScanSourceText(const std::string& src, const std::string& name,
                    std::vector<Violation>& out)
{
    for (const auto& entry : FormatEntryPoints())
    {
        size_t pos = 0;
        while ((pos = src.find(entry, pos)) != std::string::npos)
        {
            const size_t name_end = pos + entry.size();
            // Reject tails of longer identifiers (e.g. "absl::StrFormatX").
            if (name_end < src.size() &&
                (std::isalnum(static_cast<unsigned char>(src[name_end])) ||
                 src[name_end] == '_'))
            {
                pos = name_end;
                continue;
            }
            size_t i = name_end;
            SkipWhitespaceAndComments(src, i);
            if (i < src.size() && src[i] == '(')
            {
                ++i;
                bool fmt_reachable = true;
                if (HasSinkArgument(entry)) fmt_reachable = SkipArgumentAndComma(src, i);
                std::string fmt;
                if (fmt_reachable && ExtractFormatString(src, i, fmt) &&
                    std::regex_search(fmt, FmtPlaceholder()))
                {
                    const size_t line = 1 + static_cast<size_t>(std::count(
                        src.begin(), src.begin() + static_cast<long>(pos), '\n'));
                    out.push_back({name, line, entry, fmt});
                }
            }
            pos = name_end;
        }
    }
}

void ScanFile(const fs::path& path, std::vector<Violation>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return;
    std::ostringstream ss;
    ss << f.rdbuf();
    ScanSourceText(ss.str(), path.generic_string(), out);
}

void ScanTree(const fs::path& dir, std::vector<Violation>& out)
{
    std::error_code ec;
    if (!fs::exists(dir, ec))
    {
        out.push_back({dir.string() + " (MISSING)", 0, "", ""});
        return;
    }
    for (auto& e : fs::recursive_directory_iterator(dir))
    {
        if (!e.is_regular_file()) continue;
        const std::string ext = e.path().extension().string();
        if (ext != ".cpp" && ext != ".cc" && ext != ".h" && ext != ".hpp" &&
            ext != ".cuh" && ext != ".cu")
            continue;
        ScanFile(e.path(), out);
    }
}

std::vector<Violation> ScanCode(const std::string& code)
{
    std::vector<Violation> v;
    ScanSourceText(code, "snippet", v);
    return v;
}

} // namespace

TEST_CASE("absl format guard: parser flags {} and accepts printf-style",
          "[absl-format][guard]")
{
    SECTION("flags plain {} placeholder")
    {
        CHECK(ScanCode("absl::StrFormat(\"GPU {} MiB\", x);").size() == 1);
    }
    SECTION("flags typed and width placeholders")
    {
        CHECK(ScanCode("absl::StrFormat(\"{:.4f}\", v);").size() == 1);
        CHECK(ScanCode("absl::StrFormat(\"{:03d}\", n);").size() == 1);
        CHECK(ScanCode("absl::StrFormat(\"{:<22} {:>6}\", a, b);").size() == 1);
    }
    SECTION("flags multi-line concatenated literals")
    {
        const std::string code =
            "absl::StrFormat(\n"
            "    \"layer={:03d}  {} -> {}\",\n"
            "    l, e, p);";
        CHECK(ScanCode(code).size() == 1);
    }
    SECTION("sink-argument entry points check the second argument")
    {
        CHECK(ScanCode("absl::StrAppendFormat(&s, \"{}\", x);").size() == 1);
        CHECK(ScanCode("absl::StreamFormat(&o, \"{}\", x);").size() == 1);
        CHECK(ScanCode("absl::StrAppendFormat(&s, \"%d\", x);").empty());
    }
    SECTION("printf-style is clean")
    {
        CHECK(ScanCode("absl::StrFormat(\"GPU %d MiB\", x);").empty());
        CHECK(ScanCode("absl::StrFormat(\"%.1f ms\", v);").empty());
        CHECK(ScanCode("absl::StrFormat(\"%-22s\", s);").empty());
    }
    SECTION("non-absl fmt-style calls are out of scope")
    {
        CHECK(ScanCode("spdlog::info(\"GPU {} MiB\", x);").empty());
        CHECK(ScanCode("fmt::format(\"GPU {} MiB\", x);").empty());
    }
    SECTION("StrCat is concatenation, not a format call")
    {
        CHECK(ScanCode("absl::StrCat(\"GPU {} MiB\", x);").empty());
    }
    SECTION("variable format string is skipped (cannot be checked)")
    {
        CHECK(ScanCode("absl::StrFormat(fmt, x);").empty());
    }
    SECTION("all four entry points are monitored")
    {
        CHECK(ScanCode("absl::StrFormat(\"{}\", x);").size() == 1);
        CHECK(ScanCode("absl::StrAppendFormat(&s, \"{}\", x);").size() == 1);
        CHECK(ScanCode("absl::StreamFormat(&o, \"{}\", x);").size() == 1);
        CHECK(ScanCode("absl::Format(\"{}\", x);").size() == 1);
    }
    SECTION("longer identifiers do not match entry-point prefixes")
    {
        CHECK(ScanCode("absl::StrFormatX(\"{}\", x);").empty());
        CHECK(ScanCode("my_absl::StrFormat(\"{}\", x);").size() == 1);
    }
}

TEST_CASE("absl format guard: runtime semantics of conversions used by fixed sites",
          "[absl-format][guard]")
{
    // Pinned behavior for the conversion cheat-sheet (see file header). With
    // the compile-time checker inert (GCC; patched AOCC) these runtime checks
    // are the safety net for the type/conversion pairs used repo-wide.
    SECTION("%d — signed integrals")
    {
        CHECK(absl::StrFormat("%d", -42) == "-42");
        CHECK(absl::StrFormat("%d", std::int32_t{7}) == "7");
        CHECK(absl::StrFormat("%d", std::int64_t{-1234567890123LL}) ==
              "-1234567890123");
    }
    SECTION("%u — unsigned integrals")
    {
        CHECK(absl::StrFormat("%u", std::size_t{42}) == "42");
        CHECK(absl::StrFormat("%u", std::uint64_t{1ULL << 40}) ==
              "1099511627776");
    }
    SECTION("%s — string-ish")
    {
        CHECK(absl::StrFormat("%s", std::string("abc")) == "abc");
        CHECK(absl::StrFormat("%s", "lit") == "lit");
        char buf[] = "mutable";
        CHECK(absl::StrFormat("%s", buf) == "mutable");
    }
    SECTION("%f family — floats and doubles")
    {
        CHECK(absl::StrFormat("%.1f", 12.34) == "12.3");
        CHECK(absl::StrFormat("%.4f", 0.5) == "0.5000");
        CHECK(absl::StrFormat("%f", 2.5f) == "2.500000");
    }
    SECTION("width / zero-padding / alignment")
    {
        CHECK(absl::StrFormat("%03d", 7) == "007");
        CHECK(absl::StrFormat("%6d", 42) == "    42");
        CHECK(absl::StrFormat("[%-22s]", "cat") ==
              "[cat                   ]");
        CHECK(absl::StrFormat("[%-8s]", "f16") == "[f16     ]");
        CHECK(absl::StrFormat("['%s']", "logestix-qwen") ==
              "['logestix-qwen']");
    }
}

TEST_CASE("absl format guard: no fmt-style placeholders in absl format calls",
          "[absl-format][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);

    std::vector<Violation> violations;
    ScanTree(root / "libs", violations);
    ScanTree(root / "apps", violations);

    std::sort(violations.begin(), violations.end(),
              [](const Violation& a, const Violation& b) {
                  if (a.file != b.file) return a.file < b.file;
                  return a.line < b.line;
              });
    std::string joined;
    for (const auto& v : violations)
    {
        joined += v.file + ":" + std::to_string(v.line) + "  " + v.entry +
                  "(\"" + v.format + "\")\n";
    }
    INFO("absl fmt-style violations (" << violations.size() << "):\n" << joined);
    REQUIRE(violations.empty());
}
