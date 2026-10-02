// lsxcommon::json::JsonEscape — the single JSON string-escaping contract
// (host-only; no CUDA, no model).
//
// Two responsibilities:
//   1. The escape contract itself (JSON RFC 8259 §7). Everything below U+0020
//      other than the three mandatory short forms must become a \u00XX escape;
//      bytes >= 0x80 must pass through so UTF-8 survives.
//   2. A drift guard over the family-local copies of the same function. The
//      neutral header in json_escape.h is the reference, but the qwen, glm-flash
//      and mimo codecs plus lsxnetwork's openai_endpoint each carry their own
//      copy (the lsxnetwork one because that library does not link lsxcommon).
//      A copy that drifts turns a model answer into a body no JSON parser
//      accepts, and nothing else in the suite compares them.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <lsxnetwork/json_escape.h>

#ifndef LSX_SOURCE_DIR
#define LSX_SOURCE_DIR "."
#endif

namespace fs = std::filesystem;

namespace {

using lsxnetwork::json::JsonEscape;

std::string EscapeOne(unsigned char c)
{
    return JsonEscape(std::string_view(reinterpret_cast<const char *>(&c), 1));
}

// ── source scanning for the drift guard ────────────────────────────────────

std::string ReadFile(const fs::path &p)
{
    std::ifstream in(p);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Body of the function named `sym`, brace-matched from its opening brace.
std::string FunctionBody(const std::string &text, const std::string &sym)
{
    const size_t sig = text.find(sym + "(");
    if (sig == std::string::npos) return {};
    const size_t open = text.find('{', sig);
    if (open == std::string::npos) return {};
    int depth = 0;
    for (size_t i = open; i < text.size(); ++i) {
        if (text[i] == '{') ++depth;
        else if (text[i] == '}' && --depth == 0) return text.substr(open, i - open + 1);
    }
    return {};
}

// The semantic core: the ordered `case X: ... break;` arms and the fallback
// branch. Loop-variable spelling and pointer-vs-array hex tables are noise, so
// they are normalized away; a changed escape sequence is not.
std::string CanonicalArms(const std::string &body)
{
    static const std::regex kArm(
        R"X(case\s+('[^']*'|0x[0-9A-Fa-f]{2})\s*:([^;]*);)X");
    std::string out;
    for (std::sregex_iterator it(body.begin(), body.end(), kArm), end;
         it != end; ++it) {
        out += (*it)[1].str();
        std::string act = (*it)[2].str();
        // Drop identifiers and punctuation-only noise, keep literals.
        static const std::regex kKeep(
            R"X("(?:[^"\\]|\\.)*"|[A-Za-z_][A-Za-z_0-9]*)X");
        std::string kept;
        for (std::sregex_iterator k(act.begin(), act.end(), kKeep), ke;
             k != ke; ++k)
            kept += k->str() + ";";
        out += kept;
    }
    // The < 0x20 fallback must exist and use the \u00 form.
    out += body.find("\\u00") != std::string::npos ? "|u00" : "|NO-U00";
    return out;
}

struct CopyRow {
    const char *label;
    const char *rel_path;
    const char *symbol;
};

// lsxcommon now has exactly one implementation. The qwen / glm-flash / mimo
// codecs that each carried a byte-identical copy now call it, so the drift
// guard became the reverse check below. The lsxnetwork copy stays: that library
// does not link lsxcommon, and its own comment says so.
const std::vector<CopyRow> &Copies()
{
    // Exactly one implementation, in lsxnetwork: the dependency runs
    // lsxcommon -> lsxnetwork, so the shared header has to live on the lower
    // side of it. The qwen / glm-flash / mimo codecs and the OpenAI +
    // Anthropic SSE serializers all include it.
    static const std::vector<CopyRow> v = {
        {"reference (lsxnetwork/json_escape.h)",
         "libs/lsxnetwork/include/lsxnetwork/json_escape.h", "JsonEscape"},
    };
    return v;
}

} // namespace

TEST_CASE("json escape: the three mandatory short forms and the backslash",
          "[json][escape]")
{
    CHECK(JsonEscape("plain text") == "plain text");
    CHECK(JsonEscape("say \"hi\"") == "say \\\"hi\\\"");
    CHECK(JsonEscape("a\\b") == "a\\\\b");
    CHECK(JsonEscape("l1\nl2\r\tl3") == "l1\\nl2\\r\\tl3");
    CHECK(JsonEscape("") == "");
}

TEST_CASE("json escape: every other control byte becomes \\u00XX",
          "[json][escape]")
{
    // 0x00 NUL, 0x08 backspace, 0x0B vertical tab, 0x0C form feed, 0x1B ESC.
    CHECK(EscapeOne(0x00) == "\\u0000");
    CHECK(EscapeOne(0x08) == "\\u0008");
    CHECK(EscapeOne(0x0B) == "\\u000b");
    CHECK(EscapeOne(0x0C) == "\\u000c");
    CHECK(EscapeOne(0x1B) == "\\u001b");
    // DEL (0x7F) is not a JSON-mandated escape and must pass through.
    CHECK(EscapeOne(0x7F) == "\x7f");
    // 0x1F is the last byte below U+0020 that is not a short form.
    CHECK(EscapeOne(0x1F) == "\\u001f");
}

TEST_CASE("json escape: UTF-8 bytes pass through unescaped",
          "[json][escape]")
{
    // A JSON parser requires valid UTF-8 to be emitted as-is; escaping the
    // high bytes would still parse, but the apostrophe form absl::CEscape
    // produces (\\') and its octal \\001 are not valid JSON at all.
    const std::string cyrillic = "\xd0\x9f\xd1\x80\xd0\xb8";
    CHECK(JsonEscape(cyrillic) == cyrillic);
    CHECK(JsonEscape("\xe2\x80\x8b") == "\xe2\x80\x8b");
    // The apostrophe is NOT escaped: \' is invalid JSON.
    CHECK(JsonEscape("it's") == "it's");
}

TEST_CASE("json escape: a realistic model answer stays parseable",
          "[json][escape]")
{
    // The failure this guards: a control character in a model response
    // producing a body no JSON parser accepts.
    const std::string answer = "line one\nline \"two\"\tpath\\to\x01 end";
    const std::string escaped = JsonEscape(answer);
    CHECK(escaped.find('\n') == std::string::npos);
    CHECK(escaped.find('\t') == std::string::npos);
    CHECK(escaped.find('\x01') == std::string::npos);
    // Every remaining quote is part of an escape pair, never a bare one.
    int bare = 0;
    for (size_t i = 0; i < escaped.size(); ++i)
        if (escaped[i] == '"' && (i == 0 || escaped[i - 1] != '\\')) ++bare;
    CHECK(bare == 0);
}

TEST_CASE("json escape: every copy matches the reference arms",
          "[json][escape][guard]")
{
    const fs::path root(LSX_SOURCE_DIR);
    std::string reference;
    for (const CopyRow &row : Copies()) {
        INFO("copy " << row.label << " at " << row.rel_path);
        const fs::path p = root / row.rel_path;
        REQUIRE(fs::exists(p));
        const std::string body = FunctionBody(ReadFile(p), row.symbol);
        REQUIRE_FALSE(body.empty());
        const std::string arms = CanonicalArms(body);
        if (reference.empty()) {
            reference = arms;
            REQUIRE_FALSE(reference.empty());
            continue;
        }
        CHECK(arms == reference);
    }
}

TEST_CASE("json escape: lsxcommon keeps no family-local copy",
          "[json][escape][guard]")
{
    // The qwen / glm-flash / mimo codecs and lsxnetwork's openai_endpoint each
    // used to carry a byte-identical copy of the escape loop. Reintroducing one
    // is easy and silent — it would compile, pass every other test and then
    // diverge. The hex digit table is the fingerprint of a copy.
    const fs::path root(LSX_SOURCE_DIR);
    static const std::regex kHexTable(R"(static\s+const\s+char\s*\*?\s*kHex)");
    std::vector<std::string> found;
    for (const char *sub : {"lsxcommon", "lsxnetwork"}) {
        std::error_code ec;
        for (auto it = fs::recursive_directory_iterator(root / "libs" / sub, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (!it->is_regular_file()) continue;
            const auto ext = it->path().extension().string();
            if (ext != ".h" && ext != ".cpp" && ext != ".cu") continue;
            if (std::regex_search(ReadFile(it->path()), kHexTable))
                found.push_back(
                    it->path().lexically_relative(root).generic_string());
        }
    }
    std::sort(found.begin(), found.end());
    std::string joined;
    for (const auto &f : found) joined += f + "\n";
    INFO("files declaring their own hex table:\n" << joined);
    // Exactly one, in lsxnetwork. A second copy anywhere means someone decided
    // the dependency direction made it unavoidable.
    REQUIRE(found.size() == 1);
    CHECK(found[0] == "libs/lsxnetwork/include/lsxnetwork/json_escape.h");
}
