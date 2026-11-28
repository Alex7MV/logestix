#include "lsxcommon/selfcheck_common.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_format.h>
#include <absl/strings/strip.h>

#include <algorithm>
#include <cctype>

namespace lsxcommon {
namespace selfcheck {

std::string Unfence(std::string_view raw)
{
    std::string s(absl::StripAsciiWhitespace(raw));
    const std::size_t open = s.find("```");
    if (open == std::string::npos) return s;
    std::size_t begin = open + 3;
    const std::size_t line_end = s.find('\n', begin);
    if (line_end != std::string::npos) {
        const std::string first = s.substr(begin, line_end - begin);
        bool lang = !first.empty();
        for (char c : first)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
                  c == '+' || c == '-'))
                lang = false;
        if (lang) begin = line_end + 1;
    }
    const std::size_t close = s.find("```", begin);
    const std::string inner = (close == std::string::npos)
                                  ? s.substr(begin)
                                  : s.substr(begin, close - begin);
    return std::string(absl::StripAsciiWhitespace(inner));
}

namespace {

// Byte offset just past the last reasoning close, or npos when neither marker
// appears.
std::size_t AfterReasoningClose(const std::string &s)
{
    std::size_t after = std::string::npos;
    if (const std::size_t c = s.rfind("</think>"); c != std::string::npos)
        after = c + 8;
    if (const std::size_t c = s.rfind("[COT_END]");
        c != std::string::npos && (after == std::string::npos || c + 9 > after))
        after = c + 9;
    return after;
}

} // namespace

std::string StripBeforeFinalAnswer(std::string_view raw)
{
    std::string text = Unfence(raw);
    const std::size_t after = AfterReasoningClose(text);
    if (after != std::string::npos) text = text.substr(after);
    return text;
}

std::string CutPromptEcho(std::string_view text, std::string_view body)
{
    if (body.empty()) return std::string(text);
    std::string s(text);
    const std::size_t after = AfterReasoningClose(s);
    if (after != std::string::npos) s = s.substr(after);
    const std::size_t n = std::min<std::size_t>(32, body.size());
    const std::size_t pos = s.find(std::string_view(body).substr(0, n));
    if (pos != std::string::npos) s = s.substr(0, pos);
    return s;
}

std::string ExtractMinifiedJson(
    std::string_view text, std::string_view anchor,
    const std::function<bool(std::string_view)> &accept)
{
    // Balanced-brace scan with string-literal awareness. Every '{' is tried as
    // a candidate start; a candidate that never closes ends the search (there
    // is no later complete object either).
    for (std::size_t start = text.find('{'); start != std::string_view::npos;
         start = text.find('{', start + 1)) {
        int depth = 0;
        bool in_str = false;
        bool esc = false;
        std::size_t end = std::string_view::npos;
        for (std::size_t i = start; i < text.size(); ++i) {
            const char c = text[i];
            if (in_str) {
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
                continue;
            }
            if (c == '"') { in_str = true; continue; }
            if (c == '{') { ++depth; continue; }
            if (c == '}') {
                --depth;
                if (depth == 0) { end = i + 1; break; }
            }
        }
        if (end == std::string_view::npos) return {};
        const std::string_view sub = text.substr(start, end - start);
        std::string out;
        out.reserve(sub.size());
        in_str = false;
        esc = false;
        for (char c : sub) {
            if (in_str) {
                out.push_back(c);
                if (esc) esc = false;
                else if (c == '\\') esc = true;
                else if (c == '"') in_str = false;
                continue;
            }
            if (c == '"') { in_str = true; out.push_back(c); }
            else if (!std::isspace(static_cast<unsigned char>(c)))
                out.push_back(c);
        }
        if (!anchor.empty() && out.find(anchor) == std::string::npos) continue;
        if (accept && !accept(out)) continue;
        return out;
    }
    return {};
}

std::string EscapeForLog(std::string_view in)
{
    std::string out;
    out.reserve(in.size() + 16);
    for (unsigned char c : in) {
        switch (c) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\0': out += "\\0"; break;
            default:
                if (c < 0x20 || c == 0x7f) {
                    // absl::StrFormat, not a local hex table: the tree has one
                    // JSON escaper with its own table and test_json_escape
                    // forbids a second copy anywhere (measured: it caught this
                    // one on the commit that added MCP).
                    out += absl::StrFormat("\\x%02x", static_cast<unsigned>(c));
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

} // namespace selfcheck
} // namespace lsxcommon
