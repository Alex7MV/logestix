#include "lsxcommon/fsi_selfcheck.h"

#include <absl/strings/ascii.h>
#include <cctype>

namespace lsxcommon {
namespace fsi {
namespace {

bool IsComparisonOp(char c)
{
    return c == '<' || c == '>' || c == '=' || c == '!';
}

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

std::string FirstLineWith(const std::string &s, std::string_view anchor)
{
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const std::size_t nl = s.find('\n', pos);
        const std::string line =
            (nl == std::string::npos) ? s.substr(pos)
                                      : s.substr(pos, nl - pos);
        if (line.find(anchor) != std::string::npos) return line;
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    return s;
}

std::string Normalize(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(c);
    while (!out.empty() && (out.back() == '.' || out.back() == ',' ||
                            out.back() == ';'))
        out.pop_back();
    return out;
}

} // namespace

std::string ExtractFsiFormula(FsiKind kind, std::string_view anchor,
                              std::string_view raw)
{
    const std::string line = FirstLineWith(Unfence(raw), anchor);
    if (kind == FsiKind::kComparison) {
        const std::size_t a = line.find(anchor);
        if (a == std::string::npos) return {};
        std::size_t i = a + anchor.size();
        while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
        const std::size_t op = i;
        while (i < line.size() && IsComparisonOp(line[i])) ++i;
        if (i == op) return {};
        while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
        if (i >= line.size() ||
            !(std::isalpha((unsigned char)line[i]) || line[i] == '_'))
            return {};
        while (i < line.size() &&
               (std::isalnum((unsigned char)line[i]) || line[i] == '_'))
            ++i;
        return Normalize(std::string_view(line).substr(a, i - a));
    }
    const std::size_t a = line.find(anchor);
    if (a == std::string::npos) return {};
    const std::size_t eq = line.find('=', a + anchor.size());
    if (eq == std::string::npos) return {};
    std::size_t i = eq + 1;
    int depth = 0;
    bool first = true;
    bool prev_op = true;
    while (i < line.size()) {
        const char c = line[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '(') {
            ++depth; prev_op = true; first = false; ++i; continue;
        }
        if (c == ')') {
            if (depth == 0) break;
            --depth; prev_op = false; first = false; ++i; continue;
        }
        if (c == '+' || c == '-' || c == '*' || c == '/') {
            prev_op = true; first = false; ++i; continue;
        }
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' ||
            c == '.') {
            if (!first && depth == 0 && !prev_op) break;
            while (i < line.size() &&
                   (std::isalnum(static_cast<unsigned char>(line[i])) ||
                    line[i] == '_' || line[i] == '.'))
                ++i;
            prev_op = false; first = false; continue;
        }
        break;
    }
    return Normalize(std::string_view(line).substr(a, i - a));
}

}  // namespace fsi
}  // namespace lsxcommon
