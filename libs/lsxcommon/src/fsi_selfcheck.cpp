#include "lsxcommon/fsi_selfcheck.h"

#include <absl/strings/ascii.h>
#include <simdjson.h>
#include <cctype>
#include <fstream>
#include <sstream>

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

bool ParseFsiGolden(std::string_view json, FsiGolden &out, std::string *error)
{
    out = {};
    simdjson::ondemand::parser parser;
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    if (parser.iterate(pad).get(doc) != simdjson::SUCCESS) {
        if (error) *error = "invalid JSON";
        return false;
    }
    simdjson::ondemand::object root;
    if (doc.get_object().get(root) != simdjson::SUCCESS) {
        if (error) *error = "root is not an object";
        return false;
    }
    std::string_view tag;
    if (root["format"].get_string().get(tag) != simdjson::SUCCESS ||
        tag != "logestix-fsi-golden") {
        if (error) *error = "missing/unknown format tag";
        return false;
    }
    std::string_view model_type;
    if (root["model_type"].get_string().get(model_type) != simdjson::SUCCESS ||
        model_type.empty()) {
        if (error) *error = "missing model_type";
        return false;
    }
    out.model_type = std::string(model_type);
    simdjson::ondemand::array prompts;
    if (root["prompts"].get_array().get(prompts) != simdjson::SUCCESS) {
        if (error) *error = "missing prompts array";
        return false;
    }
    for (auto element : prompts) {
        simdjson::ondemand::object o;
        if (element.get_object().get(o) != simdjson::SUCCESS) {
            if (error) *error = "prompt entry is not an object";
            return false;
        }
        FsiGoldenCase c;
        std::string_view sv;
        if (o["id"].get_string().get(sv) != simdjson::SUCCESS || sv.empty()) {
            if (error) *error = "prompt missing id";
            return false;
        }
        c.id = std::string(sv);
        if (o["kind"].get_string().get(sv) != simdjson::SUCCESS) {
            if (error) *error = c.id + ": missing kind";
            return false;
        }
        if (sv == "arithmetic") c.kind = FsiKind::kArithmetic;
        else if (sv == "comparison") c.kind = FsiKind::kComparison;
        else { if (error) *error = c.id + ": unknown kind"; return false; }
        if (o["anchor"].get_string().get(sv) != simdjson::SUCCESS ||
            sv.empty()) {
            if (error) *error = c.id + ": missing anchor";
            return false;
        }
        c.anchor = std::string(sv);
        simdjson::ondemand::array ids;
        if (o["input_ids"].get_array().get(ids) != simdjson::SUCCESS) {
            if (error) *error = c.id + ": missing input_ids";
            return false;
        }
        for (auto v : ids) {
            int64_t id = 0;
            if (v.get_int64().get(id) != simdjson::SUCCESS) {
                if (error) *error = c.id + ": non-integer input id";
                return false;
            }
            c.input_ids.push_back(static_cast<int32_t>(id));
        }
        if (c.input_ids.empty()) {
            if (error) *error = c.id + ": empty input_ids";
            return false;
        }
        int64_t mnt = 0;
        if (o["max_new_tokens"].get_int64().get(mnt) == simdjson::SUCCESS &&
            mnt > 0)
            c.max_new_tokens = static_cast<int>(mnt);
        if (o["expected"].get_string().get(sv) != simdjson::SUCCESS ||
            sv.empty()) {
            if (error) *error = c.id + ": missing expected";
            return false;
        }
        c.expected = std::string(sv);
        std::string_view raw;
        if (o["raw"].get_string().get(raw) == simdjson::SUCCESS)
            c.raw = std::string(raw);
        out.cases.push_back(std::move(c));
    }
    if (out.cases.empty()) {
        if (error) *error = "no prompts in golden";
        return false;
    }
    return true;
}

bool LoadFsiGoldenFile(const std::string &path, FsiGolden &out,
                       std::string *error)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        if (error) *error = "cannot open " + path;
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::string local;
    if (!ParseFsiGolden(text, out, &local)) {
        if (error) *error = path + ": " + local;
        return false;
    }
    return true;
}

}  // namespace fsi
}  // namespace lsxcommon
