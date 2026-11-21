#include "lsxcommon/code_selfcheck.h"
#include "lsxcommon/selfcheck_common.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>

#include "lsxcommon/atem_channel.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {
namespace code {
namespace {
// Dedup Ф5: fence stripper is the shared lsxcommon::selfcheck body.
using lsxcommon::selfcheck::Unfence;


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

// Strip everything before the last </think> so a reasoning chain does not
// shadow the final answer (same rule as FSI).
std::string Preprocess(std::string_view raw)
{
    std::string text = Unfence(raw);
    const std::size_t close = text.rfind("</think>");
    if (close != std::string::npos) text = text.substr(close + 8);
    return text;
}

// Balanced {..} extraction with string-literal awareness, then minify by
// removing whitespace outside strings.
std::string ExtractJson(std::string_view text, std::string_view anchor)
{
    const std::size_t start = text.find('{');
    if (start == std::string_view::npos) return {};
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
        else if (!std::isspace(static_cast<unsigned char>(c))) out.push_back(c);
    }
    if (!anchor.empty() && out.find(anchor) == std::string::npos) return {};
    return out;
}

// Find `anchor` directly followed (spaces allowed) by '='. An assignment kind
// has no valid answer without the '='; there is deliberately no fallback.
std::size_t FindAssignmentAnchor(std::string_view s, std::string_view anchor)
{
    std::size_t pos = s.find(anchor);
    while (pos != std::string_view::npos) {
        std::size_t i = pos + anchor.size();
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
            ++i;
        if (i < s.size() && s[i] == '=') return pos;
        pos = s.find(anchor, pos + 1);
    }
    return std::string_view::npos;
}

std::string ExtractAssignment(std::string_view text, std::string_view anchor)
{
    if (anchor.empty()) return {};
    const std::size_t a = FindAssignmentAnchor(text, anchor);
    if (a == std::string_view::npos) return {};
    const std::size_t line_start = (text.rfind('\n', a) == std::string_view::npos)
                                       ? 0
                                       : text.rfind('\n', a) + 1;
    std::size_t line_end = text.find('\n', a);
    if (line_end == std::string_view::npos) line_end = text.size();
    const std::string_view line = text.substr(line_start, line_end - line_start);
    std::size_t i = a - line_start;
    const std::size_t start = i;
    int depth = 0;
    bool first = true;
    bool prev_op = true;
    while (i < line.size()) {
        const char c = line[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '(') { ++depth; prev_op = true; first = false; ++i; continue; }
        if (c == ')') {
            if (depth == 0) break;
            --depth; prev_op = false; first = false; ++i; continue;
        }
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '=') {
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
    if (i <= start) return {};
    return Normalize(line.substr(start, i - start));
}

bool IsRegexToken(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) ||
           std::strchr("\\[]{}()|.*+?^$/-", c) != nullptr;
}

std::string ExtractRegex(std::string_view text, std::string_view anchor)
{
    if (anchor.empty()) return {};
    const std::size_t a = text.find(anchor);
    if (a == std::string_view::npos) return {};
    const std::size_t line_start = (text.rfind('\n', a) == std::string_view::npos)
                                       ? 0
                                       : text.rfind('\n', a) + 1;
    std::size_t line_end = text.find('\n', a);
    if (line_end == std::string_view::npos) line_end = text.size();
    const std::string_view line = text.substr(line_start, line_end - line_start);
    std::size_t l = a - line_start;
    std::size_t r = l + anchor.size();
    while (r < line.size() && IsRegexToken(line[r])) ++r;
    while (l > 0 && IsRegexToken(line[l - 1])) --l;
    if (r <= l) return {};
    return Normalize(line.substr(l, r - l));
}

} // namespace

std::string ExtractCodeAnswer(CodeKind kind, std::string_view anchor,
                              std::string_view raw)
{
    const std::string text = Preprocess(raw);
    switch (kind) {
        case CodeKind::kJson: return ExtractJson(text, anchor);
        case CodeKind::kAssignment: return ExtractAssignment(text, anchor);
        case CodeKind::kRegex: return ExtractRegex(text, anchor);
    }
    return {};
}

bool ParseCodeGolden(std::string_view json, CodeGolden &out, std::string *error)
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
        tag != "logestix-code-golden") {
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
        CodeGoldenCase c;
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
        if (sv == "json") c.kind = CodeKind::kJson;
        else if (sv == "assignment") c.kind = CodeKind::kAssignment;
        else if (sv == "regex") c.kind = CodeKind::kRegex;
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
        bool ign = false;
        if (o["ignore_eos"].get_bool().get(ign) == simdjson::SUCCESS)
            c.ignore_eos = ign;
        int64_t ans = -1;
        if (o["answer_start_token_id"].get_int64().get(ans) ==
            simdjson::SUCCESS)
            c.answer_start_token_id = static_cast<int32_t>(ans);
        if (o["expected"].get_string().get(sv) != simdjson::SUCCESS ||
            sv.empty()) {
            if (error) *error = c.id + ": missing expected";
            return false;
        }
        c.expected = std::string(sv);
        std::string_view raw;
        if (o["raw"].get_string().get(raw) == simdjson::SUCCESS)
            c.raw = std::string(raw);
        std::string_view body;
        if (o["body"].get_string().get(body) == simdjson::SUCCESS)
            c.body = std::string(body);
        std::string_view ach;
        if (o["answer_channel"].get_string().get(ach) == simdjson::SUCCESS)
            c.answer_channel = std::string(ach);
        std::string_view rch;
        if (o["reasoning_channel"].get_string().get(rch) == simdjson::SUCCESS)
            c.reasoning_channel = std::string(rch);
        out.cases.push_back(std::move(c));
    }
    if (out.cases.empty()) {
        if (error) *error = "no prompts in golden";
        return false;
    }
    return true;
}

bool LoadCodeGoldenFile(const std::string &path, CodeGolden &out,
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
    if (!ParseCodeGolden(text, out, &local)) {
        if (error) *error = path + ": " + local;
        return false;
    }
    return true;
}

namespace {

// A fragile reasoning model can fail to emit a turn-end token and echo the
// prompt after its answer. When the golden carries the prompt body, cut the
// decoded text at the body's prefix (after the reasoning close) so the echo
// does not bleed into the extracted answer.
std::string CutPromptEcho(std::string_view text, std::string_view body)
{
    if (body.empty()) return std::string(text);
    std::string s(text);
    const std::size_t close = s.rfind("</think>");
    if (close != std::string::npos) s = s.substr(close + 8);
    const std::size_t n = std::min<std::size_t>(32, body.size());
    const std::size_t pos = s.find(std::string_view(body).substr(0, n));
    if (pos != std::string::npos) s = s.substr(0, pos);
    return s;
}

std::vector<int32_t> TailAfter(const std::vector<int32_t> &ids, int32_t marker)
{
    if (marker < 0) return ids;
    for (std::size_t i = ids.size(); i > 0; --i)
        if (ids[i - 1] == marker)
            return std::vector<int32_t>(ids.begin() + static_cast<long>(i),
                                        ids.end());
    return ids;
}


CodeCaseResult RunCodeCase(IModel &model, const CodeGoldenCase &c,
                           const CodeDecodeFn &decode)
{
    CodeCaseResult r;
    r.id = c.id;
    r.expected = c.expected;
    ModelRequest req;
    req.input_ids = c.input_ids;
    req.max_gen_tokens = c.max_new_tokens;
    req.ignore_eos = c.ignore_eos;
    const gpu::InferenceResult ir = model.Infer(req);
    if (!ir.ok) {
        r.error = ir.error_msg.empty() ? "inference failed" : ir.error_msg;
        return r;
    }
    const std::string text = decode(TailAfter(ir.gen_tokens,
                                              c.answer_start_token_id));
    r.raw = text;
    const std::string channel =
        atem::SelectAnswerChannel(text, c.answer_channel,
                                  c.reasoning_channel);
    r.extracted =
        ExtractCodeAnswer(c.kind, c.anchor, CutPromptEcho(channel, c.body));
    r.passed = (r.extracted == c.expected);
    return r;
}

} // namespace

CodeResult RunCodeSelfCheck(IModel &model, const CodeGolden &golden,
                            const CodeDecodeFn &decode)
{
    CodeResult res;
    // Progress line per case (see RunFsiSelfCheck): the verdict lines arrive
    // only after the last case, so a slow model needs a heartbeat.
    for (std::size_t i = 0; i < golden.cases.size(); ++i) {
        const CodeGoldenCase &c = golden.cases[i];
        lsxcommon::log::info(absl::StrFormat(
            "[code] %s start (%u/%u) max_new_tokens=%d ignore_eos=%s",
            c.id.c_str(), static_cast<unsigned>(i + 1),
            static_cast<unsigned>(golden.cases.size()), c.max_new_tokens,
            c.ignore_eos ? "true" : "false"));
        res.cases.push_back(RunCodeCase(model, c, decode));
    }
    res.all_passed = !res.cases.empty();
    for (const CodeCaseResult &c : res.cases) {
        lsxcommon::log::info(absl::StrFormat(
            "[code] %s expected='%s' extracted='%s' passed=%s raw='%s'",
            c.id.c_str(), c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "true" : "false", c.raw.c_str()));
        if (!c.passed) res.all_passed = false;
    }
    return res;
}

CodeResult RunCodeSelfCheck(IModel &model, const CodeGolden &golden)
{
    return RunCodeSelfCheck(
        model, golden,
        [&model](const std::vector<int32_t> &ids) {
            return runtime::DecodeGeneratedText(model.Tokenizer(), ids);
        });
}

}  // namespace code
}  // namespace lsxcommon
