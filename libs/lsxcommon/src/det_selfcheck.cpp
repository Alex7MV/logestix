#include "lsxcommon/det_selfcheck.h"
#include "lsxcommon/selfcheck_common.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include "lsxcommon/atem_channel.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {
namespace det {
namespace {
// Dedup Ф5: fence stripper, reasoning cut, echo cut and the log escaper are the
// shared lsxcommon::selfcheck bodies.
using lsxcommon::selfcheck::CutPromptEcho;
using lsxcommon::selfcheck::EscapeForLog;
using lsxcommon::selfcheck::StripBeforeFinalAnswer;

bool IsWordChar(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

// First standalone token matching [A-Z][A-Z0-9_]* with length >= 2.
std::string ExtractToken(std::string_view text)
{
    const std::size_t n = text.size();
    for (std::size_t i = 0; i < n; ++i) {
        const char c = text[i];
        if (c < 'A' || c > 'Z') continue;
        if (i > 0 && IsWordChar(text[i - 1])) continue;
        std::size_t j = i + 1;
        while (j < n) {
            const char d = text[j];
            const bool ok = (d >= 'A' && d <= 'Z') ||
                            (d >= '0' && d <= '9') || d == '_';
            if (!ok) break;
            ++j;
        }
        if (j - i < 2) { i = j; continue; }
        if (j < n && IsWordChar(text[j])) { i = j; continue; }
        return std::string(text.substr(i, j - i));
    }
    return {};
}

} // namespace

std::string ExtractDetAnswer(DetKind kind, std::string_view raw)
{
    const std::string text = StripBeforeFinalAnswer(raw);
    switch (kind) {
        case DetKind::kToken: return ExtractToken(text);
    }
    return {};
}

bool ParseDetGolden(std::string_view json, DetGolden &out, std::string *error)
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
        tag != "logestix-det-golden") {
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
        DetGoldenCase c;
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
        if (sv == "token") c.kind = DetKind::kToken;
        else { if (error) *error = c.id + ": unknown kind"; return false; }
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
        int64_t sr = 0;
        if (o["stress_runs"].get_int64().get(sr) == simdjson::SUCCESS) {
            if (sr < 1) {
                if (error) *error = c.id + ": stress_runs must be >= 1";
                return false;
            }
            c.stress_runs = static_cast<int>(sr);
        }
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

bool LoadDetGoldenFile(const std::string &path, DetGolden &out,
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
    if (!ParseDetGolden(text, out, &local)) {
        if (error) *error = path + ": " + local;
        return false;
    }
    return true;
}

namespace {

DetCaseResult RunDetCase(IModel &model, const DetGoldenCase &c,
                         const DetDecodeFn &decode, int stress_runs_override)
{
    DetCaseResult r;
    r.id = c.id;
    r.expected = c.expected;
    int runs = (stress_runs_override > 0) ? stress_runs_override
                                          : c.stress_runs;
    if (runs < 3) runs = 3;
    r.runs = runs;

    ModelRequest req;
    req.input_ids = c.input_ids;
    req.max_gen_tokens = c.max_new_tokens;
    req.ignore_eos = c.ignore_eos;

    std::vector<int32_t> first;
    bool identical = true;
    for (int i = 0; i < runs; ++i) {
        const gpu::InferenceResult ir = model.Infer(req);
        if (!ir.ok) {
            r.error = ir.error_msg.empty() ? "inference failed" : ir.error_msg;
            return r;
        }
        if (i == 0) first = ir.gen_tokens;
        else if (ir.gen_tokens != first) identical = false;
    }
    r.deterministic = identical;
    std::vector<int32_t> tail = first;
    if (c.answer_start_token_id >= 0)
    {
        for (std::size_t i = first.size(); i > 0; --i)
            if (first[i - 1] == c.answer_start_token_id)
            {
                tail.assign(first.begin() + static_cast<long>(i), first.end());
                break;
            }
    }
    const std::string text = decode(tail);
    r.raw = text;
    const std::string channel =
        atem::SelectAnswerChannel(text, c.answer_channel,
                                  c.reasoning_channel);
    r.extracted = ExtractDetAnswer(c.kind, CutPromptEcho(channel, c.body));
    r.passed = r.deterministic && (r.extracted == c.expected);
    return r;
}

} // namespace

DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden,
                          const DetDecodeFn &decode, int stress_runs_override)
{
    DetResult res;
    // Progress line per case and per determinism run (see RunFsiSelfCheck):
    // DET repeats each prompt `runs` times, so it is the slowest of the three
    // gates and the one that most needs a heartbeat.
    for (std::size_t i = 0; i < golden.cases.size(); ++i) {
        const DetGoldenCase &c = golden.cases[i];
        int runs = (stress_runs_override > 0) ? stress_runs_override
                                              : c.stress_runs;
        if (runs < 3) runs = 3;
        lsxcommon::log::info(absl::StrFormat(
            "[det] %s start (%u/%u) runs=%d max_new_tokens=%d",
            c.id.c_str(), static_cast<unsigned>(i + 1),
            static_cast<unsigned>(golden.cases.size()), runs,
            c.max_new_tokens));
        res.cases.push_back(RunDetCase(model, c, decode, stress_runs_override));
        // An inference failure is NOT a determinism failure: the runs never
        // completed, so nothing was compared. Reporting it as
        // `deterministic=false` sends the reader after a numeric-instability bug
        // when the real cause is (measured 2026-11-07) a VRAM alloc under a
        // concurrent GPU workload. Say which one it is, and name the error.
        const DetCaseResult &last = res.cases.back();
        if (!last.error.empty())
            lsxcommon::log::info(absl::StrFormat(
                "[det] %s done inference-error error='%s'", c.id.c_str(),
                last.error.c_str()));
        else
            lsxcommon::log::info(absl::StrFormat(
                "[det] %s done deterministic=%s", c.id.c_str(),
                last.deterministic ? "true" : "false"));
    }
    res.all_passed = !res.cases.empty();
    for (const DetCaseResult &c : res.cases) {
        lsxcommon::log::info(absl::StrFormat(
            "[det] %s runs=%d deterministic=%s expected='%s' extracted='%s' "
            "passed=%s error='%s' raw='%s'",
            c.id.c_str(), c.runs, c.deterministic ? "true" : "false",
            c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "true" : "false", c.error.c_str(),
            EscapeForLog(c.raw).c_str()));
        if (!c.passed) res.all_passed = false;
    }
    return res;
}

DetResult RunDetSelfCheck(IModel &model, const DetGolden &golden)
{
    return RunDetSelfCheck(
        model, golden,
        [&model](const std::vector<int32_t> &ids) {
            return runtime::DecodeGeneratedText(model.Tokenizer(), ids);
        });
}

}  // namespace det
}  // namespace lsxcommon
