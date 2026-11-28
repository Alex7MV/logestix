#include "lsxcommon/mcp_selfcheck.h"
#include "lsxcommon/selfcheck_common.h"

#include <absl/strings/str_format.h>
#include <lsxnetwork/json_escape.h>
#include <simdjson.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

#include "lsxcommon/atem_channel.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"

namespace lsxcommon {
namespace mcp {
namespace {
// Dedup Ф5: the fence stripper, the reasoning cut, the echo cut, the JSON
// extractor and the log escaper are the shared lsxcommon::selfcheck bodies.
using lsxcommon::selfcheck::CutPromptEcho;
using lsxcommon::selfcheck::EscapeForLog;
using lsxcommon::selfcheck::ExtractMinifiedJson;
using lsxcommon::selfcheck::StripBeforeFinalAnswer;

// The protocol shape a candidate payload must have before it is even considered
// the answer. It is what keeps a reasoning model's illustrative REQUEST from
// being read as the ERROR RESPONSE the prompt asked for.
bool ShapeMatches(McpKind kind, std::string_view candidate)
{
    const bool has_version = candidate.find("\"jsonrpc\"") != std::string_view::npos;
    const bool has_error = candidate.find("\"error\"") != std::string_view::npos;
    const bool has_method = candidate.find("\"method\"") != std::string_view::npos;
    switch (kind) {
        case McpKind::kToolCall:
        case McpKind::kResourceRead:
            return has_version && has_method && !has_error;
        case McpKind::kErrorResponse:
            return has_version && has_error;
    }
    return false;
}

// ── canonical JSON ────────────────────────────────────────────────────────

// The tree has ONE JSON string escaper (lsxnetwork::json::JsonEscape, an RFC
// 8259 §7 implementation with its own hex table and its own guard). The
// canonical form uses it verbatim, so the canonicalization cannot disagree with
// the serializer that produced the model's payload.
std::string AppendEscaped(std::string_view s)
{
    // JsonEscape escapes the CONTENT; the delimiters stay here, because the
    // canonical form needs them and the escaper's contract is body-only.
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    out += lsxnetwork::json::JsonEscape(s);
    out.push_back('"');
    return out;
}

std::string CanonicalScalar(simdjson::dom::element e)
{
    using simdjson::dom::element_type;
    const element_type t = e.type();
    if (t == element_type::DOUBLE) {
        double d = 0.0;
        if (e.get_double().get(d) == simdjson::SUCCESS)
            return absl::StrFormat("%.17g", d);
        return "<unreadable number>";
    }
    if (t == element_type::UINT64) {
        uint64_t u = 0;
        if (e.get_uint64().get(u) == simdjson::SUCCESS)
            return absl::StrFormat("%llu",
                                    static_cast<unsigned long long>(u));
        return "<unreadable number>";
    }
    int64_t i = 0;
    if (e.get_int64().get(i) == simdjson::SUCCESS)
        return absl::StrFormat("%lld", static_cast<long long>(i));
    return "<unreadable number>";
}

const char *TypeName(simdjson::dom::element_type t)
{
    using simdjson::dom::element_type;
    switch (t) {
        case element_type::OBJECT: return "object";
        case element_type::ARRAY: return "array";
        case element_type::STRING: return "string";
        case element_type::INT64:
        case element_type::UINT64:
        case element_type::DOUBLE: return "number";
        case element_type::BOOL: return "boolean";
        case element_type::NULL_VALUE: return "null";
    }
    return "unknown";
}

bool WriteCanonical(simdjson::dom::element e, std::string *out, std::string *error)
{
    using simdjson::dom::element_type;
    const element_type t = e.type();
    switch (t) {
        case element_type::OBJECT: {
            simdjson::dom::object obj;
            if (e.get_object().get(obj) != simdjson::SUCCESS) {
                *error = "unreadable object";
                return false;
            }
            // Key order is insignificant in JSON, so the canonical form FIXES
            // it: two documents that differ only in order are equal here.
            std::vector<std::pair<std::string_view, simdjson::dom::element>> kv;
            for (auto field : obj) kv.emplace_back(field.key, field.value);
            std::sort(kv.begin(), kv.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            out->push_back('{');
            for (std::size_t i = 0; i < kv.size(); ++i) {
                if (i) out->push_back(',');
                *out += AppendEscaped(kv[i].first);
                out->push_back(':');
                if (!WriteCanonical(kv[i].second, out, error)) return false;
            }
            out->push_back('}');
            return true;
        }
        case element_type::ARRAY: {
            simdjson::dom::array arr;
            if (e.get_array().get(arr) != simdjson::SUCCESS) {
                *error = "unreadable array";
                return false;
            }
            out->push_back('[');
            bool first = true;
            for (auto v : arr) {
                if (!first) out->push_back(',');
                first = false;
                if (!WriteCanonical(v, out, error)) return false;
            }
            out->push_back(']');
            return true;
        }
        case element_type::STRING: {
            std::string_view s;
            if (e.get_string().get(s) != simdjson::SUCCESS) {
                *error = "unreadable string";
                return false;
            }
            *out += AppendEscaped(s);
            return true;
        }
        case element_type::BOOL: {
            bool b = false;
            if (e.get_bool().get(b) != simdjson::SUCCESS) {
                *error = "unreadable boolean";
                return false;
            }
            *out += b ? "true" : "false";
            return true;
        }
        case element_type::NULL_VALUE:
            *out += "null";
            return true;
        case element_type::INT64:
        case element_type::UINT64:
        case element_type::DOUBLE:
            *out += CanonicalScalar(e);
            return true;
    }
    *error = "unknown JSON element";
    return false;
}

bool ParseDocument(const simdjson::padded_string &json,
                   simdjson::dom::parser *parser, simdjson::dom::element *doc,
                   std::string *error)
{
    // The parsed element points INTO `json`, so every caller keeps its padded
    // buffer alive for as long as it reads the element.
    const simdjson::error_code rc = parser->parse(json).get(*doc);
    if (rc != simdjson::SUCCESS) {
        if (error) *error = std::string("invalid JSON: ") + simdjson::error_message(rc);
        return false;
    }
    return true;
}

// ── document diff ─────────────────────────────────────────────────────────

std::string JoinPath(const std::string &parent, const std::string &key)
{
    return parent.empty() ? key : parent + "." + key;
}

bool DiffInto(simdjson::dom::element a, simdjson::dom::element b,
              const std::string &path, std::string *why)
{
    using simdjson::dom::element_type;
    const element_type ta = a.type();
    const element_type tb = b.type();
    if (ta != tb) {
        *why = absl::StrFormat("%s: %s vs %s", path.c_str(), TypeName(ta),
                               TypeName(tb));
        return false;
    }
    if (ta == element_type::OBJECT) {
        simdjson::dom::object oa;
        simdjson::dom::object ob;
        if (a.get_object().get(oa) != simdjson::SUCCESS ||
            b.get_object().get(ob) != simdjson::SUCCESS) {
            *why = path + ": unreadable object";
            return false;
        }
        std::map<std::string, simdjson::dom::element> mb;
        for (auto f : ob) mb[std::string(f.key)] = f.value;
        std::map<std::string, simdjson::dom::element> ma;
        for (auto f : oa) ma[std::string(f.key)] = f.value;
        for (const auto &[k, va] : ma) {
            const auto it = mb.find(k);
            if (it == mb.end()) {
                *why = absl::StrFormat("%s: missing key '%s'", path.c_str(), k.c_str());
                return false;
            }
            if (!DiffInto(va, it->second, JoinPath(path, k), why)) return false;
        }
        for (const auto &[k, vb] : mb) {
            (void)vb;
            if (!ma.count(k)) {
                *why = absl::StrFormat("%s: unexpected key '%s'", path.c_str(),
                                       k.c_str());
                return false;
            }
        }
        return true;
    }
    if (ta == element_type::ARRAY) {
        simdjson::dom::array aa;
        simdjson::dom::array ab;
        if (a.get_array().get(aa) != simdjson::SUCCESS ||
            b.get_array().get(ab) != simdjson::SUCCESS) {
            *why = path + ": unreadable array";
            return false;
        }
        const std::size_t na = aa.size();
        const std::size_t nb = ab.size();
        if (na != nb) {
            *why = absl::StrFormat("%s: %u element(s) vs %u", path.c_str(),
                                   static_cast<unsigned>(na),
                                   static_cast<unsigned>(nb));
            return false;
        }
        auto ia = aa.begin();
        auto ib = ab.begin();
        for (std::size_t i = 0; i < na; ++i, ++ia, ++ib) {
            if (!DiffInto(*ia, *ib,
                          absl::StrFormat("%s[%u]", path.c_str(),
                                          static_cast<unsigned>(i)),
                          why))
                return false;
        }
        return true;
    }
    // Scalars: compare their canonical spelling, so 1 and 1.0 are the same
    // JSON number and `"a"` vs `"b"` prints both sides.
    std::string error;
    std::string ca;
    std::string cb;
    if (!WriteCanonical(a, &ca, &error) || !WriteCanonical(b, &cb, &error)) {
        *why = path + ": " + error;
        return false;
    }
    if (ca != cb) {
        *why = absl::StrFormat("%s: %s != %s", path.c_str(), ca.c_str(), cb.c_str());
        return false;
    }
    return true;
}

} // namespace

const char *McpKindName(McpKind kind)
{
    switch (kind) {
        case McpKind::kToolCall: return "tool_call";
        case McpKind::kResourceRead: return "resource_read";
        case McpKind::kErrorResponse: return "error_response";
    }
    return "unknown";
}

std::string ExtractMcpAnswer(McpKind kind, std::string_view anchor,
                             std::string_view raw)
{
    const std::string text = StripBeforeFinalAnswer(raw);
    return ExtractMinifiedJson(
        text, anchor, [kind](std::string_view candidate) {
            return ShapeMatches(kind, candidate);
        });
}

bool CanonicalizeJson(std::string_view json, std::string *canonical,
                      std::string *error)
{
    if (canonical) canonical->clear();
    const simdjson::padded_string pad(json.data(), json.size());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (!ParseDocument(pad, &parser, &doc, error)) return false;
    std::string out;
    std::string local;
    if (!WriteCanonical(doc, &out, &local)) {
        if (error) *error = local;
        return false;
    }
    if (canonical) *canonical = std::move(out);
    return true;
}

std::string JsonDocumentDiff(std::string_view expected, std::string_view got)
{
    const simdjson::padded_string pad_a(expected.data(), expected.size());
    const simdjson::padded_string pad_b(got.data(), got.size());
    simdjson::dom::parser pa;
    simdjson::dom::parser pb;
    simdjson::dom::element a;
    simdjson::dom::element b;
    std::string error;
    if (!ParseDocument(pad_a, &pa, &a, &error)) return "expected: " + error;
    if (!ParseDocument(pad_b, &pb, &b, &error)) return "extracted: " + error;
    std::string why;
    if (DiffInto(a, b, "", &why)) return {};
    return why;
}

bool ParseMcpGolden(std::string_view json, McpGolden &out, std::string *error)
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
        tag != "logestix-mcp-golden") {
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
        McpGoldenCase c;
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
        if (sv == "tool_call") c.kind = McpKind::kToolCall;
        else if (sv == "resource_read") c.kind = McpKind::kResourceRead;
        else if (sv == "error_response") c.kind = McpKind::kErrorResponse;
        else { if (error) *error = c.id + ": unknown kind"; return false; }
        if (o["anchor"].get_string().get(sv) != simdjson::SUCCESS || sv.empty()) {
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
        if (o["answer_start_token_id"].get_int64().get(ans) == simdjson::SUCCESS)
            c.answer_start_token_id = static_cast<int32_t>(ans);
        if (o["expected"].get_string().get(sv) != simdjson::SUCCESS || sv.empty()) {
            if (error) *error = c.id + ": missing expected";
            return false;
        }
        // A golden that is not itself a protocol message would compare the
        // model against something JSON-RPC never defines, so it is a parse
        // error rather than a silently unpassable case.
        const std::string expected(sv);
        std::string canonical;
        std::string local;
        if (!CanonicalizeJson(expected, &canonical, &local)) {
            if (error) *error = c.id + ": expected " + local;
            return false;
        }
        const bool ok_version = canonical.find("\"jsonrpc\":\"2.0\"") !=
                                std::string::npos;
        const bool ok_id = canonical.find("\"id\":") != std::string::npos;
        const bool has_method = canonical.find("\"method\":") != std::string::npos;
        const bool has_error = canonical.find("\"error\":") != std::string::npos;
        if (!ok_version || !ok_id || has_method == has_error) {
            if (error) *error = c.id + ": expected is not a JSON-RPC 2.0 "
                                        "message (needs jsonrpc=\"2.0\", an id "
                                        "and exactly one of method/error)";
            return false;
        }
        c.expected = expected;
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

bool LoadMcpGoldenFile(const std::string &path, McpGolden &out,
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
    if (!ParseMcpGolden(text, out, &local)) {
        if (error) *error = path + ": " + local;
        return false;
    }
    return true;
}

namespace {

std::vector<int32_t> TailAfter(const std::vector<int32_t> &ids, int32_t marker)
{
    if (marker < 0) return ids;
    for (std::size_t i = ids.size(); i > 0; --i)
        if (ids[i - 1] == marker)
            return std::vector<int32_t>(ids.begin() + static_cast<long>(i),
                                        ids.end());
    return ids;
}

McpCaseResult RunMcpCase(IModel &model, const McpGoldenCase &c,
                         const McpDecodeFn &decode)
{
    McpCaseResult r;
    r.id = c.id;
    std::string canonical_expected;
    std::string local;
    if (!CanonicalizeJson(c.expected, &canonical_expected, &local)) {
        r.error = "golden expected: " + local;
        return r;
    }
    r.expected = canonical_expected;

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
    const std::string channel = atem::SelectAnswerChannel(
        text, c.answer_channel, c.reasoning_channel);
    const std::string extracted =
        ExtractMcpAnswer(c.kind, c.anchor, CutPromptEcho(channel, c.body));
    if (extracted.empty()) {
        r.diff = std::string("no JSON-RPC ") + McpKindName(c.kind) +
                 " payload carrying '" + c.anchor + "'";
        return r;
    }
    r.exact = (extracted == c.expected);
    std::string canonical_got;
    if (!CanonicalizeJson(extracted, &canonical_got, &local)) {
        r.diff = "extracted JSON is unreadable: " + local;
        return r;
    }
    r.extracted = canonical_got;
    r.diff = JsonDocumentDiff(canonical_expected, canonical_got);
    r.passed = r.diff.empty();
    return r;
}

} // namespace

McpResult RunMcpSelfCheck(IModel &model, const McpGolden &golden,
                          const McpDecodeFn &decode)
{
    McpResult res;
    // Progress line per case (see RunFsiSelfCheck): the verdict lines arrive
    // only after the last case, so a slow model needs a heartbeat.
    for (std::size_t i = 0; i < golden.cases.size(); ++i) {
        const McpGoldenCase &c = golden.cases[i];
        log::info(absl::StrFormat(
            "[mcp] %s start (%u/%u) kind=%s max_new_tokens=%d ignore_eos=%s",
            c.id.c_str(), static_cast<unsigned>(i + 1),
            static_cast<unsigned>(golden.cases.size()), McpKindName(c.kind),
            c.max_new_tokens, c.ignore_eos ? "true" : "false"));
        res.cases.push_back(RunMcpCase(model, c, decode));
        const McpCaseResult &last = res.cases.back();
        if (!last.error.empty())
            log::info(absl::StrFormat("[mcp] %s done inference-error error='%s'",
                                      c.id.c_str(), last.error.c_str()));
    }
    res.all_passed = !res.cases.empty();
    for (const McpCaseResult &c : res.cases) {
        log::info(absl::StrFormat(
            "[mcp] %s exact=%s passed=%s diff='%s' expected='%s' "
            "extracted='%s' raw='%s'",
            c.id.c_str(), c.exact ? "true" : "false",
            c.passed ? "true" : "false", EscapeForLog(c.diff).c_str(),
            c.expected.c_str(), c.extracted.c_str(),
            EscapeForLog(c.raw).c_str()));
        if (!c.passed) res.all_passed = false;
    }
    return res;
}

McpResult RunMcpSelfCheck(IModel &model, const McpGolden &golden)
{
    return RunMcpSelfCheck(
        model, golden,
        [&model](const std::vector<int32_t> &ids) {
            return runtime::DecodeGeneratedText(model.Tokenizer(), ids);
        });
}

} // namespace mcp
} // namespace lsxcommon
