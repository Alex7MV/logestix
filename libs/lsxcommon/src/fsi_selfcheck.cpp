#include "lsxcommon/fsi_selfcheck.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_format.h>
#include <simdjson.h>
#include <cctype>
#include <fstream>
#include <memory>
#include <sstream>

#include "lsxcommon/atem_channel.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"

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

std::string SelectAnswerLine(const std::string &s, std::string_view anchor,
                             FsiKind kind, bool prefer_last = false)
{
    std::vector<std::string> lines;
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const std::size_t nl = s.find('\n', pos);
        lines.push_back((nl == std::string::npos) ? s.substr(pos)
                                                  : s.substr(pos, nl - pos));
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    // A line whose anchor is followed by '=' (arithmetic) or a comparison op
    // (comparison) is a candidate answer. `prefer_last` scans the candidates
    // from the end, so a model that reasons inline cannot shadow its own final
    // answer with a line from the chain.
    for (int pass = 0; pass < 2; ++pass) {
        const bool forward = (pass == 0) ? !prefer_last : prefer_last;
        for (std::size_t k = 0; k < lines.size(); ++k) {
            const std::string &line =
                lines[forward ? k : lines.size() - 1 - k];
            const std::size_t a = line.find(anchor);
            if (a == std::string::npos) continue;
            std::size_t i = a + anchor.size();
            while (i < line.size() && std::isspace((unsigned char)line[i]))
                ++i;
            if (i >= line.size()) continue;
            if (kind == FsiKind::kArithmetic && line[i] == '=') return line;
            if (kind == FsiKind::kComparison && IsComparisonOp(line[i]))
                return line;
        }
    }
    for (std::size_t k = 0; k < lines.size(); ++k) {
        const std::string &line = lines[prefer_last ? lines.size() - 1 - k : k];
        if (line.find(anchor) != std::string::npos) return line;
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

struct ExprNode {
    enum Kind { kLeaf, kCall, kUnary, kBin } kind = kLeaf;
    std::string text;
    std::unique_ptr<ExprNode> left;
    std::unique_ptr<ExprNode> right;
};

int BinPrec(char op) { return (op == '+' || op == '-') ? 1 : 2; }

class ExprParser {
public:
    explicit ExprParser(std::string_view s) : s_(s) {}

    std::unique_ptr<ExprNode> Parse()
    {
        pos_ = 0;
        auto e = ParseExpr(0);
        if (!e || pos_ != s_.size()) return nullptr;
        return e;
    }

private:
    std::unique_ptr<ExprNode> ParseExpr(int min_prec)
    {
        auto left = ParseUnary();
        if (!left) return nullptr;
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c != '+' && c != '-' && c != '*' && c != '/') break;
            const int prec = BinPrec(c);
            if (prec < min_prec) break;
            ++pos_;
            auto right = ParseExpr(prec + 1);
            if (!right) return nullptr;
            auto n = std::make_unique<ExprNode>();
            n->kind = ExprNode::kBin;
            n->text = std::string(1, c);
            n->left = std::move(left);
            n->right = std::move(right);
            left = std::move(n);
        }
        return left;
    }

    std::unique_ptr<ExprNode> ParseUnary()
    {
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) {
            const char c = s_[pos_++];
            auto v = ParseUnary();
            if (!v) return nullptr;
            if (c == '+') return v;
            auto n = std::make_unique<ExprNode>();
            n->kind = ExprNode::kUnary;
            n->text = "-";
            n->left = std::move(v);
            return n;
        }
        return ParsePrimary();
    }

    std::unique_ptr<ExprNode> ParsePrimary()
    {
        if (pos_ >= s_.size()) return nullptr;
        const char c = s_[pos_];
        if (c == '(') {
            ++pos_;
            auto e = ParseExpr(0);
            if (!e || pos_ >= s_.size() || s_[pos_] != ')') return nullptr;
            ++pos_;
            return e;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            const std::size_t b = pos_;
            while (pos_ < s_.size() &&
                   (std::isalnum(static_cast<unsigned char>(s_[pos_])) ||
                    s_[pos_] == '_'))
                ++pos_;
            const std::string name(s_.substr(b, pos_ - b));
            if (pos_ < s_.size() && s_[pos_] == '(') {
                ++pos_;
                auto arg = ParseExpr(0);
                if (!arg || pos_ >= s_.size() || s_[pos_] != ')')
                    return nullptr;
                ++pos_;
                auto n = std::make_unique<ExprNode>();
                n->kind = ExprNode::kCall;
                n->text = name;
                n->left = std::move(arg);
                return n;
            }
            auto n = std::make_unique<ExprNode>();
            n->text = name;
            return n;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            const std::size_t b = pos_;
            while (pos_ < s_.size() &&
                   (std::isdigit(static_cast<unsigned char>(s_[pos_])) ||
                    s_[pos_] == '.'))
                ++pos_;
            auto n = std::make_unique<ExprNode>();
            n->text = std::string(s_.substr(b, pos_ - b));
            return n;
        }
        return nullptr;
    }

    std::string_view s_;
    std::size_t pos_ = 0;
};

std::string PrintExpr(const ExprNode *n, int min_prec)
{
    if (!n) return {};
    if (n->kind == ExprNode::kLeaf) return n->text;
    if (n->kind == ExprNode::kCall)
        return n->text + "(" + PrintExpr(n->left.get(), 0) + ")";
    if (n->kind == ExprNode::kUnary) {
        const std::string body = "-" + PrintExpr(n->left.get(), 3);
        return (min_prec > 3) ? "(" + body + ")" : body;
    }
    const int prec = BinPrec(n->text[0]);
    const std::string body = PrintExpr(n->left.get(), prec) + n->text +
                             PrintExpr(n->right.get(), prec + 1);
    return (min_prec > prec) ? "(" + body + ")" : body;
}

// Left-associate equal-precedence chains so forms that differ only by
// associativity canonicalize identically: a*(b/c) == (a*b)/c, a-(b-c) ==
// a-b+c. Each rewrite is an exact algebraic identity, so genuinely different
// expressions are never equated; commutativity is deliberately not applied.
std::unique_ptr<ExprNode> Reassociate(std::unique_ptr<ExprNode> n)
{
    if (!n || n->kind != ExprNode::kBin) return n;
    n->left = Reassociate(std::move(n->left));
    n->right = Reassociate(std::move(n->right));
    while (n->right && n->right->kind == ExprNode::kBin) {
        const char op = n->text[0];
        const char ro = n->right->text[0];
        char outer = 0;
        if ((op == '+' || op == '-') && (ro == '+' || ro == '-'))
            outer = (op == '+') ? ro : (ro == '+' ? '-' : '+');
        else if ((op == '*' || op == '/') && (ro == '*' || ro == '/'))
            outer = (op == '*') ? ro : (ro == '*' ? '/' : '*');
        if (outer == 0) break;
        std::unique_ptr<ExprNode> r = std::move(n->right);
        auto newLeft = std::make_unique<ExprNode>();
        newLeft->kind = ExprNode::kBin;
        newLeft->text = std::string(1, op);
        newLeft->left = std::move(n->left);
        newLeft->right = std::move(r->left);
        n->text = std::string(1, outer);
        n->left = Reassociate(std::move(newLeft));
        n->right = std::move(r->right);
    }
    return n;
}

std::string CanonicalizeArithmetic(std::string_view rhs)
{
    ExprParser p(rhs);
    auto root = p.Parse();
    if (!root) return std::string(rhs);
    root = Reassociate(std::move(root));
    return PrintExpr(root.get(), 0);
}

std::string CanonicalizeFsiAnswer(FsiKind kind, std::string_view anchor,
                                  std::string_view normalized)
{
    if (kind != FsiKind::kArithmetic) return std::string(normalized);
    const std::size_t eq = normalized.find('=');
    if (eq == std::string_view::npos) return std::string(normalized);
    (void)anchor;
    return std::string(normalized.substr(0, eq + 1)) +
           CanonicalizeArithmetic(normalized.substr(eq + 1));
}

} // namespace

// Where on the winning LINE the answer starts.
//
// `prefer_last` (a model that states its answer last) has to apply to the anchor
// position as well as to the line choice: a model that reasons inline emits NO
// line break between its chain and its answer, so the chain and the answer share
// one line and the FIRST `anchor=` on it belongs to the chain. Measured on
// Inkling-2-Small FSI-TEST-01 — the answer was correct and was the last thing in
// the stream, and this function's first-occurrence predecessor returned a
// paragraph of reasoning.
std::size_t AnswerAnchorPos(const std::string &line, std::string_view anchor,
                            FsiKind kind, bool prefer_last)
{
    const std::size_t first = line.find(anchor);
    if (!prefer_last) return first;
    std::size_t best = std::string::npos;
    for (std::size_t a = first; a != std::string::npos;
         a = line.find(anchor, a + anchor.size()))
    {
        std::size_t i = a + anchor.size();
        while (i < line.size() && std::isspace((unsigned char)line[i])) ++i;
        if (i >= line.size()) continue;
        if (kind == FsiKind::kArithmetic && line[i] == '=') best = a;
        else if (kind == FsiKind::kComparison && IsComparisonOp(line[i]))
            best = a;
    }
    return best == std::string::npos ? first : best;
}

std::string ExtractFsiFormula(FsiKind kind, std::string_view anchor,
                              std::string_view raw, bool prefer_last)
{
    std::string text = Unfence(raw);
    // Reasoning models emit a chain-of-thought that can contain competing
    // formulas; the final answer is the segment after the last reasoning-close
    // marker (`</think>`, or the AliceAI `[COT_END]`).
    std::size_t after = std::string::npos;
    if (const std::size_t c = text.rfind("</think>"); c != std::string::npos)
        after = c + 8;
    if (const std::size_t c = text.rfind("[COT_END]");
        c != std::string::npos && (after == std::string::npos || c + 9 > after))
        after = c + 9;
    if (after != std::string::npos) text = text.substr(after);
    const std::string line = SelectAnswerLine(text, anchor, kind, prefer_last);
    const std::size_t anchor_at =
        AnswerAnchorPos(line, anchor, kind, prefer_last);
    if (kind == FsiKind::kComparison) {
        const std::size_t a = anchor_at;
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
    const std::size_t a = anchor_at;
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
    return CanonicalizeFsiAnswer(
        kind, anchor, Normalize(std::string_view(line).substr(a, i - a)));
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
        bool ign = false;
        if (o["ignore_eos"].get_bool().get(ign) == simdjson::SUCCESS)
            c.ignore_eos = ign;
        int64_t ans = -1;
        if (o["answer_start_token_id"].get_int64().get(ans) ==
            simdjson::SUCCESS)
            c.answer_start_token_id = static_cast<int32_t>(ans);
        bool pl = false;
        if (o["prefer_last_answer"].get_bool().get(pl) == simdjson::SUCCESS)
            c.prefer_last_answer = pl;
        std::string_view ach;
        if (o["answer_channel"].get_string().get(ach) == simdjson::SUCCESS)
            c.answer_channel = std::string(ach);
        std::string_view rch;
        if (o["reasoning_channel"].get_string().get(rch) == simdjson::SUCCESS)
            c.reasoning_channel = std::string(rch);
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

namespace {

// Drops everything up to and including the LAST occurrence of the marker that
// ends the thinking block (a special id, so the decoder never sees it).
std::vector<int32_t> TailAfter(const std::vector<int32_t> &ids, int32_t marker)
{
    if (marker < 0) return ids;
    for (std::size_t i = ids.size(); i > 0; --i)
        if (ids[i - 1] == marker)
            return std::vector<int32_t>(ids.begin() + static_cast<long>(i),
                                        ids.end());
    return ids;
}

FsiCaseResult RunFsiCase(IModel &model, const FsiGoldenCase &c,
                         const FsiDecodeFn &decode)
{
    FsiCaseResult r;
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
    r.extracted = ExtractFsiFormula(c.kind, c.anchor, channel,
                                    c.prefer_last_answer);
    r.passed =
        (r.extracted == CanonicalizeFsiAnswer(c.kind, c.anchor, c.expected));
    return r;
}

} // namespace

FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden,
                          const FsiDecodeFn &decode)
{
    FsiResult res;
    // Progress line per case: a reasoning model can spend hours inside one
    // case at its long budget, and the result lines are only emitted after the
    // LAST case. Without this a slow run is indistinguishable from a hang.
    for (std::size_t i = 0; i < golden.cases.size(); ++i) {
        const FsiGoldenCase &c = golden.cases[i];
        lsxcommon::log::info(absl::StrFormat(
            "[fsi] %s start (%u/%u) max_new_tokens=%d ignore_eos=%s",
            c.id.c_str(), static_cast<unsigned>(i + 1),
            static_cast<unsigned>(golden.cases.size()), c.max_new_tokens,
            c.ignore_eos ? "true" : "false"));
        res.cases.push_back(RunFsiCase(model, c, decode));
    }
    res.all_passed = !res.cases.empty();
    for (const FsiCaseResult &c : res.cases) {
        lsxcommon::log::info(absl::StrFormat(
            "[fsi] %s expected='%s' extracted='%s' passed=%s raw='%s'",
            c.id.c_str(), c.expected.c_str(), c.extracted.c_str(),
            c.passed ? "true" : "false", c.raw.c_str()));
        if (!c.passed) res.all_passed = false;
    }
    return res;
}

FsiResult RunFsiSelfCheck(IModel &model, const FsiGolden &golden)
{
    return RunFsiSelfCheck(
        model, golden,
        [&model](const std::vector<int32_t> &ids) {
            return runtime::DecodeGeneratedText(model.Tokenizer(), ids);
        });
}

}  // namespace fsi
}  // namespace lsxcommon
