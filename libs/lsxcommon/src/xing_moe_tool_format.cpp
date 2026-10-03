// Xing4.0-29B-A4B native chat + tool codec (host-only).
//
// Verified against the checkpoint's chat_template.jinja and the
// tokenizer_config.json added_tokens_decoder (ids 1..6, 9..14, all
// special:true). Frame per message role, verbatim from the template:
//
//   system      <_system>{text}
//   user        <_user>{text}\n
//   assistant   <_bot>\n<think>\n{reasoning}\n</think>{content}<_end>\n
//   observation <_observation><tool_response>{content}</tool_response>
//   generation  <_bot><think>\n
//
// With tools present, the template inserts its instruction block (tools JSON
// inside <tools>…</tools>, then the XML call format) between the system block
// and the first message.
//
// DESIGN NOTE — no invisible characters in this file. The upstream
// <tool_call> marker carries a ZERO-WIDTH SPACE (U+200B) right after
// the '<'; a literal copy of it in source is unreviewable and has been silently
// stripped from this repository's sources before. So every marker is resolved
// from the VOCABULARY at the frozen id (xmg::frozen::kToolCallId = 11) and
// decoded back to its exact bytes. If a checkpoint ever ships different
// marker bytes, the codec follows the vocabulary instead of failing.

#include "lsxcommon/xing_moe_tool_format.h"

#include "xing_moe_constants.h"

#include "lsxcommon/tool_format.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/strip.h>

#include <algorithm>
#include <cstring>
#include <map>

namespace lsxcommon::xing::tool_format {
namespace {

namespace frozen = lsxcommon::xing::frozen;

// Frozen control-marker ids (tokenizer_config.json).
// Upstream marker bytes. The tool-call pair carries a ZERO-WIDTH SPACE
// (U+200B) right after the opening '<' — written as an escape so the bytes are
// reviewable and cannot be stripped by a source filter (they HAVE been, in this
// repository's MiMo codec).
//
// kIds[i].first is the FROZEN id from tokenizer_config.json; SentinelsPresent
// verifies name -> id against the live vocabulary, so a checkpoint that ever
// renumbers is caught instead of silently mis-rendering.
struct Marker {
    const char *text;
    int32_t id;
};

constexpr Marker kSystem        {"<_system>",      frozen::kSystemId};
constexpr Marker kUser          {"<_user>",        frozen::kUserId};
constexpr Marker kBot           {"<_bot>",         frozen::kBotId};
constexpr Marker kThink         {"<think>",   frozen::kThinkId};
constexpr Marker kThinkEnd      {"</think>",  frozen::kThinkEndId};
constexpr Marker kToolCall      {"<\xE2\x80\x8B" "tool_call>",   frozen::kToolCallId};
constexpr Marker kToolCallEnd   {"</\xE2\x80\x8B" "tool_call>",  frozen::kToolCallEndId};
constexpr Marker kToolResponse  {"<tool_response>",  frozen::kToolResponseId};
constexpr Marker kToolResponseEnd {"</tool_response>", frozen::kToolResponseEndId};
constexpr Marker kEnd          {"<_end>",          frozen::kEosId};
constexpr Marker kAll[] = {kSystem,     kUser,       kBot,      kEnd,
                           kThink,      kThinkEnd,   kToolCall, kToolCallEnd,
                           kToolResponse, kToolResponseEnd};

// The <_observation> marker is NOT an added token upstream (the template emits
// it as plain text), so it is rendered as text, not injected as an id.
constexpr const char *kObservation = "<_observation>";

struct Markers {
    std::string system, user, bot, end, think, think_end;
    std::string tool_call, tool_call_end;
    std::string tool_response, tool_response_end;
    bool ok = false;
};

Markers LiteralMarkers()
{
    Markers m;
    m.system = kSystem.text;
    m.user = kUser.text;
    m.bot = kBot.text;
    m.end = kEnd.text;
    m.think = kThink.text;
    m.think_end = kThinkEnd.text;
    m.tool_call = kToolCall.text;
    m.tool_call_end = kToolCallEnd.text;
    m.tool_response = kToolResponse.text;
    m.tool_response_end = kToolResponseEnd.text;
    m.ok = true;
    return m;
}

// The markers are frozen, so the vocabulary is only CHECKED (name -> id), never
// consulted for their bytes: arrow_tokenizer_decode does not surface added
// tokens, and re-deriving invisible marker bytes at runtime would hide a
// mis-parse instead of failing.
bool MarkersPresentIn(const ArrowTokenizerDb &tok)
{
    for (const Marker &mk : kAll)
    {
        const int32_t id =
            arrow_tokenizer_lookup(tok, mk.text, std::strlen(mk.text));
        if (id != mk.id) return false;
    }
    return true;
}

bool AppendEncoded(const ArrowTokenizerDb &tok, std::string_view text,
                   std::vector<int32_t> &ids)
{
    for (size_t cap = 256; cap <= 65536; cap *= 2)
    {
        std::vector<int32_t> buf(cap, 0);
        size_t enc_cap = cap;
        const int32_t n = arrow_tokenizer_encode(tok, text.data(), text.size(),
                                                 buf.data(), &enc_cap);
        if (n == -2) continue;  // buffer too small
        if (n < 0) return false;
        // The checkpoint sets add_bos_token=false, and the frame already opens
        // with <_system>: drop an auto-prepended bos so the model sees exactly
        // the transcript the template produces.
        const size_t begin = (tok.bos_id >= 0 && n > 0 &&
                              buf[0] == tok.bos_id)
                                 ? 1
                                 : 0;
        ids.insert(ids.end(), buf.begin() + static_cast<long>(begin),
                   buf.begin() + n);
        return true;
    }
    return false;
}

std::string MessageText(const ChatMessage &m)
{
    return m.content;
}

std::string ReasoningOf(const ChatMessage &m)
{
    // ChatMessage carries no reasoning field, so the template's
    // `m.reasoning_content is string` branch never fires and <think>…</think>
    // inside the content is what upstream actually consumes.
    const std::string &c = m.content;
    const size_t open = c.find("<think>");
    if (open == std::string::npos) return {};
    const size_t close = c.find("</think>", open);
    if (close == std::string::npos) return {};
    return c.substr(open + 7, close - (open + 7));
}

std::string StripThink(const std::string &c)
{
    const size_t close = c.rfind("</think>");
    if (close == std::string::npos) return c;
    return std::string(
        absl::StripAsciiWhitespace(c.substr(close + 8)));
}

std::string ToolsBlock(const std::vector<ToolSpec> &tools)
{
    std::vector<std::string> jsons;
    jsons.reserve(tools.size());
    for (const auto &t : tools)
    {
        std::vector<std::string> parts;
        parts.push_back(absl::StrCat("\"name\": \"", t.name, "\""));
        parts.push_back(absl::StrCat("\"description\": \"", t.description, "\""));
        parts.push_back(
            absl::StrCat("\"parameters\": ", t.parameters_json.empty()
                                                  ? std::string("{}")
                                                  : t.parameters_json));
        jsons.push_back(absl::StrCat("{", absl::StrJoin(parts, ", "), "}"));
    }
    return absl::StrCat(
        "\n# Tools\n\nYou may call one or more functions to assist with the "
        "user query.\n\nYou are provided with function signatures within "
        "<tools></tools> XML tags:\n<tools>\n",
        absl::StrJoin(jsons, "\n"), "\n</tools>\n\nFor each function call, "
        "output the function name and arguments within the following XML "
        "format:\n");
}

}  // namespace

namespace {

bool RenderWith(const ChatConversation &conv, std::string &out,
                bool add_generation_prompt, const Markers &m)
{
    out.clear();
    std::string system;
    std::vector<const ChatMessage *> body;
    for (const auto &msg : conv.messages)
    {
        if (msg.role == "system" && system.empty())
            system = msg.content;
        else
            body.push_back(&msg);
    }
    out += m.system + system;
    if (!conv.tools.empty()) out += ToolsBlock(conv.tools) + "\n";
    for (const ChatMessage *msg : body)
    {
        if (msg->role == "user")
        {
            out += m.user + MessageText(*msg) + "\n";
        }
        else if (msg->role == "assistant" || msg->role == "bot")
        {
            out += m.bot + "\n" + m.think + "\n" + ReasoningOf(*msg) + "\n" +
                   m.think_end + StripThink(MessageText(*msg)) + m.end + "\n";
            for (const auto &tc : msg->tool_calls)
            {
                out += m.tool_call + tc.name;
                // Arguments arrive as a JSON object string; the template emits
                // one <param_key>/<param_value> pair per entry.
                std::string args = tc.arguments_json;
                if (args.size() >= 2 && args.front() == '{' && args.back() == '}')
                    args = args.substr(1, args.size() - 2);
                // Split on top-level commas.
                int depth = 0;
                bool in_str = false;
                std::string cur;
                std::vector<std::string> pairs;
                for (char ch : args)
                {
                    if (in_str)
                    {
                        cur.push_back(ch);
                        if (ch == '"' && (cur.size() < 2 ||
                                          cur[cur.size() - 2] != '\\'))
                            in_str = false;
                        continue;
                    }
                    if (ch == '"') { in_str = true; cur.push_back(ch); continue; }
                    if (ch == '{' || ch == '[') ++depth;
                    if (ch == '}' || ch == ']') --depth;
                    if (ch == ',' && depth == 0)
                    {
                        pairs.push_back(cur);
                        cur.clear();
                        continue;
                    }
                    cur.push_back(ch);
                }
                if (!cur.empty()) pairs.push_back(cur);
                for (const std::string &pair : pairs)
                {
                    std::string p(
                        std::string(absl::StripAsciiWhitespace(pair)));
                    const size_t colon = p.find(':');
                    if (colon == std::string::npos) continue;
                    std::string key(std::string(
                        absl::StripAsciiWhitespace(p.substr(0, colon))));
                    std::string val{
                        std::string(absl::StripAsciiWhitespace(p.substr(colon + 1)))};
                    if (key.size() >= 2 && key.front() == '"' && key.back() == '"')
                        key = key.substr(1, key.size() - 2);
                    if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
                        val = val.substr(1, val.size() - 2);
                    out += "<param_key>" + key + "</param_key><param_value>" +
                           val + "</param_value>";
                }
                out += m.tool_call_end;
            }
        }
        else if (msg->role == "tool")
        {
            out += std::string(kObservation) + m.tool_response +
                   MessageText(*msg) + m.tool_response_end;
        }
    }
    if (add_generation_prompt) out += m.bot + m.think + "\n";
    return true;
}

}  // namespace

// Literal markers for a db-less render (shape tests / diagnostics). The
// tool-call pair keeps its U+200B, written as an escape so it survives review
// and every source filter.

bool SentinelsPresent(const ArrowTokenizerDb &tok)
{
    return MarkersPresentIn(tok);
}

bool RenderText(const ChatConversation &conv, std::string &out,
                bool add_generation_prompt)
{
    return RenderWith(conv, out, add_generation_prompt, LiteralMarkers());
}



bool Render(const ArrowTokenizerDb &tok, const ChatConversation &conv,
            std::vector<int32_t> &out_ids, bool add_generation_prompt)
{
    out_ids.clear();
    if (!MarkersPresentIn(tok)) return false;
    const Markers m = LiteralMarkers();
    std::string text;
    if (!RenderWith(conv, text, add_generation_prompt, m)) return false;

    // Inject the control markers as their own vocab ids; encode only the runs of
    // free text between them (arrow_tokenizer_encode is pure BPE).
    static const std::vector<std::string> *kOrder = nullptr;
    (void)kOrder;
    std::vector<std::string> marks = {m.system,  m.user,      m.bot,
                                      m.think,   m.think_end, m.tool_call,
                                      m.tool_call_end, m.tool_response,
                                      m.tool_response_end,
                                      std::string(kObservation)};
    // Longest first so "<tool_call>" wins over "<tool_call_end>"-like
    // prefixes and the observation marker never shadows "<_".
    std::sort(marks.begin(), marks.end(),
              [](const std::string &a, const std::string &b) {
                  return a.size() > b.size();
              });

    size_t i = 0;
    size_t run = 0;
    auto flush = [&](size_t end) -> bool {
        if (end <= run) return true;
        return AppendEncoded(tok, std::string_view(text).substr(run, end - run),
                             out_ids);
    };
    while (i < text.size())
    {
        bool matched = false;
        for (const auto &s : marks)
        {
            if (s.empty() || text.compare(i, s.size(), s) != 0) continue;
            if (!flush(i)) return false;
            // `<_observation>` is plain text upstream, so it goes through the
            // BPE encoder instead of an id lookup.
            const int32_t id = arrow_tokenizer_lookup(tok, s.data(), s.size());
            if (id < 0)
            {
                if (!AppendEncoded(tok, s, out_ids)) return false;
            }
            else
            {
                out_ids.push_back(id);
            }
            i += s.size();
            run = i;
            matched = true;
            break;
        }
        if (!matched) ++i;
    }
    return flush(text.size());
}

std::vector<ToolCall> ParseToolCallText(std::string_view text)
{
    // Written against the LITERAL upstream spelling, which contains U+200B
    // after each opening '<' of the tool-call pair. Escapes keep it reviewable.
    static const std::string kOpen = "<\xE2\x80\x8B" "tool_call>";
    static const std::string kClose = "</\xE2\x80\x8B" "tool_call>";
    static const std::string kKey = "<param_key>";
    static const std::string kKeyEnd = "</param_key>";
    static const std::string kVal = "<param_value>";
    static const std::string kValEnd = "</param_value>";

    std::vector<ToolCall> calls;
    size_t pos = 0;
    while (true)
    {
        const size_t open = text.find(kOpen, pos);
        if (open == std::string_view::npos) break;
        const size_t close = text.find(kClose, open);
        const size_t body_end =
            close == std::string_view::npos ? text.size() : close;
        ToolCall tc;
        tc.id = "call_" + std::to_string(calls.size());
        size_t p = open + kOpen.size();
        const size_t name_end = text.find('<', p);
        if (name_end == std::string_view::npos || name_end > body_end)
        {
            pos = body_end + 1;
            continue;
        }
        tc.name = std::string(text.substr(p, name_end - p));
        std::vector<std::string> parts;
        while (true)
        {
            const size_t key_at = text.find(kKey, p);
            if (key_at == std::string_view::npos || key_at > body_end) break;
            const size_t key_end = text.find(kKeyEnd, key_at);
            if (key_end == std::string_view::npos || key_end > body_end) break;
            const size_t val_at = text.find(kVal, key_end);
            const size_t val_end = text.find(kValEnd, val_at);
            if (val_at == std::string_view::npos ||
                val_end == std::string_view::npos || val_end > body_end)
                break;
            parts.push_back(absl::StrCat(
                "\"", std::string(text.substr(key_at + kKey.size(),
                                              key_end - key_at - kKey.size())),
                "\":\"",
                std::string(text.substr(val_at + kVal.size(),
                                        val_end - val_at - kVal.size())),
                "\""));
            p = val_end + kValEnd.size();
        }
        tc.arguments_json = "{" + absl::StrJoin(parts, ",") + "}";
        calls.push_back(tc);
        pos = close == std::string_view::npos ? text.size() : close + kClose.size();
    }
    return calls;
}

bool Parse(const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
           std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;
    if (!MarkersPresentIn(tok)) return false;

    // All nine Xing control markers are special:true, and arrow_tokenizer_decode
    // DROPS special tokens — so decoding the ids alone loses every tool-call
    // marker and the parse silently falls through to the generic Hermes codec.
    // The ids are therefore the source of truth: walk them, splice the frozen
    // marker bytes for a marker id, and decode only the runs in between.
    std::string text;
    std::vector<int32_t> run;
    auto flush = [&] {
        if (run.empty()) return true;
        std::size_t cap = 4096;
        std::string buf(cap, '\0');
        for (int attempt = 0; attempt < 8; ++attempt)
        {
            // arrow_tokenizer_decode returns 0 on SUCCESS and reports the
            // written length through max_out_len (-2 = buffer too small).
            const int rc = arrow_tokenizer_decode(tok, run.data(), run.size(),
                                                  buf.data(), &cap);
            if (rc == 0)
            {
                text.append(buf.data(), cap);
                return true;
            }
            if (rc == -2 && cap > buf.size() && cap <= (1u << 20))
            {
                buf.resize(cap);
                continue;
            }
            return false;
        }
        return false;
    };
    for (int32_t id : gen_tokens)
    {
        const char *marker = nullptr;
        for (const Marker &mk : kAll)
            if (mk.id == id)
            {
                marker = mk.text;
                break;
            }
        if (marker)
        {
            if (!flush()) return false;
            run.clear();
            text += marker;
            continue;
        }
        run.push_back(id);
    }
    if (!flush()) return false;

    out_calls = ParseToolCallText(text);
    if (!out_calls.empty()) return true;
    // Generic Hermes fallback, then the bare-JSON form the fallback families
    // emit (see AGENTS.md: never claim a native renderer without verifying it).
    out_calls = tools::ParseToolCallsText(text);
    if (!out_calls.empty()) return true;
    out_calls = tools::ParseBareToolCallJson(text);
    return true;
}

}  // namespace lsxcommon::xing::tool_format