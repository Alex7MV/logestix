// Mistral-Small-4-119B native chat + tool codec (host-only, step 4b).
// Frame per message role, verbatim from the checkpoint's chat_template.jinja.
// Control markers are injected/parsed as their frozen vocab ids; free text goes
// through the BPE.

#include "lsxcommon/mistral_moe_tool_format.h"

#include "mistral_moe_constants.h"
#include "lsxcommon/log.h"
#include "lsxcommon/tool_format.h"

#include <absl/strings/str_cat.h>
#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstring>
#include <string>

namespace lsxcommon {
namespace mistral_moe {
namespace tool_format {
namespace {

namespace frozen = lsxcommon::mistral_moe::frozen;

// Ids below the 1000 added-token block are special; normal BPE tokens start at
// 1000 in this tokenizer.
constexpr int32_t kSpecialCeiling = 1000;

struct Marker {
    const char *text;
    int32_t id;
};
constexpr Marker kBos{"<s>", frozen::kBosId};
constexpr Marker kEos{"</s>", frozen::kEosId};
constexpr Marker kInst{"[INST]", frozen::kInstId};
constexpr Marker kInstEnd{"[/INST]", frozen::kInstEndId};
constexpr Marker kAvail{"[AVAILABLE_TOOLS]", frozen::kAvailableToolsId};
constexpr Marker kAvailEnd{"[/AVAILABLE_TOOLS]", frozen::kAvailableToolsEndId};
constexpr Marker kToolResults{"[TOOL_RESULTS]", frozen::kToolResultsId};
constexpr Marker kToolResultsEnd{"[/TOOL_RESULTS]", frozen::kToolResultsEndId};
constexpr Marker kToolCalls{"[TOOL_CALLS]", frozen::kToolCallsId};
constexpr Marker kSystem{"[SYSTEM_PROMPT]", frozen::kSystemPromptId};
constexpr Marker kSystemEnd{"[/SYSTEM_PROMPT]", frozen::kSystemPromptEndId};
constexpr Marker kArgs{"[ARGS]", frozen::kArgsId};
constexpr Marker kThink{"[THINK]", frozen::kThinkId};
constexpr Marker kThinkEnd{"[/THINK]", frozen::kThinkEndId};
constexpr Marker kSettings{"[MODEL_SETTINGS]", frozen::kModelSettingsId};
constexpr Marker kSettingsEnd{"[/MODEL_SETTINGS]", frozen::kModelSettingsEndId};
constexpr Marker kAll[] = {kInst,   kInstEnd,  kAvail,      kAvailEnd,
                           kToolResults, kToolResultsEnd, kToolCalls,
                           kSystem, kSystemEnd, kArgs,       kThink,
                           kThinkEnd, kSettings, kSettingsEnd};

bool MarkersPresentIn(const ArrowTokenizerDb &tok)
{
    for (const Marker &mk : kAll)
    {
        const int32_t id =
            arrow_tokenizer_lookup(tok, mk.text, std::strlen(mk.text));
        if (id != mk.id)
        {
            log::error(absl::StrFormat(
                "mistral tool_format: marker '%s' lookup=%d expected=%d",
                mk.text, id, mk.id));
            return false;
        }
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
        if (n == -2) continue;
        if (n < 0) return false;
        const size_t begin =
            (tok.bos_id >= 0 && n > 0 && buf[0] == tok.bos_id) ? 1 : 0;
        ids.insert(ids.end(), buf.begin() + static_cast<long>(begin),
                   buf.begin() + n);
        return true;
    }
    return false;
}

std::string ToolsJson(const std::vector<ToolSpec> &tools)
{
    if (tools.empty()) return "";
    // One JSON array of OpenAI tool objects (chat_template's `tools|tojson`).
    std::string one = "[";
    for (size_t i = 0; i < tools.size(); ++i)
    {
        const auto &t = tools[i];
        if (i) one += ",";
        one += absl::StrCat("{\"type\":\"function\",\"function\":{\"name\":\"",
                            t.name, "\",\"description\":\"", t.description,
                            "\",\"parameters\":", t.parameters_json, "}}");
    }
    one += "]";
    return one;
}

} // namespace

bool RenderText(const ChatConversation &conv, std::string &out,
                bool add_generation_prompt)
{
    (void)add_generation_prompt;  // the template appends nothing after the last turn
    out.clear();
    out += kBos.text;
    if (!conv.tools.empty())
    {
        out += kAvail.text;
        out += ToolsJson(conv.tools);
        out += kAvailEnd.text;
        out += kSettings.text;
        out += "{\"reasoning_effort\": \"none\"}";
        out += kSettingsEnd.text;
    }
    for (const auto &m : conv.messages)
    {
        if (m.role == "system")
        {
            out += kSystem.text;
            out += m.content;
            out += kSystemEnd.text;
        }
        else if (m.role == "user")
        {
            out += kInst.text;
            out += m.content;
            out += kInstEnd.text;
        }
        else if (m.role == "assistant")
        {
            out += m.content;
            for (const auto &tc : m.tool_calls)
            {
                out += kToolCalls.text;
                out += tc.name;
                out += kArgs.text;
                out += tc.arguments_json.empty() ? "{}" : tc.arguments_json;
            }
            out += kEos.text;
        }
        else if (m.role == "tool")
        {
            out += kToolResults.text;
            out += m.content;
            out += kToolResultsEnd.text;
        }
    }
    return true;
}

bool Render(const ArrowTokenizerDb &tok, const ChatConversation &conv,
            std::vector<int32_t> &out_ids, bool add_generation_prompt)
{
    if (!MarkersPresentIn(tok)) return false;
    out_ids.clear();
    out_ids.push_back(frozen::kBosId);
    if (!conv.tools.empty())
    {
        out_ids.push_back(frozen::kAvailableToolsId);
        if (!AppendEncoded(tok, ToolsJson(conv.tools), out_ids)) return false;
        out_ids.push_back(frozen::kAvailableToolsEndId);
        out_ids.push_back(frozen::kModelSettingsId);
        if (!AppendEncoded(tok, "{\"reasoning_effort\": \"none\"}", out_ids))
            return false;
        out_ids.push_back(frozen::kModelSettingsEndId);
    }
    for (const auto &m : conv.messages)
    {
        if (m.role == "system")
        {
            out_ids.push_back(frozen::kSystemPromptId);
            if (!AppendEncoded(tok, m.content, out_ids)) return false;
            out_ids.push_back(frozen::kSystemPromptEndId);
        }
        else if (m.role == "user")
        {
            out_ids.push_back(frozen::kInstId);
            if (!AppendEncoded(tok, m.content, out_ids)) return false;
            out_ids.push_back(frozen::kInstEndId);
        }
        else if (m.role == "assistant")
        {
            if (!m.content.empty() &&
                !AppendEncoded(tok, m.content, out_ids)) return false;
            for (const auto &tc : m.tool_calls)
            {
                out_ids.push_back(frozen::kToolCallsId);
                if (!AppendEncoded(tok, tc.name, out_ids)) return false;
                out_ids.push_back(frozen::kArgsId);
                if (!AppendEncoded(tok,
                                   tc.arguments_json.empty() ? "{}"
                                                             : tc.arguments_json,
                                   out_ids))
                    return false;
            }
            out_ids.push_back(frozen::kEosId);
        }
        else if (m.role == "tool")
        {
            out_ids.push_back(frozen::kToolResultsId);
            if (!AppendEncoded(tok, m.content, out_ids)) return false;
            out_ids.push_back(frozen::kToolResultsEndId);
        }
    }
    (void)add_generation_prompt;
    return true;
}

bool SentinelsPresent(const ArrowTokenizerDb &tok) { return MarkersPresentIn(tok); }

std::vector<ToolCall> ParseToolCallText(std::string_view text)
{
    std::vector<ToolCall> out;
    const std::string tc = kToolCalls.text;
    const std::string args = kArgs.text;
    size_t pos = 0;
    while ((pos = text.find(tc, pos)) != std::string_view::npos)
    {
        const size_t name_start = pos + tc.size();
        const size_t args_pos = text.find(args, name_start);
        if (args_pos == std::string_view::npos) break;
        ToolCall call;
        call.name = std::string(text.substr(name_start, args_pos - name_start));
        // The argument object is the JSON value following [ARGS]; match braces.
        size_t j = args_pos + args.size();
        while (j < text.size() && (text[j] == ' ' || text[j] == '\n')) ++j;
        if (j < text.size() && text[j] == '{')
        {
            int depth = 0;
            const size_t obj_start = j;
            for (; j < text.size(); ++j)
            {
                if (text[j] == '{') ++depth;
                else if (text[j] == '}') { --depth; if (depth == 0) { ++j; break; } }
            }
            call.arguments_json = std::string(text.substr(obj_start, j - obj_start));
        }
        else
        {
            call.arguments_json = "{}";
        }
        out.push_back(std::move(call));
        pos = j;
    }
    return out;
}

bool Parse(const ArrowTokenizerDb &tok, const std::vector<int32_t> &gen_tokens,
           std::vector<ToolCall> &out_calls)
{
    out_calls.clear();
    if (gen_tokens.empty()) return true;

    // Rebuild the decoded text WITH the control markers spliced back in (arrow
    // decode drops special tokens). Runs of normal (>=1000) ids are decoded as
    // one batch.
    auto marker_text = [](int32_t id) -> const char * {
        for (const Marker &mk : kAll)
            if (mk.id == id) return mk.text;
        return nullptr;
    };
    std::string text;
    std::vector<int32_t> run;
    auto flush = [&]() {
        if (run.empty()) return;
        std::vector<char> buf(4096);
        size_t cap = buf.size();
        if (arrow_tokenizer_decode(tok, run.data(), run.size(), buf.data(),
                                   &cap) == 0)
            text.append(buf.data(), cap);
        run.clear();
    };
    for (int32_t id : gen_tokens)
    {
        if (id == frozen::kEosId || id == frozen::kToolCallsId ||
            id == frozen::kArgsId)
        {
            flush();
            if (const char *m = marker_text(id)) text += m;
            else if (id == frozen::kEosId) text += kEos.text;
        }
        else if (id < kSpecialCeiling)
        {
            flush();  // other control marker: not part of a tool call
        }
        else
        {
            run.push_back(id);
        }
    }
    flush();

    out_calls = ParseToolCallText(text);
    if (out_calls.empty())
    {
        out_calls = tools::ParseToolCallsText(text);
        if (out_calls.empty()) out_calls = tools::ParseBareToolCallJson(text);
    }
    return true;
}

} // namespace tool_format
} // namespace mistral_moe
} // namespace lsxcommon
