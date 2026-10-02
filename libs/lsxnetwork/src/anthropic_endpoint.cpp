#include "lsxnetwork/anthropic_endpoint.h"
#include <lsxnetwork/json_escape.h>

#include <utility>

#include <simdjson.h>

namespace lsxnetwork {
namespace {

std::string TextBlocks(simdjson::ondemand::value content) {
    // content may be a plain string or an array of {type:"text",text:...}.
    if (content.is_string().error() == simdjson::SUCCESS &&
        content.is_string().value_unsafe()) {
        std::string_view s;
        if (content.get_string().get(s) == simdjson::SUCCESS) return std::string(s);
        return {};
    }
    std::string out;
    simdjson::ondemand::array blocks;
    if (content.get_array().get(blocks) != simdjson::SUCCESS) return out;
    for (auto b : blocks) {
        simdjson::ondemand::object bo;
        if (b.get_object().get(bo) != simdjson::SUCCESS) continue;
        std::string_view type;
        if (bo["type"].get_string().get(type) != simdjson::SUCCESS) continue;
        if (type == "text") {
            std::string_view t;
            if (bo["text"].get_string().get(t) == simdjson::SUCCESS) {
                if (!out.empty()) out += "\n";
                out.append(t);
            }
        }
    }
    return out;
}

}  // namespace

simdjson::error_code ParseAnthropicRequest(const char* data, std::size_t size,
                                           std::size_t capacity,
                                           AnthropicRequest& out) {
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    simdjson::padded_string_view v(std::string_view(data, size), capacity);
    if (auto e = parser.iterate(v).get(doc)) return e;
    simdjson::ondemand::object root;
    if (auto e = doc.get_object().get(root)) return e;

    if (auto e = root["model"].get_string(out.model)) {
        if (e != simdjson::NO_SUCH_FIELD) return e;
    }
    if (auto mt = root["max_tokens"].get_int64();
        mt.error() == simdjson::SUCCESS)
        out.max_tokens = static_cast<int>(mt.value_unsafe());
    if (auto s = root["stream"].get_bool(); s.error() == simdjson::SUCCESS)
        out.stream = s.value_unsafe();

    // system: string or block array.
    {
        simdjson::ondemand::value sys;
        if (root["system"].get(sys) == simdjson::SUCCESS)
            out.system = TextBlocks(sys);
    }

    simdjson::ondemand::array messages;
    if (auto e = root["messages"].get_array().get(messages)) return e;
    for (auto m : messages) {
        simdjson::ondemand::object mo;
        if (auto e = m.get_object().get(mo)) return e;
        std::string_view role;
        if (mo["role"].get_string().get(role) != simdjson::SUCCESS) continue;
        simdjson::ondemand::value content;
        if (mo["content"].get(content) != simdjson::SUCCESS) continue;

        if (content.is_string().error() == simdjson::SUCCESS &&
            content.is_string().value_unsafe()) {
            ChatInputMessage msg;
            msg.role.assign(role);
            msg.content = TextBlocks(content);
            out.messages.push_back(std::move(msg));
            continue;
        }

        simdjson::ondemand::array blocks;
        if (content.get_array().get(blocks) != simdjson::SUCCESS) continue;
        ChatInputMessage msg;
        msg.role.assign(role);
        for (auto b : blocks) {
            simdjson::ondemand::object bo;
            if (b.get_object().get(bo) != simdjson::SUCCESS) continue;
            std::string_view type;
            if (bo["type"].get_string().get(type) != simdjson::SUCCESS) continue;
            if (type == "text") {
                std::string_view t;
                if (bo["text"].get_string().get(t) == simdjson::SUCCESS) {
                    if (!msg.content.empty()) msg.content += "\n";
                    msg.content.append(t);
                }
            } else if (type == "tool_use") {
                ChatToolCall call;
                std::string_view sv;
                if (bo["id"].get_string().get(sv) == simdjson::SUCCESS)
                    call.id.assign(sv);
                if (bo["name"].get_string().get(sv) == simdjson::SUCCESS)
                    call.name.assign(sv);
                simdjson::ondemand::value iv;
                if (bo["input"].get(iv) == simdjson::SUCCESS) {
                    if (auto rj = iv.raw_json(); rj.error() == simdjson::SUCCESS)
                        call.arguments_json.assign(rj.value_unsafe());
                }
                if (call.arguments_json.empty()) call.arguments_json = "{}";
                msg.tool_calls.push_back(std::move(call));
            } else if (type == "tool_result") {
                ChatInputMessage tool;
                tool.role = "tool";
                std::string_view id;
                if (bo["tool_use_id"].get_string().get(id) == simdjson::SUCCESS)
                    tool.tool_call_id.assign(id);
                simdjson::ondemand::value tc;
                if (bo["content"].get(tc) == simdjson::SUCCESS)
                    tool.content = TextBlocks(tc);
                out.messages.push_back(std::move(tool));
            }
        }
        if (!msg.content.empty() || !msg.tool_calls.empty())
            out.messages.push_back(std::move(msg));
    }

    simdjson::ondemand::array tools;
    if (root["tools"].get_array().get(tools) == simdjson::SUCCESS) {
        for (auto tv : tools) {
            simdjson::ondemand::object to;
            if (tv.get_object().get(to) != simdjson::SUCCESS) continue;
            ChatToolDef def;
            std::string_view sv;
            if (to["name"].get_string().get(sv) == simdjson::SUCCESS)
                def.name.assign(sv);
            if (to["description"].get_string().get(sv) == simdjson::SUCCESS)
                def.description.assign(sv);
            simdjson::ondemand::value iv;
            if (to["input_schema"].get(iv) == simdjson::SUCCESS) {
                if (auto rj = iv.raw_json(); rj.error() == simdjson::SUCCESS)
                    def.parameters_json.assign(rj.value_unsafe());
            }
            if (def.parameters_json.empty()) def.parameters_json = "{}";
            if (!def.name.empty()) out.tools.push_back(std::move(def));
        }
    }

    {
        simdjson::ondemand::array stops;
        if (root["stop_sequences"].get_array().get(stops) == simdjson::SUCCESS) {
            for (auto s : stops) {
                std::string_view sv;
                if (s.get_string().get(sv) == simdjson::SUCCESS)
                    out.stop_sequences.emplace_back(sv);
            }
        }
    }

    {
        simdjson::ondemand::object tc;
        if (root["tool_choice"].get_object().get(tc) == simdjson::SUCCESS) {
            std::string_view type;
            if (tc["type"].get_string().get(type) == simdjson::SUCCESS) {
                if (type == "auto")
                    out.tool_choice = "auto";
                else if (type == "any")
                    out.tool_choice = "required";
                else if (type == "tool") {
                    std::string_view name;
                    if (tc["name"].get_string().get(name) == simdjson::SUCCESS)
                        out.tool_choice.assign(name);
                }
            }
        }
    }
    return simdjson::SUCCESS;
}

std::string AnthropicEvent(const std::string& event, const std::string& data) {
    return "event: " + event + "\ndata: " + data + "\n\n";
}

namespace {
std::string Usage(int in, int out) {
    return "\"usage\":{\"input_tokens\":" + std::to_string(in) +
           ",\"output_tokens\":" + std::to_string(out) + "}";
}
std::string MessageHead(const std::string& id, const std::string& model,
                        const std::string& content, const std::string& stop_reason,
                        int in, int out) {
    return "{\"id\":\"" + json::JsonEscape(id) +
           "\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"" +
           json::JsonEscape(model) + "\",\"content\":" + content +
           ",\"stop_reason\":\"" + stop_reason +
           "\",\"stop_sequence\":null," + Usage(in, out) + "}";
}
}  // namespace

std::string AnthropicMessageStart(const std::string& id, const std::string& model,
                                  int input_tokens) {
    const std::string msg = "{\"id\":\"" + json::JsonEscape(id) +
                            "\",\"type\":\"message\",\"role\":\"assistant\","
                            "\"model\":\"" +
                            json::JsonEscape(model) +
                            "\",\"content\":[],\"stop_reason\":null,"
                            "\"stop_sequence\":null," +
                            Usage(input_tokens, 0) + "}";
    return AnthropicEvent("message_start",
                          "{\"type\":\"message_start\",\"message\":" + msg + "}");
}

std::string AnthropicContentBlockStartText(int index) {
    return AnthropicEvent(
        "content_block_start",
        "{\"type\":\"content_block_start\",\"index\":" + std::to_string(index) +
            ",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}");
}

std::string AnthropicContentBlockDeltaText(int index, const std::string& text) {
    return AnthropicEvent(
        "content_block_delta",
        "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(index) +
            ",\"delta\":{\"type\":\"text_delta\",\"text\":\"" + json::JsonEscape(text) +
            "\"}}");
}

std::string AnthropicContentBlockStop(int index) {
    return AnthropicEvent(
        "content_block_stop",
        "{\"type\":\"content_block_stop\",\"index\":" + std::to_string(index) + "}");
}

std::string AnthropicToolUseBlockStart(int index, const std::string& id,
                                       const std::string& name) {
    return AnthropicEvent(
        "content_block_start",
        "{\"type\":\"content_block_start\",\"index\":" + std::to_string(index) +
            ",\"content_block\":{\"type\":\"tool_use\",\"id\":\"" + json::JsonEscape(id) +
            "\",\"name\":\"" + json::JsonEscape(name) + "\",\"input\":{}}}");
}

std::string AnthropicToolUseDelta(int index, const std::string& partial_json) {
    return AnthropicEvent(
        "content_block_delta",
        "{\"type\":\"content_block_delta\",\"index\":" + std::to_string(index) +
            ",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"" +
            json::JsonEscape(partial_json) + "\"}}");
}

std::string AnthropicMessageDelta(const std::string& stop_reason,
                                  int output_tokens) {
    return AnthropicEvent(
        "message_delta",
        "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"" + stop_reason +
            "\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":" +
            std::to_string(output_tokens) + "}}");
}

std::string AnthropicMessageStop() {
    return AnthropicEvent("message_stop", "{\"type\":\"message_stop\"}");
}

std::string BuildAnthropicMessage(const std::string& id, const std::string& model,
                                  const std::string& text,
                                  const std::string& stop_reason,
                                  int input_tokens, int output_tokens) {
    const std::string content =
        "[{\"type\":\"text\",\"text\":\"" + json::JsonEscape(text) + "\"}]";
    return MessageHead(id, model, content, stop_reason, input_tokens,
                       output_tokens);
}

std::string BuildAnthropicToolMessage(const std::string& id,
                                      const std::string& model,
                                      const std::vector<ChatToolCall>& calls,
                                      const std::string& stop_reason,
                                      int input_tokens, int output_tokens) {
    std::string content = "[";
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (i) content += ",";
        content += "{\"type\":\"tool_use\",\"id\":\"" + json::JsonEscape(calls[i].id) +
                   "\",\"name\":\"" + json::JsonEscape(calls[i].name) + "\",\"input\":" +
                   (calls[i].arguments_json.empty() ? std::string("{}")
                                                     : calls[i].arguments_json) +
                   "}";
    }
    content += "]";
    return MessageHead(id, model, content, stop_reason, input_tokens,
                       output_tokens);
}

std::string BuildAnthropicError(const std::string& message,
                                const std::string& type) {
    return "{\"type\":\"error\",\"error\":{\"type\":\"" + json::JsonEscape(type) +
           "\",\"message\":\"" + json::JsonEscape(message) + "\"}}";
}

std::string BuildAnthropicModelsList(const std::vector<std::string>& ids) {
    std::string data = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) data += ",";
        data += "{\"id\":\"" + json::JsonEscape(ids[i]) + "\",\"display_name\":\"" +
                json::JsonEscape(ids[i]) +
                "\",\"description\":\"logestix\",\"type\":\"model\"}";
    }
    data += "]";
    return "{\"data\":" + data + ",\"has_more\":false}";
}

}  // namespace lsxnetwork
