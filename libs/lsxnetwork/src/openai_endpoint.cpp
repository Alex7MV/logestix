#include "lsxnetwork/openai_endpoint.h"
#include <lsxnetwork/json_escape.h>

#include <iomanip>
#include <sstream>
#include <utility>

#include <simdjson.h>

namespace lsxnetwork {

simdjson::error_code OpenAiEndpoint::Parse(const char* data, std::size_t size,
                                           std::size_t capacity,
                                           ChatCompletion& out) const {
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;

    simdjson::padded_string_view bufferview(std::string_view(data, size), capacity);
    if (auto e = parser.iterate(bufferview).get(doc)) {
        return e;
    }

    simdjson::ondemand::object root;
    if (auto e = doc.get_object().get(root)) {
        return e;
    }

    if (auto e = root["id"].get_string(out.id)) return e;
    if (auto e = root["object"].get_string(out.object)) return e;

    // Optional dispatch target. OpenAI requests/reports carry a top-level
    // "model" string; absent/weird payloads leave it empty rather than failing
    // the whole parse (a missing model routes to the orchestrator's default).
    if (auto err = root["model"].get_string(out.model)) {
        if (err != simdjson::NO_SUCH_FIELD) return err;
    }

    // "max_tokens" is request-side only; absent on responses -> keep default 0.
    if (auto mt = root["max_tokens"].get_int64(); mt.error() == simdjson::SUCCESS) {
        out.max_tokens = static_cast<int>(mt.value_unsafe());
    } else if (mt.error() != simdjson::NO_SUCH_FIELD) {
        return mt.error();
    }

    // "usage" is response-side only; requests never carry it -> keep 0s.
    simdjson::ondemand::object usage;
    if (auto e = root["usage"].get_object().get(usage)) {
        if (e != simdjson::NO_SUCH_FIELD) return e;
    } else {
        if (auto e = usage["prompt_tokens"].get_int64().get(out.prompt_tokens)) return e;
        if (auto e = usage["completion_tokens"].get_int64().get(out.completion_tokens))
            return e;
        if (auto e = usage["total_tokens"].get_int64().get(out.total_tokens)) return e;
    }

    // Request format: content lives in messages[0].{role,content}.
    simdjson::ondemand::array messages;
    if (!root["messages"].get_array().get(messages)) {
        simdjson::ondemand::value msg_value;
        if (auto e = messages.at(0).get(msg_value)) return e;
        simdjson::ondemand::object msg_obj;
        if (auto e = msg_value.get_object().get(msg_obj)) return e;
        if (auto e = msg_obj["content"].get_string(out.content)) return e;
        if (auto e = msg_obj["role"].get_string(out.role)) return e;
        return simdjson::SUCCESS;
    }

    // Response format fallback: choices[0].message.{role,content} (the legacy
    // harness fixture). A frame with neither messages nor choices fails here.
    simdjson::ondemand::array choices;
    if (auto e = root["choices"].get_array().get(choices)) return e;

    simdjson::ondemand::value choice_value;
    if (auto e = choices.at(0).get(choice_value)) return e;

    simdjson::ondemand::object choice;
    if (auto e = choice_value.get_object().get(choice)) return e;
    if (auto e = choice["finish_reason"].get_string(out.finish_reason)) return e;

    simdjson::ondemand::object message;
    if (auto e = choice["message"].get_object().get(message)) return e;
    if (auto e = message["role"].get_string(out.role)) return e;
    if (auto e = message["content"].get_string(out.content)) return e;

    return simdjson::SUCCESS;
}

bool OpenAiEndpoint::WriteCompletion(const ChatCompletion& req,
                                     const CompletionResponse& res,
                                     std::string& out) const {
    std::ostringstream oss;
    oss << "{\"id\":\"" << req.id << "\","
        << "\"object\":\"chat.completion\","
        << "\"model\":\"" << req.model << "\","
        << "\"choices\":[{\"index\":0,"
        << "\"message\":{\"role\":\"assistant\",\"content\":\"";
    // Escape the content minimally: JSON-escape '"' and '\' and control chars.
    for (char c : res.output_text) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20)
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c))
                        << std::dec << std::setfill(' ');
                else
                    oss << c;
        }
    }
    const std::int64_t comp_tokens = static_cast<std::int64_t>(res.gen_tokens.size());
    oss << "\"},\"finish_reason\":\""
        << (res.eos_terminated ? "stop" : "length")
        << "\"}],"
        << "\"usage\":{\"prompt_tokens\":" << req.prompt_tokens
        << ",\"completion_tokens\":" << comp_tokens
        << ",\"total_tokens\":" << (req.prompt_tokens + comp_tokens) << "},"
        << "\"lsx\":{\"ok\":" << (res.ok ? "true" : "false")
        << ",\"ttft_ms\":" << res.ttft_ms
        << ",\"tokens_per_sec\":" << res.tokens_per_sec << "}}";
    out = oss.str();
    return true;
}

simdjson::error_code OpenAiEndpoint::ParseChatRequest(const char* data,
                                                      std::size_t size,
                                                      std::size_t capacity,
                                                      ChatRequest& out) const {
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    simdjson::padded_string_view v(std::string_view(data, size), capacity);
    if (auto e = parser.iterate(v).get(doc)) return e;
    simdjson::ondemand::object root;
    if (auto e = doc.get_object().get(root)) return e;

    if (auto e = root["model"].get_string(out.model)) {
        if (e != simdjson::NO_SUCH_FIELD) return e;
    }
    if (auto s = root["stream"].get_bool(); s.error() == simdjson::SUCCESS)
        out.stream = s.value_unsafe();
    else if (s.error() != simdjson::NO_SUCH_FIELD)
        return s.error();

    if (auto mt = root["max_tokens"].get_int64();
        mt.error() == simdjson::SUCCESS)
        out.max_tokens = static_cast<int>(mt.value_unsafe());
    else if (mt.error() != simdjson::NO_SUCH_FIELD)
        return mt.error();

    if (auto tv = root["temperature"].get_double();
        tv.error() == simdjson::SUCCESS)
        out.sampling.temperature = static_cast<float>(tv.value_unsafe());
    else if (tv.error() != simdjson::NO_SUCH_FIELD)
        return tv.error();
    if (auto pv = root["top_p"].get_double(); pv.error() == simdjson::SUCCESS)
        out.sampling.top_p = static_cast<float>(pv.value_unsafe());
    else if (pv.error() != simdjson::NO_SUCH_FIELD)
        return pv.error();
    if (auto kv = root["top_k"].get_int64(); kv.error() == simdjson::SUCCESS)
        out.sampling.top_k = static_cast<std::int32_t>(kv.value_unsafe());
    else if (kv.error() != simdjson::NO_SUCH_FIELD)
        return kv.error();
    if (auto sv = root["seed"].get_int64(); sv.error() == simdjson::SUCCESS)
        out.sampling.seed = static_cast<std::uint64_t>(sv.value_unsafe());
    else if (sv.error() != simdjson::NO_SUCH_FIELD)
        return sv.error();

    simdjson::ondemand::object so;
    if (auto e = root["stream_options"].get_object().get(so)) {
        if (e != simdjson::NO_SUCH_FIELD) return e;
    } else if (auto iu = so["include_usage"].get_bool();
               iu.error() == simdjson::SUCCESS) {
        out.include_usage = iu.value_unsafe();
    }

    if (auto arr = root["stop"].get_array(); arr.error() == simdjson::SUCCESS) {
        for (auto s : arr.value_unsafe()) {
            std::string_view sv;
            if (auto e = s.get_string().get(sv); e) return e;
            out.stop.emplace_back(sv);
        }
    } else if (arr.error() != simdjson::NO_SUCH_FIELD) {
        if (auto sv = root["stop"].get_string();
            sv.error() == simdjson::SUCCESS)
            out.stop.emplace_back(sv.value_unsafe());
        else if (sv.error() != simdjson::NO_SUCH_FIELD)
            return sv.error();
    }

    simdjson::ondemand::array messages;
    if (auto e = root["messages"].get_array().get(messages)) return e;
    for (auto m : messages) {
        simdjson::ondemand::object mo;
        if (auto e = m.get_object().get(mo)) return e;
        ChatInputMessage msg;
        std::string_view sv;
        if (auto e = mo["role"].get_string().get(sv); !e) msg.role.assign(sv);
        if (auto c = mo["content"].get_string(); c.error() == simdjson::SUCCESS)
            msg.content.assign(c.value_unsafe());
        if (auto e = mo["tool_call_id"].get_string().get(sv); !e)
            msg.tool_call_id.assign(sv);
        simdjson::ondemand::array tcs;
        if (mo["tool_calls"].get_array().get(tcs) == simdjson::SUCCESS) {
            for (auto tv : tcs) {
                simdjson::ondemand::object to;
                if (tv.get_object().get(to)) continue;
                ChatToolCall call;
                if (auto e = to["id"].get_string().get(sv); !e)
                    call.id.assign(sv);
                simdjson::ondemand::object fn;
                if (to["function"].get_object().get(fn) == simdjson::SUCCESS) {
                    if (auto e = fn["name"].get_string().get(sv); !e)
                        call.name.assign(sv);
                    simdjson::ondemand::value av;
                    if (fn["arguments"].get(av) == simdjson::SUCCESS) {
                        const auto is = av.is_string();
                        if (is.error() == simdjson::SUCCESS && is.value_unsafe()) {
                            std::string_view as;
                            if (av.get_string().get(as) == simdjson::SUCCESS)
                                call.arguments_json.assign(as);
                        } else if (auto rj = av.raw_json();
                                   rj.error() == simdjson::SUCCESS) {
                            call.arguments_json.assign(rj.value_unsafe());
                        }
                    }
                }
                msg.tool_calls.push_back(std::move(call));
            }
        }
        out.messages.push_back(std::move(msg));
    }

    simdjson::ondemand::array tools;
    if (root["tools"].get_array().get(tools) == simdjson::SUCCESS) {
        for (auto tv : tools) {
            simdjson::ondemand::object to;
            if (tv.get_object().get(to)) continue;
            simdjson::ondemand::object fn;
            if (to["function"].get_object().get(fn)) continue;
            ChatToolDef def;
            std::string_view sv;
            if (auto e = fn["name"].get_string().get(sv); !e) def.name.assign(sv);
            if (auto e = fn["description"].get_string().get(sv); !e)
                def.description.assign(sv);
            simdjson::ondemand::value pv;
            if (fn["parameters"].get(pv) == simdjson::SUCCESS) {
                if (auto rj = pv.raw_json(); rj.error() == simdjson::SUCCESS)
                    def.parameters_json.assign(rj.value_unsafe());
            }
            if (def.parameters_json.empty()) def.parameters_json = "{}";
            out.tools.push_back(std::move(def));
        }
    }
    if (auto tc = root["tool_choice"].get_string();
        tc.error() == simdjson::SUCCESS)
        out.tool_choice.assign(tc.value_unsafe());
    return simdjson::SUCCESS;
}

simdjson::error_code OpenAiEndpoint::ParseLegacyCompletionRequest(
    const char* data, std::size_t size, std::size_t capacity,
    LegacyCompletionRequest& out) const {
    simdjson::ondemand::parser parser;
    simdjson::ondemand::document doc;
    simdjson::padded_string_view v(std::string_view(data, size), capacity);
    if (auto e = parser.iterate(v).get(doc)) return e;
    simdjson::ondemand::object root;
    if (auto e = doc.get_object().get(root)) return e;

    if (auto e = root["model"].get_string(out.model)) {
        if (e != simdjson::NO_SUCH_FIELD) return e;
    }
    if (auto s = root["stream"].get_bool(); s.error() == simdjson::SUCCESS)
        out.stream = s.value_unsafe();
    else if (s.error() != simdjson::NO_SUCH_FIELD)
        return s.error();
    if (auto mt = root["max_tokens"].get_int64();
        mt.error() == simdjson::SUCCESS)
        out.max_tokens = static_cast<int>(mt.value_unsafe());
    else if (mt.error() != simdjson::NO_SUCH_FIELD)
        return mt.error();

    simdjson::ondemand::object so;
    if (auto e = root["stream_options"].get_object().get(so)) {
        if (e != simdjson::NO_SUCH_FIELD) return e;
    } else if (auto iu = so["include_usage"].get_bool();
               iu.error() == simdjson::SUCCESS) {
        out.include_usage = iu.value_unsafe();
    }

    if (auto arr = root["stop"].get_array(); arr.error() == simdjson::SUCCESS) {
        for (auto s : arr.value_unsafe()) {
            std::string_view sv;
            if (auto e = s.get_string().get(sv); e) return e;
            out.stop.emplace_back(sv);
        }
    } else if (arr.error() != simdjson::NO_SUCH_FIELD) {
        if (auto sv = root["stop"].get_string();
            sv.error() == simdjson::SUCCESS)
            out.stop.emplace_back(sv.value_unsafe());
        else if (sv.error() != simdjson::NO_SUCH_FIELD)
            return sv.error();
    }

    if (auto ps = root["prompt"].get_string();
        ps.error() == simdjson::SUCCESS) {
        out.prompt.assign(ps.value_unsafe());
    } else if (ps.error() == simdjson::INCORRECT_TYPE) {
        simdjson::ondemand::array pa;
        if (auto e = root["prompt"].get_array().get(pa)) return e;
        for (auto s : pa) {
            std::string_view sv;
            if (auto e = s.get_string().get(sv); e) return e;
            out.prompt.append(sv);
        }
    } else if (ps.error() != simdjson::NO_SUCH_FIELD) {
        return ps.error();
    }
    return simdjson::SUCCESS;
}

// NOTE: lsxcommon/src/json_escape.h holds the canonical implementation of this
// function (JSON RFC 8259: control chars as \u00XX). It is byte-identical to the
std::string SseFrame(const std::string& data) {
    return "data: " + data + "\n\n";
}

namespace {
std::string Usage(int prompt_tokens, int completion_tokens) {
    return "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) +
           ",\"completion_tokens\":" + std::to_string(completion_tokens) +
           ",\"total_tokens\":" + std::to_string(prompt_tokens + completion_tokens) +
           "}";
}
}  // namespace

std::string BuildChatCompletion(const std::string& id, const std::string& model,
                                const std::string& text,
                                const std::string& finish_reason,
                                int prompt_tokens, int completion_tokens) {
    return "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"model\":\"" +
           model +
           "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
           "\"content\":\"" +
           json::JsonEscape(text) + "\"},\"finish_reason\":\"" + finish_reason +
           "\"}]," + Usage(prompt_tokens, completion_tokens) + "}";
}

std::string BuildChatChunk(const std::string& id, const std::string& model,
                           const std::string& delta_content, bool first_delta,
                           const std::string& finish_reason, int prompt_tokens,
                           int completion_tokens, bool include_usage) {
    std::string out =
        "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"model\":\"" +
        model + "\",\"choices\":[{\"index\":0,\"delta\":{";
    if (first_delta) out += "\"role\":\"assistant\",";
    out += "\"content\":\"" + json::JsonEscape(delta_content) + "\"},\"finish_reason\":";
    out += finish_reason.empty() ? "null" : ("\"" + finish_reason + "\"");
    out += "}]";
    if (include_usage && !finish_reason.empty())
        out += "," + Usage(prompt_tokens, completion_tokens);
    out += "}";
    return out;
}

std::string BuildTextCompletion(const std::string& id, const std::string& model,
                                const std::string& text,
                                const std::string& finish_reason,
                                int prompt_tokens, int completion_tokens) {
    return "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"model\":\"" +
           model + "\",\"choices\":[{\"index\":0,\"text\":\"" + json::JsonEscape(text) +
           "\",\"finish_reason\":\"" + finish_reason + "\"}]," +
           Usage(prompt_tokens, completion_tokens) + "}";
}

std::string BuildTextChunk(const std::string& id, const std::string& model,
                           const std::string& delta_text,
                           const std::string& finish_reason) {
    std::string out = "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"model\":\"" +
                      model + "\",\"choices\":[{\"index\":0,\"text\":\"" +
                      json::JsonEscape(delta_text) + "\",\"finish_reason\":";
    out += finish_reason.empty() ? "null" : ("\"" + finish_reason + "\"");
    out += "}]}";
    return out;
}

std::string BuildModel(const std::string& id) {
    return "{\"id\":\"" + json::JsonEscape(id) +
           "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"logestix\"}";
}

std::string BuildModelsList(const std::vector<std::string>& ids) {
    std::string data = "[";
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) data += ",";
        data += BuildModel(ids[i]);
    }
    data += "]";
    return "{\"object\":\"list\",\"data\":" + data + "}";
}

std::string BuildError(const std::string& message, const std::string& type) {
    return "{\"error\":{\"message\":\"" + json::JsonEscape(message) + "\",\"type\":\"" +
           json::JsonEscape(type) + "\",\"param\":null,\"code\":null}}";
}

std::string ToolCallsToJson(const std::vector<ChatToolCall>& calls) {
    std::string out = "[";
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (i) out += ",";
        out += "{\"id\":\"" + json::JsonEscape(calls[i].id) +
               "\",\"type\":\"function\",\"function\":{\"name\":\"" +
               json::JsonEscape(calls[i].name) + "\",\"arguments\":\"" +
               json::JsonEscape(calls[i].arguments_json.empty() ? "{}"
                                                          : calls[i].arguments_json) +
               "\"}}";
    }
    out += "]";
    return out;
}

std::string BuildChatCompletionTools(const std::string& id,
                                     const std::string& model,
                                     const std::string& tool_calls_json,
                                     int prompt_tokens, int completion_tokens) {
    return "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"model\":\"" +
           model +
           "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
           "\"content\":null,\"tool_calls\":" +
           tool_calls_json + "},\"finish_reason\":\"tool_calls\"}]," +
           Usage(prompt_tokens, completion_tokens) + "}";
}

std::string BuildChatChunkTools(const std::string& id, const std::string& model,
                                const std::string& tool_calls_json,
                                bool first_delta, int prompt_tokens,
                                int completion_tokens, bool include_usage) {
    std::string out =
        "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"model\":\"" +
        model + "\",\"choices\":[{\"index\":0,\"delta\":{";
    if (first_delta) out += "\"role\":\"assistant\",";
    out += "\"tool_calls\":" + tool_calls_json +
           "},\"finish_reason\":\"tool_calls\"}]";
    if (include_usage) out += "," + Usage(prompt_tokens, completion_tokens);
    out += "}";
    return out;
}

}  // namespace lsxnetwork