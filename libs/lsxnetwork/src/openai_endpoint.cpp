#include "lsxnetwork/openai_endpoint.h"

#include <iomanip>
#include <sstream>

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

}  // namespace lsxnetwork