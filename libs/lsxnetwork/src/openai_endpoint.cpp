#include "lsxnetwork/openai_endpoint.h"

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

    if (auto e = root["id"].get_string().get(out.id)) return e;
    if (auto e = root["object"].get_string().get(out.object)) return e;

    simdjson::ondemand::object usage;
    if (auto e = root["usage"].get_object().get(usage)) return e;
    if (auto e = usage["prompt_tokens"].get_int64().get(out.prompt_tokens)) return e;
    if (auto e = usage["completion_tokens"].get_int64().get(out.completion_tokens))
        return e;
    if (auto e = usage["total_tokens"].get_int64().get(out.total_tokens)) return e;

    simdjson::ondemand::array choices;
    if (auto e = root["choices"].get_array().get(choices)) return e;

    simdjson::ondemand::value choice_value;
    if (auto e = choices.at(0).get(choice_value)) return e;

    simdjson::ondemand::object choice;
    if (auto e = choice_value.get_object().get(choice)) return e;
    if (auto e = choice["finish_reason"].get_string().get(out.finish_reason)) return e;

    simdjson::ondemand::object message;
    if (auto e = choice["message"].get_object().get(message)) return e;
    if (auto e = message["role"].get_string().get(out.role)) return e;
    if (auto e = message["content"].get_string().get(out.content)) return e;

    return simdjson::SUCCESS;
}

}  // namespace lsxnetwork