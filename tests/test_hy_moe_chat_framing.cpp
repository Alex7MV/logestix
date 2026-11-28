// Hy-Memory-30B-A3B chat-framing parity against the checkpoint's own
// chat_template.jinja + tokenizer: the engine's native renderer must produce
// the reference ids for the control prompt, the FSI/CODE/DET gate bodies, a
// multi-turn conversation and a tool round-trip.
//
// The golden (tests/fixtures/hy/chat_framing.json) is captured by
// tools/hy_chat_framing_capture.py (tokenizer only, no weights). SKIPs (4)
// when the converted vault entry is absent.
#include <catch2/catch_test_macros.hpp>

#include "test_manifest_query_root.h"

#include <arrow/io/file.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <simdjson.h>

#include <lsxcommon/hy_moe_tool_format.h>
#include <lsxcommon/tokenizer.h>

#ifndef LSX_HY_FIXTURE_DIR
#define LSX_HY_FIXTURE_DIR "tests/fixtures/hy"
#endif

namespace {

using lsxcommon::ChatConversation;
using lsxcommon::ChatMessage;
using lsxcommon::ToolSpec;

struct Case {
    std::string name;
    ChatConversation conv;
    std::vector<int32_t> ids;
};

bool LoadVaultDb(lsxcommon::ArrowTokenizerDb &db)
{
    const std::filesystem::path root =
        manifest_query::ConvertedDirFor("hy-memory-30b-a3b") / "pinned";
    const std::filesystem::path v = root / "tokenizer_vocab.arrow";
    const std::filesystem::path m = root / "tokenizer_merges.arrow";
    if (!std::filesystem::exists(v) || !std::filesystem::exists(m)) return false;
    auto vf = arrow::io::MemoryMappedFile::Open(v.string(),
                                                arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(m.string(),
                                                arrow::io::FileMode::READ);
    if (!vf.ok() || !mf.ok()) return false;
    return lsxcommon::load_tokenizer_metadata(*vf, *mf, db).ok();
}

std::vector<Case> LoadCases()
{
    std::ifstream f(LSX_HY_FIXTURE_DIR "/chat_framing.json", std::ios::binary);
    REQUIRE(f.good());
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);
    std::vector<Case> cases;
    for (auto c : doc["cases"].get_array()) {
        Case out;
        std::string_view name;
        REQUIRE(c["name"].get_string().get(name) == simdjson::SUCCESS);
        out.name = std::string(name);
        for (auto m : c["messages"].get_array()) {
            ChatMessage msg;
            std::string_view role, content;
            REQUIRE(m["role"].get_string().get(role) == simdjson::SUCCESS);
            msg.role = std::string(role);
            if (m["content"].get_string().get(content) == simdjson::SUCCESS)
                msg.content = std::string(content);
            auto tc = m["tool_calls"];
            if (tc.error() == simdjson::SUCCESS) {
                for (auto call : tc.get_array()) {
                    lsxcommon::ToolCall out_call;
                    std::string_view cname;
                    REQUIRE(call["function"]["name"]
                                .get_string().get(cname) == simdjson::SUCCESS);
                    out_call.name = std::string(cname);
                    // Re-serialize the arguments exactly as the capture saw
                    // them; the renderer parses this JSON.
                    std::string args = simdjson::to_string(
                        call["function"]["arguments"]);
                    out_call.arguments_json = args;
                    msg.tool_calls.push_back(std::move(out_call));
                }
            }
            out.conv.messages.push_back(std::move(msg));
        }
        for (auto t : c["tools"].get_array()) {
            ToolSpec spec;
            std::string_view fn_name, desc, params;
            REQUIRE(t["function"]["name"].get_string().get(fn_name) ==
                    simdjson::SUCCESS);
            spec.name = std::string(fn_name);
            if (t["function"]["description"].get_string().get(desc) ==
                simdjson::SUCCESS)
                spec.description = std::string(desc);
            spec.parameters_json =
                simdjson::to_string(t["function"]["parameters"]);
            out.conv.tools.push_back(std::move(spec));
        }
        for (auto id : c["ids"].get_array()) {
            int64_t v = 0;
            REQUIRE(id.get_int64().get(v) == simdjson::SUCCESS);
            out.ids.push_back(static_cast<int32_t>(v));
        }
        cases.push_back(std::move(out));
    }
    return cases;
}

} // namespace

TEST_CASE("Hy native renderer reproduces the checkpoint chat framing",
          "[hy][framing]")
{
    lsxcommon::ArrowTokenizerDb db;
    if (!LoadVaultDb(db))
        SKIP(std::string("converted Hy-Memory vault entry absent: ") +
             manifest_query::ConvertedDirFor("hy-memory-30b-a3b").string() +
             "/pinned");
    const auto cases = LoadCases();
    REQUIRE(cases.size() >= 10);
    for (const auto &c : cases) {
        std::vector<int32_t> ids;
        INFO("case " << c.name);
        REQUIRE(lsxcommon::hy::RenderHyConversation(db, c.conv, ids));
        std::string want, got;
        for (auto x : c.ids) want += std::to_string(x) + " ";
        for (auto x : ids) got += std::to_string(x) + " ";
        INFO("want: " << want << "\n got: " << got);
        CHECK(ids == c.ids);
    }
}
