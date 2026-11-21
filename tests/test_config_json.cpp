// Neutral host helpers shared by family loaders/converters (dedup phase 5).
// Host-only; the bodies were byte-identical per-family copies.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/config_json.h"
#include "lsxcommon/selfcheck_common.h"

#include <simdjson.h>

#include <string>

TEST_CASE("config::GetDouble reads doubles and ints, rejects the rest",
          "[config][json]")
{
    simdjson::ondemand::parser parser;
    const std::string json = R"({"d":1.5,"i":7,"s":"x","b":true})";
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    REQUIRE(parser.iterate(pad).get(doc) == simdjson::SUCCESS);
    simdjson::ondemand::object o;
    REQUIRE(doc.get_object().get(o) == simdjson::SUCCESS);
    double v = 0.0;
    REQUIRE(lsxcommon::config::GetDouble(o, "d", v));
    CHECK(v == 1.5);
    REQUIRE(lsxcommon::config::GetDouble(o, "i", v));
    CHECK(v == 7.0);
    CHECK_FALSE(lsxcommon::config::GetDouble(o, "s", v));
    CHECK_FALSE(lsxcommon::config::GetDouble(o, "b", v));
    CHECK_FALSE(lsxcommon::config::GetDouble(o, "missing", v));
}

TEST_CASE("config::GetInt reads ints with a default", "[config][json]")
{
    simdjson::ondemand::parser parser;
    const std::string json = R"({"i":7,"s":"x"})";
    simdjson::padded_string pad(json.data(), json.size());
    simdjson::ondemand::document doc;
    REQUIRE(parser.iterate(pad).get(doc) == simdjson::SUCCESS);
    simdjson::ondemand::object o;
    REQUIRE(doc.get_object().get(o) == simdjson::SUCCESS);
    CHECK(lsxcommon::config::GetInt(o, "i", -1) == 7);
    CHECK(lsxcommon::config::GetInt(o, "s", -1) == -1);
    CHECK(lsxcommon::config::GetInt(o, "missing", 42) == 42);
}

TEST_CASE("config::ParseLayerBody splits model.layers.N.<body>",
          "[config][json]")
{
    int layer = -1;
    std::string body;
    REQUIRE(lsxcommon::config::ParseLayerBody("model.layers.12.self_attn.q_proj",
                                              layer, body));
    CHECK(layer == 12);
    CHECK(body == "self_attn.q_proj");
    CHECK_FALSE(lsxcommon::config::ParseLayerBody("model.embed_tokens.weight",
                                                  layer, body));
    CHECK_FALSE(lsxcommon::config::ParseLayerBody("model.layers.x.body",
                                                  layer, body));
    CHECK_FALSE(lsxcommon::config::ParseLayerBody("model.layers.3", layer, body));
}

TEST_CASE("selfcheck::Unfence strips code fences and language tags",
          "[selfcheck][unfence]")
{
    using lsxcommon::selfcheck::Unfence;
    CHECK(Unfence("  42  ") == "42");
    CHECK(Unfence("```python\n42\n```") == "42");
    CHECK(Unfence("```\n42\n```") == "42");
    CHECK(Unfence("```c++\nint x;\n```") == "int x;");
    // a first line that is not a language tag stays part of the content
    CHECK(Unfence("```42 is the answer\n```") == "42 is the answer");
    // unterminated fence: everything after the opener
    CHECK(Unfence("```python\n42") == "42");
    CHECK(Unfence("no fence") == "no fence");
}