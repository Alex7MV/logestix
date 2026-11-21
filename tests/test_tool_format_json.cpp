// Neutral tool-format JSON helpers (dedup phase 4).
//
// The byte-identical per-family copies (PyJsonString/PyJsonValue/PyJsonDumps
// in flint/hy/muse, SerializeJson in deepseek/gemma/gigachat,
// AppendJinjaString in dory/qwen) move to lsxcommon::tools. These goldens
// pin the two DISTINCT escape semantics — they must not be merged into one
// another:
//   * PyJson*  — Python json.dumps(ensure_ascii=True): ", ", ": " separators,
//     <, >, &, ' escaped as \u00XX, non-ASCII as \uXXXX surrogate pairs,
//     doubles via %g.
//   * SerializeJson — compact JSON via lsxnetwork::json::JsonEscape, doubles
//     via %.17g.
//   * AppendJinjaString — Jinja tojson: control chars \u00XX, no HTML escape,
//     bytes >= 0x20 pass through verbatim.
#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/tool_format.h"

#include <string>

TEST_CASE("tools::PyJsonDumps matches Python json.dumps ensure_ascii shape",
          "[tools][json]")
{
    using lsxcommon::tools::PyJsonDumps;
    // separators ", " and ": " (the template's `fn | tojson` shape)
    CHECK(PyJsonDumps(R"({"a":1,"b":[true,null]})") ==
          "{\"a\": 1, \"b\": [true, null]}");
    // HTML-significant chars and the single quote are \u-escaped
    CHECK(PyJsonDumps(R"({"s":"<>&'"})") ==
          "{\"s\": \"\\u003c\\u003e\\u0026\\u0027\"}");
    // control chars \u00XX
    CHECK(PyJsonDumps("{\"s\":\"a\\nb\"}") == "{\"s\": \"a\\nb\"}");
    CHECK(PyJsonDumps("{\"s\":\"\\u0001\"}") == "{\"s\": \"\\u0001\"}");
    // non-ASCII -> \uXXXX (BMP) and surrogate pair (above BMP)
    CHECK(PyJsonDumps("{\"s\":\"\u00e9\"}") == "{\"s\": \"\\u00e9\"}");
    CHECK(PyJsonDumps("{\"s\":\"\U0001f600\"}") ==
          "{\"s\": \"\\ud83d\\ude00\"}");
    // doubles via %g
    CHECK(PyJsonDumps("{\"d\":1.5}") == "{\"d\": 1.5}");
    // invalid JSON -> empty string (the family contract)
    CHECK(PyJsonDumps("{broken") == "");
}

TEST_CASE("tools::SerializeJson stays compact with %.17g doubles",
          "[tools][json]")
{
    using lsxcommon::tools::SerializeJson;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    const std::string j1 = R"({"a":1,"b":[true,null]})";
    const std::string j2 = R"({"s":"x\ny"})";
    const std::string j3 = "{\"d\":0.1}";
    REQUIRE(parser.parse(j1).get(doc) == simdjson::SUCCESS);
    CHECK(SerializeJson(doc) == "{\"a\":1,\"b\":[true,null]}");
    REQUIRE(parser.parse(j2).get(doc) == simdjson::SUCCESS);
    CHECK(SerializeJson(doc) == "{\"s\":\"x\\ny\"}");
    REQUIRE(parser.parse(j3).get(doc) == simdjson::SUCCESS);
    CHECK(SerializeJson(doc) == "{\"d\":0.10000000000000001}");
}

TEST_CASE("tools::AppendJinjaString escapes controls but not HTML or UTF-8",
          "[tools][json]")
{
    std::string out;
    lsxcommon::tools::AppendJinjaString(out, "a<b>\u00e9\"\\\n");
    CHECK(out == "\"a<b>\u00e9\\\"\\\\\\n\"");
    out.clear();
    lsxcommon::tools::AppendJinjaString(out, std::string("\x01", 1));
    CHECK(out == "\"\\u0001\"");
}