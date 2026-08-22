// Тест Gemma-4 chat-template framing descriptor (host-only).
// Сентинелы сверены с реальным tokenizer.json чекпоинта:
//   <|turn>=105, <turn|>=106, "\n"=107 (vocab), "user"/"model" — одиночные ключи vocab.
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "lsxcommon/prompt_framing.h"
#include "lsxcommon/tokenizer.h"

using lsxcommon::prompt::LookupFraming;

TEST_CASE("gemma framing descriptor is registered", "[gemma][framing]") {
    auto *f = LookupFraming("gemma-4-26b-a4b-it");
    REQUIRE(f != nullptr);
    // <|turn>user\n ... <turn|>\n<|turn>model\n
    REQUIRE(f->leading_count == 3);
    REQUIRE(f->trailing_count == 5);
    REQUIRE(f->leading[0] == "<|turn>");
    REQUIRE(f->leading[1] == "user");
    REQUIRE(f->leading[2] == "\n");
    REQUIRE(f->trailing[0] == "<turn|>");
    REQUIRE(f->trailing[1] == "\n");
    REQUIRE(f->trailing[2] == "<|turn>");
    REQUIRE(f->trailing[3] == "model");
    REQUIRE(f->trailing[4] == "\n");
}

TEST_CASE("glm framing unaffected by gemma entry", "[gemma][framing]") {
    auto *g = LookupFraming("glm-5.2");
    REQUIRE(g != nullptr);
    REQUIRE(g->leading_count == 3);
    REQUIRE(g->leading[0] == "[gMASK]");
    CHECK(LookupFraming("unknown-model") == nullptr);
}
