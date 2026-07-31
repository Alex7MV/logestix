#include <catch2/catch_test_macros.hpp>
#include <lsxcommon/unicode.h>

TEST_CASE("unicode classification spot checks", "[tokenizer][unicode]")
{
    using lsxcommon::unicode::CharClass;
    CHECK(lsxcommon::unicode::classify(0x41) == CharClass::Letter);       // 'A'
    CHECK(lsxcommon::unicode::classify(0x61) == CharClass::Letter);       // 'a'
    CHECK(lsxcommon::unicode::classify(0x4E2D) == CharClass::Letter);     // 中
    CHECK(lsxcommon::unicode::classify(0x9FA6) == CharClass::Letter);     // 龦 (L, outside CJK split range)
    CHECK(lsxcommon::unicode::classify(0x0301) == CharClass::Mark);       // combining acute
    CHECK(lsxcommon::unicode::classify(0x30) == CharClass::Number);       // '0'
    CHECK(lsxcommon::unicode::classify(0x00B2) == CharClass::Number);     // ² (No)
    CHECK(lsxcommon::unicode::classify(0x21) == CharClass::Punct);        // '!'
    CHECK(lsxcommon::unicode::classify(0x20AC) == CharClass::Symbol);     // €
    CHECK(lsxcommon::unicode::classify(0x20) == CharClass::Whitespace);   // space
    CHECK(lsxcommon::unicode::classify(0x00A0) == CharClass::Whitespace); // NBSP
    CHECK(lsxcommon::unicode::classify(0x0A) == CharClass::Whitespace);   // \n
    CHECK(lsxcommon::unicode::classify(0x09) == CharClass::Whitespace);   // \t
    CHECK(lsxcommon::unicode::classify(0x0001) == CharClass::Other);      // control char
}
