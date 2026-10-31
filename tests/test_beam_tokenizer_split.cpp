// Beam-1.0 pre-tokenizer (Model Ingestion Protocol stage 4): the checkpoint's
// tokenizer.json is the Qwen2/3 single-run pattern —
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]? \p{L}+ | \p{N} |
//   ' '? [^\s\p{L}\p{N}]+ [\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// — one letter run (no case split), ONE digit per number word, no CJK
// isolation and no `\p{M}` in the classes. This test pins the split itself;
// test_beam_tokenizer_parity pins the ids against the source tokenizer.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/pretokenizer.h>

#include <string>
#include <vector>

namespace {

std::vector<std::string> Split(const std::string &text)
{
    lsxcommon::BeamWordStream ws(text.data(), text.size());
    lsxcommon::WordSpan w;
    std::vector<std::string> out;
    while (ws.next(w)) out.emplace_back(w.begin, w.length);
    REQUIRE(ws.error() == 0);
    return out;
}

using Pieces = std::vector<std::string>;

} // namespace

TEST_CASE("beam pre-tokenizer keeps one letter run (no case split)",
          "[beam][pretokenizer]")
{
    REQUIRE(Split("getUserName") == Pieces{"getUserName"});
    REQUIRE(Split("hello world") == Pieces{"hello", " world"});
}

TEST_CASE("beam pre-tokenizer isolates one digit per number word",
          "[beam][pretokenizer]")
{
    REQUIRE(Split("1234567") == Pieces{"1", "2", "3", "4", "5", "6", "7"});
    REQUIRE(Split("v2beta") == Pieces{"v", "2", "beta"});
}

TEST_CASE("beam pre-tokenizer does not isolate CJK", "[beam][pretokenizer]")
{
    // Han is \p{L} (Lo), and the pattern has no CJK split.
    REQUIRE(Split("中文ab") == Pieces{"中文ab"});
}

TEST_CASE("beam pre-tokenizer keeps the Qwen whitespace lookahead",
          "[beam][pretokenizer]")
{
    REQUIRE(Split("hello  world") == Pieces{"hello", " ", " world"});
    REQUIRE(Split("a\n\nb") == Pieces{"a", "\n\n", "b"});
}

TEST_CASE("beam pre-tokenizer punctuation and contractions",
          "[beam][pretokenizer]")
{
    REQUIRE(Split("a...b") == Pieces{"a", "...", "b"});
    REQUIRE(Split("it's") == Pieces{"it", "'s"});
    REQUIRE(Split("</tool_call>") == Pieces{"</", "tool", "_call", ">"});
}
