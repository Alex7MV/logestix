// Mistral-Large-4 pre-tokenizer (Model Ingestion Protocol stage 4): the
// checkpoint's tokenizer.json uses the Small-4 o200k case-run pattern WITHOUT
// contractions, but its number alternative is o200k's RIGHT-grouped
// `\p{N}{1,3}(?=(?:\p{N}{3})*(?!\p{N}))|\p{N}{1,2}`. MistralWordStream (kind 9)
// groups digits one-per-word, so this needs its own stream (kind 12).
// This test pins the split; tokenizer parity against the real tokenizer.json is
// measured by the conversion round-trip (tools fixture, stage 4).
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/pretokenizer.h>

#include <string>
#include <vector>

namespace {

std::vector<std::string> Split(const std::string &text)
{
    lsxcommon::MistralGroupedWordStream ws(text.data(), text.size());
    lsxcommon::WordSpan w;
    std::vector<std::string> out;
    while (ws.next(w)) out.emplace_back(w.begin, w.length);
    REQUIRE(ws.error() == 0);
    return out;
}

using Pieces = std::vector<std::string>;

} // namespace

TEST_CASE("mistral-large4 pre-tokenizer right-groups number runs",
          "[mistral_large4][pretokenizer]")
{
    REQUIRE(Split("1234567") == Pieces{"1", "234", "567"});
    REQUIRE(Split("123456") == Pieces{"123", "456"});
    REQUIRE(Split("12345") == Pieces{"12", "345"});
    REQUIRE(Split("1234") == Pieces{"1", "234"});
    REQUIRE(Split("123") == Pieces{"123"});
    REQUIRE(Split("12") == Pieces{"12"});
    REQUIRE(Split("1") == Pieces{"1"});
    // A number run after a letter is still right-grouped inside the run.
    REQUIRE(Split("a1234") == Pieces{"a", "1", "234"});
}

TEST_CASE("mistral-large4 pre-tokenizer keeps the o200k case runs",
          "[mistral_large4][pretokenizer]")
{
    REQUIRE(Split("getUserName") == Pieces{"get", "User", "Name"});
    REQUIRE(Split("HTTPRequest") == Pieces{"HTTPRequest"});
    REQUIRE(Split("a...b") == Pieces{"a", "...", "b"});
    // No contraction alternative: the apostrophe is just the optional
    // `[^\r\n\p{L}\p{N}]?` prefix of the next letter run.
    REQUIRE(Split("it's") == Pieces{"it", "'s"});
}

TEST_CASE("mistral-large4 pre-tokenizer keeps the o200k whitespace rules",
          "[mistral_large4][pretokenizer]")
{
    REQUIRE(Split("hello  world") == Pieces{"hello", " ", " world"});
    REQUIRE(Split("a\n\nb") == Pieces{"a", "\n\n", "b"});
    // `[\r\n/]*` tail after a punctuation run.
    REQUIRE(Split("a//\n") == Pieces{"a", "//\n"});
}
