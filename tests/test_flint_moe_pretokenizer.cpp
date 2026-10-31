// Flint pre-tokenizer (Model Ingestion Protocol stage 4): the checkpoint's
// tokenizer.json is a 3-split Sequence —
//   1. \p{N}{1,3}   Isolated
//   2. [\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+  Isolated
//   3. a 6-alternative o200k CASE-RUN pattern whose punctuation alternative is
//      ' '? [\p{P}\p{S}]+ [\r\n]*   (NOT o200k's [^\s\p{L}\p{N}]+[\r\n/]*)
// — which neither WordStream (digit+CJK, letter runs) nor Llama3WordStream
// (case runs, no digit/CJK split) reproduces. This test pins the split itself;
// test_flint_moe_tokenizer_parity pins the ids against the source tokenizer.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/pretokenizer.h>

#include <string>
#include <vector>

namespace {

std::vector<std::string> Split(const std::string &text)
{
    lsxcommon::FlintWordStream ws(text.data(), text.size());
    lsxcommon::WordSpan w;
    std::vector<std::string> out;
    while (ws.next(w)) out.emplace_back(w.begin, w.length);
    REQUIRE(ws.error() == 0);
    return out;
}

using Pieces = std::vector<std::string>;

} // namespace

TEST_CASE("flint pre-tokenizer splits case runs like o200k", "[flint][pretokenizer]")
{
    REQUIRE(Split("getUserName") == Pieces{"get", "User", "Name"});
    // o200k's first alternative is greedy: [Upper]*[Lower]+ makes
    // "HTTPRequest" ONE piece (verified against tokenizer.json).
    REQUIRE(Split("HTTPRequest") == Pieces{"HTTPRequest"});
    REQUIRE(Split("a...b") == Pieces{"a", "...", "b"});
}

TEST_CASE("flint pre-tokenizer isolates digit groups of at most three",
          "[flint][pretokenizer]")
{
    REQUIRE(Split("1234567") == Pieces{"123", "456", "7"});
    REQUIRE(Split("v2beta") == Pieces{"v", "2", "beta"});
}

TEST_CASE("flint pre-tokenizer isolates CJK runs", "[flint][pretokenizer]")
{
    REQUIRE(Split("中文ab") == Pieces{"中文", "ab"});
    REQUIRE(Split("abcあいうえお") == Pieces{"abc", "あいうえお"});
}

TEST_CASE("flint pre-tokenizer keeps the o200k whitespace lookahead",
          "[flint][pretokenizer]")
{
    REQUIRE(Split("hello  world") == Pieces{"hello", " ", " world"});
    REQUIRE(Split("a\n\nb") == Pieces{"a", "\n\n", "b"});
}

TEST_CASE("flint contraction suffix is greedy even before more letters",
          "[flint][pretokenizer]")
{
    // The reference pattern's optional (?i:'s|'t|...|'m|...) sits at the end of
    // the letter-run alternative with no "must not be followed by a letter"
    // constraint, so "majnf'mpRKP" is ["majnf'm", "p", "RKP"].
    REQUIRE(Split("majnf'mpRKP") == Pieces{"majnf'm", "p", "RKP"});
    REQUIRE(Split("we'll") == Pieces{"we'll"});
}

TEST_CASE("flint digit/CJK stages run before the main pattern",
          "[flint][pretokenizer]")
{
    // The reference is a three-stage Sequence: the digit and CJK runs are
    // isolated first, so the main pattern sees only the gap and its whitespace
    // lookahead hits EOF. "a  1" is ["a", "  ", "1"], not two spaces.
    REQUIRE(Split("a  1") == Pieces{"a", "  ", "1"});
    REQUIRE(Split("x   9") == Pieces{"x", "   ", "9"});
    REQUIRE(Split("a  中") == Pieces{"a", "  ", "中"});
    REQUIRE(Split("1  2") == Pieces{"1", "  ", "2"});
}

TEST_CASE("flint punctuation run stops at the newline, not at a slash",
          "[flint][pretokenizer]")
{
    // Flint's alternative is [\p{P}\p{S}]+[\r\n]* : the slash is punctuation
    // (so it belongs to the run) but is NOT swallowed with the following
    // newline the way o200k's [\r\n/]* alternative would.
    REQUIRE(Split("/\n/") == Pieces{"/\n", "/"});
    // The optional [^\r\n\p{L}\p{N}]? prefix of alt1 wins over the
    // punctuation run, so the underscore joins the following word.
    REQUIRE(Split("</tool_call>") == Pieces{"</", "tool", "_call", ">"});
}
