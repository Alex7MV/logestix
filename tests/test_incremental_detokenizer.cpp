#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/incremental_detokenizer.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

using lsxcommon::IncrementalDetokenizer;

namespace {
// A fake decoder mapping token ids to raw byte segments (may split a codepoint).
IncrementalDetokenizer MakeFake(const std::map<std::int32_t, std::string>& t) {
    return IncrementalDetokenizer([t](const std::vector<std::int32_t>& ids) {
        std::string s;
        for (const auto id : ids) {
            auto it = t.find(id);
            if (it != t.end()) s += it->second;
        }
        return s;
    });
}
}  // namespace

TEST_CASE("detokenizer: ascii deltas stream immediately",
          "[detokenizer][fast]") {
    auto d = MakeFake({{1, "Hel"}, {2, "lo"}});
    CHECK(d.Push(1) == "Hel");
    CHECK(d.Push(2) == "lo");
    CHECK(d.Flush().empty());
}

TEST_CASE("detokenizer: holds back an incomplete UTF-8 sequence",
          "[detokenizer][fast]") {
    // "é" = 0xC3 0xA9 split across two tokens.
    auto d = MakeFake({{1, std::string("\xC3", 1)},
                       {2, std::string("\xA9", 1)},
                       {3, "!"}});
    CHECK(d.Push(1).empty());         // incomplete, withheld
    CHECK(d.Push(2) == "\xC3\xA9");   // now complete
    CHECK(d.Push(3) == "!");
}

// End of stream must not be a hole in the UTF-8 guarantee: Push withholds an
// incomplete sequence, and whatever Flush hands back has to be text the encoder
// will accept again, or BuildConversationInputIds fails and the next turn is
// dropped. A truncated sequence becomes one replacement character, never raw
// bytes.
TEST_CASE("detokenizer: flush substitutes rather than emits broken UTF-8",
          "[detokenizer][fast]") {
    auto d = MakeFake({{1, std::string("\xC3", 1)}});
    CHECK(d.Push(1).empty());
    CHECK(d.Flush() == "\xEF\xBF\xBD");     // U+FFFD, not the lone 0xC3
}

// A codepoint completed by a later token is real text and must survive; once
// nothing is withheld, Flush has nothing left to resolve.
TEST_CASE("detokenizer: a completed codepoint is emitted and Flush is empty",
          "[detokenizer][fast]") {
    auto d = MakeFake({{1, std::string("\xC3", 1)}, {2, std::string("\xA9", 1)}});
    CHECK(d.Push(1).empty());
    CHECK(d.Push(2) == "\xC3\xA9");         // "é" intact
    CHECK(d.Flush().empty());
}

// An overlong form is invalid UTF-8 even though its shape looks well formed:
// the encoder rejects it, so neither Push nor Flush may hand out the raw bytes.
// Two replacement characters here (one for the impossible lead 0xC1, one for the
// orphaned continuation 0xA9) are acceptable — what matters is that no raw byte
// escapes and the trailing real text still arrives.
TEST_CASE("detokenizer: never emits an overlong encoding", "[detokenizer][fast]") {
    auto d = MakeFake({{1, std::string("\xC1\xA9", 2)}, {2, "!"}});
    const std::string a = d.Push(1);
    const std::string b = d.Push(2);
    const std::string all = a + b;
    CHECK(all.find('\xC1') == std::string::npos);
    CHECK(all.find('\xA9') == std::string::npos);
    CHECK(all == "\xEF\xBF\xBD" "\xEF\xBF\xBD" "!");
}

TEST_CASE("detokenizer: multi-byte codepoint intact", "[detokenizer][fast]") {
    auto d = MakeFake({{1, "\xE2\x82"}, {2, "\xAC"}});  // €
    CHECK(d.Push(1).empty());
    CHECK(d.Push(2) == "\xE2\x82\xAC");
}
