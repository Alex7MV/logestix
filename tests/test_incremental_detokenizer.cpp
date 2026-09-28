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

TEST_CASE("detokenizer: flush emits any held bytes", "[detokenizer][fast]") {
    auto d = MakeFake({{1, std::string("\xC3", 1)}});
    CHECK(d.Push(1).empty());
    CHECK(d.Flush() == std::string("\xC3", 1));
}

TEST_CASE("detokenizer: multi-byte codepoint intact", "[detokenizer][fast]") {
    auto d = MakeFake({{1, "\xE2\x82"}, {2, "\xAC"}});  // €
    CHECK(d.Push(1).empty());
    CHECK(d.Push(2) == "\xE2\x82\xAC");
}
