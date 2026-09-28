#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/token_stream.h>

#include <atomic>
#include <cstdint>
#include <vector>

TEST_CASE("generation stream: emits chunks in order", "[token_stream][fast]") {
    std::vector<std::uint32_t> seen;
    lsxcommon::TokenCallback cb = [&](const lsxcommon::TokenChunk& c) {
        seen.push_back(c.token_id);
        CHECK(c.step_t == seen.size() - 1);
    };
    lsxcommon::GenerationStream gs;
    gs.on_token = &cb;
    gs.Emit(7u, 0u);
    gs.Emit(8u, 1u);
    REQUIRE(seen.size() == 2u);
    CHECK(seen[0] == 7u);
    CHECK(seen[1] == 8u);
}

TEST_CASE("generation stream: Aborted reflects the flag", "[token_stream][fast]") {
    std::atomic<bool> abort{false};
    lsxcommon::GenerationStream gs;
    gs.abort_flag = &abort;
    CHECK_FALSE(gs.Aborted());
    abort.store(true);
    CHECK(gs.Aborted());
}
