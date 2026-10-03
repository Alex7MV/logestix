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

TEST_CASE("generation stream: Emit carries the step's max logit into the chunk",
          "[token_stream][fast]") {
    // nemotron and gigachat publish a per-step max logit next to the token id;
    // every other family omits it. Both go through Emit, so the payload has to
    // reach TokenChunk verbatim and default to 0 when not supplied.
    std::vector<lsxcommon::TokenChunk> chunks;
    lsxcommon::TokenCallback cb = [&](const lsxcommon::TokenChunk& c) {
        chunks.push_back(c);
    };
    lsxcommon::GenerationStream gs;
    gs.on_token = &cb;
    gs.Emit(11u, 0u, -2.5f);
    gs.Emit(12u, 1u);
    REQUIRE(chunks.size() == 2u);
    CHECK(chunks[0].logit_maxv == -2.5f);  // negatives survive (logits, not counts)
    CHECK(chunks[1].logit_maxv == 0.0f);   // default when not supplied
    CHECK(chunks[0].token_id == 11u);
    CHECK(chunks[1].step_t == 1u);
}

TEST_CASE("generation stream: Emit with no callback and no flag is inert",
          "[token_stream][fast]") {
    // Both members are optional; a disabled stream must not touch anything.
    lsxcommon::GenerationStream gs;
    CHECK_FALSE(gs.Aborted());
    gs.Emit(1u, 0u, 1.0f);
    gs.Emit(2u, 1u);
    SUCCEED("no callback, no abort flag");
}
