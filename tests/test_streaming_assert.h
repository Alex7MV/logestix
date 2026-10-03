#pragma once

// Shared per-family streaming-order assertion (Plan 2 Task 12).
//
// Runs `model.Infer` with a recording token callback and asserts the callback
// observed exactly the produced gen_tokens, in order. This is the regression
// gate for the GenerationStream threading: the callback must fire once per
// generated token, in the same order the driver appends them, and must not
// invent or drop tokens.
//
// Host header (no CUDA include); included from inference-labelled test targets
// that already load a converted model.

#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/model.h>

#include <atomic>
#include <cstdint>
#include <vector>

inline void ExpectStreamMatchesTokens(lsxcommon::IModel &model,
                                      const lsxcommon::ModelRequest &base)
{
    lsxcommon::ModelRequest req = base;
    std::vector<std::int32_t> seen;
    req.token_callback = [&](const lsxcommon::TokenChunk &c) {
        seen.push_back(static_cast<std::int32_t>(c.token_id));
    };
    const auto res = model.Infer(req);
    REQUIRE(res.ok);
    REQUIRE(seen.size() == res.gen_tokens.size());
    CHECK(seen == res.gen_tokens);
}

// Cooperative-cancel assertions (OPEN_TASKS §E1a).
//
// `lsxfabric --serve` cancels a request by setting `req.abort_flag` from the
// framer thread; the decode loop must observe it between steps. A family that
// only checks the flag before the first token still "works" for every existing
// test, so these two helpers pin the two halves that were unverified:
//
//   ExpectPreSetAbortStops   — the flag is already set when Infer starts, so
//                              the loop must not decode the full budget;
//   ExpectCallbackAbortStops — the flag is raised from the token callback
//                              mid-generation (the real --serve path), so the
//                              loop must stop shortly after, not at the end.
//
// Both assert the weaker-but-sufficient contract "strictly fewer tokens than
// the budget": the exact stopping step is not part of the API, and EOS
// truncation can make a short run legitimately finish early.

inline void ExpectPreSetAbortStops(lsxcommon::IModel &model,
                                   const lsxcommon::ModelRequest &base,
                                   std::size_t budget = 24)
{
    std::atomic<bool> flag{true};
    lsxcommon::ModelRequest req = base;
    req.max_gen_tokens = static_cast<std::int32_t>(budget);
    req.abort_flag = &flag;
    std::size_t chunks = 0;
    req.token_callback = [&](const lsxcommon::TokenChunk &) { ++chunks; };
    const auto res = model.Infer(req);
    REQUIRE(res.ok);
    CHECK(res.gen_tokens.size() < budget);
    // The stream is cut with the decode loop, not left dangling: at most one
    // chunk per produced token, and none more.
    CHECK(chunks <= res.gen_tokens.size());
}

inline void ExpectCallbackAbortStops(lsxcommon::IModel &model,
                                     const lsxcommon::ModelRequest &base,
                                     std::size_t stop_after = 2,
                                     std::size_t budget = 24)
{
    std::atomic<bool> flag{false};
    lsxcommon::ModelRequest req = base;
    req.max_gen_tokens = static_cast<std::int32_t>(budget);
    req.abort_flag = &flag;
    std::size_t chunks = 0;
    req.token_callback = [&](const lsxcommon::TokenChunk &) {
        if (++chunks >= stop_after) flag.store(true, std::memory_order_release);
    };
    const auto res = model.Infer(req);
    REQUIRE(res.ok);
    // Generation really started (otherwise this proves nothing), then stopped
    // early instead of running to the budget.
    CHECK(chunks >= 1);
    CHECK(res.gen_tokens.size() < budget);
    CHECK(chunks <= res.gen_tokens.size());
}
