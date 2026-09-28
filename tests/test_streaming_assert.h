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
