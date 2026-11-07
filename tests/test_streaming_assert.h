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

// The stream contract when the family RETAINS the EOS token in gen_tokens but
// stops streaming at it. That asymmetry is deliberate and documented in the
// shared truncation helper (`MoETruncateTokensAtEos` resizes to `i + 1`, keeping
// the EOS token, while the emit loop breaks before it), so a strict
// equality check reports a "dropped token" for a family that is behaving as
// designed. This variant removes one trailing EOS from the expectation and then
// requires exact equality, so invention, reordering and mid-stream drops are
// still failures.
inline void ExpectStreamIsTokensWithoutTrailingEos(
    lsxcommon::IModel &model, const lsxcommon::ModelRequest &base)
{
    lsxcommon::ModelRequest req = base;
    std::vector<std::int32_t> seen;
    req.token_callback = [&](const lsxcommon::TokenChunk &c) {
        seen.push_back(static_cast<std::int32_t>(c.token_id));
    };
    const auto res = model.Infer(req);
    REQUIRE(res.ok);
    std::vector<std::int32_t> want = res.gen_tokens;
    int32_t eos = model.Tokenizer().eos_id;
    if (!want.empty() && eos >= 0 && want.back() == eos) want.pop_back();
    INFO("streamed " << seen.size() << " of " << res.gen_tokens.size()
                     << " gen_tokens (eos=" << eos << ")");
    REQUIRE(seen == want);
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

// The whole cooperative-cancel gate, in the three runs E1a asks for: the
// pre-set poll, the mid-generation poll raised from the token callback (the
// real `--serve` path), and the control that proves the family really would
// have produced the full budget. EOS is forced off in all three (`ignore_eos`,
// which every family maps to a sentinel no token equals), because on random
// weights an EOS hit could shorten a run for a reason unrelated to the abort,
// and the "strictly fewer than budget" check would then pass for the wrong
// reason. With EOS off, the only way a run can stop early is the abort flag, so
// the control run picking up exactly `budget` tokens makes the two cancel runs
// meaningful. The control runs last so the cancel runs see a fresh session.
template <typename Model>
inline void ExpectCooperativeCancelStops(Model &model,
                                         const lsxcommon::ModelRequest &base,
                                         std::size_t budget = 24,
                                         std::size_t stop_after = 2)
{
    lsxcommon::ModelRequest b = base;
    b.max_gen_tokens = static_cast<std::int32_t>(budget);
    b.ignore_eos = true;

    {
        std::atomic<bool> flag{true};
        lsxcommon::ModelRequest req = b;
        req.abort_flag = &flag;
        std::size_t chunks = 0;
        req.token_callback = [&](const lsxcommon::TokenChunk &) { ++chunks; };
        const auto res = model.Infer(req);
        INFO("pre-set abort: " << res.error_msg);
        REQUIRE(res.ok);
        CHECK(res.gen_tokens.size() < budget);
        CHECK(chunks <= res.gen_tokens.size());
    }
    {
        std::atomic<bool> flag{false};
        lsxcommon::ModelRequest req = b;
        req.abort_flag = &flag;
        std::size_t chunks = 0;
        req.token_callback = [&](const lsxcommon::TokenChunk &) {
            if (++chunks >= stop_after)
                flag.store(true, std::memory_order_release);
        };
        const auto res = model.Infer(req);
        INFO("mid-generation abort: " << res.error_msg);
        REQUIRE(res.ok);
        CHECK(chunks >= 1);
        CHECK(res.gen_tokens.size() < budget);
        CHECK(chunks <= res.gen_tokens.size());
    }
    {
        std::size_t chunks = 0;
        lsxcommon::ModelRequest req = b;
        req.token_callback = [&](const lsxcommon::TokenChunk &) { ++chunks; };
        const auto res = model.Infer(req);
        INFO("control: " << res.error_msg);
        REQUIRE(res.ok);
        CHECK(res.gen_tokens.size() == budget);
        CHECK(chunks == res.gen_tokens.size());
    }
}
