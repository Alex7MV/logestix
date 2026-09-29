// TDD — InferenceEngine singleton + streaming TokenChunk/TokenCallback + abort
// intercept. Host-only: no CUDA, no real model, no Arrow. Exercises the
// public model.h / model_engine.h surface exactly as .cu-less host code would.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "lsxcommon/model.h"
#include "lsxcommon/model_base.h"
#include "lsxcommon/model_engine.h"

namespace {

// Minimal IModel fake used to verify InferenceEngine dispatch semantics.
class FakeEngineModel : public lsxcommon::ModelBase {
public:
    FakeEngineModel() : ModelBase(lsxcommon::ModelInitConfig{}) {}

    bool Load() override
    {
        loaded_ = true;
        return true;
    }
    // Unload is the canonical ModelBase template method (final): clears
    // loaded_ without an override.
    lsxcommon::gpu::InferenceResult Infer(const lsxcommon::ModelRequest &req) override
    {
        infer_calls++;
        infer_last_req_ = req;
        lsxcommon::gpu::InferenceResult r;
        r.ok = true;
        r.num_tokens = 2;
        r.gen_tokens = {101, 202};
        // Every production model streams its tokens through the request's
        // GenerationStream as they are decoded (see IModel::Infer); mirror that
        // here so the engine's own streaming behavior is exercised faithfully.
        for (std::size_t i = 0; i < r.gen_tokens.size(); ++i) {
            if (!req.token_callback) break;
            lsxcommon::TokenChunk c;
            c.token_id = static_cast<std::uint32_t>(r.gen_tokens[i]);
            c.step_t = static_cast<std::uint32_t>(i);
            req.token_callback(c);
        }
        return r;
    }
    lsxcommon::gpu::InferenceResult SelfCheck(const lsxcommon::ModelRequest &req) override
    {
        return Infer(req);
    }
    lsxcommon::SafetensorsModel Inspect() override { return {}; }
    const std::string &Type() const override { return type_; }

    std::string type_ = "fake-engine";
    int infer_calls = 0;
    lsxcommon::ModelRequest infer_last_req_;
};

} // namespace

// ── TokenChunk data-layout contract (0ms / zero-copy transmission) ─────────
TEST_CASE("model_engine: TokenChunk is flat, trivially copyable, defaults", "[model_engine]")
{
    static_assert(std::is_trivially_copyable_v<lsxcommon::TokenChunk>);
    static_assert(std::is_standard_layout_v<lsxcommon::TokenChunk>);

    lsxcommon::TokenChunk c;
    REQUIRE(c.token_id == 0);
    REQUIRE(c.step_t == 0);
    REQUIRE(c.logit_maxv == 0.0f);
    REQUIRE_FALSE(c.is_special);

    c.token_id = 42;
    c.step_t = 3;
    c.logit_maxv = -0.5f;
    c.is_special = true;
    lsxcommon::TokenChunk cp = c;  // value-copy stays legal (flat POD payload)
    REQUIRE(cp.token_id == 42);
    REQUIRE(cp.step_t == 3);
    REQUIRE(cp.logit_maxv == -0.5f);
    REQUIRE(cp.is_special);
}

TEST_CASE("model_engine: TokenCallback receives chunks by const reference", "[model_engine]")
{
    lsxcommon::TokenCallback cb = [](const lsxcommon::TokenChunk &) {};
    REQUIRE(static_cast<bool>(cb));

    int seen = 0;
    lsxcommon::TokenCallback collector = [&seen](const lsxcommon::TokenChunk &c) {
        seen += static_cast<int>(c.token_id);
    };
    lsxcommon::TokenChunk c;
    c.token_id = 7;
    collector(c);
    collector(c);
    REQUIRE(seen == 14);
}

// ── ModelRequest streaming hooks ──────────────────────────────────────────
TEST_CASE("model_engine: ModelRequest carries null hook plumbing by default", "[model_engine]")
{
    lsxcommon::ModelRequest req;
    REQUIRE_FALSE(static_cast<bool>(req.token_callback));
    REQUIRE(req.abort_flag == nullptr);
}

TEST_CASE("model_engine: ModelRequest callback+abort survive a request copy", "[model_engine]")
{
    lsxcommon::TokenCallback cb = [](const lsxcommon::TokenChunk &) {};
    std::atomic<bool> cancel{false};
    lsxcommon::ModelRequest req;
    req.token_callback = cb;
    req.abort_flag = &cancel;

    lsxcommon::ModelRequest cp = req;
    REQUIRE(static_cast<bool>(cp.token_callback));
    REQUIRE(cp.abort_flag == &cancel);
}

// ── Strict Meyers Singleton ────────────────────────────────────────────────
TEST_CASE("model_engine: Instance() returns one address (Meyers singleton)", "[model_engine]")
{
    lsxcommon::InferenceEngine &a = lsxcommon::InferenceEngine::Instance();
    lsxcommon::InferenceEngine &b = lsxcommon::InferenceEngine::Instance();
    REQUIRE(&a == &b);
}

TEST_CASE("model_engine: InferenceEngine is non-copyable, non-movable", "[model_engine]")
{
    static_assert(!std::is_copy_constructible_v<lsxcommon::InferenceEngine>);
    static_assert(!std::is_copy_assignable_v<lsxcommon::InferenceEngine>);
    static_assert(!std::is_move_constructible_v<lsxcommon::InferenceEngine>);
    static_assert(!std::is_move_assignable_v<lsxcommon::InferenceEngine>);
}

TEST_CASE("model_engine: ModelBase is non-cloneable (no illegal weight dup)", "[model_engine]")
{
    static_assert(!std::is_copy_constructible_v<lsxcommon::ModelBase>);
    static_assert(!std::is_copy_assignable_v<lsxcommon::ModelBase>);
    static_assert(!std::is_move_constructible_v<lsxcommon::ModelBase>);
    static_assert(!std::is_move_assignable_v<lsxcommon::ModelBase>);
    static_assert(!std::is_constructible_v<lsxcommon::ModelBase, const lsxcommon::ModelBase &>);
}

// ── Lifecycle: Initialize / Shutdown ──────────────────────────────────────
TEST_CASE("model_engine: InitializeEngine toggles ready state idempotently", "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());
    REQUIRE(eng.InitializeEngine());
    REQUIRE(eng.IsInitialized());
    REQUIRE(eng.InitializeEngine());      // second init stays ok
    REQUIRE(eng.IsInitialized());
    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());
    eng.ShutdownEngine();                 // double shutdown is a no-op
    REQUIRE_FALSE(eng.IsInitialized());
}

TEST_CASE("model_engine: Run before Initialize returns not-initialized", "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    eng.ShutdownEngine();

    FakeEngineModel m;
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2};
    lsxcommon::gpu::InferenceResult r = eng.Run(m, req);
    REQUIRE_FALSE(r.ok);
    REQUIRE_FALSE(r.error_msg.empty());
    REQUIRE(m.infer_calls == 0);
}

// ── Atomic Interruption Intercept (abort_flag) ────────────────────────────
TEST_CASE("model_engine: Run short-circuits when abort flag is already set", "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    REQUIRE(eng.InitializeEngine());

    FakeEngineModel m;
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2};
    std::atomic<bool> cancel{true};       // aborted before dispatch
    req.abort_flag = &cancel;

    lsxcommon::gpu::InferenceResult r = eng.Run(m, req);
    REQUIRE_FALSE(r.ok);
    REQUIRE(m.infer_calls == 0);          // never dispatched to the model
}

TEST_CASE("model_engine: Run forwards request + streams when abort is clear", "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    REQUIRE(eng.InitializeEngine());

    FakeEngineModel m;
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2};
    std::atomic<bool> cancel{false};
    req.abort_flag = &cancel;

    int chunks_seen = 0;
    lsxcommon::ModelRequest req2 = req;
    req2.token_callback = [&](const lsxcommon::TokenChunk &) { chunks_seen++; };

    lsxcommon::gpu::InferenceResult r = eng.Run(m, req2);
    REQUIRE(r.ok);
    REQUIRE(m.infer_calls == 1);
    REQUIRE(m.infer_last_req_.abort_flag == &cancel);
    REQUIRE(static_cast<bool>(m.infer_last_req_.token_callback));
    REQUIRE(chunks_seen == 2);
}

TEST_CASE("model_engine: Run does not duplicate tokens already streamed by the "
          "model",
          "[model_engine]")
{
    // IModel::Infer owns the streaming contract: it emits one TokenChunk per
    // decoded token through req.token_callback (every family wires a
    // GenerationStream). The engine facade must therefore forward the callback,
    // never replay the stream a second time after Infer returns.
    auto &eng = lsxcommon::InferenceEngine::Instance();
    REQUIRE(eng.InitializeEngine());

    FakeEngineModel m;
    lsxcommon::ModelRequest req;
    req.input_ids = {1, 2};
    std::vector<std::uint32_t> seen;
    req.token_callback = [&](const lsxcommon::TokenChunk &c) {
        seen.push_back(c.token_id);
    };

    lsxcommon::gpu::InferenceResult r = eng.Run(m, req);
    REQUIRE(r.ok);
    REQUIRE(m.infer_calls == 1);
    REQUIRE(seen.size() == r.gen_tokens.size());
    REQUIRE(seen == std::vector<std::uint32_t>{101u, 202u});
}

// ── RAII Session guard ────────────────────────────────────────────────────
TEST_CASE("model_engine: nested Session guards keep the engine alive until the "
          "outermost exits",
          "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());

    {
        lsxcommon::InferenceEngine::Session outer;
        REQUIRE(eng.IsInitialized());
        {
            lsxcommon::InferenceEngine::Session inner;
            REQUIRE(eng.IsInitialized());
        }
        // The inner guard must not tear the engine down under the outer one.
        REQUIRE(eng.IsInitialized());
    }
    REQUIRE_FALSE(eng.IsInitialized());
}

TEST_CASE("model_engine: Session guard initializes and shutdowns on scope exit", "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());

    {
        lsxcommon::InferenceEngine::Session session;
        REQUIRE(eng.IsInitialized());
    }
    REQUIRE_FALSE(eng.IsInitialized());
}

// A direct InitializeEngine() is a claim the Session counter does not know
// about. When a Session is built on top of that live engine, its release must
// not tear the engine down under the direct caller: the direct owner still
// expects IsInitialized() == true until it calls ShutdownEngine() itself.
TEST_CASE("model_engine: a Session over a live direct-init engine does not shut "
          "it down on release",
          "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());

    REQUIRE(eng.InitializeEngine());   // direct, no Session guard
    REQUIRE(eng.IsInitialized());
    {
        lsxcommon::InferenceEngine::Session session;
        REQUIRE(eng.IsInitialized());
    }
    // The direct caller still owns the engine.
    REQUIRE(eng.IsInitialized());

    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());
}

// Same contract with nesting: no Session level may shut down an engine that a
// direct InitializeEngine() brought up.
TEST_CASE("model_engine: nested Sessions over a live direct-init engine do not "
          "shut it down",
          "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());

    REQUIRE(eng.InitializeEngine());
    {
        lsxcommon::InferenceEngine::Session outer;
        {
            lsxcommon::InferenceEngine::Session inner;
            REQUIRE(eng.IsInitialized());
        }
        REQUIRE(eng.IsInitialized());
    }
    REQUIRE(eng.IsInitialized());

    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());
}

// The direct-owner exemption must not leak into the pure-Session path: a
// Session that itself brought the engine up still shuts it down when it is the
// last holder. (Regression anchor for the two cases above.)
TEST_CASE("model_engine: a Session that owns the engine still shuts it down", "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    eng.ShutdownEngine();
    REQUIRE_FALSE(eng.IsInitialized());

    {
        lsxcommon::InferenceEngine::Session session;
        REQUIRE(eng.IsInitialized());
    }
    REQUIRE_FALSE(eng.IsInitialized());
}

// A direct ShutdownEngine() clears the direct claim unconditionally, so a
// later Session must not inherit a stale exemption.
TEST_CASE("model_engine: a direct ShutdownEngine clears the direct claim", "[model_engine]")
{
    auto &eng = lsxcommon::InferenceEngine::Instance();
    eng.ShutdownEngine();
    REQUIRE(eng.InitializeEngine());
    REQUIRE(eng.IsInitialized());

    eng.ShutdownEngine();               // direct owner releases
    REQUIRE_FALSE(eng.IsInitialized());

    {
        lsxcommon::InferenceEngine::Session session;
        REQUIRE(eng.IsInitialized());
    }
    REQUIRE_FALSE(eng.IsInitialized());
}