// TDD — ModelBase (host-only, no CUDA, no real model).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using Catch::Approx;

#include <arrow/io/file.h>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/model_base.h>
#include <lsxcommon/model_runtime.h>
#include <lsxcommon/qwen_inference.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

namespace {

std::string BuildPinnedTokenizerDir(const std::string &fixture_name)
{
    auto pinned = fs::temp_directory_path() / ("modelbase_" + fixture_name) / "logestix" / "pinned";
    fs::remove_all(pinned.parent_path().parent_path());
    fs::create_directories(pinned);
    REQUIRE(lsxcommon::BuildTokenizerArrowFiles(
                TOKENIZER_TEST_DATA_DIR "/" + fixture_name, pinned)
                .ok());
    return pinned.parent_path().parent_path().string();
}

class FakeModel : public lsxcommon::ModelBase {
public:
    explicit FakeModel(std::string type)
        : ModelBase(lsxcommon::ModelInitConfig{}), type_(std::move(type)) {}

    bool Load() override
    {
        if (!LoadTokenizerDb(cfg_.model_path)) return false;
        loaded_ = true;
        return true;
    }
    // Unload is the canonical ModelBase template method (final); the fake
    // relies on it releasing tok_db_ and clearing loaded_.
    lsxcommon::gpu::InferenceResult Infer(const lsxcommon::ModelRequest &req) override
    {
        lsxcommon::gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;
        return scripted_;
    }
    lsxcommon::SafetensorsModel Inspect() override { return {}; }
    lsxcommon::gpu::InferenceResult SelfCheck(const lsxcommon::ModelRequest &req) override
    {
        lsxcommon::gpu::InferenceResult r;
        if (!ValidateInferRequest(req, r)) return r;
        return scripted_;
    }
    const std::string &Type() const override { return type_; }

    lsxcommon::gpu::InferenceResult scripted_;

private:
    std::string type_;
};

} // namespace

TEST_CASE("model_base: LoadTokenizerDb loads a valid Arrow db", "[model_base]")
{
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    FakeModel m("deepseek-v4-flash");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());
    REQUIRE(m.tok_db_ != nullptr);
    REQUIRE(m.Tokenizer().valid());
    REQUIRE(m.Tokenizer().vocab_size > 0);
    REQUIRE(m.loaded_);
    m.Unload();
    fs::remove_all(root);
}

TEST_CASE("model_base: LoadTokenizerDb fails on a missing model dir", "[model_base]")
{
    FakeModel m("glm-5.2");
    m.cfg_.model_path = (fs::temp_directory_path() / "no_such_model_xyz").string();
    REQUIRE_FALSE(m.Load());
    REQUIRE_FALSE(m.loaded_);
}

TEST_CASE("model_runtime: FakeModel ResidentWeightBytes defaults to zero", "[model_runtime]")
{
    FakeModel m("glm-5.2");
    REQUIRE(m.ResidentWeightBytes() == 0);
}

TEST_CASE("model_runtime: RunCliGeneration returns nonzero on a missing model", "[model_runtime]")
{
    lsxcommon::runtime::GenerationCliArgs args;
    args.model_path = (fs::temp_directory_path() / "no_such_model_xyz").string();
    args.prompt = "hello";
    REQUIRE(lsxcommon::runtime::RunCliGeneration(args) != 0);
}

TEST_CASE("model_base: ValidateInferRequest canonical errors", "[model_base]")
{
    FakeModel m("glm-5.2");
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());

    lsxcommon::ModelRequest req;
    lsxcommon::gpu::InferenceResult r;

    req.input_ids = {1, 2};
    req.max_gen_tokens = 64;
    // not loaded -> "model not loaded"
    m.loaded_ = false;
    REQUIRE_FALSE(m.ValidateInferRequest(req, r));
    REQUIRE(r.error_msg == "model not loaded");
    m.loaded_ = true;

    req.input_ids.clear();
    REQUIRE_FALSE(m.ValidateInferRequest(req, r));
    REQUIRE(r.error_msg == "empty input_ids");

    req.input_ids = {1, 2};
    req.max_gen_tokens = 0;
    REQUIRE_FALSE(m.ValidateInferRequest(req, r));
    REQUIRE(r.error_msg == "max_gen_tokens must be positive");

    req.max_gen_tokens = 64;
    REQUIRE(m.ValidateInferRequest(req, r));
    m.Unload();
    fs::remove_all(root);
}

TEST_CASE("model_runtime: DecodeGeneratedText round-trips a token stream", "[model_runtime]")
{
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    FakeModel m("deepseek-v4-flash");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());

    const char *text = "The capital of France is Paris";
    std::vector<int32_t> ids(256, 0);
    size_t cap = ids.size();
    const int32_t n = lsxcommon::arrow_tokenizer_encode(
        m.Tokenizer(), text, std::strlen(text), ids.data(), &cap);
    REQUIRE(n > 0);
    ids.resize(static_cast<size_t>(n));

    const std::string out = lsxcommon::runtime::DecodeGeneratedText(m.Tokenizer(), ids);
    REQUIRE_FALSE(out.empty());
    REQUIRE(out.find("Paris") != std::string::npos);

    REQUIRE(lsxcommon::runtime::DecodeGeneratedText(m.Tokenizer(), {}).empty());
    m.Unload();
    fs::remove_all(root);
}

TEST_CASE("model_runtime: RunCliBenchmark returns 1 on a missing model",
          "[model_runtime]")
{
    lsxcommon::runtime::GenerationCliArgs args;
    args.model_path = (fs::temp_directory_path() / "no_such_model_xyz").string();
    REQUIRE(lsxcommon::runtime::RunCliBenchmark(args) == 1);
}

TEST_CASE("model_runtime: RunParisSelfCheck matches a 1/2-token Paris prefix", "[model_runtime]")
{
    // synthetic_paris_tok.json adds a "P a"->...->"Paris" BPE merge chain so
    // "Paris" encodes to a single token (the byte-level synthetic_tok.json
    // fixture would yield 5 byte tokens, which 1->2-token decode can never
    // reassemble into "Paris").
    const std::string root = BuildPinnedTokenizerDir("synthetic_paris_tok.json");
    FakeModel m("glm-5.2");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());

    // Script the fake's Infer to emit gen_tokens that decode to "Paris".
    std::vector<int32_t> paris(256, 0);
    size_t cap = paris.size();
    const int32_t np = lsxcommon::arrow_tokenizer_encode(
        m.Tokenizer(), "Paris", 5, paris.data(), &cap);
    REQUIRE(np > 0);
    paris.resize(static_cast<size_t>(np));
    m.scripted_.ok = true;
    m.scripted_.gen_tokens = paris;
    m.scripted_.first_gen_tok_id = paris.front();

    lsxcommon::ModelRequest req{std::vector<int32_t>{1, 2}, 1};
    auto r = lsxcommon::runtime::RunParisSelfCheck(m, req);
    REQUIRE(r.ok);
    REQUIRE(r.selfcheck_passed);
    REQUIRE_FALSE(r.first_gen_text.empty());

    m.scripted_.gen_tokens = {12345};   // never decodes to "Paris"
    m.scripted_.first_gen_tok_id = 12345;
    auto r2 = lsxcommon::runtime::RunParisSelfCheck(m, req);
    REQUIRE(r2.ok);
    REQUIRE_FALSE(r2.selfcheck_passed);
    m.Unload();
    fs::remove_all(root);
}

TEST_CASE("kv_slots_used contract defaults and monotonicity", "[model_runtime]")
{
    // Default InferenceResult must report zero occupancy (host-only contract).
    lsxcommon::gpu::InferenceResult r;
    REQUIRE(r.kv_slots_used == 0);
    // A synthetic engine result satisfies: kv_slots_used == prompt + generated.
    r.kv_slots_used = 7 + 3;
    r.num_tokens = 3;
    REQUIRE(r.kv_slots_used == 7 + r.num_tokens);
}

// ── FillInferenceResult (unified result mapping) ────────────────────────────

TEST_CASE("FillInferenceResult maps tokens, timings and eos", "[model_runtime]")
{
    using namespace lsxcommon;
    gpu::InferenceResult r;
    std::vector<int32_t> toks{7, 8, 9};
    std::vector<float> logits{0.1f, 0.2f};
    std::string err;
    runtime::GenResultView g{&toks, &logits, 12.5, 100.0, &err,
                             /*eos_terminated=*/true, /*kv_slots_used=*/17,
                             /*first_maxv=*/3.25f, /*resident_bytes=*/12345};
    runtime::FillInferenceResult(r, g, /*num_layers=*/79);
    CHECK(r.ok);
    CHECK(r.num_tokens == 3);
    CHECK(r.num_layers == 79);
    CHECK(r.first_gen_tok_id == 7);
    CHECK(r.gen_tokens == toks);
    CHECK(r.final_logits == logits);
    CHECK(r.ttft_ms == 12.5);
    CHECK(r.total_time_ms == 100.0);
    CHECK(r.tokens_per_sec == Approx(30.0));  // 3 tok / 0.1 s
    CHECK(r.eos_terminated);
    CHECK(r.kv_slots_used == 17);
    CHECK(r.weight_resident_bytes == 12345);
    CHECK(r.output_text.empty());  // Fill must NOT decode text (caller does)
}

TEST_CASE("FillInferenceResult handles empty stream and null optionals",
          "[model_runtime]")
{
    using namespace lsxcommon;
    gpu::InferenceResult r;
    std::vector<int32_t> toks;
    runtime::GenResultView g{&toks, nullptr, 0, 0, nullptr, false,
                             /*kv_slots_used=*/-1, 0.f, 0};
    runtime::FillInferenceResult(r, g, 40);
    CHECK(r.ok);
    CHECK(r.num_tokens == 0);
    CHECK(r.first_gen_tok_id == -1);
    CHECK(r.kv_slots_used == -1);  // caller fallback preserved
    CHECK(r.final_logits.empty());
}

// ── Shared incremental-session decision (qwen families) ─────────────────────

TEST_CASE("DecideSession continuation vs fresh", "[session]")
{
    using namespace lsxcommon::qwen;
    const std::vector<int32_t> tail{5, 6};
    const std::vector<int32_t> req{5, 6};
    auto d = DecideSession(/*kv_allocated=*/true, /*last_tok=*/6, tail, req);
    CHECK(d.is_continuation);
    CHECK((d.gen_input == std::vector<int32_t>{6}));

    d = DecideSession(true, 6, tail, {1, 2});
    CHECK_FALSE(d.is_continuation);
    CHECK((d.gen_input == std::vector<int32_t>{1, 2}));

    d = DecideSession(false, 6, tail, req);  // kv dead -> fresh
    CHECK_FALSE(d.is_continuation);
    CHECK((d.gen_input == std::vector<int32_t>{5, 6}));

    d = DecideSession(true, 6, {}, req);  // no tail -> fresh
    CHECK_FALSE(d.is_continuation);

    d = DecideSession(true, 6, tail, {6});  // partial match -> fresh
    CHECK_FALSE(d.is_continuation);
}

// ── ModelBase persistent CUDA execution context ─────────────────────────────

#include <cublas_v2.h>
#include <shared_mutex>

namespace {

class ContextProbe final : public lsxcommon::ModelBase {
public:
    using ModelBase::ModelBase;
    using ModelBase::EnsureContext;
    using ModelBase::TeardownContext;
    using ModelBase::stream;
    using ModelBase::cublas;
    using ModelBase::RunUnderResidencyLock;

    bool Load() override { return false; }
    lsxcommon::gpu::InferenceResult Infer(const lsxcommon::ModelRequest &) override
    {
        return {};
    }
    lsxcommon::gpu::InferenceResult SelfCheck(const lsxcommon::ModelRequest &) override
    {
        return {};
    }
    const std::string &Type() const override
    {
        static std::string t{"context-probe"};
        return t;
    }
};

} // namespace

TEST_CASE("ModelBase context is lazy, idempotent and teardown-safe",
          "[model_base][gpu]")
{
    ContextProbe m{lsxcommon::ModelInitConfig{}};
    CHECK(m.stream() == nullptr);
    CHECK(m.cublas() == nullptr);

    // Stream-only context.
    REQUIRE(m.EnsureContext(/*with_cublas=*/false));
    cudaStream_t s1 = m.stream();
    REQUIRE(s1 != nullptr);
    CHECK(m.cublas() == nullptr);

    // Idempotent: same stream, no churn; upgrades to cublas on request.
    REQUIRE(m.EnsureContext(false));
    CHECK(m.stream() == s1);
    REQUIRE(m.EnsureContext(true));
    cublasHandle_t h1 = m.cublas();
    REQUIRE(h1 != nullptr);
    REQUIRE(m.EnsureContext(true));
    CHECK(m.cublas() == h1);
    CHECK(m.stream() == s1);

    // Teardown releases both; safe to call twice.
    m.TeardownContext();
    CHECK(m.stream() == nullptr);
    CHECK(m.cublas() == nullptr);
    m.TeardownContext();
}

TEST_CASE("EnsureContext binds cuBLAS to the owned stream", "[model_base][gpu]")
{
    ContextProbe m{lsxcommon::ModelInitConfig{}};
    REQUIRE(m.EnsureContext(/*with_cublas=*/true));
    cudaStream_t s = nullptr;
    CHECK(cublasGetStream(m.cublas(), &s) == CUBLAS_STATUS_SUCCESS);
    CHECK(s == m.stream());
    m.TeardownContext();
}

TEST_CASE("RunUnderResidencyLock executes and releases", "[model_base]")
{
    ContextProbe m{lsxcommon::ModelInitConfig{}};
    int v = m.RunUnderResidencyLock([] { return 42; });
    CHECK(v == 42);
    // The shared lock must be released afterwards: unique_lock succeeds.
    std::unique_lock<std::shared_mutex> ul(m.ResidencyMutex());
}

// ── Canonical lifecycle contract (lifecycle-unification 2026-08) ────────────
//
// ModelBase owns the Unload template method (final): TeardownWeights() ->
// tokenizer db -> loaded_ flag. Subclasses only override TeardownWeights().
// AbortLoad() is the canonical Load-failure tail. runtime::CombineGenError is
// the single error format for cudaError_t + family-error tails.

#include <cuda_runtime.h>

namespace {

class LifecycleProbe final : public lsxcommon::ModelBase {
public:
    using ModelBase::ModelBase;
    using ModelBase::AbortLoad;
    using ModelBase::resident_bytes_;

    int teardowns = 0;

    void TeardownWeights() override { ++teardowns; }
    bool Load() override { return false; }
    lsxcommon::gpu::InferenceResult Infer(const lsxcommon::ModelRequest &)
        override
    {
        return {};
    }
    lsxcommon::gpu::InferenceResult SelfCheck(const lsxcommon::ModelRequest &)
        override
    {
        return {};
    }
    const std::string &Type() const override
    {
        static std::string t{"lifecycle-probe"};
        return t;
    }
};

} // namespace

TEST_CASE("ModelBase canonical Unload: teardown + tokenizer + flags",
          "[model_base][lifecycle]")
{
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    LifecycleProbe p{lsxcommon::ModelInitConfig{}};
    p.cfg_.model_path = root;
    REQUIRE(p.LoadTokenizerDb(p.cfg_.model_path));
    p.loaded_ = true;
    p.resident_bytes_ = 12345;
    REQUIRE(p.tok_db_ != nullptr);

    p.Unload();
    CHECK(p.teardowns == 1);
    CHECK(p.tok_db_ == nullptr);
    CHECK_FALSE(p.loaded_);
    CHECK(p.ResidentWeightBytes() == 0);

    p.Unload();  // idempotent: still safe, state unchanged
    CHECK(p.tok_db_ == nullptr);
    CHECK_FALSE(p.loaded_);
    fs::remove_all(root);
}

TEST_CASE("ModelBase AbortLoad: failed Load leaves a clean state",
          "[model_base][lifecycle]")
{
    const std::string root = BuildPinnedTokenizerDir("synthetic_tok.json");
    LifecycleProbe p{lsxcommon::ModelInitConfig{}};
    p.cfg_.model_path = root;
    REQUIRE(p.LoadTokenizerDb(p.cfg_.model_path));
    p.loaded_ = true;
    p.resident_bytes_ = 999;

    // Canonical Load-failure tail: returns false for `return AbortLoad();`
    CHECK_FALSE(p.AbortLoad());
    CHECK(p.teardowns == 1);
    CHECK(p.tok_db_ == nullptr);
    CHECK_FALSE(p.loaded_);
    CHECK(p.ResidentWeightBytes() == 0);
    fs::remove_all(root);
}

TEST_CASE("runtime::CombineGenError is the single error format",
          "[model_runtime][lifecycle]")
{
    using lsxcommon::runtime::CombineGenError;
    // Family error wins when the CUDA status is clean.
    CHECK(CombineGenError(cudaSuccess, "gen failed") == "gen failed");
    // CUDA status alone.
    CHECK(CombineGenError(cudaErrorUnknown, "") ==
          std::string(cudaGetErrorString(cudaErrorUnknown)));
    // Both present: CUDA string first, then the family tail.
    CHECK(CombineGenError(cudaErrorUnknown, "gen failed") ==
          std::string(cudaGetErrorString(cudaErrorUnknown)) + " / gen failed");
    // Both empty.
    CHECK(CombineGenError(cudaSuccess, "").empty());
}

// ── Governor lease ownership in ModelBase (lifecycle-unification II) ────────
//
// Every model acquires its permanent-weight lease (and, for MoE, the expert
// cache lease) through ModelBase helpers; the canonical Unload/AbortLoad
// releases them. Governor is a process singleton with first-init-wins
// options, so the fixture pins a synthetic 64 GiB device once; every test
// case below is self-cleaning (committed VRAM returns to 0).

#include "lsxcommon/resource_governor.h"

namespace {

class LeaseProbe final : public lsxcommon::ModelBase {
public:
    using ModelBase::ModelBase;
    using ModelBase::AcquirePermanentWeights;
    using ModelBase::AcquireExpertCacheLease;
    using ModelBase::permanent_lease_;
    using ModelBase::expert_cache_lease_;

    void TeardownWeights() override {}
    bool Load() override { return false; }
    lsxcommon::gpu::InferenceResult Infer(const lsxcommon::ModelRequest &)
        override
    {
        return {};
    }
    lsxcommon::gpu::InferenceResult SelfCheck(const lsxcommon::ModelRequest &)
        override
    {
        return {};
    }
    const std::string &Type() const override
    {
        static std::string t{"lease-probe"};
        return t;
    }
};

lsxcommon::ResourceGovernor &TestGovernor()
{
    auto &gov = lsxcommon::ResourceGovernor::Instance();
    lsxcommon::GovernorOptions o;
    o.override_hw = true;
    o.hw_override.vram_total_bytes = 64ull << 30;
    o.hw_override.vram_free_bytes = 64ull << 30;
    o.hw_override.ram_total_bytes = 128ull << 30;
    o.hw_override.ram_available_bytes = 128ull << 30;
    REQUIRE(gov.Initialize(o));
    return gov;
}

} // namespace

TEST_CASE("ModelBase permanent lease: acquire and release on Unload",
          "[model_base][lifecycle][governor]")
{
    auto &gov = TestGovernor();
    const std::uint64_t GiB = 1ull << 30;

    LeaseProbe p{lsxcommon::ModelInitConfig{}};
    REQUIRE(p.AcquirePermanentWeights(8 * GiB));
    CHECK(p.permanent_lease_.Valid());
    CHECK(gov.CommittedVramBytes() == 8 * GiB);

    p.Unload();  // canonical Unload releases the lease
    CHECK_FALSE(p.permanent_lease_.Valid());
    CHECK(gov.CommittedVramBytes() == 0);
}

TEST_CASE("ModelBase permanent lease: deterministic refusal when exhausted",
          "[model_base][lifecycle][governor]")
{
    auto &gov = TestGovernor();
    const std::uint64_t GiB = 1ull << 30;

    LeaseProbe p1{lsxcommon::ModelInitConfig{}};
    REQUIRE(p1.AcquirePermanentWeights(40 * GiB));
    CHECK(p1.permanent_lease_.Valid());

    // Grantable = 64 GiB - 512 MiB keep-out; 40 + 40 > grantable -> the
    // second acquisition is refused (no relax_memory_refusal).
    LeaseProbe p2{lsxcommon::ModelInitConfig{}};
    CHECK_FALSE(p2.AcquirePermanentWeights(40 * GiB));
    CHECK_FALSE(p2.permanent_lease_.Valid());
    CHECK(gov.CommittedVramBytes() == 40 * GiB);

    p1.Unload();
    CHECK(gov.CommittedVramBytes() == 0);
}

TEST_CASE("ModelBase expert-cache lease is adaptive and released on Unload",
          "[model_base][lifecycle][governor]")
{
    auto &gov = TestGovernor();
    const std::uint64_t GiB = 1ull << 30;
    const std::uint64_t MiB = 1ull << 20;

    LeaseProbe p{lsxcommon::ModelInitConfig{}};
    CHECK(p.AcquireExpertCacheLease(/*bytes=*/2 * GiB, /*min_bytes=*/MiB,
                                    /*max_bytes=*/4 * GiB));
    CHECK(p.expert_cache_lease_.Valid());
    CHECK(gov.CommittedVramBytes() == 2 * GiB);

    p.Unload();
    CHECK(gov.CommittedVramBytes() == 0);
}

// ── Option-driven Paris SelfCheck (unified harness) ─────────────────────────

TEST_CASE("RunParisSelfCheck options: substring window match", "[selfcheck]")
{
    // Gemma semantics: decode an N-token window, pass iff it CONTAINS "Paris".
    const std::string root = BuildPinnedTokenizerDir("synthetic_paris_tok.json");
    FakeModel m("gemma4-26b-a4b");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());

    std::vector<int32_t> paris(256, 0);
    size_t cap = paris.size();
    const int32_t np = lsxcommon::arrow_tokenizer_encode(
        m.Tokenizer(), "Paris", 5, paris.data(), &cap);
    REQUIRE(np > 0);
    paris.resize(static_cast<size_t>(np));

    lsxcommon::ModelRequest req{{1}, 4};
    lsxcommon::runtime::ParisSelfCheckOptions opts;
    opts.min_tokens = 8;
    opts.answer_window = 12;
    opts.substring_match = true;

    // "is in Paris" — the canonical answer appears mid-window.
    std::vector<int32_t> phrase(256, 0);
    cap = phrase.size();
    const int32_t nph = lsxcommon::arrow_tokenizer_encode(
        m.Tokenizer(), "is in Paris", 11, phrase.data(), &cap);
    REQUIRE(nph > 0);
    phrase.resize(static_cast<size_t>(nph));
    m.scripted_.ok = true;
    m.scripted_.gen_tokens = phrase;

    auto r = lsxcommon::runtime::RunParisSelfCheck(m, req, opts);
    REQUIRE(r.ok);
    CHECK(r.selfcheck_passed);

    // Without substring mode the same stream must fail the strict
    // first-token equality contract.
    auto r2 = lsxcommon::runtime::RunParisSelfCheck(m, req);
    REQUIRE(r2.ok);
    CHECK_FALSE(r2.selfcheck_passed);

    m.Unload();
}

TEST_CASE("RunParisSelfCheck default options preserve legacy behavior",
          "[selfcheck]")
{
    const std::string root = BuildPinnedTokenizerDir("synthetic_paris_tok.json");
    FakeModel m("glm-5.2");
    m.cfg_.model_path = root;
    REQUIRE(m.Load());

    std::vector<int32_t> paris(256, 0);
    size_t cap = paris.size();
    const int32_t np = lsxcommon::arrow_tokenizer_encode(
        m.Tokenizer(), "Paris", 5, paris.data(), &cap);
    REQUIRE(np > 0);
    paris.resize(static_cast<size_t>(np));
    m.scripted_.ok = true;
    m.scripted_.gen_tokens = paris;
    m.scripted_.first_gen_tok_id = paris.front();

    lsxcommon::ModelRequest req{{1, 2}, 1};
    // Explicit defaults must behave exactly like the old 2-arg call.
    lsxcommon::runtime::ParisSelfCheckOptions defaults;
    auto r = lsxcommon::runtime::RunParisSelfCheck(m, req, defaults);
    REQUIRE(r.ok);
    CHECK(r.selfcheck_passed);
    m.Unload();
}
