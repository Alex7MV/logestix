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
#include <lsxcommon/benchmark_gpu.h>

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
    void Unload() override
    {
        tok_db_.reset();
        loaded_ = false;
    }
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

TEST_CASE("model_runtime: IModel::Benchmark default returns ok=false", "[model_runtime]")
{
    // The shared test FakeModel does NOT override Benchmark -> default applies.
    FakeModel m("glm-5.2");
    auto r = m.Benchmark();
    REQUIRE_FALSE(r.ok);
}

TEST_CASE("model_runtime: RunBenchmark shim delegates to IModel::Benchmark", "[model_runtime]")
{
    // No checkpoint at this path -> factory returns nullptr -> shim reports
    // ok=false but still fills num_tokens from its argument (fill-before-dispatch).
    auto r = lsxcommon::gpu::RunBenchmark(
        (fs::temp_directory_path() / "no_such_model_xyz").string(), 64);
    REQUIRE_FALSE(r.ok);
    REQUIRE(r.num_tokens == 64);
    REQUIRE(r.resident_bytes == 0);
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

// ── FillInferenceResult / FillBenchmarkResult (unified result mapping) ──────

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

TEST_CASE("FillBenchmarkResult computes tokens-per-second", "[model_runtime]")
{
    lsxcommon::gpu::BenchmarkResult r;
    lsxcommon::runtime::FillBenchmarkResult(r, /*total_ms=*/200.0,
                                            /*num_tokens=*/100,
                                            /*num_layers=*/43,
                                            /*resident_bytes=*/9999);
    CHECK(r.ok);
    CHECK(r.num_tokens == 100);
    CHECK(r.num_layers == 43);
    CHECK(r.total_time_ms == 200.0);
    CHECK(r.tokens_per_sec == Approx(500.0));
    CHECK(r.resident_bytes == 9999);
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
    void Unload() override {}
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
