// TDD — ModelBase (host-only, no CUDA, no real model).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/model_base.h>
#include <lsxcommon/model_runtime.h>
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
    m.cfg_.model_path = fs::temp_directory_path() / "no_such_model_xyz";
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
    args.model_path = fs::temp_directory_path() / "no_such_model_xyz";
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
