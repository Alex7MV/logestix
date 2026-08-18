// TDD — ModelBase (host-only, no CUDA, no real model).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <filesystem>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/model_base.h>
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
    explicit FakeModel(std::string type) : type_(std::move(type)) {}

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
