#include <catch2/catch_all.hpp>

#include <lsxcommon/qwen_dense_convert.h>
#include <lsxcommon/safetensors.h>

#include <spdlog/spdlog.h>

#include "test_qwen_dense_fixture.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace fs = std::filesystem;

struct QwenIntegrityFixture {
    fs::path tmp;
    fs::path tok_src;

    static std::mutex& cfg_mu()
    {
        static std::mutex m;
        return m;
    }

    QwenIntegrityFixture()
    {
        tmp = fs::temp_directory_path() / "qwen_integrity_test";
        fs::remove_all(tmp);
        fs::create_directories(tmp);
        tok_src = TOKENIZER_TEST_DATA_DIR;
    }

    ~QwenIntegrityFixture() { fs::remove_all(tmp); }

    // Build a synthetic Qwen model dir; any tensor whose name contains a
    // substring listed in `drop_names` is omitted from the safetensors file.
    void build(bool with_tok = true,
               const std::vector<std::string> &drop_names = {})
    {
        std::lock_guard<std::mutex> lock(cfg_mu());  // TestCfg is a shared global

        auto tensors = qwentest::BuildSyntheticQwenModel();
        std::vector<qwentest::FakeTensor> keep;
        for (auto &t : tensors) {
            bool drop = false;
            for (auto &n : drop_names)
                if (t.name.find(n) != std::string::npos) { drop = true; break; }
            if (!drop) keep.push_back(t);
        }
        qwentest::WriteSafetensors(tmp, "model.safetensors", keep);
        qwentest::WriteConfig(tmp);
        if (with_tok) {
            fs::copy_file(tok_src / "synthetic_tok.json",
                          tmp / "tokenizer.json");
        }
    }
};

static std::string read_file(const fs::path &p)
{
    std::ifstream in(p, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
    return s;
}

TEST_CASE("Qwen: happy path converts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true);

    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    INFO("ConvertQwenModel error: " << st.ToString());
    REQUIRE(st.ok());
}

TEST_CASE("Qwen: missing config.json aborts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true);
    fs::remove(f.tmp / "config.json");

    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    INFO("ConvertQwenModel error: " << st.ToString());
    CHECK(!st.ok());
    CHECK(st.ToString().find("config.json") != std::string::npos);
}

TEST_CASE("Qwen: malformed config.json aborts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true);
    {
        std::ofstream of(f.tmp / "config.json");
        of << "not json";
    }
    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    CHECK(!st.ok());
}

TEST_CASE("Qwen: missing required tensor aborts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true, {"layers.0.mlp.down_proj.weight"});
    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    INFO("ConvertQwenModel error: " << st.ToString());
    CHECK(!st.ok());
    CHECK(st.ToString().find("down_proj") != std::string::npos);
}

TEST_CASE("Qwen: layer hole aborts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true, {"layers.2."});  // drop the whole layer 2
    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    INFO("ConvertQwenModel error: " << st.ToString());
    CHECK(!st.ok());
    CHECK(st.ToString().find("missing tensors") != std::string::npos);
}

TEST_CASE("Qwen: config geometry mismatch aborts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true);
    // Rewrite config with a wrong hidden_size: WriteConfig writes
    // "\"hidden_size\":64" for the default 64-dim geometry.
    std::string content = read_file(f.tmp / "config.json");
    auto pos = content.find("\"hidden_size\":64");
    REQUIRE(pos != std::string::npos);
    content.replace(pos, 16, "\"hidden_size\":128");
    {
        std::ofstream out(f.tmp / "config.json");
        out << content;
    }
    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    INFO("ConvertQwenModel error: " << st.ToString());
    CHECK(!st.ok());
    CHECK(st.ToString().find("geometry mismatch") != std::string::npos);
}

TEST_CASE("Qwen: missing tokenizer.json aborts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(false);  // no tokenizer

    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    INFO("ConvertQwenModel error: " << st.ToString());
    CHECK(!st.ok());
    CHECK(st.ToString().find("tokenizer.json") != std::string::npos);
}

TEST_CASE("Qwen: broken tokenizer json aborts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true);
    {
        std::ofstream of(f.tmp / "tokenizer.json");
        of << "not json";
    }
    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    CHECK(!st.ok());
}

TEST_CASE("Qwen: tokenizer without bos special still converts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true);
    // Qwen3.5-9B has no BOS token (bos_id == -1 per convention, commit df91ab9),
    // so stripping the "<|bos|>" added-token entry must NOT abort conversion.
    std::string content = read_file(f.tmp / "tokenizer.json");
    auto p0 = content.find("{\"id\":0,\"content\":\"<|bos|>\"");
    REQUIRE(p0 != std::string::npos);
    auto p1 = content.find(',', p0);
    REQUIRE(p1 != std::string::npos);
    content.erase(p0, p1 - p0 + 1);
    {
        std::ofstream out(f.tmp / "tokenizer.json");
        out << content;
    }
    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    INFO("ConvertQwenModel error: " << st.ToString());
    CHECK(st.ok());
}

TEST_CASE("Qwen: truncated safetensors aborts", "[qwen][integrity]")
{
    QwenIntegrityFixture f;
    f.build(true);
    {
        auto path = f.tmp / "model.safetensors";
        auto sz = fs::file_size(path);
        fs::resize_file(path, sz / 2);
    }
    auto st = lsxcommon::qwen::ConvertQwenModel(f.tmp.string(), f.tmp.string());
    INFO("ConvertQwenModel error: " << st.ToString());
    CHECK(!st.ok());
}
