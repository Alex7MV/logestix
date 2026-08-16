#include <catch2/catch_all.hpp>

#include <lsxcommon/model_integrity.h>
#include <lsxcommon/safetensors.h>
#include <lsxcommon/schema.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using namespace lsxcommon::integrity;

// ── CheckGeometryFields ──

TEST_CASE("CheckGeometryFields passes on matching values", "[integrity]")
{
    std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> fields = {
        {"hidden_size", {4096, 4096}},
        {"num_layers",  {32, 32}},
    };
    CHECK(CheckGeometryFields(fields).ok());
}

TEST_CASE("CheckGeometryFields fails on mismatch", "[integrity]")
{
    std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> fields = {
        {"hidden_size", {4096, 2048}},
    };
    auto st = CheckGeometryFields(fields);
    CHECK(!st.ok());
    CHECK(st.ToString().find("hidden_size") != std::string::npos);
    CHECK(st.ToString().find("4096") != std::string::npos);
    CHECK(st.ToString().find("2048") != std::string::npos);
}

TEST_CASE("CheckGeometryFields reports every mismatch", "[integrity]")
{
    std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> fields = {
        {"a", {1, 2}},
        {"b", {3, 3}},
        {"c", {5, 6}},
    };
    auto st = CheckGeometryFields(fields);
    CHECK(!st.ok());
    CHECK(st.ToString().find("a: config=1 tensor=2") != std::string::npos);
    CHECK(st.ToString().find("c: config=5 tensor=6") != std::string::npos);
}

// ── CheckTensorCoverage ──

TEST_CASE("CheckTensorCoverage passes when all expected present", "[integrity]")
{
    std::vector<std::string> expected = {"a", "b", "c"};
    std::vector<std::string> present  = {"a", "b", "c"};
    CHECK(CheckTensorCoverage(expected, present, "test").ok());
}

TEST_CASE("CheckTensorCoverage fails on missing tensor", "[integrity]")
{
    std::vector<std::string> expected = {"a", "b", "c"};
    std::vector<std::string> present  = {"a", "b"};
    auto st = CheckTensorCoverage(expected, present, "test");
    CHECK(!st.ok());
    CHECK(st.ToString().find("missing tensors") != std::string::npos);
    CHECK(st.ToString().find("test") != std::string::npos);
    CHECK(st.ToString().find(" c") != std::string::npos);
}

TEST_CASE("CheckTensorCoverage names every missing tensor", "[integrity]")
{
    std::vector<std::string> expected = {"a", "b", "c", "d"};
    std::vector<std::string> present  = {"a"};
    auto st = CheckTensorCoverage(expected, present, "x");
    CHECK(!st.ok());
    for (auto name : {"b", "c", "d"})
        CHECK(st.ToString().find(name) != std::string::npos);
}

TEST_CASE("CheckTensorCoverage warns on extra tensor, passes", "[integrity]")
{
    std::vector<std::string> expected = {"a", "b"};
    std::vector<std::string> present  = {"a", "b", "c"};
    CHECK(CheckTensorCoverage(expected, present, "test").ok());
}

TEST_CASE("CheckTensorCoverage empty expected passes", "[integrity]")
{
    std::vector<std::string> expected;
    std::vector<std::string> present = {"a"};
    CHECK(CheckTensorCoverage(expected, present, "empty").ok());
}

// ── CheckLayerContiguity ──

TEST_CASE("CheckLayerContiguity passes on full range", "[integrity]")
{
    absl::flat_hash_set<int> layers = {0, 1, 2, 3};
    CHECK(CheckLayerContiguity(4, layers).ok());
}

TEST_CASE("CheckLayerContiguity fails on hole", "[integrity]")
{
    absl::flat_hash_set<int> layers = {0, 1, 3};
    auto st = CheckLayerContiguity(4, layers);
    CHECK(!st.ok());
    CHECK(st.ToString().find("hole") != std::string::npos);
    CHECK(st.ToString().find("2") != std::string::npos);
}

TEST_CASE("CheckLayerContiguity fails on count mismatch", "[integrity]")
{
    absl::flat_hash_set<int> layers = {0, 1, 2};
    auto st = CheckLayerContiguity(4, layers);
    CHECK(!st.ok());
    CHECK(st.ToString().find("layer count mismatch") != std::string::npos);
}

TEST_CASE("CheckLayerContiguity fails on empty set", "[integrity]")
{
    absl::flat_hash_set<int> layers;
    CHECK(!CheckLayerContiguity(4, layers).ok());
}

// ── BuildAndVerifyTokenizer ──

static fs::path integrity_tok_dir()
{
    return fs::temp_directory_path() / "logestix_integrity_tok_unit";
}

static std::string read_file(const fs::path &p)
{
    std::ifstream in(p, std::ios::binary);
    std::string s((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
    return s;
}

TEST_CASE("BuildAndVerifyTokenizer: synthetic tokenizer round-trips", "[integrity][tokenizer]")
{
    auto tmp = integrity_tok_dir();
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    auto st = BuildAndVerifyTokenizer(
        TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json", tmp,
        TokenizerSpecialIds{.bos_required = true, .eos_required = true});
    INFO("BuildAndVerifyTokenizer error: " << st.ToString());
    REQUIRE(st.ok());
    REQUIRE(fs::exists(tmp / "tokenizer_vocab.arrow"));
    REQUIRE(fs::exists(tmp / "tokenizer_merges.arrow"));

    fs::remove_all(tmp);
}

TEST_CASE("BuildAndVerifyTokenizer: missing file fails", "[integrity][tokenizer]")
{
    fs::remove_all(integrity_tok_dir());
    fs::create_directories(integrity_tok_dir());

    auto st = BuildAndVerifyTokenizer(
        (integrity_tok_dir() / "nope.json").string(), integrity_tok_dir(),
        TokenizerSpecialIds{});
    CHECK(!st.ok());
    fs::remove_all(integrity_tok_dir());
}

TEST_CASE("BuildAndVerifyTokenizer: corrupt json fails", "[integrity][tokenizer]")
{
    auto tmp = integrity_tok_dir();
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    {
        std::ofstream of(tmp / "tokenizer.json");
        of << "not json";
    }

    auto st = BuildAndVerifyTokenizer(
        (tmp / "tokenizer.json").string(), tmp, TokenizerSpecialIds{});
    CHECK(!st.ok());
    fs::remove_all(tmp);
}

TEST_CASE("BuildAndVerifyTokenizer: missing required bos fails", "[integrity][tokenizer]")
{
    auto tmp = integrity_tok_dir();
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    // Copy the synthetic tokenizer, strip the "<|bos|>" added-token entry so
    // bos_id resolves to -1.
    std::string content = read_file(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json");
    auto p0 = content.find("{\"id\":0,\"content\":\"<|bos|>\"");
    REQUIRE(p0 != std::string::npos);
    auto p1 = content.find(',', p0);
    REQUIRE(p1 != std::string::npos);
    content.erase(p0, p1 - p0 + 1);

    {
        std::ofstream of(tmp / "tokenizer.json");
        of << content;
    }

    auto st = BuildAndVerifyTokenizer(
        (tmp / "tokenizer.json").string(), tmp,
        TokenizerSpecialIds{.bos_required = true, .eos_required = true});
    INFO("BuildAndVerifyTokenizer error: " << st.ToString());
    CHECK(!st.ok());
    CHECK(st.ToString().find("bos_id") != std::string::npos);
    fs::remove_all(tmp);
}