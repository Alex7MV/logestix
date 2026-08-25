// Тест Gemma-4 chat-template framing (host-only, no CUDA, no real model).
// Сентинелы сверены с реальным tokenizer.json чекпоинта:
//   <|turn>=105, <turn|>=106, "\n"=107 (vocab), "user"/"model" — одиночные ключи vocab.
//
// Фрейминг принадлежит модели: GemmaBaseModel::BuildPromptInputIds ->
// lsxcommon::gemma::BuildPromptInputIds (BOS-led шаблон, leading_bos=true).
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/prompt_framing.h>
#include <lsxcommon/tokenizer.h>

namespace fs = std::filesystem;

namespace {

struct DbFiles {
    fs::path pinned;
    lsxcommon::ArrowTokenizerDb db;
};

DbFiles BuildDb(const std::string &pinned_suffix, const std::string &tok_json)
{
    DbFiles out;
    // pinned_suffix must be a bare name: path::append() REPLACES the left side
    // when the right side is (or contains) an absolute path, so concatenating
    // a full json path here produced an invalid "logestix_gpf_C:\..." name.
    out.pinned = fs::temp_directory_path() / ("logestix_gpf_" + pinned_suffix);
    fs::remove_all(out.pinned);
    fs::create_directories(out.pinned);
    auto st = lsxcommon::BuildTokenizerArrowFiles(tok_json, out.pinned);
    if (!st.ok()) FAIL(st.ToString());
    
    auto v = arrow::io::MemoryMappedFile::Open(
        (out.pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(v.ok());
    auto m = arrow::io::MemoryMappedFile::Open(
        (out.pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(m.ok());
    REQUIRE(lsxcommon::load_tokenizer_metadata(*v, *m, out.db).ok());
    return out;
}

// Gemma-framed tokenizer fixture: synthetic_tok.json + turn sentinels as
// added special tokens and an explicit bos_token. Returns the temp json path.
std::string WriteGemmaTokJson()
{
    std::ifstream src(TOKENIZER_TEST_DATA_DIR "/synthetic_tok.json");
    REQUIRE(src.good());
    std::string json((std::istreambuf_iterator<char>(src)),
                     std::istreambuf_iterator<char>());
    // synthetic_tok.json already carries <|bos|>/<|eos|>/<|pad|> added tokens
    // with ids 0..2; declare <|bos|> the bos_token and append the turn
    // sentinels ("user"/"model" exist in every BPE vocab via single bytes).
    const size_t at = json.find("\"added_tokens\"");
    REQUIRE(at != std::string::npos);
    const size_t bracket = json.find('[', at);
    REQUIRE(bracket != std::string::npos);
    const size_t arr_end = json.find(']', bracket);
    REQUIRE(arr_end != std::string::npos);
    static constexpr const char *kTurnTokens =
        R"({"id": 268, "content": "<|turn>", "single_word": false, )"
        R"("lstrip": false, "rstrip": false, "normalized": false, )"
        R"("special": true}, {"id": 269, "content": "<turn|>", )"
        R"("single_word": false, "lstrip": false, "rstrip": false, )"
        R"("normalized": false, "special": true}, )"
        R"({"id": 270, "content": "model", "single_word": false, )"
        R"("lstrip": false, "rstrip": false, "normalized": false, )"
        R"("special": true}, {"id": 271, "content": "user", )"
        R"("single_word": false, "lstrip": false, "rstrip": false, )"
        R"("normalized": false, "special": true}, {"id": 272, )"
        R"("content": "\n", "single_word": false, "lstrip": false, )"
        R"("rstrip": false, "normalized": false, "special": true}, )"
        R"({"id": 273, "content": "<|channel>", "single_word": false, )"
        R"("lstrip": false, "rstrip": false, "normalized": false, )"
        R"("special": true}, {"id": 274, "content": "thought", )"
        R"("single_word": false, "lstrip": false, "rstrip": false, )"
        R"("normalized": false, "special": true}, {"id": 275, )"
        R"("content": "<channel|>", "single_word": false, "lstrip": false, )"
        R"("rstrip": false, "normalized": false, "special": true})";
    // Append AFTER the existing sorted entries (ids must not collide with
    // the BPE vocab ranks).
    json.insert(arr_end, "," + std::string(kTurnTokens));
    // Explicit bos_token so the tokenizer db carries bos_id >= 0.
    const size_t root = json.find('{');
    REQUIRE(root != std::string::npos);
    json.insert(root + 1, "\"bos_token\": {\"content\": \"<|bos|>\"},");
    const fs::path p = fs::temp_directory_path() / "gemma4_framing_tok.json";
    std::ofstream f(p);
    REQUIRE(f.good());
    f << json;
    return p.string();
}

} // namespace

TEST_CASE("gemma framing descriptor is registered", "[gemma][framing]") {
    using lsxcommon::prompt::FramingDescriptor;
    const FramingDescriptor &f = lsxcommon::gemma::kFraming;
    // <bos><|turn>user\n ... <turn|>\n<|turn>model\n
    REQUIRE(f.leading_count == 3);
    // Trailing includes the EMPTY thinking channel (enable_thinking=false).
    REQUIRE(f.trailing_count == 9);
    REQUIRE(f.leading[0] == "<|turn>");
    REQUIRE(f.leading[1] == "user");
    REQUIRE(f.leading[2] == "\n");
    REQUIRE(f.trailing[0] == "<turn|>");
    REQUIRE(f.trailing[1] == "\n");
    REQUIRE(f.trailing[2] == "<|turn>");
    REQUIRE(f.trailing[3] == "model");
    REQUIRE(f.trailing[4] == "\n");
    REQUIRE(f.trailing[5] == "<|channel>");
    REQUIRE(f.trailing[6] == "thought");
    REQUIRE(f.trailing[7] == "\n");
    REQUIRE(f.trailing[8] == "<channel|>");
    // Gemma templates start with {{ bos_token }} -> BOS-led framing.
    REQUIRE(f.leading_bos);
}

TEST_CASE("gemma framing places a single leading BOS outside the turn frame",
          "[gemma][framing]")
{
    DbFiles f = BuildDb("gemma4_framing", WriteGemmaTokJson());

    const int32_t bos = f.db.bos_id;
    REQUIRE(bos >= 0);
    const auto lookup = [&](const char *s) {
        return lsxcommon::arrow_tokenizer_lookup(f.db, s, std::strlen(s));
    };

    std::vector<int32_t> ids;
    REQUIRE(lsxcommon::gemma::BuildPromptInputIds(
        f.db, "The capital of France is", ids));

    // Canonical layout: [bos][<|turn>][user][\n]...[\n][<turn|>][\n][<|turn>][model][\n]
    REQUIRE(ids.size() > 3 + 5 + 9);
    REQUIRE(ids.front() == bos);          // exactly one leading BOS...
    REQUIRE(ids[1] == lookup("<|turn>"));
    REQUIRE(ids[2] == lookup("user"));
    REQUIRE(ids[3] == lookup("\n"));
    // ...and no stray BOS inside the framed body.
    for (size_t i = 1; i + 5 < ids.size(); ++i)
        CHECK(ids[i] != bos);

    f.db = lsxcommon::ArrowTokenizerDb{};
    fs::remove_all(f.pinned);
}

TEST_CASE("glm framing unaffected by gemma entry", "[gemma][framing]") {
    // Framing is model-owned: GLM keeps its own BOS-free descriptor.
    REQUIRE(lsxcommon::glm::kFraming.leading_count == 3);
    REQUIRE(lsxcommon::glm::kFraming.leading[0] == "[gMASK]");
    REQUIRE_FALSE(lsxcommon::glm::kFraming.leading_bos);
}
