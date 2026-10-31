// Granite-Switch byte-level BPE parity against the checkpoint tokenizer.
#include <catch2/catch_test_macros.hpp>

#include <arrow/io/file.h>
#include <simdjson.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <lsxcommon/converter.h>
#include <lsxcommon/path_registry.h>
#include <lsxcommon/granite_switch_tool_format.h>
#include <lsxcommon/tokenizer.h>

#ifndef LSX_GRANITE_SWITCH_FIXTURE_DIR
#define LSX_GRANITE_SWITCH_FIXTURE_DIR "tests/fixtures/granite_switch"
#endif
#ifndef LSX_GRANITE_SWITCH_TOKENIZER_PATH
#define LSX_GRANITE_SWITCH_TOKENIZER_PATH \
    "/mnt/fast_storage/huggingface/granite-switch-4.1-30b/tokenizer.json"
#endif

namespace {

struct Case {
    std::string text;
    std::vector<int32_t> ids;
};

std::string Join(const std::vector<int32_t> &v)
{
    std::string s;
    for (auto x : v) s += std::to_string(x) + " ";
    return s;
}

std::vector<Case> LoadCases()
{
    std::ifstream f(LSX_GRANITE_SWITCH_FIXTURE_DIR "/tokenizer_parity.jsonl",
                    std::ios::binary);
    REQUIRE(f.good());
    std::string line;
    std::vector<Case> cases;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        REQUIRE(parser.parse(line).get(doc) == simdjson::SUCCESS);
        Case c;
        std::string_view text;
        REQUIRE(doc["text"].get_string().get(text) == simdjson::SUCCESS);
        c.text = std::string(text);
        for (auto e : doc["ids"].get_array()) {
            int64_t v = 0;
            REQUIRE(e.get_int64().get(v) == simdjson::SUCCESS);
            c.ids.push_back(static_cast<int32_t>(v));
        }
        cases.push_back(std::move(c));
    }
    return cases;
}

} // namespace

TEST_CASE("granite-switch tokenizer reproduces the checkpoint ids",
          "[granite_switch][tokenizer]")
{
    const char *env = std::getenv("LSX_TEST_GRANITE_SWITCH_MODEL_PATH");
    const std::filesystem::path root =
        env ? std::filesystem::path(env)
            : std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
                  "granite-switch-4.1-30b";
    const auto pinned = root / "logestix" / "pinned";
    if (!std::filesystem::exists(pinned / "tokenizer_vocab.arrow"))
        SKIP("granite-switch converted tokenizer absent");

    auto vf = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_vocab.arrow").string(),
        arrow::io::FileMode::READ);
    auto mf = arrow::io::MemoryMappedFile::Open(
        (pinned / "tokenizer_merges.arrow").string(),
        arrow::io::FileMode::READ);
    REQUIRE(vf.ok());
    REQUIRE(mf.ok());
    lsxcommon::ArrowTokenizerDb db;
    auto meta = lsxcommon::load_tokenizer_metadata(*vf, *mf, db);
    INFO(meta.ToString());
    REQUIRE(meta.ok());

    const auto cases = LoadCases();
    REQUIRE(cases.size() >= 20);
    for (const auto &c : cases) {
        std::vector<int32_t> out;
        if (c.text.find("<|") != std::string::npos ||
            c.text.find("<tool_call>") != std::string::npos ||
            c.text.find("</tool_call>") != std::string::npos) {
            // Added tokens are injected by the family bridge, never by the
            // pure-BPE encoder (the same split the model uses).
            REQUIRE(lsxcommon::AppendGraniteSwitchIds(db, c.text, out));
        } else {
            out.assign(c.text.size() * 4 + 64, 0);
            size_t cap = out.size();
            const int32_t n = lsxcommon::arrow_tokenizer_encode(
                db, c.text.data(), c.text.size(), out.data(), &cap);
            INFO("text: " << c.text);
            REQUIRE(n >= 0);
            out.resize(static_cast<size_t>(n));
            if (!out.empty() && out[0] == db.bos_id)
                out.erase(out.begin());  // golden is add_special_tokens=False
        }
        INFO("text: " << c.text);
        INFO("want: " << Join(c.ids) << "\n got: " << Join(out));
        CHECK(out == c.ids);
    }
}
