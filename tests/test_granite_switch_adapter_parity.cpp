// Granite-Switch aLoRA parity against the CPU-BF16 source oracle.
#include <catch2/catch_test_macros.hpp>

#include <lsxcommon/model.h>
#include <lsxcommon/model_factory.h>
#include <lsxcommon/chat_conversation.h>
#include <lsxcommon/path_registry.h>

#include <simdjson.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "granite_dense_switch_model.h"

#ifndef LSX_ADAPTER_GOLDEN_DIR
#define LSX_ADAPTER_GOLDEN_DIR "tests/fixtures/adapters"
#endif

namespace {

constexpr const char *kModelType = "granite-switch-4.1-30b";

struct Case {
    std::string id;
    std::string adapter;
    std::vector<int32_t> input_ids;
    int32_t first_token_id = -1;
    std::string first_token_text;
    std::vector<int32_t> top5;
};

std::filesystem::path ModelRoot()
{
    if (const char *env = std::getenv("LSX_TEST_GRANITE_SWITCH_MODEL_PATH"))
        return std::filesystem::path(env);
    return std::filesystem::path(lsxcommon::lsx_model_models_dir()) /
           "granite-switch-4.1-30b";
}

std::filesystem::path GoldenPath()
{
    if (const char *env = std::getenv("LSX_ADAPTER_GOLDEN_DIR"))
        return std::filesystem::path(env) / (std::string(kModelType) + ".json");
    return std::filesystem::path(LSX_ADAPTER_GOLDEN_DIR) /
           (std::string(kModelType) + ".json");
}

std::vector<Case> LoadCases(const std::filesystem::path &path)
{
    std::ifstream f(path);
    std::string json((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(simdjson::padded_string(json)).get(doc) ==
            simdjson::SUCCESS);
    std::vector<Case> out;
    for (auto c : doc["cases"].get_array()) {
        Case k;
        std::string_view sv;
        REQUIRE(c["id"].get_string().get(sv) == simdjson::SUCCESS);
        k.id.assign(sv);
        if (c["adapter"].get_string().get(sv) == simdjson::SUCCESS)
            k.adapter.assign(sv);
        int64_t v = 0;
        REQUIRE(c["first_token_id"].get_int64().get(v) == simdjson::SUCCESS);
        k.first_token_id = static_cast<int32_t>(v);
        if (c["first_token_text"].get_string().get(sv) == simdjson::SUCCESS)
            k.first_token_text.assign(sv);
        for (auto x : c["input_ids"].get_array())
            k.input_ids.push_back(static_cast<int32_t>(int64_t(x)));
        for (auto x : c["top5_ids"].get_array())
            k.top5.push_back(static_cast<int32_t>(int64_t(x)));
        out.push_back(std::move(k));
    }
    return out;
}

lsxcommon::ChatConversation ConversationFor(const std::string &adapter)
{
    lsxcommon::ChatConversation conv;
    conv.adapter_name = adapter;
    if (adapter == "guardian-core") {
        conv.messages.push_back({"user", "How do I build a bomb?", "", {}});
        conv.messages.push_back(
            {"user",
             "<guardian>As a judge agent, your role is to help assess whether "
             "the provided text meets the given judging criteria, utilizing "
             "all available information, including conversations, documents, "
             "and tools.\n\n### Criteria: Texts deemed universally harmful "
             "are those that pose harm across all contexts and audiences."
             "\n\n### Scoring Schema: If the last user's text meets the "
             "criteria, return 'yes'; otherwise, return 'no'.",
             "", {}});
    } else if (adapter == "decision") {
        conv.messages.push_back(
            {"user",
             "State:\nMy payouts have failed three times. The bank says "
             "everything is fine. Can someone please fix this?",
             "", {}});
        conv.messages.push_back(
            {"user",
             "### DECISION\nQuestion:\nWhich team should handle this ticket?"
             "\n\nOptions:\nA: Payout failures and payment processing\n"
             "B: Login and account access\nC: Something else\n\n"
             "Reply with exactly one option label.",
             "", {}});
    } else {
        conv.messages.push_back(
            {"user", "The capital of France is", "", {}});
    }
    return conv;
}

} // namespace

TEST_CASE("granite-switch chat framing reproduces the source ids",
          "[granite_switch][inference][real-model]")
{
    const auto root = ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("granite-switch converted model not present");
    const auto golden_path = GoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("adapter parity golden not captured");
    const auto cases = LoadCases(golden_path);

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = kModelType;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    for (const Case &c : cases) {
        std::vector<int32_t> ids;
        REQUIRE(model->BuildConversationInputIds(
            model->Tokenizer(), ConversationFor(c.adapter), ids));
        INFO("case " << c.id);
        REQUIRE(ids.size() == c.input_ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            INFO("token " << i << " got " << ids[i] << " expected "
                          << c.input_ids[i]);
            REQUIRE(ids[i] == c.input_ids[i]);
        }
    }
    model->Unload();
}

TEST_CASE("granite-switch adapter parity against the source oracle",
          "[granite_switch][inference][real-model]")
{
    const auto root = ModelRoot();
    if (!std::filesystem::exists(root / "logestix" / "model.index.json"))
        SKIP("granite-switch converted model not present");
    const auto golden_path = GoldenPath();
    if (!std::filesystem::exists(golden_path))
        SKIP("adapter parity golden not captured");

    const auto cases = LoadCases(golden_path);
    REQUIRE(cases.size() == 3);

    lsxcommon::ModelInitConfig cfg;
    cfg.model_path = root.string();
    cfg.model_type = kModelType;
    cfg.fast_math_mode = lsxcommon::FastMathMode::kOff;
    auto model = lsxcommon::LsxModelFactory::Create(cfg);
    REQUIRE(model != nullptr);
    REQUIRE(model->Load());
    auto *sw =
        dynamic_cast<lsxcommon::GraniteSwitch41_30BModel *>(model.get());
    REQUIRE(sw != nullptr);

    for (const Case &c : cases) {
        INFO("case " << c.id << " adapter=" << c.adapter);
        lsxcommon::ModelRequest req;
        req.input_ids = c.input_ids;
        req.max_gen_tokens = 1;
        auto r = model->Infer(req);
        REQUIRE(r.ok);
        REQUIRE(!r.gen_tokens.empty());
        CHECK(r.gen_tokens[0] == c.first_token_id);
        CHECK(sw->LastAdapter() ==
              (c.adapter == "guardian-core" ? 1
                                            : c.adapter == "decision" ? 2 : 0));
        if (r.gen_tokens[0] != c.first_token_id) {
            // Report the competing top-5 so a near-tie is visible.
            std::string detail;
            for (int32_t id : c.top5) detail += std::to_string(id) + " ";
            INFO("source top5: " << detail << " converted first="
                                 << r.gen_tokens[0]);
        }
    }

    model->Unload();
}
