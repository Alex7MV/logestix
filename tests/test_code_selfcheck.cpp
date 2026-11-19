// CODESelfCheck host-only unit tests (extraction/golden/harness). No GPU.
#include <catch2/catch_test_macros.hpp>

#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <lsxcommon/code_selfcheck.h>
#include <lsxcommon/inference_gpu.h>
#include <lsxcommon/safetensors.h>
#include <lsxcommon/tokenizer.h>

namespace code = lsxcommon::code;

TEST_CASE("code: json extraction minifies the nested object", "[code][unit]")
{
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          R"({"root":{"nodes":{"leaf":true}}})") ==
          R"({"root":{"nodes":{"leaf":true}}})");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          "```json\n{\"root\":{\"nodes\":{\"leaf\":true}}}\n```") ==
          R"({"root":{"nodes":{"leaf":true}}})");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          R"({ "root" : { "nodes" : { "leaf" : true } } })") ==
          R"({"root":{"nodes":{"leaf":true}}})");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          R"({"root":{"nodes":{"leaf":true}}} done)") ==
          R"({"root":{"nodes":{"leaf":true}}})");
    // Braces inside a string literal must not break depth tracking.
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          R"({"root":{"a":"}"}})") ==
          R"({"root":{"a":"}"}})");
    // Whitespace inside a string literal is preserved.
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          R"({ "root" : { "msg" : "a b" } })") ==
          R"({"root":{"msg":"a b"}})");
    // Anchor not present.
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          R"({"foo":1})").empty());
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root", "").empty());
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          "no json here").empty());
}

TEST_CASE("code: assignment extraction keeps the exact parentheses",
          "[code][unit]")
{
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kAssignment, "m",
          "m=l+((h-l)/2)") == "m=l+((h-l)/2)");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kAssignment, "m",
          "m = l + ((h - l) / 2)") == "m=l+((h-l)/2)");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kAssignment, "m",
          "```c\nm=l+((h-l)/2);\n```") == "m=l+((h-l)/2)");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kAssignment, "m",
          "The split is m=l+((h-l)/2); return mid") == "m=l+((h-l)/2)");
    // No associativity rewriting: a redundant-paren variant is NOT equal.
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kAssignment, "m",
          "m=l+(h-l)/2") != "m=l+((h-l)/2)");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kAssignment, "m", "").empty());
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kAssignment, "m",
          "no assignment here").empty());
}

TEST_CASE("code: regex extraction captures the token including leading escape",
          "[code][unit]")
{
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kRegex, "[0-9A-F]",
          "\\b[0-9A-F]{4}\\b") == "\\b[0-9A-F]{4}\\b");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kRegex, "[0-9A-F]",
          "```regex\n\\b[0-9A-F]{4}\\b\n```") == "\\b[0-9A-F]{4}\\b");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kRegex, "[0-9A-F]",
          "\\b[0-9A-F]{4}\\b.") == "\\b[0-9A-F]{4}\\b");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kRegex, "[0-9A-F]",
          "").empty());
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kRegex, "[0-9A-F]",
          "no pattern here").empty());
}

TEST_CASE("code: extraction ignores chain-of-thought before </think>",
          "[code][unit]")
{
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kJson, "root",
          "we should output {\"wrong\":1}\n</think>\n"
          R"({"root":{"nodes":{"leaf":true}}})") ==
          R"({"root":{"nodes":{"leaf":true}}})");
    CHECK(code::ExtractCodeAnswer(code::CodeKind::kAssignment, "m",
          "maybe m=0\n</think>\nm=l+((h-l)/2)") == "m=l+((h-l)/2)");
}

namespace {

const char *kGoldenJson = R"JSON({
  "format": "logestix-code-golden",
  "version": 1,
  "model_type": "fake-model",
  "source": "fake",
  "runtime": "test",
  "prompts": [
    {
      "id": "CODE-TEST-01",
      "kind": "json",
      "anchor": "root",
      "input_ids": [1, 2, 3],
      "max_new_tokens": 16,
      "body": "[CONTEXT]: CODE-Validation-Core-01",
      "expected": "{\"root\":{\"nodes\":{\"leaf\":true}}}",
      "raw": "{\"root\":{\"nodes\":{\"leaf\":true}}}"
    },
    {
      "id": "CODE-TEST-03",
      "kind": "regex",
      "anchor": "[0-9A-F]",
      "input_ids": [9],
      "max_new_tokens": 16,
      "expected": "\\b[0-9A-F]{4}\\b",
      "raw": "\\b[0-9A-F]{4}\\b"
    }
  ]
})JSON";

} // namespace

TEST_CASE("code: golden JSON parses all fields", "[code][unit]")
{
    code::CodeGolden g;
    std::string err;
    REQUIRE(code::ParseCodeGolden(kGoldenJson, g, &err));
    CHECK(err.empty());
    CHECK(g.model_type == "fake-model");
    REQUIRE(g.cases.size() == 2);
    CHECK(g.cases[0].id == "CODE-TEST-01");
    CHECK(g.cases[0].kind == code::CodeKind::kJson);
    CHECK(g.cases[0].anchor == "root");
    CHECK(g.cases[0].input_ids == std::vector<int32_t>{1, 2, 3});
    CHECK(g.cases[0].max_new_tokens == 16);
    CHECK(g.cases[0].body == "[CONTEXT]: CODE-Validation-Core-01");
    CHECK(g.cases[1].kind == code::CodeKind::kRegex);
    CHECK(g.cases[1].body.empty());
}

TEST_CASE("code: malformed golden is rejected with an error", "[code][unit]")
{
    code::CodeGolden g;
    std::string err;
    CHECK_FALSE(code::ParseCodeGolden("{}", g, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(code::ParseCodeGolden(
        R"({"format":"logestix-code-golden","model_type":"m","prompts":[]})",
        g, &err));
    CHECK_FALSE(code::ParseCodeGolden(
        R"({"format":"logestix-code-golden","model_type":"m","prompts":)"
        R"([{"id":"x","kind":"nope","anchor":"a","input_ids":[1],"expected":"e"}]})",
        g, &err));
}

namespace {

class FakeCodeModel : public lsxcommon::IModel {
public:
    bool Load() override { return true; }
    void Unload() override {}
    lsxcommon::SafetensorsModel Inspect() override { return {}; }
    lsxcommon::gpu::InferenceResult Infer(
        const lsxcommon::ModelRequest &req) override
    {
        last_ = req;
        if (fail) {
            lsxcommon::gpu::InferenceResult r;
            r.ok = false;
            r.error_msg = "boom";
            return r;
        }
        lsxcommon::gpu::InferenceResult r;
        r.ok = true;
        r.gen_tokens = gen;
        return r;
    }
    lsxcommon::gpu::InferenceResult SelfCheck(
        const lsxcommon::ModelRequest &) override
    {
        throw std::runtime_error("CODESelfCheck must not call SelfCheck");
    }
    const std::string &Type() const override
    {
        static const std::string t = "fake-model";
        return t;
    }
    lsxcommon::ArrowTokenizerDb &Tokenizer() override
    {
        static lsxcommon::ArrowTokenizerDb db{};
        return db;
    }
    std::shared_mutex &ResidencyMutex() override { return m_; }

    std::vector<int32_t> gen = {7};
    bool fail = false;
    lsxcommon::ModelRequest last_;
private:
    std::shared_mutex m_;
};

code::CodeGolden OneCaseGolden()
{
    code::CodeGolden g;
    code::CodeGoldenCase c;
    c.id = "CODE-TEST-02";
    c.kind = code::CodeKind::kAssignment;
    c.anchor = "m";
    c.input_ids = {1, 2};
    c.max_new_tokens = 8;
    c.expected = "m=l+((h-l)/2)";
    g.cases.push_back(std::move(c));
    return g;
}

} // namespace

TEST_CASE("code: harness passes on a matching answer and forwards the request",
          "[code][unit]")
{
    FakeCodeModel m;
    const code::CodeGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string("m=l+((h-l)/2)");
    };
    const code::CodeResult r = code::RunCodeSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].extracted == "m=l+((h-l)/2)");
    CHECK(m.last_.input_ids == std::vector<int32_t>{1, 2});
    CHECK(m.last_.max_gen_tokens == 8);
}

TEST_CASE("code: harness fails on a mismatching answer", "[code][unit]")
{
    FakeCodeModel m;
    const code::CodeGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string("m=l+(h-l)/2");
    };
    const code::CodeResult r = code::RunCodeSelfCheck(m, g, decode);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].extracted == "m=l+(h-l)/2");
}

TEST_CASE("code: harness propagates inference errors per case", "[code][unit]")
{
    FakeCodeModel m;
    m.fail = true;
    const code::CodeGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) { return std::string(); };
    const code::CodeResult r = code::RunCodeSelfCheck(m, g, decode);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].error == "boom");
}

TEST_CASE("code: harness cuts an echoed prompt after the answer",
          "[code][unit]")
{
    FakeCodeModel m;
    code::CodeGolden g;
    code::CodeGoldenCase c;
    c.id = "CODE-TEST-03";
    c.kind = code::CodeKind::kRegex;
    c.anchor = "[0-9A-F]";
    c.input_ids = {1};
    c.max_new_tokens = 8;
    c.body = "[CONTEXT]: CODE-Validation-Core-03 // Compiler Token Escaping";
    c.expected = "\\b[0-9A-F]{4}\\b";
    g.cases.push_back(std::move(c));
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string(
            "\\b[0-9A-F]{4}\\b[CONTEXT]: CODE-Validation-Core-03 // Compiler "
            "Token Escaping\n... repeated prompt");
    };
    const code::CodeResult r = code::RunCodeSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    CHECK(r.cases[0].extracted == "\\b[0-9A-F]{4}\\b");
}

TEST_CASE("code: prompt echo in reasoning is stripped before the cut",
          "[code][unit]")
{
    FakeCodeModel m;
    code::CodeGolden g = OneCaseGolden();
    g.cases[0].body =
        "[CONTEXT]: CODE-Validation-Core-02 // Structural Control Flow";
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string(
            "Let me read [CONTEXT]: CODE-Validation-Core-02 // Structural "
            "Control Flow first.\n</think>\nm=l+((h-l)/2)");
    };
    const code::CodeResult r = code::RunCodeSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    CHECK(r.cases[0].extracted == "m=l+((h-l)/2)");
}

TEST_CASE("code: golden expected values are self-consistent", "[code][unit]")
{
    const auto canon = [](code::CodeKind k, const char *anchor,
                          const char *expected) {
        return code::ExtractCodeAnswer(k, anchor, expected);
    };
    CHECK(canon(code::CodeKind::kJson, "root",
                R"({"root":{"nodes":{"leaf":true}}})") ==
          R"({"root":{"nodes":{"leaf":true}}})");
    CHECK(canon(code::CodeKind::kAssignment, "m", "m=l+((h-l)/2)") ==
          "m=l+((h-l)/2)");
    CHECK(canon(code::CodeKind::kRegex, "[0-9A-F]", "\\b[0-9A-F]{4}\\b") ==
          "\\b[0-9A-F]{4}\\b");
}

TEST_CASE("code: ATEM answer channel is selected from a reasoning loop",
          "[code][unit]")
{
    // Muse-Glimmer emits its answer in an "assistant to=user" ATEM channel,
    // then loops back into "assistant to=self" reasoning. The channel markers
    // are special tokens (decode drops them), so the text carries the literal
    // "assistant to=user" / "assistant to=self" headers. The extractor must
    // read the answer channel, not the reasoning tail (which can be truncated
    // by the token budget mid-answer).
    FakeCodeModel m;
    code::CodeGolden g = OneCaseGolden();
    g.cases[0].answer_channel = "assistant to=user";
    g.cases[0].reasoning_channel = "assistant to=self";
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string(
            "assistant to=selfWe need m=l+((h-l)/2). Probably."
            "assistant to=userm=l+((h-l)/2)"
            "assistant to=selfDone.assistant to=userm=l+((h-l)/2)"
            "assistant to=selfOk.");
    };
    const code::CodeResult r = code::RunCodeSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    CHECK(r.cases[0].extracted == "m=l+((h-l)/2)");
}

TEST_CASE("code: ATEM channel selection tolerates a truncated last channel",
          "[code][unit]")
{
    // The LAST answer channel can be cut mid-answer by the token budget; the
    // first COMPLETE answer channel is the model's answer.
    FakeCodeModel m;
    code::CodeGolden g = OneCaseGolden();
    g.cases[0].answer_channel = "assistant to=user";
    g.cases[0].reasoning_channel = "assistant to=self";
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string(
            "assistant to=userm=l+((h-l)/2)"
            "assistant to=selfAgain.assistant to=userm=l+((h-l)");
    };
    const code::CodeResult r = code::RunCodeSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    CHECK(r.cases[0].extracted == "m=l+((h-l)/2)");
}
