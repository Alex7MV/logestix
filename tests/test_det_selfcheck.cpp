// DETSelfCheck host-only unit tests (extraction/golden/harness). No GPU.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <lsxcommon/det_selfcheck.h>
#include <lsxcommon/inference_gpu.h>
#include <lsxcommon/safetensors.h>

namespace det = lsxcommon::det;

TEST_CASE("det: token extraction picks the first standalone uppercase token",
          "[det][unit]")
{
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "PARADOX_ANTINOMY") ==
          "PARADOX_ANTINOMY");
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken,
                                "The answer is PARADOX_ANTINOMY.") ==
          "PARADOX_ANTINOMY");
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken,
                                "```\nCATASTROPHIC\n```") == "CATASTROPHIC");
    CHECK(det::ExtractDetAnswer(
              det::DetKind::kToken,
              "thinking... BIT_ZERO maybe\n</think>\nBIT_ZERO") == "BIT_ZERO");
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "**PARADOX_ANTINOMY**") ==
          "PARADOX_ANTINOMY");
}

TEST_CASE("det: token extraction ignores lowercase and short tokens",
          "[det][unit]")
{
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "the answer is a word")
              .empty());
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "").empty());
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "A").empty());
    // "DET" is a standalone uppercase run -> a valid token by contract.
    CHECK(det::ExtractDetAnswer(det::DetKind::kToken, "DET-TEST-01") == "DET");
}

namespace {

const char *kGoldenJson = R"JSON({
  "format": "logestix-det-golden",
  "version": 1,
  "model_type": "fake-model",
  "source": "fake",
  "runtime": "test",
  "prompts": [
    {
      "id": "DET-TEST-01",
      "kind": "token",
      "input_ids": [1, 2, 3],
      "max_new_tokens": 32,
      "stress_runs": 4,
      "body": "[CONTEXT]: DET-Validation-Core-01",
      "expected": "PARADOX_ANTINOMY",
      "raw": "PARADOX_ANTINOMY"
    },
    {
      "id": "DET-TEST-03",
      "kind": "token",
      "input_ids": [9],
      "expected": "BIT_ZERO"
    }
  ]
})JSON";

} // namespace

TEST_CASE("det: golden JSON parses all fields and stress_runs default",
          "[det][unit]")
{
    det::DetGolden g;
    std::string err;
    REQUIRE(det::ParseDetGolden(kGoldenJson, g, &err));
    CHECK(err.empty());
    CHECK(g.model_type == "fake-model");
    REQUIRE(g.cases.size() == 2);
    CHECK(g.cases[0].id == "DET-TEST-01");
    CHECK(g.cases[0].kind == det::DetKind::kToken);
    CHECK(g.cases[0].input_ids == std::vector<int32_t>{1, 2, 3});
    CHECK(g.cases[0].max_new_tokens == 32);
    CHECK(g.cases[0].stress_runs == 4);
    CHECK(g.cases[0].body == "[CONTEXT]: DET-Validation-Core-01");
    CHECK(g.cases[0].expected == "PARADOX_ANTINOMY");
    CHECK(g.cases[1].stress_runs == 3);  // absent -> default
    CHECK(g.cases[1].max_new_tokens == 64);
}

TEST_CASE("det: malformed golden is rejected with an error", "[det][unit]")
{
    det::DetGolden g;
    std::string err;
    CHECK_FALSE(det::ParseDetGolden("{}", g, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(det::ParseDetGolden(
        R"({"format":"logestix-det-golden","model_type":"m","prompts":[]})",
        g, &err));
    CHECK_FALSE(det::ParseDetGolden(
        R"({"format":"logestix-det-golden","model_type":"m","prompts":[
            {"id":"x","kind":"nope","input_ids":[1],"expected":"T"}]})",
        g, &err));
}


namespace {

class FakeDetModel : public lsxcommon::IModel {
public:
    bool Load() override { return true; }
    void Unload() override {}
    lsxcommon::SafetensorsModel Inspect() override { return {}; }
    lsxcommon::gpu::InferenceResult Infer(
        const lsxcommon::ModelRequest &req) override
    {
        last_ = req;
        ++calls;
        if (fail) {
            lsxcommon::gpu::InferenceResult r;
            r.ok = false;
            r.error_msg = "boom";
            return r;
        }
        lsxcommon::gpu::InferenceResult r;
        r.ok = true;
        const std::size_t idx =
            sequence.empty()
                ? 0
                : std::min<std::size_t>(static_cast<std::size_t>(calls - 1),
                                        sequence.size() - 1);
        r.gen_tokens = sequence.empty() ? std::vector<int32_t>{1}
                                        : sequence[idx];
        return r;
    }
    lsxcommon::gpu::InferenceResult SelfCheck(
        const lsxcommon::ModelRequest &) override
    {
        throw std::runtime_error("DET must not call SelfCheck");
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

    std::vector<std::vector<int32_t>> sequence;
    bool fail = false;
    int calls = 0;
    lsxcommon::ModelRequest last_;
private:
    std::shared_mutex m_;
};

det::DetGolden OneCaseGolden(int stress_runs = 3)
{
    det::DetGolden g;
    det::DetGoldenCase c;
    c.id = "DET-TEST-01";
    c.kind = det::DetKind::kToken;
    c.input_ids = {1, 2};
    c.max_new_tokens = 8;
    c.stress_runs = stress_runs;
    c.expected = "PARADOX_ANTINOMY";
    g.cases.push_back(std::move(c));
    return g;
}

std::string DecodeOne(const std::vector<int32_t> &) { return "PARADOX_ANTINOMY"; }

} // namespace

TEST_CASE("det: harness passes when runs are identical and answer matches",
          "[det][unit]")
{
    FakeDetModel m;
    const det::DetGolden g = OneCaseGolden();
    const det::DetResult r = det::RunDetSelfCheck(m, g, DecodeOne);
    REQUIRE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].deterministic);
    CHECK(r.cases[0].runs == 3);
    CHECK(r.cases[0].extracted == "PARADOX_ANTINOMY");
    CHECK(m.last_.input_ids == std::vector<int32_t>{1, 2});
    CHECK(m.last_.max_gen_tokens == 8);
    CHECK(m.calls == 3);
}

TEST_CASE("det: harness fails when a run diverges", "[det][unit]")
{
    FakeDetModel m;
    m.sequence = {{1}, {1}, {2}};
    const det::DetResult r = det::RunDetSelfCheck(m, OneCaseGolden(), DecodeOne);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK_FALSE(r.cases[0].deterministic);
}

TEST_CASE("det: harness fails on an answer mismatch", "[det][unit]")
{
    FakeDetModel m;
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string("CATASTROPHIC");
    };
    const det::DetResult r = det::RunDetSelfCheck(m, OneCaseGolden(), decode);
    CHECK_FALSE(r.all_passed);
    CHECK(r.cases[0].deterministic);
    CHECK(r.cases[0].extracted == "CATASTROPHIC");
}

TEST_CASE("det: harness propagates inference errors", "[det][unit]")
{
    FakeDetModel m;
    m.fail = true;
    const det::DetResult r = det::RunDetSelfCheck(m, OneCaseGolden(), DecodeOne);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].error == "boom");
}

TEST_CASE("det: stress_runs override is floored at 3", "[det][unit]")
{
    FakeDetModel m;
    const det::DetResult r =
        det::RunDetSelfCheck(m, OneCaseGolden(1), DecodeOne, 1);
    CHECK(r.cases[0].runs == 3);
    CHECK(m.calls == 3);
}
