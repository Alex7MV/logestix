// FSISelfCheck host-only unit tests (extraction/golden/harness). No GPU.
#include <catch2/catch_test_macros.hpp>

#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <lsxcommon/fsi_selfcheck.h>
#include <lsxcommon/inference_gpu.h>
#include <lsxcommon/safetensors.h>
#include <lsxcommon/tokenizer.h>

namespace fsi = lsxcommon::fsi;

TEST_CASE("fsi: arithmetic extraction normalizes to a bare formula",
          "[fsi][unit]")
{
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))") ==
          "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "```python\nPV = CF*((1-exp(-r*n))/(1-exp(-r)))\n```") ==
          "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "PV=CF*((1-exp(-r*n))/(1-exp(-r))) This is the formula.") ==
          "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "Balance_Alpha",
          "Balance_Alpha = (((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C") ==
          "Balance_Alpha=((100000000*Rate_A-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV", "").empty());
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "no formula here").empty());
}

TEST_CASE("fsi: comparison extraction is bounded", "[fsi][unit]")
{
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Drawdown>Threshold") == "Drawdown>Threshold");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Drawdown > Threshold") == "Drawdown>Threshold");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Drawdown>Threshold.") == "Drawdown>Threshold");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "```\nDrawdown>Threshold\n```") == "Drawdown>Threshold");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "no answer").empty());
    // A candidate that CANNOT be bounded is skipped, not fatal: a reasoning
    // stream restates the prompt before answering, and the echo's anchor is
    // followed by an operator too (`Drawdown = 14.999991` — an assignment, not
    // a comparison). The capture-side reference (`tools/fsi_capture.py`) has
    // always skipped such candidates with a regex search over the line, so the
    // committed goldens were produced by the skip rule; bailing instead
    // reported an EMPTY answer for a model that answered correctly
    // (measured: inkling-2-mini FSI-TEST-03).
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Variables: Drawdown = 14.999991; Threshold = 15.000000. "
          "Answer: Drawdown>Threshold") == "Drawdown>Threshold");
    // ... and when the echo owns its own line, the line is SKIPPED: the answer
    // is on the next one. This legal because no committed comparison golden can
    // expect an empty answer (the capture aborts on empty extraction), so
    // reaching further for a candidate cannot contradict one. It is REQUIRED
    // because the engine's decode of the same tokens breaks lines differently
    // from HF's `skip_special_tokens` decode (measured: inkling-2-mini
    // FSI-TEST-03 — the echo landed on its own line).
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Variables: Drawdown = 14.999991\nDrawdown>Threshold") ==
          "Drawdown>Threshold");
    // A stream with NO boundable candidate is still empty, not the echo.
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
          "Variables: Drawdown = 14.999991").empty());
}

namespace {

const char *kGoldenJson = R"JSON({
  "format": "logestix-fsi-golden",
  "version": 1,
  "model_type": "fake-model",
  "source": "fake",
  "runtime": "test",
  "prompts": [
    {
      "id": "FSI-TEST-01",
      "kind": "arithmetic",
      "anchor": "PV",
      "input_ids": [1, 2, 3],
      "max_new_tokens": 16,
      "answer_channel": "assistant to=user",
      "reasoning_channel": "assistant to=self",
      "expected": "PV=CF*((1-exp(-r*n))/(1-exp(-r)))",
      "raw": "PV=CF*((1-exp(-r*n))/(1-exp(-r)))"
    },
    {
      "id": "FSI-TEST-03",
      "kind": "comparison",
      "anchor": "Drawdown",
      "input_ids": [9],
      "max_new_tokens": 16,
      "expected": "Drawdown>Threshold",
      "raw": "Drawdown>Threshold"
    }
  ]
})JSON";

} // namespace

TEST_CASE("fsi: golden JSON parses all fields", "[fsi][unit]")
{
    fsi::FsiGolden g;
    std::string err;
    REQUIRE(fsi::ParseFsiGolden(kGoldenJson, g, &err));
    CHECK(err.empty());
    CHECK(g.model_type == "fake-model");
    REQUIRE(g.cases.size() == 2);
    CHECK(g.cases[0].id == "FSI-TEST-01");
    CHECK(g.cases[0].kind == fsi::FsiKind::kArithmetic);
    CHECK(g.cases[0].anchor == "PV");
    CHECK(g.cases[0].input_ids == std::vector<int32_t>{1, 2, 3});
    CHECK(g.cases[0].max_new_tokens == 16);
    CHECK(g.cases[0].answer_channel == "assistant to=user");
    CHECK(g.cases[0].reasoning_channel == "assistant to=self");
    CHECK(g.cases[1].answer_channel.empty());
    CHECK(g.cases[1].kind == fsi::FsiKind::kComparison);
}

TEST_CASE("fsi: malformed golden is rejected with an error", "[fsi][unit]")
{
    fsi::FsiGolden g;
    std::string err;
    CHECK_FALSE(fsi::ParseFsiGolden("{}", g, &err));
    CHECK_FALSE(err.empty());
    CHECK_FALSE(fsi::ParseFsiGolden(
        R"({"format":"logestix-fsi-golden","model_type":"m","prompts":[]})",
        g, &err));
}

namespace {

class FakeFsiModel : public lsxcommon::IModel {
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
        throw std::runtime_error("FSI must not call SelfCheck");
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

fsi::FsiGolden OneCaseGolden()
{
    fsi::FsiGolden g;
    fsi::FsiGoldenCase c;
    c.id = "FSI-TEST-01";
    c.kind = fsi::FsiKind::kArithmetic;
    c.anchor = "PV";
    c.input_ids = {1, 2};
    c.max_new_tokens = 8;
    c.expected = "PV=CF*((1-exp(-r*n))/(1-exp(-r)))";
    g.cases.push_back(std::move(c));
    return g;
}

} // namespace

TEST_CASE("fsi: harness passes on a matching answer and forwards the request",
          "[fsi][unit]")
{
    FakeFsiModel m;
    const fsi::FsiGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string("PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    };
    const fsi::FsiResult r = fsi::RunFsiSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].extracted == "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
    CHECK(m.last_.input_ids == std::vector<int32_t>{1, 2});
    CHECK(m.last_.max_gen_tokens == 8);
}

TEST_CASE("fsi: harness fails on a mismatching answer", "[fsi][unit]")
{
    FakeFsiModel m;
    const fsi::FsiGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string("PV=wrong");
    };
    const fsi::FsiResult r = fsi::RunFsiSelfCheck(m, g, decode);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].extracted == "PV=wrong");
}

TEST_CASE("fsi: harness propagates inference errors per case", "[fsi][unit]")
{
    FakeFsiModel m;
    m.fail = true;
    const fsi::FsiGolden g = OneCaseGolden();
    const auto decode = [](const std::vector<int32_t> &) { return std::string(); };
    const fsi::FsiResult r = fsi::RunFsiSelfCheck(m, g, decode);
    CHECK_FALSE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].error == "boom");
}

TEST_CASE("fsi: arithmetic canonicalization removes redundant parentheses",
          "[fsi][unit]")
{
    const std::string a = fsi::ExtractFsiFormula(
        fsi::FsiKind::kArithmetic, "Balance_Alpha",
        "Balance_Alpha=((((100000000*Rate_A)-Slip_A)*Rate_B)-Slip_B)*Rate_C-Slip_C");
    const std::string b = fsi::ExtractFsiFormula(
        fsi::FsiKind::kArithmetic, "Balance_Alpha",
        "Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C");
    CHECK(a == b);
    CHECK(a ==
          "Balance_Alpha=((100000000*Rate_A-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C");
}

TEST_CASE("fsi: arithmetic canonicalization is associativity-normalized",
          "[fsi][unit]")
{
    const auto canon = [](std::string rhs) {
        return fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
                                      "PV=" + rhs);
    };
    // Multiplicative chains: a*(b/c) == (a*b)/c == a*b/c, a/(b*c) == a/b/c.
    CHECK(canon("CF*((1-exp(-r*n))/(1-exp(-r)))") ==
          canon("CF*(1-exp(-r*n))/(1-exp(-r))"));
    CHECK(canon("a*(b/c)") == canon("(a*b)/c"));
    CHECK(canon("a*(b*c)") == canon("(a*b)*c"));
    CHECK(canon("a/(b*c)") == canon("a/b/c"));
    // a/(b/c) == a*c/b == (a/b)*c, but is NOT a/b/c.
    CHECK(canon("a/(b/c)") == canon("(a/b)*c"));
    CHECK(canon("a/(b/c)") != canon("a/b/c"));
    // Additive chains.
    CHECK(canon("a-(b-c)") == canon("a-b+c"));
    CHECK(canon("a+(b+c)") == canon("a+b+c"));
    CHECK(canon("a-(b+c)") == canon("a-b-c"));
    // Commutativity is deliberately not applied.
    CHECK(canon("a+b") != canon("b+a"));
}

TEST_CASE("fsi: multi-line answer selects the anchor+operator line",
          "[fsi][unit]")
{
    CHECK(fsi::ExtractFsiFormula(
              fsi::FsiKind::kArithmetic, "PV",
              "The correct formula for the present value (PV) is:\n"
              "PV=CF*((1-exp(-r*n))/(1-exp(-r)))") ==
          "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
}

TEST_CASE("fsi: extraction ignores chain-of-thought before </think>",
          "[fsi][unit]")
{
    CHECK(fsi::ExtractFsiFormula(
              fsi::FsiKind::kArithmetic, "PV",
              "We need output. Compute PV=wrong*2.\n</think>\n"
              "PV=CF*((1-exp(-r*n))/(1-exp(-r)))") ==
          "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
              "think: Drawdown<Threshold maybe\n</think>\nDrawdown>Threshold") ==
          "Drawdown>Threshold");
}

TEST_CASE("fsi: prefer_last picks the LAST anchor on an unbroken reasoning line",
          "[fsi][unit]")
{
    // A model that reasons INLINE and emits no newlines between its chain and
    // its answer puts both on ONE line: the answer is the last `anchor=` on it.
    // Taking the first occurrence there extracts the chain (measured on
    // Inkling-2-Small FSI-TEST-01: the answer was correct and was the last thing
    // in the stream, and the extractor returned the reasoning paragraph).
    const std::string raw =
        "Ordinary annuity PV continuous: CF*(1-exp(-r n))/(1-exp(-r))? Actually "
        "PV=CF*sum_{t=1}^n exp(-r t). The expected template is "
        "PV=CF*((1-exp(-r*n))/(1-exp(-r)))";
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV", raw, true) ==
          "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
    // Without prefer_last the historical first-occurrence behaviour is kept, so
    // every captured golden that does not opt in is untouched: it still reads
    // from the chain's first `PV` to the chain's first expression.
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV", raw, false) ==
          "PVcontinuous:CF*(1-exp(-rn))/(1-exp(-r))?ActuallyPV=CF*sum_");
    // Same rule for a comparison, and the last candidate line still wins.
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kComparison, "Drawdown",
                                 "maybe Drawdown<Threshold, but finally "
                                 "Drawdown>Threshold",
                                 true) == "Drawdown>Threshold");
}

TEST_CASE("fsi: ATEM answer channel is selected from a reasoning loop",
          "[fsi][unit]")
{
    // Muse-Glimmer answers inside an `assistant to=user` ATEM channel and then
    // loops back into `assistant to=self` reasoning, re-emitting the answer.
    // The LAST answer channel can be cut mid-answer by the token budget, and
    // the reasoning tail repeats the prompt body (so the prompt-echo cut would
    // remove everything). The harness must read the FIRST COMPLETE answer
    // channel. Markers are special tokens, so the decoder leaves the channel
    // headers as literal text.
    FakeFsiModel m;
    fsi::FsiGolden g = OneCaseGolden();
    g.cases[0].answer_channel = "assistant to=user";
    g.cases[0].reasoning_channel = "assistant to=self";
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string(
            "assistant to=selfWe need PV=CF*((1-exp(-r*n))/(1-exp(-r)))."
            "assistant to=userPV=CF*((1-exp(-r*n))/(1-exp(-r)))"
            "assistant to=selfDone."
            "assistant to=userPV=CF*((1-exp(-r*n))/(1-exp(");
    };
    const fsi::FsiResult r = fsi::RunFsiSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    REQUIRE(r.cases.size() == 1);
    CHECK(r.cases[0].extracted == "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
}

TEST_CASE("fsi: ATEM channel selection keeps a lone truncated answer channel",
          "[fsi][unit]")
{
    // One answer channel and the budget ran out inside it: there is no
    // complete channel to prefer, so the (truncated) channel is still the
    // answer — never the reasoning text that precedes it.
    FakeFsiModel m;
    fsi::FsiGolden g = OneCaseGolden();
    g.cases[0].answer_channel = "assistant to=user";
    g.cases[0].reasoning_channel = "assistant to=self";
    const auto decode = [](const std::vector<int32_t> &) {
        return std::string(
            "assistant to=selfCompute PV from the cash-flow series."
            "assistant to=userPV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    };
    const fsi::FsiResult r = fsi::RunFsiSelfCheck(m, g, decode);
    REQUIRE(r.all_passed);
    CHECK(r.cases[0].extracted == "PV=CF*(1-exp(-r*n))/(1-exp(-r))");
}
