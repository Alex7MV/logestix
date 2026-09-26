// FSISelfCheck host-only unit tests (extraction/golden/harness). No GPU.
#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include <lsxcommon/fsi_selfcheck.h>

namespace fsi = lsxcommon::fsi;

TEST_CASE("fsi: arithmetic extraction normalizes to a bare formula",
          "[fsi][unit]")
{
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))") ==
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "```python\nPV = CF*((1-exp(-r*n))/(1-exp(-r)))\n```") ==
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "PV",
          "PV=CF*((1-exp(-r*n))/(1-exp(-r))) This is the formula.") ==
          "PV=CF*((1-exp(-r*n))/(1-exp(-r)))");
    CHECK(fsi::ExtractFsiFormula(fsi::FsiKind::kArithmetic, "Balance_Alpha",
          "Balance_Alpha = (((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C") ==
          "Balance_Alpha=(((100000000*Rate_A)-Slip_A)*Rate_B-Slip_B)*Rate_C-Slip_C");
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
