// FSISelfCheck host-only unit tests (extraction/golden/harness). No GPU.
#include <catch2/catch_test_macros.hpp>

#include <string>

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
