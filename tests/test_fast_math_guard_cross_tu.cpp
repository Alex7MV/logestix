// E1 guard regression: the FastMathScope depth must be process-global even
// though the class is header-only and used from many translation units.
//
// Before the fix `Detail::FastMathDepth()` was an anonymous-namespace static
// in the header: every TU owned a private counter, Enter/Leave could land on
// different copies, and real FSI/DET runs printed false
// `VIOLATION: 2,3,...` messages while the depth leaked monotonically.
//
// This test lives in its own TU and drives Enter/Leave from a SECOND TU, so a
// per-TU counter cannot pass.

#include <catch2/catch_test_macros.hpp>

#include "lsxcommon/lsx_inference_nvfp4.h"

extern "C" void AuditGuardEnterOtherTu();
extern "C" void AuditGuardLeaveOtherTu();

using lsxcommon::lsx_nvfp4::Detail::ActiveFastMathScopeDepth;

TEST_CASE("E1 guard counter is shared across translation units", "[fast][guard]")
{
    REQUIRE(ActiveFastMathScopeDepth() == 0);

    AuditGuardEnterOtherTu();
    REQUIRE(ActiveFastMathScopeDepth() == 1);

    AuditGuardLeaveOtherTu();
    REQUIRE(ActiveFastMathScopeDepth() == 0);
}
