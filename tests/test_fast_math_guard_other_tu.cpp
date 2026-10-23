// Second translation unit for the E1 guard cross-TU regression.
// The guard's counter used to be an anonymous-namespace static in a header,
// so this TU and the test TU each got a private copy. These shims let the test
// prove that an Enter here is visible there.

#include "lsxcommon/lsx_inference_nvfp4.h"

extern "C" void AuditGuardEnterOtherTu()
{
    lsxcommon::lsx_nvfp4::Detail::EnterFastMathScope();
}

extern "C" void AuditGuardLeaveOtherTu()
{
    lsxcommon::lsx_nvfp4::Detail::LeaveFastMathScope();
}
