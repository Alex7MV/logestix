// Device math helpers for Xing4.0-29B-A4B: the mHC block needs exp/rsqrt at
// kOff (selfcheck parity against the BF16 host reference), so it resolves to
// the same glibc replicas as the other bit-exact families instead of the
// fast-math intrinsics.
//
// The implementations are model-neutral (lsx_device_math.h); the xing:: names
// exist so a Xing kernel never reads like a GLM one.

#ifndef LSXCOMMON_XING_DEVICE_MATH_H
#define LSXCOMMON_XING_DEVICE_MATH_H

#include <cuda_runtime.h>
#include <cstdint>

#include "lsx_device_math.h"

namespace lsxcommon {
namespace xing {
namespace device {

__device__ __forceinline__ float XingExpfGlibc(float x)
{
    return lsxcommon::device::ExpfGlibc(x);
}

__device__ __forceinline__ float XingRsqrtfGlibc(float x)
{
    return lsxcommon::device::RsqrtfGlibc(x);
}

}  // namespace device
}  // namespace xing
}  // namespace lsxcommon

#endif  // LSXCOMMON_XING_DEVICE_MATH_H
