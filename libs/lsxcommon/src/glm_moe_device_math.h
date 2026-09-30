// Device math helpers that reproduce, bit-exactly, the functions the GLM-5.2
// host reference resolves to on the target build host:
//   * std::exp (gcc, glibc)  -> glibc's __expf  (e_expf.c, table-based)
//   * std::sqrtf/rsqrtf      -> glibc rsqrtf = 1.0f/sqrtf(x) (s_rsqrt_template.c)
// The CPU reference is compiled at -O0 without FMA ISA and without
// -ffast-math, so every arithmetic step below must stay a plain (non-FMA)
// IEEE operation. The target CUDA build enforces this with -fmad=false.
//
// The implementations live in the model-neutral lsx_device_math.h; the glm::
// names are kept so all existing GLM call sites compile unchanged.

#ifndef LSXCOMMON_GLM_DEVICE_MATH_H
#define LSXCOMMON_GLM_DEVICE_MATH_H

#include <cuda_runtime.h>
#include <cstdint>

#include "lsx_device_math.h"

namespace lsxcommon {
namespace glm {
namespace device {

using lsxcommon::device::ExpfGlibc;
using lsxcommon::device::RsqrtfGlibc;

__device__ __forceinline__ float GlmExpfGlibc(float x)
{
    return ExpfGlibc(x);
}

__device__ __forceinline__ float GlmRsqrtfGlibc(float x)
{
    return RsqrtfGlibc(x);
}

}  // namespace device
}  // namespace glm
}  // namespace lsxcommon

#endif  // LSXCOMMON_GLM_DEVICE_MATH_H
