#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

namespace lsxcommon {
namespace gpu {

__global__ void rms_norm_kernel(
    const uint16_t *x,
    const uint16_t *weight,
    uint16_t *out,
    int dim,
    float eps)
{
}

} // namespace gpu
} // namespace lsxcommon
