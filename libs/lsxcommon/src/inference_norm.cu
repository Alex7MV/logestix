#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace lsxcommon {
namespace gpu {

__global__ void rms_norm_kernel(
    const __half *x,
    const __half *weight,
    __half *out,
    int dim,
    float eps)
{
    extern __shared__ float s_partial[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    float local_sum = 0.0f;
    for (int i = tid; i < dim; i += nthreads) {
        float v = __half2float(x[i]);
        local_sum += v * v;
    }

    s_partial[tid] = local_sum;
    __syncthreads();

    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_partial[tid] += s_partial[tid + stride];
        __syncthreads();
    }

    float rms = rsqrtf(s_partial[0] / static_cast<float>(dim) + eps);

    for (int i = tid; i < dim; i += nthreads)
        out[i] = __float2half(__half2float(x[i]) * rms * __half2float(weight[i]));
}

} // namespace gpu
} // namespace lsxcommon
