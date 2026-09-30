#ifndef LSXCOMMON_LSX_REDUCE_CUH
#define LSXCOMMON_LSX_REDUCE_CUH

// Model-neutral warp-shuffle block max reduction for attention kernels.
//
// Replaces the strided shared-memory tree (log2(threads) __syncthreads() and
// one smem round-trip per step) with 5 warp shuffles + a single cross-warp
// pass through shared memory. For MAX the result is identical to any
// reduction order (fmaxf is associative and commutative), so swapping the
// implementation is bit-safe by construction; SUM reductions stay on the
// strided tree because reassociation would perturb the last ulp against the
// CPU FP64 reference.
//
// Scratch contract: points at a __shared__ float array of at least
// blockDim.x/32 (rounded up) floats; all threads of the block must call.

#ifndef __CUDA_ARCH__
#include <cfloat>
#endif

#include <cuda_runtime.h>

namespace lsxcommon {
namespace kernels {

__device__ __forceinline__ float BlockMaxShfl(float v)
{
    __shared__ float warp_max[32];

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, v, off);
        v = fmaxf(v, other);
    }
    if (lane == 0) warp_max[wid] = v;
    __syncthreads();

    // Single thread folds the per-warp maxima; warp count <= 32 so this is
    // one short serial pass instead of log2(threads) block-wide barriers.
    if (threadIdx.x == 0) {
        float m = warp_max[0];
        const int nw = (static_cast<int>(blockDim.x) + 31) >> 5;
        for (int i = 1; i < nw; ++i) m = fmaxf(m, warp_max[i]);
        warp_max[0] = m;
    }
    __syncthreads();
    return warp_max[0];
}

}  // namespace kernels
}  // namespace lsxcommon

#endif // LSXCOMMON_LSX_REDUCE_CUH
