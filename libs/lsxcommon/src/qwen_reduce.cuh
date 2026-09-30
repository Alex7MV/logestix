#ifndef LSXCOMMON_QWEN_REDUCE_CUH
#define LSXCOMMON_QWEN_REDUCE_CUH

// Qwen-named alias for the model-neutral block max reduction. The canonical
// implementation lives in lsx_reduce.cuh (lsxcommon::kernels::BlockMaxShfl);
// the qwen:: name is kept so existing Qwen call sites compile unchanged.
//
// Scratch contract: points at a __shared__ float array of at least
// blockDim.x/32 (rounded up) floats; all threads of the block must call.

#include "lsx_reduce.cuh"

__device__ __forceinline__ float qwen_block_max_shfl(float v)
{
    return lsxcommon::kernels::BlockMaxShfl(v);
}

#endif // LSXCOMMON_QWEN_REDUCE_CUH
