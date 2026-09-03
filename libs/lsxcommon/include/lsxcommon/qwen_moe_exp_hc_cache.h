#ifndef LSXCOMMON_QWEN_MOE_EXP_HC_CACHE_H_
#define LSXCOMMON_QWEN_MOE_EXP_HC_CACHE_H_

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "lsxcommon/qwen_inference.h"  // QwenNvfp4Weight

namespace lsxcommon::qwen_moe_exp {

// Dequantized fp16 twin of a layer's hyper-connection weights (down/up +
// optional inject), built once at load. Eliminates the per-token
// dequantize-everything cost of the hc GatedResidual path.
struct ExpHcDequantCache {
    __half *down = nullptr;    // [lowrank, hc*H]
    __half *up = nullptr;      // [hc*H, lowrank]
    __half *inject = nullptr;  // [hc, hc*H]; null when absent (mixer)
    int64_t down_elems = 0, up_elems = 0, inject_elems = 0;

    void Free();
};

// Dequantizes the three hc weights once (on load). The mixer passes an
// empty inject_w: inject cache stays null. Strict on allocation failure.
cudaError_t BuildExpHcDequantCache(cudaStream_t stream,
                                   const qwen::QwenNvfp4Weight &down,
                                   const qwen::QwenNvfp4Weight &up,
                                   const qwen::QwenNvfp4Weight &inject_w,
                                   ExpHcDequantCache &out);

} // namespace lsxcommon::qwen_moe_exp

#endif // LSXCOMMON_QWEN_MOE_EXP_HC_CACHE_H_
