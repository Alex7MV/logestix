#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/step_moe_load.h"

namespace lsxcommon::step {

// Scratch views owned by the pipeline's arena (no allocation here).
struct StepAttnScratch {
    __half *normed = nullptr;  // [H]
    __half *q = nullptr;       // [NH*HD]
    __half *k = nullptr;       // [NKV*HD]
    __half *v = nullptr;       // [NKV*HD]
    __half *gate = nullptr;    // [NH]
    __half *attn = nullptr;    // [NH*HD]
    __half *out = nullptr;     // [H]  (o_proj output, added to the residual)
};

// One attention block: FP32 residual -> norm -> fused QKV + per-head gate
// projection -> per-head QK RMSNorm -> partial NEOX RoPE -> KV append ->
// windowed/full causal GQA attention -> per-head sigmoid gate -> o_proj.
cudaError_t RunStepAttentionBlock(cudaStream_t stream,
                                  const StepLayerWeights &lw,
                                  const StepTopology &top, int layer,
                                  const float *d_resid,
                                  const StepAttnScratch &s, __half *k_base,
                                  __half *v_base, int pos);

} // namespace lsxcommon::step
