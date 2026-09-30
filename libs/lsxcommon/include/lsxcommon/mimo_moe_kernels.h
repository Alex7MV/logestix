#pragma once

// MiMo-V2.6-Flash-RL family kernels: RMSNorm, sigmoid noaux_tc router and
// SwiGLU. NVFP4 GEMMs go through the neutral lsx_nvfp4 layer; these are the
// non-GEMM primitives. One stream, no per-token cudaDeviceSynchronize.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon {
namespace mimo {

// out[r][j] = (x[r][j] / sqrt(mean(x[r]^2)+eps)) * w[j]
cudaError_t RunMimoRmsNorm(cudaStream_t stream, const __half *x,
                           const __half *w, int rows, int hidden, float eps,
                           __half *out);

// Per-row sigmoid noaux_tc router: idx/w sized [rows * top_k].
// gate_w: BF16 [num_experts, hidden]; bias: F32 [num_experts].
cudaError_t RunMimoRouter(cudaStream_t stream, const __half *x,
                          const __nv_bfloat16 *gate_w, const float *bias,
                          int rows, int hidden, int num_experts, int top_k,
                          int *idx_out, float *w_out);

// out[i] = silu(gate[i]) * up[i]
cudaError_t RunMimoSwiGLU(cudaStream_t stream, const __half *gate,
                          const __half *up, int64_t n, __half *out);

} // namespace mimo
} // namespace lsxcommon
