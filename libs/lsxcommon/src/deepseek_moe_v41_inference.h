// deepseek_moe_v41_inference.h — DeepSeek-V4.1-Flash device kernel launchers.
// FP16 paths (uint16_t bit patterns to keep host TUs cuda_fp16-free) and FP32
// Hyper-Connection paths (the hc residual stream is carried in F32; norm/rope
// bridges cast at the boundary).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace deepseek_v41 {

// RMSNorm (F32 compute, eps = 1e-20 for V4.1): out = gamma*x*rsqrt(mean(x^2)+eps).
cudaError_t LaunchRmsNormHalf(const uint16_t *x, const uint16_t *gamma,
                              uint16_t *out, int n, float eps,
                              cudaStream_t stream);

// Rotates the last `rope_dim` elements of each of `rows` row-vectors in place.
cudaError_t LaunchRopeHalf(uint16_t *x, int rows, int row_len, int rope_dim,
                           const float *cos_row, const float *sin_row,
                           bool inverse, cudaStream_t stream);

// Hyper-Connections: mixes = (hc_fn @ x) * rsqrt(mean(x^2)+eps), then
// split+Sinkhorn into pre/post/comb. x: [hc*dim] F32.
cudaError_t LaunchHcMixes(const float *x, const float *hc_fn,
                          const float *hc_scale, const float *hc_base, int hc,
                          int dim, int sinkhorn_iters, float eps, float *pre,
                          float *post, float *comb, cudaStream_t stream);

// out[d] = sum_c pre[c] * x[c*dim + d]. x: [hc*dim] F32.
cudaError_t LaunchHcPre(const float *x, const float *pre_mix, float *out,
                        int hc, int dim, cudaStream_t stream);

// out[c][d] = post[c]*x[d] + sum_cp comb[cp*hc+c]*residual[cp][d].
cudaError_t LaunchHcPost(const float *x, const float *residual,
                         const float *post, const float *comb, float *out,
                         int hc, int dim, cudaStream_t stream);

} // namespace deepseek_v41
} // namespace lsxcommon
