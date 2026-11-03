#pragma once

// Model-neutral dense-decoder kernel layer (RMSNorm, RoPE, causal GQA
// attention with the split-KV flash-decode path, fused split-QKV, fused
// SwiGLU). The algorithm is the one Granite-4.2-30B has always used; the
// consumers (granite, granite-switch, phi-4) keep their own layer drivers and
// only delegate here. Header must stay free of abseil containers: it is
// included from .cu translation units.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace lsxcommon {
namespace dense {

// out = x * rsqrt(mean(x^2) + eps) * w, row-wise over `hidden`.
cudaError_t RunDenseRmsNormKernel(cudaStream_t stream, __half *out,
                                  const __half *x, const __half *w, int rows,
                                  int hidden, float eps);

// Full-dimension rotate-half RoPE; positions are absolute.
cudaError_t RunDenseRopeKernel(cudaStream_t stream, __half *q, __half *k,
                               const int32_t *pos, int seq, int nh, int nkv,
                               int hd, double theta);

// Causal GQA attention with a persistent KV cache: split-KV when the caller
// provides `partials` and the shape pays off, legacy block kernel otherwise.
cudaError_t RunDenseAttentionKernel(cudaStream_t stream, const __half *q,
                                    const __half *k, const __half *v,
                                    __half *out, float *partials, int seq,
                                    int nh, int nkv, int hd, float scale,
                                    int kv_base);

cudaError_t RunDenseAttentionSplitKernel(cudaStream_t stream, const __half *q,
                                         const __half *k, const __half *v,
                                         __half *out, float *partials, int seq,
                                         int nh, int nkv, int hd, float scale,
                                         int kv_base, int splits);

// Unpack an interleaved [seq, qn + 2*kn] fused projection.
cudaError_t RunDenseSplitQKVKernel(cudaStream_t stream, const __half *qkv,
                                   __half *q, __half *k, __half *v, int seq,
                                   int qn, int kn);

// out = silu(g) * u elementwise over `n`, where g/u are interleaved in `gu`.
cudaError_t RunDenseSwiGLUKernel(cudaStream_t stream, const __half *gu,
                                 __half *out, int I, int64_t n);

} // namespace dense
} // namespace lsxcommon
