#pragma once

// Device-kernel entry points for GigaChat3.5 elementwise gating math
// (Plan 2a). Compiled into lsxcommon_gigachat_cu; declared here for the .cu
// test/pipeline translation units.
#include <cstdint>

#include <cuda_runtime.h>

namespace lsxcommon::gigachat {

// ZeroCenteredGatedNorm over `rows` rows of `dim`:
//   xn[r]     = x[r] * rsqrt(mean(x[r]^2) + eps)
//   gate(r)   = Wdown @ silu(Wup @ xn[r])
//   out[r][i] = ZeroCenteredGatedNormElement(xn[r][i], w[i], gate(r)[i])
// `w` BF16 [dim]; `w_up` BF16 [16, dim]; `w_down` BF16 [dim, 16]; F32 compute.
cudaError_t RunZeroCenteredGatedNorm(const float *x, int rows, int dim,
                                     const uint8_t *w, const uint8_t *w_up,
                                     const uint8_t *w_down, float eps,
                                     float *out, cudaStream_t stream);

// GatedDeltaNet output norm (zero-centered sigmoid gated RMSNorm), one block
// per head:
//   xn = rmsnorm(o[h]) ; out[h][d] = GdnOutputNormElement(xn[d], w[d],
//   z[h][d], eps, scale)
// `w` BF16 [head_dim]; `z` is the per-head gate [heads, head_dim].
cudaError_t RunGdnOutputNorm(const float *o, const float *z, const uint8_t *w,
                             int heads, int head_dim, float eps, float scale,
                             float *out, cudaStream_t stream);

} // namespace lsxcommon::gigachat
