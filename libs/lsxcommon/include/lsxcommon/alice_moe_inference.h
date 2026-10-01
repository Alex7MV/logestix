#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lsxcommon::alice {

// Gated full-attention core (no projections): splits the fused `qg` projection
// into query + output gate, applies zero-centred q/k RMSNorm, partial
// rotate-half RoPE on the first `rot` dims, causal GQA attention, then scales
// by sigmoid(output gate).
//   qg: [seq, 2*H*D]; k/v: [(kv_base+seq), KVH, D] (cache rows pre-normed);
//   q_norm_w/k_norm_w: [D]; cos/sin: [seq, rot]; out: [seq, H*D].
cudaError_t RunAliceGatedAttentionCore(
    cudaStream_t stream, const __half *qg, const __half *k, const __half *v,
    const __half *q_norm_w, const __half *k_norm_w, const __half *cos,
    const __half *sin, __half *out, int seq, int H, int KVH, int D, int rot,
    float scale, int kv_base, __half *scratch);

} // namespace lsxcommon::alice
