#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lsxcommon::alice {

// The workspace-allocator predicate. cudaSuccess (0) must read as SUCCESS, so
// this returns true only for a real failure — the opposite of how a raw
// cudaError_t behaves in a boolean context (0 is falsy, i.e. "no error", which
// is also falsy). That double negative is why it is a named function with its
// own case in test_alice_moe_kernels.cu: commit 397ab361 turned the pipeline's
// allocation macro's arms into bool constants and inverted the meaning, so the
// first *successful* malloc aborted the whole workspace and alice generated zero
// tokens (see docs/open-tasks/2026-11-06-alice-workspace-polarity.md).
constexpr bool WsAllocFailed(cudaError_t e) { return e != cudaSuccess; }


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
