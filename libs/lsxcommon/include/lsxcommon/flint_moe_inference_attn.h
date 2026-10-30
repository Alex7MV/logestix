#pragma once

// Recursive Flint attention (family-owned, single-token decode): GQA with
// per-head QK RMSNorm applied by the caller, optional RoPE (skipped on the
// full-attention layers, which carry NO positional encoding), an optional
// sliding window and the learned per-head sink.
//
// The sink follows the vendor's FA4 learned-sink fix exactly: it is one extra
// softmax COLUMN — the max and the denominator include exp(sink), the numerator
// gets no value contribution.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lsxcommon::flint {

// Appends nothing: the caller has already written the roped (or unroped) k and
// the v into the cache at `pos`. q: [heads, head_dim] FP16 for the new token.
// k_cache/v_cache: [capacity, kv_heads, head_dim] FP16 pos-major.
// sink: FP16 [heads] (never null). window: 0 = full attention, >0 = the layer
// attends to positions [pos - window + 1, pos].
// out: [heads * head_dim] FP16, written in place.
cudaError_t RunFlintAttnDecode(cudaStream_t stream, const __half *q, int pos,
                               const __half *sink, int heads, int kv_heads,
                               int head_dim, int window, float scale,
                               const __half *k_cache, const __half *v_cache,
                               __half *out);

} // namespace lsxcommon::flint
