#pragma once

// MiMo-V2.6-Flash-RL attention (fused-QKV, asymmetric QK/V head dims, partial
// rotate-half RoPE, GQA, causal or sliding-window, optional learned sink bias).
// Correctness-first prefill kernel; the model drives decode through it with a
// one-token query against the KV cache.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lsxcommon {
namespace mimo {

struct MimoAttnShape {
    int heads = 64;
    int kv_heads = 4;
    int head_dim = 192;
    int v_head_dim = 128;
    int rope_dim = 64;
    bool is_swa = false;
    int window = 128;
    float value_scale = 0.707f;
    float scale = 0.07216878364870322f;
    double rope_theta = 1e7;
    int q_size = 12288;
    int k_size = 768;
    int v_size = 512;
};

// qkv: [seq, q_size + 2*k_size] FP16 (fused projection, v unscaled, k unroped).
// sink: FP16 [heads] or nullptr. out: [seq, heads * v_head_dim] FP16.
cudaError_t RunMimoAttentionPrefill(cudaStream_t stream, const __half *qkv,
                                    const __half *sink, int seq,
                                    const MimoAttnShape &shape, __half *out);

// Decode step with a KV cache. qkv_token: [q_size + 2*k_size] FP16 for the new
// token at position `pos`. k_cache: [cap, kv_heads, head_dim] FP16 (roped),
// v_cache: [cap, kv_heads, v_head_dim] FP16 (value-scaled). Appends the new k/v
// at index `pos`, then attends over [j0..pos] (j0 = pos-window+1 for SWA).
// out: [heads * v_head_dim] FP16.
cudaError_t RunMimoAttentionDecode(cudaStream_t stream, const __half *qkv_token,
                                   int pos, const __half *sink,
                                   const MimoAttnShape &shape, __half *k_cache,
                                   __half *v_cache, __half *out);

} // namespace mimo
} // namespace lsxcommon
