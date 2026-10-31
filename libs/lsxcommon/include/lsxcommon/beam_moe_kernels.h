#pragma once

// Beam-1.0 family kernels: plain RMSNorm (row and per-head), rotate-half RoPE,
// the NoPE SSMax query scale, per-head sigmoid attention gate, sliding-window
// decode attention, the group-limited sigmoid router, the FP32 SwiGLU product,
// the depth-scaled residual add, the FP16 embedding lookup and the final-logit
// softcap. Model-neutral primitives (NVFP4 GEMV, the routed-expert FFN) come
// from the shared layers; only Beam's own arithmetic lives here.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon::beam {

// One routed slot: selected expert id + weight (already multiplied by
// routed_scaling_factor). Layout-compatible with moe::MoERoutedSlot (8 bytes).
struct alignas(8) BeamRoutedSlot {
    int32_t eid;
    float coeff;
};

cudaError_t RunBeamNorm(cudaStream_t stream, const __half *x, const __half *w,
                        __half *out, int rows, int hidden, float eps);

cudaError_t RunBeamHeadNorm(cudaStream_t stream, const __half *x,
                            const __half *w, __half *out, int rows, int heads,
                            int head_dim, float eps);

// In-place rotate-half RoPE over the full head_dim. `q` is [q_heads, head_dim]
// per token (seq==1), `k` is [k_heads, head_dim]; either may be null.
cudaError_t RunBeamRope(cudaStream_t stream, __half *q, __half *k, int seq,
                        int q_heads, int k_heads, int head_dim, float theta,
                        int pos);

// In-place Q *= (1 + (pos+1)/2048)^0.1 (FP32 product, one rounding).
cudaError_t RunBeamSsmScale(cudaStream_t stream, __half *q, int heads,
                            int head_dim, int pos);

// Decode attention for ONE token: q [heads, head_dim], KV cache rows
// [0..pos][kv_heads, head_dim] (pos-major, row stride kv_heads*head_dim),
// sliding window `window` (<=0 = full), scale at QK, then the per-head gate
// `attn *= sigmoid(gate + gate_bias[head])` before writing [heads*head_dim].
cudaError_t RunBeamAttnDecode(cudaStream_t stream, const __half *q,
                              const __half *k_cache, const __half *v_cache,
                              const __half *gate, const __half *gate_bias,
                              __half *out, int heads, int kv_heads,
                              int head_dim, int pos, int window, float scale);

// Group-limited sigmoid router (FP32 logits, correction bias for choice,
// top-2-per-group group score, topk_group groups, top-k experts, renormalized
// plain sigmoid weights * routed_scale). Selection is thread-0 only.
cudaError_t RunBeamRouter(cudaStream_t stream, const float *router_w,
                          const float *router_bias, const __half *x, int seq,
                          int E, int K, int H, int n_group, int topk_group,
                          float routed_scale, BeamRoutedSlot *d_slots);

// SwiGLU with the activation/product evaluated in FP32 then rounded to FP16.
cudaError_t RunBeamSwiGLU(cudaStream_t stream, const __half *gate,
                          const __half *up, __half *out, int64_t n);

// out = branch/sqrt(idx) + residual*sqrt((idx-1)/idx); each scaled operand is
// rounded to FP16 before the add (sglang/transformers depth-scaled residual).
cudaError_t RunBeamDepthResidual(cudaStream_t stream, const __half *branch,
                                 const __half *residual, __half *out, int n,
                                 int idx);

cudaError_t RunBeamSoftcap(cudaStream_t stream, float *logits, int n);

cudaError_t RunBeamEmbed(cudaStream_t stream, const __half *table,
                         const int32_t *ids, __half *out, int seq, int hidden);

} // namespace lsxcommon::beam
