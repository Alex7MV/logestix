#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace lsxcommon::alice {

// Zero-centred (or plain) RMSNorm over the last dim.
//   zero_centered: scale = 1 + w; otherwise scale = w.
cudaError_t RunAliceZcRmsNorm(cudaStream_t stream, const __half *x,
                              const __half *w, __half *out, int rows,
                              int hidden, float eps, bool zero_centered);

// AttnRes depth-softmax mixer (_depth_softmax_mix).
//   sources: [n_sources, hidden] FP16; norm_w/proj_w: [hidden] FP16.
//   logit_s = dot(rmsnorm(sources[s]) * norm_w, proj_w); out = softmax(logits)·sources.
cudaError_t RunAliceDepthMix(cudaStream_t stream, const __half *sources,
                             int n_sources, const __half *norm_w,
                             const __half *proj_w, __half *out, int hidden,
                             float eps);

// Sigmoid top-k router with correction bias. scores = sigmoid(x·gateᵀ) + bias;
// top-k indices selected on scores, weights = scores[idx] renormalized.
cudaError_t RunAliceRouterTopK(cudaStream_t stream, const __half *x,
                               const __half *gate, const __half *bias,
                               int hidden, int n_experts, int top_k,
                               int32_t *out_idx, float *out_w,
                               float *out_logits);

// out = silu(a) * b (elementwise).
cudaError_t RunAliceSwiGLU(cudaStream_t stream, const __half *a,
                           const __half *b, __half *out, int64_t n);

// Depthwise causal conv1d (kernel `k`, zero left-pad) + SiLU over each of `C`
// channels. `history` is [C, k-1] floats holding the previous raw inputs
// (oldest first); updated in place to the last k-1 inputs of this chunk.
// x/out: [seq, C] FP16.
cudaError_t RunAliceKdaConv(cudaStream_t stream, const __half *x,
                            const __half *conv_w, float *history, __half *out,
                            int seq, int C, int k);

// KDA gated-delta recurrence (repeat factor 1: one K head per V head).
//   q/k: [seq, H, KD] FP16 (post-conv, pre-L2norm)
//   v:   [seq, H, VD] FP16 (post-conv)
//   alpha: [seq, H, KD] FP16 (f_b output); beta: [seq, H] FP16
//   a_log: [H] FP16; dt_bias: [H, KD] FP16
//   state: [H, KD, VD] F32 (persistent; zero-initialized for a fresh session)
//   out: [seq, H, VD] FP16
cudaError_t RunAliceKdaRecurrence(cudaStream_t stream, const __half *q,
                                  const __half *k, const __half *v,
                                  const __half *alpha, const __half *beta,
                                  const __half *a_log, const __half *dt_bias,
                                  float *state, __half *out, int seq, int H,
                                  int KD, int VD);

} // namespace lsxcommon::alice
