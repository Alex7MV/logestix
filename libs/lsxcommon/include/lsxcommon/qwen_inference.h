#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/token_stream.h"

namespace lsxcommon {
namespace qwen {

// Result of the shared incremental-session policy (model-unification 2026-08):
// a request whose input_ids exactly match the previous call's generated tail
// continues the session KV cache from {kv_last_tok}; anything else starts a
// fresh session with the full request as prefill input.
struct SessionDecision {
    bool is_continuation = false;
    std::vector<int32_t> gen_input;  // {kv_last_tok} on continuation
};

inline SessionDecision DecideSession(bool kv_allocated, int32_t kv_last_tok,
                                     const std::vector<int32_t> &session_tail,
                                     const std::vector<int32_t> &req_input_ids)
{
    SessionDecision d;
    d.is_continuation = kv_allocated && !session_tail.empty() &&
                        req_input_ids == session_tail;
    if (d.is_continuation)
        d.gen_input = std::vector<int32_t>{kv_last_tok};
    else
        d.gen_input = req_input_ids;
    return d;
}

// ── Numeric kernels (TDD-testable, FP16 in / FP16 or FP32 out) ──────────────

// RMSNorm, offset-weight form: out = rmsnorm(x) * (1 + w), eps=1e-6.
// x: [rows, hidden], w: [hidden], out: [rows, hidden].
cudaError_t RunQwenRMSNorm(
    cudaStream_t stream, const __half *x, const __half *w,
    __half *out, int rows, int hidden, double eps);

// Computes cos/sin for partial rotary on the first `rotary_dim` dims.
// positions: [seq] int32, cos/sin: [seq, rotary_dim] FP16.
cudaError_t RunQwenRopeCosSin(
    cudaStream_t stream, const int32_t *positions, int seq, int rotary_dim,
    double theta, __half *cos, __half *sin);

// Partial rotary (HF rotate_half pairing: dims j and j+rot/2 form a pair):
// rotates the first `rotary_dim` of `head_dim` dims per (seq,head) pair,
// pass-through the rest. q: [seq, q_heads, head_dim],
// k: [seq, k_heads, head_dim] (GQA: q_heads=NH, k_heads=NKV).
cudaError_t RunQwenPartialRope(
    cudaStream_t stream, const __half *q, const __half *k,
    const __half *cos, const __half *sin,
    __half *q_out, __half *k_out,
    int seq, int q_heads, int k_heads, int head_dim, int rotary_dim);

// Same-heads convenience overload (q_heads = k_heads = heads).
cudaError_t RunQwenPartialRope(
    cudaStream_t stream, const __half *q, const __half *k,
    const __half *cos, const __half *sin,
    __half *q_out, __half *k_out,
    int seq, int heads, int head_dim, int rotary_dim);

// ── Inference entry points (operate on QwenPinnedWeights) ──────────────────

// Token embedding: out[t][h] = embed[ids[t]][h].
cudaError_t RunQwenEmbed(cudaStream_t stream, const __half *d_embed,
                         const int32_t *d_ids, __half *d_out,
                         int seq, int hidden);

// NVFP4 dense GEMM: y[t][n] = sum_k x[t*K+k] * dequant(packed[n*K+k], scales, gs).
// x: [seq, K] FP16; packed: [N*K/2] nibble-packed E2M1 (row-major [N, K]);
// scales: [N*K/16] E8M0 per-16 block floats; gs: global scale. y: [seq, N] FP16
// (FP32 accumulate). Dispatches to a shared-memory tile kernel for N>=8 && K>=16,
// else a per-element kernel for tiny shapes. Numeric contract: FP32 accumulate,
// k-order free.
// Blackwell tensor-core fast-path toggle (audit Phase 3): when enabled,
// RunQwenNvfp4Gemm dispatches to the sm_120a mxf4nvf4 kernel for supported
// shapes and falls back to the scalar path otherwise. Default off.
// Qwen-named shims over the model-neutral NVFP4 routing state
// (lsxcommon/lsx_inference_nvfp4.h). New code should use lsx_nvfp4:: names
// directly; these keep the qwen-family call sites stable.
inline void SetQwenTcFastMath(bool on) { lsx_nvfp4::SetLsxTcFastMath(on); }
inline bool QwenTcFastMath() { return lsx_nvfp4::LsxTcFastMath(); }
inline void SetQwenSeq1Gemv(bool on) { lsx_nvfp4::SetLsxSeq1Gemv(on); }
inline bool QwenSeq1Gemv() { return lsx_nvfp4::LsxSeq1Gemv(); }
inline bool QwenGemmLarge() { return lsx_nvfp4::LsxGemmLarge(); }
inline bool QwenGemmLargeActive() { return lsx_nvfp4::LsxGemmLargeActive(); }

// Model-independence guard (AGENTS.md: models are independent): every model
// asserts its OWN fast_math policy for the duration of Infer/SelfCheck via
// lsx_nvfp4::LsxFastMathScope — the last-constructed model's config must not
// leak into another model's kernel selection.
using QwenFastMathScope = lsx_nvfp4::LsxFastMathScope;

// NVFP4 GEMM moved to the model-neutral layer; qwen-named aliases keep the
// dense/moe call sites unchanged.
inline cudaError_t RunQwenNvfp4Gemm(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N, bool acc = false,
    const float *seg_wgs_dev = nullptr, const int *seg_bounds_dev = nullptr) {
    return lsx_nvfp4::RunLsxNvfp4Gemm(stream, x, packed, scales, global_scale,
                                      y, seq, K, N, acc, seg_wgs_dev,
                                      seg_bounds_dev);
}
inline cudaError_t RunQwenNvfp4GemmSeq1GemmV(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int K, int N, bool acc = false) {
    return lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(stream, x, packed, scales,
                                               global_scale, y, K, N, acc);
}

// QwenNvfp4Weight is an alias of lsx_nvfp4::LsxNvfp4Weight; the launcher is
// the single model-neutral definition (a same-signature inline shim here
// would make unqualified calls ambiguous).
using lsx_nvfp4::LaunchNvfp4Gemm;

// Full attention layer: input RMSNorm -> qkv projections (NVFP4 dequant GEMM)
// -> q/k per-head norm -> partial rope -> GQA causal attention -> output gate
// -> o_proj. All buffers caller-provided (see RunQwenGeneration arena layout).
// `kv_base` offsets the k/v write rows and the causal attention bound so the
// kernel can append into a persistent KV cache (incremental decode).
cudaError_t RunQwenFullAttention(
    cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qc, __half *d_q, __half *d_gate,
    __half *d_k, __half *d_v, __half *d_attn,
    __half *d_cos, __half *d_sin, int32_t *d_pos,
    int kv_base = 0);

// Linear (GatedDeltaNet) attention layer: in_proj qkv (NVFP4 dequant GEMM) ->
// causal conv1d -> gated delta rule (recurrent) -> RMSNormGated -> out_proj.
// Buffers caller-provided (halves unless noted):
//   d_qkv: [seq, KD*2+VD], d_z: [seq,VD], d_ab: [seq*nv*2] (a then b),
//   d_g: [seq*nv] float decay, d_beta: [seq*nv] float beta,
//   d_conv: [seq, KD*2+VD], d_core: [seq,VD], d_gated: [seq,VD].
// `kv_base` offsets the conv1d history read (rows [kv_base..kv_base+seq) of
// d_qkv) for incremental decode; `d_gd_state` (per-linear-layer slice of the
// GatedDeltaNet col accumulator) seeds the recurrence and is written back by
// the last row's block when non-null.
cudaError_t RunQwenLinearAttention(
    cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qkv, __half *d_z, __half *d_ab,
    float *d_g, float *d_beta,
    __half *d_conv, __half *d_core, __half *d_gated,
    int kv_base = 0, float *d_gd_state = nullptr);

// Linear-attention layer with a fused front-end (F1 qkv+conv1d+l2norm, F2
// z+a+b+prep_gating), reducing the split path's 8 front launches to 2. The
// gated-delta recurrence, RMSNormGated, and out_proj tail are unchanged.
// Call signature and scratch layout identical to RunQwenLinearAttention.
cudaError_t RunQwenFusedLinearAttnKernel(
    cudaStream_t stream,
    const QwenLayerWeights &lw, const QwenTopology &top, int seq,
    const __half *d_normed, __half *d_out,
    __half *d_qkv, __half *d_z, __half *d_ab,
    float *d_g, float *d_beta,
    __half *d_conv, __half *d_core, __half *d_gated,
    int kv_base = 0, float *d_gd_state = nullptr,
    bool use_sigmoid_gate = false);

// Scaled dot-product causal attention (GQA), FP16 in/out, FP32 accumulate.
// q: [seq, heads, head_dim]; k/v: [seq * kv_heads, head_dim] base rows where
// the kv row for attended index s is (s * kv_heads + kvh) * head_dim;
// kv_base offsets the attended rows (s in [0, kv_base+t]) for a persistent
// cache; out: [seq, heads, head_dim]. Dispatches to the block-wide chunked
// parallel-softmax kernel; this is the numeric entry for the kernel alone
// (the high-level RunQwenFullAttention adds projections/RoPE/gate).
cudaError_t RunQwenFullAttnKernel(
    cudaStream_t stream,
    const __half *q, const __half *k, const __half *v, __half *out,
    int seq, int heads, int head_dim, int kv_heads, float scale,
    int kv_base);

// Split-KV decode attention (seq == 1, long context): flash-decode two-kernel
// path. The legacy single-block kernel is memory-latency bound at one block
// per head — its per-step cost grows linearly with the KV length (~2.3 us per
// position per step on 35B-A3B). The split path fixes the chunk geometry
// (512-row chunks, ascending chunk combine) so the per-step cost stays flat
// as the context grows. Numerically independent from the legacy kernel
// (chunk-rescaled online softmax; FP16 tolerance-envelope parity, same
// contract class as the TC-batch amendment); run-to-run bits are identical.
cudaError_t RunQwenFullAttnKernelSplitKV(
    cudaStream_t stream,
    const __half *q, const __half *k, const __half *v, __half *out,
    int seq, int heads, int head_dim, int kv_heads, float scale,
    int kv_base);

// Dense SwiGLU MLP: gate=silu(x@Wg), up=x@Wu, inter=gate*up, out=inter@Wd.
cudaError_t RunQwenMLP(
    cudaStream_t stream,
    const QwenLayerWeights &lw, int seq,
    const __half *d_in, __half *d_out, __half *d_scratch);

// Fused MLP layer: takes the ALREADY post-RMSNormed input stream (d_in) and
// accumulates the down-projection INTO d_resid (the residual stream) via the
// acc-down GEMM, absorbing the separate residual add. d_scratch is [2*seq*I].
cudaError_t RunQwenFusedMLP(
    cudaStream_t stream, const QwenLayerWeights &lw,
    const QwenTopology &top, int seq,
    const __half *d_in, __half *d_resid, __half *d_scratch);

// Producer-fused SwiGLU + NVFP4 activation quant: y = silu(a)*b in fp16 AND
// its quantized form (packed nibbles + pow2-E4M3 scales, one scale per 16
// consecutive elements). Byte-identical to running qwen_swiglu_kernel and
// then the standalone act-quant pass on y. n must be a multiple of 16 for a
// full-coverage guarantee (tail groups are still handled correctly).
cudaError_t RunQwenSwigluQuant(
    cudaStream_t stream, const __half *a, const __half *b, __half *y,
    uint8_t *packed, uint8_t *scales, int64_t n);

// Internal: Precise-flavor producer-fused swiglu+quant (MoE routed path).
cudaError_t RunQwenSwigluPreciseQuantInternal(cudaStream_t stream,
                                              const __half *a, const __half *b,
                                              __half *y, uint8_t *packed,
                                              uint8_t *scales, int64_t n);

// ── Generation ──────────────────────────────────────────────────────────────

} // namespace qwen
} // namespace lsxcommon