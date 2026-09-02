#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/sampling.h"

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
void SetQwenTcFastMath(bool on);
bool QwenTcFastMath();

// seq==1 decode-path GEMV toggle (round-18 lever). Opt-in process-wide
// flag; when enabled together with fast-math, RunQwenNvfp4Gemm routes
// eligible seq==1 shapes (K % 32 == 0, K >= 256, <= 12 MiB packed+scales,
// no fused segments) to the weight-streaming GEMV kernel.
void SetQwenSeq1Gemv(bool on);
bool QwenSeq1Gemv();

cudaError_t RunQwenNvfp4Gemm(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int seq, int K, int N, bool acc = false,
    // Optional per-segment global scales for the TC epilogue: two ranges
    // split at N/2 when bounds is null, else four bounds-driven ranges.
    // Ignored by the scalar fallback path.
    const float *seg_wgs_dev = nullptr, const int *seg_bounds_dev = nullptr);

// seq==1 decode-path weight-streaming NVFP4 GEMV (round-18 lever): y[n] =
// sum_k x[k] * dequant(packed[n*K+k]) with FP32 accumulate. Streams packed
// E2M1 weights straight from global memory (coalesced uint4 loads) -- no
// shared-memory slab staging. Requires K % 32 == 0; k-order free (tolerance
// class), run-to-run deterministic. External linkage for direct testing.
cudaError_t RunQwenNvfp4GemmSeq1GemmV(
    cudaStream_t stream, const __half *x,
    const uint8_t *packed, const uint8_t *scales, float global_scale,
    __half *y, int K, int N, bool acc = false);

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

// Fused decoder (seq=1) full-attention layer: 3 fused kernels (qkv -> online
// softmax -> o_proj) replace the ~9 split launches for a SINGLE token. x is the
// embedded token [H]; kv_base appends its k/v and bounds attended rows
// [0 .. kv_base]. kv_k/kv_v: persistent cache, row r is (r*NKV+kvh)*HD. out [H].
// KNOWN-BROKEN: diverges on the real Qwen3.5-9B checkpoint (selfcheck garbled;
// unwired from generation in 19e8d4f). TEST-ONLY — do NOT wire into generation
// without re-validating full-layer parity on the real checkpoint.
cudaError_t RunQwenFusedDecodeAttnKernel(
    cudaStream_t stream, const __half *x, const QwenLayerWeights &lw,
    const QwenTopology &top, int kv_base,
    __half *kv_k, __half *kv_v, __half *out);

// Fused decoder stage 1 (qkv): applies input RMSNorm to x ([H]), then per-head
// inline NVFP4 q/k/v/gate projection, per-head RMSNorm(q) and RMSNorm(k),
// inline RoPE (rotate-half pairing on the first rotary_dim dims), appends this
// token's k/v to the persistent KV row kv_base, and publishes q_gl/gate_gl
// [NH*HD] for the fused attention stage.
cudaError_t RunQwenFusedQkv(
    cudaStream_t stream, const __half *x,
    const QwenNvfp4Weight &qw, const QwenNvfp4Weight &kw,
    const QwenNvfp4Weight &vw,
    const __half *q_norm, const __half *k_norm,
    double theta, int rotary_dim, int kv_base,
    int NH, int NKV, int HD, int H, float eps,
    const __half *input_norm,
    __half *kv_k, __half *kv_v, __half *q_gl, __half *gate_gl);

// Fused decoder stage 2 (attention): block-wide online-softmax causal GQA over
// attended KV rows [0..kv_base], per-head output gate (sigmoid). q_gl/gate_gl
// [NH*HD], kv_k/kv_v persistent cache [..NKV*HD] rows; out_h [NH*HD].
cudaError_t RunQwenFusedAttn(
    cudaStream_t stream,
    const __half *q_gl, const __half *gate_gl,
    const __half *kv_k, const __half *kv_v, __half *out_h,
    int kv_base, int NH, int NKV, int HD, float scale);

// Fused decoder stage 3 (o_proj): accumulate each head's o_proj contribution
// into out_acc[H] (float atomicAdd), then convert to out [H]. out_h [NH*HD];
// o is the [NH*HD, H] NVFP4 output weight.
cudaError_t RunQwenFusedOProj(
    cudaStream_t stream, const __half *out_h,
    const QwenNvfp4Weight &o, float *out_acc, __half *out,
    int NH, int HD, int H);

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

struct QwenGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    int32_t first_token = -1;
    std::vector<float> final_logits;
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    std::uint64_t resident_bytes = 0;
    bool eos_terminated = false;
};

// Persistent per-session incremental KV cache (spec item 15). A non-null
// `kv` handed to RunQwenGeneration turns the generation loop into a uniform
// one-token-per-step stream: fresh sessions allocate (kv_rows == 0), later
// calls continue at kv->kv_rows with prompt = {kv->last_tok}. Full-attention
// k/v rows live in [max_total][NKV*HD] buffers (one per full-attn layer);
// linear-attention layers keep their raw qkv history [max_total][KD*2+VD]
// (for the causal conv1d) plus the persistent GatedDeltaNet col accumulator
// [nv*vd*kd] floats. Growth is handled by EnsureCapacity.
struct QwenKvCache {
    __half *d_kv_k = nullptr;        // [num_full][max_total][NKV*HD]
    __half *d_kv_v = nullptr;        // [num_full][max_total][NKV*HD]
    __half *d_qkv = nullptr;         // [num_linear][max_total][KD*2+VD]
    float  *d_col = nullptr;         // [num_linear][nv*vd*kd]  (zero-initialized)
    int32_t last_tok = -1;
    int kv_rows = 0;                 // rows written so far (== global pos base)
    int max_total = 0;
    int num_linear_layers = 0;
    int num_full_layers = 0;
    int64_t cache_bytes = 0;

    bool allocated() const { return d_kv_k != nullptr; }
    std::uint64_t Bytes() const { return static_cast<std::uint64_t>(cache_bytes); }

    cudaError_t Allocate(const QwenPinnedWeights &pw, int max_total,
                         cudaStream_t stream);
    cudaError_t EnsureCapacity(const QwenPinnedWeights &pw, int needed,
                               cudaStream_t stream);
    void Free();
};

// Full greedy multi-token generation over the frozen Qwen3.5-9B hybrid loop:
// per decoder layer, RMSNorm input -> (linear or full attention) -> residual ->
// RMSNorm -> MLP -> residual; final norm -> lm_head -> greedy argmax.
// A non-null `kv` selects the incremental path (spec item 15): each step runs
// exactly one token at the persistent kv_base and the recurrent GatedDeltaNet
// state is carried across calls; `nullptr` keeps today's full-recompute path.
cudaError_t RunQwenGeneration(
    cudaStream_t stream,
    const QwenPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new,
    int eos_id,
    QwenGenerationResult &result,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler(),
    QwenKvCache *kv = nullptr);

} // namespace qwen
} // namespace lsxcommon