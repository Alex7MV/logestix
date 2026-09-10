#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/gemma_moe_load.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/qwen_moe_inference.h"

namespace lsxcommon {
namespace gemma {

// ── Norms ───────────────────────────────────────────────────────────────────
// out[r] = rmsnorm(x[r]) * (1 + w), rows x hidden, FP16 storage / FP32 math.
cudaError_t RunGemmaRMSNorm(const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_w, __nv_bfloat16 *d_out,
                            int rows, int hidden, float eps, cudaStream_t stream);

// ── RoPE ────────────────────────────────────────────────────────────────────
// cos/sin tables [seq][rotary_dim/2] for positions pos[t], exponent base theta.
cudaError_t RunGemmaRopeCosSin(const int32_t *d_pos, __nv_bfloat16 *d_cos,
                               __nv_bfloat16 *d_sin, int seq, int rotary_dim,
                               double theta, cudaStream_t stream);

// Partial rotary embedding over packed [seq, heads, head_dim]; rotates the
// first rotary_dim dims of every head with HF rotate_half pairing.
cudaError_t RunGemmaPartialRope(const __nv_bfloat16 *d_x, const __nv_bfloat16 *d_cos,
                                const __nv_bfloat16 *d_sin, __nv_bfloat16 *d_out, int seq,
                                int heads, int head_dim, int rotary_dim,
                                cudaStream_t stream);

// ── Attention ───────────────────────────────────────────────────────────────
// Causal attention over a KV cache. One kernel serves both layer kinds:
//   window > 0  -> sliding band mask (slot >= pos_base+t-window+1)
//   window <= 0 -> full causal context
// "Global" KV geometry (full-attention layers) is just kv_heads/global_head_dim
// parameters — the layout is identical.
//
// Layouts (stride invariant):
//   q/out row (t,h):  [(t*q_heads + h) * head_dim]
//   k/v  row (s,kvh): [(s*kv_heads + kvh) * head_dim], kvh = h/(q_heads/kv_heads)
//
// Determinism: fixed slot iteration order; block reductions are tree-shaped
// with __syncthreads() barriers and no read-modify-write races.
cudaError_t RunGemmaAttention(const __nv_bfloat16 *d_q, const __nv_bfloat16 *d_k,
                              const __nv_bfloat16 *d_v, __nv_bfloat16 *d_out, int seq,
                              int total_slots, int pos_base, int q_heads,
                              int kv_heads, int head_dim, int window,
                              float scale, cudaStream_t stream);

// Split-KV decode attention (seq == 1, long context): flash-decode two-kernel
// path — a port of the qwen split-KV design (chunk 512, chunk-rescaled online
// softmax) to the gemma BF16 geometry. The legacy one-block-per-head kernel is
// memory-latency bound and its per-step cost grows linearly with the attended
// span; the split path restores memory-level parallelism through the extra
// chunk axis so the per-step cost stays flat as the context grows. Sliding
// layers pass their window so chunks outside the band contribute nothing
// (out-of-band rows never enter any reduction, matching the legacy kernel's
// band predicate). Numerics: chunk-rescale grouping differs from the legacy
// two-pass kernel only by FP reduction grouping (same tolerance-envelope
// contract class); run-to-run bits identical (ascending chunk order).
cudaError_t RunGemmaAttentionSplitKV(
    const __nv_bfloat16 *d_q, const __nv_bfloat16 *d_k,
    const __nv_bfloat16 *d_v, __nv_bfloat16 *d_out, int seq,
    int total_slots, int pos_base, int q_heads, int kv_heads, int head_dim,
    int window, float scale, cudaStream_t stream);

// ── Dense MLP + MoE ─────────────────────────────────────────────────────────
// out = down @ (gelu_tanh(gate @ x) * (up @ x)); weights [rows, hidden]
// row-major; x [seq, hidden]; out [seq, hidden].
cudaError_t RunGemmaDenseMLP(const __nv_bfloat16 *d_gate, const __nv_bfloat16 *d_up,
                             const __nv_bfloat16 *d_down, int64_t I, int64_t H,
                             const __nv_bfloat16 *d_x, __nv_bfloat16 *d_out, int seq,
                             cudaStream_t stream);

// Router: logits[e] = router_proj[e] . x; top-k selection with deterministic
// lower-index tie-break; weights = softmax(selected logits). When
// d_per_expert_scale is non-null each selected weight is additionally
// multiplied by per_expert_scale[e]. Outputs d_sel_experts [seq, top_k] and
// d_sel_weights [seq, top_k].
cudaError_t RunGemmaRouterTopK(const __nv_bfloat16 *d_router_proj,
                               const float *d_per_expert_scale,
                               const __nv_bfloat16 *d_normed, int E, int H,
                               int top_k, int32_t *d_sel_experts,
                               float *d_sel_weights, int seq,
                               cudaStream_t stream);

// Round 21: batched weight-streaming NVFP4 GEMV for the gemma routed-expert
// chain at seq==1. Replaces the dequant-to-BF16 + cuBLAS detour with direct
// streaming of the three SEPARATED slabs of the staged expert slot
// (w1 gate | w2 down | w3 up packed; s1|s2|s3 scales). The geGLU stage
// applies gelu_tanh(g)*u in FP16 and the weighted reduce accumulates in an
// FP32 buffer the caller converts/casts (gemma runs a BF16 pipeline).
// Requires K_gate_up % 32 == 0 && K_down % 32 == 0. Run-to-run deterministic.
cudaError_t RunRoutedGemvBatchSeg(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const lsxcommon::qwen_moe::QwenMoeGvSlotDev *slots_dev, int k_slots,
    int64_t off_gate_p, int64_t off_gate_s,
    int64_t off_up_p, int64_t off_up_s,
    int64_t off_down_p, int64_t off_down_s,
    int n_gate_up, int k_gate_up,
    int n_down, int k_down,
    bool geglu,
    const __half *d_x,
    __half *ygu,       // [k_slots * 2 * n_gate_up] gate|up halves per slot
    __half *hbuf,      // [k_slots * n_gate_up]
    float *acc_f32);   // [n_down] fp32 accumulator

// Weighted expert-output combine: out[h] = router_scale[h] *
// sum_k w_k * expert_out_k[h]. Null d_router_scale skips the channel scale.
cudaError_t RunGemmaMoECombine(const __nv_bfloat16 *const *d_expert_outs,
                               const float *d_weights, const float *d_router_scale,
                               __nv_bfloat16 *d_out, int hidden, int k, cudaStream_t stream);

// ── Generation ──────────────────────────────────────────────────────────────
struct GemmaGenerationResult {
    bool ok = false;
    std::string error_msg;
    std::vector<int32_t> tokens;
    std::vector<float> final_logits;   // last decode step (softcapped)
    double ttft_ms = 0.0;
    double total_ms = 0.0;
    bool eos_terminated = false;
};

// Full generation loop: prefill(input) + up to max_gen_tokens decode steps on
// ONE stream (no cudaDeviceSynchronize in the loop; per-token D2H argmax via
// pinned handshake). kv must be pre-allocated for prompt+gen capacity.
// Applies final RMSNorm, tied-head logits and softcap tanh(l/cap)*cap.
//
// Persistent generation context: cuBLAS handle + capacity-grown workspace +
// routed-expert device slot cache. Owned by the model; created lazily,
// single-threaded (one consumer). Plain host-safe members only (no absl).
struct GenWorkspace {
    __nv_bfloat16 *d_x = nullptr;        // [seq0][H]
    __nv_bfloat16 *d_normed = nullptr;   // [seq0][H]
    __nv_bfloat16 *d_tmp = nullptr;      // [seq0][q_rows_max]
    __nv_bfloat16 *d_qr = nullptr;       // [seq0][q_rows_max]
    __nv_bfloat16 *d_kr = nullptr;       // [seq0][kv_rows_max]
    __nv_bfloat16 *d_tmp2 = nullptr;     // [seq0][kv_rows_max]
    __nv_bfloat16 *d_vr = nullptr;
    __nv_bfloat16 *d_cos = nullptr;      // [seq0][rot_half]
    __nv_bfloat16 *d_sin = nullptr;
    int32_t *d_pos = nullptr;     // [seq0]
    float *d_logits = nullptr;    // [vocab]
    int32_t *d_argmax = nullptr;
    int32_t *d_sel = nullptr;     // [seq0*K]
    float *d_selw = nullptr;      // [seq0*K]
    int32_t *h_sel = nullptr;     // pinned
    float *h_selw = nullptr;      // pinned
    int32_t *h_arg = nullptr;     // pinned
    uint8_t *d_p1 = nullptr, *d_s1 = nullptr, *d_p2 = nullptr;
    uint8_t *d_s2 = nullptr, *d_p3 = nullptr, *d_s3 = nullptr;
    __nv_bfloat16 *d_w1 = nullptr, *d_w3 = nullptr, *d_w2 = nullptr;
    __nv_bfloat16 *d_eout = nullptr;     // [1][H]
    __nv_bfloat16 *d_acc = nullptr;      // [seq0][H]
    __nv_bfloat16 *d_gg = nullptr;       // [seq0][I_max] dense-branch gate/geglu
    __nv_bfloat16 *d_gu = nullptr;       // [seq0][I_max] dense-branch up
    __nv_bfloat16 *d_eg = nullptr;       // [MI] expert gate/geglu
    __nv_bfloat16 *d_eu = nullptr;       // [MI] expert up
    // Batched decode (seq==1) staging: dequant rows [K][...] for all top-K
    // experts in one gather launch; batched GEMM outputs.
    __nv_bfloat16 *d_bw1 = nullptr;      // [K][MI*H]
    __nv_bfloat16 *d_bw3 = nullptr;      // [K][MI*H]
    __nv_bfloat16 *d_bw2 = nullptr;      // [K][H*MI]
    __nv_bfloat16 *d_beg = nullptr;      // [K][MI]  gate projections
    __nv_bfloat16 *d_beu = nullptr;      // [K][MI]  up projections
    __nv_bfloat16 *d_beo = nullptr;      // [K][H]   down outputs
    float *d_bselw = nullptr;            // [K] selected weights
    // Two-phase logits/argmax scratch: per-row then per-block winners.
    // Sized [V + kLogitsMaxBlocks].
    float *d_lg_best = nullptr;
    int32_t *d_lg_idx = nullptr;
    // NVFP4 dense-MLP chain scratch (FP16): activation mirror + projections.
    __half *d_xh = nullptr;      // [seq0][H]
    __half *d_mlp_g = nullptr;   // [I_max]
    __half *d_mlp_u = nullptr;   // [I_max]
    __half *d_mlp_h = nullptr;   // [I_max]
    __half *d_mlp_y = nullptr;   // [H]
    __half *d_tmpf = nullptr;    // [q_rows_max] FP16 attention output (o_proj in)

    size_t seq0 = 0, q_rows_max = 0, kv_rows_max = 0, rot_half_max = 0;
    int K = 0;

    void Free()
    {
        auto F = [](void **p) { if (*p) { cudaFree(*p); *p = nullptr; } };
        F((void **)&d_x); F((void **)&d_normed); F((void **)&d_tmp);
        F((void **)&d_qr); F((void **)&d_kr); F((void **)&d_vr);
        F((void **)&d_tmp2);
        F((void **)&d_cos); F((void **)&d_sin); F((void **)&d_pos);
        F((void **)&d_logits); F((void **)&d_argmax);
        F((void **)&d_sel); F((void **)&d_selw);
        F((void **)&d_p1); F((void **)&d_s1); F((void **)&d_p2);
        F((void **)&d_s2); F((void **)&d_p3); F((void **)&d_s3);
        F((void **)&d_w1); F((void **)&d_w3); F((void **)&d_w2);
        F((void **)&d_eout); F((void **)&d_acc);
        F((void **)&d_gg); F((void **)&d_gu);
        F((void **)&d_eg); F((void **)&d_eu);
        F((void **)&d_bw1); F((void **)&d_bw3); F((void **)&d_bw2);
        F((void **)&d_beg); F((void **)&d_beu); F((void **)&d_beo);
        F((void **)&d_bselw);
        F((void **)&d_lg_best); F((void **)&d_lg_idx);
        F((void **)&d_xh); F((void **)&d_mlp_g); F((void **)&d_mlp_u);
        F((void **)&d_mlp_h); F((void **)&d_mlp_y); F((void **)&d_tmpf);
        if (h_sel) { cudaFreeHost(h_sel); h_sel = nullptr; }
        if (h_selw) { cudaFreeHost(h_selw); h_selw = nullptr; }
        if (h_arg) { cudaFreeHost(h_arg); h_arg = nullptr; }
    }

    ~GenWorkspace() { Free(); }
};

struct GemmaGenContext {
    ~GemmaGenContext()
    {
        if (cublas) cublasDestroy(cublas);
        if (ec.d_base) cudaFree(ec.d_base);
        delete[] ec.h_slot_of;
        delete[] ec.h_slot_owner;
        delete[] ec.h_lru;
        ec = {};
        if (ev_router) cudaEventDestroy(ev_router);
    }
    cublasHandle_t cublas = nullptr;
    GenWorkspace ws;
    bool ws_ready = false;
    size_t seq0_cap = 0, q_rows_cap = 0, kv_rows_cap = 0, rot_half_cap = 0;
    int K_cap = 0;
    // Routed-expert device slot cache: [slots] × (pack_stride + scale_stride),
    // one slot per flat expert id. Bookkeeping per gpu::ExpertDeviceCache.
    gpu::ExpertDeviceCache ec{};
    int64_t pack_stride = 0;
    int64_t scale_stride = 0;
    // Router-handoff fence: host busy-polls this event per layer instead of
    // a full cudaStreamSynchronize (created with the expert cache).
    cudaEvent_t ev_router = nullptr;
};

GemmaGenContext *CreateGemmaGenContext();
void DestroyGemmaGenContext(GemmaGenContext *ctx);

// Single-expert NVFP4 elementwise dequant (unified Nvfp4DequantKernel).
// Exposed for the batched-vs-single parity unit test.
cudaError_t RunGemmaSingleDequant(cudaStream_t stream, const uint8_t *packed,
                                  int64_t packed_bytes, const uint8_t *scales,
                                  int64_t scales_bytes, float global_scale,
                                  __nv_bfloat16 *out, int64_t num_elements);

// Dense-MLP NVFP4 weight-streaming chain: y = down @ (gelu_tanh(gate @ x) *
// (up @ x)). x is FP16 [H]; gate/up weights are [I, H], down is [H, I] —
// all NVFP4-packed with per-16 E8M0-style scales and a float global scale.
// Intermediate buffers g_out/u_out/h_out are FP16 [I] (caller-owned scratch);
// the result y is FP16 [H]. seq==1 contract class (tolerance-envelope);
// run-to-run deterministic (fixed ascending-d accumulation).
cudaError_t RunGemmaDenseMlpNvfp4Chain(
    cudaStream_t stream, const uint8_t *gate_packed, const uint8_t *gate_scales,
    float gate_gs, const uint8_t *up_packed, const uint8_t *up_scales,
    float up_gs, const uint8_t *down_packed, const uint8_t *down_scales,
    float down_gs, const __half *x, __half *g_out, __half *u_out,
    __half *h_out, __half *y, int H, int I);

// FP16 -> BF16 elementwise cast (elementwise scratch plumbing).
cudaError_t RunGemmaF16ToBf16(cudaStream_t stream, const __half *src,
                              __nv_bfloat16 *dst, int64_t n);

// Gather-dequant of @p K experts into [K][num_elements] rows. Bit-identical
// element mapping to RunGemmaSingleDequant (same decode LUTs, same scale
// indexing); one launch instead of K. Pointer arrays are passed by value.
struct GemmaDequantBatch {
    const uint8_t *packs[16];
    const uint8_t *scales[16];
    float global_scales[16];
};
cudaError_t RunGemmaBatchedDequant(cudaStream_t stream,
                                   const GemmaDequantBatch &batch,
                                   int64_t num_elements_per_expert, int k,
                                   __nv_bfloat16 *out_rows);

// Two-phase lm-head logits + deterministic argmax (test/production hook).
// Phase-1 rows keep strict ascending-d fp32 accumulation; phase-2 reduces
// block winners in ascending block order. Requires scratch buffers sized
// [vocab + 4096].
cudaError_t RunGemmaLogitsArgmax(cudaStream_t stream,
                                 const __nv_bfloat16 *d_embed,
                                 const __nv_bfloat16 *d_x, int vocab,
                                 int hidden, float cap, int32_t *d_argmax,
                                 float *d_maxlogit, float *d_logits,
                                 float *d_blkbest, int32_t *d_blkidx);

cudaError_t RunGemmaGeneration(cudaStream_t stream, const GemmaPinnedWeights &pw,
                               GemmaExpertDb &db, const std::vector<int32_t> &input_ids,
                               int max_gen_tokens, int eos_id, float softcap,
                               GemmaGenerationResult &out, GemmaKvCache &kv,
                               GemmaGenContext *ctx = nullptr);

} // namespace gemma
} // namespace lsxcommon

