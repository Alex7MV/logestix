#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/deepseek_norm_kernels.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/log.h"
#include "lsxcommon/deepseek_inference_csa.h"
#include "deepseek_inference_load.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace lsxcommon {
namespace gpu {

// Forward declarations from other .cu files
extern __global__ void dequant_nvfp4_to_half_kernel(
    const uint8_t* __restrict__, const uint8_t* __restrict__, float,
    __half* __restrict__, int);

// half_to_float_kernel / float_to_half_kernel are defined once in
// deepseek_inference_load.cuh (included above) and shared by every .cu
// file in this pipeline — see that header for why.

// float → half conversion kernel (extern — also defined in deepseek_inference_moe.cu)
__global__ void float_to_half_f2h_kernel(const __half* __restrict__ src,
                                          float* __restrict__ dst, int n)
{
    // half→float conversion (for shared FFN FP16→FP32)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

// Clamp float → half with overflow protection (prevents Inf/NaN from FP16 range)
__global__ void clamp_f32_to_fp16(const float* __restrict__ src,
                                    __half* __restrict__ dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    // Clamp to FP16 representable range: [-65504, 65504]
    v = fmaxf(-65504.0f, fminf(65504.0f, v));
    dst[i] = __float2half(v);
}

static __host__ cudaError_t run_rmsnorm(
    cudaStream_t stream, const __half* d_input, const __half* d_weight,
    __half* d_output, float* d_scratch, float* d_gamma_scratch, int dim, float eps)
{
    int n = dim;
    int block = 256;
    int grid = (n + block - 1) / block;

    half_to_float_kernel<<<grid, block, 0, stream>>>(d_input, d_scratch, n);
    half_to_float_kernel<<<grid, block, 0, stream>>>(d_weight, d_gamma_scratch, n);

    cudaError_t err = rmsnorm_float(stream, 1, dim, d_scratch,
                                     d_gamma_scratch, eps, d_scratch);
    if (err != cudaSuccess) return err;

    // Clamp FP32→FP16 (prevent overflow)
    clamp_f32_to_fp16<<<grid, block, 0, stream>>>(d_scratch, d_output, n);
    return cudaGetLastError();
}

// From deepseek_inference_moe.cu
cudaError_t route_token(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    const int64_t *h_tid2eid, int tid2eid_stride, int token_id,
    const float *h_gate_bias,
    int topk,
    int *h_top_indices, float *h_top_weights);

cudaError_t run_shared_experts_ffn(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w2, const uint8_t *d_w3,
    const uint8_t *d_w1_scales, const uint8_t *d_w2_scales, const uint8_t *d_w3_scales,
    float w1_gs, float w2_gs, float w3_gs,
    int64_t w1_bytes, int64_t w2_bytes, int64_t w3_bytes,
    float *d_intermediate_f32, float *d_gate_f32,
    float *d_dequant, __half *d_output);

cudaError_t run_routed_experts(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_normed,
    const ExpertEntry *db_entries, int layer, int experts_per_layer,
    const int *h_top_indices, const float *h_top_weights, int topk,
    float *d_output,
    float *d_gate_f32, float *d_up_f32,
    float *d_dequant,
    float *d_act_quant,
    float *d_swiglu_out,
    float *d_act_scales,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    int64_t pack_stage_bytes, int64_t scale_stage_bytes);

// run_mla_attention is declared in deepseek_inference_load.cuh (included
// above) — that header's signature now matches the real definition in
// deepseek_inference_attn.cu (float* d_dequant), so this file no longer
// needs its own local, previously-stale (__half*) redeclaration.

// From deepseek_inference_attn.cu
cudaError_t run_csa_attention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const LayerWeights& lw,
    const __half* d_x,
    __half* d_kv_cache_layer,
    CSAState& csa_state,
    int start_pos,
    __half* d_q, __half* d_qr, __half* d_qr2,
    __half* d_kv, __half* d_kv2,
    __half* d_o, __half* d_olora, __half* d_out,
    float* d_dequant,
    float* d_norm_scratch, float* d_norm_gamma_scratch);

// From deepseek_inference_head.cu
cudaError_t run_lm_head(
    cublasHandle_t cublas_h,
    cudaStream_t stream,
    const __half *d_head_weight,
    const __half *d_hidden,
    int vocab_size,
    int hidden_dim,
    float *d_logits,
    int32_t *d_next_token,
    float *d_next_maxv);

// Sanity ceiling for a model-reported vocab_size. d_logits is now allocated
// dynamically to the actual vocab_size (see RunInferenceModel), but any
// model reporting an implausibly large vocabulary fails loudly at load time
// instead of proceeding with a runaway allocation/GEMM size.
constexpr int kMaxVocabSize = 200000;

InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db)
{
    InferenceResult result;
    auto mp = model_path;

    // 1. Log the incoming prompt tokens
    std::string tok_line;
    for (size_t i = 0; i < input_ids.size(); ++i)
        tok_line += std::to_string(input_ids[i]) + " ";
    log::info("[inference] prompt=" + std::to_string(input_ids.size())
        + " tokens: " + tok_line);

    // 2. Load topology
    auto topo = load_topology(mp);
    if (topo.num_layers == 0) {
        result.error_msg = topo.error_msg.empty()
            ? "Empty topology" : topo.error_msg;
        return result;
    }
    log::info("[inference] layers=" + std::to_string(topo.num_layers)
        + " experts_per_layer=" + std::to_string(topo.experts_per_layer)
        + " total=" + std::to_string(topo.total_experts));

    // Load pinned weights FIRST (before expert index, to avoid fd exhaustion)
    // Execution block — all INFER_CHECK gotos jump to cleanup: below
    {

#define INFER_CHECK(call)                                              \
    do {                                                               \
        cudaError_t _e = call;                                         \
        if (_e != cudaSuccess) {                                       \
            result.error_msg = std::string(#call)                       \
                + " failed: " + cudaGetErrorString(_e);                \
            goto cleanup;                                              \
        }                                                              \
    } while (0)

#define CUBLAS_INFER_CHECK(call)                                       \
    do {                                                               \
        cublasStatus_t _s = call;                                      \
        if (_s != CUBLAS_STATUS_SUCCESS) {                             \
            result.error_msg = std::string(#call)                       \
                + " failed: code="                                     \
                + std::to_string(static_cast<int>(_s));                \
            goto cleanup;                                              \
        }                                                              \
    } while (0)

    // 4. Allocate device buffers
    int max_gen = std::min(max_gen_tokens, 4096);
    int total_tokens = static_cast<int>(input_ids.size()) + max_gen;
    std::vector<int> gen_ids;
    gen_ids.reserve(static_cast<size_t>(total_tokens));

    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_h = nullptr;
    float *d_hidden = nullptr;      // [hc_mult·dim] — HC state in FP32
    float *d_hidden_next = nullptr; // [hc_mult·dim] FP32
    __half *d_single = nullptr;      // [dim] hc_pre output
    __half *d_normed = nullptr;      // [dim] rmsnorm output
    __half *d_ffn_out = nullptr;     // [dim] FFN output (converted to half for hc_post)
    float *d_ffn_accum = nullptr;   // [dim] FP32 accumulation buffer (shared+routed)
    float *d_swiglu_out = nullptr;   // [intermediate_dim] FP32 SwiGLU output
    float *d_act_scales = nullptr;   // [ceil(max(hidden,inter)/128)] per-block act scales
    float *d_gate_f32 = nullptr;      // [intermediate_dim] FP32 W1 output
    float *d_up_f32 = nullptr;        // [intermediate_dim] FP32 W3 output
    CSAState *csa_states = nullptr;  // [num_layers] per-layer CSA state
    __half *d_shared_int = nullptr;
    __half *d_shared_gate = nullptr;
    __half *d_routed_buf = nullptr;
    float *d_dequant_scratch = nullptr;
    float *d_act_quant = nullptr;      // FP32 activation (was __half*)
    float *d_attn_dequant = nullptr;  // F8→FP32 scratch (≥ wq_b/wo_a/wo_b)
    __half *d_kv_cache = nullptr;      // [layers][window][head_dim]
    __half *d_q = nullptr, *d_qr = nullptr, *d_qr2 = nullptr;
    __half *d_kv = nullptr, *d_kv2 = nullptr;
    __half *d_o = nullptr, *d_olora = nullptr, *d_attn_out_h = nullptr;
    float  *d_attn_out = nullptr;    // [dim] FP32 for hc_post
    float *d_hc_post = nullptr, *d_hc_comb = nullptr, *d_hc_pre = nullptr;
    float *d_logits = nullptr;
    float *d_norm_scratch = nullptr;
    float *d_norm_gamma_scratch = nullptr;
    int32_t *d_next_token = nullptr;
    float *d_next_maxv = nullptr;
    PinnedWeights pw;
    int *h_topk = nullptr;
    float *h_top_weights = nullptr;
    uint8_t *d_pack_stage = nullptr;
    uint8_t *d_scale_stage = nullptr;
    int64_t routed_pack_total = 0, routed_scale_total = 0;
    ExpertDb expert_db;
    cudaEvent_t t_start = nullptr, t_stop = nullptr;
    float rms_eps = 1e-6f;
    int total_steps = 0;
    const int kLocalTopK = 6;
    // Shared experts may have intermediate_dim = hidden_dim; allocate full size.
    size_t dequant_sz = static_cast<size_t>(kHiddenDim * kHiddenDim) * sizeof(float);
    // Shared experts may have intermediate_dim up to hidden_dim
    size_t shared_int_sz = static_cast<size_t>(kHiddenDim) * sizeof(__half);
    // Max attention weight: wq_b/wo_a/wo_b = 32768·1024 (or 8192·4096) elements
    // FP32 dequant scratch: weights (n) + activation (k) + output (m), all FP32
    size_t max_attn_elems = static_cast<size_t>(kAttnHeads) * kHeadDim * kQloraRank;
    size_t max_attn_elems_wo = 8192LL * 4096;  // wo_a: [8192, 4096]
    size_t attn_dequant_max = std::max(max_attn_elems, max_attn_elems_wo);
    // Need: weights (n floats) + activation (max cols) + output (max rows)
    size_t act_plus_out = static_cast<size_t>(std::max(kHiddenDim, 8192));
    size_t attn_dequant_sz = (attn_dequant_max + act_plus_out) * sizeof(float);
    // KV cache: 43 layers × (128 window + max_compressed) × 512 head_dim
    int max_compressed = (max_gen_tokens + static_cast<int>(input_ids.size())) / 4 + 1;
    size_t kv_cache_sz = static_cast<size_t>(kNumLayers)
                       * (kAttnWindow + max_compressed) * kHeadDim
                       * sizeof(__half);

    INFER_CHECK(cudaStreamCreate(&stream));
    CUBLAS_INFER_CHECK(cublasCreate(&cublas_h));
    CUBLAS_INFER_CHECK(cublasSetStream(cublas_h, stream));

    INFER_CHECK(cudaMalloc(&d_hidden, static_cast<size_t>(kHcDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_hidden_next, static_cast<size_t>(kHcDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_single, static_cast<size_t>(kHiddenDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_normed, static_cast<size_t>(kHiddenDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_ffn_out, static_cast<size_t>(kHiddenDim) * sizeof(__half)));

    INFER_CHECK(cudaMalloc(&d_shared_int, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_shared_gate, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_routed_buf, shared_int_sz));
    INFER_CHECK(cudaMalloc(&d_dequant_scratch, dequant_sz));
    INFER_CHECK(cudaMalloc(&d_act_quant,
        static_cast<size_t>(std::max(kHiddenDim, kIntermediateDim)) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_ffn_accum,
        static_cast<size_t>(kHiddenDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_swiglu_out,
        static_cast<size_t>(kIntermediateDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_act_scales,
        static_cast<size_t>((std::max(kHiddenDim, kIntermediateDim) + 127) / 128) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_gate_f32,
        static_cast<size_t>(kIntermediateDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_up_f32,
        static_cast<size_t>(kIntermediateDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_attn_dequant, attn_dequant_sz));
    INFER_CHECK(cudaMalloc(&d_kv_cache, kv_cache_sz));
    INFER_CHECK(cudaMemset(d_kv_cache, 0, kv_cache_sz));

    // Routed-expert staging: 3 packed weights (~4.19MB each) + 3 E8M0 scales.
    {
        int64_t pk = static_cast<int64_t>(kIntermediateDim) * kHiddenDim / 2;   // w1/w3 packed bytes
        int64_t w2p = static_cast<int64_t>(kHiddenDim) * kIntermediateDim / 2;  // w2 packed bytes
        routed_pack_total = pk + w2p + pk;
        int64_t scl = (static_cast<int64_t>(kIntermediateDim) * kHiddenDim) / 32; // per-32 E8M0
        routed_scale_total = scl * 3;
        INFER_CHECK(cudaMalloc(&d_pack_stage, static_cast<size_t>(routed_pack_total)));
        INFER_CHECK(cudaMalloc(&d_scale_stage, static_cast<size_t>(routed_scale_total)));
    }

    INFER_CHECK(cudaMalloc(&d_q, static_cast<size_t>(kAttnHeads * kHeadDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_qr, static_cast<size_t>(kQloraRank) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_qr2, static_cast<size_t>(kQloraRank) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_kv, static_cast<size_t>(kHeadDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_kv2, static_cast<size_t>(kHeadDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_o, static_cast<size_t>(kAttnHeads * kHeadDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_olora, static_cast<size_t>(kOGroups * kOloraRank) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_attn_out_h, static_cast<size_t>(kHiddenDim) * sizeof(__half)));
    INFER_CHECK(cudaMalloc(&d_attn_out, static_cast<size_t>(kHiddenDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_hc_post, static_cast<size_t>(kHcMult) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_hc_comb, static_cast<size_t>(kHcMult * kHcMult) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_hc_pre, static_cast<size_t>(kHcMult) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_next_token, sizeof(int32_t)));
    INFER_CHECK(cudaMalloc(&d_next_maxv, sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_norm_scratch,
        static_cast<size_t>(kHiddenDim) * sizeof(float)));
    INFER_CHECK(cudaMalloc(&d_norm_gamma_scratch,
        static_cast<size_t>(kHiddenDim) * sizeof(float)));

    // ── DBG: zero ALL scratch so uninit reads can't differ run-to-run ──
    INFER_CHECK(cudaMemsetAsync(d_hidden, 0, static_cast<size_t>(kHcDim) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_hidden_next, 0, static_cast<size_t>(kHcDim) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_single, 0, static_cast<size_t>(kHiddenDim) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_normed, 0, static_cast<size_t>(kHiddenDim) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_ffn_out, 0, static_cast<size_t>(kHiddenDim) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_shared_int, 0, shared_int_sz, stream));
    INFER_CHECK(cudaMemsetAsync(d_shared_gate, 0, shared_int_sz, stream));
    INFER_CHECK(cudaMemsetAsync(d_routed_buf, 0, shared_int_sz, stream));
    INFER_CHECK(cudaMemsetAsync(d_dequant_scratch, 0, dequant_sz, stream));
    INFER_CHECK(cudaMemsetAsync(d_act_quant, 0,
        static_cast<size_t>(std::max(kHiddenDim, kIntermediateDim)) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_ffn_accum, 0,
        static_cast<size_t>(kHiddenDim) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_swiglu_out, 0,
        static_cast<size_t>(kIntermediateDim) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_act_scales, 0,
        static_cast<size_t>((std::max(kHiddenDim, kIntermediateDim) + 127) / 128) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_gate_f32, 0,
        static_cast<size_t>(kIntermediateDim) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_up_f32, 0,
        static_cast<size_t>(kIntermediateDim) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_attn_dequant, 0, attn_dequant_sz, stream));
    INFER_CHECK(cudaMemsetAsync(d_pack_stage, 0, static_cast<size_t>(routed_pack_total), stream));
    INFER_CHECK(cudaMemsetAsync(d_scale_stage, 0, static_cast<size_t>(routed_scale_total), stream));
    INFER_CHECK(cudaMemsetAsync(d_q, 0, static_cast<size_t>(kAttnHeads * kHeadDim) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_qr, 0, static_cast<size_t>(kQloraRank) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_qr2, 0, static_cast<size_t>(kQloraRank) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_kv, 0, static_cast<size_t>(kHeadDim) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_kv2, 0, static_cast<size_t>(kHeadDim) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_o, 0, static_cast<size_t>(kAttnHeads * kHeadDim) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_olora, 0, static_cast<size_t>(kOGroups * kOloraRank) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_attn_out_h, 0, static_cast<size_t>(kHiddenDim) * sizeof(__half), stream));
    INFER_CHECK(cudaMemsetAsync(d_attn_out, 0, static_cast<size_t>(kHiddenDim) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_hc_post, 0, static_cast<size_t>(kHcMult) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_hc_comb, 0, static_cast<size_t>(kHcMult * kHcMult) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_hc_pre, 0, static_cast<size_t>(kHcMult) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_norm_scratch, 0,
        static_cast<size_t>(kHiddenDim) * sizeof(float), stream));
    INFER_CHECK(cudaMemsetAsync(d_norm_gamma_scratch, 0,
        static_cast<size_t>(kHiddenDim) * sizeof(float), stream));

    // 5. Load pinned weights
    {
        auto perr = load_pinned_weights(mp, pw, stream);
        if (perr != cudaSuccess) {
            result.error_msg = "load_pinned_weights failed";
            goto cleanup;
        }
        INFER_CHECK(cudaDeviceSynchronize());
    }

    // 6. Expert index: mmap all routed-expert arrows for zero-copy access.
    {
        auto t0 = std::chrono::steady_clock::now();
        expert_db = load_expert_index(mp, topo.num_layers, topo.experts_per_layer);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        int loaded = 0;
        for (const auto &en : expert_db.entries)
            if (en.host_w1) ++loaded;
        log::info("[inference] expert mmaps: " + std::to_string(loaded)
            + "/" + std::to_string(expert_db.entries.size())
            + " loaded in " + std::to_string(ms) + " ms");
        if (loaded == 0 && !expert_db.entries.empty())
            log::warn("[inference] NO routed experts mapped — routed experts inactive");
    }
    // Ensure vocab size
    if (pw.vocab_size == 0) pw.vocab_size = 128000;
    log::info("[inference] vocab_size=" + std::to_string(pw.vocab_size));
    if (pw.vocab_size < 0 || pw.vocab_size > kMaxVocabSize) {
        result.error_msg = "vocab_size " + std::to_string(pw.vocab_size)
            + " exceeds fixed logits-buffer capacity ("
            + std::to_string(kMaxVocabSize) + ")";
        goto cleanup;
    }

    // Allocate logits AFTER load (pre-alloc buffers can be corrupted by loading).
    // Sized to the model's actual vocab_size (bounds-checked above), not a
    // fixed literal — a larger-vocab model now fails loudly at load time
    // instead of silently overflowing this buffer on every decode step.
    INFER_CHECK(cudaMalloc(&d_logits, static_cast<size_t>(pw.vocab_size) * sizeof(float)));

    // 7. Route table for top-8
    INFER_CHECK(cudaMallocHost(&h_topk,
        static_cast<size_t>(kLocalTopK) * sizeof(int)));
    INFER_CHECK(cudaMallocHost(&h_top_weights,
        static_cast<size_t>(kLocalTopK) * sizeof(float)));

    // 8. Embedding: first token
    {
        int first_tok = input_ids[0];
        size_t embed_offset = static_cast<size_t>(first_tok) * kHiddenDim;
        cudaError_t embed_err = cudaMemcpyAsync(
            d_single, pw.d_embed + embed_offset,
            static_cast<size_t>(kHiddenDim) * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream);
        if (embed_err != cudaSuccess) {
            log::error(std::string("embed copy failed: ")
                + cudaGetErrorString(embed_err));
        }
        INFER_CHECK(cudaGetLastError());  // catch any pending
        // Expand single embedding into hc_mult copies of the HC state
        INFER_CHECK(run_expand_hc(stream, d_single, d_hidden));
        // Sync after embed copy to catch async errors
        INFER_CHECK(cudaStreamSynchronize(stream));
        (void)embed_err;
    }

    total_steps = static_cast<int>(input_ids.size()) + max_gen;

    // Allocate CSA state for layers 2-41
    csa_states = new CSAState[topo.num_layers];
    {
        // Per-layer compress_ratio table (mirrors load_pinned_weights).
        std::vector<int> compress_ratios(topo.num_layers, 0);
        for (int i = 0; i < topo.num_layers && i < static_cast<int>(pw.layers.size()); ++i) {
            compress_ratios[i] = pw.layers[i].compress_ratio;
        }
        INFER_CHECK(alloc_csa_state(csa_states, topo.num_layers,
            total_steps, kHeadDim, stream, compress_ratios.data()));
    }

    // 9. Timing events
    INFER_CHECK(cudaEventCreate(&t_start));
    INFER_CHECK(cudaEventCreate(&t_stop));
    INFER_CHECK(cudaEventRecord(t_start, stream));

    // 10. Autoregressive loop
    for (int step = 0; step < total_steps; ++step) {
        int cur_tok = (step < static_cast<int>(input_ids.size()))
            ? input_ids[step]
            : gen_ids.back();  // use last generated token (not indexed by step-prompt)

        // Embed (if not first token)
        if (step > 0) {
            size_t embed_offset = static_cast<size_t>(cur_tok) * kHiddenDim;
            INFER_CHECK(cudaMemcpyAsync(
                d_single, pw.d_embed + embed_offset,
                static_cast<size_t>(kHiddenDim) * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream));
            INFER_CHECK(run_expand_hc(stream, d_single, d_hidden));
        }

        // Per-layer HC + MLA + FFN
        int max_layers = topo.num_layers;
        for (int lay = 0; lay < max_layers; ++lay) {
            auto &lw = pw.layers[lay];

            // ── Attention branch ─────────────────────────────────────────
            // x = hc_pre(h, hc_attn_*);  x = attn_norm(x);  x = MLA(x)
            // h = hc_post(x, h, post, comb)
            if (lw.d_hc_attn_fn) {
                INFER_CHECK(run_hc_pre(stream, d_hidden, lw.d_hc_attn_fn,
                                        lw.d_hc_attn_scale, lw.d_hc_attn_base,
                                        d_single, d_hc_post, d_hc_comb, d_hc_pre));
                INFER_CHECK(run_rmsnorm(stream, d_single, lw.d_attn_norm,
                                         d_normed, d_norm_scratch,
                                         d_norm_gamma_scratch,
                                         kHiddenDim, rms_eps));
                if (lw.compress_ratio > 0) {
                    INFER_CHECK(run_csa_attention(
                        cublas_h, stream, lw, d_normed,
                        d_kv_cache + static_cast<size_t>(lay) * (kAttnWindow + max_compressed) * kHeadDim,
                        csa_states[lay], step,
                        d_q, d_qr, d_qr2, d_kv, d_kv2,
                        d_o, d_olora, d_attn_out_h,
                        d_attn_dequant, d_norm_scratch, d_norm_gamma_scratch));
                 } else {
                     INFER_CHECK(run_mla_attention(
                         cublas_h, stream, lw, d_normed,
                         d_kv_cache + static_cast<size_t>(lay) * (kAttnWindow + max_compressed) * kHeadDim,
                         step,
                         d_q, d_qr, d_qr2, d_kv, d_kv2, d_o, d_olora, d_attn_out_h,
                         d_attn_dequant, d_norm_scratch, d_norm_gamma_scratch));
                }
                // Convert FP16 attention output → FP32 for hc_post
                {
                    int b = 256;
                    int g = (kHiddenDim + b - 1) / b;
                    half_to_float_kernel<<<g, b, 0, stream>>>(d_attn_out_h, d_attn_out, kHiddenDim);
                }
INFER_CHECK(run_hc_post(stream, d_attn_out, d_hidden,
                                         d_hc_post, d_hc_comb, d_hidden_next));
                float *tmp = d_hidden;
                d_hidden = d_hidden_next;
                d_hidden_next = tmp;
            }

            // ── FFN branch ───────────────────────────────────────────────
            // x = hc_pre(h, hc_ffn_*);  x = ffn_norm(x);  x = FFN(x)
            // h = hc_post(x, h, post, comb)
            if (lw.d_hc_ffn_fn) {
                INFER_CHECK(run_hc_pre(stream, d_hidden, lw.d_hc_ffn_fn,
                                        lw.d_hc_ffn_scale, lw.d_hc_ffn_base,
                                        d_single, d_hc_post, d_hc_comb, d_hc_pre));
            } else {
                // Fallback: ffn_norm on the reduced state (no HC)
                INFER_CHECK(run_hc_pre(stream, d_hidden, lw.d_hc_attn_fn,
                                        lw.d_hc_attn_scale, lw.d_hc_attn_base,
                                        d_single, d_hc_post, d_hc_comb, d_hc_pre));
            }

            if (lw.d_ffn_norm) {
                INFER_CHECK(run_rmsnorm(stream, d_single, lw.d_ffn_norm,
                                         d_normed, d_norm_scratch,
                                         d_norm_gamma_scratch,
                                         kHiddenDim, rms_eps));
            } else {
                INFER_CHECK(cudaMemcpyAsync(d_normed, d_single,
                    static_cast<size_t>(kHiddenDim) * sizeof(__half),
                    cudaMemcpyDeviceToDevice, stream));
            }

            // Router: gate GEMM + sqrt(softplus) + top-k (hash/score) + weights
            if (lw.d_gate_weight) {
                int num_routed = std::min(kNumRouted, topo.experts_per_layer);
                const int64_t *tid = lw.h_tid2eid;
                if (tid && (cur_tok < 0 || cur_tok >= lw.tid2eid_vocab)) tid = nullptr;
                INFER_CHECK(route_token(cublas_h, stream, lw.d_gate_weight,
                                        d_normed, kHiddenDim, num_routed,
                                        tid, lw.tid2eid_topk, cur_tok,
                                        lw.h_gate_bias,
                                        kLocalTopK, h_topk, h_top_weights));
            } else {
                std::memset(h_topk, 0, static_cast<size_t>(kLocalTopK) * sizeof(int));
                std::memset(h_top_weights, 0,
                            static_cast<size_t>(kLocalTopK) * sizeof(float));
            }

            // Shared experts FFN: outputs to d_ffn_accum (FP32)
            if (lw.d_shared_w1 && lw.shared_w1_bytes > 0) {
                // Shared FFN outputs FP16 to d_ffn_out, then convert to FP32 accum
                INFER_CHECK(run_shared_experts_ffn(
                    cublas_h, stream, d_normed,
                    lw.d_shared_w1, lw.d_shared_w2, lw.d_shared_w3,
                    lw.d_shared_w1_scales, lw.d_shared_w2_scales, lw.d_shared_w3_scales,
                    lw.shared_w1_global_scale, lw.shared_w2_global_scale,
                    lw.shared_w3_global_scale,
                    lw.shared_w1_bytes, lw.shared_w2_bytes, lw.shared_w3_bytes,
                    d_gate_f32, d_up_f32, d_dequant_scratch, d_ffn_out));
                // Convert shared FP16 output → FP32 accumulation buffer
                {
                    int block = 256;
                    int grid = (kHiddenDim + block - 1) / block;
                    // Reuse: half→float conversion via a simple kernel
                    // We'll use the d_ffn_accum as float output
                    INFER_CHECK(cudaMemsetAsync(d_ffn_accum, 0,
                        static_cast<size_t>(kHiddenDim) * sizeof(float), stream));
                    float_to_half_f2h_kernel<<<grid, block, 0, stream>>>(
                        d_ffn_out, d_ffn_accum, kHiddenDim);
                }
            } else {
                INFER_CHECK(cudaMemsetAsync(d_ffn_accum, 0,
                    static_cast<size_t>(kHiddenDim) * sizeof(float), stream));
            }

            // Routed experts: accumulate top-k weighted FFN outputs into d_ffn_accum (FP32)
            if (lw.d_gate_weight && !expert_db.entries.empty()) {
                for (int i = 0; i < kLocalTopK; ++i)
                    ensure_expert_resident(expert_db, lay, h_topk[i], stream);
                INFER_CHECK(run_routed_experts(
                    cublas_h, stream, d_normed,
                    expert_db.entries.data(), lay, topo.experts_per_layer,
                    h_topk, h_top_weights, kLocalTopK,
                    d_ffn_accum,
                    d_gate_f32, d_up_f32, d_dequant_scratch,
                    d_act_quant, d_swiglu_out, d_act_scales,
                    d_pack_stage, d_scale_stage,
                    routed_pack_total, routed_scale_total));
            }

            // Convert FP32 FFN accumulation → FP16 for hc_post
            // Clamp to FP16 range to prevent Inf/NaN overflow
            // Pass d_ffn_accum (FP32) directly to hc_post — no FP16 conversion
            {
                INFER_CHECK(run_hc_post(stream, d_ffn_accum, d_hidden,
                                         d_hc_post, d_hc_comb, d_hidden_next));
            }

            // hidden = hidden_next for next layer
            float *tmp = d_hidden;
            d_hidden = d_hidden_next;
            d_hidden_next = tmp;
        } // layers

        // Final: hc_head (4 copies → 1), then final RMSNorm, then LM Head
        if (pw.d_hc_head_fn) {
            INFER_CHECK(run_hc_head(stream, d_hidden, pw.d_hc_head_fn,
                                    pw.d_hc_head_scale, pw.d_hc_head_base,
                                    d_single));
        } else {
            // Fallback: first copy of HC state (float→half via kernel)
            // This path should not be taken if hc_head_fn is loaded
            log::warn("hc_head_fn not loaded, using raw first copy");
        }

        if (pw.d_final_norm) {
            INFER_CHECK(run_rmsnorm(stream, d_single, pw.d_final_norm,
                                     d_normed, d_norm_scratch,
                                     d_norm_gamma_scratch,
                                     kHiddenDim, rms_eps));
        } else {
            INFER_CHECK(cudaMemcpyAsync(d_normed, d_single,
                static_cast<size_t>(kHiddenDim) * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream));
        }

        // LM Head
        INFER_CHECK(cudaMemsetAsync(d_logits, 0,
            static_cast<size_t>(pw.vocab_size) * sizeof(float), stream));
        INFER_CHECK(run_lm_head(cublas_h, stream,
                                 pw.d_head, d_normed,
                                 pw.vocab_size, kHiddenDim,
                                 d_logits, d_next_token, d_next_maxv));

        // Read back only the 8-byte GPU-computed argmax result (token id +
        // max logit) — not the whole vocab_size-sized logits buffer — and
        // synchronize just this stream, never the whole device. This keeps
        // the per-token host handshake (needed to pick the next embedding
        // and check EOS) from stalling unrelated streams/models sharing
        // the GPU, and removes the O(vocab_size) host rescan entirely.
        int32_t next_tok_i32 = 0;
        float max_val = -1e10f;
        INFER_CHECK(cudaMemcpyAsync(&next_tok_i32, d_next_token,
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        INFER_CHECK(cudaMemcpyAsync(&max_val, d_next_maxv,
            sizeof(float), cudaMemcpyDeviceToHost, stream));
        INFER_CHECK(cudaStreamSynchronize(stream));
        int next_tok = static_cast<int>(next_tok_i32);

        // Diagnostic-only clamp of the *logged* max value. The previous
        // implementation rescaled the whole host logits array here, but
        // that array was never read again after this point — only the
        // logged max_val mattered, so only the log-facing value is clamped.
        if (max_val > 30.0f) {
            max_val = 30.0f;
        }

        // Store generated token (always — gen_ids[0] used as first gen-step input)
        gen_ids.push_back(next_tok);

        std::ostringstream step_oss;
        step_oss << "[inference] step=" << step << " tok=" << next_tok;
        char tok_buf[512];
        size_t tok_cap = sizeof(tok_buf);
        std::vector<int32_t> single_tok = {static_cast<int32_t>(next_tok)};
        if (::lsxcommon::arrow_tokenizer_decode(
                tok_db, single_tok.data(), single_tok.size(), tok_buf, &tok_cap) == 0)
            step_oss << " '" << std::string(tok_buf, tok_cap) << "'";
        step_oss << " maxv=" << std::fixed << std::setprecision(4) << max_val;
        log::info(step_oss.str());

        // Single-token self-check: capture the first generated token (pre-clamp
        // maxv) for RunSelfCheck. The first generated token is the argmax at
        // step == input_ids.size()-1 (the step whose embedding is the LAST
        // prompt token); at step == input_ids.size() the model has already been
        // fed that first generated token and argmax produces the SECOND
        // generated token (off-by-one).
        if (step == static_cast<int>(input_ids.size()) - 1
            && result.first_gen_tok_id == -1) {
            result.first_gen_tok_id = next_tok;
            result.first_gen_maxv   = max_val;
            result.first_gen_text   = std::string(tok_buf, tok_cap);
        }

        if (next_tok == tok_db.eos_id && step >= static_cast<int>(input_ids.size()))
            break;
    }

    INFER_CHECK(cudaEventRecord(t_stop, stream));
    INFER_CHECK(cudaDeviceSynchronize());

    // Compute metrics
    {
        float elapsed_ms = 0.0f;
        INFER_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));

        int prompt_len = static_cast<int>(input_ids.size());
        int gen_count = static_cast<int>(gen_ids.size()) - prompt_len;
        // Skip prompt tokens for output decoding
        auto gen_start = gen_ids.begin() + std::min(prompt_len, static_cast<int>(gen_ids.size()));
        std::vector<int32_t> gen_ids_i32(gen_start, gen_ids.end());
        size_t out_cap = 4096;
        std::string output(out_cap, '\0');
        int32_t dr = ::lsxcommon::arrow_tokenizer_decode(
            tok_db, gen_ids_i32.data(), gen_ids_i32.size(), output.data(), &out_cap);
        if (dr == -2) {   // *out_cap now reports the required size; retry once
            output.resize(out_cap);
            size_t cap2 = out_cap;
            dr = ::lsxcommon::arrow_tokenizer_decode(
                tok_db, gen_ids_i32.data(), gen_ids_i32.size(), output.data(), &cap2);
            out_cap = cap2;
        }
        if (dr == 0) output.resize(out_cap);
        else output.clear();
        result.ttft_ms = elapsed_ms / static_cast<double>(total_steps);
        result.tokens_per_sec = elapsed_ms > 0.0f
            ? (static_cast<double>(gen_count) / elapsed_ms) * 1000.0
            : 0.0;
        result.total_time_ms = static_cast<double>(elapsed_ms);
        result.num_tokens = gen_count;
        result.num_layers = topo.num_layers;
        result.output_text = output;
        result.ok = true;

        std::ostringstream res_oss;
        res_oss << "=== --testprompt Results ===\n"
                << "  prompt_tokens: \"" << input_ids.size() << "\"\n"
                << "  output:        \"" << output << "\"\n"
                << "  elapsed:       " << std::fixed << std::setprecision(1)
                << static_cast<double>(elapsed_ms) << " ms\n"
                << "  gen_tokens:    " << gen_count << "\n"
                << "  TTFT:          " << std::setprecision(2) << result.ttft_ms
                << " ms\n"
                << "  tokens/s:      " << std::setprecision(1)
                << result.tokens_per_sec;
        log::info(res_oss.str());
    }

cleanup:
    if (stream)        cudaStreamDestroy(stream);
    if (cublas_h)      cublasDestroy(cublas_h);
    if (d_hidden)      cudaFree(d_hidden);
    if (d_hidden_next) cudaFree(d_hidden_next);
    if (d_single)      cudaFree(d_single);
    if (d_normed)      cudaFree(d_normed);
    if (d_ffn_out)     cudaFree(d_ffn_out);
    if (d_shared_int)  cudaFree(d_shared_int);
    if (d_shared_gate) cudaFree(d_shared_gate);
    if (d_routed_buf)  cudaFree(d_routed_buf);
    if (d_dequant_scratch) cudaFree(d_dequant_scratch);
    if (d_act_quant)      cudaFree(d_act_quant);
    if (d_ffn_accum)      cudaFree(d_ffn_accum);
    if (d_swiglu_out)     cudaFree(d_swiglu_out);
    if (d_act_scales)     cudaFree(d_act_scales);
    if (d_gate_f32)       cudaFree(d_gate_f32);
    if (d_up_f32)         cudaFree(d_up_f32);
    if (d_attn_dequant) cudaFree(d_attn_dequant);
    if (d_kv_cache)    cudaFree(d_kv_cache);
    if (d_q)           cudaFree(d_q);
    if (d_qr)          cudaFree(d_qr);
    if (d_qr2)         cudaFree(d_qr2);
    if (d_kv)          cudaFree(d_kv);
    if (d_kv2)         cudaFree(d_kv2);
    if (d_o)           cudaFree(d_o);
    if (d_olora)       cudaFree(d_olora);
    if (d_attn_out)    cudaFree(d_attn_out);
    if (d_attn_out_h)  cudaFree(d_attn_out_h);
    if (d_hc_post)     cudaFree(d_hc_post);
    if (d_hc_comb)     cudaFree(d_hc_comb);
    if (d_hc_pre)      cudaFree(d_hc_pre);
    if (d_logits)      cudaFree(d_logits);
    if (d_norm_scratch) cudaFree(d_norm_scratch);
    if (d_norm_gamma_scratch) cudaFree(d_norm_gamma_scratch);
    if (d_next_token)  cudaFree(d_next_token);
    if (d_next_maxv)   cudaFree(d_next_maxv);
    if (h_topk)        cudaFreeHost(h_topk);
    if (h_top_weights) cudaFreeHost(h_top_weights);
    if (d_pack_stage)  cudaFree(d_pack_stage);
    if (d_scale_stage) cudaFree(d_scale_stage);
    if (csa_states) {
        free_csa_state(csa_states, topo.num_layers);
        delete[] csa_states;
    }
    if (t_start)       cudaEventDestroy(t_start);
    if (t_stop)        cudaEventDestroy(t_stop);
    free_expert_index(expert_db);
    pw.Free();

    return result;
    } // end execution block
} // end RunInference

// Single-token self-check: runs one forward pass on the given prompt and
// reports whether the first generated token is "Paris". Test/diagnostic entry
// point — may be removed later.
InferenceResult RunSelfCheckModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db)
{
    InferenceResult r = RunInferenceModel(model_path, input_ids, max_gen_tokens, tok_db);
    if (!r.ok) {
        r.selfcheck_passed = false;
        return r;
    }
    if (r.first_gen_tok_id >= 0) {
        absl::StripAsciiWhitespace(&r.first_gen_text);
        r.selfcheck_passed = (r.first_gen_text == "Paris");
    }
    log::info("[selfcheck] top1_tok=" + std::to_string(r.first_gen_tok_id)
              + " text='" + r.first_gen_text + "' passed="
              + (r.selfcheck_passed ? "true" : "false"));
    return r;
}

} // namespace gpu
} // namespace lsxcommon
