#include "lsxcommon/inference_gpu.h"
#include "lsxcommon/deepseek_moe_inference_head.h"
#include "lsxcommon/deepseek_moe_norm_kernels.h"
#include "lsxcommon/tokenizer.h"
#include "lsxcommon/log.h"
#include "lsxcommon/model_runtime.h"
#include "lsxcommon/resource_governor.h"
#include "lsxcommon/deepseek_moe_inference_csa.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/gpu_perf.h"
#include "deepseek_moe_load.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace lsxcommon {
namespace gpu {

// Forward declarations from other .cu files

// half_to_float_kernel / float_to_half_kernel are defined once in
// deepseek_moe_load.cuh (included above) and shared by every .cu
// file in this pipeline — see that header for why.

// half → float conversion kernel (local pipeline copy; the canonical
// half_to_float_kernel lives in deepseek_moe_inference_moe.cu)
__global__ void half_to_float_f2h_kernel(const __half* __restrict__ src,
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

// From deepseek_moe_inference_moe.cu
template <class T = deepseek::Passport>
cudaError_t run_shared_experts_ffn(
    cudaStream_t stream,
    const __half *d_hidden,
    const uint8_t *d_w1, const uint8_t *d_w2, const uint8_t *d_w3,
    const uint8_t *d_w1_scales, const uint8_t *d_w2_scales, const uint8_t *d_w3_scales,
    float w1_gs, float w2_gs, float w3_gs,
    int64_t w1_bytes, int64_t w2_bytes, int64_t w3_bytes,
    float *d_intermediate_f32, float *d_gate_f32,
    float *d_dequant, __half *d_output,
    const __half *d_w1_h_cache = nullptr,
    const __half *d_w3_h_cache = nullptr,
    const float *d_w2_f32_cache = nullptr);

template <class T = deepseek::Passport>
cudaError_t run_routed_experts_ex(
    cudaStream_t stream, cudaStream_t copy_stream, cudaEvent_t *stage_ev,
    cudaEvent_t pre_ev,
    const __half *d_normed,
    const ExpertEntry *db_entries, int layer, int experts_per_layer,
    const int *h_top_indices, const float *h_top_weights, int topk,
    float *d_output,
    __half *d_yg, __half *d_y2,
    uint8_t *d_pack_stage, uint8_t *d_scale_stage,
    int64_t pack_stage_bytes, int64_t scale_stage_bytes,
    const ExpertDeviceCache *cache);

// Early-issue phase: stages top-k cache misses onto copy_stream right after
// the router readback so the transfers overlap the shared-expert FFN.
template <class T = deepseek::Passport>
cudaError_t issue_routed_staging(
    cudaStream_t copy_stream, cudaEvent_t ev, ExpertDeviceCache *cache,
    const ExpertEntry *db_entries, int layer, int experts_per_layer,
    const int *h_top_indices, int topk);

// run_mla_attention is declared in deepseek_moe_load.cuh (included
// above) — that header's signature now matches the real definition in
// deepseek_moe_inference_attn.cu (float* d_dequant), so this file no longer
// needs its own local, previously-stale (__half*) redeclaration.

// From deepseek_moe_inference_attn.cu
cudaError_t run_csa_attention(
    cudaStream_t stream,
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


// Sanity ceiling for a model-reported vocab_size. d_logits is now allocated
// dynamically to the actual vocab_size (see RunInferenceModel), but any
// model reporting an implausibly large vocabulary fails loudly at load time
// instead of proceeding with a runaway allocation/GEMM size.
constexpr int kMaxVocabSize = 200000;

InferenceResult RunInferenceModel(
    const std::string &model_path,
    const std::vector<int32_t> &input_ids,
    int max_gen_tokens,
    ::lsxcommon::ArrowTokenizerDb &tok_db,
    PinnedWeights &pw,
    lsxcommon::sample::ISampler &sampler =
        lsxcommon::sample::DefaultSampler(),
    cudaStream_t ext_stream = nullptr,
    bool ignore_eos = false,
    const lsxcommon::GenerationStream *gs = nullptr)
{
    InferenceResult result;
    auto mp = model_path;
    // Persistent-context mode (ModelBase::EnsureContext): the caller owns the
    // stream across Infer calls, saving per-call stream setup.
    const bool own_context = (ext_stream == nullptr);

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


    // 4. Allocate device buffers
    int max_gen = std::min(max_gen_tokens, 4096);
    int total_tokens = static_cast<int>(input_ids.size()) + max_gen;
    std::vector<int> gen_ids;
    gen_ids.reserve(static_cast<size_t>(total_tokens));

    cudaStream_t stream = nullptr;
    // Per-run single-allocation arena (spec item 12). Local aliases below are
    // bound to arena slices right after Allocate(); d_hidden/d_hidden_next get
    // swapped by the layer loop (pointers swap, arena members stay fixed).
    DeepSeekInferArena arena;
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
    int *h_topk = nullptr;
    float *h_top_weights = nullptr;
    uint8_t *d_pack_stage = nullptr;
    uint8_t *d_scale_stage = nullptr;
    int64_t routed_pack_total = 0, routed_scale_total = 0;
    ExpertDb expert_db;
    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
    cudaEvent_t evict_done = nullptr;   // mmap-eviction barrier (event form)
    // Dual-stream expert staging: H2D miss copies run on copy_stream with a
    // per-expert event; the compute stream waits only for the expert it is
    // about to multiply (GLM-Flash pattern). Hides copy latency under GEMV.
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t stage_ev[kTopK] = {};
    cudaEvent_t ev_route = nullptr;     // router readback fence (async split)
    // Device-side resident-expert cache (see ExpertDeviceCache in
    // deepseek_moe_load.cuh): repeated expert selections skip the
    // ~13 MB per-expert H2D staging entirely.
    ExpertDeviceCache expert_cache;
    std::vector<int32_t> cache_slot_of;
    std::vector<int32_t> cache_slot_owner;
    std::vector<int32_t> cache_lru;
    std::vector<int32_t> cache_rotor;
    float rms_eps = 1e-6f;
    int total_steps = 0;
    std::uint64_t d_alloc_bytes = 0;  // arena device bytes (resident accounting)
    const int kLocalTopK = 6;
    // Routed-expert staging: 3 packed weights (~4.19MB each) + 3 E8M0 scales.
    // Sizes are replicated by DeepSeekInferArena::Allocate; these locals feed
    // run_routed_experts' bounds and the memset block below.
    {
        int64_t pk = static_cast<int64_t>(kIntermediateDim) * kHiddenDim / 2;   // w1/w3 packed bytes
        int64_t w2p = static_cast<int64_t>(kHiddenDim) * kIntermediateDim / 2;  // w2 packed bytes
        routed_pack_total = pk + w2p + pk;
        // logestix-v1-nvfp4: pow2-E4M3 scales are PER-16 elements (2x the
        // legacy per-32 MXFP4 arrays). A stale per-32 bound makes every
        // routed expert silently fail the staging-size check and skip.
        int64_t scl = (static_cast<int64_t>(kIntermediateDim) * kHiddenDim) / 16;
        routed_scale_total = scl * 3;
    }
    // KV cache: 43 layers × (128 window + max_compressed) × 512 head_dim.
    // max_compressed must be derived from the CLAMPED step count — the arena
    // sizes its compressed-KV region from total_steps below (which uses
    // max_gen, not the raw max_gen_tokens); an unclamped value here would
    // overflow the kv-cache slice via memset and desync per-layer strides.
    int max_compressed = total_tokens / 4 + 1;
    size_t kv_cache_sz = static_cast<size_t>(kNumLayers)
                       * (kAttnWindow + max_compressed) * kHeadDim
                       * sizeof(__half);

    // Spin (busy-poll) host synchronization: the per-layer router readback and
    // the per-token 8-byte handshake wake the host in ~1µs instead of the
    // ~10-30µs blocking-syscall latency. Host CPU headroom is explicitly not
    // a constrained resource for this project.
    INFER_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
    if (own_context) {
        INFER_CHECK(cudaStreamCreate(&stream));
    } else {
        stream = ext_stream;
        if (!stream) {
            result.error_msg = "external context requires stream";
            goto cleanup;
        }
    }
    // Staging copy lane — created regardless of stream ownership (the model
    // hands in its own stream, own_context is false in production).
    INFER_CHECK(cudaStreamCreate(&copy_stream));

    // The whole per-run footprint (every scratch buffer, KV cache, CSA state,
    // logits, routing staging) is carved from DeepSeekInferArena below — after
    // load_pinned_weights, so d_logits can be sized to the model's actual
    // vocab_size. One device cudaMalloc + one pinned cudaMallocHost per run.

    // 5. Pinned weights are loaded ONCE by the owning model
    //    (DeepSeekV4FlashModel::Load) and handed in here; the loader already
    //    ends synchronized (StreamDrain), so no device-wide sync is needed.
    if (!pw.d_embed || pw.layers.empty()) {
        result.error_msg = "pinned weights not loaded";
        goto cleanup;
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

    // 7. Single-allocation per-run arena (spec item 12). Allocate AFTER load so
    // d_logits is sized to the model's actual vocab_size (bounds-checked above),
    // and the CSA region to the real per-layer compress ratios.
    total_steps = static_cast<int>(input_ids.size()) + max_gen;
    {
        // Per-layer compress_ratio table (mirrors load_pinned_weights).
        std::vector<int> compress_ratios(topo.num_layers, 0);
        for (int i = 0; i < topo.num_layers && i < static_cast<int>(pw.layers.size()); ++i) {
            compress_ratios[i] = pw.layers[i].compress_ratio;
        }
        INFER_CHECK(arena.Allocate(stream, topo.num_layers, total_steps,
                                   compress_ratios.data(), pw.vocab_size));
        d_alloc_bytes = arena.bytes;
    }

    // Bind local aliases to the arena slices.
    d_hidden        = arena.d_hidden;
    d_hidden_next   = arena.d_hidden_next;
    d_single        = arena.d_single;
    d_normed        = arena.d_normed;
    d_ffn_out       = arena.d_ffn_out;
    d_shared_int    = arena.d_shared_int;
    d_shared_gate   = arena.d_shared_gate;
    d_routed_buf    = arena.d_routed_buf;
    d_dequant_scratch = arena.d_dequant_scratch;
    d_act_quant     = arena.d_act_quant;
    d_ffn_accum     = arena.d_ffn_accum;
    d_swiglu_out    = arena.d_swiglu_out;
    d_act_scales    = arena.d_act_scales;
    d_gate_f32      = arena.d_gate_f32;
    d_up_f32        = arena.d_up_f32;
    d_attn_dequant  = arena.d_attn_dequant;
    d_kv_cache      = arena.d_kv_cache;
    d_q             = arena.d_q;
    d_qr            = arena.d_qr;
    d_qr2           = arena.d_qr2;
    d_kv            = arena.d_kv;
    d_kv2           = arena.d_kv2;
    d_o             = arena.d_o;
    d_olora         = arena.d_olora;
    d_attn_out_h    = arena.d_attn_out_h;
    d_attn_out      = arena.d_attn_out;
    d_hc_post       = arena.d_hc_post;
    d_hc_comb       = arena.d_hc_comb;
    d_hc_pre        = arena.d_hc_pre;
    d_logits        = arena.d_logits;
    d_norm_scratch  = arena.d_norm_scratch;
    d_norm_gamma_scratch = arena.d_norm_gamma_scratch;
    d_next_token    = arena.d_next_token;
    d_next_maxv     = arena.d_next_maxv;
    d_pack_stage    = arena.d_pack_stage;
    d_scale_stage   = arena.d_scale_stage;
    h_topk          = arena.h_topk;
    h_top_weights   = arena.h_top_weights;
    csa_states      = arena.csa_states;

    // ── DBG: zero ALL scratch so uninit reads can't differ run-to-run ──
    // (Sizes replicated from DeepSeekInferArena::Allocate so the zero block
    // covers the exact footprint. Nested block: the goto-based INFER_CHECK
    // jumps above must not cross these initializers (nvcc C2362).)
    {
    size_t dequant_sz = static_cast<size_t>(kHiddenDim * kHiddenDim) * sizeof(float);
    size_t attn_dequant_max = std::max(
        static_cast<size_t>(kAttnHeads) * kHeadDim * kQloraRank,
        static_cast<size_t>(8192ULL * 4096));
    size_t attn_dequant_sz = (attn_dequant_max
                              + std::max<size_t>(kHiddenDim, 8192)) * sizeof(float);
    size_t shared_int_sz = static_cast<size_t>(kHiddenDim) * sizeof(__half);
    INFER_CHECK(cudaMemset(d_kv_cache, 0, kv_cache_sz));
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
    INFER_CHECK(cudaMemsetAsync(arena.d_scale_stage_tc, 0, static_cast<size_t>(routed_scale_total), stream));
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
    }

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

    // 9. Timing events
    INFER_CHECK(cudaEventCreate(&t_start));
    INFER_CHECK(cudaEventCreate(&t_ttft));
    INFER_CHECK(cudaEventCreate(&t_stop));
    INFER_CHECK(cudaEventCreateWithFlags(&evict_done, cudaEventDisableTiming));
    INFER_CHECK(cudaEventCreateWithFlags(&ev_route, cudaEventDisableTiming));
    for (int k = 0; k < kTopK; ++k)
        INFER_CHECK(cudaEventCreateWithFlags(&stage_ev[k],
                                             cudaEventDisableTiming));
    INFER_CHECK(cudaEventRecord(t_start, stream));

    // 9b. Device-side resident-expert cache. Slot count is overridable via
    // LSX_DEEPSEEK_EXPERT_CACHE_SLOTS (0 disables). Each slot holds one
    // expert's packed w1/w2/w3 + scales (~13 MB at V4-Flash geometry).
    {
        const int total_experts = topo.num_layers * topo.experts_per_layer;
        bool explicit_slots = false;
        int slots = 0;
        if (const char* env = std::getenv("LSX_DEEPSEEK_EXPERT_CACHE_SLOTS")) {
            slots = std::atoi(env);
            explicit_slots = true;   // verbatim: 0 legitimately disables
        }
        // GLM-Flash ResolveExpertSlots pattern: cache everything that fits
        // into free VRAM minus a generation reserve. The old "thrashes every
        // token" note applied to fixed small slot counts; full-free-VRAM
        // sizing measured 4.6 -> 8.5 tok/s on RTX 5090.
        if (!explicit_slots) {
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
                constexpr size_t kReserve = size_t{2} << 30;  // arenas + ctx
                const size_t budget =
                    free_bytes > kReserve ? free_bytes - kReserve : 0;
                // Perf policy: scale the VRAM budget by the resolved slot
                // percentage (default 100 = unchanged behavior).
                const int pct = perf::CurrentKnobs().expert_slots_pct;
                size_t eff = budget;
                if (pct < 100) eff = budget / 100u * static_cast<unsigned>(pct);
                const int64_t slot_bytes =
                    routed_pack_total + routed_scale_total;
                if (slot_bytes > 0)
                    slots = static_cast<int>(
                        eff / static_cast<size_t>(slot_bytes));
            }
        }
        if (slots > 0 && total_experts > 0 && !expert_db.entries.empty()) {
            expert_cache.slots = std::min(slots, total_experts);
            expert_cache.pack_stride = routed_pack_total;
            expert_cache.scale_stride = routed_scale_total;
            expert_cache.total_experts = total_experts;
            cache_slot_of.assign(static_cast<size_t>(total_experts), -1);
            cache_slot_owner.assign(static_cast<size_t>(expert_cache.slots), -1);
            cache_lru.resize(static_cast<size_t>(expert_cache.slots));
            // Per-layer slot bands are REQUIRED for the early-issue
            // prefetch (issue(L+1) must never write the band layer L's
            // GEMVs read — removes the write-after-read slot race by
            // construction). Without prefetch the global LRU measured
            // faster (11.7 vs 10.7) and stays the default.
            // With the full-resident mmap table (max_open=65536) there are
            // no munmaps at all, so the early-issue prefetch needs no slot
            // bands — global LRU measured the best hit rate.
            expert_cache.layer_slots = 0;
            expert_cache.num_layers = topo.num_layers;
            for (int i = 0; i < expert_cache.slots; ++i)
                cache_lru[static_cast<size_t>(i)] = expert_cache.slots - 1 - i; // back = LRU
            expert_cache.h_slot_of = cache_slot_of.data();
            expert_cache.h_slot_owner = cache_slot_owner.data();
            expert_cache.h_lru = cache_lru.data();
            cudaError_t ce = cudaMalloc(&expert_cache.d_base,
                                        expert_cache.DeviceBytes());
            if (ce == cudaSuccess) {
                expert_cache.d_pack = expert_cache.d_base;
                expert_cache.d_scale = expert_cache.d_base
                    + static_cast<int64_t>(expert_cache.slots) * expert_cache.pack_stride;
                log::info("[inference] expert device cache: "
                          + std::to_string(expert_cache.slots) + " slots ("
                          + std::to_string(expert_cache.DeviceBytes() / (1024 * 1024))
                          + " MiB)");
            } else {
                log::warn("[inference] expert device cache allocation failed ("
                          + std::string(cudaGetErrorString(ce))
                          + ") — falling back to per-call staging");
                expert_cache = ExpertDeviceCache{};
            }
        }
    }

    // 10. Autoregressive loop
    for (int step = 0; step < total_steps; ++step) {
        int cur_tok = (step < static_cast<int>(input_ids.size()))
            ? input_ids[step]
            : gen_ids.back();  // use last generated token (not indexed by step-prompt)
        // Guard the embed gather: a token id outside [0, vocab) is an OOB
        // device read (bad sampler output / corrupt tokenizer db).
        if (cur_tok < 0 || cur_tok >= pw.vocab_size) {
            result.error_msg = "token id " + std::to_string(cur_tok) +
                " out of vocab range (" + std::to_string(pw.vocab_size) + ")";
            goto cleanup;
        }

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
                                        d_single, d_hc_post, d_hc_comb, d_hc_pre,
                                        d_norm_scratch));
                INFER_CHECK(run_rmsnorm(stream, d_single, lw.d_attn_norm,
                                         d_normed, d_norm_scratch,
                                         d_norm_gamma_scratch,
                                         kHiddenDim, rms_eps));
                if (lw.compress_ratio > 0) {
                    INFER_CHECK(run_csa_attention(
                        stream, lw, d_normed,
                        d_kv_cache + static_cast<size_t>(lay) * (kAttnWindow + max_compressed) * kHeadDim,
                        csa_states[lay], step,
                        d_q, d_qr, d_qr2, d_kv, d_kv2,
                        d_o, d_olora, d_attn_out_h,
                        d_attn_dequant, d_norm_scratch, d_norm_gamma_scratch));
                 } else {
                     INFER_CHECK(run_mla_attention(
                         stream, lw, d_normed,
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
                                        d_single, d_hc_post, d_hc_comb, d_hc_pre,
                                        d_norm_scratch));
            } else {
                // Fallback: ffn_norm on the reduced state (no HC)
                INFER_CHECK(run_hc_pre(stream, d_hidden, lw.d_hc_attn_fn,
                                        lw.d_hc_attn_scale, lw.d_hc_attn_base,
                                        d_single, d_hc_post, d_hc_comb, d_hc_pre,
                                        d_norm_scratch));
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

            // Router (async split): begin enqueues gate GEMM + D2H logits and
            // records ev_route; the host-side scoring happens in finish BELOW,
            // after the shared-expert FFN has been enqueued — the GPU keeps
            // executing that FFN while the host waits for the router logits.
            const bool have_router = (lw.d_gate_weight != nullptr);
            const bool do_prefetch =
                std::getenv("LSX_DS_PREFETCH") != nullptr;
            int num_routed = std::min(kNumRouted, topo.experts_per_layer);
            const int64_t *tid = lw.h_tid2eid;
            if (tid && (cur_tok < 0 || cur_tok >= lw.tid2eid_vocab)) tid = nullptr;
            if (have_router) {
                INFER_CHECK(route_token_begin(stream, lw.d_gate_weight,
                                              d_normed, kHiddenDim, num_routed,
                                              arena.d_routing_scratch,
                                              arena.h_routing_scratch,
                                              ev_route));
            } else {
                std::memset(h_topk, 0, static_cast<size_t>(kLocalTopK) * sizeof(int));
                std::memset(h_top_weights, 0,
                            static_cast<size_t>(kLocalTopK) * sizeof(float));
            }

            // EXPERIMENTAL prefetch (LSX_DS_PREFETCH=1): issue the top-k
            // miss copies early so they overlap the shared-expert FFN.
            if (do_prefetch && have_router && expert_cache.slots > 0) {
                for (int i = 0; i < kLocalTopK; ++i)
                    ensure_expert_resident(expert_db, lay, h_topk[i], stream,
                                           evict_done);
                // LSX_DS_PREFETCH_SYNC: issue on the MAIN stream (no copy
                // lane concurrency at all) to isolate logic vs timing.
                const bool pf_sync =
                    std::getenv("LSX_DS_PREFETCH_SYNC") != nullptr;
                INFER_CHECK(issue_routed_staging(
                    pf_sync ? stream : copy_stream, stage_ev[0], &expert_cache,
                    expert_db.entries.data(), lay, topo.experts_per_layer,
                    h_topk, kLocalTopK));
            }

            // Shared experts FFN: outputs to d_ffn_accum (FP32).
            // Independent of the routing result; enqueued behind the router GEMM
            // so it overlaps the host's router-logit wait.
            if (lw.d_shared_w1 && lw.shared_w1_bytes > 0) {
                // Shared FFN outputs FP16 to d_ffn_out, then convert to FP32 accum
                INFER_CHECK(run_shared_experts_ffn(
                    stream, d_normed,
                    lw.d_shared_w1, lw.d_shared_w2, lw.d_shared_w3,
                    lw.d_shared_w1_scales, lw.d_shared_w2_scales, lw.d_shared_w3_scales,
                    lw.shared_w1_global_scale, lw.shared_w2_global_scale,
                    lw.shared_w3_global_scale,
                    lw.shared_w1_bytes, lw.shared_w2_bytes, lw.shared_w3_bytes,
                    d_gate_f32, d_up_f32, d_dequant_scratch, d_ffn_out,
                    lw.d_shared_w1_h, lw.d_shared_w3_h,
                    lw.d_shared_w2_f32));
                // Convert shared FP16 output → FP32 accumulation buffer
                {
                    int block = 256;
                    int grid = (kHiddenDim + block - 1) / block;
                    half_to_float_f2h_kernel<<<grid, block, 0, stream>>>(
                        d_ffn_out, d_ffn_accum, kHiddenDim);
                }
            } else {
                INFER_CHECK(cudaMemsetAsync(d_ffn_accum, 0,
                    static_cast<size_t>(kHiddenDim) * sizeof(float), stream));
            }

            // Router finish: event-scoped fence + host-side scoring
            // (sqrt(softplus) + top-k). Stream ordering guarantees the shared
            // FFN above retires before the routed-expert W2 accumulation below.
            if (have_router) {
                INFER_CHECK(route_token_finish(num_routed, cur_tok,
                                               tid, lw.tid2eid_topk,
                                               lw.h_gate_bias,
                                               kLocalTopK,
                                               arena.h_routing_scratch,
                                               h_topk, h_top_weights,
                                               ev_route));
            }

            // Routed experts: accumulate top-k weighted FFN outputs into d_ffn_accum (FP32)
            if (lw.d_gate_weight && !expert_db.entries.empty()) {
                if (!do_prefetch) {
                    for (int i = 0; i < kLocalTopK; ++i)
                        ensure_expert_resident(expert_db, lay, h_topk[i],
                                               stream, evict_done);
                }
                INFER_CHECK(run_routed_experts_ex(
                    stream, copy_stream, stage_ev,
                    (do_prefetch && have_router && expert_cache.slots > 0
                     && !std::getenv("LSX_DS_PF_NOPRE")) ? stage_ev[0] : nullptr,
                    d_normed,
                    expert_db.entries.data(), lay, topo.experts_per_layer,
                    h_topk, h_top_weights, kLocalTopK,
                    d_ffn_accum,
                    reinterpret_cast<__half *>(d_gate_f32),
                    reinterpret_cast<__half *>(d_up_f32),
                    d_pack_stage, d_scale_stage,
                    routed_pack_total, routed_scale_total,
                    expert_cache.slots > 0 ? &expert_cache : nullptr));
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
                                     d_single, d_norm_scratch));
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

        // LM Head (no fused argmax — the sampler below performs the single
        // vocab reduction; the old duplicate argmax scanned the full vocab
        // twice per token).
        INFER_CHECK(cudaMemsetAsync(d_logits, 0,
            static_cast<size_t>(pw.vocab_size) * sizeof(float), stream));
        INFER_CHECK(run_lm_head(stream,
                                 pw.d_head, d_normed,
                                 pw.vocab_size, kHiddenDim,
                                 d_logits, nullptr, nullptr));

        INFER_CHECK(sampler.Sample(stream, d_logits, pw.vocab_size,
                                   lsxcommon::sample::LogitType::kF32,
                                   d_next_token, d_next_maxv));

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
        if (gs) {
            const int idx = static_cast<int>(gen_ids.size()) - 1;
            const int first_gen = static_cast<int>(input_ids.size()) - 1;
            if (idx >= first_gen)
                gs->Emit(static_cast<std::uint32_t>(next_tok),
                         static_cast<std::uint32_t>(idx - first_gen));
            if (gs->Aborted()) break;
        }

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
            cudaEventRecord(t_ttft, stream);
        }

        if (!ignore_eos && next_tok == tok_db.eos_id &&
            step >= static_cast<int>(input_ids.size())) {
            result.eos_terminated = true;
            break;
        }
    }

    // Publish the raw [vocab] logits row of the final decode step.
    {
        result.final_logits.resize(static_cast<size_t>(pw.vocab_size));
        cudaError_t fl_e = cudaMemcpyAsync(
            result.final_logits.data(), d_logits,
            static_cast<size_t>(pw.vocab_size) * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        if (fl_e == cudaSuccess) fl_e = cudaStreamSynchronize(stream);
        if (fl_e != cudaSuccess) result.final_logits.clear();  // best-effort
    }

    INFER_CHECK(cudaEventRecord(t_stop, stream));
    // Stream-scoped teardown fence (audit F-04): this pipeline owns a single
    // stream (created above), so a device-wide barrier here would only stall
    // unrelated streams/models sharing the GPU.
    INFER_CHECK(cudaStreamSynchronize(stream));

    // Compute metrics
    {
        float elapsed_ms = 0.0f;
        INFER_CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));

        int prompt_len = static_cast<int>(input_ids.size());
        // The first generated token is the argmax at step == prompt_len-1 (the
        // step whose embedding is the LAST prompt token); the steps below it
        // only predicted prompt-continuation tokens and are not part of the
        // output. Start the generated stream there (was prompt_len — an
        // off-by-one that dropped the first generated token).
        int first_gen_idx = std::max(prompt_len - 1, 0);
        int gen_count = static_cast<int>(gen_ids.size()) - first_gen_idx;
        auto gen_start = gen_ids.begin()
            + std::min(first_gen_idx, static_cast<int>(gen_ids.size()));
        std::vector<int32_t> gen_ids_i32(gen_start, gen_ids.end());
        result.output_text = lsxcommon::runtime::DecodeGeneratedText(
            tok_db, gen_ids_i32);
        float ttft_elapsed = 0.0f;
        INFER_CHECK(cudaEventElapsedTime(&ttft_elapsed, t_start, t_ttft));
        result.ttft_ms = static_cast<double>(ttft_elapsed);
        result.tokens_per_sec = elapsed_ms > 0.0f
            ? (static_cast<double>(gen_count) / elapsed_ms) * 1000.0
            : 0.0;
        result.total_time_ms = static_cast<double>(elapsed_ms);
        result.kv_slots_used = prompt_len + gen_count;
        result.weight_resident_bytes = pw.resident_bytes + d_alloc_bytes;
        result.num_tokens = gen_count;
        result.num_layers = topo.num_layers;
        result.gen_tokens = std::move(gen_ids_i32);
        result.ok = true;

        std::ostringstream res_oss;
        res_oss << "=== --testprompt Results ===\n"
                << "  prompt_tokens: \"" << input_ids.size() << "\"\n"
                << "  output:        \"" << result.output_text << "\"\n"
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
    if (stream && own_context) cudaStreamDestroy(stream);
    arena.Free();      // single cudaFree + single cudaFreeHost for the whole run
    for (int k = 0; k < kTopK; ++k)
        if (stage_ev[k]) cudaEventDestroy(stage_ev[k]);
    if (copy_stream) cudaStreamDestroy(copy_stream);
    if (expert_cache.d_base) cudaFree(expert_cache.d_base);
    if (t_start)       cudaEventDestroy(t_start);
    if (t_ttft)        cudaEventDestroy(t_ttft);
    if (t_stop)        cudaEventDestroy(t_stop);
    if (evict_done)    cudaEventDestroy(evict_done);
    if (ev_route)      cudaEventDestroy(ev_route);
    free_expert_index(expert_db);

    return result;
    } // end execution block
} // end RunInference

} // namespace gpu
} // namespace lsxcommon
