#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include "lsxcommon/deepseek_moe_inference_csa.h"
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/fast_math_mode.h"
#include "lsxcommon/expert_store.h"
#include "lsxcommon/vram_estimate.h"

namespace lsxcommon {
namespace gpu {

constexpr int kHiddenDim = 4096;
constexpr int kNumLayers = 43;
constexpr int kNumRouted = 256;
constexpr int kTopK = 6;
constexpr int kIntermediateDim = 2048;

// Sole supported vault index format since the NVFP4 repack (Task 3-4,
// docs/superpowers/plans/2026-09-11-deepseek-nvfp4-migration.md): routed
// experts carry pow2-E4M3 per-16 scales + per-tensor global_scale.
inline constexpr char kIndexFormatNvfp4[] = "logestix-v1-nvfp4";

// Routed experts are NVFP4-native (logestix-v1-nvfp4): kernel selection
// lives in the neutral lsx_nvfp4 layer; DeepSeekV4FlashModel::Infer asserts
// its per-call policy with lsx_nvfp4::LsxFastMathScope (model-independence
// contract). The legacy family-scoped toggle is retired.
// Frozen-geometry passport (MODEL INGEST INVARIANT step 3): compile-time
// relations lock the architecture against accidental drift.
static_assert(kTopK > 0 && kTopK <= kNumRouted,
              "top-k exceeds routed expert count");
static_assert(kHiddenDim % 256 == 0,
              "hidden must tile cleanly on Blackwell (256-wide)");
static_assert(kNumLayers > 0 && kIntermediateDim > 0,
              "bad MLP/layer geometry");

// DeepSeek-V4 MLA / Hyper-Connections geometry (from config.json)
constexpr int kHcMult = 4;                 // hc_mult: copies of hidden state
constexpr int kHcMix = (2 + kHcMult) * kHcMult;  // 24
constexpr int kHcDim = kHcMult * kHiddenDim;     // 16384
constexpr int kAttnHeads = 64;
constexpr int kHeadDim = 512;
constexpr int kRopeHeadDim = 64;
constexpr int kQloraRank = 1024;
constexpr int kOloraRank = 1024;
constexpr int kOGroups = 8;
constexpr int kAttnWindow = 128;           // sliding window (pure-MLA layers)
constexpr int kSinkhornIters = 20;
constexpr float kRmsEps = 1e-6f;

struct ExpertEntry {
    const uint8_t *host_w1 = nullptr;
    const uint8_t *host_w2 = nullptr;
    const uint8_t *host_w3 = nullptr;
    const uint8_t *host_w1_scales = nullptr;
    const uint8_t *host_w2_scales = nullptr;
    const uint8_t *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;
    float w3_global_scale = 1.0f;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;
    int64_t w3_scales_bytes = 0;
    // Pinned-blob fast path (LSX pin pass): when set, the entry's data
    // lives in one locked-host blob laid out [w1|w3|w2][s1|s3|s2] — the
    // exact GEMV streaming order — so staging is 2 copies instead of 6.
    const uint8_t *pinned_blob = nullptr;
    int64_t blob_pack_bytes = 0;    // w1b + w3b + w2b
    int64_t blob_scale_bytes = 0;   // s1b + s3b + s2b
};

struct ModelTopology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
    lsxcommon::VramProfile vram;   // conversion-time VRAM requirements
    // Index format tag ("logestix-v1-nvfp4" since the NVFP4 repack).
    std::string format;
    std::string error_msg;
};

ModelTopology load_topology(const std::string &model_path);

// Read `model` field from <model_path>/logestix/model.index.json.
// Returns normalized type (lowercase, ' ' -> '-'), or "" on any error.
std::string detect_model_type(const std::string &model_path);

// THREAD-SAFETY: this struct is NOT internally synchronized. It is safe only
// because RunInferenceModel constructs, uses and destroys its own ExpertDb
// entirely within a single call (never shared across calls or threads) and
// LsxOrchestrator::ServeTaskMosaic is documented single-consumer. If a future
// change makes an IModel instance serve concurrent Infer() calls from
// multiple threads (or hoists an ExpertDb into persistent per-model state),
// ensure_expert_resident/load_expert_index/free_expert_index will need an
// external lock: concurrent LRU touch/evict mutates `lru`/`entries`/`mmaps`
// without any synchronization.
using ExpertDb = lsxcommon::gpu::expert_store::ExpertStore<ExpertEntry>;

ExpertDb load_expert_index(const std::string &model_path,
                            int num_layers, int experts_per_layer);

// Ensure the mmap for (layer, expert) is resident in the LRU cache, evicting the
// least-recently-used entries as needed. No-op if already resident.
//
// `stream` is synchronized (once, only when an eviction is actually about to
// happen) before any mmap is unmapped: run_routed_experts / run_shared_experts
// issue cudaMemcpyAsync() calls that read directly from an entry's mmap'd host
// memory (host_w1/host_w2/... — this is the whole point of the zero-copy
// design), and cudaMemcpyAsync() returning to the host does NOT mean the DMA
// engine has finished reading the source buffer yet. Without this barrier,
// evicting (munmap-ing) an entry whose async H2D copy is still in flight is a
// use-after-free read of unmapped host memory from the GPU's DMA engine.
// @p evict_done, when non-null, releases that barrier via cudaEventRecord +
// cudaStreamWaitEvent instead of a full host-blocking cudaStreamSynchronize.
void ensure_expert_resident(ExpertDb &db, int layer, int expert,
                            cudaStream_t stream,
                            cudaEvent_t evict_done = nullptr);
// Release all expert mmaps (frees CRT file descriptors).
void free_expert_index(ExpertDb &db);

// ── Device-side resident-expert cache (Blackwell H2D-traffic elimination) ────
// Unified implementation: lsxcommon/expert_device_cache.h (model-unification
// 2026-08). Keeps the packed NVFP4 payloads + E8M0 scales of the hottest
// routed experts resident in VRAM so repeated selections skip the ~13 MB
// per-expert H2D staging entirely. All device access happens on the caller's
// single compute stream, so stream ordering alone makes slot reuse safe (a
// slot is only overwritten by copies enqueued after every prior kernel
// reading it). Host bookkeeping only — no device-side structures, no absl.
#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/fast_math_mode.h"

inline int moe_route(int token_idx, int layer_idx, int experts_per_layer)
{
    unsigned h = static_cast<unsigned>(token_idx) * 2654435761U
               + static_cast<unsigned>(layer_idx) * 314159265U;
    h ^= h >> 16; h *= 0x85ebca6bU;
    h ^= h >> 13; h *= 0xc2b2ae35U;
    h ^= h >> 16;
    return static_cast<int>(h % static_cast<unsigned>(experts_per_layer));
}

// Half <-> float elementwise conversion kernels shared by every stage of the
// pipeline (attention, MoE FFNs, the top-level RMSNorm bridge). Previously
// hand-copied with external (non-`static`) linkage into three separate .cu
// files (deepseek_moe_inference_pipeline.cu, deepseek_moe_inference_moe.cu,
// deepseek_moe_inference_attn.cu) — defined once here (header-`static`, so each
// TU that includes this file gets its own internal-linkage copy, the
// standard pattern for sharing small __global__ kernels across CUDA
// translation units without relocatable device code) to remove the
// triplicated, driftable copies and a since-removed misleading `extern`
// re-declaration in pipeline.cu that claimed (incorrectly) to resolve to
// deepseek_moe_inference_moe.cu's definition.
//
// Guarded by __CUDACC__: this header is also #included from model_factory.cpp
// (a plain host .cpp compiled by cl.exe/g++, not nvcc) purely for its host-
// only declarations (e.g. detect_model_type); threadIdx/blockIdx/blockDim
// below are CUDA device-compile builtins that a non-CUDA host compiler does
// not know about.
#ifdef __CUDACC__
static __global__ void half_to_float_kernel(
    const __half* __restrict__ src, float* __restrict__ dst, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) dst[i] = __half2float(src[i]);
}

static __global__ void float_to_half_kernel(
    const float* __restrict__ src, __half* __restrict__ dst, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Block-wide sum reduction via warp shuffles: log2(warps) barriers instead of
// a full shared-memory tree walk. Requires blockDim.x % 32 == 0 and an array
// of blockDim.x/32 floats. Deterministic (fixed butterfly order).
__device__ __forceinline__ float lsx_block_reduce_sum(float v, float* s_warp)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();
    const int nwarp = blockDim.x >> 5;
    if (warp == 0) {
        float w = (lane < nwarp) ? s_warp[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            w += __shfl_xor_sync(0xFFFFFFFFu, w, off);
        if (lane == 0) s_warp[0] = w;
    }
    __syncthreads();
    return s_warp[0];
}

// Fused FP16 RMSNorm: half in → FP32 square-sum (shuffle reduction) →
// normalize × gamma → clamp → half out. Replaces the 4-launch bridge
// (convert-in, gamma convert, fp32 rmsnorm, clamp-out) with one kernel.
static __global__ void __launch_bounds__(256)
rmsnorm_half_fused_kernel(const __half* __restrict__ x,
                           const __half* __restrict__ gamma,
                           __half* __restrict__ out, int dim, float eps)
{
    __shared__ float s_red[32];
    const int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        const float v = __half2float(x[i]);
        ss += v * v;
    }
    const float total = lsx_block_reduce_sum(ss, s_red);
    const float inv = rsqrtf(total / static_cast<float>(dim) + eps);
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = __half2float(x[i]) * inv * __half2float(gamma[i]);
        v = fmaxf(-65504.0f, fminf(65504.0f, v));
        out[i] = __float2half(v);
    }
}
#endif  // __CUDACC__

struct LayerWeights {
    __half *d_ffn_norm = nullptr;
    __half *d_gate_weight = nullptr;
    // Host-side router state (routing runs on the CPU). tid2eid: [vocab][topk]
    // int64 hash-routing table (present only for hash layers 0..n_hash_layers-1);
    // gate_bias: [n_routed] float selection bias (present only for non-hash layers).
    int64_t *h_tid2eid = nullptr;
    int tid2eid_vocab = 0;
    int tid2eid_topk = 0;
    float *h_gate_bias = nullptr;
    int gate_bias_n = 0;
    uint8_t *d_shared_w1 = nullptr;
    uint8_t *d_shared_w2 = nullptr;
    uint8_t *d_shared_w3 = nullptr;
    uint8_t *d_shared_w1_scales = nullptr;
    uint8_t *d_shared_w2_scales = nullptr;
    uint8_t *d_shared_w3_scales = nullptr;
    __half *d_shared_w1_h = nullptr;   // load-time FP16 dequant cache (W1)
    __half *d_shared_w3_h = nullptr;   // load-time FP16 dequant cache (W3)
    float  *d_shared_w2_f32 = nullptr; // load-time FP32 dequant cache (W2)
    float shared_w1_global_scale = 1.0f;
    float shared_w2_global_scale = 1.0f;
    float shared_w3_global_scale = 1.0f;
    int compress_ratio = 0;          // 0 = pure MLA, >0 = CSA
    float rope_theta = 10000.0f;     // 10000 for MLA, 160000 for CSA
    int comp_coff = 1;               // 2 for ratio=4 (overlap), 1 for ratio=128

    // CSA Compressor (layers 2-41; nullptr for pure MLA layers 0,1,42)
    __half *d_comp_wkv = nullptr;        // [coff*head_dim, 4096] BF16→FP16
    __half *d_comp_wgate = nullptr;     // [coff*head_dim, 4096] BF16→FP16
    float  *d_comp_ape = nullptr;       // [ratio, coff*head_dim] F32
    __half *d_comp_norm = nullptr;      // [head_dim] BF16→FP16

    // CSA Indexer (ratio=4 only; nullptr for ratio=128 and pure MLA)
    uint8_t *d_idx_wq_b = nullptr;          // F8_E4M3 [8192, 1024] (device-malloc'd)
    uint8_t *d_idx_wq_b_scales = nullptr;   // F8_E8M0 [64, 8]
    int idx_wq_b_rows = 0, idx_wq_b_cols = 0;
    __half *d_idx_weights_proj = nullptr;   // BF16→FP16 [64, 4096]
    // Indexer's own Compressor (head_dim=128)
    __half *d_idx_comp_wkv = nullptr;        // [256, 4096] BF16→FP16
    __half *d_idx_comp_wgate = nullptr;     // [256, 4096] BF16→FP16
    float  *d_idx_comp_ape = nullptr;       // [4, 256] F32
    __half *d_idx_comp_norm = nullptr;      // [128] BF16→FP16
    int64_t shared_w1_bytes = 0;
    int64_t shared_w2_bytes = 0;
    int64_t shared_w3_bytes = 0;

    // MLA attention (F8_E4M3 raw + F8_E8M0 per-128x128 tile scales)
    const uint8_t *d_wq_a = nullptr;   const uint8_t *d_wq_a_scales = nullptr;
    const uint8_t *d_wq_b = nullptr;   const uint8_t *d_wq_b_scales = nullptr;
    const uint8_t *d_wkv  = nullptr;   const uint8_t *d_wkv_scales  = nullptr;
    const uint8_t *d_wo_a = nullptr;   const uint8_t *d_wo_a_scales = nullptr;
    const uint8_t *d_wo_b = nullptr;   const uint8_t *d_wo_b_scales = nullptr;
    int64_t wq_a_n = 0, wq_b_n = 0, wkv_n = 0, wo_a_n = 0, wo_b_n = 0;
    int64_t wq_a_s_n = 0, wq_b_s_n = 0, wkv_s_n = 0, wo_a_s_n = 0, wo_b_s_n = 0;
    int wq_a_rows = 0, wq_a_cols = 0;
    int wq_b_rows = 0, wq_b_cols = 0;
    int wkv_rows = 0, wkv_cols = 0;
    int wo_a_rows = 0, wo_a_cols = 0;
    int wo_b_rows = 0, wo_b_cols = 0;
    __half *d_q_norm = nullptr;      // [1024]
    __half *d_kv_norm = nullptr;     // [512]
    __half *d_attn_norm = nullptr;   // [4096]
    float  *d_attn_sink = nullptr;   // [64]

    // Hyper-Connections (FP32)
    float *d_hc_attn_fn = nullptr;    // [24, 16384]
    float *d_hc_attn_base = nullptr;  // [24]
    float *d_hc_attn_scale = nullptr; // [3]
    float *d_hc_ffn_fn = nullptr;    // [24, 16384]
    float *d_hc_ffn_base = nullptr;  // [24]
    float *d_hc_ffn_scale = nullptr; // [3]

    // Non-null when the layer was uploaded arena-backed: every d_* device
    // pointer above is a view into this single allocation and Free()
    // releases the arena instead of the individual pointers. Host-pinned
    // router state (h_tid2eid/h_gate_bias) is NOT part of the arena.
    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

struct PinnedWeights {
    __half *d_embed = nullptr;
    __half *d_head = nullptr;
    __half *d_final_norm = nullptr;
    float *d_hc_head_fn = nullptr;    // [4, 16384]
    float *d_hc_head_base = nullptr;  // [4]
    float *d_hc_head_scale = nullptr; // [1]
    std::vector<LayerWeights> layers;
    int vocab_size = 0;
    std::uint64_t resident_bytes = 0;   // total device bytes held on load

    void Free();
};

// ── deepseek_moe_inference_moe.cu ──────────────────────────────────────────────
// Routes one token: gate GEMM → sqrt(softplus) scores → top-k → weights.
// `d_logits_scratch` / `h_logits_scratch` are caller-owned (arena carve) — the
// function performs ZERO CUDA allocations (spec item 12, malloc-per-layer kill).
cudaError_t route_token(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    const int64_t *h_tid2eid, int tid2eid_stride, int token_id,
    const float *h_gate_bias,
    int topk,
    __half *d_logits_scratch,   // caller-owned [num_routed] __half
    float *h_logits_scratch,    // caller-owned [num_routed] float (pinned)
    int *h_top_indices, float *h_top_weights);

// Async split of route_token for pipeline overlap: begin enqueues the gate
// GEMM + D2H logits copy and records @p ev (NO host wait); finish waits on
// @p ev only (the caller is free to enqueue independent GPU work in between,
// e.g. the shared-expert FFN) and then runs the identical host-side scoring.
// Fused route_token == begin + finish, bit-exact.
cudaError_t route_token_begin(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const __half *d_gate, const __half *d_hidden,
    int hidden_dim, int num_routed,
    __half *d_logits_scratch,   // caller-owned [num_routed] __half
    float *h_logits_scratch,    // caller-owned [num_routed] float (pinned)
    cudaEvent_t ev);

cudaError_t route_token_finish(
    int num_routed, int token_id,
    const int64_t *h_tid2eid, int tid2eid_stride,
    const float *h_gate_bias,
    int topk,
    float *h_logits_scratch,    // filled by begin
    int *h_top_indices, float *h_top_weights,
    cudaEvent_t ev);

// Global CUDA-allocation counter for per-run inference footprints. The arena's
// Allocate() increments it by 2 (device + pinned host block); route_token and
// load_pinned_weights must not touch it. Header-inline function-local static →
// one shared counter across every TU (and test) that includes this header.
inline std::atomic<std::uint64_t>& InferenceAllocCount()
{
    static std::atomic<std::uint64_t> c{0};
    return c;
}

// Single-allocation per-run arena (spec item 12). One device cudaMalloc + one
// pinned cudaMallocHost carve every direct buffer, the CSA per-layer state and
// the routing/logits staging for one RunInferenceModel call. Mirrors
// GlmMoEArena::AllocateGeneration. Task 8 reads `bytes` for occupancy
// accounting.
struct DeepSeekInferArena {
    // device block
    uint8_t* d_base = nullptr;
    __half* d_single = nullptr;      // [kHiddenDim]
    __half* d_normed = nullptr;      // [kHiddenDim]
    __half* d_ffn_out = nullptr;     // [kHiddenDim]
    float*  d_hidden = nullptr;      // [kHcDim]
    float*  d_hidden_next = nullptr; // [kHcDim]
    __half* d_shared_int = nullptr;  // [kHiddenDim]
    __half* d_shared_gate = nullptr; // [kHiddenDim]
    __half* d_routed_buf = nullptr;  // [kHiddenDim]
    float*  d_dequant_scratch = nullptr;
    float*  d_act_quant = nullptr;
    float*  d_ffn_accum = nullptr;   // [kHiddenDim]
    float*  d_swiglu_out = nullptr;  // [kIntermediateDim]
    float*  d_act_scales = nullptr;
    float*  d_gate_f32 = nullptr;    // [kIntermediateDim]
    float*  d_up_f32 = nullptr;      // [kIntermediateDim]
    float*  d_attn_dequant = nullptr;
    __half* d_kv_cache = nullptr;
    __half* d_q = nullptr, *d_qr = nullptr, *d_qr2 = nullptr;
    __half* d_kv = nullptr, *d_kv2 = nullptr;
    __half* d_o = nullptr, *d_olora = nullptr, *d_attn_out_h = nullptr;
    float*  d_attn_out = nullptr;
    float*  d_hc_post = nullptr, *d_hc_comb = nullptr, *d_hc_pre = nullptr;
    float*  d_logits = nullptr;
    float*  d_norm_scratch = nullptr;
    float*  d_norm_gamma_scratch = nullptr;
    int32_t* d_next_token = nullptr;
    float*  d_next_maxv = nullptr;
    uint8_t* d_pack_stage = nullptr;
    uint8_t* d_scale_stage = nullptr;
    uint8_t* d_scale_stage_tc = nullptr;  // UE4M3 repack target (TC W4A4)
    __half* d_routing_scratch = nullptr;  // [kNumRouted] route_token logits
    CSAState* csa_states = nullptr;       // [num_layers] host-side state records

    // pinned host block
    uint8_t* h_base = nullptr;
    int*   h_topk = nullptr;         // [kTopK]
    float* h_top_weights = nullptr;  // [kTopK]
    float* h_routing_scratch = nullptr; // [kNumRouted] route_token logits

    std::size_t bytes = 0;           // total device bytes (resident accounting)
    std::size_t host_bytes = 0;

    // Allocates the whole per-run footprint: one device cudaMalloc + one pinned
    // cudaMallocHost. `num_layers`/`compress_ratios` bound the per-layer CSA
    // region (0 for pure-MLA layers); `max_seq_len` sizes the CSA compressed
    // slots + KV cache; `vocab` sizes the logits buffer. Increments
    // InferenceAllocCount() by 2. Buffers are laid out 16-byte aligned so every
    // element pointer stays properly aligned inside the single block.
    cudaError_t Allocate(cudaStream_t stream, int num_layers, int max_seq_len,
                         const int* compress_ratios, int vocab);
    void Free();
};

cudaError_t load_pinned_weights(const std::string &model_path,
                                 PinnedWeights &pw,
                                 cudaStream_t stream);

// ── deepseek_moe_inference_attn.cu ───────────────────────────────────────────────
// Hyper-Connections (HC) — HC state stored in FP32 for numerical stability.
// run_hc_pre / run_hc_head are two-phase (GPU-wide partials + finalize) and
// need a small FP32 scratch of ≥ (kHcMix + 2) * 32 floats; the caller's
// d_norm_scratch is safe to pass (stream ordering: rmsnorm runs strictly
// after hc_pre completes).
cudaError_t run_hc_pre(cudaStream_t stream,
                       const float *d_state, const float *d_fn,
                       const float *d_scale, const float *d_base,
                       __half *d_single, float *d_post, float *d_comb,
                       float *d_pre, float *d_scratch,
                       float rms_eps = 1e-6f);

cudaError_t run_hc_post(cudaStream_t stream,
                         const float *d_x, const float *d_residual,
                         const float *d_post, const float *d_comb,
                         float *d_y);

cudaError_t run_hc_head(cudaStream_t stream,
                        const float *d_state, const float *d_fn,
                        const float *d_scale, const float *d_base,
                        __half *d_single, float *d_scratch);

cudaError_t run_expand_hc(cudaStream_t stream,
                          const __half *d_single, float *d_state);

// MLA forward (one token)
cudaError_t run_mla_attention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const LayerWeights &lw,
    const __half *d_x,
    __half *d_kv_cache_layer,
    int start_pos,
    __half *d_q, __half *d_qr, __half *d_qr2,
    __half *d_kv, __half *d_kv2,
    __half *d_o, __half *d_olora, __half *d_out,
    float *d_dequant,
    float *d_norm_scratch, float *d_norm_gamma_scratch);

} // namespace gpu
} // namespace lsxcommon
