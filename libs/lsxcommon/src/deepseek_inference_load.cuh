#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>

namespace lsxcommon {
namespace gpu {

constexpr int kHiddenDim = 4096;
constexpr int kNumLayers = 43;
constexpr int kNumRouted = 256;
constexpr int kTopK = 6;
constexpr int kIntermediateDim = 2048;

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
};

struct ModelTopology {
    int num_layers = 0;
    int experts_per_layer = 0;
    int64_t total_experts = 0;
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
struct ExpertDb {
    std::vector<ExpertEntry> entries;
    std::vector<std::shared_ptr<arrow::io::MemoryMappedFile>> mmaps;
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches;
    std::vector<std::string> paths;       // file path per flat index (for lazy open)
    std::vector<size_t> lru;              // flat indices, back = least recently used
    size_t max_open = 7000;               // bounded by Windows CRT fd cap (8192)
    int experts_per_layer = 0;
    int num_layers = 0;
    std::string error_msg;
};

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
void ensure_expert_resident(ExpertDb &db, int layer, int expert,
                            cudaStream_t stream);

// Release all expert mmaps (frees CRT file descriptors).
void free_expert_index(ExpertDb &db);

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
// files (deepseek_inference_pipeline.cu, deepseek_inference_moe.cu,
// deepseek_inference_attn.cu) — defined once here (header-`static`, so each
// TU that includes this file gets its own internal-linkage copy, the
// standard pattern for sharing small __global__ kernels across CUDA
// translation units without relocatable device code) to remove the
// triplicated, driftable copies and a since-removed misleading `extern`
// re-declaration in pipeline.cu that claimed (incorrectly) to resolve to
// deepseek_inference_moe.cu's definition.
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
    float *d_hc_ffn_fn = nullptr;     // [24, 16384]
    float *d_hc_ffn_base = nullptr;   // [24]
    float *d_hc_ffn_scale = nullptr;  // [3]

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

    void Free();
};

cudaError_t load_pinned_weights(const std::string &model_path,
                                 PinnedWeights &pw,
                                 cudaStream_t stream);

// ── deepseek_inference_attn.cu ───────────────────────────────────────────────
// Hyper-Connections (HC) — HC state stored in FP32 for numerical stability
cudaError_t run_hc_pre(cudaStream_t stream,
                       const float *d_state, const float *d_fn,
                       const float *d_scale, const float *d_base,
                       __half *d_single, float *d_post, float *d_comb,
                       float *d_pre);

cudaError_t run_hc_post(cudaStream_t stream,
                         const float *d_x, const float *d_residual,
                         const float *d_post, const float *d_comb,
                         float *d_y);

cudaError_t run_hc_head(cudaStream_t stream,
                        const float *d_state, const float *d_fn,
                        const float *d_scale, const float *d_base,
                        __half *d_single);

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
