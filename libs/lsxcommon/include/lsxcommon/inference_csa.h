#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

namespace lsxcommon {
namespace gpu {

// Forward declaration
struct LayerWeights;

// CSA state per layer (allocated once, reused across steps)
struct CSAState {
    float  *d_comp_kv_state = nullptr;     // [max_coff_ratio*2, coff*head_dim] FP32
    float  *d_comp_score_state = nullptr;  // [max_coff_ratio*2, coff*head_dim] FP32
    __half *d_compressed_kv = nullptr;     // [max_compressed_slots, head_dim] FP16

    // Indexer compressor state (ratio=4 only)
    float  *d_idx_kv_state = nullptr;
    float  *d_idx_score_state = nullptr;
    __half *d_idx_compressed_kv = nullptr;

    int n_compressed = 0;
};

cudaError_t alloc_csa_state(CSAState *states, int num_layers,
                             int max_seq_len, int head_dim, cudaStream_t stream);

void free_csa_state(CSAState *states, int num_layers);

cudaError_t run_csa_attention(
    cublasHandle_t cublas_h, cudaStream_t stream,
    const LayerWeights &lw,
    const __half *d_x,
    __half *d_kv_cache_layer,
    CSAState &csa_state,
    int start_pos,
    __half *d_q, __half *d_qr, __half *d_qr2,
    __half *d_kv, __half *d_kv2,
    __half *d_o, __half *d_olora, __half *d_out,
    float *d_dequant,
    float *d_norm_scratch, float *d_norm_gamma_scratch);

} // namespace gpu
} // namespace lsxcommon
