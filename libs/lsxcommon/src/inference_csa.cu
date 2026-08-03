#include "lsxcommon/inference_csa.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace lsxcommon {
namespace gpu {

// State management only — run_csa_attention is defined in inference_attn.cu
// where it has direct access to the MLA kernels (f8_gemm_1d, run_rmsnorm_attn,
// per_head_rms_kernel, rope_apply_kernel, dequant_f8_to_half_kernel, mla_attn_kernel).

cudaError_t alloc_csa_state(CSAState *states, int num_layers,
                             int max_seq_len, int head_dim, cudaStream_t stream)
{
    int max_comp = max_seq_len / 4 + 1;
    for (int l = 0; l < num_layers; ++l) {
        auto &s = states[l];
        int state_sz = 2 * 4 * 2 * 512;
        cudaMalloc(&s.d_comp_kv_state, state_sz * sizeof(float));
        cudaMalloc(&s.d_comp_score_state, state_sz * sizeof(float));
        cudaMemsetAsync(s.d_comp_kv_state, 0, state_sz * sizeof(float), stream);
        cudaMemsetAsync(s.d_comp_score_state, 0, state_sz * sizeof(float), stream);
        cudaMalloc(&s.d_compressed_kv,
            static_cast<size_t>(max_comp) * head_dim * sizeof(__half));
        cudaMemsetAsync(s.d_compressed_kv, 0,
            static_cast<size_t>(max_comp) * head_dim * sizeof(__half), stream);

        int idx_state_sz = 2 * 4 * 2 * 128;
        cudaMalloc(&s.d_idx_kv_state, idx_state_sz * sizeof(float));
        cudaMalloc(&s.d_idx_score_state, idx_state_sz * sizeof(float));
        cudaMemsetAsync(s.d_idx_kv_state, 0, idx_state_sz * sizeof(float), stream);
        cudaMemsetAsync(s.d_idx_score_state, 0, idx_state_sz * sizeof(float), stream);
        cudaMalloc(&s.d_idx_compressed_kv,
            static_cast<size_t>(max_comp) * 128 * sizeof(__half));
        cudaMemsetAsync(s.d_idx_compressed_kv, 0,
            static_cast<size_t>(max_comp) * 128 * sizeof(__half), stream);
    }
    return cudaSuccess;
}

void free_csa_state(CSAState *states, int num_layers)
{
    for (int l = 0; l < num_layers; ++l) {
        auto &s = states[l];
        if (s.d_comp_kv_state) cudaFree(s.d_comp_kv_state);
        if (s.d_comp_score_state) cudaFree(s.d_comp_score_state);
        if (s.d_compressed_kv) cudaFree(s.d_compressed_kv);
        if (s.d_idx_kv_state) cudaFree(s.d_idx_kv_state);
        if (s.d_idx_score_state) cudaFree(s.d_idx_score_state);
        if (s.d_idx_compressed_kv) cudaFree(s.d_idx_compressed_kv);
    }
}

} // namespace gpu
} // namespace lsxcommon
