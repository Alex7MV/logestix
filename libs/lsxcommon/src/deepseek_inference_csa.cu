#include "lsxcommon/deepseek_inference_csa.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace lsxcommon {
namespace gpu {

// State management only — run_csa_attention is defined in deepseek_inference_attn.cu
// where it has direct access to the MLA kernels (f8_gemm_1d, run_rmsnorm_attn,
// per_head_rms_kernel, rope_apply_kernel, dequant_f8_to_half_kernel, mla_attn_kernel).

// score_state must start as -inf (reference: torch.full(..., -inf)),
// because the overlap pooling treats rows [0, ratio) before the first shift
// as masked (their softmax weight must be ~0).
__global__ void fill_f32_neg_inf_kernel(float* p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = -INFINITY;
}

// Per-layer compressor state sizing, matching reference Compressor.
//   kv_state:     [ratio*coff, coff*head_dim] FP32
//   idx state:    [ratio*coff_idx, coff_idx*idx_head_dim] FP32 (idx_head_dim=128)
//   compressed:   [max_seq_len / ratio + 1, head_dim] FP16
//   idx comp:     [max_seq_len / ratio + 1, idx_head_dim] FP16
cudaError_t alloc_csa_state(CSAState *states, int num_layers,
                             int max_seq_len, int head_dim, cudaStream_t stream,
                             const int *compress_ratios)
{
    constexpr int kIdxHeadDim = 128;
    // Every cudaMalloc/cudaMemsetAsync/kernel-launch below is checked. On the
    // first failure (e.g. OOM) we stop allocating further layers and return
    // the real error immediately instead of reporting cudaSuccess with a
    // null d_comp_kv_state/d_comp_score_state that a later kernel (e.g.
    // compressor_step_kernel) would silently write through. The caller is
    // responsible for calling free_csa_state() on error too (CSAState fields
    // default-initialize to nullptr, so partially-populated entries and
    // never-touched trailing layers are both safe to free).
    for (int l = 0; l < num_layers; ++l) {
        auto &s = states[l];
        const int ratio = (l < num_layers && compress_ratios) ? compress_ratios[l] : 0;
        if (ratio <= 0) continue;  // pure-MLA layer: no CSA state

        const int coff = (ratio == 4) ? 2 : 1;          // overlap -> coff=2
        const int rows = coff * ratio;
        const int state_cols = coff * head_dim;
        int state_sz = rows * state_cols;

        cudaError_t e = cudaMalloc(&s.d_comp_kv_state, static_cast<size_t>(state_sz) * sizeof(float));
        if (e != cudaSuccess) return e;
        e = cudaMalloc(&s.d_comp_score_state, static_cast<size_t>(state_sz) * sizeof(float));
        if (e != cudaSuccess) return e;
        e = cudaMemsetAsync(s.d_comp_kv_state, 0, static_cast<size_t>(state_sz) * sizeof(float), stream);
        if (e != cudaSuccess) return e;
        fill_f32_neg_inf_kernel<<<(state_sz + 255) / 256, 256, 0, stream>>>(s.d_comp_score_state, state_sz);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;

        int max_comp = max_seq_len / ratio + 1;
        e = cudaMalloc(&s.d_compressed_kv,
            static_cast<size_t>(max_comp) * head_dim * sizeof(__half));
        if (e != cudaSuccess) return e;
        e = cudaMemsetAsync(s.d_compressed_kv, 0,
            static_cast<size_t>(max_comp) * head_dim * sizeof(__half), stream);
        if (e != cudaSuccess) return e;

        // Indexer compressor state (only ratio==4 layers have an indexer).
        int idx_state_sz = 2 * 4 * 2 * kIdxHeadDim;
        e = cudaMalloc(&s.d_idx_kv_state, static_cast<size_t>(idx_state_sz) * sizeof(float));
        if (e != cudaSuccess) return e;
        e = cudaMalloc(&s.d_idx_score_state, static_cast<size_t>(idx_state_sz) * sizeof(float));
        if (e != cudaSuccess) return e;
        e = cudaMemsetAsync(s.d_idx_kv_state, 0, static_cast<size_t>(idx_state_sz) * sizeof(float), stream);
        if (e != cudaSuccess) return e;
        fill_f32_neg_inf_kernel<<<(idx_state_sz + 255) / 256, 256, 0, stream>>>(s.d_idx_score_state, idx_state_sz);
        e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        e = cudaMalloc(&s.d_idx_compressed_kv,
            static_cast<size_t>(max_comp) * kIdxHeadDim * sizeof(__half));
        if (e != cudaSuccess) return e;
        e = cudaMemsetAsync(s.d_idx_compressed_kv, 0,
            static_cast<size_t>(max_comp) * kIdxHeadDim * sizeof(__half), stream);
        if (e != cudaSuccess) return e;
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
