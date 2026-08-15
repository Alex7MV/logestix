#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "lsxcommon/qwen_convert.h"

namespace lsxcommon::qwen {

// One NVFP4-quantized dense weight on device: E2M1 nibbles + E4M3 per-16 block
// scales + global float scale + the ORIGINAL weight shape (rows x cols).
struct QwenNvfp4Weight {
    const uint8_t *d_packed = nullptr;
    const uint8_t *d_scales = nullptr;
    float global_scale = 1.0f;
    int rows = 0;   // input dim (K for GEMM: W is [K x N])
    int cols = 0;   // output dim (N)
    int64_t packed_bytes = 0;
    int64_t scale_bytes = 0;

    void Free() {
        cudaFree(const_cast<uint8_t*>(d_packed));
        cudaFree(const_cast<uint8_t*>(d_scales));
        d_packed = nullptr; d_scales = nullptr;
    }
};

// Per decoder layer weights. `is_linear` selects linear_attn vs self_attn.
struct QwenLayerWeights {
    bool is_linear = true;
    bool loaded = false;

    // Norms (raw BF16 -> FP16).
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    int norm_size = 0;

    // Linear attention (Qwen3_5GatedDeltaNet).
    QwenNvfp4Weight qkv;      // hidden -> k_dim*2 + v_dim
    QwenNvfp4Weight z;        // hidden -> v_dim
    QwenNvfp4Weight b;        // hidden -> num_v_heads
    QwenNvfp4Weight a;        // hidden -> num_v_heads
    QwenNvfp4Weight out;      // v_dim -> hidden
    __half *d_conv1d = nullptr;   // depthwise conv weight [conv_dim x kernel]
    float *d_A_log = nullptr;     // [num_v_heads]
    float *d_dt_bias = nullptr;   // [num_v_heads]
    __half *d_attn_norm = nullptr; // linear_attn.norm.weight [v_head_dim] (plain form)
    int attn_norm_size = 0;
    int conv_dim = 0;
    int conv_kernel = 0;

    // Full attention (Qwen3_5Attention).
    QwenNvfp4Weight q;        // hidden -> num_heads*head_dim*2 (q + gate)
    QwenNvfp4Weight k;        // hidden -> num_kv_heads*head_dim
    QwenNvfp4Weight v;        // hidden -> num_kv_heads*head_dim
    QwenNvfp4Weight o;        // num_heads*head_dim -> hidden
    __half *d_q_norm = nullptr;
    __half *d_k_norm = nullptr;
    int q_norm_size = 0;
    int k_norm_size = 0;

    // MLP (SwiGLU).
    QwenNvfp4Weight mlp_gate; // hidden -> intermediate
    QwenNvfp4Weight mlp_up;   // hidden -> intermediate
    QwenNvfp4Weight mlp_down; // intermediate -> hidden

    void Free();
};

struct QwenPinnedWeights {
    QwenTopology top;
    QwenNvfp4Weight embed;        // vocab x hidden
    QwenNvfp4Weight head;         // vocab x hidden
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<QwenLayerWeights> layers;

    void Free();
};

// Loads all Qwen3.5-9B weights from output _root (either <root>/logestix/pinned
// or <root>/pinned) into device memory on `stream`. Returns false on failure.
bool LoadQwenPinnedWeights(const std::string &output_root,
                           QwenPinnedWeights &pw, cudaStream_t stream);

std::vector<uint8_t> ReadFileBytes(const std::string &path);

} // namespace lsxcommon::qwen