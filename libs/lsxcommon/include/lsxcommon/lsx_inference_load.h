#pragma once
// MODEL-NEUTRAL per-layer weight bundle and attention geometry.
//
// LayerWeights holds one decoder layer's device weights: NVFP4 matrices, raw
// norms, the linear-attention recurrence constants and an optional arena. It
// carries no model semantics -- every dimension arrives with the tensors -- so
// it lives here rather than in a family loader. The attention/MLP kernels in
// lsx_attn_inference.cu are its only consumers.
//
// AttnGeometry is the subset of a model passport the kernels actually read.
// The passport itself also carries layer_types, vram estimates and error text,
// which no kernel touches, so it is deliberately not passed down.
//
// Device-safe: no abseil, no arrow, no host-only containers on the device path.

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon {
namespace attn {

struct LayerWeights {
    bool is_linear = true;
    bool loaded = false;

    // Norms (raw BF16 -> FP16).
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    int norm_size = 0;

    // Linear attention (Qwen3_5GatedDeltaNet).
    lsx_nvfp4::LsxNvfp4Weight qkv;      // hidden -> k_dim*2 + v_dim
    lsx_nvfp4::LsxNvfp4Weight z;        // hidden -> v_dim
    lsx_nvfp4::LsxNvfp4Weight b;        // hidden -> num_v_heads
    lsx_nvfp4::LsxNvfp4Weight a;        // hidden -> num_v_heads
    lsx_nvfp4::LsxNvfp4Weight out;      // v_dim -> hidden
    __half *d_conv1d = nullptr;   // depthwise conv weight [conv_dim x kernel]
    float *d_A_log = nullptr;     // [num_v_heads]
    float *d_dt_bias = nullptr;   // [num_v_heads]
    __half *d_attn_norm = nullptr; // linear_attn.norm.weight [v_head_dim] (plain form)
    int attn_norm_size = 0;
    int conv_dim = 0;
    int conv_kernel = 0;

    // Full attention (Qwen3_5Attention).
    lsx_nvfp4::LsxNvfp4Weight q;        // hidden -> num_heads*head_dim*2 (q + gate)
    lsx_nvfp4::LsxNvfp4Weight k;        // hidden -> num_kv_heads*head_dim
    lsx_nvfp4::LsxNvfp4Weight v;        // hidden -> num_kv_heads*head_dim
    lsx_nvfp4::LsxNvfp4Weight o;        // num_heads*head_dim -> hidden
    __half *d_q_norm = nullptr;
    __half *d_k_norm = nullptr;
    int q_norm_size = 0;
    int k_norm_size = 0;

    // MLP (SwiGLU).
    lsx_nvfp4::LsxNvfp4Weight mlp_gate; // hidden -> intermediate
    lsx_nvfp4::LsxNvfp4Weight mlp_up;   // hidden -> intermediate
    lsx_nvfp4::LsxNvfp4Weight mlp_down; // intermediate -> hidden
    // Optional fused gate+up ([2I, K], rows [0,I)=gate, [I,2I)=up) emitted by
    // the converter when it quantizes both projections jointly with a single
    // global scale. Present only in models converted after 2026-08-24; the
    // fast-math path prefers it (one GEMM + one act quant), the bit-exact
    // scalar path keeps using mlp_gate/mlp_up.
    lsx_nvfp4::LsxNvfp4Weight mlp_gate_up;

    // Non-null when the layer was uploaded arena-backed: every d_* 1D tensor
    // pointer above is a view into this single allocation and Free() releases
    // the arena instead of the individual pointers.
    void *d_arena = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

// The geometry scalars the attention/MLP kernels read from the model passport.
struct AttnGeometry {
    int hidden_size = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    double rms_norm_eps = 1e-6;
    double partial_rotary_factor = 0.25;
    double rope_theta = 1e7;
};

} // namespace attn
} // namespace lsxcommon
