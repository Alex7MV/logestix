#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include <cuda_fp16.h>

#include "lsxcommon/lsx_inference_load.h"  // attn::LayerWeights
#include "lsxcommon/qwen_dense_convert.h"

namespace lsxcommon::qwen {

// One NVFP4-quantized dense weight on device: E2M1 nibbles + E4M3 per-16 block
// scales + global float scale + the ORIGINAL weight shape (rows x cols).
// Device NVFP4 weight container moved to the model-neutral NVFP4 layer
// (lsxcommon/lsx_inference_nvfp4.h); the qwen-named alias keeps the
// qwen-family call sites unchanged.

// Per decoder layer weights. `is_linear` selects linear_attn vs self_attn.

struct QwenPinnedWeights {
    QwenTopology top;
    lsx_nvfp4::LsxNvfp4Weight embed;        // vocab x hidden
    lsx_nvfp4::LsxNvfp4Weight head;         // vocab x hidden
    // The same two edges as stored BF16 planes (the `bf16` codec profile, which
    // has no global_nvfp4.arrow at all). The pipeline gathers from / projects
    // with whichever pair is resident.
    gemv::DenseWeight embed_bf16;
    gemv::DenseWeight head_bf16;
    // The two dense edges share one arena (RunUploadJobs allocates per call).
    void *d_arena_bf16 = nullptr;
    size_t arena_bytes_bf16 = 0;
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<attn::LayerWeights> layers;
    std::uint64_t resident_bytes = 0;   // total device bytes held on load

    void Free();
};

// Loads all Qwen3.5-9B weights from output _root (either <root>/logestix/pinned
// or <root>/pinned) into device memory on `stream`. Returns false on failure.
// Adapts the model passport to the geometry the neutral kernels take. Lives
// here, not in lsx_inference_load.h, because the reverse dependency (a
// neutral header naming a model passport) is exactly what the neutrality
// guard forbids.
inline attn::AttnGeometry MakeAttnGeometry(const QwenTopology &top)
{
    return attn::AttnGeometry{
        .hidden_size = top.hidden_size,
        .intermediate_size = top.intermediate_size,
        .vocab_size = top.vocab_size,
        .num_attention_heads = top.num_attention_heads,
        .num_key_value_heads = top.num_key_value_heads,
        .head_dim = top.head_dim,
        .linear_num_key_heads = top.linear_num_key_heads,
        .linear_num_value_heads = top.linear_num_value_heads,
        .linear_key_head_dim = top.linear_key_head_dim,
        .linear_value_head_dim = top.linear_value_head_dim,
        .rms_norm_eps = top.rms_norm_eps,
        .partial_rotary_factor = top.partial_rotary_factor,
        .rope_theta = top.rope_theta};
}

inline attn::KvGeometry MakeKvGeometry(const QwenTopology &top)
{
    attn::KvGeometry g;
    g.num_key_value_heads = top.num_key_value_heads;
    g.head_dim = top.head_dim;
    g.linear_num_key_heads = top.linear_num_key_heads;
    g.linear_key_head_dim = top.linear_key_head_dim;
    g.linear_num_value_heads = top.linear_num_value_heads;
    g.linear_value_head_dim = top.linear_value_head_dim;
    for (const auto &lt : top.layer_types)
        (lt == "linear_attention" ? g.num_linear_layers : g.num_full_layers)++;
    return g;
}

bool LoadQwenPinnedWeights(const std::string &output_root,
                           QwenPinnedWeights &pw, cudaStream_t stream);

std::vector<uint8_t> ReadFileBytes(const std::string &path);

} // namespace lsxcommon::qwen