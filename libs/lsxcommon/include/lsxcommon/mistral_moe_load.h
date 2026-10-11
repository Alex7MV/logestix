#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_moe_inference.h"

namespace lsxcommon {
namespace mistral_moe {

// One BF16 dense matrix resident on the device. The Mistral checkpoint keeps
// every attention projection, the router and the embeddings in BF16 (its own
// quantization ignore-list excludes them), so they are uploaded raw and read
// through the model-neutral BF16 GEMV (lsx_gemv).
struct MistralBf16Weight {
    const __nv_bfloat16 *data = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t bytes = 0;
    bool valid() const { return data != nullptr; }
};

// Per-layer resident weights.
struct MistralLayerWeights {
    // MLA projections (BF16). wkv_a_with_mqa is [kv_lora + qk_rope, H].
    MistralBf16Weight wq_a;    // [q_lora, H]
    MistralBf16Weight wq_b;    // [heads*qk, q_lora]
    MistralBf16Weight wkv_a;   // [kv_lora + qk_rope, H]
    MistralBf16Weight wkv_b;   // [heads*(qk_nope + v), kv_lora]
    MistralBf16Weight wo;      // [H, heads*v]
    // Norms (BF16, never quantized).
    const __nv_bfloat16 *q_a_norm = nullptr;   // [q_lora]
    const __nv_bfloat16 *kv_a_norm = nullptr;  // [kv_lora]
    const __nv_bfloat16 *input_ln = nullptr;   // [H]
    const __nv_bfloat16 *post_ln = nullptr;    // [H]
    // Router (BF16 [E,H]: its argmax selects the experts).
    const __nv_bfloat16 *router = nullptr;
    int64_t router_bytes = 0;
    // Shared expert (NVFP4, resident).
    lsx_nvfp4::LsxNvfp4Weight shared_gate;
    lsx_nvfp4::LsxNvfp4Weight shared_up;
    lsx_nvfp4::LsxNvfp4Weight shared_down;
};

// All-resident weight set for one Mistral-Small-4-119B instance. The routed
// experts (56 GiB) are NOT resident: they live in a host pinned store and are
// streamed through a device slot LRU.
struct MistralPinnedWeights {
    std::string model_path;
    int num_layers = 0;
    int num_experts = 0;
    // Geometry carried from the converted index (so a synthetic fixture steps
    // at its own widths, not the frozen ones).
    int hidden = 0;
    int heads = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int v_head = 0;
    int moe_inter = 0;
    int top_k = 0;
    MistralBf16Weight embed;   // BF16 [vocab, H]
    MistralBf16Weight head;    // BF16 [vocab, H]
    const __nv_bfloat16 *final_norm = nullptr;  // BF16 [H]
    std::vector<MistralLayerWeights> layers;
    // Host-resident routed experts (neutral store): streamed via
    // EnsureMoEExpertResident under LSX_MISTRAL_EXPERT_CACHE_SLOTS slots.
    lsxcommon::moe::MoEExpertStore expert_store;
    std::size_t resident_bytes = 0;

    void Free();
};

// Loads the converted model rooted at <model_path>/logestix into device memory.
// Strict: returns a CUDA error code and fills `error` on any missing or
// malformed artifact, and refuses a converted index whose geometry disagrees
// with the frozen passport unless SyntheticGeometryAllowed() is set.
cudaError_t LoadMistralWeights(const std::string &model_path,
                               MistralPinnedWeights &pw, cudaStream_t stream,
                               std::string &error);

void FreeMistralWeights(MistralPinnedWeights &pw);

} // namespace mistral_moe
} // namespace lsxcommon
