#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace lsxcommon::qwen_moe {

// One routed slot: selected expert id + renormalized routing weight.
// Same layout contract as GlmRoutedSlot (8 bytes, one grid slot per token/k).
struct alignas(8) QwenMoeRoutedSlot {
    int32_t eid;
    float coeff;
};

// HF-parity MoE router (modeling_qwen3_5_moe.Qwen3_5MoeTopKRouter):
// fp32 logits = router_w[e] . x[t] -> softmax over ALL experts ->
// top-K probabilities (lower-index tie-break) -> renormalize by their sum.
//
// Launch config: <<<seq, 128, E * sizeof(float), stream>>>.
// Shared-memory RMW isolation: selection/writes happen on thread 0 only.
//
// Determinism: same input bits -> same output bits on repeated runs.
cudaError_t RunQwenMoeRouterTopK(cudaStream_t stream,
                                 const __nv_bfloat16 *router_w,  // [E,H]
                                 const __half *normed,           // [seq,H]
                                 int seq, int E, int K, int H,
                                 QwenMoeRoutedSlot *d_slots);    // [seq*K]

} // namespace lsxcommon::qwen_moe
