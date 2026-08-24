#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

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

// Forward decls from the loader side (pinned store + LRU device cache).
struct QwenMoeExpertStore;
struct QwenMoeExpertDeviceCache;

// Processes one token's K routed slots (ascending-k deterministic fp32
// accumulation): d_acc += coeff_k * FFN_eidk(x). Stages experts H2D on cache
// misses via EnsureQwenMoeExpertResident. Scratch: d_yg/d_yu [I], d_y2 [H].
cudaError_t RunQwenMoeRoutedFFN(cudaStream_t stream,
                                QwenMoeExpertStore &store,
                                QwenMoeExpertDeviceCache &cache, int layer,
                                const QwenMoeRoutedSlot *h_slots, int K,
                                const __half *d_x, int H, int I,
                                float *d_acc, __half *d_yg, __half *d_yu,
                                __half *d_y2, std::string &error);

// Casts the routed-accumulation buffer to FP16.
cudaError_t RunQwenMoeAccToFp16(cudaStream_t stream, const float *d_acc,
                                __half *d_out, int n);

} // namespace lsxcommon::qwen_moe
