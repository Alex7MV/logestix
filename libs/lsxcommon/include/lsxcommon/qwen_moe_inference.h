#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/qwen_inference.h"
#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/sampling.h"
#include "lsxcommon/qwen_moe_expert_cache.h"

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
struct QwenMoePinnedWeights;

// Processes one token's K routed slots (ascending-k deterministic fp32
// accumulation): d_acc += coeff_k * FFN_eidk(x). Stages experts H2D on cache
// misses via EnsureQwenMoeExpertResident. Scratch: d_yg/d_yu [I], d_y2 [H].
//
// Copy-stream mode: when `copy_stream` != nullptr, cache-miss H2D staging is
// enqueued there instead of `stream`; `ev_uploads` (K caller-owned disabled-
// timing events) fence each upload and `ev_front` must be recorded-or-null —
// the implementation records it on `stream` before staging so the copy stream
// never overwrites a slot still being read by previously enqueued compute.
// Numerics are unchanged (same kernels, same accumulation order).
cudaError_t RunQwenMoeRoutedFFN(cudaStream_t stream,
                                QwenMoeExpertStore &store,
                                QwenMoeExpertDeviceCache &cache, int layer,
                                const QwenMoeRoutedSlot *h_slots, int K,
                                const __half *d_x, int H, int I,
                                float *d_acc, __half *d_yg, __half *d_yu,
                                __half *d_y2, std::string &error,
                                cudaStream_t copy_stream = nullptr,
                                cudaEvent_t *ev_uploads = nullptr,
                                cudaEvent_t ev_front = nullptr,
                                const uint8_t *xq = nullptr,
                                const uint8_t *xqs = nullptr,
                                uint8_t *d_qpack_scratch = nullptr,
                                uint8_t *d_qscales_scratch = nullptr);

// Casts the routed-accumulation buffer to FP16.
cudaError_t RunQwenMoeAccToFp16(cudaStream_t stream, const float *d_acc,
                                __half *d_out, int n);

// MoE FFN epilogue (HF parity): out = routed_acc + sigmoid(gate.x) * shared.
// d_routed_acc: [seq,H] fp32; scratch_i [seq,SI], scratch_h [seq,H];
// d_gate_val [seq] fp32 workspace. sw1=gate_proj [SI,H], sw3=up_proj,
// sw2=down_proj [H,SI].
cudaError_t RunQwenMoeSharedCombine(
    cudaStream_t stream, const lsxcommon::qwen::QwenNvfp4Weight &sw1,
    const lsxcommon::qwen::QwenNvfp4Weight &sw2,
    const lsxcommon::qwen::QwenNvfp4Weight &sw3,
    const __half *d_shared_gate, const __half *d_x, int seq, int H, int SI,
    const float *d_routed_acc, __half *d_scratch_i, __half *d_scratch_h,
    float *d_gate_val, __half *d_out, const uint8_t *xq = nullptr,
    const uint8_t *xqs = nullptr);

// Persistent generation scratch owned by the model: the dequantized FP16
// embedding table (computed once, byte-identical to per-call dequant) and a
// capacity-grown workspace arena reused across Infer calls.
struct QwenMoeGenScratch {
    __half *d_embed = nullptr;      // [V*H]
    int64_t table_elems = 0;
    void *arena = nullptr;
    size_t arena_capacity = 0;
    int64_t dequant_runs = 0;       // diagnostics/tests
    int64_t arena_allocs = 0;       // diagnostics/tests
};

cudaError_t EnsureQwenMoeScratch(cudaStream_t stream,
                                 const QwenMoePinnedWeights &pw,
                                 QwenMoeGenScratch &scratch);
cudaError_t ReserveQwenMoeArena(QwenMoeGenScratch &scratch, size_t bytes);
void FreeQwenMoeGenScratch(QwenMoeGenScratch &scratch);

// Incremental single-stream generation driver (MoE FFN call site). Requires
// a KV cache session (kv != nullptr); mirrors RunQwenGeneration's per-token
// handshake and final-logits publication.
cudaError_t RunQwenMoeGeneration(
    cudaStream_t stream, const QwenMoePinnedWeights &pw,
    QwenMoeExpertStore &store, QwenMoeExpertDeviceCache &cache,
    QwenMoeGenScratch &scratch,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    lsxcommon::qwen::QwenGenerationResult &result,
    lsxcommon::sample::ISampler &sampler, lsxcommon::qwen::QwenKvCache *kv);

} // namespace lsxcommon::qwen_moe
