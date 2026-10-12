#pragma once

// Mistral-Small-4-119B decode context (Model Ingestion Protocol stage 5).
//
// One token per call. Per-layer arithmetic mirrors the Mistral Small 4 text
// decoder exactly:
//   input_norm -> MLA(q_lora 1024 / kv_lora 256 / qk_nope 64 / qk_rope 64 /
//   v_head 128, split-half YaRN RoPE, Llama-4 logit temperature)
//     -> o_proj -> residual
//   post_norm -> 128-expert top-4 softmax MoE + 1 UNGATED shared expert
//     -> residual
// then final norm -> lm_head.
//
// Attention projections, norms, router, embeddings and head are BF16 and go
// through the model-neutral GEMV (lsx_gemv); the routed MoE goes through the
// neutral lsx_moe_inference store; the shared expert through the neutral NVFP4
// GEMM. The family owns only the MLA/RoPE/attention kernels.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/lsx_moe_inference.h"
#include "lsxcommon/mistral_moe_load.h"

namespace lsxcommon {
namespace mistral_moe {

// Decode geometry, taken from the LOADED weights (production always passes the
// frozen values; synthetic fixtures pass a shrunken copy).
struct MistralDims {
    int hidden = 0;
    int vocab = 0;
    int layers = 0;
    int experts = 0;
    int top_k = 0;
    int heads = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int qk_head = 0;   // qk_nope + qk_rope
    int v_head = 0;
    int moe_inter = 0;
    float attn_scale = 0.0f;
    float llama4_beta = 0.0f;
    double llama4_orig = 0.0;
    float rms_eps = 0.0f;
};

MistralDims MistralDimsFromWeights(const MistralPinnedWeights &pw);

// Per-request decode state: a per-layer KV cache ([layers, capacity, heads,
// qk_head]/[..., v_head] F32, pos-major) plus the scratch the MLA chain and
// the MoE need. Created and destroyed INSIDE RunUnderResidencyLock.
struct MistralSession {
    int capacity = 0;
    int length = 0;
    int vocab = 0;
    // KV caches, layer-major: element (l, p, h, d) at
    // ((l * capacity + p) * heads + h) * dim + d.
    float *d_k = nullptr;  // [layers * capacity * heads * qk_head]
    float *d_v = nullptr;  // [layers * capacity * heads * v_head]
    // Cache layout via the shared geometry (the guard forbids hand-rolled
    // `* heads + h` strides). pos-major ships; a layout switch is a Make() call.
    lsxcommon::KvCacheGeometry kgeom{};
    lsxcommon::KvCacheGeometry vgeom{};
    // MLA scratch.
    // Residual hidden, FP32. This model carries massive activations: the layer-32
    // top expert drives the residual to ~2.4e5 (host reference), past the FP16
    // ceiling of 65504, so an FP16 residual silently saturates (or NaNs) there.
    float *d_h = nullptr;         // [hidden]
    // Attention activations are BF16 because the BF16 projections go through
    // the neutral GEMV, which only accepts same-dtype operands.
    __nv_bfloat16 *d_norm = nullptr;    // [hidden] normed input (BF16)
    float *d_q_lat = nullptr;           // [q_lora]
    __nv_bfloat16 *d_q_lat16 = nullptr; // [q_lora] normed (BF16)
    float *d_q = nullptr;               // [heads * qk_head]
    float *d_kv = nullptr;              // [kv_lora + qk_rope]
    __nv_bfloat16 *d_kv_lat16 = nullptr;  // [kv_lora] normed (BF16)
    float *d_kv_b = nullptr;     // [heads * (qk_nope + v_head)]
    float *d_attn = nullptr;     // [heads * v_head]
    __nv_bfloat16 *d_attn16 = nullptr;  // [heads * v_head] BF16
    float *d_attn_out = nullptr; // [hidden]
    __half *d_post = nullptr;    // [hidden] post-normed MoE input (FP16)
    // MoE scratch (neutral store handles the routed experts).
    float *d_moe_acc = nullptr;      // [hidden] F32 accumulator
    __half *d_shared = nullptr;
    float *d_shared_alpha = nullptr;   // device-side SwiGLU prescale
    float *d_swi_scratch = nullptr;   // [I+2] FP32 SwiGLU arena (routed+shared)      // [hidden] shared expert output
    __half *d_mlp_scratch = nullptr; // [2 * moe_inter]
    __half *d_yg = nullptr;          // [moe_inter]
    __half *d_yu = nullptr;
    __half *d_y2 = nullptr;          // [hidden]
    lsxcommon::moe::MoERoutedSlot *d_slots = nullptr;  // [top_k]
    lsxcommon::moe::MoERoutedSlot *h_slots = nullptr;  // [top_k] pinned
    float *d_scores = nullptr;       // [heads * capacity] attention scratch
    float *d_logits = nullptr;       // [vocab]
    lsxcommon::gpu::ExpertDeviceCache cache;

        // Session teardown must run on destruction: Infer() resets the unique_ptr
    // between requests, and without this the dense graph + expert cache leak
    // every call (the second Infer OOMs).
    ~MistralSession() { Free(); }
    void Free();
};

// Allocates the session for a `capacity`-token context. `slots` sizes the
// routed-expert device LRU.
cudaError_t AllocateMistralSession(MistralSession &s, const MistralDims &dims,
                                   int capacity, int slots, cudaStream_t stream);

void FreeMistralSession(MistralSession &s);

// Host-built YaRN cos/sin tables, [capacity][rope/2] each, unscaled
// (params.json yarn.apply_scale = false).
bool BuildMistralYarnTables(int capacity, std::vector<float> &cos_t,
                            std::vector<float> &sin_t);

cudaError_t UploadMistralYarnTables(cudaStream_t stream, int capacity,
                                    const std::vector<float> &cos_t,
                                    const std::vector<float> &sin_t,
                                    float **dst_rc, float **dst_rs);

// Shared expert (ungated SwiGLU, NVFP4) over one token:
//   y = w2 . (silu(w1 . x) * (w3 . x))
// @p scratch holds 2 * w1.rows FP16 elements (gate then up).
//
// This MUST run the neutral NVFP4 seq==1 GEMV. Routing these NVFP4 views
// through the BF16 dense GEMV (attn::RunAttnMLP) reinterprets E2M1 nibbles
// as BF16 and under-contributes the branch ~44x; the decode step then drifts
// and NaNs in the deep layers. Exported so the numeric contract is unit-tested
// against a host NVFP4 reference (test_mistral_moe_kernels).
//
// The SwiGLU product is stored into the FP16 buffer under a power-of-two
// prescale (this model overflows FP16 there), so @p y carries a 1/alpha^2
// factor. When @p d_alpha_out is non-null it receives the device-side alpha
// the caller must fold into its FP32 accumulator; pass nullptr to consume the
// unscaled y (test/diagnostic use).
cudaError_t MistralSharedExpert(cudaStream_t stream,
                                const lsx_nvfp4::LsxNvfp4Weight &gate,
                                const lsx_nvfp4::LsxNvfp4Weight &up,
                                const lsx_nvfp4::LsxNvfp4Weight &down,
                                const __half *x, __half *y, __half *scratch,
                                float *d_alpha_out = nullptr,
                                float *d_swi_scratch = nullptr);

// One decode step. `token_id` selects the embedding row, `pos` is the absolute
// KV position. Writes the next-token logits into s.d_logits.
cudaError_t MistralDecodeStep(MistralPinnedWeights &pw,
                              const MistralDims &dims, MistralSession &s,
                              int token_id, const float *d_rc,
                              const float *d_rs, int pos, cudaStream_t stream);

// Reads back the last step's logits (host pointer, length = dims.vocab).
void MistralReadLogits(const MistralSession &s, float *host_logits,
                       cudaStream_t stream);

} // namespace mistral_moe
} // namespace lsxcommon
