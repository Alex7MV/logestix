#pragma once

// Mistral-Large-4-1T-A52B decode pipeline (Model Ingestion Protocol stage 5).
// One token per call, correctness first: 72 layer steps over the neutral NVFP4
// GEMV / routed-expert FFN, the family MLA attention module, the FP32 router,
// a gated shared expert, the dense layer-0 MLP and the final logit softcap.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/mistral_moe_large4_inference_attn.h"
#include "lsxcommon/mistral_moe_large4_load.h"

namespace lsxcommon {
namespace mistral_large4 {

struct Large4Session {
    int capacity = 0;
    int length = 0;
    int vocab = 0;
    int32_t last_tok = -1;
    Large4AttnState attn;
    float *d_h = nullptr;            // [hidden] F32 residual
    __nv_bfloat16 *d_norm = nullptr; // [hidden]
    __half *d_norm16 = nullptr;      // [hidden]
    float *d_norm32 = nullptr;       // [hidden]
    float *d_attn_out = nullptr;     // [hidden]
    float *d_moe_acc = nullptr;      // [hidden]
    __half *d_mlp_g = nullptr;       // [max(I, dense_I)]
    __half *d_mlp_u = nullptr;       // [max(I, dense_I)]
    __half *d_mlp_y = nullptr;       // [hidden]
    float *d_sw_f32 = nullptr;       // [max(I, dense_I)]
    float *d_alpha = nullptr;        // [2]
    __half *d_yg = nullptr;          // [2 * I]
    __half *d_yu = nullptr;          // [I]
    __half *d_y2 = nullptr;          // [hidden]
    float *d_logits = nullptr;       // [vocab]
    float *d_embed_row = nullptr;    // [hidden]
    moe::MoERoutedSlot *d_slots = nullptr;
    moe::MoERoutedSlot *h_slots = nullptr;
    lsxcommon::gpu::ExpertDeviceCache cache;

    ~Large4Session() { Free(); }
    void Free();
};

cudaError_t AllocateLarge4Session(Large4Session &s, const Large4Dims &dims,
                                  const std::vector<Large4LayerWeights> &layers,
                                  int capacity, int slots,
                                  const std::vector<float> &cos_t,
                                  const std::vector<float> &sin_t,
                                  cudaStream_t stream);

void FreeLarge4Session(Large4Session &s);

// One decode step: embeds `token_id`, runs the 72 layers at absolute `pos`
// and writes the next-token logits (already soft-capped) into s.d_logits.
cudaError_t Large4DecodeStep(Large4PinnedWeights &pw, Large4Session &s,
                             int token_id, int pos, cudaStream_t stream);

void Large4ReadLogits(const Large4Session &s, float *host_logits,
                      cudaStream_t stream);

} // namespace mistral_large4
} // namespace lsxcommon
