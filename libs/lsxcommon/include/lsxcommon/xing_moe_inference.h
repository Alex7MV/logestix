#pragma once

// Xing4.0-29B-A4B decode context (Model Ingestion Protocol stage 5).
//
// One token per call. Per-layer arithmetic mirrors modeling_xing4_0.py exactly:
//   attn_hc -> input_layernorm -> MLA(q_lora/kv_lora, interleaved YaRN RoPE)
//           -> o_proj -> mix(post, comb)
//   ffn_hc  -> post_attention_layernorm -> dense SwiGLU | MoE(64 experts top-4
//           noaux_tc + shared expert) -> mix(post, comb)
// then mean over the four streams -> final norm -> lm_head.
//
// Everything NVFP4 goes through the model-neutral lsx_nvfp4 layer; the family
// owns only the MLA/RoPE/attention kernels and the mHC block.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/xing_moe_kernels.h"
#include "lsxcommon/xing_moe_load.h"

namespace lsxcommon::xing {

// Per-request decode state. The KV cache holds the FULL (un-absorbed) MHA
// latents — Xing4_0Attention materialises k_pass/value instead of absorbing
// kv_b_proj into q/o, so this is [heads, nope+rope] K and [heads, v] V per
// token, 40 KiB/token/layer (3.1 GiB at a 2048-token context).
struct XingSession {
    int capacity = 0;
    int length = 0;          // tokens already in the cache
    float *d_k = nullptr;    // [capacity, heads, nope+rope]
    float *d_v = nullptr;    // [capacity, heads, v]
    // Four-stream hidden state [hc_mult * hidden] and the mHC outputs.
    float *d_hidden = nullptr;    // [4 * hidden]
    float *d_collapsed = nullptr;  // [hidden]
    float *d_post = nullptr;       // [hc_mult]
    float *d_comb = nullptr;       // [hc_mult * hc_mult]
    // F32 scratch: layer norm outputs, attention scores, logits.
    float *d_norm = nullptr;    // [max(hidden, q_lora, kv_lora)]
    float *d_q = nullptr;       // [heads * qk]
    float *d_kv_latent = nullptr;  // [kv_lora + rope]
    float *d_kv_b = nullptr;    // [heads * (nope + v)]
    float *d_attn = nullptr;    // [heads * v]
    float *d_scores = nullptr;  // [capacity]
    float *d_logits = nullptr;  // [vocab]
    float *d_gate = nullptr;    // [16] router weights
    int32_t *d_expert_ids = nullptr;  // [16]
    __half *d_half = nullptr;   // [2 * dense_intermediate] GEMV staging
    __half *d_half_big = nullptr;  // [vocab] head output
    // Expert pointer tables (all experts resident -> no staging copies).
    const uint8_t **d_epacked = nullptr;  // [4] selected
    const uint8_t **d_escales = nullptr;  // [4]
    float *d_egs = nullptr;                // [4]

    void Free();
};

// Allocates every scratch buffer for a `capacity`-token context.
cudaError_t AllocateXingSession(XingSession &s, int capacity, int vocab,
                                int dense_intermediate, cudaStream_t stream);

void FreeXingSession(XingSession &s);

// Host-built YaRN cos/sin tables, [capacity][rope_dim/2] each, already
// multiplied by the upstream attention factor (mscale) — see the constants
// header for why the factor appears twice.
bool BuildXingYarnTables(int capacity, std::vector<float> &cos_t,
                         std::vector<float> &sin_t);

// One decode step at absolute position `pos`. `x_emb` is the embedding row
// (F32[hidden]); `logits_out` receives F32[vocab]. Returns a CUDA error code.
cudaError_t XingDecodeStep(const XingPinnedWeights &pw, XingSession &s,
                           const float *d_rc, const float *d_rs, int pos,
                           cudaStream_t stream);

// Reads back the last step's logits (host pointer, length = pw vocab).
void XingReadLogits(const XingPinnedWeights &pw, const XingSession &s,
                    float *host_logits, cudaStream_t stream);

}  // namespace lsxcommon::xing