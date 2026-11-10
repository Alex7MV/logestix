#pragma once

// Nemotron-3-Ultra-550B device decode session: persistent Mamba-2 recurrent
// state, the 12 attention layers' KV cache, and activation scratch. One session
// per generation run.
//
// Sizing is driven entirely by the parsed index, never by the passport: the
// session for a synthetic fixture and the session for the 108-layer model use the
// same code path (that is what makes the tiny-geometry tests meaningful).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "lsxcommon/nemotron_moe_ultra_load.h"

namespace lsxcommon::nemotron_ultra {

struct UltraExpertCache;  // nemotron_moe_ultra_expert_cache.h

struct UltraSession {
    int H = 0, num_layers = 0, num_experts = 0, topk = 0, d_inner = 0;
    int conv_dim = 0, heads = 0, head_dim = 0, ssm_groups = 0, ssm_state_size = 0;
    int kv_heads = 0, attn_head_dim = 0, max_seq = 0;
    int moe_latent = 0;          // 0 => identity latent (experts at hidden)
    int moe_intermediate = 0;
    int shared_intermediate = 0;
    int n_group = 0, topk_group = 0;
    float routed_scale = 0.0f;

    // Recurrent state (one entry per mamba layer).
    std::vector<float *> mamba_state;  // [heads*head_dim*ssm_state_size]
    std::vector<float *> conv_state;   // [conv_dim*(K-1)]
    std::vector<int> mamba_slot;       // layer -> mamba index (-1 otherwise)
    std::vector<int> attn_slot;        // layer -> attention index (-1)
    // KV cache (one entry per attention layer).
    std::vector<float *> k_cache;      // [kv_heads*max_seq*attn_head_dim]
    std::vector<float *> v_cache;
    int attn_layers = 0;
    int mamba_layers = 0;

    // Scratch.
    __half *h = nullptr;               // residual stream [H]
    __half *normed = nullptr;          // RMSNorm output [H]
    __half *sub = nullptr;             // mixer output to add [H]
    float *gate = nullptr;             // mamba gate [d_inner]
    float *xbc = nullptr;              // conv input/output [conv_dim]
    float *dt = nullptr;               // [heads]
    float *ssm_out = nullptr;          // [d_inner]
    float *gn_out = nullptr;           // gated-norm output [d_inner]
    float *q = nullptr;                // [attn_heads*attn_head_dim]
    float *k = nullptr;                // [kv_heads*attn_head_dim]
    float *v = nullptr;
    float *attn = nullptr;             // attention output [attn_heads*attn_head_dim]
    float *router_logits = nullptr;    // [num_experts]
    float *topk_w = nullptr;           // [topk]
    int *topk_i = nullptr;             // [topk]
    float *moe_acc = nullptr;          // [max(H, moe_latent)]
    float *head_logits = nullptr;      // [vocab]
    // Latent-bridge input. MUST be its own buffer: the routed up-projection
    // writes topk*I halfs into proj16 while the GEMV is still READING the latent
    // vector, so aliasing the two lets the first chunk clobber its own input
    // (I = 5120 > latent = 2048). The 30B sibling cannot hit this because it has
    // no latent bridge: its expert input is `normed`, a different buffer.
    __half *latent16 = nullptr;        // [moe_latent]
    __half *proj16 = nullptr;          // NVFP4 GEMV output scratch
    __half *out16 = nullptr;           // NVFP4 GEMV output scratch

    void Free();
};

// Allocates the session buffers for `idx`. `max_seq` bounds the KV cache.
bool AllocUltraSession(const UltraIndex &idx, int max_seq, UltraSession &s,
                       std::string &error);

void FreeUltraSession(UltraSession &s);

// Zeroes the recurrent + KV state (start of a generation run).
void ResetUltraSession(UltraSession &s, cudaStream_t stream);

// Runs ONE token through all layers; writes [vocab] F32 logits into
// s.head_logits. `pos` is the token's position in the sequence. `prefill` is true
// while consuming the prompt (the reference clamps dt in the chunk-scan form) and
// false for greedy decode steps. Routed experts are made resident in `experts` on
// demand (device slot LRU).
cudaError_t UltraForwardToken(const UltraIndex &idx, const UltraWeights &w,
                              UltraSession &s, UltraExpertCache &experts, int token,
                              int pos, bool prefill, cudaStream_t stream);

} // namespace lsxcommon::nemotron_ultra
