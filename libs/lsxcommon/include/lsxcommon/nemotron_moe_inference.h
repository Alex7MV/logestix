#pragma once

// Nemotron-3.5 device decode session: persistent Mamba-2 recurrent state, KV
// cache and activation scratch. One session per generation run.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <string>

#include "lsxcommon/nemotron_moe_load.h"

namespace lsxcommon::nemotron_moe {

struct NmExpertCache;  // nemotron_moe_expert_cache.h

struct NmSession {
    int H = 0, num_layers = 0, num_experts = 0, topk = 0, d_inner = 0;
    int conv_dim = 0, heads = 0, head_dim = 0, ssm_groups = 0,
        ssm_state_size = 0, kv_heads = 0, attn_head_dim = 0, max_seq = 0;
    int vram_scratch = 0;

    // Recurrent state (one entry per mamba layer).
    std::vector<float *> mamba_state;   // [heads*head_dim*ssm_state_size]
    std::vector<float *> conv_state;    // [conv_dim*(K-1)]
    std::vector<int> mamba_slot;         // layer -> mamba index (-1 otherwise)
    std::vector<int> attn_slot;          // layer -> attention index (-1)
    // KV cache (one entry per attention layer).
    std::vector<float *> k_cache;        // [kv_heads*max_seq*attn_head_dim]
    std::vector<float *> v_cache;
    int attn_layers = 0;
    int mamba_layers = 0;

    // Scratch.
    __half *h = nullptr;
    __half *normed = nullptr;
    __half *sub = nullptr;
    float *gate = nullptr;
    float *xbc = nullptr;
    float *dt = nullptr;
    float *ssm_out = nullptr;
    float *gn_out = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *attn = nullptr;
    float *router_logits = nullptr;
    float *topk_w = nullptr;
    int *topk_i = nullptr;
    float *expert_acc = nullptr;
    float *head_logits = nullptr;
    __half *proj16 = nullptr;  // NVFP4 GEMV output scratch [in_proj]
    __half *out16 = nullptr;   // NVFP4 GEMV output scratch [H]

    void Free();
};

// Allocates the session buffers for `idx`. `max_seq` bounds the KV cache.
bool AllocNemotronSession(const NemotronIndex &idx, int max_seq,
                          NmSession &s, std::string &error);

void FreeNemotronSession(NmSession &s);

// Zeroes the recurrent + KV state (start of a generation run).
void ResetNemotronSession(NmSession &s, cudaStream_t stream);

// Runs ONE token through all layers; writes [vocab] F32 logits into
// s.head_logits. `pos` is the token's position in the sequence. `prefill` is
// true while consuming the prompt (chunk-scan dt clamp) and false for greedy
// decode steps (no dt clamp), matching the reference. Routed experts are made
// resident in `experts` on demand (device slot LRU).
cudaError_t NemotronForwardToken(const NemotronIndex &idx, const NmWeights &w,
                                 NmSession &s, NmExpertCache &experts, int token,
                                 int pos, bool prefill, cudaStream_t stream);

} // namespace lsxcommon::nemotron_moe
