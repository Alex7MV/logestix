#pragma once

// MiMo-V2.6-Flash-RL model (kType "mimo-v2.6-flash-rl"). Hybrid SWA/full sparse
// MoE with asymmetric QK/V heads and a learned sink bias. Standalone ModelBase
// subclass; frozen geometry passport is compile-time (src/mimo_moe_constants.h)
// and Load cross-checks the converted index against it.
//
// Dense backbone resident on the GPU; 12032 routed experts stream through the
// host store + device slot-LRU (mimo_moe_expert_cache.h). All NVFP4 math goes
// through the neutral lsx_nvfp4 layer; Infer opens its own LsxFastMathScope.

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "lsxcommon/chat_conversation.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/mimo_moe_expert_cache.h"
#include "lsxcommon/mimo_moe_inference_attn.h"
#include "lsxcommon/mimo_moe_load.h"
#include "lsxcommon/model_base.h"

namespace lsxcommon {

struct MimoLinearF16 {
    __half *d_w = nullptr;  // [rows, cols] FP16
    int rows = 0;
    int cols = 0;
};

struct MimoLayerWeights {
    bool is_swa = false;
    bool is_moe = false;
    __half *d_input_norm = nullptr;
    __half *d_post_norm = nullptr;
    MimoLinearF16 qkv;            // [qkv_rows, hidden] FP16 (high precision)
    __half *d_o_proj = nullptr;   // [hidden, o_size] FP16
    __half *d_sink = nullptr;     // [heads] FP16 or nullptr
    __nv_bfloat16 *d_router_gate = nullptr;  // [num_experts, hidden] BF16
    float *d_router_bias = nullptr;          // [num_experts] F32
    MimoLinearF16 dense_gate, dense_up, dense_down;  // layer 0
};

class MiMoV26FlashModel final : public ModelBase {
public:
    static const std::string kType;  // "mimo-v2.6-flash-rl"

    explicit MiMoV26FlashModel(ModelInitConfig cfg);
    ~MiMoV26FlashModel() override;

    bool Load() override;
    gpu::InferenceResult Infer(const ModelRequest &req) override;
    gpu::InferenceResult SelfCheck(const ModelRequest &req) override;

    bool BuildPromptInputIds(const ArrowTokenizerDb &tok,
                             std::string_view raw_prompt,
                             std::vector<int32_t> &out_ids) const override;
    bool BuildConversationInputIds(const ArrowTokenizerDb &tok,
                                   const ChatConversation &conv,
                                   std::vector<int32_t> &out_ids) const override;
    bool ParseToolCalls(const ArrowTokenizerDb &tok,
                        const std::vector<int32_t> &gen_tokens,
                        std::vector<ToolCall> &out_calls) const override;

    const std::string &Type() const override { return kType; }

private:
    void TeardownWeights() override;
    bool UploadLayer(int layer);
    bool ForwardOne(int token, int pos, float *logits);
    void FreeLayer(MimoLayerWeights &l);

    mimo::MiMoModelFiles files_;
    mimo::MiMoTopology topo_;
    mimo::MimoExpertCache expert_cache_;

    __half *d_embed = nullptr;
    __half *d_head = nullptr;
    __half *d_final_norm = nullptr;
    std::vector<MimoLayerWeights> layers_;

    // Per-layer KV caches (roped K, value-scaled V).
    std::vector<__half *> kv_k_;
    std::vector<__half *> kv_v_;

    // Scratch device buffers (sized for the frozen geometry / synthetic topo).
    __half *d_x = nullptr;
    __half *d_xn = nullptr;
    __half *d_qkv = nullptr;
    __half *d_attn = nullptr;
    __half *d_oproj = nullptr;
    __half *d_gate = nullptr;
    __half *d_up = nullptr;
    __half *d_act = nullptr;
    __half *d_mlp = nullptr;
    __half *d_expert_out = nullptr;
    int *d_router_idx = nullptr;
    float *d_router_w = nullptr;
    float *d_logits = nullptr;

    int max_seq_ = 0;
    int o_size_ = 0;
    int kv_heads_ga_ = 4;
    int kv_heads_swa_ = 8;
};

} // namespace lsxcommon
