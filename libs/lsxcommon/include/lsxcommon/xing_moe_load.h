#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <string>
#include <vector>

namespace lsxcommon::xing {

// One NVFP4 weight resident on the device (E2M1 nibbles + per-16 E4M3 block
// scales + the per-tensor float global_scale produced by the converter).
struct XingNvfp4Weight {
    const uint8_t *packed = nullptr;
    const uint8_t *scales = nullptr;
    float global_scale = 1.0f;
    int64_t rows = 0;
    int64_t cols = 0;
    // Byte spans: the scales live INSIDE the packed allocation (the converter
    // writes one contiguous blob per record), so the embedding dequant needs
    // the lengths, not just the base pointer.
    int64_t packed_bytes = 0;
    int64_t scale_bytes = 0;
    bool valid() const { return packed != nullptr && scales != nullptr; }
};

// mHC block (Xing4_0HyperConnection) kept in BF16/F32 exactly as converted:
// hc_fn [24, 4H] projects the concatenated streams onto pre/post/Comb logits,
// hc_base [24] and hc_scale [3] are the per-block bias and the three output
// gains. There is no input-norm tensor (Xing4_0UnweightedRMSNorm).
struct XingHyperConn {
    const __nv_bfloat16 *hc_fn = nullptr;   // [24, 4H]
    int64_t hc_fn_rows = 0;
    int64_t hc_fn_cols = 0;
    const __nv_bfloat16 *hc_base = nullptr;  // [24]
    const float *hc_scale = nullptr;         // F32 [3]
    bool valid() const { return hc_fn != nullptr && hc_base != nullptr && hc_scale; }
};

// Per-layer resident weights. Everything the decode loop touches is uploaded
// once at Load: Xing's routed experts fit in VRAM in full (64 x 38 x 3 x 1024 x
// 3584 at NVFP4 is ~15 GiB), so there is no expert cache, no slot LRU and no
// per-token H2D staging — routing only computes offsets.
struct XingLayerWeights {
    // MLA (all NVFP4 except the two inner norms).
    XingNvfp4Weight q_a;      // [q_lora, H]
    XingNvfp4Weight q_b;      // [heads*qk, q_lora]
    XingNvfp4Weight kv_a;     // [kv_lora+rope, H]
    XingNvfp4Weight kv_b;     // [heads*(nope+v), kv_lora]
    XingNvfp4Weight o_proj;   // [H, heads*v]
    // Norms stay BF16 and are typed as such: a norm that silently reads its
    // weight as bytes is a plausible-looking garbage stream, not a crash.
    const __nv_bfloat16 *q_a_ln = nullptr;    // [q_lora]
    const __nv_bfloat16 *kv_a_ln = nullptr;   // [kv_lora]
    const __nv_bfloat16 *input_ln = nullptr;      // [H]
    const __nv_bfloat16 *post_attn_ln = nullptr;  // [H]
    // mHC: two blocks per layer, residual mixing.
    XingHyperConn attn_hc;
    XingHyperConn ffn_hc;
    // Router (noaux_tc). The weight is a [E, H] map whose ARGMAX selects the
    // experts, so it stays BF16 (never quantized); the correction bias is F32.
    const __nv_bfloat16 *router = nullptr;  // BF16 [E, H]
    int64_t router_size = 0;          // bytes
    const float *e_score_bias = nullptr;  // F32 [E]
    // Dense SwiGLU (layers [0, kFirstKDense)).
    XingNvfp4Weight dense_gate;  // [dense_inter, H]
    XingNvfp4Weight dense_up;    // [dense_inter, H]
    XingNvfp4Weight dense_down;  // [H, dense_inter]
    // Shared expert (every MoE layer; one expert of shared width).
    XingNvfp4Weight shared_gate;
    XingNvfp4Weight shared_up;
    XingNvfp4Weight shared_down;
    bool is_moe = false;
};

// Byte offsets of one expert's payloads inside its layer slab. Packed nibbles
// of w1/w2/w3 come first (each 16-B aligned), then their E4M3 block scales.
struct alignas(16) XingExpertRecord {
    int64_t w1_off = 0;
    int64_t w2_off = 0;
    int64_t w3_off = 0;
    int64_t s1_off = 0;
    int64_t s2_off = 0;
    int64_t s3_off = 0;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t s1_bytes = 0;
    int64_t s2_bytes = 0;
    int64_t s3_bytes = 0;
    float gs1 = 1.0f;
    float gs2 = 1.0f;
    float gs3 = 1.0f;
};

// All-resident weight set for one Xing4.0-29B-A4B instance.
struct XingPinnedWeights {
    std::string model_path;
    int num_layers = 0;
    int num_experts = 0;
    int first_moe_layer = 0;
    XingNvfp4Weight embed;
    XingNvfp4Weight head;
    const __nv_bfloat16 *final_norm = nullptr;  // BF16 [H]
    std::vector<XingLayerWeights> layers;
    // Routed experts, per layer: one device slab (packed + scales) plus an
    // offset table. slab_scale_bytes[i] mirrors slab_bytes[i] for the E4M3 part.
    std::vector<uint8_t *> expert_slabs;
    std::vector<size_t> slab_bytes;
    std::vector<std::vector<XingExpertRecord>> expert_records;
    // HOST pointer tables, one row per layer [2 * num_experts] (gate/up rows in
    // the first half, down rows in the second). The model-neutral expert-batch
    // GEMV takes its pointer arrays BY VALUE on the host and ships them to the
    // kernel in a by-value struct, so a device-resident table cannot be used
    // here - the decode loop copies the 4 ids back and indexes these.
    std::vector<std::vector<const uint8_t *>> expert_packed_ptrs;
    std::vector<std::vector<const uint8_t *>> expert_scale_ptrs;
    std::vector<std::vector<float>> expert_gs;  // [2 * num_experts]
    // Per-layer raw scratch (hc / norms) counted into resident_bytes.
    std::size_t resident_bytes = 0;

    void Free();
};

// Loads the converted model rooted at <model_path>/logestix into device memory.
// Strict: returns a CUDA error code and fills `error` on any missing or
// malformed artifact, and refuses a converted index whose geometry disagrees
// with the frozen passport unless SyntheticGeometryAllowed() (LSX_TEST_ALLOW_
// SYNTHETIC_GEOMETRY=1) is set.
cudaError_t LoadXingWeights(const std::string &model_path,
                            XingPinnedWeights &pw, cudaStream_t stream,
                            std::string &error);

void FreeXingWeights(XingPinnedWeights &pw);

} // namespace lsxcommon::xing