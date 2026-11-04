#pragma once

#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "lsxcommon/qwen_inference_load.h"
#include "lsxcommon/qwen_moe_expert_cache.h"
#include "lsxcommon/qwen_moe_exp_convert.h"
#include "lsxcommon/qwen_moe_load.h"  // moe::MoEExpertStore base + record types

namespace lsxcommon::qwen_moe_exp {

// Dequantized fp16 twin of a layer's hyper-connection weights (down/up +
// optional inject), built once at load. Eliminates the per-token
// dequantize-everything cost of the hc GatedResidual path.
struct ExpHcDequantCache {
    __half *down = nullptr;    // [lowrank, hc*H]
    __half *up = nullptr;      // [hc*H, lowrank]
    __half *inject = nullptr;  // [hc, hc*H]; null when absent (mixer)
    int64_t down_elems = 0, up_elems = 0, inject_elems = 0;

    void Free();
};

// Dequantizes the three hc weights once (on load). The mixer passes an
// empty inject_w: inject cache stays null. Strict on allocation failure.
cudaError_t BuildExpHcDequantCache(cudaStream_t stream,
                                   const lsx_nvfp4::LsxNvfp4Weight &down,
                                   const lsx_nvfp4::LsxNvfp4Weight &up,
                                   const lsx_nvfp4::LsxNvfp4Weight &inject_w,
                                   ExpHcDequantCache &out);

// Per-layer device weights: reused dense attention block + qwen4_exp extras
// (hyper-connection norms kept BF16 on device; hc down/up/inject NVFP4;
// router BF16 raw; shared expert NVFP4 trio + FP16 scalar gate; PLE
// projections on the PLE layer).
struct ExpLayerWeights {
    lsxcommon::attn::LayerWeights attn;  // norms + attention/GDN (no dense MLP)
    lsx_nvfp4::LsxNvfp4Weight indexer_qk;  // QSA: hidden -> (idx_h+idx_kv)*idx_hd

    // Hyper-connections (GatedResidual): group-RMSNorm weights BF16 raw,
    // low-rank projections NVFP4, block-inject NVFP4.
    uint8_t *d_attn_hc_norm = nullptr;  // [hc*H] BF16 bits
    uint8_t *d_mlp_hc_norm = nullptr;   // [hc*H] BF16 bits
    int64_t hc_norm_bytes = 0;
    lsx_nvfp4::LsxNvfp4Weight attn_hc_down;    // [lowrank, hc*H]
    lsx_nvfp4::LsxNvfp4Weight attn_hc_up;      // [hc*H, lowrank]
    lsx_nvfp4::LsxNvfp4Weight attn_hc_inject;  // [hc, hc*H]
    lsx_nvfp4::LsxNvfp4Weight mlp_hc_down;
    lsx_nvfp4::LsxNvfp4Weight mlp_hc_up;
    lsx_nvfp4::LsxNvfp4Weight mlp_hc_inject;
    // Pre-dequantized fp16 hc weights (built once at load; per-token hc
    // dequant eliminated). Null until BuildExpHcDequantCache runs.
    ExpHcDequantCache *attn_hc_cache = nullptr;
    ExpHcDequantCache *mlp_hc_cache = nullptr;

    // MoE.
    uint8_t *d_router = nullptr;  // raw BF16 bits [E, H]
    int64_t router_bytes = 0;
    lsx_nvfp4::LsxNvfp4Weight shared_w1;  // gate_proj [SI, H]
    lsx_nvfp4::LsxNvfp4Weight shared_w2;  // down_proj [H, SI]
    lsx_nvfp4::LsxNvfp4Weight shared_w3;  // up_proj    [SI, H]
    __half *d_shared_gate = nullptr;  // [H] fp16
    bool moe_loaded = false;

    // PLE (only on the PLE layer, otherwise null).
    lsx_nvfp4::LsxNvfp4Weight ple_key;    // [hc*H, ple_embed]
    lsx_nvfp4::LsxNvfp4Weight ple_value;  // [H, ple_embed]
    uint8_t *d_ple_norm_key = nullptr;    // [hc*H] BF16 bits
    uint8_t *d_ple_norm_query = nullptr;  // [hc*H] BF16 bits
    uint8_t *d_ple_norm_conv = nullptr;   // [hc*H] BF16 bits
    uint8_t *d_ple_conv_bf16 = nullptr;   // [hc*H, k] BF16 bits (raw file)
    int64_t ple_conv_bytes = 0;
    int ple_conv_state_len = 0;
    bool is_ple_layer = false;
};

// PLE n-gram hash meta: multipliers (bit-exact splitmix64) + per-head prime
// vocab sizes and offsets. Verified against checkpoint buffers at load.
struct ExpPleMeta {
    static constexpr int kMaxHeads = 32;
    int64_t multipliers[3] = {0, 0, 0};
    int64_t head_vocab_sizes[kMaxHeads] = {};
    int64_t head_offsets[kMaxHeads] = {};
    int num_heads = 0;
    int head_dim = 0;       // 160 on the real model
    int context_len = 0;    // ngram_size - 1
    int64_t padded_vocab = 0;
};

// Non-owning mmap view of one PLE n-gram shard.
struct ExpPleShardView {
    const uint8_t *data = nullptr;
    size_t bytes = 0;
    int shard = -1;
    std::shared_ptr<void> keepalive;  // keeps the mmap alive
};

struct ExpPinnedWeights {
    qwen::QwenTopology top;    // dense view for kernel reuse
    Qwen4ExpTopology exp_top;  // full topology incl. qwen4_exp fields
    lsx_nvfp4::LsxNvfp4Weight embed;
    lsx_nvfp4::LsxNvfp4Weight head;   // legacy NVFP4 view (unused for qwen38)
    // LM head as a persistent FP16 table ([V, H], from BF16 raw global.arrow)
    // for the FP16 head GEMV: NVFP4 logit noise exceeds the top-1/top-2
    // margin of this model.
    __half *d_head_f16 = nullptr;
    int64_t head_elems = 0;
    // Global hyper-connection mixer (GatedResidual, use_combine=false):
    // norm BF16 raw + down/up NVFP4; output [H] (mean over streams).
    uint8_t *d_mixer_norm = nullptr;  // [hc*H] BF16 bits
    int64_t mixer_norm_bytes = 0;
    lsx_nvfp4::LsxNvfp4Weight mixer_down;  // [lowrank, hc*H]
    lsx_nvfp4::LsxNvfp4Weight mixer_up;    // [hc*H, lowrank]
    ExpHcDequantCache *mixer_hc_cache = nullptr;  // pre-dequant mixer weights
    __half *d_final_norm = nullptr;
    int final_norm_size = 0;
    std::vector<ExpLayerWeights> layers;
    ExpPleMeta ple_meta;
    std::size_t resident_bytes = 0;

    void Free();
};

// Loads dense parts (globals, hc mixer, attention/GDN, hc, norms, router,
// shared expert, PLE projections) from <model_path>/logestix/. Routed experts
// are NOT resident; they go through ExpExpertStore separately. Strict: returns
// false and fills `error` on any missing/malformed input.
bool LoadQwen4ExpDenseWeights(const std::string &model_path,
                              ExpPinnedWeights &pw, cudaStream_t stream,
                              std::string &error);

void FreeQwen4ExpPinnedWeights(ExpPinnedWeights &pw);

// Host-resident routed-expert payloads (pinned slabs), laid out per layer.
// The generation context stages slices of these into the LRU device cache.
// Record layout is IDENTICAL to lsxcommon::moe::MoEExpertRecord (fused staging
// consumes both stores through the same kernels); aliasing the type makes
// the pairing structural instead of convention.
using ExpExpertRecord = moe::MoEExpertRecord;
using ExpExpertMetaDev = moe::MoEExpertMetaDev;

// Qwen3.8-Flash-Next expert store: EXTENDS lsxcommon::moe::MoEExpertStore so
// the shared routed-expert kernels (RunMoERoutedFFN, RunMoERouterTopK,
// moe::EnsureMoEExpertResident family) consume it directly — no
// reinterpret_cast pairing, no layout-drift bug class.
struct ExpExpertStore : public lsxcommon::moe::MoEExpertStore {
    // Single pinned arena backing every layer slab (one page-lock instead
    // of num_layers; per-slab cudaMallocHost cost ~660ms each on load).
    // All h_slabs entries point at arena_base; record offsets are global.
    uint8_t *arena_base = nullptr;
    size_t arena_bytes = 0;

    void Free();
};

// Loads every experts/layer_<l>/exp_<e>.arrow under
// <model_path>/logestix/ into pinned host memory. Strict errors.
bool LoadQwen4ExpExpertStore(const std::string &model_path,
                             ExpExpertStore &store, std::string &error);

// Allocates the LRU device cache sized to the largest expert payload.
bool AllocateQwen4ExpExpertCache(ExpExpertStore &store, int slots,
                                 gpu::ExpertDeviceCache &cache,
                                 std::string &error);
void FreeQwen4ExpExpertCache(gpu::ExpertDeviceCache &cache);

// Ensures expert `eid` of `layer` is resident in the device cache; stages its
// payload H2D from the pinned slab on miss. Returns slot index, -1 on error
// (error text goes to `error` when non-null). When `h2d_stream` is provided,
// miss staging is enqueued on that (copy) stream instead of `stream`; set
// *missed so the caller can fence the upload before consuming the slot.
int EnsureQwen4ExpExpertResident(ExpExpertStore &store, int layer, int eid,
                                 gpu::ExpertDeviceCache &cache,
                                 cudaStream_t stream, std::string &error,
                                 cudaStream_t h2d_stream = nullptr,
                                 bool *missed = nullptr);

// Memory-maps all ple/shard_<S>.arrow files (S in 0..split_parts-1) under
// <model_path>/logestix/. `shards` must have room for split_ngram_parts
// entries (read from pw.exp_top). Strict errors.
bool LoadQwen4ExpPleShards(const std::string &model_path, ExpPinnedWeights &pw,
                           ExpPleShardView *shards, std::string &error);

// Device bytes the qwen38 generation context allocates AFTER the expert
// cache: KV history (KvCache::Allocate) + the max_seq-scaled arena
// (RunExpGeneration carving, incl. the two max_seq x V fp32 logits buffers).
// Both grow linearly with the context, so the LRU sizing reserve must be a
// function of the supported context, not a fixed constant (the model-neutral
// 8 GiB reserve was tuned for models with ~150k vocab; qwen38's V=248320
// alone makes the arena ~4.5 GiB at a 2k context — the max-gen-2048 OOM).
std::size_t GenContextBytes(int max_seq);

// Model-aware LRU slot sizing: lifecycle::ExpertCacheSlots with a reserve of
// GenContextBytes(max_seq) + kPostCacheFloorBytes (embed dequant + TC fused
// builds + driver margin — the max_seq-independent post-cache floor).
//
// `already_resident_bytes` is the part of that footprint the process already
// holds: this family keeps the generation arena and the fp16 embed table in
// process-lifetime statics (RunExpGeneration's "single session per process"
// etalon), so after the first decode they are inside `free_bytes`'s
// complement. Charging them again through the reserve double-counts the
// context and starves the cache on every subsequent Load in the same process
// (measured: the second load of one perf target sized 5350 -> 2202 slots).
// Pass the resident bytes to refund them; 0 (the default) is a cold load.
int SizeExpertCacheSlots(std::size_t free_bytes, std::size_t per_expert_bytes,
                         int total_experts, int max_seq, int slots_pct,
                         std::size_t already_resident_bytes = 0);

// Bytes of the generation-context footprint (arena + fp16 embed table) that are
// CURRENTLY allocated by this process. 0 before the first decode. Feeds
// SizeExpertCacheSlots' refund argument; defined in the inference TU, which owns
// the buffers.
std::size_t GenContextResidentBytes();

} // namespace lsxcommon::qwen_moe_exp
