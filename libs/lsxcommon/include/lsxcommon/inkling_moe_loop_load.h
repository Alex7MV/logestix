#pragma once

// Inkling-2-Mini loader. The converted tree keeps the release's precision:
//   * `pinned/global.arrow`        embed/unembed (BF16->FP16 on upload), norms,
//                                  AttnRes queries (F32), merged message_queries
//                                  [loop_layers, 4, hidden] F32
//   * `pinned/layer_<l>.arrow`     dense projections (BF16->FP16), norms,
//                                  L2-clip scales, rel bank, sinks, conv taps,
//                                  router weight/bias, and (loop layers only)
//                                  the four MessageGate controller triples
//   * `experts/layer_<l>/exp_<e>.arrow`  routed experts (neutral ExpertStore;
//                                  NVFP4 layers 0..19, MXFP8 20..23)
//
// The dense plane is device-resident; the routed set is only ever an mmap view.
// The loop adds no weight planes: passes share the loop layers' weights, and
// only the KV/conv STATE is per-pass (owned by the session, not the loader).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "lsxcommon/expert_store.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

namespace lsxcommon::inkling {

// ── runtime topology (cross-checked against the frozen passport) ───────────
struct InklingLoopTopology {
    // Schema layout generation declared by model.index.json
    // (lsxcommon::kIndexLayout). 0 = absent.
    int64_t layout = 0;
    std::string source_model_type;
    int num_layers = 0;
    int hidden = 0;
    int vocab = 0;
    int unpadded_vocab = 0;
    int num_heads = 0;
    int num_kv_heads = 0;
    int head_dim = 0;
    int d_rel = 0;
    int rel_extent = 0;
    int sliding_window = 0;
    int conv_kernel = 0;
    int attn_res_block = 0;
    int num_experts = 0;
    int top_k = 0;
    int moe_intermediate = 0;
    int shared_experts = 0;
    bool moe_renormalize = true;
    double route_scale = 1.0;
    double attn_scale_denom = 128.0;
    double rms_eps = 1e-6;
    double qk_l2_clip = 0.0;
    double rel_l2_clip = 0.0;
    int log_scaling_n_floor = 0;
    double log_scaling_alpha = 0.0;
    double mlp_gate_softcap = 0.0;
    double mlp_up_softcap = 0.0;
    double final_logit_softcap = 0.0;
    // the loop core
    int loop_start = 0;
    int loop_end = 0;
    // The count this run actually executes (the checkpoint's declared default,
    // or the operator's override inside `loop_passes_max`).
    int loop_passes = 1;
    // The trained ceiling recorded by the converter. A fixed-pass tree records
    // its own declared count here, so the slab arithmetic has ONE shape.
    int loop_passes_max = 0;
    // `loop_config.is_varloop`: the pass count may be chosen at inference time.
    bool varloop = false;
    int loop_gate_rank = 0;
    int bos_id = -1;
    int eos_id = -1;
    int stop_id_alt = -1;
    int pad_id = -1;
    int tokenizer_kind = 0;
    int skipped_visual = 0;
    std::vector<int> full_layers;
    std::vector<int> sliding_layers;
    std::vector<bool> expert_nvfp4;
    std::vector<bool> expert_fp8;
    std::string error_msg;

    bool IsFull(int layer) const
    {
        for (int l : full_layers)
            if (l == layer) return true;
        return false;
    }
    bool IsLoop(int layer) const
    {
        return layer >= loop_start && layer < loop_end;
    }
    int LoopLayers() const { return loop_end - loop_start; }
    // Layer APPLICATIONS per token under the ACTIVE pass count: what the trace
    // records and what a decode really executes.
    int LayerApplications() const
    {
        return num_layers + (loop_passes - 1) * LoopLayers();
    }
    // KV + conv STATE SLOTS. Sized for the MAXIMUM pass count the artifact can
    // run (equal to LayerApplications() for a fixed-pass tree), because the slab
    // must not be re-allocated when the count changes between requests. The
    // active count addresses a prefix: slot(l, p) is num_layers + (p-1)*L + ...
    int CacheLayers() const
    {
        return num_layers +
               (std::max(loop_passes, loop_passes_max) - 1) * LoopLayers();
    }
    // Cache slot of (physical layer, pass); pass 0 uses the physical slot.
    int CacheSlot(int layer, int pass) const
    {
        return pass == 0 ? layer
                         : num_layers + (pass - 1) * LoopLayers() +
                               (layer - loop_start);
    }
    bool ExpertIsFp8(int layer) const
    {
        return layer >= 0 && layer < static_cast<int>(expert_fp8.size()) &&
               expert_fp8[static_cast<size_t>(layer)];
    }
    bool IsBlockBoundary(int layer) const
    {
        return attn_res_block > 0 && layer % attn_res_block == 0;
    }
};

// ── routed-expert store entry (field order matches ExpertStore::DecodeEntry) ─
struct InklingLoopExpertEntry {
    const uint8_t *host_w1 = nullptr;  // gate [I, H] packed
    const uint8_t *host_w2 = nullptr;  // down [H, I] packed
    const uint8_t *host_w3 = nullptr;  // up   [I, H] packed
    const uint8_t *host_w1_scales = nullptr;
    const uint8_t *host_w2_scales = nullptr;
    const uint8_t *host_w3_scales = nullptr;
    float w1_global_scale = 1.0f;
    float w2_global_scale = 1.0f;
    float w3_global_scale = 1.0f;
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t w1_scales_bytes = 0;
    int64_t w2_scales_bytes = 0;
    int64_t w3_scales_bytes = 0;
};
static_assert(std::is_trivially_copyable_v<InklingLoopExpertEntry>,
              "ExpertStore entries must stay trivially copyable views");
static_assert(sizeof(InklingLoopExpertEntry) <= 128,
              "ExpertStore entry must stay a compact view");

using InklingLoopExpertDb = gpu::expert_store::ExpertStore<InklingLoopExpertEntry>;

// ── an expert's six planes inside one 256-byte-aligned block ────────────────
// ExpertStore::PinChunked<kStream> packs a pinned expert as
// [w1 | w3 | w2 | w1_scales | w3_scales | w2_scales], each plane starting on a
// 256-byte boundary, and re-points the entry at those offsets. The inference
// pipeline lays its device staging slab out identically, which is what lets it
// move an expert with ONE cudaMemcpyAsync instead of six. Both sides compute
// the geometry here so the contract has a single definition; the loader test
// pins it against the real pinned pointers.
struct InklingLoopExpertPlanes {
    int64_t w1 = 0;
    int64_t w3 = 0;
    int64_t w2 = 0;
    int64_t w1_scales = 0;
    int64_t w3_scales = 0;
    int64_t w2_scales = 0;
    int64_t block_bytes = 0;
};

// `inter` is the MoE intermediate width (I), `hidden` the model width (H).
// `mxfp8` selects one byte per element (E4M3) instead of packed NVFP4 nibbles;
// the per-16 scale plane is the same shape either way.
InklingLoopExpertPlanes InklingLoopExpertPlaneOffsets(int inter, int hidden,
                                                     bool mxfp8);

// ── device-resident routed experts ─────────────────────────────────────────
// Mini's whole routed set is 14.7 GiB and the dense plane is 3.2 GiB, so both
// fit on a 32 GiB card with room for the KV cache. Keeping the experts in VRAM
// turns the per-token expert traffic from PCIe H2D into HBM reads: the measured
// profile before this cache (nsys 2026-11-22) showed 1.79 s of H2D inside a
// 5.3 s span, i.e. the copies were a third of the wall, because 48 layer
// applications x top_k 8 experts is 384 expert fetches per token.
//
// When the card cannot hold them the cache stays disabled and the staging path
// falls back to host-memory copies, so this is a residency decision, not a
// correctness one.
struct InklingLoopDeviceExperts {
    uint8_t *base = nullptr;
    // Per-LAYER block geometry: the NVFP4 layers pack two elements per byte and
    // the MXFP8 layers one, so the two classes have different block sizes.
    std::vector<int64_t> block_bytes;   // [num_layers]
    std::vector<int64_t> layer_offset;  // [num_layers], base-relative
    std::vector<int32_t> layer_experts; // [num_layers], experts per layer
    int64_t total_bytes = 0;
    size_t experts = 0;                 // resident expert blocks in total

    bool Enabled() const { return base != nullptr; }
    // Byte offset of one expert's block inside `base`, or -1 when out of range.
    int64_t ExpertOffset(int layer, int expert) const
    {
        if (layer < 0 || layer >= static_cast<int>(block_bytes.size())) return -1;
        if (expert < 0 || expert >= layer_experts[static_cast<size_t>(layer)])
            return -1;
        return layer_offset[static_cast<size_t>(layer)] +
               block_bytes[static_cast<size_t>(layer)] *
                   static_cast<int64_t>(expert);
    }
    void Free()
    {
        if (base) cudaFree(base);
        base = nullptr;
        total_bytes = 0;
        experts = 0;
        block_bytes.clear();
        layer_offset.clear();
        layer_experts.clear();
    }
};

// Uploads every routed expert of every layer into one device slab. Returns an
// empty (disabled) cache when the allocation does not fit, so the caller can
// keep streaming instead of failing the load.
InklingLoopDeviceExperts LoadInklingLoopDeviceExperts(
    InklingLoopExpertDb &db, const InklingLoopTopology &top, int64_t inter,
    int64_t hidden, cudaStream_t stream);

// ── pinned weights (device-resident for the whole session) ──────────────────
// One FP16 dense projection: [rows, cols], row-major.
struct InklingLoopDenseF16 {
    __half *d = nullptr;
    int rows = 0;
    int cols = 0;
};

// One MessageGate controller: down [rank, 2*hidden] FP16, forget/gain
// [hidden, rank] FP32 (the reference computes the gates in FP32).
struct InklingLoopMessageGate {
    InklingLoopDenseF16 down;
    float *forget = nullptr;
    float *gain = nullptr;
};

struct InklingLoopLayerWeights {
    bool full = false;
    bool loop = false;
    // BF16 in the checkpoint, FP16 on device.
    InklingLoopDenseF16 wq, wk, wv, wo, wr, attn_gate;
    InklingLoopDenseF16 ksconv_proj, vsconv_proj;  // full layers only
    // Norms (FP16 weights, never quantized).
    __half *attn_norm = nullptr;    // sliding layers
    __half *q_in_norm = nullptr;    // full layers
    __half *k_in_norm = nullptr;
    __half *v_in_norm = nullptr;
    __half *mlp_norm = nullptr;
    // FP32 side tensors.
    __half *sink_proj = nullptr;    // [heads, hidden] FP16 (matvec)
    float *q_clip = nullptr;        // [heads, head_dim] learned scale
    float *rel_clip = nullptr;      // [heads, d_rel]   learned scale
    float *rel_proj = nullptr;      // [d_rel, rel_extent]
    float *kconv_w = nullptr;       // [kv_dim, k] conv1d taps (sliding)
    float *vconv_w = nullptr;
    float *kconv_static = nullptr;  // [k, kv_dim] lag-0-first taps (full)
    float *vconv_static = nullptr;
    float *attn_conv_w = nullptr;   // [hidden, k] conv1d taps
    float *mlp_conv_w = nullptr;
    float *router_w = nullptr;      // [experts, hidden] FP32
    float *router_bias = nullptr;   // [experts] FP32 selection bias
    // Loop controllers: q, k, v, mlp message gates (loop layers only).
    InklingLoopMessageGate msg[4];
    // Recurrent conv state, FOUR convolutions per layer (k/v + post-attn +
    // post-MLP): [k][kv_dim] x2 then [k][hidden] x2. PER-SEQUENCE state; the
    // session owns one copy per cache slot (the loader's copy is the pass-0
    // template and is zeroed at every request start).
    float *conv_state = nullptr;
    int64_t conv_state_bytes = 0;
};

struct InklingLoopPinnedWeights {
    InklingLoopTopology top;
    // embed/unembed are FP16 (the residual stream at full width).
    __half *embed = nullptr;           // [vocab, hidden] FP16
    __half *head = nullptr;            // [vocab, hidden] FP16 (unembed)
    int embed_rows = 0;
    int embed_cols = 0;
    __half *embed_norm = nullptr;
    __half *final_norm = nullptr;
    // AttnRes query banks, FP32: [layers][nq][hidden], nq = 3 on full layers.
    float *attn_queries = nullptr;     // [sum(nq_l)][hidden]
    int *attn_query_offset = nullptr;  // [layers] host array of row offsets
    float *mlp_queries = nullptr;      // [layers][hidden]
    float *out_query = nullptr;        // [hidden]
    // Merged cross-pass message queries, FP32 [loop_layers][4][hidden],
    // order [q, k, v, mlp] per loop layer.
    float *message_queries = nullptr;
    std::vector<InklingLoopLayerWeights> layers;
    // The routed experts, device-resident for the whole session when the card
    // can hold them (see InklingLoopDeviceExperts). Disabled = host streaming.
    InklingLoopDeviceExperts device_experts;
    std::uint64_t resident_bytes = 0;
};

// ── API ────────────────────────────────────────────────────────────────────
// Reads <model_path>/model.index.json. The first gate is the schema GENERATION
// (`layout`, design §2.4.3), which the index declares itself; the distinction
// between the fixed-pass and varloop rows is the checkpoint's own `varloop`
// flag, checked by the row that owns the pass policy — not by a per-model tag
// handed down from the caller. There was an `expected_format` parameter here
// for exactly that tag comparison; it survived as a string interpolated into an
// error message, saying "expected format <tag>" about a layout check.
arrow::Status LoadInklingLoopTopology(const std::string &model_path,
                                      InklingLoopTopology &out);

// Cross-checks the loaded index against the frozen passport (relaxed under
// LSX_TEST_ALLOW_SYNTHETIC_GEOMETRY). The pass count is checked SEPARATELY by
// the model row that owns the pass policy: `MatchesFrozenGeometry` holds the
// numbers every row of this architecture shares, `LoopCoreMatches` the range a
// row accepts.
bool InklingLoopTopologyMatchesFrozen(const InklingLoopTopology &top);
bool InklingLoopTopologyMatchesFrozenGeometry(const InklingLoopTopology &top);
bool InklingLoopLoopCoreMatches(const InklingLoopTopology &top, int passes_min,
                                int passes_max);

// Uploads the pinned plane. Returns false on any missing/short tensor (a
// partially loaded model is never left behind).
bool LoadInklingLoopPinnedWeights(const std::string &root,
                                  InklingLoopPinnedWeights &pw,
                                  cudaStream_t stream);

void FreeInklingLoopPinnedWeights(InklingLoopPinnedWeights &pw);

// The neutral routed-expert store (mmap + LRU, no host copy).
InklingLoopExpertDb LoadInklingLoopExpertIndex(const std::string &model_path,
                                               int num_layers,
                                               int experts_per_layer,
                                               size_t max_open,
                                               bool warm_on_load = false,
                                               bool pin_host_memory = true);
// The routed-expert window floor: top_k draws from every layer APPLICATION of
// the artifact's MAXIMUM pass count (the KV/conv slab's), so a varloop tree that
// changes its count between loads never falls out of the window. The window is
// retention (mapped mmap views capped by `max_open`), so a floor above this
// run's working set costs page-cache, not correctness.
size_t InklingLoopExpertWindowFloor(const InklingLoopTopology &top);
size_t InklingLoopResolveExpertWindow(const InklingLoopTopology &top,
                                      long requested);
void EnsureInklingLoopExpertResident(InklingLoopExpertDb &db, int layer,
                                     int expert, cudaStream_t stream);
void FreeInklingLoopExpertIndex(InklingLoopExpertDb &db);

}  // namespace lsxcommon::inkling