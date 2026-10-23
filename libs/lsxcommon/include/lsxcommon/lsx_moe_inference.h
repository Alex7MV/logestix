#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/expert_device_cache.h"
#include "lsxcommon/expert_store.h"
#include "lsxcommon/lsx_attn_inference.h"

namespace lsxcommon::moe {

// One routed slot: selected expert id + renormalized routing weight.
// Same layout contract as GlmRoutedSlot (8 bytes, one grid slot per token/k).
struct alignas(8) MoERoutedSlot {
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
cudaError_t RunMoERouterTopK(cudaStream_t stream,
                                 const __nv_bfloat16 *router_w,  // [E,H]
                                 const __half *normed,           // [seq,H]
                                 int seq, int E, int K, int H,
                                 MoERoutedSlot *d_slots);    // [seq*K]

// Forward decls from the loader side (pinned store + LRU device cache).
// Upper bound on the routed top-k; the generation loop sizes its router
// scratch from it, so it is shared state rather than TU-local.
inline constexpr int kMaxTopK = 16;

// Host-resident routed-expert payloads (pinned slabs), laid out per layer.
// The generation context stages slices of these into the LRU device cache.
// Per-expert GEOMETRY (immutable metadata), not a pointer into host memory.
// The payload lives in MoEExpertDb and is mapped on demand, so an entry here
// must stay valid for the whole session even after the payload has been
// retired by the LRU — the device-slot sizing loops read these numbers
// independently of residency.
struct MoEExpertRecord {
    int64_t w1_bytes = 0;
    int64_t w2_bytes = 0;
    int64_t w3_bytes = 0;
    int64_t s1_bytes = 0;
    int64_t s2_bytes = 0;
    int64_t s3_bytes = 0;
    float gs1 = 1.f, gs2 = 1.f, gs3 = 1.f;

    // Byte offset inside a PINNED ARENA (the pre-ExpertStore storage, still
    // used only by glm-5.3-flash — see docs/OPEN_TASKS.md §V). Zero for an
    // arena-free store, whose payload is reached through MoEExpertDb instead.
    // The slab_* prefix is deliberate: reading these on a view-backed store
    // would be a stale-pointer bug, so the name has to say "arena" on sight.
    int64_t slab_w1_off = 0;
    int64_t slab_w2_off = 0;
    int64_t slab_w3_off = 0;
    int64_t slab_s1_off = 0;
    int64_t slab_s2_off = 0;
    int64_t slab_s3_off = 0;
};

// View into one expert's mmap, in the field order the shared
// ExpertStore<TEntry>::DecodeEntry() aggregate-fills. Never owns bytes.
struct MoEExpertEntry {
    const uint8_t *host_w1 = nullptr;
    const uint8_t *host_w2 = nullptr;
    const uint8_t *host_w3 = nullptr;
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

// The neutral zero-copy host store. max_open bounds how many experts stay
// mapped at once, so host memory is a function of the cap, not of the model.
using MoEExpertDb = gpu::expert_store::ExpertStore<MoEExpertEntry>;

// Cap on simultaneously mapped experts for the shared MoE loader. Sized so a
// routing burst stays resident without pinning the whole expert set: qwen MoE
// and ornith are 17 GiB each, mistral 61 GiB (docs/OPEN_TASKS.md §V).
constexpr size_t kMoEExpertMaxOpen = 4096;

// Maps expert (layer, eid) into host memory, retiring the LRU tail past
// max_open; a no-op when already resident. Distinct from
// EnsureMoEExpertResident below, which stages an ALREADY-MAPPED expert into a
// device slot.
void EnsureMoEExpertView(MoEExpertDb &db, int layer, int eid,
                         cudaStream_t stream);

// Per-expert static device meta for no-handshake decode (16 B): residency
// slot index + the three global scales. Maintained by
// EnsureMoEExpertResident; dev_slot == -1 marks a not-yet-resident
// expert.
struct alignas(16) MoEExpertMetaDev {
    int32_t dev_slot;
    float gs1, gs2, gs3;
};

struct MoEExpertStore {
    // The neutral zero-copy host store: mmap views, LRU-bounded. This REPLACED
    // per-layer pinned slabs holding every expert for the life of the session
    // (17 GiB for qwen MoE / ornith, 61 GiB for mistral — see
    // docs/OPEN_TASKS.md §V). records[] keeps only the immutable geometry.
    MoEExpertDb experts;
    // LEGACY arena storage — glm-5.3-flash only (docs/OPEN_TASKS.md §V). Empty
    // for every arena-free store. Deleted together with the slab_* offsets
    // when that family migrates.
    std::vector<uint8_t *> h_slabs;
    std::vector<size_t> slab_bytes;
    std::vector<std::vector<MoEExpertRecord>> records;  // [layers][expert]
    int num_layers = 0;
    int num_experts = 0;

    // Per-expert fused gate/up global scales for the single w13 GEMM:
    // d_seg_wgs[flat*2] = gs1, [flat*2+1] = gs3; bounds device triplet
    // {I, I, I} selects segment by output column. Null when unused.
    float *d_seg_wgs = nullptr;
    int *d_seg_bounds = nullptr;

    // No-handshake decode support: host mirror of per-flat
    // {dev_slot(-1 while missing), gs1..gs3}; device mirror uploaded once
    // every entry is resident. gemv_ready == true unlocks the
    // router-readback-free decode path (full-residency guarantee).
    std::vector<MoEExpertMetaDev> h_gemv_meta;
    MoEExpertMetaDev *d_gemv_meta = nullptr;
    bool gemv_ready = false;

    void Free();
};

// Ensures an expert is resident in the device cache (see below).

// Per-slot device table for the batched routed GEMV chain (built host-side
// in the handshake flow or on device by the no-handshake gather kernel;
// 8-byte-aligned 24B payload).
struct alignas(8) MoEGvSlotDev {
    int32_t dev_slot;
    float gs1, gs3, gs2;
    float coeff;
};

// Persistent scratch for the batched TC routed chain. When every pointer is
// non-null and `ygu_batch` is supplied, the per-slot SwiGLU prequant is
// hoisted: all K slots quantize in three launches per layer instead of three
// per slot (bit-identical math, same per-slot alpha).
struct MoERoutedTcScratch {
    uint8_t *qpack = nullptr;    // [K][I/2] packed nibbles
    uint8_t *qscales = nullptr;  // [K][ceil(I/16)] E4M3 block scales
    float *sw_f32 = nullptr;     // [K][I] FP32 SwiGLU product
    float *alpha = nullptr;      // [2][K] absmax / prescale per slot
};

// Processes one token's K routed slots (ascending-k deterministic fp32
// accumulation): d_acc += coeff_k * FFN_eidk(x). Stages experts H2D on cache
// misses via EnsureMoEExpertResident. Scratch: d_yg/d_yu [I], d_y2 [H].
//
// Copy-stream mode: when `copy_stream` != nullptr, cache-miss H2D staging is
// enqueued there instead of `stream`; `ev_uploads` (K caller-owned disabled-
// timing events) fence each upload and `ev_front` must be recorded-or-null —
// the implementation records it on `stream` before staging so the copy stream
// never overwrites a slot still being read by previously enqueued compute.
// Numerics are unchanged (same kernels, same accumulation order).
// `d_yg` is 2*I halves when the fused w13 route is eligible (fast-math and a
// non-null store.d_seg_wgs), because the gate and up halves are one GEMM
// output [gate|up]; the split route uses only the first I. `d_yu` is I halves
// and is used by the split route only — the fused route must never read it.
cudaError_t RunMoERoutedFFN(cudaStream_t stream,
                                MoEExpertStore &store,
                                gpu::ExpertDeviceCache &cache, int layer,
                                const MoERoutedSlot *h_slots, int K,
                                const __half *d_x, int H, int I,
                                float *d_acc, __half *d_yg, __half *d_yu,
                                __half *d_y2, std::string &error,
                                cudaStream_t copy_stream = nullptr,
                                cudaEvent_t *ev_uploads = nullptr,
                                cudaEvent_t ev_front = nullptr,
                                const uint8_t *xq = nullptr,
                                const uint8_t *xqs = nullptr,
                                uint8_t *d_qpack_scratch = nullptr,
                                uint8_t *d_qscales_scratch = nullptr,
                                __half *ygu_batch = nullptr,
                                __half *hbuf_batch = nullptr,
                                __half *ypart_batch = nullptr,
                                MoEGvSlotDev *d_gemv_tbl = nullptr,
                                // Optional caller-owned arena for the overflow-
                                // proof scalar SwiGLU: I floats for the FP32
                                // product plus 2 floats for alpha/inv_alpha.
                                // nullptr keeps the cudaMallocAsync fallback.
                                float *d_swi_scratch = nullptr,
                                // Optional batched-prequant scratch. Requires
                                // ygu_batch (K*2I) and (I % 16) == 0; nullptr
                                // keeps the per-slot path.
                                const MoERoutedTcScratch *tc_scratch = nullptr);

// Casts the routed-accumulation buffer to FP16.
cudaError_t RunMoEAccToFp16(cudaStream_t stream, const float *d_acc,
                                __half *d_out, int n);

// Test/parity helper: seeds the MoE fp32 accumulator from the fp16 residual
// (acc[i] = f32(resid[i])) -- the exact moe_init_acc_kernel math that
// the driver runs before the legacy routed chain.
cudaError_t RunMoEAccInitForTest(cudaStream_t stream,
                                     const __half *d_resid, float *d_acc,
                                     int n);

// Batched routed-expert chain for seq==1 GEMV mode: replaces the per-slot
// {w1,w3,w2} launch loop (K x 5 kernels/layer) with 4 launches total:
// w13 GEMV batch ([2I,K]*x per slot), elementwise swiglu, down GEMV batch
// ([H,I]*h per slot -> per-slot partials), ordered coeff reduce
// acc[r] += sum_s coeff_s * ypart[s][r] (ascending s == legacy add order).
//
// Device addressing mirrors the STAGED FUSED cache slot layout: per-slot
// packed base = d_pack + dev_slot*pack_stride with [w13][w2] packed and
// [s13][s2] scales separated by w13_bytes / s13_bytes respectively.
// Requires H % 32 == 0 && I % 32 == 0 (GEMV row-alignment contract).
// Scratch ygu/hbuf/ypart are caller-owned [Ktop*...] buffers. Deterministic:
// fixed partitioning + fixed-order reduce.
cudaError_t RunMoERoutedGemmVBatch(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const MoEGvSlotDev *slots_host_tbl, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc);

// Device-table variant of the batched chain (round-20c/20e): launches the
// four GEMV-batch kernels directly against an already-resident device slot
// table. External linkage so the graph-capture transparency test can pin
// launch semantics; the decode driver uses it for both the chain and the
// per-layer graph path.
cudaError_t RunMoERoutedGemmVBatchCore(
    cudaStream_t stream,
    const uint8_t *d_pack_base, const uint8_t *d_scale_base,
    int64_t pack_stride, int64_t scale_stride,
    const MoEGvSlotDev *slots_dev, int k_slots,
    int64_t w13_bytes, int64_t s13_bytes,
    int H, int I, const __half *d_x,
    __half *ygu, __half *hbuf, __half *ypart, float *acc);

// Composes the batch GEMV slot table from device-side routing results:
// tbl[k] = {meta[d_slots[k].eid].*, coeff = d_slots[k].coeff}. Used by the
// no-handshake decode path where the host never reads back router output.
// Launch config: <<<1, 32>>>, K <= 16.
cudaError_t RunMoEGemvGatherTable(
    cudaStream_t stream,
    const MoERoutedSlot *d_slots, int k_slots,
    const MoEExpertMetaDev *d_meta,   // [num_experts] slice of the layer
    MoEGvSlotDev *d_tbl_out);         // [k_slots]

// True once every routed expert is resident and mirrored to device meta
// (full-residency steady state): unlocks the router-readback-free decode
// path. One-time lazy upload of the mirror on first readiness.
bool MoEGemvNohandshake(const MoEExpertStore &store);

// Same routed GEMV without the device->host handshake; the generation loop
// calls it on the resident path.
cudaError_t RunMoERoutedGemmVNoHandshake(
    cudaStream_t stream, MoEExpertStore &store,
    gpu::ExpertDeviceCache &cache, int layer,
    const MoERoutedSlot *d_slots, int K,
    const __half *d_x, int H, int I, float *d_acc,
    __half *ygu_b, __half *hbuf_b, __half *ypart_b,
    MoEGvSlotDev *d_tbl,
    std::string &error);

// Batched decode swiglu-quant: per slot s, gate = ygu[s*2I + j],
// up = ygu[s*2I + I + j]; h[s] = silu(gate)*up (FP16) + NVFP4 encode of h.
// Overflow-proof: the product stays FP32, each slot is prescaled by a power
// of two so neither the FP16 store nor the E4M3 block scale saturates, and
// the reciprocal is written to slot_prescale[s] for the reduce to fold back
// in (1.0 when the slot already fits). slot_absmax is scratch the caller
// zeroes first (kMaxTopK floats). Matches the per-slot
// RunAttnSwigluPreScaledQuantInternal bit-for-bit.
cudaError_t RunMoESwigluQuantBatch(cudaStream_t stream,
                                       const __half *ygu, int k_slots, int I,
                                       __half *h, uint8_t *packed,
                                       uint8_t *scales, float *slot_absmax,
                                       float *slot_prescale);

// Batched TC routed-expert chain (no host readback): gather (device table)
// -> w13 batch TC GEMM -> batched swiglu-quant -> w2 batch TC GEMM ->
// ascending coeff reduce. d_acc = f32(d_resid) + sum_k coeff_k * FFN_k(x)
// -- the identical fp32 add order as the legacy chain (acc-init folded into
// the reduce; legacy: acc = f32(resid) then += coeff_k * y_k, k ascending).
// Requires TC fast_math, full residency mirror (MoEGemvNohandshake),
// I % 16 == 0. splitk13: caller buffer for the w13 split-K partials
// (QwenTcBatchSplitK(H, 2*I) * K * 2I floats; nullptr OK when the helper
// returns 1). d_tbl: [16] reused gather output (MoEGvSlotDev).
cudaError_t RunMoERoutedTcBatch(
    cudaStream_t stream, MoEExpertStore &store,
    const gpu::ExpertDeviceCache &cache, int layer,
    const MoERoutedSlot *d_routes, const uint8_t *xq, const uint8_t *xqs,
    int K, int H, int I, const __half *d_resid, float *d_acc,
    __half *ygu_batch, __half *h_batch, __half *yp_batch,
    uint8_t *qpack_batch, uint8_t *qscales_batch, float *splitk13,
    MoEGvSlotDev *d_tbl);

// Device-table variant of the batched TC routed chain for streaming
// (partial-residency) models whose router-readback handshake already knows
// the per-slot dev_slot and per-segment global scales on the host: skips the
// device gather and runs w13 batch TC GEMM -> batched swiglu-quant -> w2
// batch TC GEMM -> ascending coeff reduce against a caller-supplied table
// (layout-identical to lsx_tc::LsxTcBatchSlot). Same numerics as
// RunMoERoutedTcBatch (bit-exact per slot). The caller must have staged
// every routed slot and zero-seeded d_acc.
cudaError_t RunMoERoutedTcBatchTbl(
    cudaStream_t stream, const gpu::ExpertDeviceCache &cache,
    const MoEGvSlotDev *d_tbl, const uint8_t *xq, const uint8_t *xqs,
    int K, int H, int I, int64_t w13_bytes, int64_t s13_bytes,
    float *d_acc, __half *ygu_batch, __half *h_batch, __half *yp_batch,
    uint8_t *qpack_batch, uint8_t *qscales_batch, float *splitk13);

// Host-side EOS truncation (decode loop runs to max_new without per-step
// EOS readback). Keeps the EOS token (legacy break semantics), returns
// true when a cut happened. eos_id < 0 disables.
bool MoETruncateTokensAtEos(std::vector<int32_t> &tokens, int eos_id);

// MoE FFN epilogue (HF parity): out = routed_acc + sigmoid(gate.x) * shared.
// d_routed_acc: [seq,H] fp32; scratch_i [seq,SI], scratch_h [seq,H];
// d_gate_val [seq] fp32 workspace. sw1=gate_proj [SI,H], sw3=up_proj,
// sw2=down_proj [H,SI].
cudaError_t RunMoESharedCombine(
    cudaStream_t stream, const lsxcommon::lsx_nvfp4::LsxNvfp4Weight &sw1,
    const lsxcommon::lsx_nvfp4::LsxNvfp4Weight &sw2,
    const lsxcommon::lsx_nvfp4::LsxNvfp4Weight &sw3,
    const __half *d_shared_gate, const __half *d_x, int seq, int H, int SI,
    const float *d_routed_acc, __half *d_scratch_i, __half *d_scratch_h,
    float *d_gate_val, __half *d_out, const uint8_t *xq = nullptr,
    const uint8_t *xqs = nullptr);

// Ensures expert `eid` of `layer` is resident in the device cache; stages its
// payload H2D from the pinned slab on miss. Returns slot index, -1 on error
// (error text goes to `error` when non-null). When `h2d_stream` is provided,
// miss staging is enqueued on that (copy) stream instead of `stream`; set
// *missed so the caller can fence the upload before consuming the slot.
// *evicted_slot receives the recycled slot index on eviction (-1 otherwise):
// the caller must fence in-flight readers of that slot (event wait) before
// the staged upload overwrites it.
int EnsureMoEExpertResident(MoEExpertStore &store, int layer, int eid,
                                gpu::ExpertDeviceCache &cache,
                                cudaStream_t stream, std::string &error,
                                cudaStream_t h2d_stream = nullptr,
                                bool *missed = nullptr,
                                int *evicted_slot = nullptr);

// Loads every experts/layer_<l>/exp_<e>.arrow under
// <model_path>/logestix/ into pinned host memory. Strict errors.
bool LoadMoEExpertStore(const std::string &model_path,
                            lsxcommon::moe::MoEExpertStore &store, std::string &error);

// Allocates the LRU device cache sized to the largest expert payload.
bool AllocateMoEExpertCache(moe::MoEExpertStore &store, int slots,
                                gpu::ExpertDeviceCache &cache,
                                std::string &error);
void FreeMoEExpertCache(gpu::ExpertDeviceCache &cache);

} // namespace lsxcommon::moe
