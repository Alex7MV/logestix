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
#include <vector>
#include <string>
#include <vector>

#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_kv_cache.h"
#include "lsxcommon/xing_moe_kernels.h"
#include "lsxcommon/xing_moe_load.h"

namespace lsxcommon::xing {

// Decode geometry, taken from the LOADED weights rather than from the frozen
// constants. Production always passes the frozen values (Load refuses a
// mismatch); a synthetic fixture passes a shrunken copy so the whole step can
// be exercised on a synthetic checkpoint - the same escape hatch
// SyntheticGeometryAllowed() provides everywhere else in this family.
struct XingDims {
    int hidden = 0;
    int hc_mult = 0;
    int vocab = 0;
    int layers = 0;
    int experts = 0;
    int top_k = 0;
    int first_moe_layer = 0;
    int q_lora = 0;
    int kv_lora = 0;
    int heads = 0;
    int qk_nope = 0;
    int qk_rope = 0;
    int qk_head = 0;   // qk_nope + qk_rope
    int v_head = 0;
    int dense_inter = 0;
    int moe_inter = 0;
    int sinkhorn_iters = 0;
    float attn_scale = 0.0f;
    float routed_scale = 0.0f;
    bool norm_topk_prob = true;
    float hc_eps = 0.0f;
    float clamp_min = 0.0f;
    float clamp_max = 0.0f;
    float post_gain = 0.0f;
};

// Derives the dims from an uploaded weight set (never from the constants).
XingDims XingDimsFromWeights(const XingPinnedWeights &pw);

// Per-request decode state. The KV cache holds the FULL (un-absorbed) MHA
// latents — Xing4_0Attention materialises k_pass/value instead of absorbing
// kv_b_proj into q/o, so this is [heads, nope+rope] K and [heads, v] V per
// token, 40 KiB/token/layer (3.1 GiB at a 2048-token context).
// Position chunks per head in the split-K attention.
//
// 8, and it stays 8 because splits sets the chunk boundaries, hence the reduction
// order, hence THE BITS. A project decision (2026-10-05) took bit-identity over
// throughput: 4 is measurably faster and is NOT bit-identical, so it was reverted.
//
// Measured on the AOCC perf tree (RTX 5090, 170 SMs, 32 heads) - kept because the
// number is worth having even though the value is not taken:
//
//   splits  blocks   KV read GB/s @8191   attention us @2047
//      2      64          688               52.7
//      3      96         1129               44.9
//      4     128         1215               40.7    <- fastest, REJECTED
//      5     160         1133               40.7
//      6     192         1020               40.5
//      8     256          867               43.1    <- shipped
//
// So the tail is wave quantisation, not the exp: 8 x 32 = 256 blocks on 170 SMs runs
// the second wave at half occupancy. An earlier comment here justified 8 as "fills
// the SMs twice over, more splits only add combine work", which was exactly
// backwards. Two effects cross at 4 - parallelism wants >= one block per SM, while
// per-block sequential MLP and the partials->combine traffic (131 KB at 8, 65 KB at
// 4) want fewer, fatter blocks - and "fill the SM array" is the wrong rule: 128
// blocks beats 160.
//
// Taking splits=4 would have been +2% at 2048 and +10% at 8191. It was not taken.
// If that trade is ever revisited, the cost is one constant and the numbers above
// are the measurement to re-check against; nothing else has to change, because the
// layout knob and the parity gate make the addressing independent of this choice.
constexpr int kAttnSplits = 8;

struct XingSession {
    int capacity = 0;
    int length = 0;          // tokens already in the cache
    int kv_layers = 0;       // layers with their own cache slice
    int64_t kv_layer_k = 0;  // floats per layer in d_k
    int64_t kv_layer_v = 0;  // floats per layer in d_v
    // Stride geometry of the two caches. K and V are separate because their rows
    // differ in width (kQkHeadDim 192 vs kVHeadDim 128). LayerStride() is
    // layout-invariant, so kv_layer_k / kv_layer_v above stay correct whichever
    // layout is in force; only the addressing changes.
    KvCacheGeometry kgeom{};
    KvCacheGeometry vgeom{};
    // PER-LAYER KV cache: every decoder layer keeps its own K/V (upstream's
    // past_key_values.update(..., layer_idx)), so layer l starts at
    // l * kv_layer_k / l * kv_layer_v. One shared buffer would make each layer
    // overwrite the previous layer's token rows, and the damage is invisible in
    // a single-token prefill (nothing is read back) yet destroys every
    // multi-token prompt.
    // head-major [layers, heads, capacity, nope+rope] / [..., v] by default; the
    // layout in force is kgeom / vgeom, not this comment.
    float *d_k = nullptr;
    float *d_v = nullptr;
    // Four-stream hidden state [hc_mult * hidden] and the mHC outputs.
    float *d_hidden = nullptr;    // [4 * hidden]
    float *d_collapsed = nullptr;  // [hidden]
    float *d_post = nullptr;       // [hc_mult]
    float *d_comb = nullptr;       // [hc_mult * hc_mult]
    // mHC scratch: the hc_fn row dot products followed by the pre gates, sized
    // kMaxMix + kMaxHcMult (80 + 8 floats) rather than the actual mix.
    float *d_rows = nullptr;
    // Split-K attention scratch: per (head, chunk) max/sum and the unnormalised
    // value accumulation, combined by xing_attn_combine_kernel.
    float *d_attn_m = nullptr;     // [heads * kAttnSplits]
    float *d_attn_l = nullptr;     // [heads * kAttnSplits]
    float *d_attn_part = nullptr;  // [heads * kAttnSplits * v_head]
    // F32 scratch: layer norm outputs, attention scores, logits.
    // d_act keeps the LAYERNORMED HIDDEN alive across the MLA chain: q_a's
    // inner RMSNorm overwrites d_norm in place, and kv_a must still consume the
    // 3584-wide activation, not the 768-wide q_lora vector.
    float *d_act = nullptr;     // [hidden]
    float *d_norm = nullptr;    // [max(hidden, q_lora, kv_lora)] scratch
    float *d_q = nullptr;       // [heads * qk]
    float *d_kv_latent = nullptr;  // [kv_lora + rope]
    float *d_kv_b = nullptr;    // [heads * (nope + v)]
    float *d_attn = nullptr;    // [heads * v]
    float *d_scores = nullptr;  // [capacity]
    float *d_logits = nullptr;  // [vocab]
    int64_t logit_count = 0;    // elements in d_logits (set at allocation)
    // Router staging: the BF16 activation the family router reads.
    __nv_bfloat16 *d_router_act = nullptr;   // [hidden]
    float *d_router_scores = nullptr;  // [experts] between the router phases
    float *d_gate = nullptr;    // [16] router weights
    int32_t *d_expert_ids = nullptr;  // [16]
    int32_t token_id = 0;             // current step's input token
    // Diagnostics (tests only): the last layer index that ran, and the first
    // non-finite value seen while scanning the four streams.
    int last_layer = -1;
    int fail_line = 0;   // source line of the failing stage (diagnostics)
    // Last routed expert ids per layer (diagnostics: a decode whose expert
    // ORDER varies run to run loses bit-exactness).
    std::vector<int32_t> last_routed_ids;
    __half *d_half = nullptr;   // [2 * dense_intermediate] GEMV staging
    __half *d_half2 = nullptr;  // [max(hidden, dense_inter)] second F16 slot
    // Device-resident routed-expert pointer tables, one per matrix
    // (gate | down | up), each [layers * experts] of LsxExpertPtrRow. Built once
    // from the loader's host tables so the decode step resolves the routed ids ON
    // THE DEVICE - no D2H and no stream synchronize inside a layer, which is what
    // makes a CUDA-graph capture of the step possible at all.
    // {pos, token_id} for the current step, in DEVICE memory. Every kernel that
    // needs them reads them from here instead of taking them as arguments, which
    // is what lets a captured CUDA graph be replayed for the next position: a
    // graph bakes its kernel arguments in, so a per-token scalar must travel in
    // memory. The host fills it with one 8-byte H2D copy before each step -
    // outside the graph, which is legal there.
    int *d_scalars = nullptr;
    // The captured decode step, once recorded. Recorded on the first generated
    // token and replayed for every one after it; see XingCaptureDecodeGraph.
    cudaGraphExec_t graph_exec = nullptr;
    // Pinned host mirror of d_scalars (source of that copy).
    int *h_scalars = nullptr;
    lsx_nvfp4::LsxExpertPtrRow *d_rows_gate = nullptr;
    lsx_nvfp4::LsxExpertPtrRow *d_rows_down = nullptr;
    lsx_nvfp4::LsxExpertPtrRow *d_rows_up = nullptr;
    bool expert_tables_uploaded = false;
    // Dedicated "up" output. Writing the up projection into d_half + I ALIASES
    // its own input whenever I < K (shared expert: 1024 < 3584), and whether a
    // given x[k] is read before the overlapping store lands depends on block
    // order - that is what made the MoE layer non-deterministic run to run.
    __half *d_half_up = nullptr;  // [dense_intermediate]
    // Persistent FP16 embedding table (dequantized once from the NVFP4 pack;
    // a per-token dequant of 131072x3584 would dominate the step time).
    __half *d_embed_table = nullptr;
    __half *d_half_big = nullptr;  // [vocab] head output
    // Routed-expert down-projection OUTPUT. It must not alias the swiglu
    // input: the batched GEMV writes y + e*hidden while later experts still
    // read x + e*moe_intermediate, and the overlap races with block order.
    __half *d_expert_out = nullptr;  // [top_k * hidden]
    // Expert pointer tables (all experts resident -> no staging copies).
    const uint8_t **d_epacked = nullptr;  // [4] selected
    const uint8_t **d_escales = nullptr;  // [4]
    float *d_egs = nullptr;                // [4]

    void Free();
    // The model owns the session as a unique_ptr and drops it with reset()
    // (TeardownWeights, non-continuation Infer); without this destructor
    // every reset leaked the whole session (A3.5). MistralSession parity.
    ~XingSession() { Free(); }
};

// Allocates every scratch buffer for a `capacity`-token context.
cudaError_t AllocateXingSession(XingSession &s, const XingDims &dims,
                                int capacity, KvCacheLayout layout,
                                cudaStream_t stream);

void FreeXingSession(XingSession &s);

// Host-built YaRN cos/sin tables, [capacity][rope_dim/2] each, already
// multiplied by the upstream attention factor (mscale) — see the constants
// header for why the factor appears twice. The rope geometry comes from the
// passport (YaRN is frozen); only `capacity` is caller-supplied.
bool BuildXingYarnTables(int capacity, std::vector<float> &cos_t,
                         std::vector<float> &sin_t);

// Uploads yarn tables into `dst_rc`/`dst_rs` ([capacity, rope/2] F32 each).
cudaError_t UploadXingYarnTables(cudaStream_t stream, int capacity,
                                 const std::vector<float> &cos_t,
                                 const std::vector<float> &sin_t,
                                 float **dst_rc, float **dst_rs);

// One decode step. `token_id` selects the embedding row (NOT the position),
// `pos` is the absolute KV-cache position. Returns a CUDA error code.
//
// `capture_safe` must be true exactly when this call is being recorded into a CUDA
// graph: it switches the attention to its fixed launch geometry. Nothing else in
// the step is position-dependent any more (the scalars live in device memory), so
// this is the single flag a captured step needs.
cudaError_t XingDecodeStep(const XingPinnedWeights &pw, const XingDims &dims,
                           XingSession &s, int token_id, const float *d_rc,
                           const float *d_rs, int pos, cudaStream_t stream,
                           int layers_override = -1, bool capture_safe = false);

// Interleaved YaRN rotation of q's rope subspace ([heads, qk_head_dim] with
// `pairs` rope dims at offset `qk_head_dim - 2*pairs`) and of the shared k_rot
// ([2*pairs]). Both launchers derive every element offset from `blockIdx`
// internally: the per-head base must never be a caller-supplied scalar, because
// a single shared offset silently rotates one head and leaves the rest
// unrotated — a failure that a one-token prefill cannot observe (the softmax is
// trivial there).
// `cos_t`/`sin_t` are the BASE of the YaRN tables; the row for the step's
// position is selected inside the kernel from d_scalars[0], so the launch
// arguments are the same for every token and a captured graph stays valid.
// Publish this step's {pos, token_id} for the device-side kernels. Call before
// XingDecodeStep, and before every graph launch (never during a capture).
cudaError_t XingSetStepScalars(XingSession &s, int pos, int token_id,
                               cudaStream_t stream);

// Record the decode step into a CUDA graph and instantiate it into the session
// (once; later calls are no-ops). A recorded step does not execute, so the caller
// must launch it with XingLaunchDecodeGraph.
//
// Returns cudaErrorStreamCaptureUnsupported when the step is not capture-legal in
// this configuration - nothing was started, nothing is broken, and the caller
// should simply keep using the eager step (that is the legacy fast-math-off path,
// not an error). Any other failure means a capture actually began and was
// invalidated, which leaves the context unusable: the caller must report it rather
// than continue.
//
// Legal only when nothing in the step allocates or synchronises, which the caller
// guarantees: prefill has already run (so the FP16 embedding table and the routed
// expert pointer tables exist), and LSX_TEST_XING_LAYER_DEBUG is off (its staging
// does a blocking D2H readback). Refuses otherwise instead of capturing a step
// that would fail later.
cudaError_t XingCaptureDecodeGraph(const XingPinnedWeights &pw,
                                   const XingDims &dims, XingSession &s,
                                   const float *d_rc, const float *d_rs,
                                   cudaStream_t stream,
                                   int layers_override = -1);

// True once XingCaptureDecodeGraph has produced a graph for this session.
bool XingDecodeGraphReady(const XingSession &s);

// Replay the captured step. The caller must have published this token's scalars
// with XingSetStepScalars since the last launch - that copy is outside the graph
// on purpose.
cudaError_t XingLaunchDecodeGraph(XingSession &s, cudaStream_t stream);

cudaError_t LaunchXingRopeQ(float *q, int heads, int qk_head_dim, int pairs,
                            const float *cos_t, const float *sin_t,
                            const int *d_scalars, cudaStream_t stream);
cudaError_t LaunchXingRopeK(float *krot, int pairs, const float *cos_t,
                            const float *sin_t, const int *d_scalars,
                            cudaStream_t stream);

// Source line of the stage that failed the last XingDecodeStep (0 = none).
int XingLastFailureLine();

// Diagnostic (tests): copies one layer's K/V row for `pos` into host memory -
// `host_k` receives kv_layer_k floats laid out [heads, head_dim], `host_v`
// kv_layer_v the same way. Under head-major that row is not contiguous in the
// cache, so this gathers it with one strided 2D copy. Lets a test see WHICH
// layer's values a cache slot holds without a device-side formatter.
cudaError_t XingCopyLayerKV(const XingSession &s, int layer, int pos,
                            float *host_k, float *host_v, cudaStream_t stream);

// Counts non-finite elements of a device F32 buffer (diagnostics/tests: a
// decode pipeline that silently produces NaN logits is otherwise invisible).
cudaError_t XingCountNonFinite(const float *d, int64_t n, int *out_count,
                               cudaStream_t stream);

// Largest |value| of a device F32 buffer (diagnostics: a finite-but-huge
// activation is how an FP32 GEMV silently overflows to inf -> NaN).
cudaError_t XingMaxAbs(const float *d, int64_t n, float *out_max,
                       cudaStream_t stream);

// Reads back the last step's logits (host pointer, length = pw vocab).
void XingReadLogits(const XingSession &s, float *host_logits,
                    cudaStream_t stream);

}  // namespace lsxcommon::xing