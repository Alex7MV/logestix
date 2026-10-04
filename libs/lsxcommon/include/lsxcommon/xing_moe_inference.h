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
// Position chunks per head in the split-K attention. 8 x 32 = 256 blocks fills
// the 170 SMs of the target part twice over at a 2k context; more splits only
// add combine work once the chunks get shorter than a block.
constexpr int kAttnSplits = 8;

struct XingSession {
    int capacity = 0;
    int length = 0;          // tokens already in the cache
    int kv_layers = 0;       // layers with their own cache slice
    int64_t kv_layer_k = 0;  // floats per layer in d_k
    int64_t kv_layer_v = 0;  // floats per layer in d_v
    // PER-LAYER KV cache: every decoder layer keeps its own K/V (upstream's
    // past_key_values.update(..., layer_idx)), so layer l starts at
    // l * kv_layer_k / l * kv_layer_v. One shared buffer would make each layer
    // overwrite the previous layer's token rows, and the damage is invisible in
    // a single-token prefill (nothing is read back) yet destroys every
    // multi-token prompt.
    float *d_k = nullptr;    // [layers, capacity, heads, nope+rope]
    float *d_v = nullptr;    // [layers, capacity, heads, v]
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
    // Router staging: the neutral device-independent GEMV dispatches matching
    // dtypes only, so the BF16 activation and the BF16 logits need their own
    // buffers (the router kernel then converts per element).
    __nv_bfloat16 *d_router_act = nullptr;   // [hidden]
    __nv_bfloat16 *d_router_logits = nullptr;  // [experts]
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
};

// Allocates every scratch buffer for a `capacity`-token context.
cudaError_t AllocateXingSession(XingSession &s, const XingDims &dims,
                                int capacity, cudaStream_t stream);

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
cudaError_t XingDecodeStep(const XingPinnedWeights &pw, const XingDims &dims,
                           XingSession &s, int token_id, const float *d_rc,
                           const float *d_rs, int pos, cudaStream_t stream,
                           int layers_override = -1);

// Interleaved YaRN rotation of q's rope subspace ([heads, qk_head_dim] with
// `pairs` rope dims at offset `qk_head_dim - 2*pairs`) and of the shared k_rot
// ([2*pairs]). Both launchers derive every element offset from `blockIdx`
// internally: the per-head base must never be a caller-supplied scalar, because
// a single shared offset silently rotates one head and leaves the rest
// unrotated — a failure that a one-token prefill cannot observe (the softmax is
// trivial there).
cudaError_t LaunchXingRopeQ(float *q, int heads, int qk_head_dim, int pairs,
                            const float *cos_t, const float *sin_t,
                            cudaStream_t stream);
cudaError_t LaunchXingRopeK(float *krot, int pairs, const float *cos_t,
                            const float *sin_t, cudaStream_t stream);

// Source line of the stage that failed the last XingDecodeStep (0 = none).
int XingLastFailureLine();

// Diagnostic (tests): copies one layer's K/V row for `pos` into host memory -
// `host_k` receives kv_layer_k floats, `host_v` kv_layer_v. Lets a test see
// WHICH layer's values a cache slot holds without a device-side formatter.
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