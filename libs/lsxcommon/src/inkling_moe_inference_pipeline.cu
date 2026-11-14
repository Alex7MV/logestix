// Inkling-2-Small generation pipeline (Model Ingestion Protocol stage 5).
//
// STATE OF THIS FILE (2026-11-14): the residency scaffolding is complete and
// reviewed — the KV session, the decode scratch arena, and the engine's
// position-dependent log-scaling factor. The per-layer forward pass is NOT
// written: `RunInklingGeneration` returns a hard error, and the model refuses to
// Load so that nothing downstream can mistake a half-built engine for a working
// one. `models/inkling-2-small.json` carries the corresponding golden_debt rows.
//
// What remains, in dependency order (see
// docs/open-tasks/2026-11-14-inkling-2-small-ingest.md § "Plan"):
//   1. Forward(): the 12-step layer body documented below, wiring the
//      already-tested kernels of `inkling_moe_kernels.h`.
//   2. the routed-expert staging + 3 batch GEMVs per layer
//      (`RunLsxNvfp4GemmSeq1GemmVExpertBatch`), FP32 accumulation.
//   3. the sampling loop, GenerationStream emission and abort polling.
//
// Reference order for one decode step, with the two orderings that are easy to
// invert called out:
//
//   embed(row, NVFP4) -> weighted RMSNorm (`embed_norm`)
//   per layer:
//     (a) AttnRes mixture of the attention input (1 query on the sliding
//         layers, 3 on the full layers for separate q/k/v streams; layer 0 has
//         no mixture at all)
//     (b) input norms -> q/k/v/r/gate/sink projections (NVFP4 GEMV)
//     (c) k/v short convolutions on the RAW projections — the k L2 clip and the
//         v value-RMSNorm run AFTER the conv, not before
//     (d) learned L2 clips (q and rel with a learned per-element scale, k
//         weight-free), the weight-free value RMSNorm, and on the full layers
//         the log scaling of q and rel
//     (e) KV append, relative-position bias, learned sinks, sliding window
//     (f) output gate, o_proj, post-attention convolution
//     (g) residual/block-stack bookkeeping — at a block boundary the attention
//         output is NOT added to the residual, the layer input joins the stack
//         instead — then the MLP AttnRes mixture
//     (h) router -> 8 staged experts -> softcapped SwiGLU -> FP32 accumulation
//     (i) post-MLP convolution, residual, and at a boundary the stack append
//   final AttnRes mixture (`output_query`) -> norm -> unembed -> softcap
#include "lsxcommon/inkling_moe_inference_pipeline.h"

#include "inkling_moe_constants.h"

#include <algorithm>
#include <cmath>

namespace lsxcommon::inkling {

float InklingLogScaleTau(const InklingTopology &top, int pos)
{
    if (top.log_scaling_n_floor <= 0) return 1.0f;
    const double effective = static_cast<double>(pos) + 1.0;
    const double ratio =
        std::max(effective / static_cast<double>(top.log_scaling_n_floor), 1.0);
    return static_cast<float>(1.0 + top.log_scaling_alpha * std::log(ratio));
}

std::int64_t InklingSession::Bytes() const
{
    if (!d_k) return 0;
    return 2 * static_cast<int64_t>(geom.layers) * geom.LayerStride() *
               static_cast<int64_t>(sizeof(__half)) +
           static_cast<int64_t>(gen_tail.capacity()) * 4;
}

void InklingSession::Free()
{
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    d_k = nullptr;
    d_v = nullptr;
    geom = {};
    capacity = 0;
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

void InklingSession::Reset()
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

cudaError_t InklingSession::EnsureCapacity(const InklingTopology &top,
                                           int64_t want, cudaStream_t stream)
{
    if (want <= capacity && d_k) return cudaSuccess;
    const int64_t cap = std::max<int64_t>(want, capacity * 2);
    // A single geometry for every layer keeps the address arithmetic identical
    // everywhere; the sliding layers pay for positions they never read (34 MiB
    // per 2048 positions for the whole stack) and that cost is measured in the
    // model report rather than hidden behind a per-layer capacity.
    const KvCacheGeometry g = KvCacheGeometry::Make(
        KvCacheLayout::kPosMajor, cap, top.num_kv_heads, top.head_dim,
        top.num_layers);
    const int64_t elems = g.Elements();
    __half *nk = nullptr;
    __half *nv = nullptr;
    if (cudaMalloc(&nk, static_cast<size_t>(elems) * sizeof(__half)) != cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(&nv, static_cast<size_t>(elems) * sizeof(__half)) != cudaSuccess) {
        cudaFree(nk);
        return cudaGetLastError();
    }
    if (d_k && kv_rows > 0) {
        for (int l = 0; l < static_cast<int>(g.layers); ++l) {
            const size_t bytes = static_cast<size_t>(kv_rows) *
                                 static_cast<size_t>(g.LayerStride()) *
                                 sizeof(__half);
            cudaMemcpyAsync(nk + l * g.LayerStride(),
                            d_k + l * geom.LayerStride(), bytes,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(nv + l * g.LayerStride(),
                            d_v + l * geom.LayerStride(), bytes,
                            cudaMemcpyDeviceToDevice, stream);
        }
        cudaStreamSynchronize(stream);
    }
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    d_k = nk;
    d_v = nv;
    geom = g;
    capacity = cap;
    return cudaSuccess;
}

void InklingScratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    if (expert_stage) cudaFree(expert_stage);
    if (d_ids) cudaFree(d_ids);
    if (d_slots) cudaFree(d_slots);
    if (d_token) cudaFree(d_token);
    if (d_maxv) cudaFree(d_maxv);
    expert_stage = nullptr;
    expert_stage_bytes = 0;
    d_ids = nullptr;
    d_slots = nullptr;
    d_token = nullptr;
    d_maxv = nullptr;
    x_h = nullptr;
    x_f = nullptr;
    mix = nullptr;
}

bool InklingScratch::Allocate(const InklingTopology &top, int64_t capacity,
                              cudaStream_t stream)
{
    Free();
    const int H = top.hidden;
    const int q_dim = top.num_heads * top.head_dim;
    const int kv_dim = top.num_kv_heads * top.head_dim;
    const int rel_dim = top.num_heads * top.d_rel;
    const int k = top.conv_kernel;
    const int I = top.moe_intermediate;
    const int topk = top.top_k;
    const int maxch = std::max(H, kv_dim);

    int64_t off = 0;
    auto take = [&off](int64_t bytes) {
        off = (off + 255) & ~static_cast<int64_t>(255);
        const int64_t at = off;
        off += bytes;
        return at;
    };
    const int64_t o_xh = take(int64_t(H) * 2);
    const int64_t o_xf = take(int64_t(H) * 4);
    const int64_t o_mix = take(int64_t(3) * H * 4);
    const int64_t o_hh = take(int64_t(H) * 2);
    const int64_t o_qh = take(int64_t(H) * 2);
    const int64_t o_kh = take(int64_t(H) * 2);
    const int64_t o_vh = take(int64_t(H) * 2);
    const int64_t o_q = take(int64_t(q_dim) * 2);
    const int64_t o_kv = take(int64_t(kv_dim) * 2);
    const int64_t o_vv = take(int64_t(kv_dim) * 2);
    const int64_t o_qf = take(int64_t(q_dim) * 4);
    const int64_t o_kf = take(int64_t(kv_dim) * 4);
    const int64_t o_vf = take(int64_t(kv_dim) * 4);
    const int64_t o_rel = take(int64_t(rel_dim) * 4);
    const int64_t o_sink = take(int64_t(top.num_heads) * 4);
    const int64_t o_gate = take(int64_t(q_dim) * 2);
    const int64_t o_attn = take(int64_t(q_dim) * 2);
    const int64_t o_oo = take(int64_t(H) * 2);
    const int64_t o_conv = take(int64_t(maxch) * 2);
    const int64_t o_conff = take(int64_t(maxch) * 4);
    const int64_t o_res = take(int64_t(H) * 4);
    const int64_t o_bias = take(int64_t(top.num_heads) * capacity * 4);
    const int64_t o_taps = take(int64_t(kv_dim) * k * 4);
    const int64_t o_proj = take(int64_t(top.num_kv_heads) * k * 4);
    const int64_t o_stack = take(int64_t(frozen::kAttnResMaxSources) * H * 4);
    const int64_t o_mg = take(int64_t(topk) * I * 2);
    const int64_t o_mu = take(int64_t(topk) * I * 2);
    const int64_t o_ma = take(int64_t(topk) * I * 2);
    const int64_t o_md = take(int64_t(topk) * H * 2);
    const int64_t o_macc = take(int64_t(H) * 4);
    const int64_t o_logits = take(int64_t(top.vocab) * 4);
    arena_bytes = off;

    if (cudaMalloc(&arena, static_cast<size_t>(arena_bytes)) != cudaSuccess)
        return false;
    if (cudaMemsetAsync(arena, 0, static_cast<size_t>(arena_bytes), stream) !=
        cudaSuccess)
        return false;
    auto *base = static_cast<uint8_t *>(arena);
    auto at = [base](int64_t o) { return base + o; };
    x_h = reinterpret_cast<__half *>(at(o_xh));
    x_f = reinterpret_cast<float *>(at(o_xf));
    mix = reinterpret_cast<float *>(at(o_mix));
    hh = reinterpret_cast<__half *>(at(o_hh));
    qh = reinterpret_cast<__half *>(at(o_qh));
    kh = reinterpret_cast<__half *>(at(o_kh));
    vh = reinterpret_cast<__half *>(at(o_vh));
    q = reinterpret_cast<__half *>(at(o_q));
    kv_key = reinterpret_cast<__half *>(at(o_kv));
    kv_val = reinterpret_cast<__half *>(at(o_vv));
    qf = reinterpret_cast<float *>(at(o_qf));
    kf = reinterpret_cast<float *>(at(o_kf));
    vf = reinterpret_cast<float *>(at(o_vf));
    relf = reinterpret_cast<float *>(at(o_rel));
    sink = reinterpret_cast<float *>(at(o_sink));
    attn_gate = reinterpret_cast<__half *>(at(o_gate));
    attn_o = reinterpret_cast<__half *>(at(o_attn));
    o_out = reinterpret_cast<__half *>(at(o_oo));
    conv_out = reinterpret_cast<__half *>(at(o_conv));
    conv_f = reinterpret_cast<float *>(at(o_conff));
    residual = reinterpret_cast<float *>(at(o_res));
    rel_bias = reinterpret_cast<float *>(at(o_bias));
    taps_k = reinterpret_cast<float *>(at(o_taps));
    taps_v = reinterpret_cast<float *>(at(o_taps + int64_t(kv_dim) * k * 4));
    proj_k = reinterpret_cast<float *>(at(o_proj));
    proj_v =
        reinterpret_cast<float *>(at(o_proj + int64_t(top.num_kv_heads) * k * 4));
    stack = reinterpret_cast<float *>(at(o_stack));
    moe_gate = reinterpret_cast<__half *>(at(o_mg));
    moe_up = reinterpret_cast<__half *>(at(o_mu));
    moe_act = reinterpret_cast<__half *>(at(o_ma));
    moe_down = reinterpret_cast<__half *>(at(o_md));
    moe_acc = reinterpret_cast<float *>(at(o_macc));
    logits = reinterpret_cast<float *>(at(o_logits));

    const int64_t w1 = int64_t(I) * H / 2;
    const int64_t s1 = int64_t(I) * H / 16;
    const int64_t w2 = int64_t(H) * I / 2;
    const int64_t s2 = int64_t(H) * I / 16;
    const int64_t per = w1 + s1 + w2 + s2 + w1 + s1;
    expert_stage_bytes = per * topk;
    if (cudaMalloc(&expert_stage, static_cast<size_t>(expert_stage_bytes)) !=
        cudaSuccess)
        return false;
    if (cudaMalloc(&d_ids, sizeof(int32_t) * topk) != cudaSuccess) return false;
    if (cudaMalloc(&d_slots, sizeof(InklingRoutedSlot) * topk) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_token, sizeof(int32_t)) != cudaSuccess) return false;
    if (cudaMalloc(&d_maxv, sizeof(float)) != cudaSuccess) return false;
    return true;
}

cudaError_t RunInklingGeneration(cudaStream_t stream,
                                 const InklingPinnedWeights &pw,
                                 InklingExpertDb &experts,
                                 InklingSession &session, InklingScratch &scratch,
                                 const std::vector<int32_t> &input_ids,
                                 int max_gen_tokens,
                                 const std::vector<int32_t> &eos_ids,
                                 float logit_softcap, int unpadded_vocab,
                                 sample::ISampler &sampler,
                                 InklingGenerationResult &out,
                                 const GenerationStream *gs)
{
    (void)stream;
    (void)pw;
    (void)experts;
    (void)session;
    (void)scratch;
    (void)input_ids;
    (void)max_gen_tokens;
    (void)eos_ids;
    (void)logit_softcap;
    (void)unpadded_vocab;
    (void)sampler;
    (void)gs;
    out.ok = false;
    out.tokens.clear();
    // Deliberately a hard error rather than an empty success: a caller that
    // sees `ok == false` cannot mistake this for a model that produced no
    // tokens, and `InklingMoeModel::Load` refuses the model for the same reason.
    out.error_msg =
        "inkling-2-small: the decode pipeline is not implemented yet "
        "(stage 5 in progress; see docs/open-tasks/"
        "2026-11-14-inkling-2-small-ingest.md)";
    return cudaErrorNotSupported;
}

}  // namespace lsxcommon::inkling
