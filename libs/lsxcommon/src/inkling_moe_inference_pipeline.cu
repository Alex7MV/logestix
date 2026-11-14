// Inkling-2-Small generation pipeline (Model Ingestion Protocol stage 5).
//
// One decode step, in the order the reference forward pass uses. Four things
// are easy to get backwards and are marked where they happen:
//
//   * the k/v short convolutions run on the RAW projections — the k L2 clip and
//     the v weight-free RMSNorm are applied AFTER the conv;
//   * at an AttnRes block boundary the attention output is NOT added to the
//     residual: the layer input joins the block stack instead, and the MLP
//     mixture then sees it as a block source;
//   * the router's selection bias steers only WHICH experts are chosen — the
//     weights are the raw sigmoid scores, renormalised to 1, then x8.0;
//   * on the full layers the attention mixture asks for THREE streams (separate
//     q/k/v inputs), every other mixture (and every MLP input) asks for one.
//
//   embed(row, NVFP4) -> weighted RMSNorm (`embed_norm`)
//   per layer: (a) attention depth mixture (b) input norms + projections
//     (c) k/v short convs (d) L2 clips / value norm / log scaling (e) KV append,
//     relative bias, sinks, window (f) output gate, o_proj, post-attn conv
//     (g) residual + block stack, MLP mixture (h) router + 8 routed experts
//     (i) post-MLP conv, residual
//   final mixture (`output_query`) -> norm -> unembed -> softcap -> sample
#include "lsxcommon/inkling_moe_inference_pipeline.h"

#include "inkling_moe_constants.h"
#include "lsxcommon/inkling_moe_kernels.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace lsxcommon::inkling {
namespace {

using steady = std::chrono::steady_clock;

std::string LayerTag(const char *what, int layer)
{
    return absl::StrFormat("%s l%d", what, layer);
}

// One NVFP4 projection of a single token row.
cudaError_t Proj(cudaStream_t stream, const lsx_nvfp4::LsxNvfp4Weight &w,
                 const __half *x, __half *y)
{
    return lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(stream, x, w.d_packed,
                                               w.d_scales, w.global_scale, y,
                                               w.cols, w.rows);
}

// The eight staged experts of one layer: three pointer planes plus the two
// global-scale classes the checkpoint carries (gate/up share w13's amax).
struct ExpertStage {
    const uint8_t *gate_p[8] = {};
    const uint8_t *gate_s[8] = {};
    const uint8_t *up_p[8] = {};
    const uint8_t *up_s[8] = {};
    const uint8_t *down_p[8] = {};
    const uint8_t *down_s[8] = {};
    float w13_gs[8] = {};
    float w2_gs[8] = {};
};

}  // namespace

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
    const int maxch = std::max(H, std::max(kv_dim, rel_dim));

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
    // Sized by the engine bound (not `capacity`): the KV session grows
    // lazily and the bias gather must never outrun this buffer.
    const int64_t o_bias = take(int64_t(top.num_heads) * kMaxContext * 4);
    (void)capacity;
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

    // Expert staging: gate/up are [I, H] and down is [H, I], each paired with a
    // per-16 E4M3 scale plane. One contiguous slab, six planes per expert.
    const int64_t w1 = int64_t(I) * H / 2;
    const int64_t s1 = int64_t(I) * H / 16;
    const int64_t w2 = int64_t(H) * I / 2;
    const int64_t s2 = int64_t(H) * I / 16;
    const int64_t per = w1 + s1 + w1 + s1 + w2 + s2;
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

namespace {

// Stages the top-k experts of one layer into the reusable slab and fills the
// three pointer planes the batch GEMV consumes. A missing plane is a hard
// error: a half-staged expert multiplies garbage and would surface as text.
cudaError_t StageExperts(InklingExpertDb &db, int layer, const int32_t *ids,
                         int count, InklingScratch &ws, int I, int H,
                         cudaStream_t stream, ExpertStage &out,
                         std::string &err)
{
    const int64_t w1 = int64_t(I) * H / 2;
    const int64_t s1 = int64_t(I) * H / 16;
    const int64_t w2 = int64_t(H) * I / 2;
    const int64_t s2 = int64_t(H) * I / 16;
    const int64_t per = w1 + s1 + w1 + s1 + w2 + s2;
    if (per * count > ws.expert_stage_bytes) {
        err = absl::StrFormat("expert staging slab too small (need %lld, have %lld)",
                              static_cast<long long>(per * count),
                              static_cast<long long>(ws.expert_stage_bytes));
        return cudaErrorInvalidValue;
    }
    for (int k = 0; k < count; ++k) {
        const int eid = ids[k];
        if (eid < 0 || eid >= db.experts_per_layer) {
            err = absl::StrFormat("router returned expert id %d (layer %d)", eid,
                                  layer);
            return cudaErrorInvalidValue;
        }
        db.EnsureResident(layer, eid, stream);
        const InklingExpertEntry &e =
            db.entries[static_cast<size_t>(layer) *
                           static_cast<size_t>(db.experts_per_layer) +
                       static_cast<size_t>(eid)];
        if (e.host_w1 == nullptr || e.host_w2 == nullptr ||
            e.host_w3 == nullptr || e.host_w1_scales == nullptr ||
            e.host_w2_scales == nullptr || e.host_w3_scales == nullptr) {
            err = absl::StrFormat(
                "routed expert %d of layer %d is not readable from the store",
                eid, layer);
            return cudaErrorInvalidValue;
        }
        if (e.w1_bytes != w1 || e.w3_bytes != w1 || e.w2_bytes != w2 ||
            e.w1_scales_bytes != s1 || e.w3_scales_bytes != s1 ||
            e.w2_scales_bytes != s2) {
            err = absl::StrFormat(
                "routed expert %d of layer %d has an unexpected plane geometry "
                "(w1 %lld/%lld, w2 %lld/%lld, scales %lld,%lld/%lld,%lld)",
                eid, layer, static_cast<long long>(e.w1_bytes),
                static_cast<long long>(w1), static_cast<long long>(e.w2_bytes),
                static_cast<long long>(w2),
                static_cast<long long>(e.w1_scales_bytes),
                static_cast<long long>(e.w3_scales_bytes),
                static_cast<long long>(s1), static_cast<long long>(s2));
            return cudaErrorInvalidValue;
        }
        uint8_t *dst = ws.expert_stage + per * k;
        // gate (w1)
        cudaMemcpyAsync(dst, e.host_w1, static_cast<size_t>(w1),
                        cudaMemcpyHostToDevice, stream);
        out.gate_p[k] = dst;
        cudaMemcpyAsync(dst + w1, e.host_w1_scales, static_cast<size_t>(s1),
                        cudaMemcpyHostToDevice, stream);
        out.gate_s[k] = dst + w1;
        // up (w3)
        dst += w1 + s1;
        cudaMemcpyAsync(dst, e.host_w3, static_cast<size_t>(w1),
                        cudaMemcpyHostToDevice, stream);
        out.up_p[k] = dst;
        cudaMemcpyAsync(dst + w1, e.host_w3_scales, static_cast<size_t>(s1),
                        cudaMemcpyHostToDevice, stream);
        out.up_s[k] = dst + w1;
        // down (w2)
        dst += w1 + s1;
        cudaMemcpyAsync(dst, e.host_w2, static_cast<size_t>(w2),
                        cudaMemcpyHostToDevice, stream);
        out.down_p[k] = dst;
        cudaMemcpyAsync(dst + w2, e.host_w2_scales, static_cast<size_t>(s2),
                        cudaMemcpyHostToDevice, stream);
        out.down_s[k] = dst + w2;
        out.w13_gs[k] = e.w1_global_scale;
        out.w2_gs[k] = e.w2_global_scale;
    }
    return cudaSuccess;
}

// One token through the whole stack. Writes the softcapped (+ sliced) logits row
// into `ws.logits`.
cudaError_t Forward(cudaStream_t stream, const InklingPinnedWeights &pw,
                    InklingExpertDb &experts, InklingSession &session,
                    InklingScratch &ws, int32_t tok, int pos,
                    float logit_softcap, int unpadded_vocab,
                    std::string &stage_err,
                    std::vector<float> *trace = nullptr)
{
    const InklingTopology &top = pw.top;
    const int H = top.hidden;
    const int heads = top.num_heads;
    const int kv_heads = top.num_kv_heads;
    const int hd = top.head_dim;
    const int q_dim = heads * hd;
    const int kv_dim = kv_heads * hd;
    const int rel_dim = heads * top.d_rel;
    const int kconv = top.conv_kernel;
    const int I = top.moe_intermediate;
    const int topk = top.top_k;
    const float eps = static_cast<float>(top.rms_eps);
    const float scale = static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(top.attn_scale_denom)));
    const float inv_static = 1.0f / std::sqrt(static_cast<float>(kconv + 1));
    const float inv_dyn = 1.0f / std::sqrt(static_cast<float>(2 * kconv + 1));

    std::string stage = "embed";
    auto fail = [&](const char *why, cudaError_t e) {
        stage_err = absl::StrFormat("[%s] %s (cuda %d)", stage, why,
                                    static_cast<int>(e));
        return e;
    };
    cudaError_t e = cudaSuccess;
    // ── embed + embed_norm ─────────────────────────────────────────────────
    e = cudaMemcpyAsync(ws.d_token, &tok, sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream);
    if (e != cudaSuccess) return fail("token upload", e);
    e = RunInklingEmbedRow(stream, pw.embed.d_packed, pw.embed.d_scales,
                           pw.embed.global_scale, ws.d_token, ws.x_h, H);
    if (e != cudaSuccess) return fail("embed row", e);
    e = RunInklingHalfToF32(stream, ws.x_h, ws.x_f, H);
    if (e != cudaSuccess) return fail("embed to f32", e);
    e = RunInklingNormF32In(stream, ws.x_f, pw.embed_norm, ws.x_h, 1, H, eps);
    if (e != cudaSuccess) return fail("embed_norm", e);


    if (trace != nullptr) {
        // The kernels run on a non-default stream, so every trace copy must be
        // ordered after them; a bare cudaMemcpy here reads a stale row (the
        // first attempt reported layer 1 identical to layer 0 for this reason).
        cudaStreamSynchronize(stream);
        trace->assign(static_cast<size_t>(top.num_layers + 1) * H, 0.0f);
        std::vector<__half> row(static_cast<size_t>(H));
        cudaMemcpy(row.data(), ws.x_h, row.size() * sizeof(__half),
                   cudaMemcpyDeviceToHost);
        for (int i = 0; i < H; ++i) (*trace)[static_cast<size_t>(i)] = __half2float(row[i]);
    }

    int n_blocks = 0;
    for (int l = 0; l < top.num_layers; ++l) {
        const InklingLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        const bool full = lw.full;
        const bool boundary = top.IsBlockBoundary(l);
        const int nq = full ? 3 : 1;

        // ── (a) attention-input depth mixture ──────────────────────────────
        stage = LayerTag("attn_res", l);
        e = RunInklingHalfToF32(stream, ws.x_h, ws.x_f, H);
        if (e != cudaSuccess) return fail("layer input to f32", e);
        e = cudaMemcpyAsync(ws.stack + static_cast<int64_t>(n_blocks) * H, ws.x_f,
                            sizeof(float) * H, cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return fail("live slot", e);
        const float *attn_input = ws.x_f;
        if (l > 0) {
            const float *attn_q = pw.attn_queries +
                                  static_cast<int64_t>(pw.attn_query_offset[l]) * H;
            e = RunInklingDepthMix(stream, ws.stack, n_blocks + 1, attn_q, nq,
                                   ws.mix, H, eps);
            if (e != cudaSuccess) return fail("attention depth mix", e);
            attn_input = ws.mix;
        }

        // ── (b) input norms + projections ──────────────────────────────────
        stage = LayerTag("attn_proj", l);
        const __half *qsrc = nullptr;
        const __half *ksrc = nullptr;
        const __half *vsrc = nullptr;
        if (full) {
            e = RunInklingNormF32In(stream, attn_input, lw.q_in_norm, ws.qh, 1, H,
                                    eps);
            if (e != cudaSuccess) return fail("q input norm", e);
            e = RunInklingNormF32In(stream, attn_input + H, lw.k_in_norm, ws.kh, 1,
                                    H, eps);
            if (e != cudaSuccess) return fail("k input norm", e);
            e = RunInklingNormF32In(stream, attn_input + 2 * H, lw.v_in_norm,
                                    ws.vh, 1, H, eps);
            if (e != cudaSuccess) return fail("v input norm", e);
            qsrc = ws.qh;
            ksrc = ws.kh;
            vsrc = ws.vh;
        } else {
            e = RunInklingNormF32In(stream, attn_input, lw.attn_norm, ws.hh, 1, H,
                                    eps);
            if (e != cudaSuccess) return fail("attn norm", e);
            qsrc = ksrc = vsrc = ws.hh;
        }
        e = Proj(stream, lw.wq, qsrc, ws.q);
        if (e != cudaSuccess) return fail("wq", e);
        e = Proj(stream, lw.wk, ksrc, ws.kv_key);
        if (e != cudaSuccess) return fail("wk", e);
        e = Proj(stream, lw.wv, vsrc, ws.kv_val);
        if (e != cudaSuccess) return fail("wv", e);
        // The relative states ([heads*d_rel]) get their OWN buffer: `conv_out`
        // is recycled by the k/v convolutions in step (c), so parking them
        // there reads a convolution result as the position bias (a defect that
        // compiles, runs, and answers confidently with the wrong tokens).
        e = Proj(stream, lw.wr, qsrc, ws.o_out);
        if (e != cudaSuccess) return fail("wr", e);
        e = Proj(stream, lw.attn_gate, qsrc, ws.attn_gate);
        if (e != cudaSuccess) return fail("attn gate", e);
        e = RunInklingMatVecF16(stream, lw.sink_proj, qsrc, ws.sink, heads, H);
        if (e != cudaSuccess) return fail("sink proj", e);

        // ── (c) k/v short convolutions on the RAW projections ──────────────
        // The reference convolves `k_proj(x)` / `v_proj(x)` and only then clips
        // k and normalises v. Inverting that order is a silent parity defect.
        stage = LayerTag("kv_conv", l);
        for (int pass = 0; pass < 2; ++pass) {
            const __half *raw = pass == 0 ? ws.kv_key : ws.kv_val;
            float *conv_state = lw.conv_state + pass * kconv * kv_dim;
            const float *taps = nullptr;
            float inv = inv_static;
            if (full) {
                const lsx_nvfp4::LsxNvfp4Weight &proj_w =
                    pass == 0 ? lw.ksconv_proj : lw.vsconv_proj;
                float *proj_out = pass == 0 ? ws.proj_k : ws.proj_v;
                float *dyn = pass == 0 ? ws.taps_k : ws.taps_v;
                const float *static_w =
                    pass == 0 ? lw.kconv_static : lw.vconv_static;
                e = Proj(stream, proj_w, pass == 0 ? ksrc : vsrc, ws.conv_out);
                if (e != cudaSuccess)
                    return fail(pass == 0 ? "k conv proj" : "v conv proj", e);
                e = RunInklingHalfToF32(stream, ws.conv_out, proj_out,
                                        kv_heads * kconv);
                if (e != cudaSuccess) return fail("conv proj to f32", e);
                e = RunInklingDynamicTaps(stream, static_w, proj_out, dyn, kv_dim,
                                          hd, kconv);
                if (e != cudaSuccess) return fail("dynamic taps", e);
                taps = dyn;
                inv = inv_dyn;
            } else {
                taps = pass == 0 ? lw.kconv_w : lw.vconv_w;
            }
            e = cudaMemcpyAsync(ws.conv_out, raw, sizeof(__half) * kv_dim,
                                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) return fail("conv input stage", e);
            e = full ? RunInklingShortConv(stream, ws.conv_out, conv_state,
                                           nullptr, taps, ws.conv_out, kv_dim,
                                           kconv, inv)
                     : RunInklingShortConv(stream, ws.conv_out, conv_state, taps,
                                           nullptr, ws.conv_out, kv_dim, kconv,
                                           inv);
            if (e != cudaSuccess) return fail("short conv", e);
            e = RunInklingHalfToF32(stream, ws.conv_out,
                                    pass == 0 ? ws.kf : ws.vf, kv_dim);
            if (e != cudaSuccess) return fail("conv out to f32", e);
        }

        // ── (d) clips, value norm, log scaling ────────────────────────────
        stage = LayerTag("clip", l);
        e = RunInklingHalfToF32(stream, ws.q, ws.qf, q_dim);
        if (e != cudaSuccess) return fail("q to f32", e);
        e = RunInklingL2Clip(stream, ws.qf, lw.q_clip, heads, hd,
                             static_cast<float>(top.qk_l2_clip));
        if (e != cudaSuccess) return fail("q l2 clip", e);
        e = RunInklingL2Clip(stream, ws.kf, nullptr, kv_heads, hd,
                             static_cast<float>(top.qk_l2_clip));
        if (e != cudaSuccess) return fail("k l2 clip", e);
        e = RunInklingUnweightedNorm(stream, ws.vf, kv_heads, hd, eps);
        if (e != cudaSuccess) return fail("v value norm", e);
        e = RunInklingHalfToF32(stream, ws.o_out, ws.relf, rel_dim);
        if (e != cudaSuccess) return fail("rel to f32", e);
        e = RunInklingL2Clip(stream, ws.relf, lw.rel_clip, heads, top.d_rel,
                             static_cast<float>(top.rel_l2_clip));
        if (e != cudaSuccess) return fail("rel l2 clip", e);
        if (full) {
            const float tau = InklingLogScaleTau(top, pos);
            e = RunInklingLogScale(stream, ws.qf, ws.relf, q_dim, rel_dim, tau);
            if (e != cudaSuccess) return fail("log scale", e);
        }
        e = RunInklingScaleToHalf(stream, ws.qf, ws.q, q_dim, 1.0f);
        if (e != cudaSuccess) return fail("q store", e);
        e = RunInklingScaleToHalf(stream, ws.kf, ws.kv_key, kv_dim, 1.0f);
        if (e != cudaSuccess) return fail("k store", e);
        e = RunInklingScaleToHalf(stream, ws.vf, ws.kv_val, kv_dim, 1.0f);
        if (e != cudaSuccess) return fail("v store", e);

        // ── (e) KV append, relative bias, attention ────────────────────────
        stage = LayerTag("attention", l);
        const int64_t kv_offset =
            static_cast<int64_t>(l) * session.geom.LayerStride();
        __half *d_k = session.d_k + kv_offset;
        __half *d_v = session.d_v + kv_offset;
        e = RunInklingKvAppend(stream, ws.kv_key, ws.kv_val, d_k, d_v,
                               session.geom, pos);
        if (e != cudaSuccess) return fail("kv append", e);
        // The cache is full-context (no ring), so key positions are 0..pos and
        // the bias gather sees absolute positions on both sides.
        const int rel_extent = full ? top.rel_extent : top.sliding_window;
        e = RunInklingRelBias(stream, ws.relf, lw.rel_proj, ws.rel_bias, heads,
                              top.d_rel, rel_extent, pos, pos + 1);
        if (e != cudaSuccess) return fail("rel bias", e);
        e = RunInklingAttention(stream, ws.q, d_k, d_v, session.geom, heads, pos,
                                ws.sink, ws.rel_bias,
                                full ? 0 : top.sliding_window, scale, ws.attn_o);
        if (e != cudaSuccess) return fail("attention", e);

        // ── (f) output gate + o_proj + post-attention conv ─────────────────
        stage = LayerTag("attn_out", l);
        e = RunInklingGateMul(stream, ws.attn_o, ws.attn_gate, ws.attn_o, q_dim);
        if (e != cudaSuccess) return fail("output gate", e);
        e = Proj(stream, lw.wo, ws.attn_o, ws.o_out);
        if (e != cudaSuccess) return fail("wo", e);
        e = RunInklingShortConv(stream, ws.o_out,
                                lw.conv_state + 2 * kconv * kv_dim, lw.attn_conv_w,
                                nullptr, ws.conv_out, H, kconv, inv_static);
        if (e != cudaSuccess) return fail("post-attention conv", e);

        // ── (g) residual / block stack, then the MLP mixture ──────────────
        stage = LayerTag("residual", l);
        e = RunInklingHalfToF32(stream, ws.conv_out, ws.residual, H);
        if (e != cudaSuccess) return fail("post-attn to f32", e);
        if (!boundary) {
            e = RunInklingAddF32(stream, ws.residual, ws.x_f, H);
            if (e != cudaSuccess) return fail("attention residual add", e);
        } else {
            // The layer input becomes a real block source; the residual is the
            // attention output alone.
            ++n_blocks;
        }
        if (n_blocks >= frozen::kAttnResMaxSources) {
            stage_err = absl::StrFormat("[%s] AttnRes stack overflow (%d)",
                                        stage, n_blocks);
            return cudaErrorInvalidValue;
        }
        e = cudaMemcpyAsync(ws.stack + static_cast<int64_t>(n_blocks) * H,
                            ws.residual, sizeof(float) * H,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return fail("mlp partial", e);
        {
            const float *mlp_q = pw.mlp_queries + static_cast<int64_t>(l) * H;
            e = RunInklingDepthMix(stream, ws.stack, n_blocks + 1, mlp_q, 1,
                                   ws.mix, H, eps);
            if (e != cudaSuccess) return fail("mlp depth mix", e);
        }
        e = RunInklingNormF32In(stream, ws.mix, lw.mlp_norm, ws.hh, 1, H, eps);
        if (e != cudaSuccess) return fail("mlp_norm", e);

        // ── (h) router + routed experts ────────────────────────────────────
        stage = LayerTag("moe", l);
        e = RunInklingRouter(stream, lw.router_w, lw.router_bias, ws.hh,
                             top.num_experts, topk, H,
                             static_cast<float>(top.route_scale), ws.d_slots,
                             ws.d_ids);
        if (e != cudaSuccess) return fail("router", e);
        InklingRoutedSlot slots[8];
        int32_t ids[8];
        e = cudaMemcpyAsync(slots, ws.d_slots, sizeof(InklingRoutedSlot) * topk,
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) return fail("slots download", e);
        e = cudaMemcpyAsync(ids, ws.d_ids, sizeof(int32_t) * topk,
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) return fail("ids download", e);
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return fail("router sync", e);

        ExpertStage st;
        e = StageExperts(experts, l, ids, topk, ws, I, H, stream, st, stage_err);
        if (e != cudaSuccess) return e;

        // gate: [I, H] x hh; up: [I, H] x hh; both share w13's global scale.
        e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
            stream, ws.hh, st.gate_p, st.gate_s, st.w13_gs, topk, 0, ws.moe_gate,
            H, I, false);
        if (e != cudaSuccess) return fail("routed gate gemv", e);
        e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
            stream, ws.hh, st.up_p, st.up_s, st.w13_gs, topk, 0, ws.moe_up, H, I,
            false);
        if (e != cudaSuccess) return fail("routed up gemv", e);
        e = RunInklingSoftcapSwiGLU(stream, ws.moe_gate, ws.moe_up, ws.moe_act,
                                    int64_t(topk) * I,
                                    static_cast<float>(top.mlp_gate_softcap),
                                    static_cast<float>(top.mlp_up_softcap));
        if (e != cudaSuccess) return fail("softcapped swiglu", e);
        // down: [H, I] per expert over that expert's own activation slice.
        e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
            stream, ws.moe_act, st.down_p, st.down_s, st.w2_gs, topk, I,
            ws.moe_down, I, H, false);
        if (e != cudaSuccess) return fail("routed down gemv", e);
        e = RunInklingExpertAccumulate(stream, ws.moe_acc, ws.moe_down, ws.d_slots,
                                       topk, H);
        if (e != cudaSuccess) return fail("expert accumulate", e);

        // ── (i) single FP16 cast, post-MLP conv, residual ─────────────────
        stage = LayerTag("mlp_out", l);
        e = RunInklingCastToHalf(stream, ws.moe_acc, ws.o_out, H);
        if (e != cudaSuccess) return fail("moe cast", e);
        e = RunInklingShortConv(stream, ws.o_out,
                                lw.conv_state + 2 * kconv * kv_dim + kconv * H,
                                lw.mlp_conv_w, nullptr, ws.conv_out, H, kconv,
                                inv_static);
        if (e != cudaSuccess) return fail("post-mlp conv", e);
        e = RunInklingHalfToF32(stream, ws.conv_out, ws.conv_f, H);
        if (e != cudaSuccess) return fail("post-mlp to f32", e);
        e = RunInklingAddF32(stream, ws.conv_f, ws.residual, H);
        if (e != cudaSuccess) return fail("mlp residual add", e);
        e = RunInklingScaleToHalf(stream, ws.conv_f, ws.x_h, H, 1.0f);
        if (e != cudaSuccess) return fail("layer output store", e);
        if (trace != nullptr) {
            cudaStreamSynchronize(stream);
            std::vector<__half> row(static_cast<size_t>(H));
            cudaMemcpy(row.data(), ws.x_h, row.size() * sizeof(__half),
                       cudaMemcpyDeviceToHost);
            float *dst = trace->data() +
                         static_cast<size_t>(l + 1) * static_cast<size_t>(H);
            for (int i = 0; i < H; ++i) dst[i] = __half2float(row[i]);
        }
    }

    // ── head: final mixture -> norm -> unembed -> softcap -> slice ─────────
    stage = "head";
    cudaError_t e2 = RunInklingHalfToF32(stream, ws.x_h, ws.x_f, H);
    if (e2 != cudaSuccess) return fail("final residual", e2);
    e2 = cudaMemcpyAsync(ws.stack + static_cast<int64_t>(n_blocks) * H, ws.x_f,
                         sizeof(float) * H, cudaMemcpyDeviceToDevice, stream);
    if (e2 != cudaSuccess) return fail("final partial", e2);
    e2 = RunInklingDepthMix(stream, ws.stack, n_blocks + 1, pw.out_query, 1,
                            ws.mix, H, eps);
    if (e2 != cudaSuccess) return fail("final depth mix", e2);
    e2 = RunInklingNormF32In(stream, ws.mix, pw.final_norm, ws.hh, 1, H, eps);
    if (e2 != cudaSuccess) return fail("final norm", e2);
    e2 = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVF32(
        stream, ws.hh, pw.head.d_packed, pw.head.d_scales, pw.head.global_scale,
        ws.logits, H, pw.head.rows, false);
    if (e2 != cudaSuccess) return fail("unembed", e2);
    if (unpadded_vocab > 0 && unpadded_vocab < pw.head.rows) {
        e2 = RunInklingLogitSoftcap(stream, ws.logits, ws.logits, unpadded_vocab,
                                    logit_softcap);
        if (e2 != cudaSuccess) return fail("logit softcap", e2);
    }
    return cudaSuccess;
}

}  // namespace

cudaError_t RunInklingForwardTrace(cudaStream_t stream, InklingPinnedWeights &pw,
                                   InklingExpertDb &experts,
                                   InklingSession &session, InklingScratch &scratch,
                                   int32_t tok, int pos, float logit_softcap,
                                   int unpadded_vocab,
                                   std::vector<float> &out_states)
{
    const InklingTopology &top = pw.top;
    cudaError_t e = session.EnsureCapacity(top, static_cast<int64_t>(pos) + 1, stream);
    if (e != cudaSuccess) return e;
    if (scratch.arena == nullptr &&
        !scratch.Allocate(top, session.capacity, stream))
        return cudaErrorMemoryAllocation;
    session.Reset();
    for (auto &lw : pw.layers)
        if (lw.conv_state != nullptr && lw.conv_state_bytes > 0)
            cudaMemsetAsync(lw.conv_state, 0,
                            static_cast<size_t>(lw.conv_state_bytes), stream);
    std::string err;
    return Forward(stream, pw, experts, session, scratch, tok, pos, logit_softcap,
                   unpadded_vocab, err, &out_states);
}

cudaError_t RunInklingGeneration(cudaStream_t stream,
                                 InklingPinnedWeights &pw,
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
    out = InklingGenerationResult{};
    if (input_ids.empty()) {
        out.error_msg = "inkling: empty prompt";
        return cudaErrorInvalidValue;
    }
    const InklingTopology &top = pw.top;
    const int ctx = kMaxContext;
    const int64_t needed = static_cast<int64_t>(input_ids.size()) +
                           std::max(0, max_gen_tokens);
    if (needed > ctx) {
        out.error_msg = absl::StrFormat(
            "inkling: %lld positions exceed the engine context bound (%d)",
            static_cast<long long>(needed), ctx);
        return cudaErrorInvalidValue;
    }
    cudaError_t e = session.EnsureCapacity(top, needed, stream);
    if (e != cudaSuccess) {
        out.error_msg = absl::StrFormat("inkling: KV allocation failed (cuda %d)",
                                        static_cast<int>(e));
        return e;
    }
    if (scratch.arena == nullptr) {
        if (!scratch.Allocate(top, session.capacity, stream)) {
            out.error_msg = "inkling: decode scratch allocation failed";
            return cudaErrorMemoryAllocation;
        }
    }
    const int vocab = (unpadded_vocab > 0 && unpadded_vocab < pw.head.rows)
                          ? unpadded_vocab
                          : pw.head.rows;

    // A new request is a new sequence: clear the recurrent short-convolution
    // state of every layer and the KV bookkeeping. Skipping this makes the
    // second identical request decode differently from the first.
    session.Reset();
    for (auto &lw : pw.layers)
        if (lw.conv_state != nullptr && lw.conv_state_bytes > 0) {
            e = cudaMemsetAsync(lw.conv_state, 0,
                                static_cast<size_t>(lw.conv_state_bytes), stream);
            if (e != cudaSuccess) {
                out.error_msg = absl::StrFormat(
                    "inkling: conv-state reset failed (cuda %d)",
                    static_cast<int>(e));
                return e;
            }
        }

    std::string err;
    const auto t_start = steady::now();
    out.ttft_ms = 0.0;

    // ── prompt ────────────────────────────────────────────────────────────
    int pos = 0;
    for (std::size_t i = 0; i < input_ids.size(); ++i) {
        e = Forward(stream, pw, experts, session, scratch, input_ids[i], pos,
                    logit_softcap, unpadded_vocab, err);
        if (e != cudaSuccess) {
            out.error_msg = "inkling prompt: " + err;
            return e;
        }
        ++pos;
    }

    // ── decode ────────────────────────────────────────────────────────────
    const bool want_eos = !eos_ids.empty();
    float maxv = 0.0f;
    int32_t tok = -1;
    bool first = true;
    for (int step = 0; step < max_gen_tokens; ++step) {
        if (gs != nullptr && gs->Aborted()) break;
        // Forward() already applied the softcap and sliced to the unpadded
        // vocabulary; softcapping a second time would compress the row twice.
        e = sampler.Sample(stream, scratch.logits, vocab, sample::LogitType::kF32,
                           scratch.d_token, scratch.d_maxv);
        if (e != cudaSuccess) {
            out.error_msg = absl::StrFormat("inkling: sampler (cuda %d)",
                                            static_cast<int>(e));
            return e;
        }
        e = cudaMemcpyAsync(&tok, scratch.d_token, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) {
            out.error_msg = "inkling: token download failed";
            return e;
        }
        e = cudaMemcpyAsync(&maxv, scratch.d_maxv, sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) {
            out.error_msg = "inkling: max-logit download failed";
            return e;
        }
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            out.error_msg = absl::StrFormat("inkling: decode sync (cuda %d)",
                                            static_cast<int>(e));
            return e;
        }
        if (tok < 0 || tok >= vocab) {
            out.error_msg = absl::StrFormat("inkling: sampler returned id %d", tok);
            return cudaErrorInvalidValue;
        }
        // The softcapped row that produced THIS token is the recorded logits.
        const std::size_t n = static_cast<std::size_t>(vocab);
        out.final_logits.resize(n);
        e = cudaMemcpyAsync(out.final_logits.data(), scratch.logits,
                            n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) return e;
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return e;

        if (first) {
            out.first_token = tok;
            out.ttft_ms =
                std::chrono::duration<double, std::milli>(steady::now() - t_start)
                    .count();
            first = false;
        }
        out.tokens.push_back(tok);
        session.gen_tail.push_back(tok);
        if (gs != nullptr) gs->Emit(static_cast<std::uint32_t>(tok),
                                    static_cast<std::uint32_t>(step), maxv);
        bool stop = false;
        if (want_eos) {
            for (int32_t id : eos_ids)
                if (id == tok) stop = true;
        }
        if (stop) {
            out.eos_terminated = true;
            break;
        }
        if (step + 1 >= max_gen_tokens) break;
        // Feed the token back so the NEXT iteration has its logits row.
        e = Forward(stream, pw, experts, session, scratch, tok, pos, logit_softcap,
                    unpadded_vocab, err);
        if (e != cudaSuccess) {
            out.error_msg = "inkling decode: " + err;
            return e;
        }
        ++pos;
    }

    session.kv_rows = pos;
    session.last_tok = out.tokens.empty() ? input_ids.back() : out.tokens.back();
    out.total_ms =
        std::chrono::duration<double, std::milli>(steady::now() - t_start).count();
    out.ok = true;
    return cudaSuccess;
}

}  // namespace lsxcommon::inkling
