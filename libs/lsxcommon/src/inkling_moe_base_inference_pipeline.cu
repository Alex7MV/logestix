// Inkling (base) generation pipeline (Model Ingestion Protocol stage 5).
//
// One decode step, in the order the reference forward pass uses
// (transformers models/inkling). The things that are easy to get backwards,
// marked where they happen:
//
//   * the k/v short convolutions run on the RAW projections — the per-head
//     q/k RMS norms are applied AFTER the conv;
//   * the convolutions are `conv(x) + x` with NO normalizer (inkling2's
//     1/sqrt(k+1) does not exist here);
//   * the attention scale is 1/head_dim, there are NO sinks and NO output
//     gate; the full layers log-scale q AND the relative bias by tau(pos);
//   * the router renormalises jointly over the top-6 routed + 2 shared
//     logits (softmax of logsigmoid), x route_scale x gate.global_scale;
//     the selection bias steers only WHICH routed experts run;
//   * the head divides the final hidden by logits_mup_width_multiplier (24)
//     BEFORE the unembed matvec and applies NO softcap.
//
//   embed(row, FP16) -> weighted RMSNorm (`embed_norm`)
//   per layer: attn_norm -> q/k/v/r projections -> k/v short convs ->
//     q/k head norms -> (full: tau on q) -> KV append, relative bias
//     (full: tau on bias), window -> o_proj -> post-attn conv -> residual
//     -> mlp_norm -> [dense SwiGLU x mlp_gs | router + 6 routed experts +
//     2 shared experts x gammas] -> post-MLP conv -> residual
//   final norm -> /24 -> unembed -> slice -> sample
#include "lsxcommon/inkling_moe_base_inference_pipeline.h"

#include "inkling_moe_base_constants.h"
#include "lsxcommon/inkling_moe_base_kernels.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/log.h"

#include <absl/strings/str_format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace lsxcommon::inkling::base {
namespace {

using steady = std::chrono::steady_clock;

std::string LayerTag(const char *what, int layer)
{
    return absl::StrFormat("%s l%d", what, layer);
}

// One dense NVFP4 projection of a single token row through the neutral layer.
cudaError_t Proj(cudaStream_t stream, const DenseNvfp4 &w, const __half *x,
                 __half *y)
{
    if (w.d_packed == nullptr) return cudaErrorInvalidValue;
    return lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmV(stream, x, w.d_packed, w.d_scales,
                                               w.global_scale, y, w.cols,
                                               w.rows, false);
}

// FP32-output projection: the base release's residual stream reaches ~1e6 at
// the deep layers, so a projection that feeds the residual (wo) can exceed the
// FP16 range. The F32 GEMV keeps the magnitude; the caller convolves in FP32.
cudaError_t ProjF32(cudaStream_t stream, const DenseNvfp4 &w, const __half *x,
                    float *y)
{
    if (w.d_packed == nullptr) return cudaErrorInvalidValue;
    return lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVF32(stream, x, w.d_packed,
                                                  w.d_scales, w.global_scale,
                                                  y, w.cols, w.rows, false);
}

// The six staged experts of one layer: three pointer planes + global scales.
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

// Stages the top-k experts of one layer into the reusable slab and fills the
// pointer planes the batch GEMV consumes. A missing plane is a hard error.
cudaError_t StageExperts(ExpertDb &db, int layer, const int32_t *ids, int count,
                         Scratch &ws, int I, int H, cudaStream_t stream,
                         ExpertStage &out, std::string &err)
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
            err = absl::StrFormat("router returned expert id %d (layer %d)",
                                  eid, layer);
            return cudaErrorInvalidValue;
        }
        db.EnsureResident(layer, eid, stream);
        const ExpertEntry &e =
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
                "routed expert %d of layer %d has an unexpected plane geometry",
                eid, layer);
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

// One token through the whole stack. Writes the sliced (unpadded-vocab)
// logits row into `ws.logits`.
cudaError_t Forward(cudaStream_t stream, const PinnedWeights &pw,
                    ExpertDb &experts, Session &session, Scratch &ws,
                    int32_t tok, int pos, int /*unpadded_vocab*/,
                    std::string &stage_err,
                    std::vector<float> *trace = nullptr)
{
    const Topology &top = pw.top;
    const int H = top.hidden;
    const int heads = top.num_heads;
    const int hd = top.head_dim;
    const int q_dim = heads * hd;
    const int rel_dim = heads * top.d_rel;
    const int kconv = top.conv_kernel;
    const int I = top.moe_intermediate;
    const int topk = top.top_k;
    const float eps = static_cast<float>(top.rms_eps);
    // q/k are per-head RMS-normed: the reference scales by 1/head_dim.
    const float scale = static_cast<float>(1.0 / top.attn_scale_denom);

    std::string stage = "embed";
    auto fail = [&](const char *why, cudaError_t e) {
        stage_err = absl::StrFormat("[%s] %s (cuda %d)", stage, why,
                                    static_cast<int>(e));
        return e;
    };
    cudaError_t e = cudaSuccess;
    // ── embed + embed_norm ─────────────────────────────────────────────────
    // The residual stream is carried in FP32 (ws.x_f): the base release's
    // residual reaches ~2e5 at the early positions of the deeper layers,
    // beyond the FP16 range (65504). Narrowing it to FP16 produced NaN K rows
    // by layer 9 on the real tree. The norm weights stay FP16.
    e = cudaMemcpyAsync(ws.d_token, &tok, sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return fail("token upload", e);
    e = RunEmbedGather(stream, pw.embed, ws.d_token, ws.x_h, H);
    if (e != cudaSuccess) return fail("embed row", e);
    e = RunHalfToF32(stream, ws.x_h, ws.x_f, H);
    if (e != cudaSuccess) return fail("embed to f32", e);
    e = RunNormF32(stream, ws.x_f, pw.embed_norm, ws.x_f, 1, H, eps);
    if (e != cudaSuccess) return fail("embed_norm", e);

    if (trace != nullptr) {
        // The kernels run on a non-default stream, so every trace copy must
        // be ordered after them.
        cudaStreamSynchronize(stream);
        trace->assign(static_cast<size_t>(top.num_layers + 1) * H, 0.0f);
        cudaMemcpy(trace->data(), ws.x_f, static_cast<size_t>(H) * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    for (int l = 0; l < top.num_layers; ++l) {
        const LayerWeights &lw = pw.layers[static_cast<size_t>(l)];
        const bool full = lw.full;
        const int kv_heads = top.KvHeadsOf(l);
        const int kv_dim = kv_heads * hd;

        // ── (a) input norm + projections ───────────────────────────────────
        stage = LayerTag("attn_proj", l);
        // The residual stream is already FP32 (ws.x_f); the norm reads it
        // directly and narrows only the projection input to FP16.
        e = RunNormF32In(stream, ws.x_f, lw.attn_norm, ws.hh, 1, H, eps);
        if (e != cudaSuccess) return fail("attn norm", e);
        e = Proj(stream, lw.wq, ws.hh, ws.q);
        if (e != cudaSuccess) return fail("wq", e);
        e = Proj(stream, lw.wk, ws.hh, ws.kv_key);
        if (e != cudaSuccess) return fail("wk", e);
        e = Proj(stream, lw.wv, ws.hh, ws.kv_val);
        if (e != cudaSuccess) return fail("wv", e);
        // The relative states get their OWN buffer: conv_out is recycled by
        // the k/v convolutions below.
        e = Proj(stream, lw.wr, ws.hh, ws.o_out);
        if (e != cudaSuccess) return fail("wr", e);
        e = RunHalfToF32(stream, ws.o_out, ws.relf, rel_dim);
        if (e != cudaSuccess) return fail("rel to f32", e);

        // ── (b) k/v short convolutions on the RAW projections ──────────────
        // The reference convolves k_proj(x)/v_proj(x) and only then norms
        // per head. Inverting that order is a silent parity defect.
        stage = LayerTag("kv_conv", l);
        for (int pass = 0; pass < 2; ++pass) {
            const __half *raw = pass == 0 ? ws.kv_key : ws.kv_val;
            float *conv_state = lw.conv_state + pass * kconv * kv_dim;
            const float *taps = pass == 0 ? lw.kconv_w : lw.vconv_w;
            e = cudaMemcpyAsync(ws.conv_out, raw, sizeof(__half) * kv_dim,
                                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) return fail("conv input stage", e);
            e = RunShortConv(stream, ws.conv_out, conv_state, taps, ws.conv_out,
                             kv_dim, kconv);
            if (e != cudaSuccess) return fail("short conv", e);
            if (pass == 0) {
                // k: conv output -> per-head RMSNorm (FP32) -> FP16 cache row.
                e = RunHalfToF32(stream, ws.conv_out, ws.qf, kv_dim);
                if (e != cudaSuccess) return fail("k conv to f32", e);
                e = RunHeadNorm(stream, ws.qf, lw.k_norm, ws.qf, kv_heads, hd,
                                eps);
                if (e != cudaSuccess) return fail("k head norm", e);
                e = RunScaleToHalf(stream, ws.qf, ws.kv_key, kv_dim, 1.0f);
                if (e != cudaSuccess) return fail("k store", e);
            } else {
                // v: the conv output IS the cache value (no value norm in this
                // family). Store it back — leaving ws.kv_val at the raw
                // projection silently skips the v convolution (the defect
                // that produced NaNs by layer 3 on the real tree).
                e = cudaMemcpyAsync(ws.kv_val, ws.conv_out,
                                    sizeof(__half) * kv_dim,
                                    cudaMemcpyDeviceToDevice, stream);
                if (e != cudaSuccess) return fail("v store", e);
            }
        }

        // ── (c) q head norm, log scaling ───────────────────────────────────
        stage = LayerTag("q_norm", l);
        e = RunHalfToF32(stream, ws.q, ws.qf, q_dim);
        if (e != cudaSuccess) return fail("q to f32", e);
        e = RunHeadNorm(stream, ws.qf, lw.q_norm, ws.qf, heads, hd, eps);
        if (e != cudaSuccess) return fail("q head norm", e);
        const float tau = LogScaleTau(top, pos);
        if (full && tau != 1.0f) {
            e = RunLogScale(stream, ws.qf, q_dim, nullptr, 0, tau);
            if (e != cudaSuccess) return fail("q log scale", e);
        }
        e = RunScaleToHalf(stream, ws.qf, ws.q, q_dim, 1.0f);
        if (e != cudaSuccess) return fail("q store", e);

        // ── (d) KV append, relative bias, attention ────────────────────────
        stage = LayerTag("attention", l);
        const KvCacheGeometry &geom = session.geom[static_cast<size_t>(l)];
        __half *d_k = session.d_k[static_cast<size_t>(l)];
        __half *d_v = session.d_v[static_cast<size_t>(l)];
        e = RunKvAppend(stream, ws.kv_key, ws.kv_val, d_k, d_v, geom, pos);
        if (e != cudaSuccess) return fail("kv append", e);
        const int rel_extent = top.RelExtentOf(l);
        e = RunRelBias(stream, ws.relf, lw.rel_proj, ws.rel_bias, heads,
                       top.d_rel, rel_extent, pos, pos + 1);
        if (e != cudaSuccess) return fail("rel bias", e);
        if (full && tau != 1.0f) {
            e = RunLogScale(stream, ws.rel_bias, int64_t(heads) * (pos + 1),
                            nullptr, 0, tau);
            if (e != cudaSuccess) return fail("bias log scale", e);
        }
        e = RunAttention(stream, ws.q, d_k, d_v, geom, heads, pos, ws.rel_bias,
                         full ? 0 : top.sliding_window, scale, ws.attn_o);
        if (e != cudaSuccess) return fail("attention", e);
        stage = LayerTag("attn_out", l);
        // The wo projection feeds the FP32 residual: at the deep layers the
        // base release's attention output (~881) times wo can exceed the FP16
        // range, so the GEMV writes FP32 directly (no FP16 narrowing).
        e = ProjF32(stream, lw.wo, ws.attn_o, ws.moe_f32);
        if (e != cudaSuccess) return fail("wo", e);
        e = RunShortConvF32(stream, ws.moe_f32,
                            lw.conv_state + 2 * kconv * kv_dim, lw.attn_conv_w,
                            ws.conv_f32, H, kconv);
        if (e != cudaSuccess) return fail("post-attention conv", e);

        // ── (e) residual, then the MLP ─────────────────────────────────────
        stage = LayerTag("residual", l);
        e = cudaMemcpyAsync(ws.residual, ws.conv_f32, sizeof(float) * H,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return fail("post-attn stage", e);
        e = RunAddF32(stream, ws.residual, ws.x_f, H);
        if (e != cudaSuccess) return fail("attention residual add", e);
        e = RunNormF32In(stream, ws.residual, lw.mlp_norm, ws.hh, 1, H, eps);
        if (e != cudaSuccess) return fail("mlp_norm", e);

        stage = LayerTag("mlp", l);
        if (top.IsDenseMlp(l)) {
            // Dense SwiGLU prefix, output x the learned global scale.
            e = Proj(stream, lw.dense_gate, ws.hh, ws.moe_gate);
            if (e != cudaSuccess) return fail("dense gate", e);
            e = Proj(stream, lw.dense_up, ws.hh, ws.moe_up);
            if (e != cudaSuccess) return fail("dense up", e);
            e = RunSwiGLU(stream, ws.moe_gate, ws.moe_up, ws.moe_act,
                          top.dense_intermediate);
            if (e != cudaSuccess) return fail("dense swiglu", e);
            e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVF32(
                stream, ws.moe_act, lw.dense_down.d_packed,
                lw.dense_down.d_scales, lw.dense_down.global_scale,
                ws.moe_acc, top.dense_intermediate, H, false);
            if (e != cudaSuccess) return fail("dense down", e);
            float mlp_gs = 1.0f;
            e = cudaMemcpyAsync(&mlp_gs, lw.mlp_gs, sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
            if (e != cudaSuccess) return fail("mlp gs download", e);
            e = cudaStreamSynchronize(stream);
            if (e != cudaSuccess) return fail("mlp gs sync", e);
            e = RunScaleF32(stream, ws.moe_acc, H, mlp_gs);
            if (e != cudaSuccess) return fail("mlp gs scale", e);
        } else {
            // ── router + routed experts + shared-sink experts ─────────────
            float router_gs = 1.0f;
            e = cudaMemcpyAsync(&router_gs, lw.router_gs, sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
            if (e != cudaSuccess) return fail("router gs download", e);
            e = cudaStreamSynchronize(stream);
            if (e != cudaSuccess) return fail("router gs sync", e);
            e = RunRouter(stream, lw.router_w, lw.router_bias, ws.hh,
                          top.num_experts, top.shared_experts, topk, H,
                          static_cast<float>(top.route_scale), router_gs,
                          static_cast<RoutedSlot *>(ws.d_slots), ws.d_ids,
                          ws.d_gammas);
            if (e != cudaSuccess) return fail("router", e);
            RoutedSlot slots[8];
            int32_t ids[8];
            float gammas[8];
            e = cudaMemcpyAsync(slots, ws.d_slots, sizeof(RoutedSlot) * topk,
                                cudaMemcpyDeviceToHost, stream);
            if (e != cudaSuccess) return fail("slots download", e);
            e = cudaMemcpyAsync(ids, ws.d_ids, sizeof(int32_t) * topk,
                                cudaMemcpyDeviceToHost, stream);
            if (e != cudaSuccess) return fail("ids download", e);
            e = cudaMemcpyAsync(gammas, ws.d_gammas,
                                sizeof(float) * top.shared_experts,
                                cudaMemcpyDeviceToHost, stream);
            if (e != cudaSuccess) return fail("gammas download", e);
            e = cudaStreamSynchronize(stream);
            if (e != cudaSuccess) return fail("router sync", e);

            ExpertStage st;
            e = StageExperts(experts, l, ids, topk, ws, I, H, stream, st,
                             stage_err);
            if (e != cudaSuccess) return e;
            e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, ws.hh, st.gate_p, st.gate_s, st.w13_gs, topk, 0,
                ws.moe_gate, H, I, false);
            if (e != cudaSuccess) return fail("routed gate gemv", e);
            e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
                stream, ws.hh, st.up_p, st.up_s, st.w13_gs, topk, 0,
                ws.moe_up, H, I, false);
            if (e != cudaSuccess) return fail("routed up gemv", e);
            e = RunSwiGLU(stream, ws.moe_gate, ws.moe_up, ws.moe_act,
                          int64_t(topk) * I);
            if (e != cudaSuccess) return fail("routed swiglu", e);
            // The down projection writes FP32: the base release's expert down
            // output can exceed the FP16 range (a shared expert reached ~9.6e5
            // at an early position). One F32 GEMV per expert into its own slice.
            for (int k = 0; k < topk; ++k) {
                e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVF32(
                    stream, ws.moe_act + int64_t(k) * I, st.down_p[k],
                    st.down_s[k], st.w2_gs[k],
                    ws.moe_down_f32 + int64_t(k) * H, I, H, false);
                if (e != cudaSuccess) return fail("routed down gemv", e);
            }
            e = RunExpertAccumulateF32(stream, ws.moe_acc, ws.moe_down_f32,
                                       static_cast<RoutedSlot *>(ws.d_slots),
                                       topk, H);
            if (e != cudaSuccess) return fail("expert accumulate", e);

            // Shared experts: each sees the SAME mlp_norm input, weighted by
            // its own gamma from the joint renorm. The down projection writes
            // FP32 (same overflow reason as the routed down).
            for (int s = 0; s < top.shared_experts; ++s) {
                const SharedExpertWeights &se =
                    lw.shared[static_cast<size_t>(s)];
                e = Proj(stream, se.gate, ws.hh, ws.sh_gate);
                if (e != cudaSuccess) return fail("shared gate", e);
                e = Proj(stream, se.up, ws.hh, ws.sh_up);
                if (e != cudaSuccess) return fail("shared up", e);
                e = RunSwiGLU(stream, ws.sh_gate, ws.sh_up, ws.sh_act, I);
                if (e != cudaSuccess) return fail("shared swiglu", e);
                e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVF32(
                    stream, ws.sh_act, se.down.d_packed, se.down.d_scales,
                    se.down.global_scale, ws.sh_down_f32, I, H, false);
                if (e != cudaSuccess) return fail("shared down", e);
                e = RunSharedAccumulateF32(stream, ws.moe_acc, ws.sh_down_f32,
                                           gammas[s], H);
                if (e != cudaSuccess) return fail("shared accumulate", e);
            }
        }

        // ── (f) post-MLP conv, residual ────────────────────────────────────
        stage = LayerTag("mlp_out", l);
        // The post-MLP conv feeds the FP32 residual: convolve the FP32 MLP
        // accumulator directly (no FP16 narrowing — the base release's conv
        // output reaches ~1.9e5 at the early positions of the deeper layers).
        e = RunShortConvF32(stream, ws.moe_acc,
                            lw.conv_state + 2 * kconv * kv_dim + kconv * H,
                            lw.mlp_conv_w, ws.conv_f32, H, kconv);
        if (e != cudaSuccess) return fail("post-mlp conv", e);
        e = RunAddF32(stream, ws.conv_f32, ws.residual, H);
        if (e != cudaSuccess) return fail("mlp residual add", e);
        // The layer output IS the FP32 residual stream (no FP16 narrowing).
        e = cudaMemcpyAsync(ws.x_f, ws.conv_f32, sizeof(float) * H,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return fail("layer output store", e);
        if (trace != nullptr) {
            cudaStreamSynchronize(stream);
            float *dst = trace->data() +
                         static_cast<size_t>(l + 1) * static_cast<size_t>(H);
            cudaMemcpy(dst, ws.x_f, static_cast<size_t>(H) * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    }

    // ── head: final norm -> /mup -> unembed -> slice (NO softcap) ──────────
    stage = "head";
    // The residual stream is already FP32 (ws.x_f).
    e = RunNormF32In(stream, ws.x_f, pw.final_norm, ws.hh, 1, H, eps);
    if (e != cudaSuccess) return fail("final norm", e);
    // muP: the reference divides the final hidden state by the width
    // multiplier BEFORE lm_head. The normed row is FP16; widen, divide once
    // in FP32, narrow back — the same rounding order as the reference's
    // bf16 hidden / fp32 constant.
    {
        e = RunHalfToF32(stream, ws.hh, ws.x_f, H);
        if (e != cudaSuccess) return fail("head to f32", e);
        const float inv_mup =
            static_cast<float>(1.0 / top.logits_mup_div);
        e = RunScaleF32(stream, ws.x_f, H, inv_mup);
        if (e != cudaSuccess) return fail("mup scale", e);
        e = RunScaleToHalf(stream, ws.x_f, ws.hh, H, 1.0f);
        if (e != cudaSuccess) return fail("mup store", e);
    }
    e = RunMatVecF16(stream, pw.head, ws.hh, ws.logits, pw.embed_rows, H);
    if (e != cudaSuccess) return fail("unembed", e);
    return cudaSuccess;
}

}  // namespace

float LogScaleTau(const Topology &top, int pos)
{
    if (top.log_scaling_n_floor <= 0) return 1.0f;
    const double effective = static_cast<double>(pos) + 1.0;
    const double ratio =
        std::max(effective / static_cast<double>(top.log_scaling_n_floor), 1.0);
    return static_cast<float>(1.0 + top.log_scaling_alpha * std::log(ratio));
}

std::int64_t Session::Bytes() const
{
    int64_t total = 0;
    for (size_t l = 0; l < geom.size(); ++l)
        if (d_k[l])
            total += 2 * geom[l].Elements() * static_cast<int64_t>(sizeof(__half));
    return total + static_cast<int64_t>(gen_tail.capacity()) * 4;
}

void Session::Free()
{
    for (auto *p : d_k)
        if (p) cudaFree(p);
    for (auto *p : d_v)
        if (p) cudaFree(p);
    d_k.clear();
    d_v.clear();
    geom.clear();
    capacity = 0;
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

void Session::Reset()
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

cudaError_t Session::EnsureCapacity(const Topology &top, int64_t want,
                                    cudaStream_t stream)
{
    if (want <= capacity && !d_k.empty()) return cudaSuccess;
    const int64_t cap = std::max<int64_t>(want, capacity * 2);
    const int L = top.num_layers;
    // Per-layer geometry: the two layer classes have different kv widths.
    // Pos-major (the measured default; see the ingestion skill).
    std::vector<KvCacheGeometry> g(static_cast<size_t>(L));
    std::vector<__half *> nk(static_cast<size_t>(L), nullptr);
    std::vector<__half *> nv(static_cast<size_t>(L), nullptr);
    auto release_new = [&]() {
        for (auto *p : nk)
            if (p) cudaFree(p);
        for (auto *p : nv)
            if (p) cudaFree(p);
    };
    for (int l = 0; l < L; ++l) {
        g[static_cast<size_t>(l)] = KvCacheGeometry::Make(
            KvCacheLayout::kPosMajor, cap, top.KvHeadsOf(l), top.head_dim, 1);
        const int64_t elems = g[static_cast<size_t>(l)].Elements();
        if (cudaMalloc(&nk[static_cast<size_t>(l)],
                       static_cast<size_t>(elems) * sizeof(__half)) !=
            cudaSuccess) {
            release_new();
            return cudaGetLastError();
        }
        if (cudaMalloc(&nv[static_cast<size_t>(l)],
                       static_cast<size_t>(elems) * sizeof(__half)) !=
            cudaSuccess) {
            release_new();
            return cudaGetLastError();
        }
    }
    if (!d_k.empty() && kv_rows > 0) {
        // The rows of one layer are contiguous: kv_rows * pos_stride of them.
        // `LayerStride()` INCLUDES the capacity — copying kv_rows *
        // LayerStride() over-reads past the old allocation (the inkling2
        // growth defect, measured as a cuda 1 on the NEXT request).
        for (int l = 0; l < L; ++l) {
            const size_t li = static_cast<size_t>(l);
            const size_t bytes = static_cast<size_t>(kv_rows) *
                                 static_cast<size_t>(geom[li].pos_stride) *
                                 sizeof(__half);
            cudaError_t ce = cudaMemcpyAsync(nk[li], d_k[li], bytes,
                                             cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) { release_new(); return ce; }
            ce = cudaMemcpyAsync(nv[li], d_v[li], bytes,
                                 cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) { release_new(); return ce; }
        }
        const cudaError_t se = cudaStreamSynchronize(stream);
        if (se != cudaSuccess) { release_new(); return se; }
    }
    for (auto *p : d_k)
        if (p) cudaFree(p);
    for (auto *p : d_v)
        if (p) cudaFree(p);
    d_k = std::move(nk);
    d_v = std::move(nv);
    geom = std::move(g);
    capacity = cap;
    return cudaSuccess;
}

void Scratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    if (expert_stage) cudaFree(expert_stage);
    expert_stage = nullptr;
    expert_stage_bytes = 0;
    if (d_ids) cudaFree(d_ids);
    if (d_slots) cudaFree(d_slots);
    if (d_gammas) cudaFree(d_gammas);
    if (d_token) cudaFree(d_token);
    if (d_maxv) cudaFree(d_maxv);
    d_ids = nullptr;
    d_slots = nullptr;
    d_gammas = nullptr;
    d_token = nullptr;
    d_maxv = nullptr;
    x_h = nullptr;
    x_f = nullptr;
}

bool Scratch::Allocate(const Topology &top, cudaStream_t stream)
{
    Free();
    const int H = top.hidden;
    const int q_dim = top.num_heads * top.head_dim;
    const int rel_dim = top.num_heads * top.d_rel;
    const int max_kv_dim =
        std::max(top.num_kv_heads, top.swa_kv_heads) * top.head_dim;
    const int I = top.moe_intermediate;
    const int DI = top.dense_intermediate;
    const int topk = top.top_k;
    const int maxch = std::max(H, max_kv_dim);
    const int max_inter = std::max(I, DI);

    int64_t off = 0;
    auto take = [&off](int64_t bytes) {
        off = (off + 255) & ~static_cast<int64_t>(255);
        const int64_t at = off;
        off += bytes;
        return at;
    };
    const int64_t o_xh = take(int64_t(H) * 2);
    const int64_t o_xf = take(int64_t(H) * 4);
    const int64_t o_hh = take(int64_t(H) * 2);
    const int64_t o_q = take(int64_t(q_dim) * 2);
    const int64_t o_kv = take(int64_t(max_kv_dim) * 2);
    const int64_t o_vv = take(int64_t(max_kv_dim) * 2);
    const int64_t o_qf = take(int64_t(q_dim) * 4);
    const int64_t o_rel = take(int64_t(rel_dim) * 4);
    const int64_t o_attn = take(int64_t(q_dim) * 2);
    const int64_t o_oo = take(int64_t(H) * 2);
    const int64_t o_conv = take(int64_t(maxch) * 2);
    const int64_t o_conff = take(int64_t(H) * 4);
    const int64_t o_conv32 = take(int64_t(H) * 4);
    const int64_t o_moef32 = take(int64_t(H) * 4);
    const int64_t o_res = take(int64_t(H) * 4);
    // Sized by the engine bound (not the session capacity): the bias gather
    // must never outrun this buffer.
    const int64_t o_bias = take(int64_t(top.num_heads) * kMaxContext * 4);
    const int64_t o_mg = take(int64_t(topk) * max_inter * 2);
    const int64_t o_mu = take(int64_t(topk) * max_inter * 2);
    const int64_t o_ma = take(int64_t(topk) * max_inter * 2);
    const int64_t o_md = take(int64_t(topk) * H * 2);
    const int64_t o_md32 = take(int64_t(topk) * H * 4);
    const int64_t o_macc = take(int64_t(H) * 4);
    const int64_t o_sg = take(int64_t(I) * 2);
    const int64_t o_su = take(int64_t(I) * 2);
    const int64_t o_sa = take(int64_t(I) * 2);
    const int64_t o_sd = take(int64_t(H) * 2);
    const int64_t o_sd32 = take(int64_t(H) * 4);
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
    hh = reinterpret_cast<__half *>(at(o_hh));
    q = reinterpret_cast<__half *>(at(o_q));
    kv_key = reinterpret_cast<__half *>(at(o_kv));
    kv_val = reinterpret_cast<__half *>(at(o_vv));
    qf = reinterpret_cast<float *>(at(o_qf));
    relf = reinterpret_cast<float *>(at(o_rel));
    attn_o = reinterpret_cast<__half *>(at(o_attn));
    o_out = reinterpret_cast<__half *>(at(o_oo));
    conv_out = reinterpret_cast<__half *>(at(o_conv));
    conv_f = reinterpret_cast<float *>(at(o_conff));
    conv_f32 = reinterpret_cast<float *>(at(o_conv32));
    moe_f32 = reinterpret_cast<float *>(at(o_moef32));
    residual = reinterpret_cast<float *>(at(o_res));
    rel_bias = reinterpret_cast<float *>(at(o_bias));
    moe_gate = reinterpret_cast<__half *>(at(o_mg));
    moe_up = reinterpret_cast<__half *>(at(o_mu));
    moe_act = reinterpret_cast<__half *>(at(o_ma));
    moe_down = reinterpret_cast<__half *>(at(o_md));
    moe_down_f32 = reinterpret_cast<float *>(at(o_md32));
    moe_acc = reinterpret_cast<float *>(at(o_macc));
    sh_gate = reinterpret_cast<__half *>(at(o_sg));
    sh_up = reinterpret_cast<__half *>(at(o_su));
    sh_act = reinterpret_cast<__half *>(at(o_sa));
    sh_down = reinterpret_cast<__half *>(at(o_sd));
    sh_down_f32 = reinterpret_cast<float *>(at(o_sd32));
    logits = reinterpret_cast<float *>(at(o_logits));

    // Expert staging: gate/up are [I, H] and down is [H, I], each paired with
    // a per-16 E4M3 scale plane. One contiguous slab, six planes per expert.
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
    if (cudaMalloc(&d_slots, sizeof(RoutedSlot) * topk) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_gammas, sizeof(float) * std::max(1, top.shared_experts)) !=
        cudaSuccess)
        return false;
    if (cudaMalloc(&d_token, sizeof(int32_t)) != cudaSuccess) return false;
    if (cudaMalloc(&d_maxv, sizeof(float)) != cudaSuccess) return false;
    return true;
}

cudaError_t RunForwardTrace(cudaStream_t stream, PinnedWeights &pw,
                            ExpertDb &experts, Session &session,
                            Scratch &scratch,
                            const std::vector<int32_t> &tokens,
                            int unpadded_vocab, std::vector<float> &out_states,
                            std::vector<float> *out_logits)
{
    if (tokens.empty()) return cudaErrorInvalidValue;
    const Topology &top = pw.top;
    cudaError_t e =
        session.EnsureCapacity(top, static_cast<int64_t>(tokens.size()), stream);
    if (e != cudaSuccess) return e;
    if (scratch.arena == nullptr && !scratch.Allocate(top, stream))
        return cudaErrorMemoryAllocation;
    // A trace is a fresh sequence: zero the conv rings.
    session.Reset();
    for (auto &lw : pw.layers)
        if (lw.conv_state != nullptr && lw.conv_state_bytes > 0)
            cudaMemsetAsync(lw.conv_state, 0,
                            static_cast<size_t>(lw.conv_state_bytes), stream);
    std::string err;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const bool last = (i + 1 == tokens.size());
        e = Forward(stream, pw, experts, session, scratch, tokens[i],
                    static_cast<int>(i), unpadded_vocab, err,
                    last ? &out_states : nullptr);
        if (e != cudaSuccess) return e;
    }
    if (out_logits != nullptr) {
        const int n = (unpadded_vocab > 0 && unpadded_vocab < pw.embed_rows)
                          ? unpadded_vocab
                          : pw.embed_rows;
        out_logits->assign(static_cast<size_t>(n), 0.0f);
        cudaStreamSynchronize(stream);
        e = cudaMemcpy(out_logits->data(), scratch.logits,
                       out_logits->size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

cudaError_t RunGeneration(cudaStream_t stream, PinnedWeights &pw,
                          ExpertDb &experts, Session &session, Scratch &scratch,
                          const std::vector<int32_t> &input_ids,
                          int max_gen_tokens, const std::vector<int32_t> &eos_ids,
                          int unpadded_vocab, sample::ISampler &sampler,
                          GenerationResult &out, const GenerationStream *gs)
{
    out = GenerationResult{};
    if (input_ids.empty()) {
        out.error_msg = "inkling: empty prompt";
        return cudaErrorInvalidValue;
    }
    const Topology &top = pw.top;
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
        out.error_msg =
            absl::StrFormat("inkling: KV allocation failed (cuda %d)",
                            static_cast<int>(e));
        return e;
    }
    if (scratch.arena == nullptr && !scratch.Allocate(top, stream)) {
        out.error_msg = "inkling: decode scratch allocation failed";
        return cudaErrorMemoryAllocation;
    }
    const int vocab = (unpadded_vocab > 0 && unpadded_vocab < pw.embed_rows)
                          ? unpadded_vocab
                          : pw.embed_rows;

    // A new request is a new sequence: clear the recurrent short-convolution
    // state of every layer and the KV bookkeeping. Skipping this makes the
    // second identical request decode differently from the first.
    session.Reset();
    for (auto &lw : pw.layers)
        if (lw.conv_state != nullptr && lw.conv_state_bytes > 0) {
            e = cudaMemsetAsync(lw.conv_state, 0,
                                static_cast<size_t>(lw.conv_state_bytes),
                                stream);
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
                    unpadded_vocab, err);
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
        e = sampler.Sample(stream, scratch.logits, vocab, sample::LogitType::kF32,
                           scratch.d_token, scratch.d_maxv);
        if (e != cudaSuccess) {
            out.error_msg =
                absl::StrFormat("inkling: sampler (cuda %d)",
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
            out.error_msg =
                absl::StrFormat("inkling: sampler returned id %d", tok);
            return cudaErrorInvalidValue;
        }
        // The row that produced THIS token is the recorded logits.
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
        if (gs != nullptr)
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(step), maxv);
        bool stop = false;
        if (want_eos)
            for (int32_t id : eos_ids)
                if (id == tok) stop = true;
        if (stop) {
            out.eos_terminated = true;
            break;
        }
        if (step + 1 >= max_gen_tokens) break;
        // Feed the token back so the NEXT iteration has its logits row.
        e = Forward(stream, pw, experts, session, scratch, tok, pos,
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

}  // namespace lsxcommon::inkling::base