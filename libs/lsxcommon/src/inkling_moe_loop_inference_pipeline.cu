// Inkling-2-Mini generation pipeline (Model Ingestion Protocol stage 5).
//
// The loop scheduling is the delta over Inkling-2-Small; the per-layer
// arithmetic (rel-bias attention, clips, convs, depth mix, softcapped SwiGLU
// MoE) is the same contract and runs through the same geometry-parameterized
// kernels. Per token:
//
//   embed -> embed_norm
//   pre-loop layers 0..loop_start-1 (inline, growing block stack)
//   snapshot (hidden, block stack) at loop entry
//   pass 0: loop layers loop_start..loop_end-1 (inline, messages = none)
//   pass p=1..passes-1:
//     restore the pre-loop snapshots; the message sources are the PREVIOUS
//     pass's block stack (skipping its first appended state) + final hidden;
//     per loop layer: retrieve 4 messages (message_queries), gate q/k/v/mlp
//     inputs through the MessageGates, run the layer into cache slot
//     CacheSlot(layer, p)
//   post-loop layers loop_end..num_layers-1
//   final mix (output_query) -> norm -> unembed -> softcap -> sample
//
// Every pass owns its own KV rows and conv rings (session slots), so pass p
// never reads pass p-1's recurrent state.
#include "lsxcommon/inkling_moe_loop_inference_pipeline.h"

#include "inkling_moe_loop_constants.h"
#include "lsxcommon/inkling_moe_kernels.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"

#include <absl/strings/str_format.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <utility>

namespace lsxcommon::inkling {

std::int64_t InklingLoopSession::Bytes() const
{
    if (!d_k) return 0;
    return 2 * static_cast<int64_t>(geom.layers) * geom.LayerStride() *
               static_cast<int64_t>(sizeof(__half)) +
           conv_state_bytes +
           static_cast<int64_t>(gen_tail.capacity()) * 4;
}

void InklingLoopSession::Free()
{
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    d_k = nullptr;
    d_v = nullptr;
    if (conv_state) cudaFree(conv_state);
    conv_state = nullptr;
    conv_slot_floats = 0;
    conv_state_bytes = 0;
    geom = {};
    capacity = 0;
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

void InklingLoopSession::Reset(cudaStream_t stream)
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
    if (conv_state != nullptr && conv_state_bytes > 0)
        cudaMemsetAsync(conv_state, 0, static_cast<size_t>(conv_state_bytes),
                        stream);
}

cudaError_t InklingLoopSession::EnsureCapacity(const InklingLoopTopology &top,
                                               int64_t want,
                                               cudaStream_t stream)
{
    if (want <= capacity && d_k) return cudaSuccess;
    const int64_t cap = std::max<int64_t>(want, capacity * 2);
    // One geometry for every CACHE SLOT (physical layers + extra loop passes):
    // pos-major, the measured default. The sliding layers pay for positions
    // they never read; that cost is reported, not hidden.
    const KvCacheGeometry g = KvCacheGeometry::Make(
        KvCacheLayout::kPosMajor, cap, top.num_kv_heads, top.head_dim,
        top.CacheLayers());
    const int64_t elems = g.Elements();
    __half *nk = nullptr;
    __half *nv = nullptr;
    if (cudaMalloc(&nk, static_cast<size_t>(elems) * sizeof(__half)) !=
        cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(&nv, static_cast<size_t>(elems) * sizeof(__half)) !=
        cudaSuccess) {
        cudaFree(nk);
        return cudaGetLastError();
    }
    if (d_k && kv_rows > 0) {
        // The rows of one slot are contiguous: kv_rows * pos_stride of them.
        // `kv_rows * LayerStride()` would over-read by capacity/kv_rows (the
        // Small defect: cudaErrorInvalidValue surfacing at the NEXT request).
        const size_t bytes = static_cast<size_t>(kv_rows) *
                             static_cast<size_t>(geom.pos_stride) *
                             sizeof(__half);
        for (int l = 0; l < static_cast<int>(g.layers); ++l) {
            cudaError_t ce = cudaMemcpyAsync(
                nk + l * g.LayerStride(), d_k + l * geom.LayerStride(), bytes,
                cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) {
                cudaFree(nk);
                cudaFree(nv);
                return ce;
            }
            ce = cudaMemcpyAsync(nv + l * g.LayerStride(),
                                 d_v + l * geom.LayerStride(), bytes,
                                 cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) {
                cudaFree(nk);
                cudaFree(nv);
                return ce;
            }
        }
        const cudaError_t se = cudaStreamSynchronize(stream);
        if (se != cudaSuccess) {
            cudaFree(nk);
            cudaFree(nv);
            return se;
        }
    }
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    d_k = nk;
    d_v = nv;
    geom = g;
    capacity = cap;

    // Conv rings: one per cache slot, allocated once at the full slot count.
    const int64_t kv_dim =
        static_cast<int64_t>(top.num_kv_heads) * top.head_dim;
    const int64_t slot_floats = 2 * static_cast<int64_t>(top.conv_kernel) *
                                    kv_dim +
                                2 * static_cast<int64_t>(top.conv_kernel) *
                                    top.hidden;
    const int64_t total_bytes =
        static_cast<int64_t>(top.CacheLayers()) * slot_floats *
        static_cast<int64_t>(sizeof(float));
    if (conv_state == nullptr || conv_state_bytes != total_bytes) {
        if (conv_state) cudaFree(conv_state);
        conv_state = nullptr;
        if (cudaMalloc(&conv_state, static_cast<size_t>(total_bytes)) !=
            cudaSuccess)
            return cudaGetLastError();
        conv_slot_floats = slot_floats;
        conv_state_bytes = total_bytes;
        if (cudaMemsetAsync(conv_state, 0, static_cast<size_t>(total_bytes),
                            stream) != cudaSuccess)
            return cudaGetLastError();
    }
    return cudaSuccess;
}

void InklingLoopScratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    if (expert_stage) cudaFree(expert_stage);
    expert_stage = nullptr;
    expert_stage_bytes = 0;
    if (fp8_stage) cudaFree(fp8_stage);
    fp8_stage = nullptr;
    fp8_stage_bytes = 0;
    if (d_ids) cudaFree(d_ids);
    d_ids = nullptr;
    if (d_slots) cudaFree(d_slots);
    d_slots = nullptr;
    if (d_token) cudaFree(d_token);
    d_token = nullptr;
    if (d_maxv) cudaFree(d_maxv);
    d_maxv = nullptr;
    pre_loop_hidden = nullptr;
    pre_loop_stack = nullptr;
    prev_stack = nullptr;
    prev_hidden = nullptr;
    msg = nullptr;
    gate_h = nullptr;
    gate_rank = nullptr;
    gate_out = nullptr;
}

bool InklingLoopScratch::Allocate(const InklingLoopTopology &top,
                                  int64_t capacity, cudaStream_t stream)
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
    const int max_sources = top.attn_res_block > 0
                                ? top.num_layers / top.attn_res_block + 1
                                : top.num_layers + 1;

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
    // Sized by the engine bound (not `capacity`): the KV session grows lazily
    // and the bias gather must never outrun this buffer.
    const int64_t o_bias = take(int64_t(top.num_heads) * kLoopMaxContext * 4);
    (void)capacity;
    // TWO arrays each (k and v): deriving the second from base + one_array
    // silently overruns into the next region (the Small arena-overlap defect).
    const int64_t o_taps = take(2 * int64_t(kv_dim) * k * 4);
    const int64_t o_proj = take(2 * int64_t(top.num_kv_heads) * k * 4);
    const int64_t o_stack = take(int64_t(max_sources) * H * 4);
    const int64_t o_mg = take(int64_t(topk) * I * 2);
    const int64_t o_mu = take(int64_t(topk) * I * 2);
    const int64_t o_ma = take(int64_t(topk) * I * 2);
    const int64_t o_md = take(int64_t(topk) * H * 2);
    const int64_t o_macc = take(int64_t(H) * 4);
    const int64_t o_logits = take(int64_t(top.vocab) * 4);
    // ── loop machinery ────────────────────────────────────────────────────
    const int64_t o_plh = take(int64_t(H) * 4);
    const int64_t o_pls = take(int64_t(max_sources) * H * 4);
    const int64_t o_pvs = take(int64_t(max_sources) * H * 4);
    const int64_t o_pvh = take(int64_t(H) * 4);
    const int64_t o_msg = take(int64_t(4) * H * 4);
    const int64_t o_gh = take(int64_t(2) * H * 4);
    const int64_t o_gr = take(int64_t(top.loop_gate_rank) * 4);
    const int64_t o_go = take(int64_t(H) * 4);
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
    pre_loop_hidden = reinterpret_cast<float *>(at(o_plh));
    pre_loop_stack = reinterpret_cast<float *>(at(o_pls));
    prev_stack = reinterpret_cast<float *>(at(o_pvs));
    prev_hidden = reinterpret_cast<float *>(at(o_pvh));
    msg = reinterpret_cast<float *>(at(o_msg));
    gate_h = reinterpret_cast<float *>(at(o_gh));
    gate_rank = reinterpret_cast<float *>(at(o_gr));
    gate_out = reinterpret_cast<float *>(at(o_go));

    // Expert staging: gate/up [I, H] + down [H, I], each with a per-16 E4M3
    // scale plane; six planes per expert, one contiguous slab.
    const int64_t w1 = int64_t(I) * H / 2;
    const int64_t s1 = int64_t(I) * H / 16;
    const int64_t w2 = int64_t(H) * I / 2;
    const int64_t s2 = int64_t(H) * I / 16;
    const int64_t per = w1 + s1 + w1 + s1 + w2 + s2;
    expert_stage_bytes = per * topk;
    if (cudaMalloc(&expert_stage, static_cast<size_t>(expert_stage_bytes)) !=
        cudaSuccess)
        return false;
    bool any_fp8 = false;
    for (bool b : top.expert_fp8) any_fp8 = any_fp8 || b;
    if (any_fp8) {
        const int64_t gate = int64_t(I) * H;
        const int64_t gate_s = int64_t(I) * (H / 16);
        const int64_t down = int64_t(H) * I;
        const int64_t down_s = int64_t(H) * (I / 16);
        fp8_stage_bytes = (2 * (gate + gate_s) + down + down_s) * topk;
        if (cudaMalloc(&fp8_stage, static_cast<size_t>(fp8_stage_bytes)) !=
            cudaSuccess)
            return false;
    }
    if (cudaMalloc(&d_ids, sizeof(int32_t) * topk) != cudaSuccess) return false;
    if (cudaMalloc(&d_slots, sizeof(InklingRoutedSlot) * topk) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_token, sizeof(int32_t)) != cudaSuccess) return false;
    if (cudaMalloc(&d_maxv, sizeof(float)) != cudaSuccess) return false;
    return true;
}

namespace {

using steady = std::chrono::steady_clock;

std::string LayerTag(const char *what, int layer)
{
    return absl::StrFormat("%s l%d", what, layer);
}

// tau = 1 + alpha * log(max((pos + 1) / floor, 1)), the full layers' log
// scaling of q and the relative states. Mini's floor is 8192 (Small's 128000
// would leave the whole 8k context unscaled).
float LoopLogScaleTau(const InklingLoopTopology &top, int pos)
{
    if (top.log_scaling_n_floor <= 0) return 1.0f;
    const double effective = static_cast<double>(pos) + 1.0;
    const double ratio = std::max(
        effective / static_cast<double>(top.log_scaling_n_floor), 1.0);
    return static_cast<float>(1.0 + top.log_scaling_alpha * std::log(ratio));
}

// One dense projection of a single token row.
cudaError_t Proj(cudaStream_t stream, const InklingLoopDenseF16 &w,
                 const __half *x, __half *y)
{
    if (w.d == nullptr) return cudaErrorInvalidValue;
    return RunInklingMatVecF16ToHalf(stream, w.d, x, y, w.rows, w.cols);
}

// The staged top-k experts of one layer: three NVFP4 pointer planes plus the
// two global-scale classes the checkpoint carries (gate/up share w13's amax).
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
// pointer planes the batch GEMV consumes. A missing plane is a hard error: a
// half-staged expert multiplies garbage and would surface as text.
cudaError_t StageExperts(InklingLoopExpertDb &db, int layer, const int32_t *ids,
                         int count, InklingLoopScratch &ws, int I, int H,
                         cudaStream_t stream, ExpertStage &out,
                         std::string &err)
{
    const int64_t w1 = int64_t(I) * H / 2;
    const int64_t s1 = int64_t(I) * H / 16;
    const int64_t w2 = int64_t(H) * I / 2;
    const int64_t s2 = int64_t(H) * I / 16;
    const int64_t per = w1 + s1 + w1 + s1 + w2 + s2;
    if (per * count > ws.expert_stage_bytes) {
        err = absl::StrFormat(
            "expert staging slab too small (need %lld, have %lld)",
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
        const InklingLoopExpertEntry &e =
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

// The four conv rings of ONE cache slot: [k][kv_dim] for k, [k][kv_dim] for v,
// then [k][hidden] for the post-attention and post-MLP convolutions. The
// session owns one set per cache slot; the loader's copy is only the pass-0
// template, so nothing here reads `lw.conv_state`.
float *ConvSlotBase(const InklingLoopSession &session,
                    const InklingLoopTopology &top, int layer, int pass)
{
    return session.ConvSlot(top.CacheSlot(layer, pass));
}

// ── one layer application ──────────────────────────────────────────────────
// Runs ONE application of physical layer `l` (depth mix -> attention ->
// residual/block stack -> MLP mixture -> MoE), reading and writing the residual
// stream in `ws.x_h` and appending to the block stack in `ws.stack`
// (`n_blocks` is in/out; stack[n_blocks] is the live partial). `pass` selects
// the KV + conv cache slot. `messages` (4 x hidden FP32, order q/k/v/mlp) is
// non-null only for the loop layers of passes >= 1, where it is blended into the
// q/k/v inputs before their norms and into the MLP input before post_attention
// norm.
cudaError_t RunLoopLayer(cudaStream_t stream, InklingLoopPinnedWeights &pw,
                         InklingLoopExpertDb &experts,
                         InklingLoopSession &session, InklingLoopScratch &ws,
                         int l, int pass, int pos, const float *messages,
                         int &n_blocks, std::string &stage_err)
{
    const InklingLoopTopology &top = pw.top;
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
    const int gate_rank = top.loop_gate_rank;

    InklingLoopLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
    const bool full = lw.full;
    const bool boundary = top.IsBlockBoundary(l);
    // Layer 0 has no attention_input in the reference: its three streams ARE the
    // layer input, so the bank holds no query rows for it.
    const int nq = (full && l > 0) ? 3 : 1;
    float *const conv_base = ConvSlotBase(session, top, l, pass);

    std::string stage = "layer_in";
    auto fail = [&](const char *why, cudaError_t e) {
        stage_err = absl::StrFormat("[%s] %s (cuda %d)", stage, why,
                                    static_cast<int>(e));
        return e;
    };
    cudaError_t e = cudaSuccess;

    // ── (a) attention-input depth mixture ──────────────────────────────────
    stage = LayerTag("attn_res", l);
    e = RunInklingHalfToF32(stream, ws.x_h, ws.x_f, H);
    if (e != cudaSuccess) return fail("layer input to f32", e);
    if (n_blocks < 0 || n_blocks >= loop::frozen::kAttnResMaxSources) {
        stage_err = absl::StrFormat("[%s] AttnRes stack overflow (%d)", stage,
                                    n_blocks);
        return cudaErrorInvalidValue;
    }
    e = cudaMemcpyAsync(ws.stack + static_cast<int64_t>(n_blocks) * H, ws.x_f,
                        sizeof(float) * H, cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return fail("live slot", e);
    const float *attn_input = ws.x_f;
    if (l > 0) {
        const float *attn_q =
            pw.attn_queries + static_cast<int64_t>(pw.attn_query_offset[l]) * H;
        e = RunInklingDepthMix(stream, ws.stack, n_blocks + 1, attn_q, nq, ws.mix,
                               H, eps);
        if (e != cudaSuccess) return fail("attention depth mix", e);
        attn_input = ws.mix;
    }

    // ── (b) input norms (+ message gates) and projections ──────────────────
    // The gates run on the MIXED input and BEFORE the input norm, and the three
    // attention streams are gated separately: a gate applied after the norm (or
    // once, shared) is a silent parity defect that still produces text.
    stage = LayerTag("attn_proj", l);
    const __half *qsrc = nullptr;
    const __half *ksrc = nullptr;
    const __half *vsrc = nullptr;
    if (messages == nullptr && !full) {
        // A sliding layer without messages normalises ONE stream and feeds it to
        // q, k and v. (Writing the norm after a `break` here — or skipping it —
        // leaves q/k/v pointing at stale scratch: the layer then "runs" with a
        // zero key/value, answers confidently and never fails.)
        e = RunInklingNormF32In(stream, attn_input, lw.attn_norm, ws.qh, 1, H,
                                eps);
        if (e != cudaSuccess) return fail("attn norm", e);
        qsrc = ksrc = vsrc = ws.qh;
    } else {
        for (int s = 0; s < 3; ++s) {
            const float *row = (nq >= 3)
                                   ? attn_input + static_cast<int64_t>(s) * H
                                   : attn_input;
            if (messages != nullptr) {
                const InklingLoopMessageGate &mg = lw.msg[s];
                if (mg.down.d == nullptr || mg.forget == nullptr ||
                    mg.gain == nullptr) {
                    stage_err = absl::StrFormat(
                        "[%s] layer %d has no message gate %d", stage, l, s);
                    return cudaErrorInvalidValue;
                }
                e = RunInklingLoopMessageGate(
                    stream, row, messages + static_cast<int64_t>(s) * H,
                    mg.down.d, mg.forget, mg.gain, ws.gate_out, H, gate_rank,
                    eps, ws.gate_h, ws.gate_rank);
                if (e != cudaSuccess) return fail("message gate", e);
                row = ws.gate_out;
            }
            const __half *norm_w = full ? (s == 0   ? lw.q_in_norm
                                           : s == 1 ? lw.k_in_norm
                                                    : lw.v_in_norm)
                                        : lw.attn_norm;
            __half *dst = s == 0 ? ws.qh : (s == 1 ? ws.kh : ws.vh);
            e = RunInklingNormF32In(stream, row, norm_w, dst, 1, H, eps);
            if (e != cudaSuccess) return fail("input norm", e);
        }
        qsrc = ws.qh;
        ksrc = ws.kh;
        vsrc = ws.vh;
    }

    e = Proj(stream, lw.wq, qsrc, ws.q);
    if (e != cudaSuccess) return fail("wq", e);
    e = Proj(stream, lw.wk, ksrc, ws.kv_key);
    if (e != cudaSuccess) return fail("wk", e);
    e = Proj(stream, lw.wv, vsrc, ws.kv_val);
    if (e != cudaSuccess) return fail("wv", e);
    // The relative states ([heads*d_rel]) get their OWN buffer: `conv_out` is
    // recycled by the k/v convolutions below, so parking them there reads a
    // convolution result as the position bias.
    e = Proj(stream, lw.wr, qsrc, ws.o_out);
    if (e != cudaSuccess) return fail("wr", e);
    e = Proj(stream, lw.attn_gate, qsrc, ws.attn_gate);
    if (e != cudaSuccess) return fail("attn gate", e);
    e = RunInklingMatVecF16(stream, lw.sink_proj, qsrc, ws.sink, heads, H);
    if (e != cudaSuccess) return fail("sink proj", e);

    // ── (c) k/v short convolutions on the RAW projections ──────────────────
    // The reference convolves k_proj(x) / v_proj(x) and only then clips k and
    // normalises v; inverting that order is silent.
    stage = LayerTag("kv_conv", l);
    for (int half = 0; half < 2; ++half) {
        const __half *raw = half == 0 ? ws.kv_key : ws.kv_val;
        float *conv_state = conv_base + half * kconv * kv_dim;
        const float *taps = nullptr;
        float inv = inv_static;
        if (full) {
            const InklingLoopDenseF16 &proj_w =
                half == 0 ? lw.ksconv_proj : lw.vsconv_proj;
            float *proj_out = half == 0 ? ws.proj_k : ws.proj_v;
            float *dyn = half == 0 ? ws.taps_k : ws.taps_v;
            const float *static_w =
                half == 0 ? lw.kconv_static : lw.vconv_static;
            e = Proj(stream, proj_w, half == 0 ? ksrc : vsrc, ws.conv_out);
            if (e != cudaSuccess)
                return fail(half == 0 ? "k conv proj" : "v conv proj", e);
            e = RunInklingHalfToF32(stream, ws.conv_out, proj_out,
                                    kv_heads * kconv);
            if (e != cudaSuccess) return fail("conv proj to f32", e);
            e = RunInklingDynamicTaps(stream, static_w, proj_out, dyn, kv_dim, hd,
                                      kconv);
            if (e != cudaSuccess) return fail("dynamic taps", e);
            taps = dyn;
            inv = inv_dyn;
        } else {
            taps = half == 0 ? lw.kconv_w : lw.vconv_w;
        }
        e = cudaMemcpyAsync(ws.conv_out, raw, sizeof(__half) * kv_dim,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return fail("conv input stage", e);
        e = full ? RunInklingShortConv(stream, ws.conv_out, conv_state, nullptr,
                                       taps, ws.conv_out, kv_dim, kconv, inv)
                 : RunInklingShortConv(stream, ws.conv_out, conv_state, taps,
                                       nullptr, ws.conv_out, kv_dim, kconv, inv);
        if (e != cudaSuccess) return fail("short conv", e);
        e = RunInklingHalfToF32(stream, ws.conv_out,
                                half == 0 ? ws.kf : ws.vf, kv_dim);
        if (e != cudaSuccess) return fail("conv out to f32", e);
    }

    // ── (d) clips, value norm, log scaling ────────────────────────────────
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
        const float tau = LoopLogScaleTau(top, pos);
        e = RunInklingLogScale(stream, ws.qf, ws.relf, q_dim, rel_dim, tau);
        if (e != cudaSuccess) return fail("log scale", e);
    }
    e = RunInklingScaleToHalf(stream, ws.qf, ws.q, q_dim, 1.0f);
    if (e != cudaSuccess) return fail("q store", e);
    e = RunInklingScaleToHalf(stream, ws.kf, ws.kv_key, kv_dim, 1.0f);
    if (e != cudaSuccess) return fail("k store", e);
    e = RunInklingScaleToHalf(stream, ws.vf, ws.kv_val, kv_dim, 1.0f);
    if (e != cudaSuccess) return fail("v store", e);

    // ── (e) KV append into this pass's slot, relative bias, attention ──────
    stage = LayerTag("attention", l);
    const int64_t slot = static_cast<int64_t>(top.CacheSlot(l, pass));
    __half *d_k = session.d_k + slot * session.geom.LayerStride();
    __half *d_v = session.d_v + slot * session.geom.LayerStride();
    e = RunInklingKvAppend(stream, ws.kv_key, ws.kv_val, d_k, d_v, session.geom,
                           pos);
    if (e != cudaSuccess) return fail("kv append", e);
    const int rel_extent = full ? top.rel_extent : top.sliding_window;
    e = RunInklingRelBias(stream, ws.relf, lw.rel_proj, ws.rel_bias, heads,
                          top.d_rel, rel_extent, pos,
                          pos + 1);
    if (e != cudaSuccess) return fail("rel bias", e);
    e = RunInklingAttention(stream, ws.q, d_k, d_v, session.geom, heads,
                            pos, ws.sink, ws.rel_bias,
                            full ? 0 : top.sliding_window, scale, ws.attn_o);
    if (e != cudaSuccess) return fail("attention", e);

    stage = LayerTag("attn_out", l);
    e = RunInklingGateMul(stream, ws.attn_o, ws.attn_gate, ws.attn_o, q_dim);
    if (e != cudaSuccess) return fail("output gate", e);
    e = Proj(stream, lw.wo, ws.attn_o, ws.o_out);
    if (e != cudaSuccess) return fail("wo", e);
    e = RunInklingShortConv(stream, ws.o_out, conv_base + 2 * kconv * kv_dim,
                            lw.attn_conv_w, nullptr, ws.conv_out, H, kconv,
                            inv_static);
    if (e != cudaSuccess) return fail("post-attention conv", e);

    // ── (f) residual / block stack, then the MLP mixture ───────────────────
    stage = LayerTag("residual", l);
    e = RunInklingHalfToF32(stream, ws.conv_out, ws.residual, H);
    if (e != cudaSuccess) return fail("post-attn to f32", e);
    if (!boundary) {
        e = RunInklingAddF32(stream, ws.residual, ws.x_f, H);
        if (e != cudaSuccess) return fail("attention residual add", e);
    } else {
        // The layer INPUT (already parked at stack[n_blocks]) becomes a real
        // block source; the residual is the attention output alone.
        ++n_blocks;
        if (n_blocks >= loop::frozen::kAttnResMaxSources) {
            stage_err = absl::StrFormat("[%s] AttnRes stack overflow (%d)",
                                        stage, n_blocks);
            return cudaErrorInvalidValue;
        }
    }
    e = cudaMemcpyAsync(ws.stack + static_cast<int64_t>(n_blocks) * H,
                        ws.residual, sizeof(float) * H, cudaMemcpyDeviceToDevice,
                        stream);
    if (e != cudaSuccess) return fail("mlp partial", e);
    {
        const float *mlp_q = pw.mlp_queries + static_cast<int64_t>(l) * H;
        e = RunInklingDepthMix(stream, ws.stack, n_blocks + 1, mlp_q, 1, ws.mix, H,
                               eps);
        if (e != cudaSuccess) return fail("mlp depth mix", e);
    }
    const float *mlp_in = ws.mix;
    if (messages != nullptr) {
        const InklingLoopMessageGate &mg = lw.msg[3];
        if (mg.down.d == nullptr || mg.forget == nullptr || mg.gain == nullptr) {
            stage_err = absl::StrFormat("[%s] layer %d has no mlp message gate",
                                        stage, l);
            return cudaErrorInvalidValue;
        }
        e = RunInklingLoopMessageGate(stream, ws.mix,
                                      messages + static_cast<int64_t>(3) * H,
                                      mg.down.d, mg.forget, mg.gain, ws.gate_out,
                                      H, gate_rank, eps, ws.gate_h, ws.gate_rank);
        if (e != cudaSuccess) return fail("mlp message gate", e);
        mlp_in = ws.gate_out;
    }
    e = RunInklingNormF32In(stream, mlp_in, lw.mlp_norm, ws.hh, 1, H, eps);
    if (e != cudaSuccess) return fail("mlp_norm", e);

    // ── (g) router + routed experts ────────────────────────────────────────
    stage = LayerTag("moe", l);
    e = RunInklingRouter(stream, lw.router_w, lw.router_bias, ws.hh,
                         top.num_experts, topk, H,
                         static_cast<float>(top.route_scale), ws.d_slots,
                         ws.d_ids);
    if (e != cudaSuccess) return fail("router", e);
    InklingRoutedSlot slots[8];
    int32_t ids[8];
    if (topk > 8) {
        stage_err = absl::StrFormat("[%s] top_k %d exceeds the 8-slot stack",
                                    stage, topk);
        return cudaErrorInvalidValue;
    }
    e = cudaMemcpyAsync(slots, ws.d_slots, sizeof(InklingRoutedSlot) * topk,
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return fail("slots download", e);
    e = cudaMemcpyAsync(ids, ws.d_ids, sizeof(int32_t) * topk,
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return fail("ids download", e);
    e = cudaStreamSynchronize(stream);
    if (e != cudaSuccess) return fail("router sync", e);

    if (top.ExpertIsFp8(l)) {
        // The release's MXFP8 plane, read DIRECTLY: the expert's E4M3 bytes and
        // the duplicated ue8m0 exponents are staged to the device (the store
        // hands out mmap pointers, which a kernel cannot dereference) and the
        // FP8 GEMV decodes them in place.
        const int64_t gate = int64_t(I) * H;
        const int64_t gate_s = int64_t(I) * (H / 16);
        const int64_t down = int64_t(H) * I;
        const int64_t down_s = int64_t(H) * (I / 16);
        const int64_t per = 2 * (gate + gate_s) + down + down_s;
        if (per * topk > ws.fp8_stage_bytes) {
            stage_err = absl::StrFormat(
                "[%s] FP8 staging slab too small (need %lld, have %lld)", stage,
                static_cast<long long>(per * topk),
                static_cast<long long>(ws.fp8_stage_bytes));
            return cudaErrorInvalidValue;
        }
        InklingFp8ExpertPtrs gate_ptrs, up_ptrs, down_ptrs;
        for (int k = 0; k < topk; ++k) {
            const int eid = ids[k];
            if (eid < 0 || eid >= top.num_experts) {
                stage_err =
                    absl::StrFormat("[%s] router returned id %d", stage, eid);
                return cudaErrorInvalidValue;
            }
            experts.EnsureResident(l, eid, stream);
            const InklingLoopExpertEntry &ent =
                experts.entries[static_cast<size_t>(l) *
                                    static_cast<size_t>(top.num_experts) +
                                static_cast<size_t>(eid)];
            if (ent.host_w1 == nullptr || ent.host_w2 == nullptr ||
                ent.host_w3 == nullptr || ent.host_w1_scales == nullptr ||
                ent.host_w2_scales == nullptr ||
                ent.host_w3_scales == nullptr) {
                stage_err = absl::StrFormat(
                    "[%s] routed expert %d of layer %d is not readable", stage,
                    eid, l);
                return cudaErrorInvalidValue;
            }
            if (ent.w1_bytes != gate || ent.w3_bytes != gate ||
                ent.w2_bytes != down || ent.w1_scales_bytes != gate_s ||
                ent.w3_scales_bytes != gate_s || ent.w2_scales_bytes != down_s) {
                stage_err = absl::StrFormat(
                    "[%s] expert %d of layer %d has no FP8 plane "
                    "(w1 %lld/%lld, scales %lld/%lld)",
                    stage, eid, l, static_cast<long long>(ent.w1_bytes),
                    static_cast<long long>(gate),
                    static_cast<long long>(ent.w1_scales_bytes),
                    static_cast<long long>(gate_s));
                return cudaErrorInvalidValue;
            }
            uint8_t *dst = ws.fp8_stage + per * k;
            int64_t at = 0;
            auto put = [&](const uint8_t *src, int64_t n) {
                if (at + n > per) return static_cast<const uint8_t *>(nullptr);
                cudaMemcpyAsync(dst + at, src, static_cast<size_t>(n),
                                cudaMemcpyHostToDevice, stream);
                const uint8_t *dev = dst + at;
                at += n;
                return dev;
            };
            gate_ptrs.packed[k] = put(ent.host_w1, ent.w1_bytes);
            gate_ptrs.scales[k] = put(ent.host_w1_scales, ent.w1_scales_bytes);
            up_ptrs.packed[k] = put(ent.host_w3, ent.w3_bytes);
            up_ptrs.scales[k] = put(ent.host_w3_scales, ent.w3_scales_bytes);
            down_ptrs.packed[k] = put(ent.host_w2, ent.w2_bytes);
            down_ptrs.scales[k] = put(ent.host_w2_scales, ent.w2_scales_bytes);
            if (gate_ptrs.packed[k] == nullptr ||
                gate_ptrs.scales[k] == nullptr || up_ptrs.packed[k] == nullptr ||
                up_ptrs.scales[k] == nullptr || down_ptrs.packed[k] == nullptr ||
                down_ptrs.scales[k] == nullptr) {
                stage_err = absl::StrFormat(
                    "[%s] FP8 staging slab too small for layer %d", stage, l);
                return cudaErrorInvalidValue;
            }
        }
        e = RunInklingFp8ExpertGemm(stream, ws.hh, gate_ptrs, topk, 0,
                                    ws.moe_gate, H, I, false);
        if (e != cudaSuccess) return fail("mxfp8 gate gemv", e);
        e = RunInklingFp8ExpertGemm(stream, ws.hh, up_ptrs, topk, 0, ws.moe_up, H,
                                    I, false);
        if (e != cudaSuccess) return fail("mxfp8 up gemv", e);
        e = RunInklingSoftcapSwiGLU(
            stream, ws.moe_gate, ws.moe_up, ws.moe_act, int64_t(topk) * I,
            static_cast<float>(top.mlp_gate_softcap),
            static_cast<float>(top.mlp_up_softcap));
        if (e != cudaSuccess) return fail("softcapped swiglu", e);
        e = RunInklingFp8ExpertGemm(stream, ws.moe_act, down_ptrs, topk, I,
                                    ws.moe_down, I, H, false);
        if (e != cudaSuccess) return fail("mxfp8 down gemv", e);
        e = RunInklingExpertAccumulate(stream, ws.moe_acc, ws.moe_down, ws.d_slots,
                                       topk, H);
        if (e != cudaSuccess) return fail("expert accumulate", e);
    } else {
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
        e = RunInklingSoftcapSwiGLU(
            stream, ws.moe_gate, ws.moe_up, ws.moe_act, int64_t(topk) * I,
            static_cast<float>(top.mlp_gate_softcap),
            static_cast<float>(top.mlp_up_softcap));
        if (e != cudaSuccess) return fail("softcapped swiglu", e);
        e = lsx_nvfp4::RunLsxNvfp4GemmSeq1GemmVExpertBatch(
            stream, ws.moe_act, st.down_p, st.down_s, st.w2_gs, topk, I,
            ws.moe_down, I, H, false);
        if (e != cudaSuccess) return fail("routed down gemv", e);
        e = RunInklingExpertAccumulate(stream, ws.moe_acc, ws.moe_down, ws.d_slots,
                                       topk, H);
        if (e != cudaSuccess) return fail("expert accumulate", e);
    }

    // ── (h) single FP16 cast, post-MLP conv, residual ─────────────────────
    stage = LayerTag("mlp_out", l);
    e = RunInklingCastToHalf(stream, ws.moe_acc, ws.o_out, H);
    if (e != cudaSuccess) return fail("moe cast", e);
    e = RunInklingShortConv(stream, ws.o_out,
                            conv_base + 2 * kconv * kv_dim + kconv * H,
                            lw.mlp_conv_w, nullptr, ws.conv_out, H, kconv,
                            inv_static);
    if (e != cudaSuccess) return fail("post-mlp conv", e);
    e = RunInklingHalfToF32(stream, ws.conv_out, ws.conv_f, H);
    if (e != cudaSuccess) return fail("post-mlp to f32", e);
    e = RunInklingAddF32(stream, ws.conv_f, ws.residual, H);
    if (e != cudaSuccess) return fail("mlp residual add", e);
    e = RunInklingScaleToHalf(stream, ws.conv_f, ws.x_h, H, 1.0f);
    if (e != cudaSuccess) return fail("layer output store", e);
    return cudaSuccess;
}

// ── one token through the whole stack ──────────────────────────────────────
// Writes the softcapped (+ sliced) logits row into `ws.logits`. The block stack
// lives in `ws.stack` and is threaded through every application in execution
// order, so the loop passes see exactly what the reference sees.
cudaError_t Forward(cudaStream_t stream, InklingLoopPinnedWeights &pw,
                    InklingLoopExpertDb &experts, InklingLoopSession &session,
                    InklingLoopScratch &ws, int32_t tok, int pos,
                    float logit_softcap, int unpadded_vocab,
                    std::string &stage_err, std::vector<float> *trace)
{
    const InklingLoopTopology &top = pw.top;
    const int H = top.hidden;
    const float eps = static_cast<float>(top.rms_eps);
    std::string stage = "embed";
    auto fail = [&](const char *why, cudaError_t e) {
        stage_err = absl::StrFormat("[%s] %s (cuda %d)", stage, why,
                                    static_cast<int>(e));
        return e;
    };
    if (pos < 0 || pos + 1 > kLoopMaxContext) {
        stage_err = absl::StrFormat("inkling loop: position %d outside the "
                                    "engine context bound (%d)",
                                    pos, kLoopMaxContext);
        return cudaErrorInvalidValue;
    }
    cudaError_t e = cudaSuccess;
    // ── embed + embed_norm ─────────────────────────────────────────────────
    e = cudaMemcpyAsync(ws.d_token, &tok, sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream);
    if (e != cudaSuccess) return fail("token upload", e);
    e = RunInklingEmbedGather(stream, pw.embed, ws.d_token, ws.x_h, H);
    if (e != cudaSuccess) return fail("embed row", e);
    e = RunInklingHalfToF32(stream, ws.x_h, ws.x_f, H);
    if (e != cudaSuccess) return fail("embed to f32", e);
    e = RunInklingNormF32In(stream, ws.x_f, pw.embed_norm, ws.x_h, 1, H, eps);
    if (e != cudaSuccess) return fail("embed_norm", e);

    auto record = [&](int row) -> cudaError_t {
        if (trace == nullptr) return cudaSuccess;
        // The kernels run on a non-default stream, so every trace copy must be
        // ordered after them or it reads a stale row.
        cudaError_t ce = cudaStreamSynchronize(stream);
        if (ce != cudaSuccess) return ce;
        std::vector<__half> tmp(static_cast<size_t>(H));
        ce = cudaMemcpy(tmp.data(), ws.x_h, tmp.size() * sizeof(__half),
                        cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) return ce;
        float *dst = trace->data() + static_cast<size_t>(row) * H;
        for (int i = 0; i < H; ++i)
            dst[i] = __half2float(tmp[static_cast<size_t>(i)]);
        return cudaSuccess;
    };
    if (trace != nullptr) {
        trace->assign(static_cast<size_t>(top.CacheLayers() + 1) * H, 0.0f);
        e = record(0);
        if (e != cudaSuccess) return fail("trace row 0", e);
    }

    int n_blocks = 0;
    int row = 1;
    const int passes = std::max(1, top.loop_passes);
    const int loop_start = top.loop_start;
    const int loop_end = top.loop_end;

    // ── pre-loop layers ────────────────────────────────────────────────────
    for (int l = 0; l < loop_start; ++l) {
        e = RunLoopLayer(stream, pw, experts, session, ws, l, 0, pos, nullptr,
                         n_blocks, stage_err);
        if (e != cudaSuccess) return e;
        e = record(row++);
        if (e != cudaSuccess) return fail("trace row", e);
    }

    // ── snapshot at loop entry ─────────────────────────────────────────────
    // Every extra pass restarts from the hidden state and block stack the loop
    // core was entered with; only the messages (and the per-pass KV/conv slots)
    // differ. A pass that continues from the previous pass's output instead is
    // a loop that composes, not the release's loop.
    e = RunInklingHalfToF32(stream, ws.x_h, ws.pre_loop_hidden, H);
    if (e != cudaSuccess) return fail("pre-loop hidden snapshot", e);
    if (n_blocks > 0) {
        e = cudaMemcpyAsync(ws.pre_loop_stack, ws.stack,
                            sizeof(float) * H * n_blocks,
                            cudaMemcpyDeviceToDevice, stream);
        if (e != cudaSuccess) return fail("pre-loop stack snapshot", e);
    }
    ws.pre_loop_sources = n_blocks;

    // ── pass 0 over the loop core (messages = none) ────────────────────────
    for (int l = loop_start; l < loop_end; ++l) {
        e = RunLoopLayer(stream, pw, experts, session, ws, l, 0, pos, nullptr,
                         n_blocks, stage_err);
        if (e != cudaSuccess) return e;
        e = record(row++);
        if (e != cudaSuccess) return fail("trace row", e);
    }

    // The message sources of a pass: its block states SKIPPING the first one
    // appended (which merely re-states the pre-loop hidden) plus its final
    // hidden state, packed contiguously for the retrieval kernel.
    auto snapshot_prev = [&](int blocks) -> cudaError_t {
        const int appended = blocks - ws.pre_loop_sources;
        const int keep = std::max(0, appended - 1);
        if (keep + 1 > loop::frozen::kAttnResMaxSources) return cudaErrorInvalidValue;
        if (keep > 0) {
            cudaError_t ce = cudaMemcpyAsync(
                ws.prev_stack,
                ws.stack + static_cast<int64_t>(ws.pre_loop_sources + 1) * H,
                sizeof(float) * H * keep, cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) return ce;
        }
        cudaError_t ce = RunInklingHalfToF32(stream, ws.x_h, ws.prev_hidden, H);
        if (ce != cudaSuccess) return ce;
        ce = cudaMemcpyAsync(ws.prev_stack + static_cast<int64_t>(keep) * H,
                             ws.prev_hidden, sizeof(float) * H,
                             cudaMemcpyDeviceToDevice, stream);
        if (ce != cudaSuccess) return ce;
        ws.prev_sources = keep + 1;
        return cudaSuccess;
    };
    e = snapshot_prev(n_blocks);
    if (e != cudaSuccess) return fail("pass-0 message snapshot", e);

    // ── passes 1..passes-1 ─────────────────────────────────────────────────
    for (int p = 1; p < passes; ++p) {
        stage = absl::StrFormat("loop pass %d", p);
        e = RunInklingScaleToHalf(stream, ws.pre_loop_hidden, ws.x_h, H, 1.0f);
        if (e != cudaSuccess) return fail("pre-loop hidden restore", e);
        if (ws.pre_loop_sources > 0) {
            e = cudaMemcpyAsync(ws.stack, ws.pre_loop_stack,
                                sizeof(float) * H * ws.pre_loop_sources,
                                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) return fail("pre-loop stack restore", e);
        }
        n_blocks = ws.pre_loop_sources;
        for (int l = loop_start; l < loop_end; ++l) {
            const int64_t q_off = static_cast<int64_t>(l - loop_start) * 4 * H;
            e = RunInklingLoopMessageMix(stream, ws.prev_stack, ws.prev_sources,
                                         pw.message_queries + q_off, ws.msg, H,
                                         eps);
            if (e != cudaSuccess) return fail("message retrieval", e);
            e = RunLoopLayer(stream, pw, experts, session, ws, l, p, pos, ws.msg,
                             n_blocks, stage_err);
            if (e != cudaSuccess) return e;
            e = record(row++);
            if (e != cudaSuccess) return fail("trace row", e);
        }
        e = snapshot_prev(n_blocks);
        if (e != cudaSuccess) return fail("message snapshot", e);
    }

    // ── post-loop layers ───────────────────────────────────────────────────
    for (int l = loop_end; l < top.num_layers; ++l) {
        e = RunLoopLayer(stream, pw, experts, session, ws, l, 0, pos, nullptr,
                         n_blocks, stage_err);
        if (e != cudaSuccess) return e;
        e = record(row++);
        if (e != cudaSuccess) return fail("trace row", e);
    }

    // ── head: final mixture -> norm -> unembed -> softcap -> slice ─────────
    stage = "head";
    e = RunInklingHalfToF32(stream, ws.x_h, ws.x_f, H);
    if (e != cudaSuccess) return fail("final residual", e);
    e = cudaMemcpyAsync(ws.stack + static_cast<int64_t>(n_blocks) * H, ws.x_f,
                        sizeof(float) * H, cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return fail("final partial", e);
    e = RunInklingDepthMix(stream, ws.stack, n_blocks + 1, pw.out_query, 1, ws.mix,
                           H, eps);
    if (e != cudaSuccess) return fail("final depth mix", e);
    e = RunInklingNormF32In(stream, ws.mix, pw.final_norm, ws.hh, 1, H, eps);
    if (e != cudaSuccess) return fail("final norm", e);
    e = RunInklingMatVecF16(stream, pw.head, ws.hh, ws.logits, pw.embed_rows, H);
    if (e != cudaSuccess) return fail("unembed", e);
    if (unpadded_vocab > 0 && unpadded_vocab < pw.embed_rows) {
        e = RunInklingLogitSoftcap(stream, ws.logits, ws.logits, unpadded_vocab,
                                   logit_softcap);
        if (e != cudaSuccess) return fail("logit softcap", e);
    }
    return cudaSuccess;
}

}  // namespace

cudaError_t RunInklingLoopForwardTrace(cudaStream_t stream,
                                       InklingLoopPinnedWeights &pw,
                                       InklingLoopExpertDb &experts,
                                       InklingLoopSession &session,
                                       InklingLoopScratch &scratch,
                                       const std::vector<int32_t> &tokens,
                                       float logit_softcap, int unpadded_vocab,
                                       std::vector<float> &out_states,
                                       std::vector<float> *out_logits)
{
    if (tokens.empty()) return cudaErrorInvalidValue;
    const InklingLoopTopology &top = pw.top;
    cudaError_t e =
        session.EnsureCapacity(top, static_cast<int64_t>(tokens.size()), stream);
    if (e != cudaSuccess) return e;
    if (scratch.arena == nullptr &&
        !scratch.Allocate(top, session.capacity, stream))
        return cudaErrorMemoryAllocation;
    // A trace is a fresh sequence: zero every cache slot's conv ring and the KV
    // bookkeeping so the history matches what a real request sees.
    session.Reset(stream);
    std::string err;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const bool last = (i + 1 == tokens.size());
        e = Forward(stream, pw, experts, session, scratch, tokens[i],
                    static_cast<int>(i), logit_softcap, unpadded_vocab, err,
                    last ? &out_states : nullptr);
        if (e != cudaSuccess) return e;
    }
    // The prompt's positions are now in the cache; a following generation call
    // continues from them.
    session.kv_rows = static_cast<int>(tokens.size());
    session.last_tok = tokens.back();
    if (out_logits != nullptr) {
        const int n = (unpadded_vocab > 0 && unpadded_vocab < pw.embed_rows)
                          ? unpadded_vocab
                          : pw.embed_rows;
        out_logits->assign(static_cast<size_t>(n), 0.0f);
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return e;
        e = cudaMemcpy(out_logits->data(), scratch.logits,
                       out_logits->size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

cudaError_t RunInklingLoopGeneration(cudaStream_t stream,
                                     InklingLoopPinnedWeights &pw,
                                     InklingLoopExpertDb &experts,
                                     InklingLoopSession &session,
                                     InklingLoopScratch &scratch,
                                     const std::vector<int32_t> &input_ids,
                                     int max_gen_tokens,
                                     const std::vector<int32_t> &eos_ids,
                                     float logit_softcap, int unpadded_vocab,
                                     sample::ISampler &sampler,
                                     InklingLoopGenerationResult &out,
                                     const GenerationStream *gs)
{
    out = InklingLoopGenerationResult{};
    if (input_ids.empty()) {
        out.error_msg = "inkling loop: empty prompt";
        return cudaErrorInvalidValue;
    }
    const InklingLoopTopology &top = pw.top;
    const int64_t needed = static_cast<int64_t>(input_ids.size()) +
                           std::max(0, max_gen_tokens);
    if (needed > kLoopMaxContext) {
        out.error_msg = absl::StrFormat(
            "inkling loop: %lld positions exceed the engine context bound (%d)",
            static_cast<long long>(needed), kLoopMaxContext);
        return cudaErrorInvalidValue;
    }
    cudaError_t e = session.EnsureCapacity(top, needed, stream);
    if (e != cudaSuccess) {
        out.error_msg = absl::StrFormat(
            "inkling loop: KV allocation failed (cuda %d)", static_cast<int>(e));
        return e;
    }
    if (scratch.arena == nullptr) {
        if (!scratch.Allocate(top, session.capacity, stream)) {
            out.error_msg = "inkling loop: decode scratch allocation failed";
            return cudaErrorMemoryAllocation;
        }
    }
    const int vocab = (unpadded_vocab > 0 && unpadded_vocab < pw.embed_rows)
                          ? unpadded_vocab
                          : pw.embed_rows;

    // A new request is a new sequence: the session owns every conv ring (one per
    // cache slot), so Reset() clears all of them plus the KV bookkeeping.
    session.Reset(stream);

    std::string err;
    const auto t_start = steady::now();
    out.ttft_ms = 0.0;

    // ── prompt ────────────────────────────────────────────────────────────
    int pos = 0;
    for (std::size_t i = 0; i < input_ids.size(); ++i) {
        e = Forward(stream, pw, experts, session, scratch, input_ids[i], pos,
                    logit_softcap, unpadded_vocab, err, nullptr);
        if (e != cudaSuccess) {
            out.error_msg = "inkling loop prompt: " + err;
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
            out.error_msg = absl::StrFormat("inkling loop: sampler (cuda %d)",
                                            static_cast<int>(e));
            return e;
        }
        e = cudaMemcpyAsync(&tok, scratch.d_token, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) {
            out.error_msg = "inkling loop: token download failed";
            return e;
        }
        e = cudaMemcpyAsync(&maxv, scratch.d_maxv, sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
        if (e != cudaSuccess) {
            out.error_msg = "inkling loop: max-logit download failed";
            return e;
        }
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            out.error_msg = absl::StrFormat("inkling loop: decode sync (cuda %d)",
                                            static_cast<int>(e));
            return e;
        }
        if (tok < 0 || tok >= vocab) {
            out.error_msg =
                absl::StrFormat("inkling loop: sampler returned id %d", tok);
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
                    unpadded_vocab, err, nullptr);
        if (e != cudaSuccess) {
            out.error_msg = "inkling loop decode: " + err;
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