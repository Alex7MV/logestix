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

#include <absl/strings/str_format.h>

#include <algorithm>
#include <cstring>

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
    if (cudaMalloc(&d_slots, sizeof(InklingLoopRoutedSlot) * topk) != cudaSuccess)
        return false;
    if (cudaMalloc(&d_token, sizeof(int32_t)) != cudaSuccess) return false;
    if (cudaMalloc(&d_maxv, sizeof(float)) != cudaSuccess) return false;
    return true;
}

}  // namespace lsxcommon::inkling