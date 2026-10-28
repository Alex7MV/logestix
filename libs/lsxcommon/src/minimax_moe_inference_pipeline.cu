#include "lsxcommon/minimax_moe_inference_pipeline.h"

#include "minimax_moe_constants.h"
#include "lsx_kernels_math.cuh"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/model_runtime.h"

#include <absl/strings/str_cat.h>

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace lsxcommon::minimax {
namespace {

constexpr int kThreads = 256;

std::string LayerTag(const char *what, int layer)
{
    return absl::StrCat(what, " l", layer);
}

} // namespace

std::int64_t MinimaxSession::Bytes() const
{
    if (!d_k) return 0;
    const int64_t per_layer = geom.LayerStride();
    return 2 * static_cast<int64_t>(geom.layers) * per_layer * 2 +
           static_cast<int64_t>(geom.layers) * capacity * index_head_dim * 2 +
           static_cast<int64_t>(gen_tail.capacity()) * 4;
}

void MinimaxSession::Free()
{
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    if (d_idx_k) cudaFree(d_idx_k);
    d_k = nullptr;
    d_v = nullptr;
    d_idx_k = nullptr;
    geom = {};
    capacity = 0;
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

void MinimaxSession::Reset()
{
    kv_rows = 0;
    last_tok = -1;
    gen_tail.clear();
}

cudaError_t MinimaxSession::EnsureCapacity(const MinimaxTopology &top,
                                           int64_t want,
                                           cudaStream_t stream)
{
    if (want <= capacity && d_k) return cudaSuccess;
    // Grow to at least `want` with headroom so a long generation does not
    // reallocate every step.
    const int64_t cap = std::max<int64_t>(want, capacity * 2);
    const KvCacheGeometry g = MinimaxKvGeometry(top, cap);
    const int64_t layer_elems = g.LayerStride();
    const int64_t kv_elems =
        static_cast<int64_t>(g.layers) * layer_elems;
    const int64_t idx_elems = static_cast<int64_t>(g.layers) * cap *
                              top.index_head_dim;

    __half *nk = nullptr;
    __half *nv = nullptr;
    __half *ni = nullptr;
    if (cudaMalloc(&nk, static_cast<size_t>(kv_elems) * sizeof(__half)) !=
        cudaSuccess)
        return cudaGetLastError();
    if (cudaMalloc(&nv, static_cast<size_t>(kv_elems) * sizeof(__half)) !=
        cudaSuccess) {
        cudaFree(nk);
        return cudaGetLastError();
    }
    if (cudaMalloc(&ni, static_cast<size_t>(idx_elems) * sizeof(__half)) !=
        cudaSuccess) {
        cudaFree(nk);
        cudaFree(nv);
        return cudaGetLastError();
    }
    // Preserve the cached prefix (session continuation / growth mid-request).
    if (d_k && kv_rows > 0) {
        for (int l = 0; l < g.layers; ++l) {
            const size_t bytes = static_cast<size_t>(kv_rows) *
                                 static_cast<size_t>(g.pos_stride) *
                                 sizeof(__half);
            cudaMemcpyAsync(nk + l * layer_elems, d_k + l * layer_elems, bytes,
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(nv + l * layer_elems, d_v + l * layer_elems, bytes,
                            cudaMemcpyDeviceToDevice, stream);
            const size_t ibytes = static_cast<size_t>(kv_rows) *
                                  static_cast<size_t>(top.index_head_dim) *
                                  sizeof(__half);
            cudaMemcpyAsync(ni + l * cap * top.index_head_dim,
                            d_idx_k + l * capacity * top.index_head_dim,
                            ibytes, cudaMemcpyDeviceToDevice, stream);
        }
        cudaStreamSynchronize(stream);
    }
    if (d_k) cudaFree(d_k);
    if (d_v) cudaFree(d_v);
    if (d_idx_k) cudaFree(d_idx_k);
    d_k = nk;
    d_v = nv;
    d_idx_k = ni;
    geom = g;
    capacity = cap;
    index_head_dim = top.index_head_dim;
    return cudaSuccess;
}

void MinimaxScratch::Free()
{
    if (arena) cudaFree(arena);
    arena = nullptr;
    arena_bytes = 0;
    half_elems = 0;
    float_elems = 0;
    mask_bytes = 0;
    seq_capacity = 0;
    score_floats = 0;
    logits_elems = 0;
    d_ids = nullptr;
    d_argmax = nullptr;
    d_maxv = nullptr;
    d_slots = nullptr;
    d_logits = nullptr;
    if (h_slots) cudaFreeHost(h_slots);
    h_slots = nullptr;
}

namespace {

// One dense SwiGLU-OAI MLP (also used for the ungated shared expert):
// h = swigluoai(gate @ x, up @ x); out = down @ h.
cudaError_t DenseMlp(cudaStream_t stream, const __half *x,
                     const lsx_nvfp4::LsxNvfp4Weight &gate,
                     const lsx_nvfp4::LsxNvfp4Weight &up,
                     const lsx_nvfp4::LsxNvfp4Weight &down, int H, int I,
                     float alpha, float limit, __half *d_yg, __half *d_yu,
                     __half *d_h, __half *out)
{
    cudaError_t e = lsx_nvfp4::RunLsxNvfp4Gemm(
        stream, x, gate.d_packed, gate.d_scales, gate.global_scale, d_yg, 1, H,
        I, false);
    if (e != cudaSuccess) return e;
    e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, x, up.d_packed, up.d_scales,
                                   up.global_scale, d_yu, 1, H, I, false);
    if (e != cudaSuccess) return e;
    e = RunMinimaxSwiGLUOai(stream, d_yg, d_yu, d_h, I, alpha, limit);
    if (e != cudaSuccess) return e;
    return lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_h, down.d_packed,
                                      down.d_scales, down.global_scale, out, 1,
                                      I, H, false);
}

} // namespace

gpu::InferenceResult RunMinimaxInference(
    const std::string &model_path, const std::vector<int32_t> &input_ids,
    int max_gen_tokens, const MinimaxPinnedWeights &pw,
    MinimaxExpertStore &experts, gpu::ExpertDeviceCache &cache,
    MinimaxSession &session, MinimaxScratch &scratch, sample::ISampler &sampler,
    cudaStream_t stream, bool ignore_eos, const GenerationStream *gs)
{
    gpu::InferenceResult out;
    out.num_layers = pw.top.num_layers;
    (void)model_path;

    const MinimaxTopology &top = pw.top;
    const int H = top.hidden_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const int V = top.vocab_size;
    const int E = top.num_experts;
    const int K = top.num_experts_per_tok;
    const int I = top.moe_intermediate_size;
    const int SI = top.shared_intermediate_size;
    const int DI = top.dense_intermediate_size;
    const int IH = top.index_n_heads;
    const int ID = top.index_head_dim;
    const int L = top.num_layers;
    const int imax = std::max(std::max(I, SI), DI);
    const float eps = static_cast<float>(top.rms_norm_eps);
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    const int eos_id = top.eos_token_id;

    if (H <= 0 || V <= 0 || NH <= 0 || NKV <= 0 || HD <= 0 || L <= 0 ||
        E <= 0 || K <= 0 || E > 512 || K > 32) {
        out.error_msg = "invalid MiniMax-M3 topology";
        return out;
    }
    if (input_ids.empty() || max_gen_tokens <= 0) {
        out.error_msg = "empty prompt or max_gen_tokens <= 0";
        return out;
    }

    // ── Session decision (shared incremental-session policy) ──
    const qwen::SessionDecision decision = qwen::DecideSession(
        session.allocated(), session.last_tok, session.gen_tail, input_ids);
    const std::vector<int32_t> prefill = decision.gen_input;
    if (!decision.is_continuation) session.Reset();

    const int prompt_steps = static_cast<int>(prefill.size());
    const int total_steps = prompt_steps + max_gen_tokens - 1;
    const int64_t need_cap =
        session.kv_rows + prompt_steps + max_gen_tokens + 8;
    if (cudaError_t ce = session.EnsureCapacity(top, need_cap, stream);
        ce != cudaSuccess) {
        out.error_msg = absl::StrCat("kv cache alloc failed: ",
                                     cudaGetErrorString(ce));
        return out;
    }

    // ── Scratch (grown on demand; one arena for halves, floats and bytes) ──
    const int num_blocks_max =
        static_cast<int>((session.capacity + top.index_block_size - 1) /
                         top.index_block_size);
    size_t half_sz[] = {
        static_cast<size_t>(H),            // d_act
        static_cast<size_t>(H),            // d_normed
        static_cast<size_t>(H),            // d_attn_out
        static_cast<size_t>(H),            // d_postin
        static_cast<size_t>(NH * HD),      // d_q
        static_cast<size_t>(NKV * HD),     // d_kp
        static_cast<size_t>(NKV * HD),     // d_vp
        static_cast<size_t>(NH * HD),      // d_attn
        static_cast<size_t>(IH * ID),      // d_idx_q
        static_cast<size_t>(ID),           // d_idx_k
        static_cast<size_t>(2 * imax),     // d_yg (dense/shared gate)
        static_cast<size_t>(2 * imax),     // d_yu (dense/shared up)
        static_cast<size_t>(imax),         // d_h
        static_cast<size_t>(H),            // d_shared
        static_cast<size_t>(H),            // d_mlp_out
        static_cast<size_t>(I),            // d_yg_slot
        static_cast<size_t>(I),            // d_yu_slot
        static_cast<size_t>(I),            // d_h_slot
        static_cast<size_t>(H),            // d_y2_slot
    };
    size_t half_bytes = 0;
    for (auto s : half_sz) half_bytes += ((s + 7) / 8) * 8 * sizeof(__half);
    half_bytes = (half_bytes + 15) & ~static_cast<size_t>(15);
    size_t float_sz[] = {
        static_cast<size_t>(H),   // d_moe_acc
    };
    size_t float_bytes = 0;
    for (auto s : float_sz) float_bytes += s * sizeof(float);
    float_bytes = (float_bytes + 15) & ~static_cast<size_t>(15);
    const size_t mask_bytes =
        ((static_cast<size_t>(IH) * num_blocks_max) + 63) / 64 * 64;
    const size_t score_bytes =
        (((static_cast<size_t>(NH) * session.capacity) * sizeof(float)) + 63) /
        64 * 64;
    const size_t arena_need = half_bytes + float_bytes + mask_bytes + score_bytes;
    if (!scratch.arena || scratch.arena_bytes < arena_need) {
        if (scratch.arena) cudaFree(scratch.arena);
        scratch.arena = nullptr;
        if (cudaMalloc(&scratch.arena, arena_need) != cudaSuccess) {
            out.error_msg = "scratch alloc failed";
            return out;
        }
        scratch.arena_bytes = arena_need;
    }
    if (!scratch.d_ids) {
        cudaMalloc(&scratch.d_ids, sizeof(int32_t));
        cudaMalloc(&scratch.d_argmax, sizeof(int32_t));
        cudaMalloc(&scratch.d_maxv, sizeof(float));
        cudaMalloc(&scratch.d_slots,
                   sizeof(MinimaxRoutedSlot) * static_cast<size_t>(K));
        cudaMallocHost(&scratch.h_slots,
                       sizeof(MinimaxRoutedSlot) * static_cast<size_t>(K));
    }
    if (scratch.logits_elems < V) {
        if (scratch.d_logits) cudaFree(scratch.d_logits);
        if (cudaMalloc(&scratch.d_logits,
                       static_cast<size_t>(V) * sizeof(float)) != cudaSuccess) {
            out.error_msg = "logits alloc failed";
            return out;
        }
        scratch.logits_elems = V;
    }
    __half *ph = static_cast<__half *>(scratch.arena);
    auto adv = [&](size_t elems) {
        __half *r = ph;
        ph += ((elems + 7) / 8) * 8;
        return r;
    };
    __half *d_act = adv(half_sz[0]);
    __half *d_normed = adv(half_sz[1]);
    __half *d_attn_out = adv(half_sz[2]);
    __half *d_postin = adv(half_sz[3]);
    __half *d_q = adv(half_sz[4]);
    __half *d_kp = adv(half_sz[5]);
    __half *d_vp = adv(half_sz[6]);
    __half *d_attn = adv(half_sz[7]);
    __half *d_idx_q = adv(half_sz[8]);
    __half *d_idx_k = adv(half_sz[9]);
    __half *d_yg = adv(half_sz[10]);
    __half *d_yu = adv(half_sz[11]);
    __half *d_h = adv(half_sz[12]);
    __half *d_shared = adv(half_sz[13]);
    __half *d_mlp_out = adv(half_sz[14]);
    __half *d_yg_slot = adv(half_sz[15]);
    __half *d_yu_slot = adv(half_sz[16]);
    __half *d_h_slot = adv(half_sz[17]);
    __half *d_y2_slot = adv(half_sz[18]);
    float *d_moe_acc = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(scratch.arena) + half_bytes);
    uint8_t *d_mask = reinterpret_cast<uint8_t *>(scratch.arena) + half_bytes +
                      float_bytes;
    float *d_scores = reinterpret_cast<float *>(d_mask + mask_bytes);

    int32_t *d_ids = scratch.d_ids;
    int32_t *d_argmax = scratch.d_argmax;
    float *d_maxv = scratch.d_maxv;
    MinimaxRoutedSlot *d_slots = scratch.d_slots;
    MinimaxRoutedSlot *h_slots = scratch.h_slots;
    float *d_logits = scratch.d_logits;

    cudaEvent_t t_start = nullptr, t_ttft = nullptr, t_stop = nullptr;
    cudaEvent_t tok_ready = nullptr;
    cudaEventCreate(&t_start);
    cudaEventCreate(&t_ttft);
    cudaEventCreate(&t_stop);
    cudaEventCreateWithFlags(&tok_ready, cudaEventDisableTiming);
    cudaEventRecord(t_start, stream);

    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(max_gen_tokens));
    float first_maxv = 0.0f;
    bool eos_terminated = false;
    std::string moe_error;

    auto fail = [&](cudaError_t err, const std::string &msg) {
        out.error_msg = msg;
        if (err != cudaSuccess)
            out.error_msg += absl::StrCat(" (", cudaGetErrorString(err), ")");
        if (t_start) cudaEventDestroy(t_start);
        if (t_ttft) cudaEventDestroy(t_ttft);
        if (t_stop) cudaEventDestroy(t_stop);
        if (tok_ready) cudaEventDestroy(tok_ready);
        return out;
    };

    int32_t cur = prefill[0];
    int generated = 0;
    int64_t emitted = 0;
    bool aborted = false;

    for (int step = 0; step < total_steps; ++step) {
        const int64_t pos = session.kv_rows;
        // Token id for this step: the next prefill token, or the previous
        // step's sampled id. One small H2D either way.
        cudaMemcpyAsync(d_ids, &cur, sizeof(int32_t), cudaMemcpyHostToDevice,
                        stream);

        cudaError_t e =
            RunMinimaxEmbed(stream, pw.d_embed, d_ids, d_act, 1, H);
        if (e != cudaSuccess) return fail(e, "embed");
        // Diagnostic: the embedded row (stage "embed", layer -1).
        if (const char *dump = std::getenv("LSX_MINIMAX_DUMP_LAYERS");
            dump && step == prompt_steps - 1) {
            std::vector<__half> hh(16);
            cudaMemcpyAsync(hh.data(), d_act, 16 * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
            FILE *df = std::fopen(dump, "w");
            if (df) {
                std::fprintf(df, "embed");
                for (int i = 0; i < 16; ++i)
                    std::fprintf(df, " %.6f",
                                 __half2float(hh[static_cast<size_t>(i)]));
                std::fprintf(df, "\n");
                std::fclose(df);
            }
        }

        for (int l = 0; l < L; ++l) {
            const MinimaxLayerWeights &lw = pw.layers[static_cast<size_t>(l)];
            const bool sparse_attn =
                top.layer_types[static_cast<size_t>(l)] == "sparse_attention";

            // ── attention block ──
            e = RunMinimaxNorm(stream, d_act, lw.attn.d_input_norm, d_normed, 1,
                               H, eps);
            if (e != cudaSuccess) return fail(e, LayerTag("norm", l));
            e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_normed, lw.attn.q.d_packed,
                                           lw.attn.q.d_scales,
                                           lw.attn.q.global_scale, d_q, 1, H,
                                           NH * HD, false);
            if (e != cudaSuccess) return fail(e, LayerTag("q", l));
            e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_normed, lw.attn.k.d_packed,
                                           lw.attn.k.d_scales,
                                           lw.attn.k.global_scale, d_kp, 1, H,
                                           NKV * HD, false);
            if (e != cudaSuccess) return fail(e, LayerTag("k", l));
            e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_normed, lw.attn.v.d_packed,
                                           lw.attn.v.d_scales,
                                           lw.attn.v.global_scale, d_vp, 1, H,
                                           NKV * HD, false);
            if (e != cudaSuccess) return fail(e, LayerTag("v", l));
            e = RunMinimaxHeadNorm(stream, d_q, lw.attn.d_q_norm, d_q, 1, NH, HD,
                                   eps);
            if (e != cudaSuccess) return fail(e, LayerTag("qnorm", l));
            e = RunMinimaxHeadNorm(stream, d_kp, lw.attn.d_k_norm, d_kp, 1, NKV,
                                   HD, eps);
            if (e != cudaSuccess) return fail(e, LayerTag("knorm", l));
            e = RunMinimaxPartialRope(stream, d_q, d_kp, 1, NH, NKV, HD,
                                      top.rotary_dim, static_cast<float>(
                                                         top.rope_theta),
                                      static_cast<int>(pos));
            if (e != cudaSuccess) return fail(e, LayerTag("rope", l));

            // Append k/v to the layer's cache slice (pos-major rows).
            const int64_t layer_elems = session.geom.LayerStride();
            __half *k_base = session.d_k + l * layer_elems;
            __half *v_base = session.d_v + l * layer_elems;
            {
                const size_t row = static_cast<size_t>(session.geom.pos_stride) *
                                   sizeof(__half);
                cudaMemcpyAsync(k_base + pos * session.geom.pos_stride, d_kp,
                                row, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(v_base + pos * session.geom.pos_stride, d_vp,
                                row, cudaMemcpyDeviceToDevice, stream);
            }

            // ── MiniMax Sparse Attention selection ──
            const int kv_len = static_cast<int>(pos) + 1;
            const int num_blocks =
                (kv_len + top.index_block_size - 1) / top.index_block_size;
            const uint8_t *mask = nullptr;
            if (sparse_attn && num_blocks > top.index_topk_blocks) {
                e = lsx_nvfp4::RunLsxNvfp4Gemm(
                    stream, d_normed, lw.attn.index_q.d_packed,
                    lw.attn.index_q.d_scales,
                    lw.attn.index_q.global_scale, d_idx_q, 1, H, IH * ID,
                    false);
                if (e != cudaSuccess) return fail(e, LayerTag("idxq", l));
                e = lsx_nvfp4::RunLsxNvfp4Gemm(
                    stream, d_normed, lw.attn.index_k.d_packed,
                    lw.attn.index_k.d_scales,
                    lw.attn.index_k.global_scale, d_idx_k, 1, H, ID, false);
                if (e != cudaSuccess) return fail(e, LayerTag("idxk", l));
                e = RunMinimaxHeadNorm(stream, d_idx_q, lw.attn.d_index_q_norm,
                                       d_idx_q, 1, IH, ID, eps);
                if (e != cudaSuccess) return fail(e, LayerTag("idxqnorm", l));
                e = RunMinimaxHeadNorm(stream, d_idx_k, lw.attn.d_index_k_norm,
                                       d_idx_k, 1, 1, ID, eps);
                if (e != cudaSuccess) return fail(e, LayerTag("idxknorm", l));
                e = RunMinimaxPartialRope(
                    stream, d_idx_q, d_idx_k, 1, IH, 1, ID, top.rotary_dim,
                    static_cast<float>(top.rope_theta),
                    static_cast<int>(pos));
                if (e != cudaSuccess) return fail(e, LayerTag("idxrope", l));
                __half *idx_base = session.d_idx_k +
                                   static_cast<int64_t>(l) * session.capacity * ID;
                cudaMemcpyAsync(idx_base + pos * ID, d_idx_k,
                                static_cast<size_t>(ID) * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                e = RunMinimaxIndexerSelect(
                    stream, d_idx_q, idx_base, kv_len, static_cast<int>(pos), IH,
                    ID, top.index_block_size, top.index_topk_blocks,
                    top.index_local_blocks, d_mask);
                if (e != cudaSuccess) return fail(e, LayerTag("indexer", l));
                mask = d_mask;
            }

            if (!mask) {
                // Dense-equivalent path (every block selected): the neutral
                // coalesced kernel, which is the hot path up to the block
                // budget (topk_blocks * block_size = 2048 tokens). Past 512
                // cached rows the single-block-per-head kernel is
                // memory-latency bound, so the split-KV decode path takes over
                // (same policy as RunAttnFullAttention).
                if (pos + 1 > 512) {
                    e = qwen::RunAttnFullAttnKernelSplitKV(
                        stream, d_q, k_base, v_base, d_attn, 1, NH, HD, NKV,
                        scale, static_cast<int>(pos));
                } else {
                    e = qwen::RunAttnFullAttnKernel(
                        stream, d_q, k_base, v_base, d_attn, 1, NH, HD, NKV,
                        scale, static_cast<int>(pos));
                }
            } else {
                e = RunMinimaxMaskedAttn(
                    stream, d_q, k_base, v_base, kv_len, NH, NKV, HD, scale,
                    session.geom.pos_stride, session.geom.head_stride, mask,
                    num_blocks, top.index_block_size, NH / IH,
                    session.capacity, d_scores, d_attn);
            }
            if (e != cudaSuccess) return fail(e, LayerTag("attn", l));

            e = lsx_nvfp4::RunLsxNvfp4Gemm(stream, d_attn, lw.attn.o.d_packed,
                                           lw.attn.o.d_scales,
                                           lw.attn.o.global_scale, d_attn_out, 1,
                                           NH * HD, H, false);
            if (e != cudaSuccess) return fail(e, LayerTag("o", l));
            lsxcommon::kernels::AddInPlaceKernel<__half>
                <<<(H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    d_attn_out, d_act, H);
            e = cudaGetLastError();
            if (e != cudaSuccess) return fail(e, LayerTag("res1", l));

            // ── MLP block ──
            e = RunMinimaxNorm(stream, d_act, lw.attn.d_post_norm, d_postin, 1,
                               H, eps);
            if (e != cudaSuccess) return fail(e, LayerTag("postnorm", l));

            if (!lw.is_moe) {
                e = DenseMlp(stream, d_postin, lw.mlp_gate, lw.mlp_up,
                             lw.mlp_down, H, DI,
                             static_cast<float>(top.swiglu_alpha),
                             static_cast<float>(top.swiglu_limit), d_yg, d_yu,
                             d_h, d_mlp_out);
                if (e != cudaSuccess) return fail(e, LayerTag("mlp", l));
            } else {
                // Router (host readback of the top-k ids, one fence per layer).
                e = RunMinimaxRouter(stream, lw.d_router, lw.d_router_bias,
                                     d_postin, 1, E, K, H, d_slots);
                if (e != cudaSuccess) return fail(e, LayerTag("router", l));
                cudaMemcpyAsync(h_slots, d_slots,
                                sizeof(MinimaxRoutedSlot) *
                                    static_cast<size_t>(K),
                                cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(tok_ready, stream);
                cudaEventSynchronize(tok_ready);

                // Ungated shared expert.
                e = DenseMlp(stream, d_postin, lw.shared_gate, lw.shared_up,
                             lw.shared_down, H, SI,
                             static_cast<float>(top.swiglu_alpha),
                             static_cast<float>(top.swiglu_limit), d_yg, d_yu,
                             d_h, d_shared);
                if (e != cudaSuccess) return fail(e, LayerTag("shared", l));

                cudaMemsetAsync(d_moe_acc, 0,
                                static_cast<size_t>(H) * sizeof(float), stream);
                for (int k = 0; k < K; ++k) {
                    const int eid = h_slots[k].eid;
                    const float coeff = h_slots[k].coeff;
                    int evicted = -1;
                    const int slot = EnsureMinimaxExpertResident(
                        experts, cache, l, eid, stream, moe_error, nullptr,
                        &evicted);
                    if (slot < 0) return fail(cudaErrorUnknown, moe_error);
                    (void)evicted;
                    const auto &rec =
                        experts.records[static_cast<size_t>(l)]
                                       [static_cast<size_t>(eid)];
                    const uint8_t *pk = cache.d_pack +
                                        static_cast<int64_t>(slot) *
                                            cache.pack_stride;
                    const uint8_t *sc = cache.d_scale +
                                        static_cast<int64_t>(slot) *
                                            cache.scale_stride;
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(
                        stream, d_postin, pk, sc, rec.gs1, d_yg_slot, 1, H, I,
                        false);
                    if (e != cudaSuccess) return fail(e, LayerTag("w1", l));
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(
                        stream, d_postin, pk + rec.w1_bytes,
                        sc + rec.s1_bytes, rec.gs3, d_yu_slot, 1, H, I, false);
                    if (e != cudaSuccess) return fail(e, LayerTag("w3", l));
                    e = RunMinimaxSwiGLUOai(
                        stream, d_yg_slot, d_yu_slot, d_h_slot, I,
                        static_cast<float>(top.swiglu_alpha),
                        static_cast<float>(top.swiglu_limit));
                    if (e != cudaSuccess) return fail(e, LayerTag("swiglu", l));
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(
                        stream, d_h_slot, pk + rec.w1_bytes + rec.w3_bytes,
                        sc + rec.s1_bytes + rec.s3_bytes, rec.gs2, d_y2_slot,
                        1, I, H, false);
                    if (e != cudaSuccess) return fail(e, LayerTag("w2", l));
                    e = RunMinimaxRoutedAccumulate(stream, d_moe_acc, d_y2_slot,
                                                   coeff, H);
                    if (e != cudaSuccess) return fail(e, LayerTag("acc", l));
                }
                e = RunMinimaxMoECombine(
                    stream, d_moe_acc, d_shared, d_mlp_out, 1, H,
                    static_cast<float>(top.routed_scaling_factor));
                if (e != cudaSuccess) return fail(e, LayerTag("combine", l));
            }
            lsxcommon::kernels::AddInPlaceKernel<__half>
                <<<(H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    d_mlp_out, d_act, H);
            e = cudaGetLastError();
            if (e != cudaSuccess) return fail(e, LayerTag("res2", l));

            // Diagnostic (LSX_MINIMAX_DUMP_LAYERS=<file>): the first 16
            // hidden values after each layer of the FIRST generated step, so
            // tools/minimax_reference.py can bisect a numeric disagreement to
            // the layer that introduced it. Diagnostic only, never a result.
            if (const char *dump = std::getenv("LSX_MINIMAX_DUMP_LAYERS");
                dump && step == prompt_steps - 1) {
                std::vector<__half> hh(static_cast<size_t>(H));
                cudaMemcpyAsync(hh.data(), d_act, H * sizeof(__half),
                                cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(tok_ready, stream);
                cudaEventSynchronize(tok_ready);
                FILE *df = std::fopen(dump, "a");
                if (df) {
                    std::fprintf(df, "layer %d", l);
                    for (int i = 0; i < H; ++i)
                        std::fprintf(df, " %.6f",
                                     __half2float(hh[static_cast<size_t>(i)]));
                    std::fprintf(df, "\n");
                    std::fclose(df);
                }
            }
        }

        // ── final norm + lm_head ──
        e = RunMinimaxNorm(stream, d_act, pw.d_final_norm, d_postin, 1, H, eps);
        if (e != cudaSuccess) return fail(e, "final norm");
        // Diagnostic: the full pre/post final-norm rows of the step that
        // produced logits (LSX_MINIMAX_DUMP_FINAL=<prefix>).
        if (const char *dump = std::getenv("LSX_MINIMAX_DUMP_FINAL");
            dump && step == prompt_steps - 1) {
            std::vector<__half> hpre(static_cast<size_t>(H));
            std::vector<__half> hpost(static_cast<size_t>(H));
            cudaMemcpyAsync(hpre.data(), d_act, H * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(hpost.data(), d_postin, H * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
            FILE *fp = std::fopen(absl::StrCat(dump, ".pre.txt").c_str(), "w");
            FILE *fq = std::fopen(absl::StrCat(dump, ".post.txt").c_str(), "w");
            for (int i = 0; i < H; ++i) {
                if (fp) std::fprintf(fp, "%.6f\n", __half2float(hpre[static_cast<size_t>(i)]));
                if (fq) std::fprintf(fq, "%.6f\n", __half2float(hpost[static_cast<size_t>(i)]));
            }
            if (fp) std::fclose(fp);
            if (fq) std::fclose(fq);
        }
        e = gemv::RunLxGemmVec(stream, pw.d_head, d_postin, 1, V, H,
                               gemv::LxDtype::kF16, gemv::LxDtype::kF16, 1.0f,
                               d_logits, gemv::LxDtype::kF32, 0.0f);
        if (e != cudaSuccess) return fail(e, "head");
        session.kv_rows += 1;

        if (step < prompt_steps - 1) {
            cur = prefill[static_cast<size_t>(step) + 1];
            continue;
        }

        // Cooperative cancel: a request that was aborted before its first
        // decoded step must publish no tokens at all.
        if (gs && gs->Aborted()) {
            aborted = true;
            break;
        }

        // ── sample ──
        e = sampler.Sample(stream, d_logits, V, sample::LogitType::kF32,
                           d_argmax, d_maxv);
        if (e != cudaSuccess) return fail(e, "argmax");
        int32_t tok = -1;
        float maxv = 0.0f;
        cudaMemcpyAsync(&tok, d_argmax, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&maxv, d_maxv, sizeof(float), cudaMemcpyDeviceToHost,
                        stream);
        cudaEventRecord(tok_ready, stream);
        cudaEventSynchronize(tok_ready);
        if (generated == 0) {
            first_maxv = maxv;
            cudaEventRecord(t_ttft, stream);
        }
        tokens.push_back(tok);
        // Diagnostic (LSX_MINIMAX_DEBUG=1): the first step's prompt ids and
        // top-5 logits, for the host-reference oracle
        // (tools/minimax_reference.py). stderr only, never the result.
        if (generated == 0 && std::getenv("LSX_MINIMAX_DEBUG")) {
            std::vector<float> hl(static_cast<size_t>(V));
            cudaMemcpyAsync(hl.data(), d_logits,
                            static_cast<size_t>(V) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(tok_ready, stream);
            cudaEventSynchronize(tok_ready);
            std::vector<int> top(5, -1);
            std::vector<float> topv(5, -1e30f);
            for (int i = 0; i < V; ++i) {
                const float v = hl[static_cast<size_t>(i)];
                for (int k = 0; k < 5; ++k)
                    if (v > topv[static_cast<size_t>(k)]) {
                        for (int j = 4; j > k; --j) {
                            topv[static_cast<size_t>(j)] =
                                topv[static_cast<size_t>(j - 1)];
                            top[static_cast<size_t>(j)] =
                                top[static_cast<size_t>(j - 1)];
                        }
                        topv[static_cast<size_t>(k)] = v;
                        top[static_cast<size_t>(k)] = i;
                        break;
                    }
            }
            std::string pids;
            for (size_t i = 0; i < prefill.size(); ++i)
                pids += absl::StrCat(i ? "," : "", prefill[i]);
            std::string t5;
            for (int k = 0; k < 5; ++k)
                t5 += absl::StrCat(k ? "," : "", top[static_cast<size_t>(k)],
                                   ":", topv[static_cast<size_t>(k)]);
            fprintf(stderr, "[minimax-debug] prompt_ids=%s\n", pids.c_str());
            fprintf(stderr, "[minimax-debug] top5=%s\n", t5.c_str());
        }
        if (gs) {
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(generated), maxv);
            emitted = tokens.size();
            if (gs->Aborted()) {
                aborted = true;
                break;
            }
        }
        ++generated;
        if (!ignore_eos && eos_id >= 0 && tok == eos_id) {
            eos_terminated = true;
            break;
        }
        cur = tok;
    }

    // ── publish session state and drain results ──
    if (!tokens.empty()) session.last_tok = tokens.back();
    session.gen_tail = tokens;
    {
        std::vector<float> hlast(static_cast<size_t>(V));
        cudaMemcpyAsync(hlast.data(), d_logits,
                        static_cast<size_t>(V) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaError_t le = cudaGetLastError();
        cudaEventRecord(t_stop, stream);
        if (cudaEventSynchronize(t_stop) != cudaSuccess)
            return fail(cudaGetLastError(), "logits fence");
        if (le != cudaSuccess) return fail(le, "logits copy");
        out.final_logits = std::move(hlast);
    }
    float ms_total = 0.0f, ms_ttft = 0.0f;
    cudaEventElapsedTime(&ms_total, t_start, t_stop);
    // t_ttft is only recorded when a token was actually decoded; querying an
    // unrecorded event raises cudaErrorInvalidResourceHandle and leaves it as
    // the sticky last error, which then surfaces inside the NEXT request.
    ms_ttft = ms_total;
    if (!tokens.empty())
        (void)cudaEventElapsedTime(&ms_ttft, t_start, t_ttft);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_ttft);
    cudaEventDestroy(t_stop);
    cudaEventDestroy(tok_ready);

    out.gen_tokens = tokens;
    out.ttft_ms = static_cast<double>(ms_ttft);
    out.total_time_ms = static_cast<double>(ms_total);
    out.first_gen_tok_id = tokens.empty() ? -1 : tokens.front();
    out.first_gen_maxv = first_maxv;
    out.eos_terminated = eos_terminated;
    out.ok = true;
    out.weight_resident_bytes =
        pw.resident_bytes + scratch.arena_bytes + session.Bytes() +
        cache.DeviceBytes();
    out.num_tokens = static_cast<int>(tokens.size());
    if (out.total_time_ms > 0.0)
        out.tokens_per_sec =
            static_cast<double>(out.num_tokens) / (out.total_time_ms / 1000.0);
    out.kv_slots_used = static_cast<int>(session.kv_rows);
    (void)aborted;
    return out;
}

} // namespace lsxcommon::minimax
