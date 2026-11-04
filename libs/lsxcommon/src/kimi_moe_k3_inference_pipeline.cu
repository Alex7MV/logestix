// Kimi-K3 generation driver. One forward per token; the layer graph reproduces
// modeling_kimi_linear.KimiDecoderLayer._forward_attn_residual:
//
//   prefix_sum = hidden
//   [mix the attention input over (block_residual..., prefix_sum)]
//   [if layer % 12 == 0: push prefix_sum into block_residual; prefix_sum = none]
//   prefix_sum (+)= attn_out
//   hidden = mix(block_residual..., prefix_sum)          (mlp_res)
//   prefix_sum (+)= mlp_out
//   ... after the stack: mix(block_residual..., prefix_sum) with the output
//   AttnRes projection, then the final RMSNorm.
//
// Attention: KDA layers run the channel-wise delta recurrence; MLA layers are
// gated NoPE MLA with the value absorbed into the output projection, so the
// cache holds the 576-wide latent per position (KvCacheGeometry, pos-major).
#include "lsxcommon/kimi_moe_k3_inference_pipeline.h"

#include "kimi_moe_k3_constants.h"

#include "lsxcommon/kimi_moe_k3_kernels.h"

#include "lsxcommon/log.h"
#include "lsxcommon/lsx_gemv.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/lsx_moe_inference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace lsxcommon::kimi_k3 {
namespace {

using lsx_nvfp4::LsxNvfp4Weight;

struct Workspace {
    __half *x = nullptr;          // running prefix_sum
    __half *mixed = nullptr;
    __half *attn_out = nullptr;
    __half *sources = nullptr;    // [17][hidden]
    __half *res_blocks = nullptr; // [16][hidden]
    __half *qa = nullptr;
    __half *q = nullptr;
    __half *qabs = nullptr;
    __half *kv = nullptr;
    __half *attn_raw = nullptr;   // [heads*v] or the KDA output gate
    __half *gate = nullptr;       // f_a intermediate / MLA output gate
    __half *beta = nullptr;
    __half *kda_q = nullptr, *kda_k = nullptr, *kda_v = nullptr;
    __half *kda_g = nullptr, *kda_o = nullptr;
    __half *conv_q = nullptr, *conv_k = nullptr, *conv_v = nullptr;
    __half *lat = nullptr;
    __half *egate = nullptr, *eup = nullptr, *eact = nullptr, *edown = nullptr;
    __half *eacc = nullptr;
    __half *sg = nullptr, *su = nullptr, *sa = nullptr, *sd = nullptr;
    __half *dg = nullptr, *du = nullptr, *dd = nullptr;
    __half *logits = nullptr;
    float *sel = nullptr, *raw = nullptr;
    int32_t *idx = nullptr;
    float *wts = nullptr;
    uint8_t *expert_stage = nullptr;
    std::vector<void *> owned;

    void *Alloc(size_t bytes)
    {
        void *p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
        owned.push_back(p);
        return p;
    }
    void Free()
    {
        for (void *p : owned) cudaFree(p);
        owned.clear();
    }
};

template <typename T>
T *A(Workspace &w, size_t n)
{
    return static_cast<T *>(w.Alloc(n * sizeof(T)));
}

bool AllocWorkspace(Workspace &w, const KimiK3Topology &t, int64_t expert_bytes)
{
    const int H = t.hidden;
    const int Lat = t.latent;
    const int MI = t.moe_intermediate;
    const int SI = t.shared_intermediate;
    const int DI = t.dense_intermediate;
    const int q_rows = t.mla_heads * (t.qk_nope + t.qk_rope);
    const int attn_raw = t.mla_heads * t.v_head;
    const int kda_rows = t.kda_heads * t.kda_head_dim;

    w.x = A<__half>(w, H);
    w.mixed = A<__half>(w, H);
    w.attn_out = A<__half>(w, H);
    w.sources = A<__half>(w, static_cast<size_t>(17) * H);
    w.res_blocks = A<__half>(w, static_cast<size_t>(16) * H);
    w.qa = A<__half>(w, t.q_lora);
    w.q = A<__half>(w, std::max(q_rows, kda_rows));
    w.qabs = A<__half>(w, static_cast<size_t>(t.mla_heads) * t.kv_lora);
    w.kv = A<__half>(w, t.kv_lora + t.qk_rope);
    w.attn_raw = A<__half>(w, std::max(attn_raw, kda_rows));
    w.gate = A<__half>(w, std::max(kda_rows, attn_raw));
    w.beta = A<__half>(w, t.kda_heads);
    w.kda_q = A<__half>(w, kda_rows);
    w.kda_k = A<__half>(w, kda_rows);
    w.kda_v = A<__half>(w, kda_rows);
    w.kda_g = A<__half>(w, kda_rows);
    w.kda_o = A<__half>(w, kda_rows);
    w.conv_q = A<__half>(w, kda_rows);
    w.conv_k = A<__half>(w, kda_rows);
    w.conv_v = A<__half>(w, kda_rows);
    w.lat = A<__half>(w, Lat);
    w.egate = A<__half>(w, std::max(MI, DI));
    w.eup = A<__half>(w, std::max(MI, DI));
    w.eact = A<__half>(w, std::max(MI, DI));
    w.edown = A<__half>(w, std::max(Lat, H));
    w.eacc = A<__half>(w, Lat);
    w.sg = A<__half>(w, std::max(SI, DI));
    w.su = A<__half>(w, std::max(SI, DI));
    w.sa = A<__half>(w, std::max(SI, DI));
    w.sd = A<__half>(w, std::max(H, DI));
    w.dg = A<__half>(w, DI);
    w.du = A<__half>(w, DI);
    w.dd = A<__half>(w, DI);
    w.logits = A<__half>(w, t.vocab);
    w.sel = A<float>(w, t.num_experts);
    w.raw = A<float>(w, t.num_experts);
    w.idx = A<int32_t>(w, t.top_k);
    w.wts = A<float>(w, t.top_k);
    w.expert_stage = static_cast<uint8_t *>(w.Alloc(static_cast<size_t>(expert_bytes)));
    if (!(w.x && w.mixed && w.attn_out && w.sources && w.res_blocks && w.qa &&
           w.q && w.qabs && w.kv && w.attn_raw && w.gate && w.beta && w.kda_q &&
           w.kda_k && w.kda_v && w.kda_g && w.kda_o && w.conv_q && w.conv_k &&
           w.conv_v && w.lat && w.egate && w.eup && w.eact && w.edown &&
           w.eacc && w.sg && w.su && w.sa && w.sd && w.dg && w.du && w.dd &&
           w.logits && w.sel && w.raw && w.idx && w.wts && w.expert_stage)) {
        log::error("kimi-k3: workspace allocation failed");
        return false;
    }
    return true;
}

inline cudaError_t Proj(cudaStream_t s, const LsxNvfp4Weight &w,
                        const __half *x, __half *y)
{
    if (!w.d_packed) return cudaErrorInvalidValue;
    return lsx_nvfp4::RunLsxNvfp4Gemm(s, x, w.d_packed, w.d_scales,
                                      w.global_scale, y, 1, w.cols, w.rows,
                                      false);
}

int ExpertStageBytes(const KimiK3Topology &t)
{
    const int64_t lat = t.latent;
    const int64_t mi = t.moe_intermediate;
    const int64_t one = lat * mi / 2 + lat * mi / 16;
    return static_cast<int>(3 * one);
}

}  // namespace

cudaError_t RunKimiK3Generation(cudaStream_t stream, KimiK3PinnedWeights &pw,
                                KimiK3ExpertDb &experts,
                                const std::vector<int32_t> &prompt, int max_new,
                                int eos_id, int max_context,
                                sample::ISampler &sampler,
                                KimiK3GenerationResult &result,
                                GenerationStream *gs)
{
    const KimiK3Topology &top = pw.top;
    const int H = top.hidden;
    const int L = top.num_layers;
    const int KD = top.kda_head_dim;
    const int VD = top.kda_head_dim;
    const int KH = top.kda_heads;
    const int QH = top.qk_nope + top.qk_rope;
    const int rows_per_head = top.qk_nope + top.v_head;
    const int kvb_row_packed = top.kv_lora / 2;
    const int kvb_row_scales = top.kv_lora / 16;

    result.ok = false;
    result.dsa_steps = 0;
    result.dsa_keys_selected = 0;
    if (max_context <= 0) max_context = kDefaultMaxContext;
    if (static_cast<int>(prompt.size()) + max_new + 1 > max_context) {
        result.error_msg =
            "kimi-k3: prompt + max_new exceeds the context budget";
        return cudaErrorInvalidValue;
    }

    KvCacheGeometry kv_geo = KvCacheGeometry::Make(
        KvCacheLayout::kPosMajor, max_context, /*kv_heads=*/1,
        top.kv_lora + top.qk_rope, top.mla_layers.size());
    std::vector<int> mla_slot(static_cast<size_t>(L), -1);
    std::vector<int> kda_slot(static_cast<size_t>(L), -1);
    int mla_ord = 0, kda_ord = 0;
    for (int l = 0; l < L; ++l) {
        if (top.IsMla(l))
            mla_slot[static_cast<size_t>(l)] = mla_ord++;
        else
            kda_slot[static_cast<size_t>(l)] = kda_ord++;
    }

    __half *latent_cache = nullptr;
    float *kda_state = nullptr;
    float *conv_hist = nullptr;
    // Gated-DSA indexer state (Kimi-K3.1-Pro only; every pointer stays null on
    // a Kimi-K3 tree). The key cache is per DSA layer because the key is
    // LayerNorm(wk @ h) of THAT layer's attention input, not the shared latent.
    __half *index_k = nullptr;      // [dsa layers][max_context][index_dim]
    __half *index_q = nullptr;      // [index_heads * index_dim]
    __half *index_w = nullptr;      // [index_heads]
    float *index_score = nullptr;   // [max_context]
    int32_t *index_sel = nullptr;   // [index_topk]
    int *index_n = nullptr;         // device: the selected count
    const int dsa_count =
        top.dsa ? static_cast<int>(top.dsa_layers.size()) : 0;
    const int index_dim = top.index_head_dim;
    const int index_heads = top.index_heads;
    const int index_topk = top.index_topk;
    // v_beta^-0.5 and num_heads^-0.5: derived from the index geometry, so the
    // shared driver needs no include of the variant's passport. The passport's
    // kIndexQScale / kIndexWeightScale are the same two numbers.
    const float index_q_scale = 1.0f / std::sqrt(static_cast<float>(index_dim));
    const float index_w_scale =
        1.0f / std::sqrt(static_cast<float>(index_heads));
    const size_t index_stride = static_cast<size_t>(max_context) * index_dim;
    const size_t kv_bytes =
        static_cast<size_t>(kv_geo.Elements()) * sizeof(__half);
    const size_t index_bytes =
        static_cast<size_t>(dsa_count) * index_stride * sizeof(__half);
    const size_t kda_bytes =
        static_cast<size_t>(kda_ord) * KH * KD * VD * sizeof(float);
    // One conv history is (kernel-1) channels; three (q/k/v) per KDA layer.
    const size_t hist_stride = static_cast<size_t>(3 * (top.conv_kernel - 1) * KH * KD);
    const size_t hist_bytes = static_cast<size_t>(kda_ord) * hist_stride * sizeof(float);

    Workspace ws;
    const bool alloc_ok = AllocWorkspace(ws, top, ExpertStageBytes(top));
    const bool cuda_ok =
        alloc_ok && cudaMalloc(&latent_cache, kv_bytes) == cudaSuccess &&
        cudaMalloc(&kda_state, kda_bytes) == cudaSuccess &&
        cudaMalloc(&conv_hist, hist_bytes) == cudaSuccess &&
        (!top.dsa ||
         (cudaMalloc(&index_k, index_bytes) == cudaSuccess &&
          cudaMalloc(&index_q,
                     static_cast<size_t>(index_heads) * index_dim *
                         sizeof(__half)) == cudaSuccess &&
          cudaMalloc(&index_w,
                     static_cast<size_t>(index_heads) * sizeof(__half)) ==
              cudaSuccess &&
          cudaMalloc(&index_score,
                     static_cast<size_t>(max_context) * sizeof(float)) ==
              cudaSuccess &&
          cudaMalloc(&index_sel,
                     static_cast<size_t>(index_topk) * sizeof(int32_t)) ==
              cudaSuccess &&
          cudaMalloc(&index_n, sizeof(int)) == cudaSuccess));
    auto cleanup = [&]() {
        ws.Free();
        if (latent_cache) cudaFree(latent_cache);
        if (kda_state) cudaFree(kda_state);
        if (conv_hist) cudaFree(conv_hist);
        if (index_k) cudaFree(index_k);
        if (index_q) cudaFree(index_q);
        if (index_w) cudaFree(index_w);
        if (index_score) cudaFree(index_score);
        if (index_sel) cudaFree(index_sel);
        if (index_n) cudaFree(index_n);
        latent_cache = nullptr;
        kda_state = nullptr;
        conv_hist = nullptr;
        index_k = nullptr;
        index_q = nullptr;
        index_w = nullptr;
        index_score = nullptr;
        index_sel = nullptr;
        index_n = nullptr;
    };
    if (!cuda_ok) {
        result.error_msg = "kimi-k3: workspace allocation failed";
        cleanup();
        return cudaErrorMemoryAllocation;
    }
    cudaMemsetAsync(latent_cache, 0, kv_bytes, stream);
    cudaMemsetAsync(kda_state, 0, kda_bytes, stream);
    cudaMemsetAsync(conv_hist, 0, hist_bytes, stream);
    if (index_k) cudaMemsetAsync(index_k, 0, index_bytes, stream);

    const float eps = static_cast<float>(top.rms_eps);
    const float index_eps = static_cast<float>(top.index_norm_eps);
    const float situ_beta = static_cast<float>(top.situ_beta);
    const float situ_lb = static_cast<float>(top.situ_linear_beta);
    const float mla_scale = 1.0f / std::sqrt(static_cast<float>(QH));
    const int64_t embed_packed_stride = top.hidden / 2;
    const int64_t embed_scale_stride = top.hidden / 16;

    std::vector<__half> hlog(static_cast<size_t>(top.vocab));
    std::string stage = "start";
    // The last step that selected keys, for the end-of-run trace.
    int dsa_last_n_pos = 0;
    int dsa_last_n_sel = 0;

    int32_t *d_tok = nullptr;
    float *d_maxv = nullptr;
    cudaMalloc(&d_tok, sizeof(int32_t));
    cudaMalloc(&d_maxv, sizeof(float));

    auto release = [&](cudaError_t e) {
        if (d_tok) cudaFree(d_tok);
        if (d_maxv) cudaFree(d_maxv);
        cleanup();
        return e;
    };
    auto fail = [&](const char *why) {
        result.error_msg = std::string("kimi-k3 [") + stage + "]: " + why;
        return cudaErrorInvalidValue;
    };

    auto forward = [&](int32_t tok, int pos, bool want_logits) -> cudaError_t {
        stage = "embed";
        const uint8_t *erow =
            pw.embed.d_packed + (size_t)tok * embed_packed_stride;
        const uint8_t *srow =
            pw.embed.d_scales + (size_t)tok * embed_scale_stride;
        cudaError_t e = RunKimiK3DequantRow(stream, erow, srow,
                                            pw.embed.global_scale, ws.x, H);
        if (e) return e;

        int n_blocks = 0;
        bool have_prefix = false;

        for (int l = 0; l < L; ++l) {
            const KimiK3LayerWeights &lw = pw.layers[static_cast<size_t>(l)];

            // (a) attention-input mixture over the CURRENT block list.
            if (n_blocks > 0) {
                stage = "attn_res_mix";
                cudaMemcpyAsync(ws.sources, ws.res_blocks,
                                static_cast<size_t>(n_blocks) * H *
                                    sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(ws.sources + (size_t)n_blocks * H, ws.x,
                                static_cast<size_t>(H) * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                e = RunKimiK3DepthMix(stream, ws.sources, n_blocks + 1,
                                      lw.attn_res_norm, lw.attn_res_proj,
                                      ws.mixed, H, eps);
                if (e) return e;
            } else {
                cudaMemcpyAsync(ws.mixed, ws.x,
                                static_cast<size_t>(H) * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
            }
            // (b) block boundary: push the incoming prefix_sum.
            if (l % top.attn_res_block == 0) {
                if (n_blocks >= 16) return fail("AttnRes block overflow");
                cudaMemcpyAsync(ws.res_blocks + (size_t)n_blocks * H, ws.x,
                                static_cast<size_t>(H) * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                ++n_blocks;
                have_prefix = false;
            }
            stage = "input_norm";
            e = RunKimiK3RmsNorm(stream, ws.mixed, lw.input_norm, ws.mixed, 1,
                                 H, eps);
            if (e) return e;

            if (lw.mla) {
                const int dsa_slot = mla_slot[static_cast<size_t>(l)];
                // Gated DSA: the key of THIS position is
                // LayerNorm(wk @ h) over the vector the MLA projections
                // consume, cached per position because a later position's
                // selection needs it and the attention input is not retained.
                if (lw.idx_wk) {
                    stage = "mla_indexer_key";
                    e = RunKimiK3IndexerKey(
                        stream, ws.mixed, lw.idx_wk, lw.idx_knorm_w,
                        lw.idx_knorm_b,
                        index_k + static_cast<size_t>(dsa_slot) * index_stride +
                            static_cast<size_t>(pos) * index_dim,
                        H, index_dim, index_eps);
                    if (e) return e;
                }
                stage = "mla_qa";
                e = Proj(stream, lw.mla_qa, ws.mixed, ws.qa);
                if (e) return e;
                e = RunKimiK3RmsNorm(stream, ws.qa, lw.mla_qanorm, ws.qa, 1,
                                     top.q_lora, eps);
                if (e) return e;
                // The indexer query reads the normalised q-lora (`qr`) and its
                // head weights read the same attention input, so both are
                // computed here, once per token per DSA layer.
                if (lw.idx_wk) {
                    stage = "mla_indexer_query";
                    e = RunKimiK3IndexerQuery(
                        stream, ws.qa, ws.mixed, lw.idx_wqb, lw.idx_wproj,
                        index_q, index_w, top.q_lora, H, index_heads,
                        index_dim, index_q_scale, index_w_scale);
                    if (e) return e;
                }
                stage = "mla_qb";
                e = Proj(stream, lw.mla_qb, ws.qa, ws.q);
                if (e) return e;
                stage = "mla_kva";
                e = Proj(stream, lw.mla_kva, ws.mixed, ws.kv);
                if (e) return e;
                stage = "mla_latent_norm";
                e = RunKimiK3RmsNorm(stream, ws.kv, lw.mla_kvanorm, ws.kv, 1,
                                     top.kv_lora, eps);
                if (e) return e;
                {
                    const int slot = mla_slot[static_cast<size_t>(l)];
                    __half *layer_cache =
                        latent_cache +
                        static_cast<size_t>(slot) * kv_geo.LayerStride();
                    __half *row = layer_cache +
                                  static_cast<size_t>(pos) * kv_geo.pos_stride;
                    stage = "mla_cache";
                    cudaMemcpyAsync(row, ws.kv,
                                    static_cast<size_t>(top.kv_lora) *
                                        sizeof(__half),
                                    cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(row + top.kv_lora,
                                    ws.kv + top.kv_lora,
                                    static_cast<size_t>(top.qk_rope) *
                                        sizeof(__half),
                                    cudaMemcpyDeviceToDevice, stream);
                    stage = "mla_absorb";
                    e = RunKimiK3MlaAbsorbQ(
                        stream, ws.q, lw.mla_kvb.d_packed, lw.mla_kvb.d_scales,
                        lw.mla_kvb.global_scale, ws.qabs, top.mla_heads,
                        top.qk_nope, top.kv_lora, rows_per_head, 0, QH,
                        kvb_row_packed, kvb_row_scales);
                    if (e) return e;
                    stage = "mla_attn";
                    // Above index_topk the visible keys are selected first:
                    // score[s] over every causal position, then the exact
                    // top-k, then the attention walks only those positions.
                    // At or below it the selection is the identity and is
                    // bypassed (sel == nullptr reproduces the dense loop
                    // verbatim).
                    const int32_t *sel = nullptr;
                    int n_sel = 0;
                    if (lw.idx_wk && pos + 1 > index_topk) {
                        stage = "mla_indexer_scores";
                        e = RunKimiK3IndexerScores(
                            stream, index_q, index_w,
                            index_k + static_cast<size_t>(dsa_slot) *
                                          index_stride,
                            pos + 1, index_heads, index_dim, index_score);
                        if (e) return e;
                        stage = "mla_indexer_topk";
                        e = RunKimiK3IndexerTopK(stream, index_score, pos + 1,
                                                 index_topk, index_sel,
                                                 index_n);
                        if (e) return e;
                        // RunKimiK3IndexerTopK always emits exactly
                        // min(k, n_pos) positions, so the count is known on
                        // the host and no device-to-host read is needed.
                        n_sel = std::min(index_topk, pos + 1);
                        sel = index_sel;
                    }
                    stage = "mla_attn";
                    e = RunKimiK3MlaAttention(
                        stream, layer_cache, pos + 1, ws.qabs,
                        ws.q + top.qk_nope, lw.mla_kvb.d_packed,
                        lw.mla_kvb.d_scales, lw.mla_kvb.global_scale,
                        ws.attn_raw, top.mla_heads, top.kv_lora, top.qk_rope,
                        top.v_head, rows_per_head, top.qk_nope, QH, mla_scale,
                        kvb_row_packed, kvb_row_scales, sel, n_sel);
                    if (e) return e;
                    if (sel) {
                        ++result.dsa_steps;
                        result.dsa_keys_selected += n_sel;
                        dsa_last_n_pos = pos + 1;
                        dsa_last_n_sel = n_sel;
                    }
                }
                stage = "mla_gate";
                e = Proj(stream, lw.mla_g, ws.mixed, ws.gate);
                if (e) return e;
                e = RunKimiK3SigmoidMul(stream, ws.attn_raw, ws.gate,
                                        static_cast<int64_t>(top.mla_heads) *
                                            top.v_head);
                if (e) return e;
                stage = "mla_o";
                e = Proj(stream, lw.mla_o, ws.attn_raw, ws.attn_out);
                if (e) return e;
            } else {
                stage = "kda_proj";
                e = Proj(stream, lw.kda_q, ws.mixed, ws.kda_q);
                if (e) return e;
                e = Proj(stream, lw.kda_k, ws.mixed, ws.kda_k);
                if (e) return e;
                e = Proj(stream, lw.kda_v, ws.mixed, ws.kda_v);
                if (e) return e;
                stage = "kda_decay";
                // The decay gate is a low-rank pair (f_a: hidden -> head_dim,
                // f_b: head_dim -> heads*head_dim), never a fused projection.
                const int gate_rank = top.kda_head_dim;
                e = gemv::RunLxGemmVec(stream, lw.kda_fa, ws.mixed, 1, gate_rank,
                                       H, gemv::LxDtype::kF16,
                                       gemv::LxDtype::kF16, 1.0f, ws.gate,
                                       gemv::LxDtype::kF16, 0.0f);
                if (e) return e;
                e = gemv::RunLxGemmVec(stream, lw.kda_fb, ws.gate, 1, KH * KD,
                                       gate_rank, gemv::LxDtype::kF16,
                                       gemv::LxDtype::kF16, 1.0f, ws.kda_g,
                                       gemv::LxDtype::kF16, 0.0f);
                if (e) return e;
                e = gemv::RunLxGemmVec(stream, lw.kda_b, ws.mixed, 1, KH, H,
                                       gemv::LxDtype::kF16, gemv::LxDtype::kF16,
                                       1.0f, ws.beta, gemv::LxDtype::kF16, 0.0f);
                if (e) return e;
                stage = "kda_o_gate";
                e = Proj(stream, lw.kda_g, ws.mixed, ws.attn_raw);
                if (e) return e;

                const int slot = kda_slot[static_cast<size_t>(l)];
                float *hq = conv_hist + (size_t)slot * hist_stride;
                float *hk = hq + (size_t)KH * KD * 3;
                float *hv = hk + (size_t)KH * KD * 3;
                stage = "kda_conv";
                e = RunKimiK3KdaConv(stream, ws.kda_q, lw.kda_conv_q, hq,
                                     ws.conv_q, KH * KD, top.conv_kernel);
                if (e) return e;
                e = RunKimiK3KdaConv(stream, ws.kda_k, lw.kda_conv_k, hk,
                                     ws.conv_k, KH * KD, top.conv_kernel);
                if (e) return e;
                e = RunKimiK3KdaConv(stream, ws.kda_v, lw.kda_conv_v, hv,
                                     ws.conv_v, KH * KD, top.conv_kernel);
                if (e) return e;
                stage = "kda_recur";
                e = RunKimiK3KdaRecur(
                    stream, ws.conv_q, ws.conv_k, ws.conv_v, ws.kda_g, ws.beta,
                    lw.kda_alog, lw.kda_dt,
                    kda_state + (size_t)slot * KH * KD * VD, ws.kda_o, KH, KH,
                    KD, VD, static_cast<float>(top.kda_gate_lower_bound));
                if (e) return e;
                stage = "kda_o_norm";
                e = RunKimiK3GatedRmsNorm(stream, ws.kda_o, lw.kda_onorm,
                                          ws.attn_raw, ws.kda_o, KH, KD, eps);
                if (e) return e;
                stage = "kda_o";
                e = Proj(stream, lw.kda_o, ws.kda_o, ws.attn_out);
                if (e) return e;
            }

            stage = "prefix_attn";
            if (have_prefix) {
                e = RunKimiK3Axpy(stream, ws.x, ws.attn_out, 1.0f, H);
                if (e) return e;
            } else {
                cudaMemcpyAsync(ws.x, ws.attn_out,
                                static_cast<size_t>(H) * sizeof(__half),
                                cudaMemcpyDeviceToDevice, stream);
                have_prefix = true;
            }

            stage = "mlp_res_mix";
            cudaMemcpyAsync(ws.sources, ws.res_blocks,
                            static_cast<size_t>(n_blocks) * H * sizeof(__half),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(ws.sources + (size_t)n_blocks * H, ws.x,
                            static_cast<size_t>(H) * sizeof(__half),
                            cudaMemcpyDeviceToDevice, stream);
            e = RunKimiK3DepthMix(stream, ws.sources, n_blocks + 1,
                                  lw.mlp_res_norm, lw.mlp_res_proj, ws.mixed, H,
                                  eps);
            if (e) return e;
            stage = "post_norm";
            e = RunKimiK3RmsNorm(stream, ws.mixed, lw.post_norm, ws.mixed, 1, H,
                                 eps);
            if (e) return e;

            if (!lw.has_moe) {
                stage = "dense_gate_up";
                e = Proj(stream, lw.dense_gate, ws.mixed, ws.dg);
                if (e) return e;
                e = Proj(stream, lw.dense_up, ws.mixed, ws.du);
                if (e) return e;
                e = RunKimiK3Situ(stream, ws.dg, ws.du, ws.dg,
                                  top.dense_intermediate, situ_beta, situ_lb);
                if (e) return e;
                stage = "dense_down";
                e = Proj(stream, lw.dense_down, ws.dg, ws.attn_out);
                if (e) return e;
            } else {
                stage = "moe_stage";
                e = StageKimiK3Layer(pw, l, stream);
                if (e) return e;
                stage = "moe_router";
                e = RunKimiK3RouterScores(stream, ws.mixed, lw.router_gate,
                                          lw.router_bias, H, top.num_experts,
                                          ws.sel, ws.raw);
                if (e) return e;
                e = RunKimiK3RouterTopK(stream, ws.sel, ws.raw, top.num_experts,
                                        top.top_k,
                                        static_cast<float>(top.route_scale),
                                        top.moe_renormalize ? 1 : 0, ws.idx,
                                        ws.wts);
                if (e) return e;
                std::vector<int32_t> hidx(static_cast<size_t>(top.top_k));
                std::vector<float> hw(static_cast<size_t>(top.top_k));
                cudaMemcpyAsync(hidx.data(), ws.idx,
                                hidx.size() * sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(hw.data(), ws.wts, hw.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                cudaMemsetAsync(ws.eacc, 0,
                                static_cast<size_t>(top.latent) *
                                    sizeof(__half),
                                stream);
                stage = "moe_latent_down";
                e = Proj(stream, pw.stream_dev.latent_down, ws.mixed, ws.lat);
                if (e) return e;

                const int64_t mi = top.moe_intermediate;
                const int64_t lat = top.latent;
                const int64_t packed_w = lat * mi / 2;
                const int64_t scales_w = lat * mi / 16;
                const int64_t one = packed_w + scales_w;
                uint8_t *p1 = ws.expert_stage;
                uint8_t *s1 = p1 + packed_w;
                uint8_t *p2 = ws.expert_stage + one;
                uint8_t *s2 = p2 + packed_w;
                uint8_t *p3 = ws.expert_stage + 2 * one;
                uint8_t *s3 = p3 + packed_w;
                for (int k = 0; k < top.top_k; ++k) {
                    const int eid = hidx[static_cast<size_t>(k)];
                    EnsureKimiK3ExpertResident(experts, l, eid, stream);
                    const size_t pe_idx =
                        static_cast<size_t>(l) * top.num_experts +
                        static_cast<size_t>(eid);
                    if (pe_idx >= experts.entries.size())
                        return fail("expert index out of range");
                    const KimiK3ExpertEntry &pe = experts.entries[pe_idx];
                    if (!pe.host_w1 || !pe.host_w2 || !pe.host_w3)
                        return fail("routed expert view is not resident");
                    stage = "moe_h2d";
                    cudaMemcpyAsync(p1, pe.host_w1,
                                    static_cast<size_t>(pe.w1_bytes),
                                    cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(s1, pe.host_w1_scales,
                                    static_cast<size_t>(pe.w1_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(p2, pe.host_w3,
                                    static_cast<size_t>(pe.w3_bytes),
                                    cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(s2, pe.host_w3_scales,
                                    static_cast<size_t>(pe.w3_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(p3, pe.host_w2,
                                    static_cast<size_t>(pe.w2_bytes),
                                    cudaMemcpyHostToDevice, stream);
                    cudaMemcpyAsync(s3, pe.host_w2_scales,
                                    static_cast<size_t>(pe.w2_scales_bytes),
                                    cudaMemcpyHostToDevice, stream);
                    stage = "moe_expert";
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(
                        stream, ws.lat, p1, s1, pe.w1_global_scale, ws.egate, 1,
                        static_cast<int>(lat), static_cast<int>(mi), false);
                    if (e) return e;
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(
                        stream, ws.lat, p2, s2, pe.w3_global_scale, ws.eup, 1,
                        static_cast<int>(lat), static_cast<int>(mi), false);
                    if (e) return e;
                    e = RunKimiK3Situ(stream, ws.egate, ws.eup, ws.eact, mi,
                                      situ_beta, situ_lb);
                    if (e) return e;
                    e = lsx_nvfp4::RunLsxNvfp4Gemm(
                        stream, ws.eact, p3, s3, pe.w2_global_scale, ws.edown, 1,
                        static_cast<int>(mi), static_cast<int>(lat), false);
                    if (e) return e;
                    e = RunKimiK3Axpy(stream, ws.eacc, ws.edown,
                                      hw[static_cast<size_t>(k)],
                                      static_cast<int64_t>(lat));
                    if (e) return e;
                }
                stage = "moe_latent_norm";
                if (top.latent_norm) {
                    e = RunKimiK3RmsNorm(stream, ws.eacc, lw.latent_norm,
                                         ws.eacc, 1, top.latent, eps);
                    if (e) return e;
                }
                stage = "moe_latent_up";
                e = Proj(stream, pw.stream_dev.latent_up, ws.eacc, ws.attn_out);
                if (e) return e;

                stage = "moe_shared";
                e = Proj(stream, pw.stream_dev.shared_gate, ws.mixed, ws.sg);
                if (e) return e;
                e = Proj(stream, pw.stream_dev.shared_up, ws.mixed, ws.su);
                if (e) return e;
                e = RunKimiK3Situ(stream, ws.sg, ws.su, ws.sa,
                                  top.shared_intermediate, situ_beta, situ_lb);
                if (e) return e;
                e = Proj(stream, pw.stream_dev.shared_down, ws.sa, ws.sd);
                if (e) return e;
                e = RunKimiK3Axpy(stream, ws.attn_out, ws.sd, 1.0f, H);
                if (e) return e;
            }

            stage = "prefix_mlp";
            e = RunKimiK3Axpy(stream, ws.x, ws.attn_out, 1.0f, H);
            if (e) return e;
        }

        if (!want_logits) return cudaSuccess;

        stage = "output_mix";
        cudaMemcpyAsync(ws.sources, ws.res_blocks,
                        static_cast<size_t>(n_blocks) * H * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(ws.sources + (size_t)n_blocks * H, ws.x,
                        static_cast<size_t>(H) * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
        e = RunKimiK3DepthMix(stream, ws.sources, n_blocks + 1,
                              pw.out_res_norm, pw.out_res_proj, ws.mixed, H, eps);
        if (e) return e;
        stage = "final_norm";
        e = RunKimiK3RmsNorm(stream, ws.mixed, pw.final_norm, ws.mixed, 1, H,
                             eps);
        if (e) return e;
        stage = "head";
        return Proj(stream, pw.head, ws.mixed, ws.logits);
    };

    const auto t_start = std::chrono::steady_clock::now();

    // ── prefill: one forward per prompt token ──
    int pos = 0;
    for (; pos < static_cast<int>(prompt.size()); ++pos) {
        if (gs != nullptr && gs->Aborted()) {
            result.ok = true;
            return release(cudaSuccess);
        }
        const bool last = (pos == static_cast<int>(prompt.size()) - 1);
        cudaError_t e = forward(prompt[static_cast<size_t>(pos)], pos, last);
        if (e) return release(e);
    }
    result.ttft_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_start)
                         .count();

    // ── decode ──
    for (int step = 0; step < max_new; ++step) {
        if (gs != nullptr && gs->Aborted()) break;
        stage = "sample";
        cudaError_t e = sampler.Sample(stream, ws.logits, top.vocab,
                                       sample::LogitType::kF16, d_tok, d_maxv);
        if (e) return release(e);
        int32_t best = -1;
        float maxv = 0.0f;
        cudaMemcpyAsync(&best, d_tok, sizeof(best), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(&maxv, d_maxv, sizeof(maxv), cudaMemcpyDeviceToHost,
                        stream);
        cudaMemcpyAsync(hlog.data(), ws.logits,
                        static_cast<size_t>(top.vocab) * sizeof(__half),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (result.first_token < 0) result.first_token = best;
        result.final_logits.resize(static_cast<size_t>(top.vocab));
        for (int i = 0; i < top.vocab; ++i)
            result.final_logits[static_cast<size_t>(i)] =
                __half2float(hlog[static_cast<size_t>(i)]);
        result.tokens.push_back(best);
        if (gs != nullptr)
            gs->Emit(static_cast<std::uint32_t>(best),
                     static_cast<std::uint32_t>(step), maxv);
        if (eos_id >= 0 && best == eos_id) {
            result.eos_terminated = true;
            break;
        }
        if (step + 1 >= max_new) break;
        e = forward(best, pos, true);
        if (e) return release(e);
        ++pos;
    }
    result.total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_start)
                          .count();
    // The device buffers of the last engaged step are still live, so the trace
    // is one copy at the end of the run rather than one per step.
    if (result.dsa_steps > 0) {
        result.dsa_last_n_pos = dsa_last_n_pos;
        result.dsa_last_score.resize(static_cast<size_t>(dsa_last_n_pos));
        result.dsa_last_selection.resize(static_cast<size_t>(dsa_last_n_sel));
        cudaMemcpyAsync(result.dsa_last_score.data(), index_score,
                        static_cast<size_t>(dsa_last_n_pos) * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(result.dsa_last_selection.data(), index_sel,
                        static_cast<size_t>(dsa_last_n_sel) * sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    result.ok = true;
    return release(cudaSuccess);
}

}  // namespace lsxcommon::kimi_k3
