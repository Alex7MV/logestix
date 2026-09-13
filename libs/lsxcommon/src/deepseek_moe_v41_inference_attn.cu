// deepseek_moe_v41_inference_attn.cu — DeepSeek-V4.1-Flash attention.
//
// Two single-token decode paths, both composing the neutral dense kernels
// (GEMM/RMSNorm/RoPE/FP8-FP4 quant/sparse-attn):
//   * AttentionForwardDecode      — pure sliding-window layers (ratio == 0)
//   * CedAttentionForwardDecode   — Compressed-Extended-Data layers (ratio > 0):
//     the window KV plus, on kv_source steps, a pooled compressed latent
//     (Compressor, model.py:429) that also feeds an Indexer (model.py:488) and
//     the two-level candidate mask (model.py:583).
#include "deepseek_moe_v41_inference.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <cuda_fp16.h>

#include "deepseek_moe_v41_dequant.h"

namespace lsxcommon {
namespace deepseek_v41 {

namespace {

__global__ void window_idxs_kernel(int *__restrict__ idxs, int window,
                                   int start_pos)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= window) return;
    int v;
    if (start_pos == 0) {
        v = t;
    } else {
        const int oldest = start_pos % window + 1;
        v = (t < window - oldest) ? (oldest + t) : (t - (window - oldest));
    }
    if (v > start_pos) v = -1;
    idxs[t] = v;
}

// Keep the `topk` best of `n_pos` scores (ties -> lower position), sorted
// ascending and shifted by `offset`; the tail up to `n_pad` is -1.
__global__ void indexer_topk_kernel(const float *__restrict__ score, int n_pos,
                                    int topk, int offset,
                                    int *__restrict__ out, int n_pad)
{
    extern __shared__ unsigned char smem[];
    int *sel = reinterpret_cast<int *>(smem);
    unsigned char *used = smem + static_cast<size_t>(topk) * sizeof(int);
    for (int i = threadIdx.x; i < n_pos; i += blockDim.x) used[i] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int k = 0; k < topk; ++k) {
        int best = -1;
        float bestv = 0.0f;
        for (int i = 0; i < n_pos; ++i) {
            if (used[i]) continue;
            const float v = score[i];
            if (best < 0 || v > bestv) { bestv = v; best = i; }
        }
        used[best] = 1;
        sel[k] = best;
    }
    for (int a = 1; a < topk; ++a) {
        const int key = sel[a];
        int b = a - 1;
        while (b >= 0 && sel[b] > key) { sel[b + 1] = sel[b]; --b; }
        sel[b + 1] = key;
    }
    for (int i = 0; i < n_pad; ++i) out[i] = (i < topk) ? sel[i] + offset : -1;
}

// Level-one candidate mask: block score = max over the block; the block
// holding the newest reachable position is pinned to +inf; the best
// `topk_blocks` finite blocks are kept.
__global__ void select_candidate_blocks_kernel(
    const float *__restrict__ score, int n_pos, int num_valid, int topk_blocks,
    int block_size, unsigned char *__restrict__ mask)
{
    extern __shared__ unsigned char smem[];
    const int nblocks = (n_pos + block_size - 1) / block_size;
    float *bs = reinterpret_cast<float *>(smem);
    unsigned char *used = smem + static_cast<size_t>(nblocks) * sizeof(float);
    for (int b = threadIdx.x; b < nblocks; b += blockDim.x) {
        float m = -INFINITY;
        for (int j = 0; j < block_size; ++j) {
            const int p = b * block_size + j;
            if (p < n_pos) m = fmaxf(m, score[p]);
        }
        bs[b] = m;
        used[b] = 0;
    }
    for (int i = threadIdx.x; i < n_pos; i += blockDim.x) mask[i] = 0;
    __syncthreads();
    if (threadIdx.x != 0) return;
    const int last = (num_valid - 1) / block_size;
    if (last >= 0 && last < nblocks) bs[last] = INFINITY;
    const int keep = topk_blocks < nblocks ? topk_blocks : nblocks;
    for (int k = 0; k < keep; ++k) {
        int best = -1;
        float bestv = 0.0f;
        for (int b = 0; b < nblocks; ++b) {
            if (used[b]) continue;
            const float v = bs[b];
            if (best < 0 || v > bestv) { bestv = v; best = b; }
        }
        used[best] = 1;
        if (bs[best] == -INFINITY) continue;
        for (int j = 0; j < block_size; ++j) {
            const int p = best * block_size + j;
            if (p < n_pos) mask[p] = 1;
        }
    }
}

__global__ void mask_indexer_scores_kernel(float *__restrict__ score, int n,
                                           const unsigned char *__restrict__ mask)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) score[i] = mask[i] ? score[i] : -INFINITY;
}

__global__ void scale_f32_kernel(float *__restrict__ x, int n, float s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void store_comp_state_kernel(const __half *__restrict__ kv,
                                        const __half *__restrict__ sc,
                                        float *__restrict__ kv_state,
                                        float *__restrict__ score_state, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    kv_state[i] = __half2float(kv[i]);
    score_state[i] = __half2float(sc[i]);
}

struct Scratch {
    uint16_t *qr = nullptr, *q = nullptr, *kv = nullptr, *o = nullptr,
             *olora = nullptr;
    int32_t *idxs = nullptr;
    void *base = nullptr;

    cudaError_t Alloc(int q_lora, int heads, int head_dim, int olora_total,
                      int window)
    {
        size_t bytes = 0;
        auto bump = [&](size_t n) { bytes += (n + 255) / 256 * 256; };
        bump(static_cast<size_t>(q_lora) * 2);
        bump(static_cast<size_t>(heads) * head_dim * 2);
        bump(static_cast<size_t>(head_dim) * 2);
        bump(static_cast<size_t>(heads) * head_dim * 2);
        bump(static_cast<size_t>(olora_total) * 2);
        bump(static_cast<size_t>(window) * sizeof(int32_t));
        if (cudaMalloc(&base, bytes) != cudaSuccess) return cudaGetLastError();
        uint8_t *p = static_cast<uint8_t *>(base);
        auto take = [&](size_t n) {
            uint8_t *r = p;
            p += (n + 255) / 256 * 256;
            return r;
        };
        qr = reinterpret_cast<uint16_t *>(take(q_lora * 2));
        q = reinterpret_cast<uint16_t *>(take(static_cast<size_t>(heads) * head_dim * 2));
        kv = reinterpret_cast<uint16_t *>(take(head_dim * 2));
        o = reinterpret_cast<uint16_t *>(take(static_cast<size_t>(heads) * head_dim * 2));
        olora = reinterpret_cast<uint16_t *>(take(olora_total * 2));
        idxs = reinterpret_cast<int32_t *>(take(window * sizeof(int32_t)));
        return cudaSuccess;
    }
    void Free()
    {
        if (base) cudaFree(base);
        base = nullptr;
    }
};

} // namespace

cudaError_t LaunchIndexerTopk(const float *score, int n_pos, int topk,
                              int offset, int32_t *out_idx, int n_pos_pad,
                              cudaStream_t stream)
{
    if (topk <= 0 || n_pos <= 0) return cudaSuccess;
    const size_t sh = static_cast<size_t>(topk) * sizeof(int) +
                      static_cast<size_t>(n_pos);
    indexer_topk_kernel<<<1, 256, sh, stream>>>(
        score, n_pos, topk, offset, reinterpret_cast<int *>(out_idx), n_pos_pad);
    return cudaGetLastError();
}

cudaError_t LaunchSelectCandidateBlocks(const float *score, int n_pos,
                                        int num_valid, int topk_blocks,
                                        int block_size, uint8_t *mask,
                                        cudaStream_t stream)
{
    if (n_pos <= 0 || block_size <= 0) return cudaSuccess;
    const int nblocks = (n_pos + block_size - 1) / block_size;
    const size_t sh = static_cast<size_t>(nblocks) * sizeof(float) +
                      static_cast<size_t>(nblocks);
    select_candidate_blocks_kernel<<<1, 256, sh, stream>>>(
        score, n_pos, num_valid, topk_blocks, block_size,
        reinterpret_cast<unsigned char *>(mask));
    return cudaGetLastError();
}

cudaError_t LaunchMaskIndexerScores(float *score, int n_pos, const uint8_t *mask,
                                    cudaStream_t stream)
{
    if (n_pos <= 0) return cudaSuccess;
    mask_indexer_scores_kernel<<<(n_pos + 255) / 256, 256, 0, stream>>>(
        score, n_pos, reinterpret_cast<const unsigned char *>(mask));
    return cudaGetLastError();
}

cudaError_t AttentionForwardDecode(
    cublasHandle_t handle, cudaStream_t stream, const uint16_t *x, int hidden,
    const uint16_t *wq_a, int q_lora, const uint16_t *q_norm,
    const uint16_t *wq_b, int heads, int head_dim, int rope_dim,
    const uint16_t *wkv, const uint16_t *kv_norm, const float *attn_sink,
    const uint16_t *wo_a, int o_groups, int o_lora_rank, const uint16_t *wo_b,
    const float *rope_cos, const float *rope_sin, uint16_t *kv_cache,
    int window, int start_pos, uint16_t *out)
{
    const int q_dim = heads * head_dim;
    Scratch sc;
    if (sc.Alloc(q_lora, heads, head_dim, o_groups * o_lora_rank, window) !=
        cudaSuccess)
        return cudaErrorInvalidValue;

    cudaError_t e = cudaSuccess;
    do {
        if ((e = LaunchGemmFp16(handle, stream, wq_a, x, sc.qr, q_lora, 1,
                                hidden)) != cudaSuccess)
            break;
        if ((e = LaunchRmsNormHalf(sc.qr, q_norm, sc.qr, q_lora, 1e-20f,
                                   stream)) != cudaSuccess)
            break;
        if ((e = LaunchGemmFp16(handle, stream, wq_b, sc.qr, sc.q, q_dim, 1,
                                q_lora)) != cudaSuccess)
            break;
        if ((e = LaunchRopeHalf(sc.q, heads, head_dim, rope_dim, rope_cos,
                                rope_sin, false, stream)) != cudaSuccess)
            break;
        if ((e = LaunchGemmFp16(handle, stream, wkv, x, sc.kv, head_dim, 1,
                                hidden)) != cudaSuccess)
            break;
        if ((e = LaunchRmsNormHalf(sc.kv, kv_norm, sc.kv, head_dim, 1e-20f,
                                   stream)) != cudaSuccess)
            break;
        if ((e = LaunchRopeHalf(sc.kv, 1, head_dim, rope_dim, rope_cos,
                                rope_sin, false, stream)) != cudaSuccess)
            break;
        if ((e = LaunchFp8ActQuantInplace(sc.kv, 1, head_dim, 32, stream)) !=
            cudaSuccess)
            break;

        const int slot = (start_pos % window) * head_dim;
        if ((e = cudaMemcpyAsync(kv_cache + slot, sc.kv,
                                 static_cast<size_t>(head_dim) * 2,
                                 cudaMemcpyDeviceToDevice, stream)) !=
            cudaSuccess)
            break;
        window_idxs_kernel<<<(window + 63) / 64, 64, 0, stream>>>(
            sc.idxs, window, start_pos);
        if ((e = cudaGetLastError()) != cudaSuccess) break;

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        if ((e = LaunchSparseAttnHalf(sc.q, 1, heads, head_dim, kv_cache,
                                      window, attn_sink, sc.idxs, window,
                                      sc.o, scale, stream)) != cudaSuccess)
            break;
        if ((e = LaunchRopeHalf(sc.o, heads, head_dim, rope_dim, rope_cos,
                                rope_sin, true, stream)) != cudaSuccess)
            break;

        const int per_group = head_dim * (heads / o_groups);
        for (int g = 0; g < o_groups; ++g) {
            const uint16_t *w_g = wo_a +
                static_cast<size_t>(g) * o_lora_rank * per_group;
            const uint16_t *x_g = sc.o + static_cast<size_t>(g) * per_group;
            uint16_t *y_g = sc.olora +
                static_cast<size_t>(g) * o_lora_rank;
            if ((e = LaunchGemmFp16(handle, stream, w_g, x_g, y_g, o_lora_rank,
                                    1, per_group)) != cudaSuccess)
                break;
        }
        if (e != cudaSuccess) break;

        if ((e = LaunchGemmFp16(handle, stream, wo_b, sc.olora, out, hidden, 1,
                                o_groups * o_lora_rank)) != cudaSuccess)
            break;
    } while (false);

    sc.Free();
    return e;
}

cudaError_t CedAttentionForwardDecode(
    cublasHandle_t handle, cudaStream_t stream, const uint16_t *x, int hidden,
    int window, int layer_id, const CedLayerParams &p, CedLayerBuffers &b,
    const float *rope_cos, const float *rope_sin, int start_pos,
    CsaRuntime &csa, uint16_t *out)
{
    (void)layer_id;
    const int hd = p.head_dim;
    const int rd = p.rope_dim;
    const int half = rd / 2;
    const int ratio = p.compress_ratio;
    const int end_pos = start_pos + 1;
    const int compress_len = end_pos / ratio;
    const int slot = start_pos % ratio;
    const int group_pos = start_pos + 1 - ratio;
    const float *cos_tok = rope_cos + static_cast<size_t>(start_pos) * half;
    const float *sin_tok = rope_sin + static_cast<size_t>(start_pos) * half;
    const float *cos_grp = rope_cos + static_cast<size_t>(group_pos) * half;
    const float *sin_grp = rope_sin + static_cast<size_t>(group_pos) * half;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

#define V41_TRY(expr)                     \
    do {                                  \
        cudaError_t _e = (expr);          \
        if (_e != cudaSuccess) return _e; \
    } while (false)

    // q path.
    V41_TRY(LaunchGemmFp16(handle, stream, p.wq_a, x, b.qr, p.q_lora, 1,
                           hidden));
    V41_TRY(LaunchRmsNormHalf(b.qr, p.q_norm, b.qr, p.q_lora, 1e-20f, stream));
    V41_TRY(LaunchGemmFp16(handle, stream, p.wq_b, b.qr, b.q, p.heads * hd, 1,
                           p.q_lora));
    V41_TRY(LaunchRopeHalf(b.q, p.heads, hd, rd, cos_tok, sin_tok, false,
                           stream));

    // Window KV: norm -> rope -> fp8 act-quant -> ring write.
    V41_TRY(LaunchGemmFp16(handle, stream, p.wkv, x, b.kv, hd, 1, hidden));
    V41_TRY(LaunchRmsNormHalf(b.kv, p.kv_norm, b.kv, hd, 1e-20f, stream));
    V41_TRY(LaunchRopeHalf(b.kv, 1, hd, rd, cos_tok, sin_tok, false, stream));
    V41_TRY(LaunchFp8ActQuantInplace(b.kv, 1, hd, 32, stream));
    V41_TRY(cudaMemcpyAsync(b.window_kv +
                                static_cast<size_t>(start_pos % window) * hd,
                            b.kv, static_cast<size_t>(hd) * 2,
                            cudaMemcpyDeviceToDevice, stream));
    window_idxs_kernel<<<(window + 63) / 64, 64, 0, stream>>>(b.win_idx,
                                                              window,
                                                              start_pos);
    V41_TRY(cudaGetLastError());

    // Compressor: a ratio-1 layer yields a latent every token; ratio > 1 fills
    // the partial-group state and only yields when the group completes.
    bool have_latent = false;
    if (p.is_kv_source) {
        if (ratio == 1) {
            V41_TRY(LaunchGemmFp16(handle, stream, p.comp_wkv, x, b.latent, hd,
                                   1, hidden));
            V41_TRY(LaunchRmsNormHalf(b.latent, p.comp_norm, b.latent, hd,
                                      1e-20f, stream));
            have_latent = true;
        } else {
            V41_TRY(LaunchGemmFp16(handle, stream, p.comp_wkv, x, b.kv, hd, 1,
                                   hidden));
            V41_TRY(LaunchGemmFp16(handle, stream, p.comp_wgate, x, b.latent,
                                   hd, 1, hidden));
            store_comp_state_kernel<<<(hd + 255) / 256, 256, 0, stream>>>(
                reinterpret_cast<const __half *>(b.kv),
                reinterpret_cast<const __half *>(b.latent),
                b.kv_state + static_cast<size_t>(slot) * hd,
                b.score_state + static_cast<size_t>(slot) * hd, hd);
            V41_TRY(cudaGetLastError());
            if (end_pos % ratio == 0) {
                V41_TRY(LaunchCompressorPool(b.kv_state, b.score_state, 1,
                                             ratio, hd, 0, b.latent_f, stream));
                V41_TRY(LaunchCastF32ToHalf(b.latent_f, b.latent, hd, stream));
                V41_TRY(LaunchRmsNormHalf(b.latent, p.comp_norm, b.latent, hd,
                                          1e-20f, stream));
                have_latent = true;
            }
        }
        // Consumers slice this cache by `compress_len`; publish the base.
        csa.PublishCompressKv(reinterpret_cast<const float *>(b.compress_kv),
                              compress_len);
    }

    // Indexer key ownership: derive keys from the pre-RoPE latent, before the
    // same buffer is rotated for the compressed-KV cache.
    if (p.owns_k && have_latent) {
        const int id = p.idx_head_dim;
        V41_TRY(LaunchGemmFp16(handle, stream, p.idx_wk, b.latent, b.idx_knew,
                               id, 1, hd));
        V41_TRY(LaunchRmsNormHalf(b.idx_knew, p.idx_k_norm, b.idx_knew, id,
                                  1e-20f, stream));
        V41_TRY(LaunchRopeHalf(b.idx_knew, 1, id, rd, cos_grp, sin_grp, false,
                               stream));
        V41_TRY(LaunchFp4ActQuantInplace(b.idx_knew, 1, id, 32, true, stream));
        V41_TRY(cudaMemcpyAsync(
            b.index_k + static_cast<size_t>(start_pos / ratio) * id,
            b.idx_knew, static_cast<size_t>(id) * 2,
            cudaMemcpyDeviceToDevice, stream));
    }
    if (p.owns_k)
        csa.PublishIndexK(b.index_k, compress_len);

    // Compressed-KV cache write (post-RoPE, fp4 block-16 E4M3).
    if (have_latent) {
        V41_TRY(LaunchRopeHalf(b.latent, 1, hd, rd, cos_grp, sin_grp, false,
                               stream));
        V41_TRY(LaunchFp4ActQuantInplace(b.latent, 1, hd, 16, false, stream));
        V41_TRY(cudaMemcpyAsync(
            b.compress_kv + static_cast<size_t>(start_pos / ratio) * hd,
            b.latent, static_cast<size_t>(hd) * 2,
            cudaMemcpyDeviceToDevice, stream));
    }

    // Indexer: score the compressed positions, select top-k (candidate mask
    // for layers downstream of the candidate source).
    int ctopk = 0;
    if (p.is_index_source) {
        const int id = p.idx_head_dim;
        V41_TRY(LaunchGemmFp16(handle, stream, p.idx_wq_b, b.qr, b.idx_q,
                               p.idx_n_heads * id, 1, p.q_lora));
        V41_TRY(LaunchRopeHalf(b.idx_q, p.idx_n_heads, id, rd, cos_tok,
                               sin_tok, false, stream));
        V41_TRY(LaunchFp4ActQuantInplace(b.idx_q, p.idx_n_heads, id, 32, true,
                                         stream));
        if (compress_len > 0) {
            V41_TRY(LaunchGemvFp16ToF32(handle, stream, p.idx_wproj, x,
                                        b.idx_w, p.idx_n_heads, hidden));
            const float ws = (1.0f / std::sqrt(static_cast<float>(id))) *
                             (1.0f / std::sqrt(static_cast<float>(
                                  p.idx_n_heads)));
            scale_f32_kernel<<<(p.idx_n_heads + 255) / 256, 256, 0, stream>>>(
                b.idx_w, p.idx_n_heads, ws);
            V41_TRY(cudaGetLastError());
            V41_TRY(LaunchIndexerScores(
                b.idx_q, p.idx_n_heads, id,
                reinterpret_cast<const uint16_t *>(csa.index_k), compress_len,
                b.idx_w, b.idx_score, stream));
            if (p.is_candidate_source) {
                V41_TRY(LaunchSelectCandidateBlocks(
                    b.idx_score, compress_len, compress_len,
                    p.candidate_topk_blocks, p.candidate_block_size,
                    b.candidates, stream));
                csa.PublishCandidates(b.candidates, compress_len);
            } else if (p.uses_candidates) {
                V41_TRY(LaunchMaskIndexerScores(b.idx_score, compress_len,
                                                csa.candidates, stream));
            }
            ctopk = std::min(p.idx_topk, compress_len);
            V41_TRY(LaunchIndexerTopk(b.idx_score, compress_len, ctopk, window,
                                      b.comp_idx, ctopk, stream));
        }
        csa.PublishTopk(b.comp_idx, ctopk);
    } else {
        ctopk = csa.topk;
    }

    // Assemble [window ring | compressed] and attend.
    V41_TRY(cudaMemcpyAsync(b.full_kv, b.window_kv,
                            static_cast<size_t>(window) * hd * 2,
                            cudaMemcpyDeviceToDevice, stream));
    const uint16_t *ckv = reinterpret_cast<const uint16_t *>(csa.compress_kv);
    if (compress_len > 0 && ckv)
        V41_TRY(cudaMemcpyAsync(
            b.full_kv + static_cast<size_t>(window) * hd, ckv,
            static_cast<size_t>(compress_len) * hd * 2,
            cudaMemcpyDeviceToDevice, stream));
    V41_TRY(cudaMemcpyAsync(b.full_idx, b.win_idx,
                            static_cast<size_t>(window) * sizeof(int32_t),
                            cudaMemcpyDeviceToDevice, stream));
    if (ctopk > 0 && csa.topk_idxs)
        V41_TRY(cudaMemcpyAsync(
            b.full_idx + window, csa.topk_idxs,
            static_cast<size_t>(ctopk) * sizeof(int32_t),
            cudaMemcpyDeviceToDevice, stream));

    V41_TRY(LaunchSparseAttnHalf(b.q, 1, p.heads, hd, b.full_kv,
                                 window + compress_len, p.attn_sink, b.full_idx,
                                 window + ctopk, b.o, scale, stream));
    V41_TRY(LaunchRopeHalf(b.o, p.heads, hd, rd, cos_tok, sin_tok, true,
                           stream));

    const int per_group = hd * (p.heads / p.o_groups);
    for (int g = 0; g < p.o_groups; ++g) {
        const uint16_t *w_g = p.wo_a +
            static_cast<size_t>(g) * p.o_lora_rank * per_group;
        const uint16_t *x_g = b.o + static_cast<size_t>(g) * per_group;
        uint16_t *y_g = b.olora + static_cast<size_t>(g) * p.o_lora_rank;
        V41_TRY(LaunchGemmFp16(handle, stream, w_g, x_g, y_g, p.o_lora_rank, 1,
                               per_group));
    }
    V41_TRY(LaunchGemmFp16(handle, stream, p.wo_b, b.olora, out, hidden, 1,
                           p.o_groups * p.o_lora_rank));
#undef V41_TRY
    return cudaSuccess;
}

} // namespace deepseek_v41
} // namespace lsxcommon
