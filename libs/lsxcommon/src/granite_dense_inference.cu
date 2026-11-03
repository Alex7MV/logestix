// Granite-4.2-30B dense decoder runtime (Model Ingestion Protocol step 5).
//
// One stream, no per-token cudaDeviceSynchronize. All dense projections go
// through the model-neutral NVFP4 layer (lsx_nvfp4::RunLsxNvfp4Gemm); norms,
// SwiGLU and residual adds use the shared model-neutral lsx_kernels_*.cuh
// primitives. The only family-local kernels are full-dimension RoPE and the
// causal GQA attention (both deterministic, block-reduction based).

#include "lsxcommon/granite_dense_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>

#include "lsxcommon/granite_dense_constants.h"
#include "lsxcommon/lsx_dense_inference.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/shared_experts.h"

#include "lsx_kernels_act.cuh"
#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"

namespace lsxcommon {
namespace granite {

using lsx_nvfp4::RunLsxNvfp4Gemm;

namespace {

constexpr int kThreads = 128;

// Split-KV geometry constants stay here because the driver sizes its partials
// slab from them; the kernels live in the neutral layer (lsx_dense_inference).
constexpr int kAttnSplitMax = 32;
constexpr int kAttnSplitMaxSeq = 8;

// ── Launchers (delegates to the neutral dense layer) ────────────────────────

cudaError_t GraniteRmsNorm(cudaStream_t stream, __half *out, const __half *x,
                           const __half *w, int rows, int hidden, float eps)
{
    return dense::RunDenseRmsNormKernel(stream, out, x, w, rows, hidden, eps);
}

cudaError_t GraniteRope(cudaStream_t stream, __half *q, __half *k,
                        const int32_t *pos, int seq, int nh, int nkv, int hd,
                        double theta)
{
    return dense::RunDenseRopeKernel(stream, q, k, pos, seq, nh, nkv, hd, theta);
}

cudaError_t GraniteAttentionSplit(cudaStream_t stream, const __half *q,
                                  const __half *k, const __half *v, __half *out,
                                  float *partials, int seq, int nh, int nkv,
                                  int hd, float scale, int kv_base, int splits)
{
    return dense::RunDenseAttentionSplitKernel(stream, q, k, v, out, partials,
                                               seq, nh, nkv, hd, scale, kv_base,
                                               splits);
}

cudaError_t GraniteAttention(cudaStream_t stream, const __half *q,
                             const __half *k, const __half *v, __half *out,
                             float *partials, int seq, int nh, int nkv, int hd,
                             float scale, int kv_base)
{
    return dense::RunDenseAttentionKernel(stream, q, k, v, out, partials, seq,
                                          nh, nkv, hd, scale, kv_base);
}

cudaError_t GraniteSplitQKV(cudaStream_t stream, const __half *qkv, __half *q,
                            __half *k, __half *v, int seq, int qn, int kn)
{
    return dense::RunDenseSplitQKVKernel(stream, qkv, q, k, v, seq, qn, kn);
}

cudaError_t GraniteSwiGLU(cudaStream_t stream, const __half *gu, __half *out,
                          int I, int64_t n)
{
    return dense::RunDenseSwiGLUKernel(stream, gu, out, I, n);
}

// ── Workspace ───────────────────────────────────────────────────────────────

struct Workspace {
    __half *x = nullptr;
    __half *norm = nullptr;
    __half *q = nullptr;
    __half *k = nullptr;
    __half *v = nullptr;
    __half *qkv = nullptr;
    __half *attn = nullptr;
    __half *proj = nullptr;
    __half *gate_up = nullptr;
    __half *act = nullptr;
    __half *down = nullptr;
    __half *logits = nullptr;
    __half *embed_h = nullptr;
    int32_t *ids = nullptr;
    int32_t *pos = nullptr;
    int32_t *tok = nullptr;
    float *maxv = nullptr;
    float *attn_part = nullptr;   // split-KV softmax partials [splits,seq,nh,hd+2]
    int max_seq = 0;

    cudaError_t Alloc(const GraniteTopology &top, int max_seq_);
    void Free();
};

void Workspace::Free()
{
    cudaFree(x); cudaFree(norm); cudaFree(q); cudaFree(k); cudaFree(v);
    cudaFree(attn); cudaFree(proj); cudaFree(gate_up); cudaFree(act);
    cudaFree(qkv);
    cudaFree(down); cudaFree(logits); cudaFree(embed_h);
    cudaFree(ids); cudaFree(pos); cudaFree(tok); cudaFree(maxv);
    cudaFree(attn_part);
    *this = Workspace{};
}

cudaError_t Workspace::Alloc(const GraniteTopology &top, int max_seq_)
{
    max_seq = max_seq_;
    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const int V = top.vocab_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const int64_t S = max_seq;

    auto alloc = [](void **p, int64_t bytes) -> cudaError_t {
        return cudaMalloc(p, static_cast<size_t>(bytes));
    };
    cudaError_t e = cudaSuccess;
    if ((e = alloc(reinterpret_cast<void **>(&x), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&norm), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&q), S * NH * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&k), S * NKV * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&v), S * NKV * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&attn), S * NH * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&qkv),
                   S * (NH + 2 * NKV) * HD * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&proj), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&gate_up), S * 2 * I * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&act), S * I * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&down), S * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&logits),
                   static_cast<int64_t>(V) * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&embed_h),
                   static_cast<int64_t>(V) * H * 2))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&ids), S * sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&pos), S * sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&tok), sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&maxv), sizeof(float)))) return e;
    // Split-KV partials only ever hold the short-seq (decode) regime; long-seq
    // prefill uses the unsplit kernel (seq*nh blocks are plentiful there).
    const int64_t sp_seq = std::min<int64_t>(S, kAttnSplitMaxSeq);
    if ((e = alloc(reinterpret_cast<void **>(&attn_part),
                   static_cast<int64_t>(kAttnSplitMax) * sp_seq * NH * (HD + 2) *
                       sizeof(float))))
        return e;
    return cudaSuccess;
}

} // namespace

// ── KV cache ────────────────────────────────────────────────────────────────

void GraniteKvCache::Free()
{
    cudaFree(d_k);
    cudaFree(d_v);
    d_k = nullptr;
    d_v = nullptr;
    last_tok = -1;
    kv_rows = 0;
    max_total = 0;
    cache_bytes = 0;
}

cudaError_t GraniteKvCache::Allocate(const GranitePinnedWeights &pw,
                                     int max_total_, cudaStream_t stream)
{
    Free();
    const int64_t row = static_cast<int64_t>(pw.top.num_key_value_heads) *
                        pw.top.head_dim;
    const int64_t bytes = static_cast<int64_t>(pw.top.num_layers) * max_total_ *
                          row * sizeof(__half);
    cudaError_t e = cudaMalloc(reinterpret_cast<void **>(&d_k),
                               static_cast<size_t>(bytes));
    if (e != cudaSuccess) return e;
    e = cudaMalloc(reinterpret_cast<void **>(&d_v), static_cast<size_t>(bytes));
    if (e != cudaSuccess) { Free(); return e; }
    e = cudaMemsetAsync(d_k, 0, static_cast<size_t>(bytes), stream);
    if (e != cudaSuccess) { Free(); return e; }
    e = cudaMemsetAsync(d_v, 0, static_cast<size_t>(bytes), stream);
    if (e != cudaSuccess) { Free(); return e; }
    max_total = max_total_;
    cache_bytes = bytes * 2;
    return cudaSuccess;
}

cudaError_t GraniteKvCache::EnsureCapacity(const GranitePinnedWeights &pw,
                                           int needed, cudaStream_t stream)
{
    if (d_k && needed <= max_total) return cudaSuccess;
    const int new_total = std::max(needed, max_total * 2);
    // Preserve written rows by copying into the fresh allocation.
    const int64_t row = static_cast<int64_t>(pw.top.num_key_value_heads) *
                        pw.top.head_dim;
    const int rows = kv_rows;
    __half *old_k = d_k;
    __half *old_v = d_v;
    const int old_total = max_total;
    d_k = nullptr;
    d_v = nullptr;
    max_total = 0;
    cache_bytes = 0;
    cudaError_t e = Allocate(pw, new_total, stream);
    if (e != cudaSuccess) { cudaFree(old_k); cudaFree(old_v); return e; }
    if (rows > 0 && old_k) {
        const size_t layer_bytes = static_cast<size_t>(old_total) * row *
                                   sizeof(__half);
        const size_t copy_bytes = static_cast<size_t>(rows) * row * sizeof(__half);
        for (int l = 0; l < pw.top.num_layers; ++l) {
            e = cudaMemcpyAsync(d_k + static_cast<int64_t>(l) * new_total * row,
                                old_k + static_cast<int64_t>(l) * old_total * row,
                                copy_bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) break;
            e = cudaMemcpyAsync(d_v + static_cast<int64_t>(l) * new_total * row,
                                old_v + static_cast<int64_t>(l) * old_total * row,
                                copy_bytes, cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) break;
        }
        (void)layer_bytes;
    }
    cudaFree(old_k);
    cudaFree(old_v);
    return e;
}

// ── Layer / generation ──────────────────────────────────────────────────────

namespace {

cudaError_t RunLayer(cudaStream_t stream, const GranitePinnedWeights &pw,
                     const GraniteLayerWeights &lw, int seq, int kv_base,
                     const int32_t *d_pos, Workspace &ws, __half *cache_k,
                     __half *cache_v)
{
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;

    cudaError_t e = GraniteRmsNorm(stream, ws.norm, ws.x, lw.d_input_norm, seq,
                                   H, static_cast<float>(top.rms_norm_eps));
    if (e != cudaSuccess) return e;

    if (lw.qkv.d_packed) {
        // One GEMV over the fused [NH*HD + 2*NKV*HD, K] weight, then split.
        e = RunLsxNvfp4Gemm(stream, ws.norm, lw.qkv.d_packed, lw.qkv.d_scales,
                            lw.qkv.global_scale, ws.qkv, seq, H,
                            (NH + 2 * NKV) * HD);
        if (e != cudaSuccess) return e;
        e = GraniteSplitQKV(stream, ws.qkv, ws.q, ws.k, ws.v, seq, NH * HD,
                            NKV * HD);
        if (e != cudaSuccess) return e;
    } else {
        e = RunLsxNvfp4Gemm(stream, ws.norm, lw.q.d_packed, lw.q.d_scales,
                            lw.q.global_scale, ws.q, seq, H, NH * HD);
        if (e != cudaSuccess) return e;
        e = RunLsxNvfp4Gemm(stream, ws.norm, lw.k.d_packed, lw.k.d_scales,
                            lw.k.global_scale, ws.k, seq, H, NKV * HD);
        if (e != cudaSuccess) return e;
        e = RunLsxNvfp4Gemm(stream, ws.norm, lw.v.d_packed, lw.v.d_scales,
                            lw.v.global_scale, ws.v, seq, H, NKV * HD);
        if (e != cudaSuccess) return e;
    }

    e = GraniteRope(stream, ws.q, ws.k, d_pos, seq, NH, NKV, HD,
                    top.rope_theta);
    if (e != cudaSuccess) return e;

    // Append this chunk's k/v rows into the persistent cache at kv_base.
    const int64_t kv_row = static_cast<int64_t>(NKV) * HD;
    const size_t layer_stride = static_cast<size_t>(ws.max_seq) * kv_row;
    // cache layer offset is handled by the caller passing the layer slice,
    // but the capacity may exceed ws.max_seq; the caller passes the true
    // cache stride via cache_k/cache_v pointers already offset per layer and
    // we compute the destination from kv_base using the cache's own stride.
    e = cudaMemcpyAsync(cache_k + static_cast<int64_t>(kv_base) * kv_row,
                        ws.k, static_cast<size_t>(seq) * kv_row * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(cache_v + static_cast<int64_t>(kv_base) * kv_row,
                        ws.v, static_cast<size_t>(seq) * kv_row * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    (void)layer_stride;

    e = GraniteAttention(stream, ws.q, cache_k, cache_v, ws.attn, ws.attn_part,
                         seq, NH, NKV, HD,
                         static_cast<float>(top.attention_multiplier), kv_base);
    if (e != cudaSuccess) return e;

    e = RunLsxNvfp4Gemm(stream, ws.attn, lw.o.d_packed, lw.o.d_scales,
                        lw.o.global_scale, ws.proj, seq, NH * HD, H);
    if (e != cudaSuccess) return e;
    kernels::AddInPlaceKernel<__half>
        <<<(seq * H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            ws.proj, ws.x, static_cast<int64_t>(seq) * H);
    e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    e = GraniteRmsNorm(stream, ws.norm, ws.x, lw.d_post_norm, seq, H,
                       static_cast<float>(top.rms_norm_eps));
    if (e != cudaSuccess) return e;

    e = RunLsxNvfp4Gemm(stream, ws.norm, lw.mlp_gate_up.d_packed,
                        lw.mlp_gate_up.d_scales, lw.mlp_gate_up.global_scale,
                        ws.gate_up, seq, H, 2 * I);
    if (e != cudaSuccess) return e;
    e = GraniteSwiGLU(stream, ws.gate_up, ws.act, I,
                      static_cast<int64_t>(seq) * I);
    if (e != cudaSuccess) return e;
    e = RunLsxNvfp4Gemm(stream, ws.act, lw.mlp_down.d_packed,
                        lw.mlp_down.d_scales, lw.mlp_down.global_scale, ws.down,
                        seq, I, H);
    if (e != cudaSuccess) return e;
    kernels::AddInPlaceKernel<__half>
        <<<(seq * H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            ws.down, ws.x, static_cast<int64_t>(seq) * H);
    return cudaGetLastError();
}

cudaError_t SampleNext(cudaStream_t stream, Workspace &ws,
                       const GranitePinnedWeights &pw, int seq,
                       lsxcommon::sample::ISampler &sampler, int32_t &tok,
                       float &maxv)
{
    const int H = pw.top.hidden_size;
    const int V = pw.top.vocab_size;
    cudaError_t e = GraniteRmsNorm(
        stream, ws.norm, ws.x, pw.d_final_norm, seq, H,
        static_cast<float>(pw.top.rms_norm_eps));
    if (e != cudaSuccess) return e;
    const __half *last = ws.norm + static_cast<int64_t>(seq - 1) * H;
    e = RunLsxNvfp4Gemm(stream, last, pw.head.d_packed, pw.head.d_scales,
                        pw.head.global_scale, ws.logits, 1, H, V);
    if (e != cudaSuccess) return e;
    e = sampler.Sample(stream, ws.logits, V,
                       lsxcommon::sample::LogitType::kF16, ws.tok, ws.maxv);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(&tok, ws.tok, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(&maxv, ws.maxv, sizeof(float), cudaMemcpyDeviceToHost,
                        stream);
    return e;
}

} // namespace

// ── Testable kernel wrappers ────────────────────────────────────────────────

cudaError_t RunGraniteRopeKernel(cudaStream_t stream, __half *q, __half *k,
                                 const int32_t *pos, int seq, int nh, int nkv,
                                 int hd, double theta)
{
    return GraniteRope(stream, q, k, pos, seq, nh, nkv, hd, theta);
}

cudaError_t RunGraniteAttentionKernel(cudaStream_t stream, const __half *q,
                                      const __half *k, const __half *v,
                                      __half *out, int seq, int nh, int nkv,
                                      int hd, float scale, int kv_base)
{
    return GraniteAttention(stream, q, k, v, out, nullptr, seq, nh, nkv, hd,
                            scale, kv_base);
}

cudaError_t RunGraniteAttentionSplitKernel(cudaStream_t stream, const __half *q,
                                           const __half *k, const __half *v,
                                           __half *out, int seq, int nh, int nkv,
                                           int hd, float scale, int kv_base)
{
    const int64_t slots = static_cast<int64_t>(kAttnSplitMax) * seq * nh;
    const size_t bytes = static_cast<size_t>(slots) * (hd + 2) * sizeof(float);
    float *partials = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&partials), bytes) != cudaSuccess)
        return cudaGetLastError();
    // Forced-split: exercise the maximum split count (many short/empty
    // boundary ranges) regardless of shape.
    const cudaError_t e =
        GraniteAttentionSplit(stream, q, k, v, out, partials, seq, nh, nkv, hd,
                              scale, kv_base, kAttnSplitMax);
    cudaFree(partials);
    return e;
}

cudaError_t RunGraniteRmsNormKernel(cudaStream_t stream, __half *out,
                                    const __half *x, const __half *w, int rows,
                                    int hidden, float eps)
{
    return GraniteRmsNorm(stream, out, x, w, rows, hidden, eps);
}

// ── Generation ──────────────────────────────────────────────────────────────

cudaError_t RunGraniteGeneration(cudaStream_t stream,
                                 const GranitePinnedWeights &pw,
                                 const std::vector<int32_t> &prompt, int max_new,
                                 int eos_id, GraniteGenerationResult &result,
                                 lsxcommon::sample::ISampler &sampler,
                                 GraniteKvCache *kv,
                                 const lsxcommon::GenerationStream *gs)
{
    const auto t_start = std::chrono::steady_clock::now();
    result = GraniteGenerationResult{};
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int V = top.vocab_size;
    const int P = static_cast<int>(prompt.size());
    if (P <= 0) {
        result.error_msg = "empty prompt";
        return cudaErrorInvalidValue;
    }
    if (top.num_layers <= 0 || pw.layers.size() <
                                   static_cast<size_t>(top.num_layers)) {
        result.error_msg = "model has no loaded layers";
        return cudaErrorInvalidValue;
    }
    if (!pw.embed.d_packed || !pw.head.d_packed || !pw.d_final_norm) {
        result.error_msg = "embed/head/final-norm missing";
        return cudaErrorInvalidValue;
    }

    Workspace ws;
    int work_seq = P;
    if (!kv) work_seq = P + std::max(max_new, 1);
    cudaError_t e = ws.Alloc(top, work_seq);
    if (e != cudaSuccess) { ws.Free(); result.error_msg = "workspace alloc"; return e; }

    // Dequant embed to FP16 for the gather.
    e = lsxcommon::gpu::dequant_shared_nvfp4(
        stream, pw.embed.d_packed, pw.embed.packed_bytes, pw.embed.d_scales,
        pw.embed.scale_bytes, pw.embed.global_scale, ws.embed_h,
        static_cast<int64_t>(V) * H);
    if (e != cudaSuccess) { ws.Free(); result.error_msg = "dequant embed"; return e; }

    GraniteKvCache local_kv;
    GraniteKvCache *cache = kv;
    // A continuation passes prompt={last_tok} and MUST append at the existing
    // kv_rows, not at position 0: the old path prefilled at 0 and reset
    // kv_rows to seq, discarding the session context (audit X2).
    const int pos0 = cache ? cache->kv_rows : 0;
    const int needed = pos0 + P + std::max(max_new, 1);
    if (!cache) {
        e = local_kv.Allocate(pw, needed, stream);
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "kv alloc"; return e; }
        cache = &local_kv;
    } else {
        e = cache->EnsureCapacity(pw, needed, stream);
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "kv grow"; return e; }
    }

    const int64_t kv_row = static_cast<int64_t>(top.num_key_value_heads) *
                           top.head_dim;

    // ── Prefill ──
    {
        const int seq = P;
        std::vector<int32_t> pos(static_cast<size_t>(seq));
        for (int i = 0; i < seq; ++i)
            pos[static_cast<size_t>(i)] = pos0 + i;
        e = cudaMemcpyAsync(ws.ids, prompt.data(),
                            static_cast<size_t>(seq) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(ws.pos, pos.data(),
                                static_cast<size_t>(seq) * sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            kernels::EmbedGatherKernel<__half>
                <<<(seq * H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    ws.embed_h, ws.ids, ws.x, seq, H);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "prefill embed"; return e; }

        for (int l = 0; l < top.num_layers && e == cudaSuccess; ++l) {
            __half *ck = cache->d_k + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            __half *cv = cache->d_v + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            e = RunLayer(stream, pw, pw.layers[static_cast<size_t>(l)], seq,
                         pos0, ws.pos, ws, ck, cv);
        }
        if (e == cudaSuccess) {
            int32_t tok = -1;
            float maxv = 0.f;
            e = SampleNext(stream, ws, pw, seq, sampler, tok, maxv);
            if (e == cudaSuccess) {
                e = cudaStreamSynchronize(stream);
                if (e == cudaSuccess) {
                    cache->kv_rows = pos0 + seq;
                    cache->last_tok = tok;
                    result.first_token = tok;
                    result.tokens.push_back(tok);
                    if (gs) gs->Emit(static_cast<std::uint32_t>(tok), 0u);
                    result.resident_bytes = pw.resident_bytes;
                }
            }
        }
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "prefill"; return e; }
    }

    const double t_ttft = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t_start)
                              .count();
    result.ttft_ms = t_ttft;

    if (result.first_token == eos_id) result.eos_terminated = true;

    // ── Decode ──
    for (int step = 1; step < max_new && !result.eos_terminated; ++step) {
        const int seq = 1;
        const int kv_base = cache->kv_rows;
        const int32_t next = cache->last_tok;
        const int32_t p0 = kv_base;
        e = cudaMemcpyAsync(ws.ids, &next, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            e = cudaMemcpyAsync(ws.pos, &p0, sizeof(int32_t),
                                cudaMemcpyHostToDevice, stream);
        if (e == cudaSuccess)
            kernels::EmbedGatherKernel<__half>
                <<<(H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    ws.embed_h, ws.ids, ws.x, seq, H);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "decode embed"; return e; }

        for (int l = 0; l < top.num_layers && e == cudaSuccess; ++l) {
            __half *ck = cache->d_k + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            __half *cv = cache->d_v + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            e = RunLayer(stream, pw, pw.layers[static_cast<size_t>(l)], seq, kv_base,
                         ws.pos, ws, ck, cv);
        }
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "decode layer"; return e; }

        int32_t tok = -1;
        float maxv = 0.f;
        e = SampleNext(stream, ws, pw, seq, sampler, tok, maxv);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "decode sample"; return e; }

        cache->last_tok = tok;
        cache->kv_rows += 1;
        result.tokens.push_back(tok);
        if (gs)
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(result.tokens.size() - 1));
        if (tok == eos_id) result.eos_terminated = true;
        if (gs && gs->Aborted()) break;
    }

    // Capture the final logits row (determinism evidence for SelfCheck). The
    // GEMM emits FP16 logits; promote to FP32 on the host.
    {
        std::vector<__half> h(static_cast<size_t>(V));
        e = cudaMemcpyAsync(h.data(), ws.logits,
                            static_cast<size_t>(V) * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) { ws.Free(); result.error_msg = "logits copy"; return e; }
        result.final_logits.resize(static_cast<size_t>(V));
        for (int i = 0; i < V; ++i)
            result.final_logits[static_cast<size_t>(i)] =
                __half2float(h[static_cast<size_t>(i)]);
    }

    const bool has_last = !result.tokens.empty();
    result.ok = has_last;
    if (!has_last && result.error_msg.empty())
        result.error_msg = "no tokens generated";
    result.total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_start)
                          .count();
    result.resident_bytes = pw.resident_bytes;

    ws.Free();
    if (!kv) local_kv.Free();
    return cudaSuccess;
}

} // namespace granite
} // namespace lsxcommon
