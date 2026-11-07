// Granite-Switch 4.1 30B low-level kernels (Model Ingestion Protocol step 5).
//
// The RoPE / causal GQA attention / split-KV / RMSNorm / SwiGLU / QKV-split
// kernels are the deterministic Granite kernels (same numeric class). The
// switch-specific additions are the multiplier scale/scale-add and the aLoRA
// shrink/expand passes. One stream, no per-token synchronisation.

#include "lsxcommon/granite_dense_switch_inference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "lsxcommon/granite_dense_switch_constants.h"
#include "lsxcommon/lsx_dense_inference.h"
#include "lsxcommon/log.h"
#include "lsxcommon/lsx_inference_nvfp4.h"
#include "lsxcommon/shared_experts.h"

#include "lsx_kernels_act.cuh"
#include "lsx_kernels_common.cuh"
#include "lsx_kernels_math.cuh"
#include "lsx_kernels_norm.cuh"

namespace lsxcommon {
namespace granite_switch {

using lsx_nvfp4::RunLsxNvfp4Gemm;

namespace {

constexpr int kThreads = 128;
// Split-KV geometry constants stay here because the driver sizes its partials
// slab from them; the kernels live in the neutral layer (lsx_dense_inference).
constexpr int kAttnSplitMax = 32;
constexpr int kAttnSplitMaxSeq = 8;
constexpr int kAttnSplitKeys = 64;

// ── Launchers (delegates to the neutral dense layer) ────────────────────────

cudaError_t GraniteSwitchRmsNorm(cudaStream_t stream, __half *out,
                                 const __half *x, const __half *w, int rows,
                                 int hidden, float eps)
{
    return dense::RunDenseRmsNormKernel(stream, out, x, w, rows, hidden, eps);
}

cudaError_t GraniteSwitchRope(cudaStream_t stream, __half *q, __half *k,
                              const int32_t *pos, int seq, int nh, int nkv,
                              int hd, double theta)
{
    return dense::RunDenseRopeKernel(stream, q, k, pos, seq, nh, nkv, hd, theta);
}

cudaError_t GraniteSwitchAttentionSplit(cudaStream_t stream, const __half *q,
                                        const __half *k, const __half *v,
                                        __half *out, float *partials, int seq,
                                        int nh, int nkv, int hd, float scale,
                                        int kv_base, int splits)
{
    return dense::RunDenseAttentionSplitKernel(stream, q, k, v, out, partials,
                                               seq, nh, nkv, hd, scale, kv_base,
                                               splits);
}

cudaError_t GraniteSwitchAttention(cudaStream_t stream, const __half *q,
                                   const __half *k, const __half *v,
                                   __half *out, float *partials, int seq,
                                   int nh, int nkv, int hd, float scale,
                                   int kv_base)
{
    return dense::RunDenseAttentionKernel(stream, q, k, v, out, partials, seq,
                                          nh, nkv, hd, scale, kv_base);
}

cudaError_t GraniteSwitchSplitQKV(cudaStream_t stream, const __half *qkv,
                                  __half *q, __half *k, __half *v, int seq,
                                  int qn, int kn)
{
    return dense::RunDenseSplitQKVKernel(stream, qkv, q, k, v, seq, qn, kn);
}

cudaError_t GraniteSwitchSwiGLU(cudaStream_t stream, const __half *gu,
                                __half *out, int I, int64_t n)
{
    return dense::RunDenseSwiGLUKernel(stream, gu, out, I, n);
}

// ── Switch-specific kernels ─────────────────────────────────────────────────

__global__ void switch_scale_kernel(__half *__restrict__ x, int64_t n,
                                    float s)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half_rn(__half2float(x[i]) * s);
}

__global__ void switch_scale_add_kernel(__half *__restrict__ dst,
                                        const __half *__restrict__ src,
                                        int64_t n, float s)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __float2half_rn(__half2float(dst[i]) +
                                 __half2float(src[i]) * s);
}

// y[t, 0:rank] = A[0:rank, 0:K] * x[t, 0:K], skipped for adapter-0 tokens.
__global__ void switch_lora_shrink_kernel(const __half *__restrict__ x,
                                          const __half *__restrict__ A,
                                          const int32_t *__restrict__ adapter,
                                          __half *__restrict__ y, int seq, int K,
                                          int rank, int y_stride, int target)
{
    const int t = blockIdx.x;
    if (t >= seq) return;
    if (adapter && (adapter[t] == 0 ||
                    (target > 0 && adapter[t] != target))) return;
    const int r = threadIdx.x;
    if (r >= rank) return;
    float acc = 0.f;
    const __half *xr = x + static_cast<int64_t>(t) * K;
    const __half *ar = A + static_cast<int64_t>(r) * K;
    for (int k = 0; k < K; ++k)
        acc = __fmaf_rn(__half2float(xr[k]), __half2float(ar[k]), acc);
    y[static_cast<int64_t>(t) * y_stride + r] = __float2half_rn(acc);
}

// out[t, off + 0:N] += B[0:N, 0:rank] * y[t, 0:rank], skipped for adapter-0.
__global__ void switch_lora_expand_add_kernel(
    const __half *__restrict__ y, const __half *__restrict__ B,
    const int32_t *__restrict__ adapter, __half *__restrict__ out, int seq,
    int rank, int N, int out_stride, int out_off, int y_stride, int target)
{
    const int t = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq || n >= N) return;
    if (adapter && (adapter[t] == 0 ||
                    (target > 0 && adapter[t] != target))) return;
    float acc = 0.f;
    const __half *yr = y + static_cast<int64_t>(t) * y_stride;
    const __half *br = B + static_cast<int64_t>(n) * rank;
    for (int r = 0; r < rank; ++r)
        acc = __fmaf_rn(__half2float(yr[r]), __half2float(br[r]), acc);
    __half *dst = out + static_cast<int64_t>(t) * out_stride + out_off + n;
    *dst = __float2half_rn(__half2float(*dst) + acc);
}

} // namespace

cudaError_t RunGraniteSwitchScale(cudaStream_t stream, __half *x, int64_t n,
                                  float s)
{
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    switch_scale_kernel<<<blocks, kThreads, 0, stream>>>(x, n, s);
    return cudaGetLastError();
}

cudaError_t RunGraniteSwitchScaleAdd(cudaStream_t stream, __half *dst,
                                     const __half *src, int64_t n, float s)
{
    const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
    switch_scale_add_kernel<<<blocks, kThreads, 0, stream>>>(dst, src, n, s);
    return cudaGetLastError();
}

cudaError_t RunGraniteSwitchLoraShrink(
    cudaStream_t stream, const __half *x, const __half *A,
    const int32_t *d_adapter, __half *y, int seq, int K, int rank,
    int y_stride, int target)
{
    if (rank <= 0) return cudaSuccess;
    switch_lora_shrink_kernel<<<seq, kThreads, 0, stream>>>(
        x, A, d_adapter, y, seq, K, rank, y_stride, target);
    return cudaGetLastError();
}

cudaError_t RunGraniteSwitchLoraExpandAdd(
    cudaStream_t stream, const __half *y, const __half *B,
    const int32_t *d_adapter, __half *out, int seq, int rank, int N,
    int out_stride, int out_off, int y_stride, int target)
{
    if (rank <= 0 || N <= 0) return cudaSuccess;
    const dim3 grid(static_cast<unsigned>((N + 255) / 256),
                    static_cast<unsigned>(seq));
    switch_lora_expand_add_kernel<<<grid, 256, 0, stream>>>(
        y, B, d_adapter, out, seq, rank, N, out_stride, out_off, y_stride,
        target);
    return cudaGetLastError();
}

cudaError_t RunGraniteSwitchRopeKernel(cudaStream_t stream, __half *q,
                                       __half *k, const int32_t *pos, int seq,
                                       int nh, int nkv, int hd, double theta)
{
    return GraniteSwitchRope(stream, q, k, pos, seq, nh, nkv, hd, theta);
}

cudaError_t RunGraniteSwitchAttentionKernel(
    cudaStream_t stream, const __half *q, const __half *k, const __half *v,
    __half *out, int seq, int nh, int nkv, int hd, float scale, int kv_base)
{
    return GraniteSwitchAttention(stream, q, k, v, out, nullptr, seq, nh, nkv,
                                  hd, scale, kv_base);
}

cudaError_t RunGraniteSwitchRmsNormKernel(cudaStream_t stream, __half *out,
                                          const __half *x, const __half *w,
                                          int rows, int hidden, float eps)
{
    return GraniteSwitchRmsNorm(stream, out, x, w, rows, hidden, eps);
}

cudaError_t RunGraniteSwitchSplitQKV(cudaStream_t stream,
                                     const __half *qkv, __half *q, __half *k,
                                     __half *v, int seq, int qn, int kn)
{
    return GraniteSwitchSplitQKV(stream, qkv, q, k, v, seq, qn, kn);
}

cudaError_t RunGraniteSwitchSwiGLU(cudaStream_t stream, const __half *gu,
                                   __half *out, int I, int64_t n)
{
    return GraniteSwitchSwiGLU(stream, gu, out, I, n);
}

cudaError_t RunGraniteSwitchAttentionSplitKernel(
    cudaStream_t stream, const __half *q, const __half *k, const __half *v,
    __half *out, int seq, int nh, int nkv, int hd, float scale, int kv_base)
{
    const int64_t slots = static_cast<int64_t>(kAttnSplitMax) * seq * nh;
    const size_t bytes =
        static_cast<size_t>(slots) * (hd + 2) * sizeof(float);
    float *partials = nullptr;
    if (cudaMalloc(reinterpret_cast<void **>(&partials), bytes) != cudaSuccess)
        return cudaGetLastError();
    const cudaError_t e = GraniteSwitchAttentionSplit(
        stream, q, k, v, out, partials, seq, nh, nkv, hd, scale, kv_base,
        kAttnSplitMax);
    cudaFree(partials);
    return e;
}


// ── KV cache ────────────────────────────────────────────────────────────────

void GraniteSwitchKvCache::Free()
{
    cudaFree(d_k);
    cudaFree(d_v);
    d_k = nullptr;
    d_v = nullptr;
    last_tok = -1;
    last_adapter = 0;
    kv_rows = 0;
    max_total = 0;
    cache_bytes = 0;
}

cudaError_t GraniteSwitchKvCache::Allocate(const GraniteSwitchPinnedWeights &pw,
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
    if (e != cudaSuccess) {
        Free();
        return e;
    }
    e = cudaMemsetAsync(d_k, 0, static_cast<size_t>(bytes), stream);
    if (e != cudaSuccess) {
        Free();
        return e;
    }
    e = cudaMemsetAsync(d_v, 0, static_cast<size_t>(bytes), stream);
    if (e != cudaSuccess) {
        Free();
        return e;
    }
    max_total = max_total_;
    cache_bytes = bytes * 2;
    return cudaSuccess;
}

cudaError_t GraniteSwitchKvCache::EnsureCapacity(
    const GraniteSwitchPinnedWeights &pw, int needed, cudaStream_t stream)
{
    if (d_k && needed <= max_total) return cudaSuccess;
    const int new_total = std::max(needed, max_total * 2);
    const int64_t row = static_cast<int64_t>(pw.top.num_key_value_heads) *
                        pw.top.head_dim;
    const int rows = kv_rows;
    const int32_t saved_tok = last_tok;
    const int saved_adapter = last_adapter;
    __half *old_k = d_k;
    __half *old_v = d_v;
    const int old_total = max_total;
    d_k = nullptr;
    d_v = nullptr;
    max_total = 0;
    cache_bytes = 0;
    cudaError_t e = Allocate(pw, new_total, stream);
    if (e != cudaSuccess) {
        cudaFree(old_k);
        cudaFree(old_v);
        return e;
    }
    if (rows > 0 && old_k) {
        const size_t copy_bytes =
            static_cast<size_t>(rows) * row * sizeof(__half);
        for (int l = 0; l < pw.top.num_layers; ++l) {
            e = cudaMemcpyAsync(
                d_k + static_cast<int64_t>(l) * new_total * row,
                old_k + static_cast<int64_t>(l) * old_total * row, copy_bytes,
                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) break;
            e = cudaMemcpyAsync(
                d_v + static_cast<int64_t>(l) * new_total * row,
                old_v + static_cast<int64_t>(l) * old_total * row, copy_bytes,
                cudaMemcpyDeviceToDevice, stream);
            if (e != cudaSuccess) break;
        }
    }
    cudaFree(old_k);
    cudaFree(old_v);
    last_tok = saved_tok;
    last_adapter = saved_adapter;
    return e;
}

// ── Generation ──────────────────────────────────────────────────────────────

namespace {

struct SwitchWorkspace {
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
    __half *lora_y = nullptr;
    int32_t *ids = nullptr;
    int32_t *adapter = nullptr;
    int32_t *pos = nullptr;
    int32_t *tok = nullptr;
    float *maxv = nullptr;
    float *attn_part = nullptr;
    int max_seq = 0;

    cudaError_t Alloc(const GraniteSwitchTopology &top, int max_seq_);
    void Free();
};

void SwitchWorkspace::Free()
{
    cudaFree(x); cudaFree(norm); cudaFree(q); cudaFree(k); cudaFree(v);
    cudaFree(qkv); cudaFree(attn); cudaFree(proj); cudaFree(gate_up);
    cudaFree(act); cudaFree(down); cudaFree(logits); cudaFree(embed_h);
    cudaFree(lora_y); cudaFree(ids); cudaFree(adapter); cudaFree(pos);
    cudaFree(tok); cudaFree(maxv); cudaFree(attn_part);
    *this = SwitchWorkspace{};
}

cudaError_t SwitchWorkspace::Alloc(const GraniteSwitchTopology &top,
                                   int max_seq_)
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
    if ((e = alloc(reinterpret_cast<void **>(&lora_y),
                   S * 64 * sizeof(__half)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&ids), S * sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&adapter), S * sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&pos), S * sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&tok), sizeof(int32_t)))) return e;
    if ((e = alloc(reinterpret_cast<void **>(&maxv), sizeof(float)))) return e;
    const int64_t sp_seq = std::min<int64_t>(S, kAttnSplitMaxSeq);
    if ((e = alloc(reinterpret_cast<void **>(&attn_part),
                   static_cast<int64_t>(kAttnSplitMax) * sp_seq * NH * (HD + 2) *
                       sizeof(float))))
        return e;
    return cudaSuccess;
}

cudaError_t ApplyLoraSlice(cudaStream_t stream, const __half *x,
                           const GraniteSwitchLora &lora,
                           const int32_t *d_adapter, __half *out, int seq, int K,
                           int N, int out_stride, int out_off, __half *y,
                           int target)
{
    if (lora.rank <= 0 || !lora.d_A || !lora.d_B) return cudaSuccess;
    cudaError_t e = RunGraniteSwitchLoraShrink(stream, x, lora.d_A, d_adapter,
                                               y, seq, K, lora.rank, 64,
                                               target);
    if (e != cudaSuccess) return e;
    return RunGraniteSwitchLoraExpandAdd(stream, y, lora.d_B, d_adapter, out,
                                         seq, lora.rank, N, out_stride, out_off,
                                         64, target);
}

// Base dot, then (when the tree carries one) the residual plane as a second dot
// accumulated into the same buffer. The arithmetic is "base rounds to FP16, then
// the residual dot adds and rounds once more" — the phi/gemma residual path —
// and it is skipped entirely on a legacy 4-bit-only tree (res.d_packed == null).
static inline cudaError_t GemmWithResidual(cudaStream_t stream, const __half *x,
                                           const lsx_nvfp4::LsxNvfp4Weight &base,
                                           const lsx_nvfp4::LsxNvfp4Weight &res,
                                           __half *y, int seq, int K, int N)
{
    cudaError_t e = RunLsxNvfp4Gemm(stream, x, base.d_packed, base.d_scales,
                                    base.global_scale, y, seq, K, N);
    if (e != cudaSuccess || res.d_packed == nullptr) return e;
    return RunLsxNvfp4Gemm(stream, x, res.d_packed, res.d_scales,
                           res.global_scale, y, seq, K, N, /*acc=*/true);
}

cudaError_t RunLayerSwitch(cudaStream_t stream,
                           const GraniteSwitchPinnedWeights &pw,
                           const GraniteSwitchLayerWeights &lw, int seq,
                           int kv_base, const int32_t *d_pos,
                           const int32_t *d_adapter, SwitchWorkspace &ws,
                           __half *cache_k, __half *cache_v)
{
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int I = top.intermediate_size;
    const int NH = top.num_attention_heads;
    const int NKV = top.num_key_value_heads;
    const int HD = top.head_dim;
    const int qn = NH * HD;
    const int kn = NKV * HD;
    const int num_adapters = top.num_adapters;

    cudaError_t e = GraniteSwitchRmsNorm(stream, ws.norm, ws.x, lw.d_input_norm,
                                         seq, H,
                                         static_cast<float>(top.rms_norm_eps));
    if (e != cudaSuccess) return e;

    e = GemmWithResidual(stream, ws.norm, lw.qkv.base, lw.qkv.res, ws.qkv,
                         seq, H, qn + 2 * kn);
    if (e != cudaSuccess) return e;

    {
        const int offs[3] = {0, qn, qn + kn};
        const int ns[3] = {qn, kn, kn};
        for (int a = 0; a < num_adapters; ++a)
            for (int sl = 0;
                 sl < static_cast<int>(lw.qkv.lora[a].size()) && sl < 3; ++sl) {
                e = ApplyLoraSlice(stream, ws.norm, lw.qkv.lora[a][sl],
                                   d_adapter, ws.qkv, seq, H, ns[sl], qn + 2 * kn,
                                   offs[sl], ws.lora_y, a + 1);
                if (e != cudaSuccess) return e;
            }
    }

    e = GraniteSwitchSplitQKV(stream, ws.qkv, ws.q, ws.k, ws.v, seq, qn, kn);
    if (e != cudaSuccess) return e;
    e = GraniteSwitchRope(stream, ws.q, ws.k, d_pos, seq, NH, NKV, HD,
                          top.rope_theta);
    if (e != cudaSuccess) return e;

    const int64_t kv_row = static_cast<int64_t>(NKV) * HD;
    e = cudaMemcpyAsync(cache_k + static_cast<int64_t>(kv_base) * kv_row, ws.k,
                        static_cast<size_t>(seq) * kv_row * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(cache_v + static_cast<int64_t>(kv_base) * kv_row, ws.v,
                        static_cast<size_t>(seq) * kv_row * sizeof(__half),
                        cudaMemcpyDeviceToDevice, stream);
    if (e != cudaSuccess) return e;

    e = GraniteSwitchAttention(stream, ws.q, cache_k, cache_v, ws.attn,
                               ws.attn_part, seq, NH, NKV, HD,
                               static_cast<float>(top.attention_multiplier),
                               kv_base);
    if (e != cudaSuccess) return e;

    e = GemmWithResidual(stream, ws.attn, lw.o.base, lw.o.res, ws.proj, seq,
                     qn, H);
    if (e != cudaSuccess) return e;
    for (int a = 0; a < num_adapters; ++a)
        if (!lw.o.lora[a].empty()) {
            e = ApplyLoraSlice(stream, ws.attn, lw.o.lora[a][0], d_adapter,
                               ws.proj, seq, qn, H, H, 0, ws.lora_y, a + 1);
            if (e != cudaSuccess) return e;
        }
    e = RunGraniteSwitchScaleAdd(stream, ws.x, ws.proj,
                                 static_cast<int64_t>(seq) * H,
                                 static_cast<float>(top.residual_multiplier));
    if (e != cudaSuccess) return e;

    e = GraniteSwitchRmsNorm(stream, ws.norm, ws.x, lw.d_post_norm, seq, H,
                             static_cast<float>(top.rms_norm_eps));
    if (e != cudaSuccess) return e;

    e = GemmWithResidual(stream, ws.norm, lw.gate_up.base, lw.gate_up.res,
                         ws.gate_up, seq, H, 2 * I);
    if (e != cudaSuccess) return e;
    for (int a = 0; a < num_adapters; ++a)
        for (int sl = 0;
             sl < static_cast<int>(lw.gate_up.lora[a].size()) && sl < 2; ++sl) {
            e = ApplyLoraSlice(stream, ws.norm, lw.gate_up.lora[a][sl],
                               d_adapter, ws.gate_up, seq, H, I, 2 * I, sl * I,
                               ws.lora_y, a + 1);
            if (e != cudaSuccess) return e;
        }

    e = GraniteSwitchSwiGLU(stream, ws.gate_up, ws.act, I,
                            static_cast<int64_t>(seq) * I);
    if (e != cudaSuccess) return e;

    e = GemmWithResidual(stream, ws.act, lw.down.base, lw.down.res, ws.down,
                     seq, I, H);
    if (e != cudaSuccess) return e;
    for (int a = 0; a < num_adapters; ++a)
        if (!lw.down.lora[a].empty()) {
            e = ApplyLoraSlice(stream, ws.act, lw.down.lora[a][0], d_adapter,
                               ws.down, seq, I, H, H, 0, ws.lora_y, a + 1);
            if (e != cudaSuccess) return e;
        }
    return RunGraniteSwitchScaleAdd(stream, ws.x, ws.down,
                                    static_cast<int64_t>(seq) * H,
                                    static_cast<float>(top.residual_multiplier));
}

cudaError_t SampleNextSwitch(cudaStream_t stream, SwitchWorkspace &ws,
                             const GraniteSwitchPinnedWeights &pw, int seq,
                             lsxcommon::sample::ISampler &sampler, int32_t &tok,
                             float &maxv)
{
    const int H = pw.top.hidden_size;
    const int V = pw.top.vocab_size;
    cudaError_t e = GraniteSwitchRmsNorm(
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

cudaError_t UploadTokens(cudaStream_t stream, SwitchWorkspace &ws,
                         const std::vector<int32_t> &ids,
                         const std::vector<int32_t> &adapters, int pos0)
{
    const int seq = static_cast<int>(ids.size());
    cudaError_t e = cudaMemcpyAsync(ws.ids, ids.data(),
                                    static_cast<size_t>(seq) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    e = cudaMemcpyAsync(ws.adapter, adapters.data(),
                        static_cast<size_t>(seq) * sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return e;
    std::vector<int32_t> pos(static_cast<size_t>(seq));
    for (int i = 0; i < seq; ++i) pos[static_cast<size_t>(i)] = pos0 + i;
    return cudaMemcpyAsync(ws.pos, pos.data(),
                           static_cast<size_t>(seq) * sizeof(int32_t),
                           cudaMemcpyHostToDevice, stream);
}

} // namespace

cudaError_t RunGraniteSwitchGeneration(
    cudaStream_t stream, const GraniteSwitchPinnedWeights &pw,
    const std::vector<int32_t> &prompt, int max_new, int eos_id,
    GraniteSwitchGenerationResult &result,
    lsxcommon::sample::ISampler &sampler, GraniteSwitchKvCache *kv,
    const lsxcommon::GenerationStream *gs)
{
    const auto t_start = std::chrono::steady_clock::now();
    result = GraniteSwitchGenerationResult{};
    const auto &top = pw.top;
    const int H = top.hidden_size;
    const int V = top.vocab_size;
    const int P = static_cast<int>(prompt.size());
    if (P <= 0) {
        result.error_msg = "empty prompt";
        return cudaErrorInvalidValue;
    }
    if (top.num_layers <= 0 ||
        pw.layers.size() < static_cast<size_t>(top.num_layers)) {
        result.error_msg = "model has no loaded layers";
        return cudaErrorInvalidValue;
    }
    if (!pw.embed.d_packed || !pw.head.d_packed || !pw.d_final_norm) {
        result.error_msg = "embed/head/final-norm missing";
        return cudaErrorInvalidValue;
    }

    SwitchWorkspace ws;
    int work_seq = P + std::max(max_new, 1);
    cudaError_t e = ws.Alloc(top, work_seq);
    if (e != cudaSuccess) {
        ws.Free();
        result.error_msg = "workspace alloc";
        return e;
    }

    e = lsxcommon::gpu::dequant_shared_nvfp4(
        stream, pw.embed.d_packed, pw.embed.packed_bytes, pw.embed.d_scales,
        pw.embed.scale_bytes, pw.embed.global_scale, ws.embed_h,
        static_cast<int64_t>(V) * H);
    if (e != cudaSuccess) {
        ws.Free();
        result.error_msg = "dequant embed";
        return e;
    }

    GraniteSwitchKvCache local_kv;
    GraniteSwitchKvCache *cache = kv;
    const int pos0 = cache ? cache->kv_rows : 0;
    const int needed = pos0 + P + std::max(max_new, 1);
    if (!cache) {
        e = local_kv.Allocate(pw, needed, stream);
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "kv alloc";
            return e;
        }
        cache = &local_kv;
    } else {
        e = cache->EnsureCapacity(pw, needed, stream);
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "kv grow";
            return e;
        }
    }

    const int64_t kv_row =
        static_cast<int64_t>(top.num_key_value_heads) * top.head_dim;
    const int carry = pos0 > 0 ? cache->last_adapter : 0;

    // ── Prefill ──
    {
        const int seq = P;
        auto prep = PrepareGraniteSwitchTokens(top, prompt, carry);
        e = UploadTokens(stream, ws, prep.embed_ids, prep.adapter_ids, pos0);
        if (e == cudaSuccess)
            kernels::EmbedGatherKernel<__half>
                <<<(seq * H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    ws.embed_h, ws.ids, ws.x, seq, H);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e == cudaSuccess)
            e = RunGraniteSwitchScale(stream, ws.x,
                                      static_cast<int64_t>(seq) * H,
                                      static_cast<float>(top.embedding_multiplier));
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "prefill embed";
            return e;
        }

        for (int l = 0; l < top.num_layers && e == cudaSuccess; ++l) {
            __half *ck = cache->d_k + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            __half *cv = cache->d_v + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            e = RunLayerSwitch(stream, pw, pw.layers[static_cast<size_t>(l)],
                               seq, pos0, ws.pos, ws.adapter, ws, ck, cv);
        }
        if (e == cudaSuccess) {
            int32_t tok = -1;
            float maxv = 0.f;
            e = SampleNextSwitch(stream, ws, pw, seq, sampler, tok, maxv);
            if (e == cudaSuccess) {
                e = cudaStreamSynchronize(stream);
                if (e == cudaSuccess) {
                    cache->kv_rows = pos0 + seq;
                    cache->last_tok = tok;
                    cache->last_adapter = prep.last_adapter;
                    result.first_token = tok;
                    result.tokens.push_back(tok);
                    if (gs) gs->Emit(static_cast<std::uint32_t>(tok), 0u);
                    result.resident_bytes = pw.resident_bytes;
                }
            }
        }
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "prefill";
            return e;
        }
    }

    result.ttft_ms = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t_start)
                         .count();
    if (result.first_token == eos_id) result.eos_terminated = true;

    // ── Decode ──
    for (int step = 1; step < max_new && !result.eos_terminated; ++step) {
        const int32_t next = cache->last_tok;
        auto prep = PrepareGraniteSwitchTokens(top, {next}, cache->last_adapter);
        e = UploadTokens(stream, ws, prep.embed_ids, prep.adapter_ids,
                         cache->kv_rows);
        if (e == cudaSuccess)
            kernels::EmbedGatherKernel<__half>
                <<<(H + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
                    ws.embed_h, ws.ids, ws.x, 1, H);
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e == cudaSuccess)
            e = RunGraniteSwitchScale(stream, ws.x, H,
                                      static_cast<float>(top.embedding_multiplier));
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "decode embed";
            return e;
        }

        const int kv_base = cache->kv_rows;
        for (int l = 0; l < top.num_layers && e == cudaSuccess; ++l) {
            __half *ck = cache->d_k + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            __half *cv = cache->d_v + static_cast<int64_t>(l) *
                                          cache->max_total * kv_row;
            e = RunLayerSwitch(stream, pw, pw.layers[static_cast<size_t>(l)], 1,
                               kv_base, ws.pos, ws.adapter, ws, ck, cv);
        }
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "decode layer";
            return e;
        }

        int32_t tok = -1;
        float maxv = 0.f;
        e = SampleNextSwitch(stream, ws, pw, 1, sampler, tok, maxv);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "decode sample";
            return e;
        }

        cache->last_tok = tok;
        cache->kv_rows += 1;
        cache->last_adapter = prep.last_adapter;
        result.tokens.push_back(tok);
        if (gs)
            gs->Emit(static_cast<std::uint32_t>(tok),
                     static_cast<std::uint32_t>(result.tokens.size() - 1));
        if (tok == eos_id) result.eos_terminated = true;
        if (gs && gs->Aborted()) break;
    }

    {
        std::vector<__half> h(static_cast<size_t>(V));
        e = cudaMemcpyAsync(h.data(), ws.logits,
                            static_cast<size_t>(V) * sizeof(__half),
                            cudaMemcpyDeviceToHost, stream);
        if (e == cudaSuccess) e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            ws.Free();
            result.error_msg = "logits copy";
            return e;
        }
        result.final_logits.resize(static_cast<size_t>(V));
        for (int i = 0; i < V; ++i)
            result.final_logits[static_cast<size_t>(i)] =
                __half2float(h[static_cast<size_t>(i)]);
    }

    result.ok = !result.tokens.empty();
    if (!result.ok && result.error_msg.empty())
        result.error_msg = "no tokens generated";
    result.last_adapter = cache->last_adapter;
    result.total_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t_start)
                          .count();
    result.resident_bytes = pw.resident_bytes;

    ws.Free();
    if (!kv) local_kv.Free();
    return cudaSuccess;
}

} // namespace granite_switch
} // namespace lsxcommon
